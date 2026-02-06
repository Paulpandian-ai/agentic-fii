"""
Stock Screener Module - Scans NYSE/NASDAQ stocks to identify value and growth opportunities.

This module provides functionality to:
- Screen stocks from NYSE and NASDAQ
- Identify value stocks based on fundamental metrics
- Identify growth stocks based on growth metrics
- Generate stock recommendations
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import pandas as pd
import numpy as np
from loguru import logger

from src.utils.yfinance_cache import get_ticker_info, get_ticker_history, yf_cache


class StockCategory(str, Enum):
    """Categories for stock classification."""
    VALUE = "value"
    GROWTH = "growth"
    BLEND = "blend"
    DIVIDEND = "dividend"
    MOMENTUM = "momentum"
    QUALITY = "quality"


class Recommendation(str, Enum):
    """Stock recommendation levels."""
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


@dataclass
class ScreenedStock:
    """Represents a screened stock with analysis."""
    symbol: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: float = 0
    current_price: float = 0

    # Value metrics
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None

    # Growth metrics
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    eps_growth: Optional[float] = None

    # Profitability
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None

    # Dividend
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None

    # Performance
    performance_1m: Optional[float] = None
    performance_3m: Optional[float] = None
    performance_ytd: Optional[float] = None
    performance_1y: Optional[float] = None

    # Scores
    value_score: float = 0
    growth_score: float = 0
    quality_score: float = 0
    momentum_score: float = 0
    overall_score: float = 0

    # Classification
    category: StockCategory = StockCategory.BLEND
    recommendation: Recommendation = Recommendation.HOLD
    analysis_summary: str = ""

    # Flags
    is_value: bool = False
    is_growth: bool = False
    is_dividend: bool = False


@dataclass
class ScreenerCriteria:
    """Criteria for stock screening."""
    # Value criteria
    max_pe_ratio: float = 20.0
    max_pb_ratio: float = 3.0
    max_ps_ratio: float = 2.0
    max_peg_ratio: float = 1.5

    # Growth criteria
    min_revenue_growth: float = 0.10  # 10%
    min_earnings_growth: float = 0.10  # 10%

    # Quality criteria
    min_profit_margin: float = 0.05  # 5%
    min_roe: float = 0.10  # 10%
    max_debt_to_equity: float = 1.5

    # Dividend criteria
    min_dividend_yield: float = 0.02  # 2%
    max_payout_ratio: float = 0.80  # 80%

    # Size criteria
    min_market_cap: float = 1e9  # $1B minimum
    max_market_cap: float = float('inf')

    # Momentum criteria
    min_momentum_1m: float = -0.10  # -10%
    min_momentum_3m: float = -0.15  # -15%


class StockScreener:
    """
    Stock screener for identifying value and growth stocks.

    Scans NYSE and NASDAQ listed stocks and classifies them based on:
    - Value metrics (P/E, P/B, P/S, PEG)
    - Growth metrics (revenue growth, earnings growth)
    - Quality metrics (profit margin, ROE, ROA)
    - Momentum (price performance)
    """

    def __init__(self):
        """Initialize the stock screener."""
        self.criteria = ScreenerCriteria()
        self._screened_stocks: list[ScreenedStock] = []

    async def screen_stocks(
        self,
        symbols: list[str],
        criteria: Optional[ScreenerCriteria] = None,
        progress_callback: Optional[callable] = None,
    ) -> list[ScreenedStock]:
        """
        Screen a list of stocks based on criteria.

        Args:
            symbols: List of stock symbols to screen
            criteria: Screening criteria (uses default if None)
            progress_callback: Optional callback for progress updates

        Returns:
            List of screened stocks with analysis
        """
        if criteria:
            self.criteria = criteria

        screened = []
        total = len(symbols)

        for i, symbol in enumerate(symbols):
            try:
                stock = await self._analyze_stock(symbol)
                if stock:
                    screened.append(stock)

                if progress_callback:
                    progress_callback(i + 1, total, symbol)

            except Exception as e:
                logger.warning(f"Error screening {symbol}: {e}")
                continue

        self._screened_stocks = screened
        return screened

    async def _analyze_stock(self, symbol: str) -> Optional[ScreenedStock]:
        """Analyze a single stock."""
        try:
            info = get_ticker_info(symbol)

            if not info or not info.get("currentPrice") and not info.get("regularMarketPrice"):
                return None

            stock = ScreenedStock(symbol=symbol)

            # Basic info
            stock.name = info.get("longName") or info.get("shortName") or symbol
            stock.sector = info.get("sector", "Unknown")
            stock.industry = info.get("industry", "Unknown")
            stock.market_cap = info.get("marketCap", 0)
            stock.current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)

            # Skip if market cap is too small
            if stock.market_cap < self.criteria.min_market_cap:
                return None

            # Value metrics
            stock.pe_ratio = info.get("trailingPE")
            stock.forward_pe = info.get("forwardPE")
            stock.pb_ratio = info.get("priceToBook")
            stock.ps_ratio = info.get("priceToSalesTrailing12Months")
            stock.peg_ratio = info.get("pegRatio")

            # Growth metrics
            stock.revenue_growth = info.get("revenueGrowth")
            stock.earnings_growth = info.get("earningsGrowth")
            stock.eps_growth = info.get("earningsQuarterlyGrowth")

            # Profitability
            stock.profit_margin = info.get("profitMargins")
            stock.operating_margin = info.get("operatingMargins")
            stock.roe = info.get("returnOnEquity")
            stock.roa = info.get("returnOnAssets")

            # Dividend
            stock.dividend_yield = info.get("dividendYield")
            stock.payout_ratio = info.get("payoutRatio")

            # Get price performance
            await self._calculate_performance(stock)

            # Calculate scores
            self._calculate_scores(stock)

            # Classify stock
            self._classify_stock(stock)

            # Generate recommendation
            self._generate_recommendation(stock)

            return stock

        except Exception as e:
            logger.warning(f"Error analyzing {symbol}: {e}")
            return None

    async def _calculate_performance(self, stock: ScreenedStock) -> None:
        """Calculate price performance metrics."""
        try:
            hist = get_ticker_history(stock.symbol, period="1y")

            if hist.empty or len(hist) < 20:
                return

            current = hist['Close'].iloc[-1]

            # 1 month performance
            if len(hist) >= 21:
                month_ago = hist['Close'].iloc[-21]
                stock.performance_1m = (current - month_ago) / month_ago

            # 3 month performance
            if len(hist) >= 63:
                three_months_ago = hist['Close'].iloc[-63]
                stock.performance_3m = (current - three_months_ago) / three_months_ago

            # YTD performance
            ytd_hist = get_ticker_history(stock.symbol, period="ytd")
            if not ytd_hist.empty and len(ytd_hist) > 1:
                ytd_start = ytd_hist['Close'].iloc[0]
                stock.performance_ytd = (current - ytd_start) / ytd_start

            # 1 year performance
            if len(hist) >= 252:
                year_ago = hist['Close'].iloc[0]
                stock.performance_1y = (current - year_ago) / year_ago

        except Exception as e:
            logger.debug(f"Error calculating performance for {stock.symbol}: {e}")

    def _calculate_scores(self, stock: ScreenedStock) -> None:
        """Calculate value, growth, quality, and momentum scores."""
        # Value Score (0-100)
        value_points = 0
        value_max = 0

        if stock.pe_ratio is not None and stock.pe_ratio > 0:
            value_max += 25
            if stock.pe_ratio < 10:
                value_points += 25
            elif stock.pe_ratio < 15:
                value_points += 20
            elif stock.pe_ratio < 20:
                value_points += 15
            elif stock.pe_ratio < 25:
                value_points += 10
            elif stock.pe_ratio < 30:
                value_points += 5

        if stock.pb_ratio is not None and stock.pb_ratio > 0:
            value_max += 25
            if stock.pb_ratio < 1:
                value_points += 25
            elif stock.pb_ratio < 2:
                value_points += 20
            elif stock.pb_ratio < 3:
                value_points += 15
            elif stock.pb_ratio < 4:
                value_points += 10
            elif stock.pb_ratio < 5:
                value_points += 5

        if stock.ps_ratio is not None and stock.ps_ratio > 0:
            value_max += 25
            if stock.ps_ratio < 1:
                value_points += 25
            elif stock.ps_ratio < 2:
                value_points += 20
            elif stock.ps_ratio < 3:
                value_points += 15
            elif stock.ps_ratio < 5:
                value_points += 10

        if stock.peg_ratio is not None and stock.peg_ratio > 0:
            value_max += 25
            if stock.peg_ratio < 1:
                value_points += 25
            elif stock.peg_ratio < 1.5:
                value_points += 20
            elif stock.peg_ratio < 2:
                value_points += 15
            elif stock.peg_ratio < 2.5:
                value_points += 10

        stock.value_score = (value_points / value_max * 100) if value_max > 0 else 50

        # Growth Score (0-100)
        growth_points = 0
        growth_max = 0

        if stock.revenue_growth is not None:
            growth_max += 35
            if stock.revenue_growth > 0.30:
                growth_points += 35
            elif stock.revenue_growth > 0.20:
                growth_points += 28
            elif stock.revenue_growth > 0.10:
                growth_points += 21
            elif stock.revenue_growth > 0.05:
                growth_points += 14
            elif stock.revenue_growth > 0:
                growth_points += 7

        if stock.earnings_growth is not None:
            growth_max += 35
            if stock.earnings_growth > 0.30:
                growth_points += 35
            elif stock.earnings_growth > 0.20:
                growth_points += 28
            elif stock.earnings_growth > 0.10:
                growth_points += 21
            elif stock.earnings_growth > 0.05:
                growth_points += 14
            elif stock.earnings_growth > 0:
                growth_points += 7

        if stock.eps_growth is not None:
            growth_max += 30
            if stock.eps_growth > 0.30:
                growth_points += 30
            elif stock.eps_growth > 0.20:
                growth_points += 24
            elif stock.eps_growth > 0.10:
                growth_points += 18
            elif stock.eps_growth > 0:
                growth_points += 12

        stock.growth_score = (growth_points / growth_max * 100) if growth_max > 0 else 50

        # Quality Score (0-100)
        quality_points = 0
        quality_max = 0

        if stock.profit_margin is not None:
            quality_max += 25
            if stock.profit_margin > 0.20:
                quality_points += 25
            elif stock.profit_margin > 0.15:
                quality_points += 20
            elif stock.profit_margin > 0.10:
                quality_points += 15
            elif stock.profit_margin > 0.05:
                quality_points += 10
            elif stock.profit_margin > 0:
                quality_points += 5

        if stock.operating_margin is not None:
            quality_max += 25
            if stock.operating_margin > 0.25:
                quality_points += 25
            elif stock.operating_margin > 0.20:
                quality_points += 20
            elif stock.operating_margin > 0.15:
                quality_points += 15
            elif stock.operating_margin > 0.10:
                quality_points += 10
            elif stock.operating_margin > 0:
                quality_points += 5

        if stock.roe is not None:
            quality_max += 25
            if stock.roe > 0.25:
                quality_points += 25
            elif stock.roe > 0.20:
                quality_points += 20
            elif stock.roe > 0.15:
                quality_points += 15
            elif stock.roe > 0.10:
                quality_points += 10
            elif stock.roe > 0:
                quality_points += 5

        if stock.roa is not None:
            quality_max += 25
            if stock.roa > 0.15:
                quality_points += 25
            elif stock.roa > 0.10:
                quality_points += 20
            elif stock.roa > 0.05:
                quality_points += 15
            elif stock.roa > 0.02:
                quality_points += 10
            elif stock.roa > 0:
                quality_points += 5

        stock.quality_score = (quality_points / quality_max * 100) if quality_max > 0 else 50

        # Momentum Score (0-100)
        momentum_points = 0
        momentum_max = 0

        if stock.performance_1m is not None:
            momentum_max += 25
            if stock.performance_1m > 0.10:
                momentum_points += 25
            elif stock.performance_1m > 0.05:
                momentum_points += 20
            elif stock.performance_1m > 0:
                momentum_points += 15
            elif stock.performance_1m > -0.05:
                momentum_points += 10
            elif stock.performance_1m > -0.10:
                momentum_points += 5

        if stock.performance_3m is not None:
            momentum_max += 25
            if stock.performance_3m > 0.15:
                momentum_points += 25
            elif stock.performance_3m > 0.08:
                momentum_points += 20
            elif stock.performance_3m > 0:
                momentum_points += 15
            elif stock.performance_3m > -0.08:
                momentum_points += 10
            elif stock.performance_3m > -0.15:
                momentum_points += 5

        if stock.performance_ytd is not None:
            momentum_max += 25
            if stock.performance_ytd > 0.25:
                momentum_points += 25
            elif stock.performance_ytd > 0.15:
                momentum_points += 20
            elif stock.performance_ytd > 0.05:
                momentum_points += 15
            elif stock.performance_ytd > -0.05:
                momentum_points += 10
            elif stock.performance_ytd > -0.15:
                momentum_points += 5

        if stock.performance_1y is not None:
            momentum_max += 25
            if stock.performance_1y > 0.30:
                momentum_points += 25
            elif stock.performance_1y > 0.20:
                momentum_points += 20
            elif stock.performance_1y > 0.10:
                momentum_points += 15
            elif stock.performance_1y > 0:
                momentum_points += 10
            elif stock.performance_1y > -0.10:
                momentum_points += 5

        stock.momentum_score = (momentum_points / momentum_max * 100) if momentum_max > 0 else 50

        # Overall Score (weighted average)
        stock.overall_score = (
            stock.value_score * 0.25 +
            stock.growth_score * 0.25 +
            stock.quality_score * 0.30 +
            stock.momentum_score * 0.20
        )

    def _classify_stock(self, stock: ScreenedStock) -> None:
        """Classify stock into categories."""
        # Value stock criteria
        stock.is_value = (
            (stock.pe_ratio is not None and stock.pe_ratio < self.criteria.max_pe_ratio and stock.pe_ratio > 0) and
            (stock.pb_ratio is None or stock.pb_ratio < self.criteria.max_pb_ratio)
        )

        # Growth stock criteria
        stock.is_growth = (
            (stock.revenue_growth is not None and stock.revenue_growth > self.criteria.min_revenue_growth) or
            (stock.earnings_growth is not None and stock.earnings_growth > self.criteria.min_earnings_growth)
        )

        # Dividend stock criteria
        stock.is_dividend = (
            stock.dividend_yield is not None and
            stock.dividend_yield > self.criteria.min_dividend_yield and
            (stock.payout_ratio is None or stock.payout_ratio < self.criteria.max_payout_ratio)
        )

        # Determine primary category
        if stock.is_value and stock.is_growth:
            stock.category = StockCategory.BLEND
        elif stock.is_value:
            stock.category = StockCategory.VALUE
        elif stock.is_growth:
            stock.category = StockCategory.GROWTH
        elif stock.is_dividend:
            stock.category = StockCategory.DIVIDEND
        elif stock.momentum_score > 70:
            stock.category = StockCategory.MOMENTUM
        elif stock.quality_score > 70:
            stock.category = StockCategory.QUALITY
        else:
            stock.category = StockCategory.BLEND

    def _generate_recommendation(self, stock: ScreenedStock) -> None:
        """Generate buy/sell recommendation."""
        score = stock.overall_score

        # Adjust based on category and specific metrics
        adjusted_score = score

        # Bonus for value stocks with good quality
        if stock.is_value and stock.quality_score > 60:
            adjusted_score += 5

        # Bonus for growth stocks with good momentum
        if stock.is_growth and stock.momentum_score > 60:
            adjusted_score += 5

        # Penalty for negative momentum
        if stock.performance_3m is not None and stock.performance_3m < -0.15:
            adjusted_score -= 10

        # Penalty for very high valuation
        if stock.pe_ratio is not None and stock.pe_ratio > 50:
            adjusted_score -= 10

        # Generate recommendation
        if adjusted_score >= 75:
            stock.recommendation = Recommendation.STRONG_BUY
        elif adjusted_score >= 60:
            stock.recommendation = Recommendation.BUY
        elif adjusted_score >= 45:
            stock.recommendation = Recommendation.HOLD
        elif adjusted_score >= 30:
            stock.recommendation = Recommendation.SELL
        else:
            stock.recommendation = Recommendation.STRONG_SELL

        # Generate summary
        stock.analysis_summary = self._generate_summary(stock)

    def _generate_summary(self, stock: ScreenedStock) -> str:
        """Generate analysis summary for a stock."""
        parts = []

        # Category
        parts.append(f"{stock.name} is classified as a {stock.category.value} stock.")

        # Value assessment
        if stock.is_value:
            pe_str = f"P/E of {stock.pe_ratio:.1f}" if stock.pe_ratio else ""
            pb_str = f"P/B of {stock.pb_ratio:.1f}" if stock.pb_ratio else ""
            metrics = ", ".join(filter(None, [pe_str, pb_str]))
            if metrics:
                parts.append(f"Shows value characteristics with {metrics}.")

        # Growth assessment
        if stock.is_growth:
            growth_parts = []
            if stock.revenue_growth:
                growth_parts.append(f"revenue growth of {stock.revenue_growth*100:.1f}%")
            if stock.earnings_growth:
                growth_parts.append(f"earnings growth of {stock.earnings_growth*100:.1f}%")
            if growth_parts:
                parts.append(f"Demonstrates strong growth with {' and '.join(growth_parts)}.")

        # Quality assessment
        if stock.quality_score > 70:
            parts.append("High quality business with strong profitability metrics.")
        elif stock.quality_score < 40:
            parts.append("Quality metrics suggest room for improvement.")

        # Momentum
        if stock.momentum_score > 70:
            parts.append("Strong positive price momentum.")
        elif stock.momentum_score < 30:
            parts.append("Recent price weakness may present opportunity or concern.")

        return " ".join(parts)

    def get_value_stocks(self, min_score: float = 60) -> list[ScreenedStock]:
        """Get stocks classified as value with minimum score."""
        return [s for s in self._screened_stocks
                if s.is_value and s.value_score >= min_score]

    def get_growth_stocks(self, min_score: float = 60) -> list[ScreenedStock]:
        """Get stocks classified as growth with minimum score."""
        return [s for s in self._screened_stocks
                if s.is_growth and s.growth_score >= min_score]

    def get_dividend_stocks(self, min_yield: float = 0.02) -> list[ScreenedStock]:
        """Get dividend stocks with minimum yield."""
        return [s for s in self._screened_stocks
                if s.is_dividend and s.dividend_yield and s.dividend_yield >= min_yield]

    def get_top_recommendations(
        self,
        n: int = 10,
        category: Optional[StockCategory] = None
    ) -> list[ScreenedStock]:
        """Get top N stock recommendations."""
        stocks = self._screened_stocks

        if category:
            stocks = [s for s in stocks if s.category == category]

        # Filter to only buy recommendations
        buy_stocks = [s for s in stocks
                      if s.recommendation in [Recommendation.STRONG_BUY, Recommendation.BUY]]

        # Sort by overall score
        sorted_stocks = sorted(buy_stocks, key=lambda x: x.overall_score, reverse=True)

        return sorted_stocks[:n]

    def get_stocks_by_sector(self, sector: str) -> list[ScreenedStock]:
        """Get screened stocks by sector."""
        return [s for s in self._screened_stocks if s.sector == sector]

    def get_screening_summary(self) -> dict[str, Any]:
        """Get summary statistics of screening results."""
        if not self._screened_stocks:
            return {}

        total = len(self._screened_stocks)

        return {
            "total_screened": total,
            "value_stocks": len([s for s in self._screened_stocks if s.is_value]),
            "growth_stocks": len([s for s in self._screened_stocks if s.is_growth]),
            "dividend_stocks": len([s for s in self._screened_stocks if s.is_dividend]),
            "strong_buy": len([s for s in self._screened_stocks if s.recommendation == Recommendation.STRONG_BUY]),
            "buy": len([s for s in self._screened_stocks if s.recommendation == Recommendation.BUY]),
            "hold": len([s for s in self._screened_stocks if s.recommendation == Recommendation.HOLD]),
            "sell": len([s for s in self._screened_stocks if s.recommendation == Recommendation.SELL]),
            "strong_sell": len([s for s in self._screened_stocks if s.recommendation == Recommendation.STRONG_SELL]),
            "avg_value_score": np.mean([s.value_score for s in self._screened_stocks]),
            "avg_growth_score": np.mean([s.growth_score for s in self._screened_stocks]),
            "avg_quality_score": np.mean([s.quality_score for s in self._screened_stocks]),
            "sectors": list(set(s.sector for s in self._screened_stocks if s.sector != "Unknown")),
        }
