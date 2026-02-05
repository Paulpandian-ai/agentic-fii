"""Fundamental Analysis Agent - Analyzes financial metrics and company fundamentals."""

from typing import Any, Optional

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.schemas import FundamentalMetrics
from src.utils.yfinance_cache import get_ticker_info


class FundamentalAnalysisAgent(BaseAgent):
    """
    Agent responsible for fundamental analysis of stocks.

    Analyzes financial metrics including:
    - Valuation ratios (P/E, P/B, P/S, PEG)
    - Profitability metrics (margins, ROE, ROA)
    - Growth metrics (revenue, earnings growth)
    - Financial health (debt ratios, current ratio)
    - Dividend metrics
    """

    def __init__(self, name: str = "FundamentalAgent"):
        super().__init__(name=name, agent_type="fundamental")

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """
        Perform fundamental analysis on the given stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing fundamental metrics and analysis
        """
        self.log_info(f"Fetching fundamental data for {symbol}")

        try:
            info = get_ticker_info(symbol)

            # Extract fundamental metrics
            metrics = self._extract_metrics(symbol, info)

            # Calculate fundamental score
            score = self._calculate_score(metrics)

            # Generate analysis summary
            summary = self._generate_summary(metrics, score)

            return {
                "metrics": metrics.model_dump(),
                "score": score,
                "summary": summary,
            }

        except Exception as e:
            self.log_warning(f"Error fetching fundamental data: {str(e)}")
            return {
                "metrics": FundamentalMetrics(symbol=symbol).model_dump(),
                "score": None,
                "summary": f"Unable to fetch fundamental data: {str(e)}",
            }

    def _extract_metrics(self, symbol: str, info: dict) -> FundamentalMetrics:
        """Extract fundamental metrics from Yahoo Finance data."""
        return FundamentalMetrics(
            symbol=symbol,
            # Valuation
            pe_ratio=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            peg_ratio=info.get("pegRatio"),
            price_to_book=info.get("priceToBook"),
            price_to_sales=info.get("priceToSalesTrailing12Months"),
            # Profitability
            profit_margin=info.get("profitMargins"),
            operating_margin=info.get("operatingMargins"),
            return_on_equity=info.get("returnOnEquity"),
            return_on_assets=info.get("returnOnAssets"),
            # Growth
            revenue_growth=info.get("revenueGrowth"),
            earnings_growth=info.get("earningsGrowth"),
            # Financial Health
            debt_to_equity=info.get("debtToEquity"),
            current_ratio=info.get("currentRatio"),
            quick_ratio=info.get("quickRatio"),
            # Dividends
            dividend_yield=info.get("dividendYield"),
            payout_ratio=info.get("payoutRatio"),
            # Earnings
            eps=info.get("trailingEps"),
            eps_growth=info.get("earningsQuarterlyGrowth"),
        )

    def _calculate_score(self, metrics: FundamentalMetrics) -> float:
        """
        Calculate a fundamental score from 0-100.

        Scoring criteria:
        - Valuation (25%): Lower P/E and P/B are better
        - Profitability (25%): Higher margins and returns are better
        - Growth (25%): Higher growth rates are better
        - Financial Health (25%): Lower debt, higher liquidity is better
        """
        score = 50.0  # Start at neutral
        weights_applied = 0

        # Valuation Score (25 points max)
        valuation_score = 0
        valuation_factors = 0

        if metrics.pe_ratio is not None:
            if metrics.pe_ratio < 0:
                valuation_score += 0  # Negative earnings
            elif metrics.pe_ratio < 15:
                valuation_score += 25
            elif metrics.pe_ratio < 25:
                valuation_score += 18
            elif metrics.pe_ratio < 35:
                valuation_score += 10
            else:
                valuation_score += 5
            valuation_factors += 1

        if metrics.peg_ratio is not None and metrics.peg_ratio > 0:
            if metrics.peg_ratio < 1:
                valuation_score += 25
            elif metrics.peg_ratio < 1.5:
                valuation_score += 18
            elif metrics.peg_ratio < 2:
                valuation_score += 12
            else:
                valuation_score += 5
            valuation_factors += 1

        if valuation_factors > 0:
            score += (valuation_score / valuation_factors) - 12.5
            weights_applied += 25

        # Profitability Score (25 points max)
        profit_score = 0
        profit_factors = 0

        if metrics.profit_margin is not None:
            if metrics.profit_margin > 0.20:
                profit_score += 25
            elif metrics.profit_margin > 0.10:
                profit_score += 18
            elif metrics.profit_margin > 0.05:
                profit_score += 12
            elif metrics.profit_margin > 0:
                profit_score += 6
            profit_factors += 1

        if metrics.return_on_equity is not None:
            if metrics.return_on_equity > 0.20:
                profit_score += 25
            elif metrics.return_on_equity > 0.15:
                profit_score += 18
            elif metrics.return_on_equity > 0.10:
                profit_score += 12
            elif metrics.return_on_equity > 0:
                profit_score += 6
            profit_factors += 1

        if profit_factors > 0:
            score += (profit_score / profit_factors) - 12.5
            weights_applied += 25

        # Growth Score (25 points max)
        growth_score = 0
        growth_factors = 0

        if metrics.revenue_growth is not None:
            if metrics.revenue_growth > 0.25:
                growth_score += 25
            elif metrics.revenue_growth > 0.15:
                growth_score += 20
            elif metrics.revenue_growth > 0.10:
                growth_score += 15
            elif metrics.revenue_growth > 0.05:
                growth_score += 10
            elif metrics.revenue_growth > 0:
                growth_score += 5
            growth_factors += 1

        if metrics.earnings_growth is not None:
            if metrics.earnings_growth > 0.25:
                growth_score += 25
            elif metrics.earnings_growth > 0.15:
                growth_score += 20
            elif metrics.earnings_growth > 0.10:
                growth_score += 15
            elif metrics.earnings_growth > 0:
                growth_score += 8
            growth_factors += 1

        if growth_factors > 0:
            score += (growth_score / growth_factors) - 12.5
            weights_applied += 25

        # Financial Health Score (25 points max)
        health_score = 0
        health_factors = 0

        if metrics.debt_to_equity is not None:
            if metrics.debt_to_equity < 0.3:
                health_score += 25
            elif metrics.debt_to_equity < 0.5:
                health_score += 20
            elif metrics.debt_to_equity < 1.0:
                health_score += 15
            elif metrics.debt_to_equity < 2.0:
                health_score += 8
            else:
                health_score += 3
            health_factors += 1

        if metrics.current_ratio is not None:
            if metrics.current_ratio > 2.0:
                health_score += 25
            elif metrics.current_ratio > 1.5:
                health_score += 20
            elif metrics.current_ratio > 1.0:
                health_score += 12
            else:
                health_score += 5
            health_factors += 1

        if health_factors > 0:
            score += (health_score / health_factors) - 12.5
            weights_applied += 25

        # Normalize score to 0-100
        return max(0, min(100, score))

    def _generate_summary(self, metrics: FundamentalMetrics, score: float) -> str:
        """Generate a summary of the fundamental analysis."""
        strengths = []
        weaknesses = []

        # Analyze valuation
        if metrics.pe_ratio is not None:
            if metrics.pe_ratio < 15 and metrics.pe_ratio > 0:
                strengths.append("attractively valued with low P/E ratio")
            elif metrics.pe_ratio > 35:
                weaknesses.append("premium valuation with high P/E ratio")

        # Analyze profitability
        if metrics.profit_margin is not None and metrics.profit_margin > 0.15:
            strengths.append("strong profit margins")
        if metrics.return_on_equity is not None and metrics.return_on_equity > 0.15:
            strengths.append("excellent return on equity")

        # Analyze growth
        if metrics.revenue_growth is not None and metrics.revenue_growth > 0.15:
            strengths.append("solid revenue growth")
        if metrics.earnings_growth is not None and metrics.earnings_growth > 0.15:
            strengths.append("strong earnings growth")
        elif metrics.earnings_growth is not None and metrics.earnings_growth < 0:
            weaknesses.append("declining earnings")

        # Analyze financial health
        if metrics.debt_to_equity is not None:
            if metrics.debt_to_equity < 0.5:
                strengths.append("low debt levels")
            elif metrics.debt_to_equity > 2.0:
                weaknesses.append("high debt burden")

        # Build summary
        summary_parts = []

        if score >= 70:
            summary_parts.append("Strong fundamentals overall.")
        elif score >= 50:
            summary_parts.append("Moderate fundamentals.")
        else:
            summary_parts.append("Weak fundamentals requiring attention.")

        if strengths:
            summary_parts.append(f"Strengths: {', '.join(strengths)}.")
        if weaknesses:
            summary_parts.append(f"Concerns: {', '.join(weaknesses)}.")

        return " ".join(summary_parts)
