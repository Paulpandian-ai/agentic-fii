"""
Investment Management Module

Provides intelligent stock selection and portfolio construction using:
- Multi-factor scoring from screener and technical analysis
- Sharpe ratio optimization for portfolio weights
- Monte Carlo simulation for 12-month return projections
- Integration with all platform modules for comprehensive analysis
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum
import numpy as np
import random

from src.utils.yfinance_cache import get_ticker_info, get_ticker_history
from src.screener.stock_screener import StockScreener, ScreenerCriteria, Recommendation
from src.screener.technical_analysis import get_technical_analyzer, SignalType
from src.screener.portfolio_analytics import get_portfolio_analytics
from src.screener.institutional_tracking import get_institutional_tracker
from src.screener.stock_universe import (
    get_sp500_symbols,
    get_nasdaq100_symbols,
    get_dow30_symbols,
    get_dividend_aristocrats,
)


class RiskProfile(Enum):
    CONSERVATIVE = "Conservative"
    MODERATE = "Moderate"
    AGGRESSIVE = "Aggressive"


class InvestmentStrategy(Enum):
    VALUE = "Value Investing"
    GROWTH = "Growth Investing"
    DIVIDEND = "Dividend Income"
    MOMENTUM = "Momentum Trading"
    BALANCED = "Balanced Portfolio"
    QUALITY = "Quality Focus"


@dataclass
class StockPick:
    """Individual stock recommendation."""
    symbol: str
    name: str
    sector: str
    current_price: float

    # Scores from various modules
    fundamental_score: float = 0.0
    technical_score: float = 0.0
    sentiment_score: float = 0.0
    quality_score: float = 0.0
    momentum_score: float = 0.0

    # Combined scores
    composite_score: float = 0.0

    # Allocation
    weight: float = 0.0
    shares_to_buy: int = 0
    investment_amount: float = 0.0

    # Projections
    expected_return_12m: float = 0.0
    projected_price_12m: float = 0.0
    upside_potential: float = 0.0

    # Monte Carlo results
    mc_mean_return: float = 0.0
    mc_median_return: float = 0.0
    mc_std_return: float = 0.0
    mc_percentile_5: float = 0.0  # Bear case
    mc_percentile_25: float = 0.0
    mc_percentile_75: float = 0.0
    mc_percentile_95: float = 0.0  # Bull case
    mc_probability_positive: float = 0.0

    # Risk metrics
    volatility: float = 0.0
    beta: float = 0.0
    sharpe_contribution: float = 0.0

    # Rationale
    buy_reasons: list = field(default_factory=list)
    risk_factors: list = field(default_factory=list)


@dataclass
class PortfolioRecommendation:
    """Complete portfolio recommendation."""
    strategy: InvestmentStrategy
    risk_profile: RiskProfile
    investment_amount: float

    # Stock picks
    stocks: list[StockPick] = field(default_factory=list)

    # Portfolio metrics
    expected_portfolio_return: float = 0.0
    portfolio_volatility: float = 0.0
    portfolio_sharpe: float = 0.0
    portfolio_beta: float = 0.0

    # Monte Carlo portfolio projections
    mc_portfolio_mean: float = 0.0
    mc_portfolio_median: float = 0.0
    mc_portfolio_std: float = 0.0
    mc_portfolio_5th: float = 0.0  # Worst case
    mc_portfolio_25th: float = 0.0
    mc_portfolio_75th: float = 0.0
    mc_portfolio_95th: float = 0.0  # Best case
    mc_probability_profit: float = 0.0
    mc_probability_beat_market: float = 0.0

    # Value projections
    projected_value_12m: float = 0.0
    projected_gain_12m: float = 0.0

    # Sector allocation
    sector_weights: dict = field(default_factory=dict)

    # Analysis metadata
    stocks_analyzed: int = 0
    analysis_date: str = ""
    num_simulations: int = 10000


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results."""
    simulations: int
    time_horizon_days: int

    # Return distribution
    mean_return: float = 0.0
    median_return: float = 0.0
    std_return: float = 0.0
    min_return: float = 0.0
    max_return: float = 0.0

    # Percentiles
    percentile_5: float = 0.0
    percentile_10: float = 0.0
    percentile_25: float = 0.0
    percentile_75: float = 0.0
    percentile_90: float = 0.0
    percentile_95: float = 0.0

    # Probabilities
    prob_positive: float = 0.0
    prob_above_10pct: float = 0.0
    prob_above_20pct: float = 0.0
    prob_loss_10pct: float = 0.0
    prob_loss_20pct: float = 0.0

    # Simulation paths (sample)
    sample_paths: list = field(default_factory=list)


class InvestmentManager:
    """
    Intelligent investment manager that combines multiple analysis modules
    to provide optimized stock picks and portfolio recommendations.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252
        self.screener = StockScreener()
        self.technical_analyzer = get_technical_analyzer()
        self.portfolio_analytics = get_portfolio_analytics()
        self.institutional_tracker = get_institutional_tracker()

    async def generate_recommendations(
        self,
        investment_amount: float,
        strategy: InvestmentStrategy = InvestmentStrategy.BALANCED,
        risk_profile: RiskProfile = RiskProfile.MODERATE,
        num_stocks: int = 10,
        universe: str = "sp500",
        num_simulations: int = 10000
    ) -> PortfolioRecommendation:
        """
        Generate portfolio recommendations with optimal stock picks.

        Args:
            investment_amount: Total amount to invest
            strategy: Investment strategy to follow
            risk_profile: Risk tolerance level
            num_stocks: Number of stocks to include
            universe: Stock universe to screen
            num_simulations: Monte Carlo simulation count

        Returns:
            PortfolioRecommendation with optimized picks
        """
        # Get stock universe
        symbols = self._get_universe_symbols(universe)

        # Screen stocks based on strategy
        criteria = self._get_strategy_criteria(strategy, risk_profile)
        screened = await self.screener.screen_stocks(symbols[:100], criteria)

        if not screened:
            return PortfolioRecommendation(
                strategy=strategy,
                risk_profile=risk_profile,
                investment_amount=investment_amount,
                analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M")
            )

        # Score and rank stocks
        scored_stocks = self._score_stocks(screened, strategy)

        # Select top stocks
        top_stocks = sorted(scored_stocks, key=lambda x: x.composite_score, reverse=True)[:num_stocks * 2]

        # Filter for diversification
        diversified = self._ensure_diversification(top_stocks, num_stocks)

        # Optimize weights using Sharpe ratio
        optimized = self._optimize_weights_sharpe(diversified, risk_profile)

        # Allocate investment
        allocated = self._allocate_investment(optimized, investment_amount)

        # Run Monte Carlo simulations
        self._run_monte_carlo_simulations(allocated, num_simulations)

        # Calculate portfolio metrics
        portfolio = self._calculate_portfolio_metrics(
            allocated, investment_amount, strategy, risk_profile, num_simulations
        )

        portfolio.stocks_analyzed = len(screened)

        return portfolio

    def _get_universe_symbols(self, universe: str) -> list[str]:
        """Get symbols for the specified universe."""
        if universe == "sp500":
            return get_sp500_symbols()
        elif universe == "nasdaq100":
            return get_nasdaq100_symbols()
        elif universe == "dow30":
            return get_dow30_symbols()
        elif universe == "dividend":
            return get_dividend_aristocrats()
        else:
            return get_sp500_symbols()

    def _get_strategy_criteria(
        self,
        strategy: InvestmentStrategy,
        risk_profile: RiskProfile
    ) -> ScreenerCriteria:
        """Get screening criteria based on strategy."""

        base_criteria = {
            'min_market_cap': 10e9 if risk_profile == RiskProfile.CONSERVATIVE else 2e9,
        }

        if strategy == InvestmentStrategy.VALUE:
            return ScreenerCriteria(
                max_pe_ratio=18,
                max_pb_ratio=2.5,
                max_peg_ratio=1.5,
                min_profit_margin=0.08,
                min_roe=0.10,
                **base_criteria
            )
        elif strategy == InvestmentStrategy.GROWTH:
            return ScreenerCriteria(
                max_pe_ratio=40,
                max_pb_ratio=10,
                min_revenue_growth=0.15,
                min_earnings_growth=0.15,
                **base_criteria
            )
        elif strategy == InvestmentStrategy.DIVIDEND:
            return ScreenerCriteria(
                max_pe_ratio=25,
                min_dividend_yield=0.02,
                min_profit_margin=0.10,
                **base_criteria
            )
        elif strategy == InvestmentStrategy.MOMENTUM:
            return ScreenerCriteria(
                max_pe_ratio=35,
                min_revenue_growth=0.10,
                **base_criteria
            )
        elif strategy == InvestmentStrategy.QUALITY:
            return ScreenerCriteria(
                max_pe_ratio=30,
                min_profit_margin=0.15,
                min_roe=0.15,
                **base_criteria
            )
        else:  # BALANCED
            return ScreenerCriteria(
                max_pe_ratio=25,
                max_pb_ratio=5,
                min_profit_margin=0.05,
                min_roe=0.08,
                **base_criteria
            )

    def _score_stocks(
        self,
        screened_stocks: list,
        strategy: InvestmentStrategy
    ) -> list[StockPick]:
        """Score stocks using multiple factors."""
        scored = []

        for stock in screened_stocks:
            pick = StockPick(
                symbol=stock.symbol,
                name=stock.name,
                sector=stock.sector,
                current_price=stock.current_price or 0
            )

            # Fundamental scores from screener
            pick.fundamental_score = (stock.value_score + stock.quality_score) / 2
            pick.quality_score = stock.quality_score
            pick.momentum_score = stock.momentum_score

            # Technical analysis
            try:
                tech = self.technical_analyzer.analyze(stock.symbol, period="6mo")
                if tech:
                    pick.technical_score = tech.signal_strength

                    # Add buy reasons based on signals
                    if tech.overall_signal in [SignalType.STRONG_BUY, SignalType.BUY]:
                        pick.buy_reasons.append(f"Technical: {tech.overall_signal.value}")
                    if tech.moving_averages.golden_cross:
                        pick.buy_reasons.append("Golden Cross pattern")
                    if tech.rsi.is_oversold:
                        pick.buy_reasons.append("RSI indicates oversold")
            except:
                pick.technical_score = 50

            # Institutional sentiment
            try:
                sentiment = self.institutional_tracker.analyze_insider_sentiment(stock.symbol, 90)
                if sentiment.sentiment_score > 20:
                    pick.sentiment_score = min(100, 50 + sentiment.sentiment_score)
                    pick.buy_reasons.append(f"Insider buying: {sentiment.sentiment_label}")
                elif sentiment.sentiment_score < -20:
                    pick.sentiment_score = max(0, 50 + sentiment.sentiment_score)
                    pick.risk_factors.append("Recent insider selling")
                else:
                    pick.sentiment_score = 50
            except:
                pick.sentiment_score = 50

            # Add fundamental buy reasons
            if stock.recommendation in [Recommendation.STRONG_BUY, Recommendation.BUY]:
                pick.buy_reasons.append(f"Screener: {stock.recommendation.value}")
            if stock.value_score > 70:
                pick.buy_reasons.append(f"Undervalued (Value Score: {stock.value_score:.0f})")
            if stock.growth_score > 70:
                pick.buy_reasons.append(f"Strong growth (Growth Score: {stock.growth_score:.0f})")
            if stock.quality_score > 70:
                pick.buy_reasons.append(f"High quality (Quality Score: {stock.quality_score:.0f})")

            # Add risk factors
            if stock.pe_ratio and stock.pe_ratio > 30:
                pick.risk_factors.append(f"High P/E ratio ({stock.pe_ratio:.1f})")
            if stock.beta and stock.beta > 1.5:
                pick.risk_factors.append(f"High beta ({stock.beta:.2f})")
                pick.beta = stock.beta

            # Calculate composite score based on strategy
            pick.composite_score = self._calculate_composite_score(pick, strategy)

            # Get volatility
            try:
                hist = get_ticker_history(stock.symbol, period="1y")
                if hist is not None and not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    pick.volatility = returns.std() * np.sqrt(252)
            except:
                pick.volatility = 0.25  # Default 25%

            scored.append(pick)

        return scored

    def _calculate_composite_score(
        self,
        pick: StockPick,
        strategy: InvestmentStrategy
    ) -> float:
        """Calculate composite score based on strategy weights."""

        weights = {
            InvestmentStrategy.VALUE: {
                'fundamental': 0.40, 'technical': 0.15, 'sentiment': 0.15,
                'quality': 0.20, 'momentum': 0.10
            },
            InvestmentStrategy.GROWTH: {
                'fundamental': 0.25, 'technical': 0.20, 'sentiment': 0.15,
                'quality': 0.15, 'momentum': 0.25
            },
            InvestmentStrategy.DIVIDEND: {
                'fundamental': 0.35, 'technical': 0.10, 'sentiment': 0.15,
                'quality': 0.30, 'momentum': 0.10
            },
            InvestmentStrategy.MOMENTUM: {
                'fundamental': 0.15, 'technical': 0.35, 'sentiment': 0.15,
                'quality': 0.10, 'momentum': 0.25
            },
            InvestmentStrategy.QUALITY: {
                'fundamental': 0.25, 'technical': 0.15, 'sentiment': 0.15,
                'quality': 0.35, 'momentum': 0.10
            },
            InvestmentStrategy.BALANCED: {
                'fundamental': 0.25, 'technical': 0.20, 'sentiment': 0.15,
                'quality': 0.25, 'momentum': 0.15
            },
        }

        w = weights.get(strategy, weights[InvestmentStrategy.BALANCED])

        return (
            pick.fundamental_score * w['fundamental'] +
            pick.technical_score * w['technical'] +
            pick.sentiment_score * w['sentiment'] +
            pick.quality_score * w['quality'] +
            pick.momentum_score * w['momentum']
        )

    def _ensure_diversification(
        self,
        stocks: list[StockPick],
        target_count: int
    ) -> list[StockPick]:
        """Ensure sector diversification in selected stocks."""
        selected = []
        sector_counts = {}
        max_per_sector = max(2, target_count // 4)

        for stock in stocks:
            sector = stock.sector or "Unknown"
            current_count = sector_counts.get(sector, 0)

            if current_count < max_per_sector:
                selected.append(stock)
                sector_counts[sector] = current_count + 1

                if len(selected) >= target_count:
                    break

        return selected

    def _optimize_weights_sharpe(
        self,
        stocks: list[StockPick],
        risk_profile: RiskProfile
    ) -> list[StockPick]:
        """Optimize portfolio weights to maximize Sharpe ratio."""
        n = len(stocks)
        if n == 0:
            return stocks

        # Get historical returns
        returns_data = []
        valid_stocks = []

        for stock in stocks:
            hist = get_ticker_history(stock.symbol, period="1y")
            if hist is not None and not hist.empty and len(hist) > 60:
                rets = hist['Close'].pct_change().dropna().values
                returns_data.append(rets[-252:] if len(rets) > 252 else rets)
                valid_stocks.append(stock)

        if len(valid_stocks) < 2:
            # Equal weight if not enough data
            for stock in stocks:
                stock.weight = 1.0 / n
            return stocks

        # Align return series
        min_len = min(len(r) for r in returns_data)
        aligned = np.array([r[-min_len:] for r in returns_data])

        # Calculate expected returns and covariance
        mean_returns = np.mean(aligned, axis=1) * 252  # Annualized
        cov_matrix = np.cov(aligned) * 252

        # Optimize using Monte Carlo weight sampling
        best_sharpe = -np.inf
        best_weights = np.ones(len(valid_stocks)) / len(valid_stocks)

        # Risk profile constraints
        if risk_profile == RiskProfile.CONSERVATIVE:
            max_weight = 0.20
            target_vol = 0.12
        elif risk_profile == RiskProfile.AGGRESSIVE:
            max_weight = 0.35
            target_vol = 0.25
        else:  # MODERATE
            max_weight = 0.25
            target_vol = 0.18

        # Random search for optimal weights
        for _ in range(5000):
            # Generate random weights
            weights = np.random.random(len(valid_stocks))
            weights = weights / weights.sum()

            # Apply max weight constraint
            weights = np.minimum(weights, max_weight)
            weights = weights / weights.sum()

            # Calculate portfolio metrics
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            # Skip if volatility too high for risk profile
            if port_vol > target_vol * 1.5:
                continue

            sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights.copy()

        # Assign optimized weights
        for i, stock in enumerate(valid_stocks):
            stock.weight = best_weights[i]
            stock.sharpe_contribution = (
                (np.mean(aligned[i]) * 252 - self.risk_free_rate) /
                (np.std(aligned[i]) * np.sqrt(252))
            ) * stock.weight

        return valid_stocks

    def _allocate_investment(
        self,
        stocks: list[StockPick],
        total_amount: float
    ) -> list[StockPick]:
        """Allocate investment amount to stocks."""
        for stock in stocks:
            stock.investment_amount = total_amount * stock.weight
            if stock.current_price > 0:
                stock.shares_to_buy = int(stock.investment_amount / stock.current_price)
                # Adjust investment amount to actual shares
                stock.investment_amount = stock.shares_to_buy * stock.current_price

        return stocks

    def _run_monte_carlo_simulations(
        self,
        stocks: list[StockPick],
        num_simulations: int = 10000
    ):
        """Run Monte Carlo simulations for each stock."""
        trading_days = 252  # 12 months

        for stock in stocks:
            hist = get_ticker_history(stock.symbol, period="2y")

            if hist is None or hist.empty:
                # Use default estimates
                stock.expected_return_12m = 0.08
                stock.mc_mean_return = 0.08
                stock.mc_median_return = 0.07
                stock.mc_probability_positive = 0.60
                continue

            returns = hist['Close'].pct_change().dropna().values

            if len(returns) < 30:
                continue

            # Calculate parameters
            mu = np.mean(returns)
            sigma = np.std(returns)

            # Run simulations
            simulated_returns = []

            for _ in range(num_simulations):
                # Geometric Brownian Motion
                daily_returns = np.random.normal(mu, sigma, trading_days)
                cumulative_return = np.prod(1 + daily_returns) - 1
                simulated_returns.append(cumulative_return)

            simulated_returns = np.array(simulated_returns)

            # Store results
            stock.mc_mean_return = np.mean(simulated_returns)
            stock.mc_median_return = np.median(simulated_returns)
            stock.mc_std_return = np.std(simulated_returns)
            stock.mc_percentile_5 = np.percentile(simulated_returns, 5)
            stock.mc_percentile_25 = np.percentile(simulated_returns, 25)
            stock.mc_percentile_75 = np.percentile(simulated_returns, 75)
            stock.mc_percentile_95 = np.percentile(simulated_returns, 95)
            stock.mc_probability_positive = np.mean(simulated_returns > 0)

            # Expected return and projected price
            stock.expected_return_12m = stock.mc_median_return
            stock.projected_price_12m = stock.current_price * (1 + stock.expected_return_12m)
            stock.upside_potential = stock.expected_return_12m

    def _calculate_portfolio_metrics(
        self,
        stocks: list[StockPick],
        investment_amount: float,
        strategy: InvestmentStrategy,
        risk_profile: RiskProfile,
        num_simulations: int
    ) -> PortfolioRecommendation:
        """Calculate overall portfolio metrics and projections."""

        portfolio = PortfolioRecommendation(
            strategy=strategy,
            risk_profile=risk_profile,
            investment_amount=investment_amount,
            stocks=stocks,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            num_simulations=num_simulations
        )

        if not stocks:
            return portfolio

        # Calculate weighted metrics
        weights = np.array([s.weight for s in stocks])
        expected_returns = np.array([s.expected_return_12m for s in stocks])
        volatilities = np.array([s.volatility for s in stocks])
        betas = np.array([s.beta if s.beta else 1.0 for s in stocks])

        # Portfolio expected return
        portfolio.expected_portfolio_return = np.dot(weights, expected_returns)

        # Portfolio volatility (simplified - assumes some correlation)
        avg_correlation = 0.3  # Typical equity correlation
        portfolio.portfolio_volatility = np.sqrt(
            np.sum((weights * volatilities) ** 2) +
            2 * avg_correlation * np.sum(
                weights[i] * weights[j] * volatilities[i] * volatilities[j]
                for i in range(len(weights))
                for j in range(i + 1, len(weights))
            )
        )

        # Portfolio Sharpe
        portfolio.portfolio_sharpe = (
            (portfolio.expected_portfolio_return - self.risk_free_rate) /
            portfolio.portfolio_volatility
        ) if portfolio.portfolio_volatility > 0 else 0

        # Portfolio Beta
        portfolio.portfolio_beta = np.dot(weights, betas)

        # Run portfolio-level Monte Carlo
        self._run_portfolio_monte_carlo(portfolio, num_simulations)

        # Calculate sector weights
        sector_weights = {}
        for stock in stocks:
            sector = stock.sector or "Unknown"
            sector_weights[sector] = sector_weights.get(sector, 0) + stock.weight
        portfolio.sector_weights = sector_weights

        # Projected values
        portfolio.projected_value_12m = investment_amount * (1 + portfolio.mc_portfolio_median)
        portfolio.projected_gain_12m = portfolio.projected_value_12m - investment_amount

        return portfolio

    def _run_portfolio_monte_carlo(
        self,
        portfolio: PortfolioRecommendation,
        num_simulations: int
    ):
        """Run Monte Carlo simulation for the entire portfolio."""
        if not portfolio.stocks:
            return

        trading_days = 252

        # Get correlated returns data
        returns_data = []
        valid_stocks = []

        for stock in portfolio.stocks:
            hist = get_ticker_history(stock.symbol, period="2y")
            if hist is not None and not hist.empty:
                rets = hist['Close'].pct_change().dropna().values
                if len(rets) > 60:
                    returns_data.append(rets)
                    valid_stocks.append(stock)

        if len(returns_data) < 2:
            # Use individual stock simulations
            portfolio.mc_portfolio_mean = portfolio.expected_portfolio_return
            portfolio.mc_portfolio_median = portfolio.expected_portfolio_return * 0.95
            portfolio.mc_portfolio_std = portfolio.portfolio_volatility
            portfolio.mc_portfolio_5th = portfolio.expected_portfolio_return - 2 * portfolio.portfolio_volatility
            portfolio.mc_portfolio_95th = portfolio.expected_portfolio_return + 2 * portfolio.portfolio_volatility
            portfolio.mc_probability_profit = 0.65
            return

        # Align returns
        min_len = min(len(r) for r in returns_data)
        aligned = np.array([r[-min_len:] for r in returns_data])

        # Calculate parameters
        means = np.mean(aligned, axis=1)
        cov = np.cov(aligned)
        weights = np.array([s.weight for s in valid_stocks])

        # Normalize weights
        weights = weights / weights.sum()

        # Run simulations
        portfolio_returns = []

        for _ in range(num_simulations):
            # Generate correlated returns
            try:
                # Add small regularization to covariance matrix
                cov_reg = cov + np.eye(len(cov)) * 1e-6
                L = np.linalg.cholesky(cov_reg)

                cumulative_return = 0
                for _ in range(trading_days):
                    z = np.random.standard_normal(len(valid_stocks))
                    daily_returns = means + L @ z
                    portfolio_daily = np.dot(weights, daily_returns)
                    cumulative_return = (1 + cumulative_return) * (1 + portfolio_daily) - 1

                portfolio_returns.append(cumulative_return)
            except:
                # Fallback to uncorrelated simulation
                daily_returns = np.random.normal(
                    np.dot(weights, means),
                    np.sqrt(np.dot(weights**2, np.var(aligned, axis=1))),
                    trading_days
                )
                portfolio_returns.append(np.prod(1 + daily_returns) - 1)

        portfolio_returns = np.array(portfolio_returns)

        # Store results
        portfolio.mc_portfolio_mean = np.mean(portfolio_returns)
        portfolio.mc_portfolio_median = np.median(portfolio_returns)
        portfolio.mc_portfolio_std = np.std(portfolio_returns)
        portfolio.mc_portfolio_5th = np.percentile(portfolio_returns, 5)
        portfolio.mc_portfolio_25th = np.percentile(portfolio_returns, 25)
        portfolio.mc_portfolio_75th = np.percentile(portfolio_returns, 75)
        portfolio.mc_portfolio_95th = np.percentile(portfolio_returns, 95)
        portfolio.mc_probability_profit = np.mean(portfolio_returns > 0)
        portfolio.mc_probability_beat_market = np.mean(portfolio_returns > 0.10)  # Beat 10% market return

    def run_scenario_analysis(
        self,
        portfolio: PortfolioRecommendation,
        scenarios: dict[str, float] = None
    ) -> dict:
        """Run scenario analysis with different market conditions."""
        if scenarios is None:
            scenarios = {
                "Bull Market (+20%)": 0.20,
                "Moderate Growth (+10%)": 0.10,
                "Flat Market (0%)": 0.00,
                "Correction (-10%)": -0.10,
                "Bear Market (-20%)": -0.20,
                "Crash (-30%)": -0.30,
            }

        results = {}

        for scenario_name, market_return in scenarios.items():
            portfolio_return = 0
            for stock in portfolio.stocks:
                # Adjust return based on beta
                beta = stock.beta if stock.beta else 1.0
                stock_return = stock.expected_return_12m + beta * (market_return - 0.10)
                portfolio_return += stock.weight * stock_return

            projected_value = portfolio.investment_amount * (1 + portfolio_return)

            results[scenario_name] = {
                'market_return': market_return,
                'portfolio_return': portfolio_return,
                'projected_value': projected_value,
                'gain_loss': projected_value - portfolio.investment_amount
            }

        return results


# Singleton instance
_manager_instance = None

def get_investment_manager() -> InvestmentManager:
    """Get singleton InvestmentManager instance."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = InvestmentManager()
    return _manager_instance
