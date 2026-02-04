"""Portfolio Optimizer Agent - Builds optimal portfolios using Sharpe ratio optimization."""

from typing import Any, Optional
import asyncio

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.schemas import (
    PortfolioOptimization,
    PortfolioStock,
    AnalysisReport,
)


class PortfolioOptimizerAgent(BaseAgent):
    """
    Agent responsible for portfolio optimization.

    Builds optimal portfolios using:
    - Mean-Variance Optimization (Markowitz)
    - Maximum Sharpe Ratio optimization
    - Risk parity considerations
    - Sector/position constraints
    """

    def __init__(
        self,
        name: str = "PortfolioOptimizer",
        risk_free_rate: float = 0.05,
    ):
        super().__init__(name=name, agent_type="portfolio")
        self.risk_free_rate = risk_free_rate

    async def optimize(
        self,
        stock_reports: list[AnalysisReport],
        investment_amount: float = 100000,
        risk_tolerance: str = "moderate",
        max_position_size: float = 0.25,
        min_position_size: float = 0.02,
        max_stocks: int = 15,
        **kwargs,
    ) -> PortfolioOptimization:
        """
        Optimize portfolio based on stock analysis reports.

        Args:
            stock_reports: List of analysis reports for candidate stocks
            investment_amount: Total investment amount
            risk_tolerance: "conservative", "moderate", or "aggressive"
            max_position_size: Maximum weight for single position
            min_position_size: Minimum weight for positions
            max_stocks: Maximum number of stocks in portfolio

        Returns:
            PortfolioOptimization with optimal weights and metrics
        """
        self.log_info(f"Optimizing portfolio with {len(stock_reports)} candidates")

        try:
            # Filter to stocks with positive scores
            valid_reports = [
                r for r in stock_reports
                if r.overall_score and r.overall_score > 40
            ]

            if len(valid_reports) < 3:
                self.log_warning("Not enough valid stocks for optimization")
                return PortfolioOptimization(
                    portfolio_summary="Insufficient stocks meeting criteria for optimization"
                )

            # Get historical returns for optimization
            symbols = [r.symbol for r in valid_reports[:max_stocks]]
            returns_data = await self._get_returns_data(symbols)

            if returns_data.empty:
                return PortfolioOptimization(
                    portfolio_summary="Unable to fetch historical data for optimization"
                )

            # Calculate expected returns and covariance
            mean_returns = returns_data.mean() * 252  # Annualized
            cov_matrix = returns_data.cov() * 252  # Annualized

            # Adjust expected returns based on analysis scores
            adjusted_returns = self._adjust_returns_by_score(
                mean_returns, valid_reports
            )

            # Run optimization
            optimal_weights = self._optimize_sharpe(
                adjusted_returns,
                cov_matrix,
                risk_tolerance,
                max_position_size,
                min_position_size,
            )

            # Build portfolio
            portfolio = self._build_portfolio(
                optimal_weights,
                valid_reports,
                returns_data,
                cov_matrix,
                investment_amount,
            )

            # Add risk metrics
            portfolio = self._add_risk_metrics(portfolio, returns_data, cov_matrix)

            # Add benchmark comparison
            portfolio = await self._add_benchmark_comparison(portfolio)

            # Set constraints used
            portfolio.max_position_size = max_position_size
            portfolio.min_position_size = min_position_size
            portfolio.risk_tolerance = risk_tolerance
            portfolio.risk_free_rate = self.risk_free_rate

            # Generate summary
            portfolio.portfolio_summary = self._generate_summary(portfolio)
            portfolio.investment_rationale = self._generate_rationale(
                portfolio, valid_reports
            )

            return portfolio

        except Exception as e:
            self.log_warning(f"Error in portfolio optimization: {str(e)}")
            return PortfolioOptimization(
                portfolio_summary=f"Optimization failed: {str(e)}"
            )

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """
        Analyze method for compatibility with base agent.
        For portfolio optimization, use the optimize() method instead.
        """
        return {
            "message": "Use optimize() method for portfolio optimization",
            "score": None,
            "summary": "Portfolio optimizer requires multiple stocks",
        }

    async def _get_returns_data(
        self, symbols: list[str], period: str = "2y"
    ) -> pd.DataFrame:
        """Get historical returns for stocks."""
        try:
            data = yf.download(
                symbols,
                period=period,
                progress=False,
                auto_adjust=True,
            )["Close"]

            if isinstance(data, pd.Series):
                data = data.to_frame()

            # Calculate daily returns
            returns = data.pct_change().dropna()

            # Remove stocks with insufficient data
            min_observations = 252  # At least 1 year
            valid_cols = [col for col in returns.columns if returns[col].count() >= min_observations]
            returns = returns[valid_cols]

            return returns

        except Exception as e:
            logger.error(f"Error fetching returns data: {e}")
            return pd.DataFrame()

    def _adjust_returns_by_score(
        self,
        historical_returns: pd.Series,
        reports: list[AnalysisReport],
    ) -> pd.Series:
        """Adjust expected returns based on analysis scores."""
        adjusted = historical_returns.copy()

        for report in reports:
            if report.symbol in adjusted.index and report.overall_score:
                # Score-based adjustment factor
                # Score of 70 = 1.2x, Score of 50 = 1.0x, Score of 30 = 0.8x
                adjustment = 0.8 + (report.overall_score - 30) / 100
                adjusted[report.symbol] *= adjustment

        return adjusted

    def _optimize_sharpe(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        risk_tolerance: str,
        max_weight: float,
        min_weight: float,
    ) -> dict[str, float]:
        """
        Optimize portfolio for maximum Sharpe ratio.
        Uses numerical optimization with constraints.
        """
        n_assets = len(expected_returns)
        symbols = list(expected_returns.index)

        # Risk tolerance adjustment
        risk_multipliers = {
            "conservative": 0.5,
            "moderate": 1.0,
            "aggressive": 1.5,
        }
        risk_mult = risk_multipliers.get(risk_tolerance, 1.0)

        # Monte Carlo simulation for optimization
        # (Simple approach - production would use scipy.optimize)
        n_portfolios = 10000
        best_sharpe = -np.inf
        best_weights = None

        np.random.seed(42)

        for _ in range(n_portfolios):
            # Generate random weights
            weights = np.random.random(n_assets)

            # Apply constraints
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()  # Normalize

            # Ensure constraints are met
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()

            # Calculate portfolio metrics
            port_return = np.dot(weights, expected_returns)
            port_volatility = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix.values, weights))
            )

            # Calculate Sharpe ratio with risk adjustment
            excess_return = port_return - self.risk_free_rate
            adjusted_vol = port_volatility / risk_mult
            sharpe = excess_return / adjusted_vol if adjusted_vol > 0 else 0

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights.copy()

        # Convert to dictionary
        weights_dict = {}
        for i, symbol in enumerate(symbols):
            weight = best_weights[i] if best_weights is not None else 1/n_assets
            if weight >= min_weight:
                weights_dict[symbol] = float(weight)

        # Renormalize
        total = sum(weights_dict.values())
        weights_dict = {k: v/total for k, v in weights_dict.items()}

        return weights_dict

    def _build_portfolio(
        self,
        weights: dict[str, float],
        reports: list[AnalysisReport],
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
        investment_amount: float,
    ) -> PortfolioOptimization:
        """Build portfolio object with stock details."""
        portfolio = PortfolioOptimization()
        portfolio.stocks = []

        reports_dict = {r.symbol: r for r in reports}

        for symbol, weight in sorted(weights.items(), key=lambda x: -x[1]):
            report = reports_dict.get(symbol)

            # Get current price
            try:
                ticker = yf.Ticker(symbol)
                price = ticker.info.get("currentPrice") or ticker.info.get("regularMarketPrice")
            except Exception:
                price = None

            position_value = investment_amount * weight
            shares = position_value / price if price else None

            # Calculate individual stock metrics
            if symbol in returns.columns:
                stock_returns = returns[symbol]
                volatility = stock_returns.std() * np.sqrt(252)
                expected_return = stock_returns.mean() * 252
            else:
                volatility = None
                expected_return = None

            stock = PortfolioStock(
                symbol=symbol,
                name=report.company_name if report else None,
                weight=weight,
                shares=shares,
                current_price=price,
                position_value=position_value,
                overall_score=report.overall_score if report else None,
                fundamental_score=report.fundamental_analysis.score if report and report.fundamental_analysis else None,
                technical_score=report.technical_analysis.score if report and report.technical_analysis else None,
                risk_score=report.risk_assessment.score if report and report.risk_assessment else None,
                expected_return=expected_return,
                volatility=volatility,
                investment_thesis=report.investment_thesis if report else None,
                key_risks=report.key_risks[:3] if report else [],
            )

            portfolio.stocks.append(stock)

        portfolio.total_stocks = len(portfolio.stocks)

        # Calculate portfolio-level metrics
        weight_array = np.array([s.weight for s in portfolio.stocks])
        symbols_in_portfolio = [s.symbol for s in portfolio.stocks]

        # Filter returns and cov matrix to portfolio stocks
        portfolio_returns = returns[[s for s in symbols_in_portfolio if s in returns.columns]]
        portfolio_cov = cov_matrix.loc[
            [s for s in symbols_in_portfolio if s in cov_matrix.index],
            [s for s in symbols_in_portfolio if s in cov_matrix.columns]
        ]

        if not portfolio_returns.empty:
            mean_returns = portfolio_returns.mean() * 252

            # Align weights with available data
            available_weights = []
            for stock in portfolio.stocks:
                if stock.symbol in portfolio_returns.columns:
                    available_weights.append(stock.weight)

            if available_weights:
                weight_array = np.array(available_weights)
                weight_array = weight_array / weight_array.sum()

                portfolio.expected_return = float(np.dot(weight_array, mean_returns.values))
                portfolio.portfolio_volatility = float(np.sqrt(
                    np.dot(weight_array.T, np.dot(portfolio_cov.values, weight_array))
                ))

                if portfolio.portfolio_volatility > 0:
                    portfolio.sharpe_ratio = (
                        portfolio.expected_return - self.risk_free_rate
                    ) / portfolio.portfolio_volatility

        # Sector diversification
        sector_weights = {}
        for report in reports:
            if report.symbol in weights and report.sector:
                sector_weights[report.sector] = sector_weights.get(report.sector, 0) + weights[report.symbol]
        portfolio.sector_weights = sector_weights

        # Concentration metrics
        sorted_weights = sorted(weights.values(), reverse=True)
        portfolio.concentration_top5 = sum(sorted_weights[:5])

        return portfolio

    def _add_risk_metrics(
        self,
        portfolio: PortfolioOptimization,
        returns: pd.DataFrame,
        cov_matrix: pd.DataFrame,
    ) -> PortfolioOptimization:
        """Add detailed risk metrics to portfolio."""
        symbols = [s.symbol for s in portfolio.stocks]
        weights = np.array([s.weight for s in portfolio.stocks])

        # Filter to available data
        available_symbols = [s for s in symbols if s in returns.columns]
        if not available_symbols:
            return portfolio

        portfolio_returns_data = returns[available_symbols]

        # Calculate portfolio daily returns
        available_weights = []
        for stock in portfolio.stocks:
            if stock.symbol in available_symbols:
                available_weights.append(stock.weight)

        if not available_weights:
            return portfolio

        weight_array = np.array(available_weights)
        weight_array = weight_array / weight_array.sum()

        portfolio_daily_returns = portfolio_returns_data.dot(weight_array)

        # Value at Risk (Historical simulation)
        portfolio.var_95 = float(-np.percentile(portfolio_daily_returns, 5))
        portfolio.var_99 = float(-np.percentile(portfolio_daily_returns, 1))

        # Conditional VaR (Expected Shortfall)
        var_95_threshold = np.percentile(portfolio_daily_returns, 5)
        tail_returns = portfolio_daily_returns[portfolio_daily_returns <= var_95_threshold]
        if len(tail_returns) > 0:
            portfolio.cvar_95 = float(-tail_returns.mean())

        # Sortino Ratio
        downside_returns = portfolio_daily_returns[portfolio_daily_returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            if downside_std > 0 and portfolio.expected_return:
                portfolio.sortino_ratio = (
                    portfolio.expected_return - self.risk_free_rate
                ) / downside_std

        # Maximum Drawdown
        cumulative = (1 + portfolio_daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        portfolio.max_drawdown = float(abs(drawdowns.min()))

        # Average correlation
        corr_matrix = portfolio_returns_data.corr()
        n = len(corr_matrix)
        if n > 1:
            # Get upper triangle excluding diagonal
            upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]
            portfolio.correlation_avg = float(np.mean(upper_tri))

        # Diversification ratio
        individual_vols = portfolio_returns_data.std() * np.sqrt(252)
        weighted_vol = np.dot(weight_array, individual_vols)
        if portfolio.portfolio_volatility and portfolio.portfolio_volatility > 0:
            portfolio.diversification_ratio = weighted_vol / portfolio.portfolio_volatility

        return portfolio

    async def _add_benchmark_comparison(
        self, portfolio: PortfolioOptimization
    ) -> PortfolioOptimization:
        """Add benchmark comparison metrics."""
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="2y")

            if len(spy_hist) > 252:
                spy_returns = spy_hist['Close'].pct_change().dropna()

                portfolio.benchmark = "SPY"
                portfolio.benchmark_return = float(spy_returns.mean() * 252)
                portfolio.benchmark_volatility = float(spy_returns.std() * np.sqrt(252))

                # Alpha (simplified)
                if portfolio.expected_return and portfolio.benchmark_return:
                    portfolio.alpha = portfolio.expected_return - portfolio.benchmark_return

                # Information ratio
                if portfolio.alpha and portfolio.portfolio_volatility and portfolio.benchmark_volatility:
                    tracking_error = abs(portfolio.portfolio_volatility - portfolio.benchmark_volatility)
                    if tracking_error > 0:
                        portfolio.information_ratio = portfolio.alpha / tracking_error
                        portfolio.tracking_error = tracking_error

        except Exception as e:
            logger.warning(f"Error calculating benchmark metrics: {e}")

        return portfolio

    def _generate_summary(self, portfolio: PortfolioOptimization) -> str:
        """Generate portfolio summary."""
        parts = []

        parts.append(f"Optimized portfolio with {portfolio.total_stocks} stocks.")

        if portfolio.expected_return:
            parts.append(f"Expected annual return: {portfolio.expected_return:.1%}.")

        if portfolio.portfolio_volatility:
            parts.append(f"Portfolio volatility: {portfolio.portfolio_volatility:.1%}.")

        if portfolio.sharpe_ratio:
            parts.append(f"Sharpe ratio: {portfolio.sharpe_ratio:.2f}.")

        if portfolio.max_drawdown:
            parts.append(f"Max drawdown: {portfolio.max_drawdown:.1%}.")

        if portfolio.concentration_top5:
            parts.append(f"Top 5 holdings: {portfolio.concentration_top5:.1%} of portfolio.")

        if portfolio.alpha and portfolio.alpha > 0:
            parts.append(f"Expected alpha vs S&P 500: {portfolio.alpha:.1%}.")

        return " ".join(parts)

    def _generate_rationale(
        self,
        portfolio: PortfolioOptimization,
        reports: list[AnalysisReport],
    ) -> str:
        """Generate investment rationale."""
        parts = []

        # Top holdings rationale
        top_stocks = portfolio.stocks[:3]
        if top_stocks:
            top_names = [f"{s.symbol} ({s.weight:.0%})" for s in top_stocks]
            parts.append(f"Top holdings: {', '.join(top_names)}.")

        # Sector allocation
        if portfolio.sector_weights:
            top_sectors = sorted(
                portfolio.sector_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            sector_str = ", ".join([f"{s[0]} ({s[1]:.0%})" for s in top_sectors])
            parts.append(f"Sector focus: {sector_str}.")

        # Risk-return profile
        if portfolio.sharpe_ratio:
            if portfolio.sharpe_ratio > 1.5:
                parts.append("Excellent risk-adjusted returns.")
            elif portfolio.sharpe_ratio > 1.0:
                parts.append("Good risk-adjusted returns.")
            elif portfolio.sharpe_ratio > 0.5:
                parts.append("Moderate risk-adjusted returns.")

        # Diversification
        if portfolio.correlation_avg:
            if portfolio.correlation_avg < 0.3:
                parts.append("Well-diversified with low correlations.")
            elif portfolio.correlation_avg < 0.5:
                parts.append("Moderately diversified.")
            else:
                parts.append("Concentrated exposure - higher correlation.")

        return " ".join(parts)
