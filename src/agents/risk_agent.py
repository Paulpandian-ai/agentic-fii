"""Risk Assessment Agent - Evaluates risk factors and volatility."""

from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.schemas import RiskMetrics


class RiskAssessmentAgent(BaseAgent):
    """
    Agent responsible for risk assessment of stocks.

    Analyzes risk metrics including:
    - Volatility (daily and annualized)
    - Beta coefficient
    - Risk ratios (Sharpe, Sortino)
    - Value at Risk (VaR)
    - Maximum drawdown
    - Risk factors identification
    """

    def __init__(self, name: str = "RiskAgent", lookback_period: int = 252):
        """
        Initialize the Risk Assessment Agent.

        Args:
            name: Agent name
            lookback_period: Number of trading days for historical analysis
        """
        super().__init__(name=name, agent_type="risk")
        self.lookback_period = lookback_period
        self.risk_free_rate = 0.05  # Assume 5% risk-free rate

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """
        Perform risk assessment on the given stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing risk metrics and analysis
        """
        self.log_info(f"Assessing risk for {symbol}")

        try:
            # Fetch historical data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1y")

            if df.empty or len(df) < 30:
                raise ValueError(f"Insufficient price data for {symbol}")

            # Fetch market benchmark (S&P 500)
            market = yf.Ticker("SPY")
            market_df = market.history(period="1y")

            # Calculate risk metrics
            metrics = self._calculate_metrics(symbol, df, market_df)

            # Identify risk factors
            metrics.risk_factors = self._identify_risk_factors(metrics, ticker.info)

            # Determine risk level
            metrics.risk_level = self._determine_risk_level(metrics)

            # Calculate risk score (higher = safer)
            score = self._calculate_score(metrics)

            # Generate summary
            summary = self._generate_summary(metrics, score)

            return {
                "risk_metrics": metrics.model_dump(),
                "score": score,
                "summary": summary,
            }

        except Exception as e:
            self.log_warning(f"Error in risk assessment: {str(e)}")
            return {
                "risk_metrics": RiskMetrics(symbol=symbol).model_dump(),
                "score": None,
                "summary": f"Unable to perform risk assessment: {str(e)}",
            }

    def _calculate_metrics(
        self, symbol: str, df: pd.DataFrame, market_df: pd.DataFrame
    ) -> RiskMetrics:
        """Calculate all risk metrics from price data."""
        close = df["Close"]
        returns = close.pct_change().dropna()

        metrics = RiskMetrics(symbol=symbol)

        # Daily and annualized volatility
        metrics.volatility_daily = returns.std()
        metrics.volatility_annual = metrics.volatility_daily * np.sqrt(252)

        # Beta calculation
        if len(market_df) > 0:
            market_returns = market_df["Close"].pct_change().dropna()
            # Align dates
            common_dates = returns.index.intersection(market_returns.index)
            if len(common_dates) > 30:
                aligned_returns = returns.loc[common_dates]
                aligned_market = market_returns.loc[common_dates]
                covariance = aligned_returns.cov(aligned_market)
                market_variance = aligned_market.var()
                if market_variance > 0:
                    metrics.beta = covariance / market_variance

        # Sharpe Ratio (annualized)
        excess_returns = returns.mean() * 252 - self.risk_free_rate
        if metrics.volatility_annual and metrics.volatility_annual > 0:
            metrics.sharpe_ratio = excess_returns / metrics.volatility_annual

        # Sortino Ratio (only considering downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std() * np.sqrt(252)
            if downside_std > 0:
                metrics.sortino_ratio = excess_returns / downside_std

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics.max_drawdown = abs(drawdown.min())

        # Current Drawdown
        current_dd = (cumulative.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1]
        metrics.current_drawdown = abs(current_dd)

        # Value at Risk (VaR)
        metrics.var_95 = abs(np.percentile(returns, 5))
        metrics.var_99 = abs(np.percentile(returns, 1))

        return metrics

    def _identify_risk_factors(
        self, metrics: RiskMetrics, info: dict
    ) -> list[str]:
        """Identify key risk factors."""
        risk_factors = []

        # Volatility risks
        if metrics.volatility_annual and metrics.volatility_annual > 0.4:
            risk_factors.append("High volatility (>40% annualized)")
        elif metrics.volatility_annual and metrics.volatility_annual > 0.25:
            risk_factors.append("Moderate volatility (>25% annualized)")

        # Beta risks
        if metrics.beta is not None:
            if metrics.beta > 1.5:
                risk_factors.append("High market sensitivity (beta > 1.5)")
            elif metrics.beta < 0:
                risk_factors.append("Negative correlation with market")

        # Drawdown risks
        if metrics.max_drawdown and metrics.max_drawdown > 0.3:
            risk_factors.append(f"Significant max drawdown ({metrics.max_drawdown:.1%})")

        if metrics.current_drawdown and metrics.current_drawdown > 0.1:
            risk_factors.append(f"Currently in drawdown ({metrics.current_drawdown:.1%})")

        # Sharpe ratio
        if metrics.sharpe_ratio is not None and metrics.sharpe_ratio < 0:
            risk_factors.append("Negative risk-adjusted returns (Sharpe < 0)")

        # Check fundamental risk factors from info
        debt_to_equity = info.get("debtToEquity")
        if debt_to_equity and debt_to_equity > 200:
            risk_factors.append("High debt levels (D/E > 200%)")

        current_ratio = info.get("currentRatio")
        if current_ratio and current_ratio < 1:
            risk_factors.append("Liquidity concern (current ratio < 1)")

        profit_margin = info.get("profitMargins")
        if profit_margin and profit_margin < 0:
            risk_factors.append("Negative profit margins")

        return risk_factors

    def _determine_risk_level(self, metrics: RiskMetrics) -> str:
        """Determine overall risk level."""
        risk_score = 0

        # Volatility contribution
        if metrics.volatility_annual:
            if metrics.volatility_annual > 0.5:
                risk_score += 3
            elif metrics.volatility_annual > 0.35:
                risk_score += 2
            elif metrics.volatility_annual > 0.2:
                risk_score += 1

        # Beta contribution
        if metrics.beta is not None:
            if metrics.beta > 1.5:
                risk_score += 2
            elif metrics.beta > 1.2:
                risk_score += 1

        # Max drawdown contribution
        if metrics.max_drawdown:
            if metrics.max_drawdown > 0.4:
                risk_score += 2
            elif metrics.max_drawdown > 0.25:
                risk_score += 1

        # Risk factors contribution
        risk_score += len(metrics.risk_factors) * 0.5

        # Determine level
        if risk_score >= 6:
            return "very_high"
        elif risk_score >= 4:
            return "high"
        elif risk_score >= 2:
            return "medium"
        else:
            return "low"

    def _calculate_score(self, metrics: RiskMetrics) -> float:
        """
        Calculate risk score from 0-100.
        Higher score = SAFER (less risky).
        """
        score = 70.0  # Start slightly above neutral

        # Volatility penalty (up to -30 points)
        if metrics.volatility_annual:
            if metrics.volatility_annual > 0.5:
                score -= 30
            elif metrics.volatility_annual > 0.35:
                score -= 20
            elif metrics.volatility_annual > 0.25:
                score -= 10
            elif metrics.volatility_annual < 0.15:
                score += 10

        # Beta adjustment (up to +/- 15 points)
        if metrics.beta is not None:
            if 0.8 <= metrics.beta <= 1.2:
                score += 5  # Market-like beta is neutral
            elif metrics.beta > 1.5:
                score -= 15
            elif metrics.beta < 0.5:
                score += 10  # Low beta is safer

        # Sharpe ratio bonus (up to +15 points)
        if metrics.sharpe_ratio is not None:
            if metrics.sharpe_ratio > 1.5:
                score += 15
            elif metrics.sharpe_ratio > 1.0:
                score += 10
            elif metrics.sharpe_ratio > 0.5:
                score += 5
            elif metrics.sharpe_ratio < 0:
                score -= 10

        # Max drawdown penalty (up to -20 points)
        if metrics.max_drawdown:
            if metrics.max_drawdown > 0.4:
                score -= 20
            elif metrics.max_drawdown > 0.25:
                score -= 10
            elif metrics.max_drawdown < 0.15:
                score += 5

        # Risk factors penalty
        score -= len(metrics.risk_factors) * 3

        return max(0, min(100, score))

    def _generate_summary(self, metrics: RiskMetrics, score: float) -> str:
        """Generate a summary of the risk assessment."""
        parts = []

        # Risk level
        if metrics.risk_level:
            level_desc = {
                "low": "low risk",
                "medium": "moderate risk",
                "high": "high risk",
                "very_high": "very high risk",
            }
            parts.append(f"This stock presents {level_desc.get(metrics.risk_level, metrics.risk_level)}.")

        # Volatility
        if metrics.volatility_annual:
            parts.append(f"Annualized volatility is {metrics.volatility_annual:.1%}.")

        # Beta
        if metrics.beta is not None:
            if metrics.beta > 1:
                parts.append(f"Beta of {metrics.beta:.2f} indicates higher market sensitivity.")
            elif metrics.beta < 1:
                parts.append(f"Beta of {metrics.beta:.2f} indicates lower market sensitivity.")

        # Sharpe ratio
        if metrics.sharpe_ratio is not None:
            if metrics.sharpe_ratio > 1:
                parts.append("Risk-adjusted returns are favorable.")
            elif metrics.sharpe_ratio < 0:
                parts.append("Risk-adjusted returns are negative.")

        # Key risk factors
        if metrics.risk_factors:
            if len(metrics.risk_factors) <= 2:
                parts.append(f"Key risks: {', '.join(metrics.risk_factors)}.")
            else:
                parts.append(f"Identified {len(metrics.risk_factors)} risk factors.")

        # Score interpretation
        if score >= 70:
            parts.append("Overall risk profile is manageable.")
        elif score <= 30:
            parts.append("Overall risk profile warrants caution.")

        return " ".join(parts)
