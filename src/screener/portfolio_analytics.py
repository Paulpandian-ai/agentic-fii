"""
Portfolio Analytics Module for Asset Managers

Provides comprehensive portfolio risk and performance analytics including:
- Risk metrics (VaR, Sharpe, Sortino, Max Drawdown)
- Correlation analysis
- Sector allocation
- Benchmark comparison
- Position sizing recommendations
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np

from src.utils.yfinance_cache import (
    get_ticker_info,
    get_ticker_history,
    batch_download,
)


@dataclass
class RiskMetrics:
    """Portfolio risk metrics."""
    volatility_annual: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    var_95: float = 0.0  # 95% Value at Risk (daily)
    var_99: float = 0.0  # 99% Value at Risk (daily)
    cvar_95: float = 0.0  # Conditional VaR (Expected Shortfall)
    beta: float = 0.0
    alpha: float = 0.0
    r_squared: float = 0.0
    treynor_ratio: float = 0.0
    information_ratio: float = 0.0
    calmar_ratio: float = 0.0


@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    ytd_return: float = 0.0
    mtd_return: float = 0.0
    return_1w: float = 0.0
    return_1m: float = 0.0
    return_3m: float = 0.0
    return_6m: float = 0.0
    return_1y: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0
    positive_days_pct: float = 0.0
    avg_daily_return: float = 0.0


@dataclass
class SectorAllocation:
    """Sector allocation breakdown."""
    sector: str
    weight: float
    value: float
    stock_count: int


@dataclass
class CorrelationData:
    """Correlation matrix data."""
    symbols: list[str] = field(default_factory=list)
    matrix: list[list[float]] = field(default_factory=list)


@dataclass
class PositionAnalysis:
    """Individual position analysis."""
    symbol: str
    name: str
    weight: float
    value: float
    shares: float
    avg_cost: float
    current_price: float
    gain_loss: float
    gain_loss_pct: float
    contribution_to_return: float
    sector: str
    beta: float
    correlation_to_portfolio: float


@dataclass
class BenchmarkComparison:
    """Comparison against benchmark."""
    benchmark_symbol: str
    benchmark_name: str
    portfolio_return: float
    benchmark_return: float
    excess_return: float  # Alpha
    tracking_error: float
    information_ratio: float
    up_capture: float  # Upside capture ratio
    down_capture: float  # Downside capture ratio
    batting_average: float  # % of periods outperforming


class PortfolioAnalytics:
    """
    Comprehensive portfolio analytics for asset managers.

    Calculates risk metrics, performance attribution, and provides
    insights for portfolio optimization.
    """

    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize portfolio analytics.

        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf = risk_free_rate / 252

    def calculate_risk_metrics(
        self,
        returns: list[float],
        benchmark_returns: list[float] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.

        Args:
            returns: Daily portfolio returns
            benchmark_returns: Daily benchmark returns (optional)

        Returns:
            RiskMetrics object with all risk calculations
        """
        if not returns or len(returns) < 20:
            return RiskMetrics()

        returns_arr = np.array(returns)

        # Basic volatility
        volatility_daily = np.std(returns_arr)
        volatility_annual = volatility_daily * np.sqrt(252)

        # Sharpe Ratio
        excess_returns = returns_arr - self.daily_rf
        sharpe = (np.mean(excess_returns) / volatility_daily) * np.sqrt(252) if volatility_daily > 0 else 0

        # Sortino Ratio (only downside deviation)
        downside_returns = returns_arr[returns_arr < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = (np.mean(excess_returns) / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns_arr)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = np.min(drawdowns)

        # Drawdown duration
        dd_duration = self._calculate_drawdown_duration(drawdowns)

        # Value at Risk
        var_95 = np.percentile(returns_arr, 5)
        var_99 = np.percentile(returns_arr, 1)

        # Conditional VaR (Expected Shortfall)
        cvar_95 = np.mean(returns_arr[returns_arr <= var_95])

        # Calmar Ratio
        annualized_return = np.mean(returns_arr) * 252
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        # Beta, Alpha, R-squared (if benchmark provided)
        beta, alpha, r_squared = 0, 0, 0
        treynor, info_ratio = 0, 0

        if benchmark_returns and len(benchmark_returns) == len(returns):
            bench_arr = np.array(benchmark_returns)

            # Beta calculation
            covariance = np.cov(returns_arr, bench_arr)[0, 1]
            bench_variance = np.var(bench_arr)
            beta = covariance / bench_variance if bench_variance > 0 else 0

            # Alpha (Jensen's Alpha)
            portfolio_excess = np.mean(returns_arr) - self.daily_rf
            benchmark_excess = np.mean(bench_arr) - self.daily_rf
            alpha = (portfolio_excess - beta * benchmark_excess) * 252

            # R-squared
            correlation = np.corrcoef(returns_arr, bench_arr)[0, 1]
            r_squared = correlation ** 2

            # Treynor Ratio
            treynor = (annualized_return - self.risk_free_rate) / beta if beta != 0 else 0

            # Information Ratio
            tracking_diff = returns_arr - bench_arr
            tracking_error = np.std(tracking_diff) * np.sqrt(252)
            info_ratio = np.mean(tracking_diff) * 252 / tracking_error if tracking_error > 0 else 0

        return RiskMetrics(
            volatility_annual=volatility_annual,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration_days=dd_duration,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            r_squared=r_squared,
            treynor_ratio=treynor,
            information_ratio=info_ratio,
            calmar_ratio=calmar
        )

    def _calculate_drawdown_duration(self, drawdowns: np.ndarray) -> int:
        """Calculate the longest drawdown duration in days."""
        if len(drawdowns) == 0:
            return 0

        in_drawdown = drawdowns < 0
        max_duration = 0
        current_duration = 0

        for dd in in_drawdown:
            if dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def calculate_performance_metrics(
        self,
        returns: list[float],
        dates: list[datetime] = None
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics over various time periods.

        Args:
            returns: Daily returns
            dates: Corresponding dates (optional)

        Returns:
            PerformanceMetrics object
        """
        if not returns:
            return PerformanceMetrics()

        returns_arr = np.array(returns)

        # Total and annualized return
        cumulative_return = np.prod(1 + returns_arr) - 1
        n_days = len(returns_arr)
        annualized = (1 + cumulative_return) ** (252 / n_days) - 1 if n_days > 0 else 0

        # Period returns
        return_1w = np.prod(1 + returns_arr[-5:]) - 1 if len(returns_arr) >= 5 else 0
        return_1m = np.prod(1 + returns_arr[-21:]) - 1 if len(returns_arr) >= 21 else 0
        return_3m = np.prod(1 + returns_arr[-63:]) - 1 if len(returns_arr) >= 63 else 0
        return_6m = np.prod(1 + returns_arr[-126:]) - 1 if len(returns_arr) >= 126 else 0
        return_1y = np.prod(1 + returns_arr[-252:]) - 1 if len(returns_arr) >= 252 else cumulative_return

        # Best/Worst
        best_day = np.max(returns_arr)
        worst_day = np.min(returns_arr)

        # Monthly returns for best/worst month
        if len(returns_arr) >= 21:
            monthly_returns = []
            for i in range(0, len(returns_arr) - 20, 21):
                month_ret = np.prod(1 + returns_arr[i:i+21]) - 1
                monthly_returns.append(month_ret)
            best_month = max(monthly_returns) if monthly_returns else 0
            worst_month = min(monthly_returns) if monthly_returns else 0
        else:
            best_month = worst_month = 0

        # Positive days percentage
        positive_days = np.sum(returns_arr > 0) / len(returns_arr) if len(returns_arr) > 0 else 0

        # YTD and MTD (approximate if no dates)
        ytd_return = return_1y  # Approximation
        mtd_return = return_1m  # Approximation

        return PerformanceMetrics(
            total_return=cumulative_return,
            annualized_return=annualized,
            ytd_return=ytd_return,
            mtd_return=mtd_return,
            return_1w=return_1w,
            return_1m=return_1m,
            return_3m=return_3m,
            return_6m=return_6m,
            return_1y=return_1y,
            best_day=best_day,
            worst_day=worst_day,
            best_month=best_month,
            worst_month=worst_month,
            positive_days_pct=positive_days,
            avg_daily_return=np.mean(returns_arr)
        )

    def calculate_correlation_matrix(
        self,
        symbols: list[str],
        period: str = "1y"
    ) -> CorrelationData:
        """
        Calculate correlation matrix for a list of symbols.

        Args:
            symbols: List of stock symbols
            period: Time period for historical data

        Returns:
            CorrelationData with symbols and correlation matrix
        """
        if not symbols or len(symbols) < 2:
            return CorrelationData(symbols=symbols, matrix=[])

        # Get historical data
        returns_data = {}
        for symbol in symbols:
            hist = get_ticker_history(symbol, period=period)
            if hist is not None and not hist.empty and 'Close' in hist.columns:
                returns_data[symbol] = hist['Close'].pct_change().dropna().values

        if len(returns_data) < 2:
            return CorrelationData(symbols=symbols, matrix=[])

        # Align all return series to same length
        min_length = min(len(r) for r in returns_data.values())
        aligned_returns = {s: r[-min_length:] for s, r in returns_data.items()}

        # Build correlation matrix
        valid_symbols = list(aligned_returns.keys())
        n = len(valid_symbols)
        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    corr = np.corrcoef(
                        aligned_returns[valid_symbols[i]],
                        aligned_returns[valid_symbols[j]]
                    )[0, 1]
                    matrix[i, j] = corr

        return CorrelationData(
            symbols=valid_symbols,
            matrix=matrix.tolist()
        )

    def calculate_sector_allocation(
        self,
        positions: dict[str, float]  # symbol -> value
    ) -> list[SectorAllocation]:
        """
        Calculate sector allocation for portfolio positions.

        Args:
            positions: Dict mapping symbol to position value

        Returns:
            List of SectorAllocation objects
        """
        if not positions:
            return []

        total_value = sum(positions.values())
        sector_data = {}

        for symbol, value in positions.items():
            info = get_ticker_info(symbol)
            sector = info.get('sector', 'Unknown') if info else 'Unknown'

            if sector not in sector_data:
                sector_data[sector] = {'value': 0, 'count': 0}
            sector_data[sector]['value'] += value
            sector_data[sector]['count'] += 1

        allocations = []
        for sector, data in sector_data.items():
            allocations.append(SectorAllocation(
                sector=sector,
                weight=data['value'] / total_value if total_value > 0 else 0,
                value=data['value'],
                stock_count=data['count']
            ))

        return sorted(allocations, key=lambda x: x.weight, reverse=True)

    def compare_to_benchmark(
        self,
        portfolio_returns: list[float],
        benchmark_symbol: str = "SPY",
        period: str = "1y"
    ) -> BenchmarkComparison:
        """
        Compare portfolio performance to a benchmark.

        Args:
            portfolio_returns: Daily portfolio returns
            benchmark_symbol: Benchmark ticker (default SPY)
            period: Comparison period

        Returns:
            BenchmarkComparison object
        """
        if not portfolio_returns:
            return BenchmarkComparison(
                benchmark_symbol=benchmark_symbol,
                benchmark_name="S&P 500 ETF",
                portfolio_return=0, benchmark_return=0,
                excess_return=0, tracking_error=0,
                information_ratio=0, up_capture=0,
                down_capture=0, batting_average=0
            )

        # Get benchmark data
        bench_hist = get_ticker_history(benchmark_symbol, period=period)
        if bench_hist is None or bench_hist.empty:
            return BenchmarkComparison(
                benchmark_symbol=benchmark_symbol,
                benchmark_name="",
                portfolio_return=0, benchmark_return=0,
                excess_return=0, tracking_error=0,
                information_ratio=0, up_capture=0,
                down_capture=0, batting_average=0
            )

        bench_returns = bench_hist['Close'].pct_change().dropna().values

        # Align lengths
        min_len = min(len(portfolio_returns), len(bench_returns))
        port_arr = np.array(portfolio_returns[-min_len:])
        bench_arr = bench_returns[-min_len:]

        # Calculate returns
        port_total = np.prod(1 + port_arr) - 1
        bench_total = np.prod(1 + bench_arr) - 1
        excess = port_total - bench_total

        # Tracking error
        tracking_diff = port_arr - bench_arr
        tracking_error = np.std(tracking_diff) * np.sqrt(252)

        # Information ratio
        info_ratio = np.mean(tracking_diff) * 252 / tracking_error if tracking_error > 0 else 0

        # Up/Down capture
        up_periods = bench_arr > 0
        down_periods = bench_arr < 0

        up_capture = 0
        if np.sum(up_periods) > 0:
            port_up = np.mean(port_arr[up_periods])
            bench_up = np.mean(bench_arr[up_periods])
            up_capture = port_up / bench_up if bench_up > 0 else 0

        down_capture = 0
        if np.sum(down_periods) > 0:
            port_down = np.mean(port_arr[down_periods])
            bench_down = np.mean(bench_arr[down_periods])
            down_capture = port_down / bench_down if bench_down < 0 else 0

        # Batting average (% of periods outperforming)
        batting_avg = np.mean(port_arr > bench_arr)

        # Get benchmark name
        bench_info = get_ticker_info(benchmark_symbol)
        bench_name = bench_info.get('shortName', benchmark_symbol) if bench_info else benchmark_symbol

        return BenchmarkComparison(
            benchmark_symbol=benchmark_symbol,
            benchmark_name=bench_name,
            portfolio_return=port_total,
            benchmark_return=bench_total,
            excess_return=excess,
            tracking_error=tracking_error,
            information_ratio=info_ratio,
            up_capture=up_capture,
            down_capture=down_capture,
            batting_average=batting_avg
        )

    def calculate_position_sizes(
        self,
        portfolio_value: float,
        symbols: list[str],
        method: str = "equal_weight",
        risk_budget: float = 0.02,
        target_volatility: float = 0.15
    ) -> dict[str, float]:
        """
        Calculate optimal position sizes.

        Args:
            portfolio_value: Total portfolio value
            symbols: List of symbols to allocate
            method: Allocation method (equal_weight, risk_parity, min_variance)
            risk_budget: Max risk per position (for risk-based methods)
            target_volatility: Target portfolio volatility

        Returns:
            Dict mapping symbol to recommended position value
        """
        if not symbols:
            return {}

        if method == "equal_weight":
            weight = 1.0 / len(symbols)
            return {s: portfolio_value * weight for s in symbols}

        elif method == "risk_parity":
            # Calculate volatility for each symbol
            volatilities = {}
            for symbol in symbols:
                hist = get_ticker_history(symbol, period="1y")
                if hist is not None and not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    vol = returns.std() * np.sqrt(252)
                    volatilities[symbol] = max(vol, 0.01)  # Min 1% vol
                else:
                    volatilities[symbol] = 0.20  # Default 20%

            # Inverse volatility weighting
            inv_vols = {s: 1/v for s, v in volatilities.items()}
            total_inv_vol = sum(inv_vols.values())
            weights = {s: iv / total_inv_vol for s, iv in inv_vols.items()}

            return {s: portfolio_value * w for s, w in weights.items()}

        else:
            # Default to equal weight
            weight = 1.0 / len(symbols)
            return {s: portfolio_value * weight for s in symbols}

    def get_concentration_risk(
        self,
        positions: dict[str, float]
    ) -> dict:
        """
        Analyze portfolio concentration risk.

        Args:
            positions: Dict mapping symbol to position value

        Returns:
            Dict with concentration metrics
        """
        if not positions:
            return {}

        total = sum(positions.values())
        weights = sorted([v/total for v in positions.values()], reverse=True)

        # Herfindahl-Hirschman Index (HHI)
        hhi = sum(w**2 for w in weights)

        # Effective number of positions
        effective_n = 1 / hhi if hhi > 0 else len(positions)

        # Top holdings concentration
        top_5_weight = sum(weights[:5]) if len(weights) >= 5 else sum(weights)
        top_10_weight = sum(weights[:10]) if len(weights) >= 10 else sum(weights)

        return {
            'hhi': hhi,
            'effective_positions': effective_n,
            'total_positions': len(positions),
            'top_5_concentration': top_5_weight,
            'top_10_concentration': top_10_weight,
            'max_position_weight': weights[0] if weights else 0,
            'diversification_ratio': effective_n / len(positions) if positions else 0
        }


# Singleton instance
_analytics_instance = None

def get_portfolio_analytics(risk_free_rate: float = 0.05) -> PortfolioAnalytics:
    """Get singleton PortfolioAnalytics instance."""
    global _analytics_instance
    if _analytics_instance is None:
        _analytics_instance = PortfolioAnalytics(risk_free_rate)
    return _analytics_instance
