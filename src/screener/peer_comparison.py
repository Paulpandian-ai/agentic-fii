"""
Peer Comparison Module for Stock Analysis

Provides comprehensive peer/sector comparison including:
- Valuation comparison (P/E, P/B, P/S, EV/EBITDA)
- Growth comparison (Revenue, Earnings, EPS growth)
- Profitability comparison (Margins, ROE, ROA)
- Performance comparison (Returns over various periods)
- Size comparison (Market Cap, Revenue, Employees)
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np

from src.utils.yfinance_cache import get_ticker_info, get_ticker_history
from src.screener.stock_universe import get_sector_stocks


class ComparisonMetric(Enum):
    VALUATION = "Valuation"
    GROWTH = "Growth"
    PROFITABILITY = "Profitability"
    PERFORMANCE = "Performance"
    SIZE = "Size"
    EFFICIENCY = "Efficiency"


@dataclass
class PeerMetrics:
    """Metrics for a single peer."""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float = 0.0

    # Valuation
    pe_ratio: float = 0.0
    forward_pe: float = 0.0
    pb_ratio: float = 0.0
    ps_ratio: float = 0.0
    ev_ebitda: float = 0.0
    peg_ratio: float = 0.0

    # Growth
    revenue_growth: float = 0.0
    earnings_growth: float = 0.0
    eps_growth: float = 0.0

    # Profitability
    gross_margin: float = 0.0
    operating_margin: float = 0.0
    profit_margin: float = 0.0
    roe: float = 0.0
    roa: float = 0.0
    roic: float = 0.0

    # Performance
    return_1m: float = 0.0
    return_3m: float = 0.0
    return_6m: float = 0.0
    return_1y: float = 0.0
    return_ytd: float = 0.0

    # Other
    dividend_yield: float = 0.0
    beta: float = 0.0
    current_price: float = 0.0
    target_price: float = 0.0
    analyst_rating: str = ""

    # Rankings (percentile within peer group)
    valuation_rank: int = 0
    growth_rank: int = 0
    profitability_rank: int = 0
    performance_rank: int = 0
    overall_rank: int = 0


@dataclass
class PeerGroupStats:
    """Aggregate statistics for peer group."""
    metric_name: str
    peer_count: int
    mean: float
    median: float
    std: float
    min_val: float
    max_val: float
    percentile_25: float
    percentile_75: float
    target_value: float  # Value for the target company
    target_percentile: float  # Where target ranks


@dataclass
class PeerComparison:
    """Complete peer comparison result."""
    target_symbol: str
    target_name: str
    sector: str
    industry: str
    peers: list[PeerMetrics] = field(default_factory=list)
    stats: dict[str, PeerGroupStats] = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    overall_ranking: str = ""  # "Top Tier", "Above Average", "Average", "Below Average"


class PeerAnalyzer:
    """
    Peer comparison analysis engine.

    Compares a target stock against its sector/industry peers
    across multiple dimensions.
    """

    def __init__(self):
        self.cache = {}

    def get_peers(
        self,
        symbol: str,
        max_peers: int = 15,
        same_industry: bool = True
    ) -> list[str]:
        """
        Get peer companies for a given symbol.

        Args:
            symbol: Target stock symbol
            max_peers: Maximum number of peers to return
            same_industry: If True, prioritize same industry peers

        Returns:
            List of peer symbols
        """
        info = get_ticker_info(symbol)
        if not info:
            return []

        sector = info.get('sector', '')
        industry = info.get('industry', '')
        market_cap = info.get('marketCap', 0)

        if not sector:
            return []

        # Get all stocks in sector
        sector_stocks = get_sector_stocks(sector)
        sector_stocks = [s for s in sector_stocks if s != symbol]

        if not sector_stocks:
            return []

        # Get info for each stock and filter/sort
        peer_info = []
        for peer_symbol in sector_stocks[:50]:  # Limit initial fetch
            peer_data = get_ticker_info(peer_symbol)
            if peer_data:
                peer_info.append({
                    'symbol': peer_symbol,
                    'industry': peer_data.get('industry', ''),
                    'market_cap': peer_data.get('marketCap', 0)
                })

        # Sort by relevance
        def relevance_score(peer):
            score = 0
            # Same industry bonus
            if same_industry and peer['industry'] == industry:
                score += 100
            # Similar market cap bonus
            if market_cap > 0 and peer['market_cap'] > 0:
                ratio = peer['market_cap'] / market_cap
                if 0.2 <= ratio <= 5:
                    score += 50
                if 0.5 <= ratio <= 2:
                    score += 25
            return score

        peer_info.sort(key=relevance_score, reverse=True)

        return [p['symbol'] for p in peer_info[:max_peers]]

    def analyze_peer(self, symbol: str) -> Optional[PeerMetrics]:
        """
        Get comprehensive metrics for a single stock.

        Args:
            symbol: Stock symbol

        Returns:
            PeerMetrics object or None
        """
        info = get_ticker_info(symbol)
        if not info:
            return None

        hist = get_ticker_history(symbol, period="1y")

        # Calculate returns from history
        returns = {}
        if hist is not None and not hist.empty and 'Close' in hist.columns:
            close = hist['Close'].values
            if len(close) >= 5:
                returns['1m'] = (close[-1] / close[-21] - 1) if len(close) >= 21 else 0
                returns['3m'] = (close[-1] / close[-63] - 1) if len(close) >= 63 else 0
                returns['6m'] = (close[-1] / close[-126] - 1) if len(close) >= 126 else 0
                returns['1y'] = (close[-1] / close[0] - 1)

        return PeerMetrics(
            symbol=symbol,
            name=info.get('shortName', symbol),
            sector=info.get('sector', 'Unknown'),
            industry=info.get('industry', 'Unknown'),
            market_cap=info.get('marketCap', 0),
            pe_ratio=info.get('trailingPE', 0) or 0,
            forward_pe=info.get('forwardPE', 0) or 0,
            pb_ratio=info.get('priceToBook', 0) or 0,
            ps_ratio=info.get('priceToSalesTrailing12Months', 0) or 0,
            ev_ebitda=info.get('enterpriseToEbitda', 0) or 0,
            peg_ratio=info.get('pegRatio', 0) or 0,
            revenue_growth=info.get('revenueGrowth', 0) or 0,
            earnings_growth=info.get('earningsGrowth', 0) or 0,
            eps_growth=info.get('earningsQuarterlyGrowth', 0) or 0,
            gross_margin=info.get('grossMargins', 0) or 0,
            operating_margin=info.get('operatingMargins', 0) or 0,
            profit_margin=info.get('profitMargins', 0) or 0,
            roe=info.get('returnOnEquity', 0) or 0,
            roa=info.get('returnOnAssets', 0) or 0,
            roic=0,  # Not directly available
            return_1m=returns.get('1m', 0),
            return_3m=returns.get('3m', 0),
            return_6m=returns.get('6m', 0),
            return_1y=returns.get('1y', 0),
            return_ytd=returns.get('1y', 0),  # Approximation
            dividend_yield=info.get('dividendYield', 0) or 0,
            beta=info.get('beta', 0) or 0,
            current_price=info.get('currentPrice') or info.get('regularMarketPrice', 0) or 0,
            target_price=info.get('targetMeanPrice', 0) or 0,
            analyst_rating=info.get('recommendationKey', '') or ''
        )

    def compare(
        self,
        symbol: str,
        peers: list[str] = None,
        max_peers: int = 10
    ) -> PeerComparison:
        """
        Perform comprehensive peer comparison.

        Args:
            symbol: Target stock symbol
            peers: Optional list of peer symbols (auto-detected if not provided)
            max_peers: Maximum number of peers to compare

        Returns:
            PeerComparison object with full analysis
        """
        # Get target metrics
        target_metrics = self.analyze_peer(symbol)
        if not target_metrics:
            return PeerComparison(
                target_symbol=symbol,
                target_name=symbol,
                sector="Unknown",
                industry="Unknown"
            )

        # Get peers if not provided
        if peers is None:
            peers = self.get_peers(symbol, max_peers=max_peers)

        # Analyze all peers
        peer_metrics = [target_metrics]  # Include target in analysis
        for peer in peers[:max_peers]:
            metrics = self.analyze_peer(peer)
            if metrics:
                peer_metrics.append(metrics)

        if len(peer_metrics) < 2:
            return PeerComparison(
                target_symbol=symbol,
                target_name=target_metrics.name,
                sector=target_metrics.sector,
                industry=target_metrics.industry,
                peers=[target_metrics]
            )

        # Calculate statistics and rankings
        stats = self._calculate_stats(target_metrics, peer_metrics)

        # Calculate rankings
        self._calculate_rankings(peer_metrics)

        # Identify strengths and weaknesses
        strengths, weaknesses = self._identify_strengths_weaknesses(target_metrics, stats)

        # Determine overall ranking
        overall_ranking = self._determine_overall_ranking(target_metrics, peer_metrics)

        return PeerComparison(
            target_symbol=symbol,
            target_name=target_metrics.name,
            sector=target_metrics.sector,
            industry=target_metrics.industry,
            peers=peer_metrics,
            stats=stats,
            strengths=strengths,
            weaknesses=weaknesses,
            overall_ranking=overall_ranking
        )

    def _calculate_stats(
        self,
        target: PeerMetrics,
        peers: list[PeerMetrics]
    ) -> dict[str, PeerGroupStats]:
        """Calculate statistical comparison for each metric."""
        metrics_config = {
            'P/E Ratio': ('pe_ratio', False),  # (attr, higher_is_better)
            'Forward P/E': ('forward_pe', False),
            'P/B Ratio': ('pb_ratio', False),
            'P/S Ratio': ('ps_ratio', False),
            'EV/EBITDA': ('ev_ebitda', False),
            'PEG Ratio': ('peg_ratio', False),
            'Revenue Growth': ('revenue_growth', True),
            'Earnings Growth': ('earnings_growth', True),
            'Gross Margin': ('gross_margin', True),
            'Operating Margin': ('operating_margin', True),
            'Profit Margin': ('profit_margin', True),
            'ROE': ('roe', True),
            'ROA': ('roa', True),
            '1M Return': ('return_1m', True),
            '3M Return': ('return_3m', True),
            '1Y Return': ('return_1y', True),
            'Dividend Yield': ('dividend_yield', True),
            'Beta': ('beta', None),  # Neutral
        }

        stats = {}

        for metric_name, (attr, higher_better) in metrics_config.items():
            values = [getattr(p, attr) for p in peers if getattr(p, attr) != 0]

            if not values:
                continue

            target_val = getattr(target, attr)

            # Calculate percentile
            sorted_vals = sorted(values)
            if target_val in sorted_vals:
                idx = sorted_vals.index(target_val)
            else:
                idx = sum(1 for v in sorted_vals if v < target_val)
            percentile = (idx / len(sorted_vals)) * 100 if sorted_vals else 50

            # For valuation metrics, lower is better, so invert percentile
            if higher_better is False:
                percentile = 100 - percentile

            stats[metric_name] = PeerGroupStats(
                metric_name=metric_name,
                peer_count=len(values),
                mean=np.mean(values),
                median=np.median(values),
                std=np.std(values),
                min_val=min(values),
                max_val=max(values),
                percentile_25=np.percentile(values, 25),
                percentile_75=np.percentile(values, 75),
                target_value=target_val,
                target_percentile=percentile
            )

        return stats

    def _calculate_rankings(self, peers: list[PeerMetrics]):
        """Calculate rankings for each peer across categories."""
        n = len(peers)

        # Valuation rank (lower P/E is better)
        pe_sorted = sorted(range(n), key=lambda i: peers[i].pe_ratio if peers[i].pe_ratio > 0 else float('inf'))
        for rank, idx in enumerate(pe_sorted):
            peers[idx].valuation_rank = rank + 1

        # Growth rank (higher growth is better)
        growth_sorted = sorted(range(n), key=lambda i: peers[i].revenue_growth + peers[i].earnings_growth, reverse=True)
        for rank, idx in enumerate(growth_sorted):
            peers[idx].growth_rank = rank + 1

        # Profitability rank (higher margins is better)
        prof_sorted = sorted(range(n), key=lambda i: peers[i].profit_margin + peers[i].roe, reverse=True)
        for rank, idx in enumerate(prof_sorted):
            peers[idx].profitability_rank = rank + 1

        # Performance rank (higher returns is better)
        perf_sorted = sorted(range(n), key=lambda i: peers[i].return_1y, reverse=True)
        for rank, idx in enumerate(perf_sorted):
            peers[idx].performance_rank = rank + 1

        # Overall rank (average of other ranks)
        for peer in peers:
            peer.overall_rank = round((
                peer.valuation_rank +
                peer.growth_rank +
                peer.profitability_rank +
                peer.performance_rank
            ) / 4)

    def _identify_strengths_weaknesses(
        self,
        target: PeerMetrics,
        stats: dict[str, PeerGroupStats]
    ) -> tuple[list[str], list[str]]:
        """Identify key strengths and weaknesses vs peers."""
        strengths = []
        weaknesses = []

        for metric_name, stat in stats.items():
            percentile = stat.target_percentile

            if percentile >= 75:
                if 'Return' in metric_name or 'Growth' in metric_name or 'Margin' in metric_name or 'RO' in metric_name:
                    strengths.append(f"Strong {metric_name}: Top quartile at {stat.target_value:.1%}"
                                   if stat.target_value < 1 else f"Strong {metric_name}: {stat.target_value:.1f}")
                elif 'P/' in metric_name or 'EV' in metric_name or 'PEG' in metric_name:
                    strengths.append(f"Attractive {metric_name}: {stat.target_value:.1f} vs peer median {stat.median:.1f}")

            elif percentile <= 25:
                if 'Return' in metric_name or 'Growth' in metric_name or 'Margin' in metric_name or 'RO' in metric_name:
                    weaknesses.append(f"Weak {metric_name}: Bottom quartile")
                elif 'P/' in metric_name or 'EV' in metric_name or 'PEG' in metric_name:
                    if stat.target_value > 0:
                        weaknesses.append(f"Expensive {metric_name}: {stat.target_value:.1f} vs peer median {stat.median:.1f}")

        return strengths[:5], weaknesses[:5]

    def _determine_overall_ranking(
        self,
        target: PeerMetrics,
        peers: list[PeerMetrics]
    ) -> str:
        """Determine overall ranking tier."""
        n = len(peers)
        target_rank = target.overall_rank

        percentile = 100 * (1 - (target_rank - 1) / n) if n > 1 else 50

        if percentile >= 80:
            return "Top Tier"
        elif percentile >= 60:
            return "Above Average"
        elif percentile >= 40:
            return "Average"
        elif percentile >= 20:
            return "Below Average"
        else:
            return "Bottom Tier"

    def get_industry_leaders(
        self,
        sector: str,
        metric: str = "market_cap",
        limit: int = 10
    ) -> list[PeerMetrics]:
        """
        Get industry leaders by a specific metric.

        Args:
            sector: Sector name
            metric: Metric to rank by
            limit: Number of leaders to return

        Returns:
            List of PeerMetrics for top companies
        """
        stocks = get_sector_stocks(sector)
        if not stocks:
            return []

        metrics = []
        for symbol in stocks[:30]:
            m = self.analyze_peer(symbol)
            if m:
                metrics.append(m)

        # Sort by metric
        metrics.sort(key=lambda x: getattr(x, metric, 0), reverse=True)

        return metrics[:limit]


# Singleton instance
_analyzer_instance = None

def get_peer_analyzer() -> PeerAnalyzer:
    """Get singleton PeerAnalyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = PeerAnalyzer()
    return _analyzer_instance
