"""Stock Screener Module - Comprehensive tools for asset managers."""

from .stock_screener import (
    StockScreener,
    ScreenedStock,
    ScreenerCriteria,
    StockCategory,
    Recommendation,
)
from .stock_universe import (
    get_sp500_symbols,
    get_nasdaq100_symbols,
    get_dow30_symbols,
    get_sector_stocks,
    get_all_major_stocks,
    get_dividend_aristocrats,
    get_all_sectors,
    SECTOR_ETFS,
)
from .watchlist_manager import (
    WatchlistManager,
    Watchlist,
    WatchlistStock,
    get_watchlist_manager,
)
from .portfolio_analytics import (
    PortfolioAnalytics,
    RiskMetrics,
    PerformanceMetrics,
    SectorAllocation,
    CorrelationData,
    BenchmarkComparison,
    get_portfolio_analytics,
)
from .technical_analysis import (
    TechnicalAnalyzer,
    TechnicalSummary,
    MovingAverages,
    RSIIndicator,
    MACDIndicator,
    BollingerBands,
    SupportResistance,
    TrendDirection,
    SignalType,
    get_technical_analyzer,
)
from .peer_comparison import (
    PeerAnalyzer,
    PeerMetrics,
    PeerComparison,
    PeerGroupStats,
    get_peer_analyzer,
)
from .events_calendar import (
    EventsCalendar,
    EarningsEvent,
    DividendEvent,
    AnalystEvent,
    CalendarSummary,
    get_events_calendar,
)
from .institutional_tracking import (
    InstitutionalTracker,
    InstitutionalHolder,
    InsiderTransaction,
    InsiderSentiment,
    OwnershipSummary,
    get_institutional_tracker,
)
from .investment_manager import (
    InvestmentManager,
    StockPick,
    PortfolioRecommendation,
    MonteCarloResult,
    RiskProfile,
    InvestmentStrategy,
    get_investment_manager,
)
from .paper_portfolio import (
    PaperPortfolioManager,
    PaperPortfolio,
    PaperHolding,
    PortfolioSnapshot,
    get_paper_portfolio_manager,
)

__all__ = [
    # Stock Screener
    "StockScreener",
    "ScreenedStock",
    "ScreenerCriteria",
    "StockCategory",
    "Recommendation",
    # Stock Universe
    "get_sp500_symbols",
    "get_nasdaq100_symbols",
    "get_dow30_symbols",
    "get_sector_stocks",
    "get_all_major_stocks",
    "get_dividend_aristocrats",
    "get_all_sectors",
    "SECTOR_ETFS",
    # Watchlist
    "WatchlistManager",
    "Watchlist",
    "WatchlistStock",
    "get_watchlist_manager",
    # Portfolio Analytics
    "PortfolioAnalytics",
    "RiskMetrics",
    "PerformanceMetrics",
    "SectorAllocation",
    "CorrelationData",
    "BenchmarkComparison",
    "get_portfolio_analytics",
    # Technical Analysis
    "TechnicalAnalyzer",
    "TechnicalSummary",
    "MovingAverages",
    "RSIIndicator",
    "MACDIndicator",
    "BollingerBands",
    "SupportResistance",
    "TrendDirection",
    "SignalType",
    "get_technical_analyzer",
    # Peer Comparison
    "PeerAnalyzer",
    "PeerMetrics",
    "PeerComparison",
    "PeerGroupStats",
    "get_peer_analyzer",
    # Events Calendar
    "EventsCalendar",
    "EarningsEvent",
    "DividendEvent",
    "AnalystEvent",
    "CalendarSummary",
    "get_events_calendar",
    # Institutional Tracking
    "InstitutionalTracker",
    "InstitutionalHolder",
    "InsiderTransaction",
    "InsiderSentiment",
    "OwnershipSummary",
    "get_institutional_tracker",
    # Investment Manager
    "InvestmentManager",
    "StockPick",
    "PortfolioRecommendation",
    "MonteCarloResult",
    "RiskProfile",
    "InvestmentStrategy",
    "get_investment_manager",
    # Paper Portfolio
    "PaperPortfolioManager",
    "PaperPortfolio",
    "PaperHolding",
    "PortfolioSnapshot",
    "get_paper_portfolio_manager",
]
