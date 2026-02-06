"""Stock Screener Module - Scan markets for value and growth opportunities."""

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

__all__ = [
    "StockScreener",
    "ScreenedStock",
    "ScreenerCriteria",
    "StockCategory",
    "Recommendation",
    "get_sp500_symbols",
    "get_nasdaq100_symbols",
    "get_dow30_symbols",
    "get_sector_stocks",
    "get_all_major_stocks",
    "get_dividend_aristocrats",
    "get_all_sectors",
    "SECTOR_ETFS",
    "WatchlistManager",
    "Watchlist",
    "WatchlistStock",
    "get_watchlist_manager",
]
