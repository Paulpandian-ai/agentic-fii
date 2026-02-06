"""
Watchlist Manager - Manage multiple stock watchlists.

Provides functionality to:
- Create and delete watchlists
- Add and remove stocks from watchlists
- Persist watchlists to file
- Load watchlists on startup
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from pathlib import Path

from loguru import logger


@dataclass
class WatchlistStock:
    """Represents a stock in a watchlist."""
    symbol: str
    name: str = ""
    added_date: str = ""
    added_price: Optional[float] = None
    notes: str = ""
    category: str = ""  # value, growth, dividend, etc.

    def __post_init__(self):
        if not self.added_date:
            self.added_date = datetime.now().strftime("%Y-%m-%d %H:%M")


@dataclass
class Watchlist:
    """Represents a watchlist."""
    name: str
    description: str = ""
    stocks: list[WatchlistStock] = field(default_factory=list)
    created_date: str = ""

    def __post_init__(self):
        if not self.created_date:
            self.created_date = datetime.now().strftime("%Y-%m-%d %H:%M")

    def add_stock(self, stock: WatchlistStock) -> bool:
        """Add a stock to the watchlist if not already present."""
        if not any(s.symbol == stock.symbol for s in self.stocks):
            self.stocks.append(stock)
            return True
        return False

    def remove_stock(self, symbol: str) -> bool:
        """Remove a stock from the watchlist."""
        original_count = len(self.stocks)
        self.stocks = [s for s in self.stocks if s.symbol != symbol]
        return len(self.stocks) < original_count

    def has_stock(self, symbol: str) -> bool:
        """Check if a stock is in the watchlist."""
        return any(s.symbol == symbol for s in self.stocks)

    def get_symbols(self) -> list[str]:
        """Get list of symbols in the watchlist."""
        return [s.symbol for s in self.stocks]


class WatchlistManager:
    """
    Manages multiple stock watchlists with persistence.

    Features:
    - Create, rename, and delete watchlists
    - Add and remove stocks from watchlists
    - Save and load watchlists from JSON file
    - Track when stocks were added and at what price
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the watchlist manager.

        Args:
            storage_path: Path to store watchlist data. If None, uses default location.
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            # Default to user's home directory
            self.storage_path = Path.home() / ".stock_screener" / "watchlists.json"

        self.watchlists: dict[str, Watchlist] = {}
        self._load_watchlists()

    def _ensure_directory(self) -> None:
        """Ensure the storage directory exists."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_watchlists(self) -> None:
        """Load watchlists from storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for name, wl_data in data.items():
                        stocks = [WatchlistStock(**s) for s in wl_data.get('stocks', [])]
                        self.watchlists[name] = Watchlist(
                            name=wl_data['name'],
                            description=wl_data.get('description', ''),
                            stocks=stocks,
                            created_date=wl_data.get('created_date', '')
                        )
                logger.info(f"Loaded {len(self.watchlists)} watchlists")
        except Exception as e:
            logger.warning(f"Error loading watchlists: {e}")
            self.watchlists = {}

    def _save_watchlists(self) -> None:
        """Save watchlists to storage."""
        try:
            self._ensure_directory()
            data = {}
            for name, wl in self.watchlists.items():
                data[name] = {
                    'name': wl.name,
                    'description': wl.description,
                    'created_date': wl.created_date,
                    'stocks': [asdict(s) for s in wl.stocks]
                }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.watchlists)} watchlists")
        except Exception as e:
            logger.error(f"Error saving watchlists: {e}")

    def create_watchlist(self, name: str, description: str = "") -> bool:
        """
        Create a new watchlist.

        Args:
            name: Watchlist name (must be unique)
            description: Optional description

        Returns:
            True if created, False if name already exists
        """
        if name in self.watchlists:
            return False

        self.watchlists[name] = Watchlist(name=name, description=description)
        self._save_watchlists()
        return True

    def delete_watchlist(self, name: str) -> bool:
        """
        Delete a watchlist.

        Args:
            name: Watchlist name

        Returns:
            True if deleted, False if not found
        """
        if name in self.watchlists:
            del self.watchlists[name]
            self._save_watchlists()
            return True
        return False

    def rename_watchlist(self, old_name: str, new_name: str) -> bool:
        """
        Rename a watchlist.

        Args:
            old_name: Current watchlist name
            new_name: New watchlist name

        Returns:
            True if renamed, False if not found or new name exists
        """
        if old_name not in self.watchlists or new_name in self.watchlists:
            return False

        wl = self.watchlists[old_name]
        wl.name = new_name
        self.watchlists[new_name] = wl
        del self.watchlists[old_name]
        self._save_watchlists()
        return True

    def get_watchlist(self, name: str) -> Optional[Watchlist]:
        """Get a watchlist by name."""
        return self.watchlists.get(name)

    def get_all_watchlists(self) -> list[Watchlist]:
        """Get all watchlists."""
        return list(self.watchlists.values())

    def get_watchlist_names(self) -> list[str]:
        """Get names of all watchlists."""
        return list(self.watchlists.keys())

    def add_stock_to_watchlist(
        self,
        watchlist_name: str,
        symbol: str,
        name: str = "",
        price: Optional[float] = None,
        notes: str = "",
        category: str = ""
    ) -> bool:
        """
        Add a stock to a watchlist.

        Args:
            watchlist_name: Name of the watchlist
            symbol: Stock symbol
            name: Company name
            price: Current price when added
            notes: Optional notes
            category: Stock category (value, growth, etc.)

        Returns:
            True if added, False if watchlist not found or stock already exists
        """
        wl = self.watchlists.get(watchlist_name)
        if not wl:
            return False

        stock = WatchlistStock(
            symbol=symbol.upper(),
            name=name,
            added_price=price,
            notes=notes,
            category=category
        )

        if wl.add_stock(stock):
            self._save_watchlists()
            return True
        return False

    def remove_stock_from_watchlist(self, watchlist_name: str, symbol: str) -> bool:
        """
        Remove a stock from a watchlist.

        Args:
            watchlist_name: Name of the watchlist
            symbol: Stock symbol to remove

        Returns:
            True if removed, False if not found
        """
        wl = self.watchlists.get(watchlist_name)
        if not wl:
            return False

        if wl.remove_stock(symbol.upper()):
            self._save_watchlists()
            return True
        return False

    def is_stock_in_watchlist(self, watchlist_name: str, symbol: str) -> bool:
        """Check if a stock is in a specific watchlist."""
        wl = self.watchlists.get(watchlist_name)
        if not wl:
            return False
        return wl.has_stock(symbol.upper())

    def get_watchlists_containing_stock(self, symbol: str) -> list[str]:
        """Get names of all watchlists containing a specific stock."""
        symbol = symbol.upper()
        return [name for name, wl in self.watchlists.items() if wl.has_stock(symbol)]

    def update_stock_notes(self, watchlist_name: str, symbol: str, notes: str) -> bool:
        """Update notes for a stock in a watchlist."""
        wl = self.watchlists.get(watchlist_name)
        if not wl:
            return False

        for stock in wl.stocks:
            if stock.symbol == symbol.upper():
                stock.notes = notes
                self._save_watchlists()
                return True
        return False

    def get_all_watched_symbols(self) -> list[str]:
        """Get a deduplicated list of all symbols across all watchlists."""
        symbols = set()
        for wl in self.watchlists.values():
            symbols.update(wl.get_symbols())
        return sorted(list(symbols))

    def export_watchlist(self, watchlist_name: str) -> Optional[dict]:
        """Export a watchlist as a dictionary."""
        wl = self.watchlists.get(watchlist_name)
        if not wl:
            return None

        return {
            'name': wl.name,
            'description': wl.description,
            'created_date': wl.created_date,
            'stocks': [asdict(s) for s in wl.stocks]
        }

    def import_watchlist(self, data: dict) -> bool:
        """
        Import a watchlist from a dictionary.

        Args:
            data: Watchlist data dictionary

        Returns:
            True if imported successfully
        """
        try:
            name = data['name']
            if name in self.watchlists:
                # Generate unique name
                i = 1
                while f"{name}_{i}" in self.watchlists:
                    i += 1
                name = f"{name}_{i}"

            stocks = [WatchlistStock(**s) for s in data.get('stocks', [])]
            self.watchlists[name] = Watchlist(
                name=name,
                description=data.get('description', ''),
                stocks=stocks,
                created_date=data.get('created_date', '')
            )
            self._save_watchlists()
            return True
        except Exception as e:
            logger.error(f"Error importing watchlist: {e}")
            return False


# Global instance for easy access
_watchlist_manager: Optional[WatchlistManager] = None


def get_watchlist_manager() -> WatchlistManager:
    """Get the global watchlist manager instance."""
    global _watchlist_manager
    if _watchlist_manager is None:
        _watchlist_manager = WatchlistManager()
    return _watchlist_manager
