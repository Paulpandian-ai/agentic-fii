"""
YFinance caching and rate limiting utility.

Implements best practices to avoid rate limiting:
- Caching: Store fetched data to avoid repeated requests
- Rate limiting: Add delays between requests (300ms)
- Batch requests: Fetch multiple tickers in a single call
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock
from typing import Any, Optional

import pandas as pd
import yfinance as yf
from loguru import logger


@dataclass
class CacheEntry:
    """Represents a cached data entry."""
    data: Any
    timestamp: datetime
    ttl_seconds: int = 300  # Default 5 minutes

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)


class YFinanceCache:
    """
    Thread-safe caching and rate-limiting wrapper for yfinance.

    Features:
    - In-memory caching with configurable TTL
    - Rate limiting with delays between requests (300ms default)
    - Batch requests for fetching multiple tickers
    - Thread-safe operations
    """

    _instance: Optional['YFinanceCache'] = None
    _lock = Lock()

    def __new__(cls):
        """Singleton pattern to ensure single cache instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the cache."""
        if self._initialized:
            return

        self._cache: dict[str, CacheEntry] = {}
        self._last_request_time: float = 0
        self._request_delay: float = 0.3  # 300ms delay between requests
        self._cache_ttl: int = 300  # 5 minutes default TTL
        self._history_cache_ttl: int = 60  # 1 minute for historical data
        self._request_lock = Lock()
        self._initialized = True

        logger.info("YFinanceCache initialized with rate limiting and caching")

    def _get_cache_key(self, symbol: str, data_type: str, **kwargs) -> str:
        """Generate a unique cache key."""
        key_parts = [symbol.upper(), data_type]
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")
        return ":".join(key_parts)

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        with self._request_lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._request_delay:
                sleep_time = self._request_delay - elapsed
                time.sleep(sleep_time)
            self._last_request_time = time.time()

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if available and not expired."""
        if key in self._cache:
            entry = self._cache[key]
            if not entry.is_expired():
                logger.debug(f"Cache hit for {key}")
                return entry.data
            else:
                # Remove expired entry
                del self._cache[key]
                logger.debug(f"Cache expired for {key}")
        return None

    def _set_cache(self, key: str, data: Any, ttl_seconds: Optional[int] = None) -> None:
        """Store data in cache."""
        ttl = ttl_seconds or self._cache_ttl
        self._cache[key] = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl_seconds=ttl
        )
        logger.debug(f"Cached {key} with TTL {ttl}s")

    def get_ticker_info(self, symbol: str) -> dict:
        """
        Get ticker info with caching.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing ticker info
        """
        cache_key = self._get_cache_key(symbol, "info")

        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # Wait for rate limit
        self._wait_for_rate_limit()

        # Fetch from yfinance
        logger.debug(f"Fetching ticker info for {symbol}")
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            self._set_cache(cache_key, info)
            return info
        except Exception as e:
            logger.warning(f"Error fetching ticker info for {symbol}: {e}")
            raise

    def get_ticker_history(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get ticker history with caching.

        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

        Returns:
            DataFrame with historical data
        """
        cache_key = self._get_cache_key(symbol, "history", period=period, interval=interval)

        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # Wait for rate limit
        self._wait_for_rate_limit()

        # Fetch from yfinance
        logger.debug(f"Fetching history for {symbol} (period={period}, interval={interval})")
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval=interval)
            self._set_cache(cache_key, history, ttl_seconds=self._history_cache_ttl)
            return history
        except Exception as e:
            logger.warning(f"Error fetching history for {symbol}: {e}")
            raise

    def get_multiple_tickers_info(self, symbols: list[str]) -> dict[str, dict]:
        """
        Get info for multiple tickers using batch requests.

        Args:
            symbols: List of stock ticker symbols

        Returns:
            Dictionary mapping symbols to their info
        """
        results = {}
        uncached_symbols = []

        # Check cache for each symbol
        for symbol in symbols:
            cache_key = self._get_cache_key(symbol, "info")
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                results[symbol] = cached
            else:
                uncached_symbols.append(symbol)

        # Fetch uncached symbols with rate limiting
        for symbol in uncached_symbols:
            self._wait_for_rate_limit()
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                cache_key = self._get_cache_key(symbol, "info")
                self._set_cache(cache_key, info)
                results[symbol] = info
            except Exception as e:
                logger.warning(f"Error fetching info for {symbol}: {e}")
                results[symbol] = {}

        return results

    def get_multiple_tickers_history(
        self,
        symbols: list[str],
        period: str = "1y",
        interval: str = "1d"
    ) -> dict[str, pd.DataFrame]:
        """
        Get history for multiple tickers using batch download.

        This is more efficient than fetching individually as yfinance
        supports batch downloads.

        Args:
            symbols: List of stock ticker symbols
            period: Data period
            interval: Data interval

        Returns:
            Dictionary mapping symbols to their historical DataFrames
        """
        results = {}
        uncached_symbols = []

        # Check cache for each symbol
        for symbol in symbols:
            cache_key = self._get_cache_key(symbol, "history", period=period, interval=interval)
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                results[symbol] = cached
            else:
                uncached_symbols.append(symbol)

        if uncached_symbols:
            # Wait for rate limit
            self._wait_for_rate_limit()

            # Batch download uncached symbols
            logger.debug(f"Batch downloading history for {len(uncached_symbols)} symbols")
            try:
                # Use yf.download for batch requests
                data = yf.download(
                    uncached_symbols,
                    period=period,
                    interval=interval,
                    group_by='ticker',
                    progress=False,
                    threads=False  # Disable threading to avoid rate limits
                )

                # Process results for each symbol
                for symbol in uncached_symbols:
                    try:
                        if len(uncached_symbols) == 1:
                            # Single ticker - data is not grouped
                            symbol_data = data
                        else:
                            # Multiple tickers - data is grouped by ticker
                            symbol_data = data[symbol] if symbol in data.columns.get_level_values(0) else pd.DataFrame()

                        if not symbol_data.empty:
                            # Reset index to match individual ticker.history() format
                            if isinstance(symbol_data.columns, pd.MultiIndex):
                                symbol_data = symbol_data.droplevel(0, axis=1)

                            cache_key = self._get_cache_key(symbol, "history", period=period, interval=interval)
                            self._set_cache(cache_key, symbol_data, ttl_seconds=self._history_cache_ttl)
                            results[symbol] = symbol_data
                        else:
                            results[symbol] = pd.DataFrame()
                    except Exception as e:
                        logger.warning(f"Error processing history for {symbol}: {e}")
                        results[symbol] = pd.DataFrame()

            except Exception as e:
                logger.warning(f"Error in batch download: {e}")
                # Fallback to individual requests
                for symbol in uncached_symbols:
                    try:
                        results[symbol] = self.get_ticker_history(symbol, period, interval)
                    except Exception:
                        results[symbol] = pd.DataFrame()

        return results

    def get_ticker_news(self, symbol: str) -> list[dict]:
        """
        Get ticker news with caching.

        Args:
            symbol: Stock ticker symbol

        Returns:
            List of news articles
        """
        cache_key = self._get_cache_key(symbol, "news")

        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # Wait for rate limit
        self._wait_for_rate_limit()

        # Fetch from yfinance
        logger.debug(f"Fetching news for {symbol}")
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []
            self._set_cache(cache_key, news, ttl_seconds=180)  # 3 minute TTL for news
            return news
        except Exception as e:
            logger.warning(f"Error fetching news for {symbol}: {e}")
            return []

    def get_ticker_recommendations(self, symbol: str) -> pd.DataFrame:
        """
        Get ticker analyst recommendations with caching.

        Args:
            symbol: Stock ticker symbol

        Returns:
            DataFrame with analyst recommendations
        """
        cache_key = self._get_cache_key(symbol, "recommendations")

        # Check cache first
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached

        # Wait for rate limit
        self._wait_for_rate_limit()

        # Fetch from yfinance
        logger.debug(f"Fetching recommendations for {symbol}")
        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations
            if recommendations is None:
                recommendations = pd.DataFrame()
            self._set_cache(cache_key, recommendations, ttl_seconds=300)  # 5 minute TTL
            return recommendations
        except Exception as e:
            logger.warning(f"Error fetching recommendations for {symbol}: {e}")
            return pd.DataFrame()

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cache entries.

        Args:
            symbol: If provided, clear only entries for this symbol.
                   If None, clear all entries.
        """
        if symbol is None:
            self._cache.clear()
            logger.info("Cleared entire cache")
        else:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(symbol.upper())]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared cache for {symbol} ({len(keys_to_remove)} entries)")

    def get_cache_stats(self) -> dict:
        """Get statistics about the cache."""
        total_entries = len(self._cache)
        expired_entries = sum(1 for entry in self._cache.values() if entry.is_expired())
        valid_entries = total_entries - expired_entries

        return {
            "total_entries": total_entries,
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
        }

    def set_rate_limit_delay(self, delay_seconds: float) -> None:
        """
        Set the delay between API requests.

        Args:
            delay_seconds: Delay in seconds (recommended: 0.2 to 0.5)
        """
        self._request_delay = max(0.1, delay_seconds)  # Minimum 100ms
        logger.info(f"Rate limit delay set to {self._request_delay}s")

    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """
        Set the default cache TTL.

        Args:
            ttl_seconds: Time-to-live in seconds
        """
        self._cache_ttl = max(60, ttl_seconds)  # Minimum 60 seconds
        logger.info(f"Cache TTL set to {self._cache_ttl}s")


# Global instance
yf_cache = YFinanceCache()


# Convenience functions for direct use
def get_ticker_info(symbol: str) -> dict:
    """Get ticker info with caching and rate limiting."""
    return yf_cache.get_ticker_info(symbol)


def get_ticker_history(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Get ticker history with caching and rate limiting."""
    return yf_cache.get_ticker_history(symbol, period, interval)


def get_multiple_tickers_info(symbols: list[str]) -> dict[str, dict]:
    """Get info for multiple tickers with batch optimization."""
    return yf_cache.get_multiple_tickers_info(symbols)


def get_multiple_tickers_history(
    symbols: list[str],
    period: str = "1y",
    interval: str = "1d"
) -> dict[str, pd.DataFrame]:
    """Get history for multiple tickers with batch optimization."""
    return yf_cache.get_multiple_tickers_history(symbols, period, interval)


def get_ticker_news(symbol: str) -> list[dict]:
    """Get ticker news with caching and rate limiting."""
    return yf_cache.get_ticker_news(symbol)


def get_ticker_recommendations(symbol: str) -> pd.DataFrame:
    """Get ticker analyst recommendations with caching and rate limiting."""
    return yf_cache.get_ticker_recommendations(symbol)
