"""
Events Calendar Module for Asset Managers

Tracks important stock events including:
- Earnings announcements and estimates
- Dividend ex-dates and payment dates
- Stock splits
- Analyst upgrades/downgrades
- Economic calendar events
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum
import yfinance as yf

from src.utils.yfinance_cache import get_ticker_info


class EventType(Enum):
    EARNINGS = "Earnings"
    DIVIDEND = "Dividend"
    SPLIT = "Split"
    ANALYST_UPGRADE = "Upgrade"
    ANALYST_DOWNGRADE = "Downgrade"
    EX_DIVIDEND = "Ex-Dividend"
    ECONOMIC = "Economic"


class EarningsSurprise(Enum):
    BEAT = "Beat"
    MISS = "Miss"
    INLINE = "In-Line"
    PENDING = "Pending"


@dataclass
class EarningsEvent:
    """Earnings announcement event."""
    symbol: str
    company_name: str
    earnings_date: str
    is_confirmed: bool = False
    time_of_day: str = ""  # "BMO" (Before Market Open), "AMC" (After Market Close)

    # Estimates
    eps_estimate: float = 0.0
    revenue_estimate: float = 0.0

    # Actuals (if reported)
    eps_actual: float = 0.0
    revenue_actual: float = 0.0

    # Surprise
    eps_surprise: float = 0.0
    eps_surprise_pct: float = 0.0
    surprise_type: EarningsSurprise = EarningsSurprise.PENDING

    # Historical
    beat_rate_4q: float = 0.0  # % of last 4 quarters beat
    avg_surprise_4q: float = 0.0


@dataclass
class DividendEvent:
    """Dividend event."""
    symbol: str
    company_name: str
    ex_dividend_date: str
    payment_date: str = ""
    record_date: str = ""
    dividend_amount: float = 0.0
    dividend_yield: float = 0.0
    frequency: str = ""  # Quarterly, Monthly, Annual
    is_special: bool = False
    year_over_year_change: float = 0.0


@dataclass
class SplitEvent:
    """Stock split event."""
    symbol: str
    company_name: str
    split_date: str
    split_ratio: str = ""  # e.g., "4:1"
    split_factor: float = 1.0


@dataclass
class AnalystEvent:
    """Analyst rating change event."""
    symbol: str
    company_name: str
    date: str
    firm: str
    analyst: str = ""
    old_rating: str = ""
    new_rating: str = ""
    old_price_target: float = 0.0
    new_price_target: float = 0.0
    action: str = ""  # Upgrade, Downgrade, Initiate, Reiterate


@dataclass
class CalendarSummary:
    """Summary of upcoming events for a watchlist."""
    total_events: int = 0
    earnings_this_week: int = 0
    dividends_this_week: int = 0
    events_by_symbol: dict = field(default_factory=dict)
    earnings_events: list[EarningsEvent] = field(default_factory=list)
    dividend_events: list[DividendEvent] = field(default_factory=list)


class EventsCalendar:
    """
    Events calendar for tracking important stock dates.

    Aggregates earnings, dividends, and other events for
    portfolio monitoring.
    """

    def __init__(self):
        self.cache = {}

    def get_earnings_calendar(
        self,
        symbols: list[str],
        days_ahead: int = 30
    ) -> list[EarningsEvent]:
        """
        Get upcoming earnings for a list of symbols.

        Args:
            symbols: List of stock symbols
            days_ahead: Number of days to look ahead

        Returns:
            List of EarningsEvent sorted by date
        """
        events = []
        today = datetime.now()
        end_date = today + timedelta(days=days_ahead)

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                calendar = ticker.calendar

                if calendar is not None and not calendar.empty:
                    # Get earnings date
                    if 'Earnings Date' in calendar.index:
                        earnings_dates = calendar.loc['Earnings Date']
                        if hasattr(earnings_dates, '__iter__') and not isinstance(earnings_dates, str):
                            earnings_date = earnings_dates.iloc[0] if len(earnings_dates) > 0 else None
                        else:
                            earnings_date = earnings_dates

                        if earnings_date:
                            if hasattr(earnings_date, 'strftime'):
                                date_str = earnings_date.strftime('%Y-%m-%d')
                            else:
                                date_str = str(earnings_date)[:10]

                            # Get estimates
                            eps_est = 0
                            rev_est = 0
                            if 'Earnings Average' in calendar.index:
                                eps_est = float(calendar.loc['Earnings Average'].iloc[0]) if hasattr(calendar.loc['Earnings Average'], 'iloc') else float(calendar.loc['Earnings Average'])
                            if 'Revenue Average' in calendar.index:
                                rev_est = float(calendar.loc['Revenue Average'].iloc[0]) if hasattr(calendar.loc['Revenue Average'], 'iloc') else float(calendar.loc['Revenue Average'])

                            info = get_ticker_info(symbol)
                            name = info.get('shortName', symbol) if info else symbol

                            events.append(EarningsEvent(
                                symbol=symbol,
                                company_name=name,
                                earnings_date=date_str,
                                eps_estimate=eps_est,
                                revenue_estimate=rev_est
                            ))
            except Exception:
                continue

        # Sort by date
        events.sort(key=lambda x: x.earnings_date)

        return events

    def get_dividend_calendar(
        self,
        symbols: list[str],
        days_ahead: int = 60
    ) -> list[DividendEvent]:
        """
        Get upcoming dividend events for a list of symbols.

        Args:
            symbols: List of stock symbols
            days_ahead: Number of days to look ahead

        Returns:
            List of DividendEvent sorted by ex-date
        """
        events = []

        for symbol in symbols:
            info = get_ticker_info(symbol)
            if not info:
                continue

            ex_date = info.get('exDividendDate')
            div_rate = info.get('dividendRate', 0) or 0
            div_yield = info.get('dividendYield', 0) or 0

            if ex_date and div_rate > 0:
                # Convert timestamp to date string
                if isinstance(ex_date, (int, float)):
                    ex_date_str = datetime.fromtimestamp(ex_date).strftime('%Y-%m-%d')
                else:
                    ex_date_str = str(ex_date)[:10]

                # Check if within range
                try:
                    ex_dt = datetime.strptime(ex_date_str, '%Y-%m-%d')
                    if ex_dt > datetime.now() + timedelta(days=days_ahead):
                        continue
                except:
                    pass

                # Estimate quarterly dividend
                quarterly_div = div_rate / 4  # Assuming quarterly

                events.append(DividendEvent(
                    symbol=symbol,
                    company_name=info.get('shortName', symbol),
                    ex_dividend_date=ex_date_str,
                    dividend_amount=quarterly_div,
                    dividend_yield=div_yield,
                    frequency="Quarterly"
                ))

        # Sort by ex-dividend date
        events.sort(key=lambda x: x.ex_dividend_date)

        return events

    def get_recent_analyst_actions(
        self,
        symbols: list[str],
        days_back: int = 30
    ) -> list[AnalystEvent]:
        """
        Get recent analyst rating changes.

        Args:
            symbols: List of stock symbols
            days_back: Number of days to look back

        Returns:
            List of AnalystEvent sorted by date (most recent first)
        """
        events = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                recommendations = ticker.recommendations

                if recommendations is not None and not recommendations.empty:
                    # Get recent recommendations
                    cutoff = datetime.now() - timedelta(days=days_back)

                    for idx, row in recommendations.tail(10).iterrows():
                        try:
                            if hasattr(idx, 'strftime'):
                                date_str = idx.strftime('%Y-%m-%d')
                                event_date = idx
                            else:
                                date_str = str(idx)[:10]
                                event_date = datetime.strptime(date_str, '%Y-%m-%d')

                            if event_date < cutoff:
                                continue

                            # Determine action type
                            grade = str(row.get('To Grade', ''))
                            from_grade = str(row.get('From Grade', ''))

                            action = row.get('Action', '')
                            if not action:
                                if grade and from_grade:
                                    # Simple upgrade/downgrade detection
                                    buy_terms = ['buy', 'outperform', 'overweight', 'positive']
                                    sell_terms = ['sell', 'underperform', 'underweight', 'negative']

                                    grade_lower = grade.lower()
                                    from_lower = from_grade.lower()

                                    new_is_buy = any(t in grade_lower for t in buy_terms)
                                    old_is_buy = any(t in from_lower for t in buy_terms)
                                    new_is_sell = any(t in grade_lower for t in sell_terms)
                                    old_is_sell = any(t in from_lower for t in sell_terms)

                                    if new_is_buy and (old_is_sell or not old_is_buy):
                                        action = "Upgrade"
                                    elif new_is_sell and (old_is_buy or not old_is_sell):
                                        action = "Downgrade"
                                    else:
                                        action = "Reiterate"

                            info = get_ticker_info(symbol)
                            name = info.get('shortName', symbol) if info else symbol

                            events.append(AnalystEvent(
                                symbol=symbol,
                                company_name=name,
                                date=date_str,
                                firm=str(row.get('Firm', 'Unknown')),
                                old_rating=from_grade,
                                new_rating=grade,
                                action=action
                            ))
                        except:
                            continue
            except:
                continue

        # Sort by date descending
        events.sort(key=lambda x: x.date, reverse=True)

        return events

    def get_calendar_summary(
        self,
        symbols: list[str],
        days_ahead: int = 14
    ) -> CalendarSummary:
        """
        Get a summary of all upcoming events for a watchlist.

        Args:
            symbols: List of stock symbols
            days_ahead: Number of days to look ahead

        Returns:
            CalendarSummary with all event types
        """
        earnings = self.get_earnings_calendar(symbols, days_ahead)
        dividends = self.get_dividend_calendar(symbols, days_ahead)

        # Count events this week
        today = datetime.now()
        week_end = today + timedelta(days=7)

        def is_this_week(date_str):
            try:
                dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
                return today <= dt <= week_end
            except:
                return False

        earnings_this_week = sum(1 for e in earnings if is_this_week(e.earnings_date))
        dividends_this_week = sum(1 for e in dividends if is_this_week(e.ex_dividend_date))

        # Group events by symbol
        events_by_symbol = {}
        for e in earnings:
            if e.symbol not in events_by_symbol:
                events_by_symbol[e.symbol] = []
            events_by_symbol[e.symbol].append(('earnings', e.earnings_date))

        for e in dividends:
            if e.symbol not in events_by_symbol:
                events_by_symbol[e.symbol] = []
            events_by_symbol[e.symbol].append(('dividend', e.ex_dividend_date))

        return CalendarSummary(
            total_events=len(earnings) + len(dividends),
            earnings_this_week=earnings_this_week,
            dividends_this_week=dividends_this_week,
            events_by_symbol=events_by_symbol,
            earnings_events=earnings,
            dividend_events=dividends
        )

    def get_earnings_history(
        self,
        symbol: str,
        quarters: int = 8
    ) -> list[dict]:
        """
        Get historical earnings data for a symbol.

        Args:
            symbol: Stock symbol
            quarters: Number of past quarters

        Returns:
            List of earnings history dicts
        """
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.earnings_history

            if history is None or history.empty:
                return []

            results = []
            for idx, row in history.tail(quarters).iterrows():
                eps_est = row.get('epsEstimate', 0) or 0
                eps_act = row.get('epsActual', 0) or 0
                surprise = eps_act - eps_est if eps_est else 0
                surprise_pct = (surprise / abs(eps_est) * 100) if eps_est else 0

                results.append({
                    'quarter': str(idx),
                    'eps_estimate': eps_est,
                    'eps_actual': eps_act,
                    'surprise': surprise,
                    'surprise_pct': surprise_pct,
                    'beat': eps_act > eps_est if eps_est else None
                })

            return results
        except:
            return []

    def get_dividend_history(
        self,
        symbol: str,
        years: int = 5
    ) -> list[dict]:
        """
        Get dividend payment history.

        Args:
            symbol: Stock symbol
            years: Number of years of history

        Returns:
            List of dividend history dicts
        """
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends

            if dividends is None or dividends.empty:
                return []

            # Filter to requested period
            cutoff = datetime.now() - timedelta(days=years*365)

            results = []
            prev_div = None
            for date, amount in dividends.items():
                if hasattr(date, 'to_pydatetime'):
                    dt = date.to_pydatetime()
                else:
                    dt = date

                if dt < cutoff:
                    continue

                yoy_change = 0
                if prev_div:
                    yoy_change = (amount / prev_div - 1) * 100 if prev_div else 0

                results.append({
                    'date': dt.strftime('%Y-%m-%d'),
                    'amount': float(amount),
                    'yoy_change': yoy_change
                })

                prev_div = amount

            return results
        except:
            return []


# Singleton instance
_calendar_instance = None

def get_events_calendar() -> EventsCalendar:
    """Get singleton EventsCalendar instance."""
    global _calendar_instance
    if _calendar_instance is None:
        _calendar_instance = EventsCalendar()
    return _calendar_instance
