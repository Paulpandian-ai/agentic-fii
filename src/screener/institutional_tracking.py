"""
Institutional & Insider Activity Tracking Module

Tracks smart money movements including:
- Institutional ownership changes
- Insider buying/selling activity
- Major holder analysis
- Fund ownership trends
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum
import yfinance as yf

from src.utils.yfinance_cache import get_ticker_info


class TransactionType(Enum):
    BUY = "Buy"
    SELL = "Sell"
    EXERCISE = "Exercise"
    GIFT = "Gift"


class InsiderRole(Enum):
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    DIRECTOR = "Director"
    VP = "VP"
    OFFICER = "Officer"
    MAJOR_SHAREHOLDER = "10% Owner"
    OTHER = "Other"


@dataclass
class InstitutionalHolder:
    """Institutional holder information."""
    holder_name: str
    shares: int = 0
    value: float = 0.0
    pct_held: float = 0.0
    date_reported: str = ""
    change_shares: int = 0  # Change from previous filing
    change_pct: float = 0.0


@dataclass
class InsiderTransaction:
    """Insider transaction record."""
    symbol: str
    insider_name: str
    role: str
    transaction_type: TransactionType
    transaction_date: str
    shares: int = 0
    value: float = 0.0
    price_per_share: float = 0.0
    shares_owned_after: int = 0
    filing_date: str = ""


@dataclass
class OwnershipSummary:
    """Summary of institutional and insider ownership."""
    symbol: str
    company_name: str

    # Institutional
    institutional_holders_count: int = 0
    institutional_pct: float = 0.0
    institutional_value: float = 0.0
    top_institutional_holders: list[InstitutionalHolder] = field(default_factory=list)

    # Mutual Funds
    fund_holders_count: int = 0
    fund_pct: float = 0.0
    top_fund_holders: list[InstitutionalHolder] = field(default_factory=list)

    # Insider
    insider_pct: float = 0.0
    insider_transactions_3m: int = 0
    net_insider_shares_3m: int = 0  # Positive = net buying
    insider_buy_ratio: float = 0.0  # Buy transactions / total

    # Trend
    institutional_trend: str = ""  # "Increasing", "Decreasing", "Stable"
    insider_sentiment: str = ""  # "Bullish", "Bearish", "Neutral"


@dataclass
class InsiderSentiment:
    """Aggregated insider sentiment analysis."""
    symbol: str
    period_days: int = 90
    total_transactions: int = 0
    buy_transactions: int = 0
    sell_transactions: int = 0
    total_buy_value: float = 0.0
    total_sell_value: float = 0.0
    net_value: float = 0.0
    unique_insiders_buying: int = 0
    unique_insiders_selling: int = 0
    sentiment_score: float = 0.0  # -100 to +100
    sentiment_label: str = ""  # "Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell"


class InstitutionalTracker:
    """
    Tracks institutional and insider activity for stocks.

    Monitors ownership changes and transactions to identify
    smart money movements.
    """

    def __init__(self):
        self.cache = {}

    def get_institutional_holders(
        self,
        symbol: str,
        top_n: int = 10
    ) -> list[InstitutionalHolder]:
        """
        Get top institutional holders for a symbol.

        Args:
            symbol: Stock symbol
            top_n: Number of top holders to return

        Returns:
            List of InstitutionalHolder objects
        """
        try:
            ticker = yf.Ticker(symbol)
            holders = ticker.institutional_holders

            if holders is None or holders.empty:
                return []

            results = []
            for _, row in holders.head(top_n).iterrows():
                holder = InstitutionalHolder(
                    holder_name=str(row.get('Holder', 'Unknown')),
                    shares=int(row.get('Shares', 0)),
                    value=float(row.get('Value', 0)),
                    pct_held=float(row.get('% Out', 0)) if '% Out' in row else 0,
                    date_reported=str(row.get('Date Reported', ''))[:10] if 'Date Reported' in row else ''
                )
                results.append(holder)

            return results
        except Exception:
            return []

    def get_fund_holders(
        self,
        symbol: str,
        top_n: int = 10
    ) -> list[InstitutionalHolder]:
        """
        Get top mutual fund holders for a symbol.

        Args:
            symbol: Stock symbol
            top_n: Number of top holders to return

        Returns:
            List of InstitutionalHolder objects
        """
        try:
            ticker = yf.Ticker(symbol)
            holders = ticker.mutualfund_holders

            if holders is None or holders.empty:
                return []

            results = []
            for _, row in holders.head(top_n).iterrows():
                holder = InstitutionalHolder(
                    holder_name=str(row.get('Holder', 'Unknown')),
                    shares=int(row.get('Shares', 0)),
                    value=float(row.get('Value', 0)),
                    pct_held=float(row.get('% Out', 0)) if '% Out' in row else 0,
                    date_reported=str(row.get('Date Reported', ''))[:10] if 'Date Reported' in row else ''
                )
                results.append(holder)

            return results
        except Exception:
            return []

    def get_insider_transactions(
        self,
        symbol: str,
        days_back: int = 90
    ) -> list[InsiderTransaction]:
        """
        Get recent insider transactions.

        Args:
            symbol: Stock symbol
            days_back: Number of days to look back

        Returns:
            List of InsiderTransaction objects
        """
        try:
            ticker = yf.Ticker(symbol)
            transactions = ticker.insider_transactions

            if transactions is None or transactions.empty:
                return []

            cutoff = datetime.now() - timedelta(days=days_back)
            results = []

            for _, row in transactions.iterrows():
                try:
                    # Parse date
                    start_date = row.get('Start Date')
                    if start_date is None:
                        continue

                    if hasattr(start_date, 'to_pydatetime'):
                        trans_date = start_date.to_pydatetime()
                    elif hasattr(start_date, 'strftime'):
                        trans_date = start_date
                    else:
                        trans_date = datetime.strptime(str(start_date)[:10], '%Y-%m-%d')

                    if trans_date < cutoff:
                        continue

                    # Determine transaction type
                    text = str(row.get('Text', '')).lower()
                    if 'sale' in text or 'sold' in text:
                        trans_type = TransactionType.SELL
                    elif 'purchase' in text or 'buy' in text or 'acquisition' in text:
                        trans_type = TransactionType.BUY
                    elif 'exercise' in text:
                        trans_type = TransactionType.EXERCISE
                    elif 'gift' in text:
                        trans_type = TransactionType.GIFT
                    else:
                        trans_type = TransactionType.SELL  # Default

                    shares = int(row.get('Shares', 0))
                    value = float(row.get('Value', 0))
                    price = value / shares if shares > 0 else 0

                    transaction = InsiderTransaction(
                        symbol=symbol,
                        insider_name=str(row.get('Insider', 'Unknown')),
                        role=str(row.get('Position', 'Unknown')),
                        transaction_type=trans_type,
                        transaction_date=trans_date.strftime('%Y-%m-%d'),
                        shares=shares,
                        value=value,
                        price_per_share=price
                    )
                    results.append(transaction)
                except Exception:
                    continue

            # Sort by date descending
            results.sort(key=lambda x: x.transaction_date, reverse=True)

            return results
        except Exception:
            return []

    def get_ownership_summary(self, symbol: str) -> OwnershipSummary:
        """
        Get comprehensive ownership summary.

        Args:
            symbol: Stock symbol

        Returns:
            OwnershipSummary object
        """
        info = get_ticker_info(symbol)
        name = info.get('shortName', symbol) if info else symbol

        # Get institutional holders
        inst_holders = self.get_institutional_holders(symbol)
        fund_holders = self.get_fund_holders(symbol)
        insider_trans = self.get_insider_transactions(symbol, days_back=90)

        # Calculate institutional stats
        inst_pct = info.get('heldPercentInstitutions', 0) if info else 0
        inst_value = sum(h.value for h in inst_holders)

        # Calculate fund stats
        fund_pct = sum(h.pct_held for h in fund_holders)

        # Calculate insider stats
        insider_pct = info.get('heldPercentInsiders', 0) if info else 0

        buy_count = sum(1 for t in insider_trans if t.transaction_type == TransactionType.BUY)
        sell_count = sum(1 for t in insider_trans if t.transaction_type == TransactionType.SELL)

        buy_shares = sum(t.shares for t in insider_trans if t.transaction_type == TransactionType.BUY)
        sell_shares = sum(t.shares for t in insider_trans if t.transaction_type == TransactionType.SELL)

        net_shares = buy_shares - sell_shares
        buy_ratio = buy_count / len(insider_trans) if insider_trans else 0.5

        # Determine trends
        if inst_pct > 0.7:
            inst_trend = "High Institutional Interest"
        elif inst_pct > 0.5:
            inst_trend = "Moderate Institutional Interest"
        else:
            inst_trend = "Low Institutional Interest"

        if buy_ratio > 0.7:
            insider_sentiment = "Bullish"
        elif buy_ratio < 0.3:
            insider_sentiment = "Bearish"
        else:
            insider_sentiment = "Neutral"

        return OwnershipSummary(
            symbol=symbol,
            company_name=name,
            institutional_holders_count=len(inst_holders),
            institutional_pct=inst_pct,
            institutional_value=inst_value,
            top_institutional_holders=inst_holders[:5],
            fund_holders_count=len(fund_holders),
            fund_pct=fund_pct,
            top_fund_holders=fund_holders[:5],
            insider_pct=insider_pct,
            insider_transactions_3m=len(insider_trans),
            net_insider_shares_3m=net_shares,
            insider_buy_ratio=buy_ratio,
            institutional_trend=inst_trend,
            insider_sentiment=insider_sentiment
        )

    def analyze_insider_sentiment(
        self,
        symbol: str,
        period_days: int = 90
    ) -> InsiderSentiment:
        """
        Analyze insider trading sentiment.

        Args:
            symbol: Stock symbol
            period_days: Analysis period in days

        Returns:
            InsiderSentiment object
        """
        transactions = self.get_insider_transactions(symbol, days_back=period_days)

        if not transactions:
            return InsiderSentiment(
                symbol=symbol,
                period_days=period_days,
                sentiment_label="No Data"
            )

        buy_trans = [t for t in transactions if t.transaction_type == TransactionType.BUY]
        sell_trans = [t for t in transactions if t.transaction_type == TransactionType.SELL]

        buy_value = sum(t.value for t in buy_trans)
        sell_value = sum(t.value for t in sell_trans)
        net_value = buy_value - sell_value

        unique_buyers = len(set(t.insider_name for t in buy_trans))
        unique_sellers = len(set(t.insider_name for t in sell_trans))

        # Calculate sentiment score (-100 to +100)
        total_value = buy_value + sell_value
        if total_value > 0:
            sentiment_score = ((buy_value - sell_value) / total_value) * 100
        else:
            sentiment_score = 0

        # Adjust for number of unique insiders
        if unique_buyers > unique_sellers:
            sentiment_score = min(100, sentiment_score + 10)
        elif unique_sellers > unique_buyers:
            sentiment_score = max(-100, sentiment_score - 10)

        # Determine label
        if sentiment_score >= 50:
            label = "Strong Buy Signal"
        elif sentiment_score >= 20:
            label = "Buy Signal"
        elif sentiment_score <= -50:
            label = "Strong Sell Signal"
        elif sentiment_score <= -20:
            label = "Sell Signal"
        else:
            label = "Neutral"

        return InsiderSentiment(
            symbol=symbol,
            period_days=period_days,
            total_transactions=len(transactions),
            buy_transactions=len(buy_trans),
            sell_transactions=len(sell_trans),
            total_buy_value=buy_value,
            total_sell_value=sell_value,
            net_value=net_value,
            unique_insiders_buying=unique_buyers,
            unique_insiders_selling=unique_sellers,
            sentiment_score=sentiment_score,
            sentiment_label=label
        )

    def get_smart_money_signals(
        self,
        symbols: list[str]
    ) -> list[dict]:
        """
        Get smart money signals for multiple symbols.

        Identifies stocks with notable institutional or insider activity.

        Args:
            symbols: List of stock symbols

        Returns:
            List of signal dicts sorted by strength
        """
        signals = []

        for symbol in symbols:
            try:
                sentiment = self.analyze_insider_sentiment(symbol, period_days=60)
                summary = self.get_ownership_summary(symbol)

                signal_strength = 0
                signal_reasons = []

                # Strong insider buying
                if sentiment.sentiment_score > 30:
                    signal_strength += 30
                    signal_reasons.append(f"Insider buying: ${sentiment.net_value:,.0f} net")

                # Multiple insiders buying
                if sentiment.unique_insiders_buying >= 3:
                    signal_strength += 20
                    signal_reasons.append(f"{sentiment.unique_insiders_buying} insiders buying")

                # High institutional ownership
                if summary.institutional_pct > 0.8:
                    signal_strength += 15
                    signal_reasons.append(f"{summary.institutional_pct:.0%} institutional ownership")

                # Low insider selling
                if sentiment.sell_transactions == 0 and sentiment.buy_transactions > 0:
                    signal_strength += 15
                    signal_reasons.append("No insider selling")

                if signal_strength > 0:
                    signals.append({
                        'symbol': symbol,
                        'company': summary.company_name,
                        'signal_strength': signal_strength,
                        'insider_sentiment': sentiment.sentiment_label,
                        'insider_score': sentiment.sentiment_score,
                        'institutional_pct': summary.institutional_pct,
                        'reasons': signal_reasons
                    })
            except Exception:
                continue

        # Sort by signal strength
        signals.sort(key=lambda x: x['signal_strength'], reverse=True)

        return signals

    def screen_insider_buying(
        self,
        symbols: list[str],
        min_buy_value: float = 100000,
        min_buyers: int = 2
    ) -> list[dict]:
        """
        Screen for stocks with significant insider buying.

        Args:
            symbols: List of symbols to screen
            min_buy_value: Minimum total buy value
            min_buyers: Minimum number of unique buyers

        Returns:
            List of stocks meeting criteria
        """
        results = []

        for symbol in symbols:
            sentiment = self.analyze_insider_sentiment(symbol, period_days=90)

            if (sentiment.total_buy_value >= min_buy_value and
                sentiment.unique_insiders_buying >= min_buyers and
                sentiment.sentiment_score > 0):

                info = get_ticker_info(symbol)

                results.append({
                    'symbol': symbol,
                    'company': info.get('shortName', symbol) if info else symbol,
                    'sector': info.get('sector', 'Unknown') if info else 'Unknown',
                    'buy_value': sentiment.total_buy_value,
                    'unique_buyers': sentiment.unique_insiders_buying,
                    'sentiment_score': sentiment.sentiment_score,
                    'sentiment': sentiment.sentiment_label
                })

        # Sort by buy value
        results.sort(key=lambda x: x['buy_value'], reverse=True)

        return results


# Singleton instance
_tracker_instance = None

def get_institutional_tracker() -> InstitutionalTracker:
    """Get singleton InstitutionalTracker instance."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = InstitutionalTracker()
    return _tracker_instance
