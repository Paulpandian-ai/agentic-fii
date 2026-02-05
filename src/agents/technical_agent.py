"""Technical Analysis Agent - Analyzes price patterns and technical indicators."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.schemas import TechnicalIndicators
from src.utils.yfinance_cache import get_ticker_history


class TechnicalAnalysisAgent(BaseAgent):
    """
    Agent responsible for technical analysis of stocks.

    Analyzes technical indicators including:
    - Moving Averages (SMA, EMA)
    - Momentum Indicators (RSI, MACD, Stochastic)
    - Volatility Indicators (Bollinger Bands, ATR)
    - Volume Indicators (OBV, Volume SMA)
    - Trend Indicators (ADX)
    """

    def __init__(self, name: str = "TechnicalAgent", lookback_period: int = 200):
        """
        Initialize the Technical Analysis Agent.

        Args:
            name: Agent name
            lookback_period: Number of days of historical data to fetch
        """
        super().__init__(name=name, agent_type="technical")
        self.lookback_period = lookback_period

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """
        Perform technical analysis on the given stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing technical indicators and analysis
        """
        self.log_info(f"Fetching price data for {symbol}")

        try:
            # Fetch historical data with caching
            df = get_ticker_history(symbol, period=f"{self.lookback_period}d")

            if df.empty:
                raise ValueError(f"No price data available for {symbol}")

            # Calculate technical indicators
            indicators = self._calculate_indicators(symbol, df)

            # Generate signals
            buy_signals, sell_signals = self._generate_signals(indicators, df)
            indicators.buy_signals = buy_signals
            indicators.sell_signals = sell_signals

            # Determine trend
            indicators.trend_direction = self._determine_trend(indicators, df)

            # Calculate technical score
            score = self._calculate_score(indicators)

            # Generate analysis summary
            summary = self._generate_summary(indicators, score)

            return {
                "indicators": indicators.model_dump(),
                "score": score,
                "summary": summary,
            }

        except Exception as e:
            self.log_warning(f"Error in technical analysis: {str(e)}")
            return {
                "indicators": TechnicalIndicators(symbol=symbol).model_dump(),
                "score": None,
                "summary": f"Unable to perform technical analysis: {str(e)}",
            }

    def _calculate_indicators(self, symbol: str, df: pd.DataFrame) -> TechnicalIndicators:
        """Calculate all technical indicators from price data."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        indicators = TechnicalIndicators(symbol=symbol)

        # Moving Averages
        if len(close) >= 20:
            indicators.sma_20 = close.rolling(window=20).mean().iloc[-1]
        if len(close) >= 50:
            indicators.sma_50 = close.rolling(window=50).mean().iloc[-1]
        if len(close) >= 200:
            indicators.sma_200 = close.rolling(window=200).mean().iloc[-1]

        # EMA
        if len(close) >= 12:
            indicators.ema_12 = close.ewm(span=12, adjust=False).mean().iloc[-1]
        if len(close) >= 26:
            indicators.ema_26 = close.ewm(span=26, adjust=False).mean().iloc[-1]

        # RSI (14-day)
        if len(close) >= 15:
            indicators.rsi = self._calculate_rsi(close, 14)

        # MACD
        if len(close) >= 26:
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            indicators.macd = macd_line.iloc[-1]
            indicators.macd_signal = signal_line.iloc[-1]
            indicators.macd_histogram = indicators.macd - indicators.macd_signal

        # Stochastic Oscillator
        if len(close) >= 14:
            low_14 = low.rolling(window=14).min()
            high_14 = high.rolling(window=14).max()
            stoch_k = 100 * (close - low_14) / (high_14 - low_14)
            indicators.stochastic_k = stoch_k.iloc[-1]
            indicators.stochastic_d = stoch_k.rolling(window=3).mean().iloc[-1]

        # Bollinger Bands
        if len(close) >= 20:
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            indicators.bollinger_middle = sma_20.iloc[-1]
            indicators.bollinger_upper = (sma_20 + 2 * std_20).iloc[-1]
            indicators.bollinger_lower = (sma_20 - 2 * std_20).iloc[-1]

        # ATR (Average True Range)
        if len(close) >= 15:
            indicators.atr = self._calculate_atr(high, low, close, 14)

        # Volume SMA
        if len(volume) >= 20:
            indicators.volume_sma = volume.rolling(window=20).mean().iloc[-1]

        # OBV (On-Balance Volume)
        if len(close) >= 2:
            indicators.obv = self._calculate_obv(close, volume)

        # ADX (Average Directional Index)
        if len(close) >= 28:
            indicators.adx = self._calculate_adx(high, low, close, 14)

        return indicators

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> float:
        """Calculate Average True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.iloc[-1]

    def _calculate_obv(self, close: pd.Series, volume: pd.Series) -> float:
        """Calculate On-Balance Volume."""
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        return obv[-1]

    def _calculate_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> float:
        """Calculate Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr = self._calculate_atr(high, low, close, period)

        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else None

    def _generate_signals(
        self, indicators: TechnicalIndicators, df: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        """Generate buy and sell signals based on indicators."""
        buy_signals = []
        sell_signals = []
        current_price = df["Close"].iloc[-1]

        # RSI signals
        if indicators.rsi is not None:
            if indicators.rsi < 30:
                buy_signals.append("RSI oversold (<30)")
            elif indicators.rsi > 70:
                sell_signals.append("RSI overbought (>70)")

        # MACD signals
        if indicators.macd is not None and indicators.macd_signal is not None:
            if indicators.macd > indicators.macd_signal:
                buy_signals.append("MACD bullish crossover")
            else:
                sell_signals.append("MACD bearish crossover")

        # Moving Average signals
        if indicators.sma_50 is not None and indicators.sma_200 is not None:
            if indicators.sma_50 > indicators.sma_200:
                buy_signals.append("Golden cross (SMA50 > SMA200)")
            else:
                sell_signals.append("Death cross (SMA50 < SMA200)")

        # Price vs SMA
        if indicators.sma_20 is not None:
            if current_price > indicators.sma_20:
                buy_signals.append("Price above SMA20")
            else:
                sell_signals.append("Price below SMA20")

        # Bollinger Band signals
        if indicators.bollinger_lower is not None and indicators.bollinger_upper is not None:
            if current_price < indicators.bollinger_lower:
                buy_signals.append("Price at lower Bollinger Band")
            elif current_price > indicators.bollinger_upper:
                sell_signals.append("Price at upper Bollinger Band")

        # Stochastic signals
        if indicators.stochastic_k is not None:
            if indicators.stochastic_k < 20:
                buy_signals.append("Stochastic oversold")
            elif indicators.stochastic_k > 80:
                sell_signals.append("Stochastic overbought")

        return buy_signals, sell_signals

    def _determine_trend(
        self, indicators: TechnicalIndicators, df: pd.DataFrame
    ) -> str:
        """Determine the overall trend direction."""
        current_price = df["Close"].iloc[-1]
        bullish_points = 0
        bearish_points = 0

        # Check moving averages
        if indicators.sma_20 is not None:
            if current_price > indicators.sma_20:
                bullish_points += 1
            else:
                bearish_points += 1

        if indicators.sma_50 is not None:
            if current_price > indicators.sma_50:
                bullish_points += 1
            else:
                bearish_points += 1

        if indicators.sma_200 is not None:
            if current_price > indicators.sma_200:
                bullish_points += 2  # More weight for long-term
            else:
                bearish_points += 2

        # Check MACD
        if indicators.macd_histogram is not None:
            if indicators.macd_histogram > 0:
                bullish_points += 1
            else:
                bearish_points += 1

        # ADX for trend strength
        if indicators.adx is not None and indicators.adx > 25:
            # Strong trend - amplify the direction
            if bullish_points > bearish_points:
                bullish_points += 1
            else:
                bearish_points += 1

        if bullish_points > bearish_points + 1:
            return "bullish"
        elif bearish_points > bullish_points + 1:
            return "bearish"
        else:
            return "neutral"

    def _calculate_score(self, indicators: TechnicalIndicators) -> float:
        """Calculate a technical score from 0-100."""
        score = 50.0  # Start neutral

        # RSI contribution (0-20 points)
        if indicators.rsi is not None:
            if 40 <= indicators.rsi <= 60:
                score += 10  # Neutral RSI
            elif 30 <= indicators.rsi < 40 or 60 < indicators.rsi <= 70:
                score += 5
            elif indicators.rsi < 30:
                score += 15  # Oversold = potential opportunity
            else:  # > 70
                score -= 5  # Overbought risk

        # Trend alignment (0-20 points)
        if indicators.trend_direction == "bullish":
            score += 15
        elif indicators.trend_direction == "bearish":
            score -= 10

        # MACD (0-15 points)
        if indicators.macd_histogram is not None:
            if indicators.macd_histogram > 0:
                score += 10
            else:
                score -= 5

        # Signal balance
        buy_count = len(indicators.buy_signals)
        sell_count = len(indicators.sell_signals)
        signal_diff = buy_count - sell_count
        score += signal_diff * 3  # Each net buy signal adds 3 points

        return max(0, min(100, score))

    def _generate_summary(self, indicators: TechnicalIndicators, score: float) -> str:
        """Generate a summary of the technical analysis."""
        parts = []

        # Trend summary
        if indicators.trend_direction:
            parts.append(f"The stock is in a {indicators.trend_direction} trend.")

        # RSI summary
        if indicators.rsi is not None:
            if indicators.rsi < 30:
                parts.append("RSI indicates oversold conditions.")
            elif indicators.rsi > 70:
                parts.append("RSI indicates overbought conditions.")
            else:
                parts.append(f"RSI at {indicators.rsi:.1f} shows neutral momentum.")

        # Signals summary
        buy_count = len(indicators.buy_signals)
        sell_count = len(indicators.sell_signals)
        if buy_count > sell_count:
            parts.append(f"Technical signals lean bullish ({buy_count} buy vs {sell_count} sell).")
        elif sell_count > buy_count:
            parts.append(f"Technical signals lean bearish ({sell_count} sell vs {buy_count} buy).")
        else:
            parts.append("Technical signals are mixed.")

        # Score interpretation
        if score >= 65:
            parts.append("Overall technical outlook is positive.")
        elif score <= 35:
            parts.append("Overall technical outlook is negative.")
        else:
            parts.append("Overall technical outlook is neutral.")

        return " ".join(parts)
