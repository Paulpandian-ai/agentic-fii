"""
Technical Analysis Module for Stock Screener

Provides comprehensive technical indicators including:
- Moving Averages (SMA, EMA, WMA)
- Momentum Indicators (RSI, MACD, Stochastic)
- Volatility Indicators (Bollinger Bands, ATR)
- Volume Indicators (OBV, VWAP)
- Trend Indicators (ADX, Parabolic SAR)
- Support/Resistance Levels
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum
import numpy as np

from src.utils.yfinance_cache import get_ticker_history


class TrendDirection(Enum):
    STRONG_BULLISH = "Strong Bullish"
    BULLISH = "Bullish"
    NEUTRAL = "Neutral"
    BEARISH = "Bearish"
    STRONG_BEARISH = "Strong Bearish"


class SignalType(Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


@dataclass
class MovingAverages:
    """Moving average values."""
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_100: float = 0.0
    sma_200: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    ema_50: float = 0.0
    price_vs_sma_20: float = 0.0  # % above/below
    price_vs_sma_50: float = 0.0
    price_vs_sma_200: float = 0.0
    golden_cross: bool = False  # SMA50 > SMA200
    death_cross: bool = False   # SMA50 < SMA200


@dataclass
class RSIIndicator:
    """RSI indicator data."""
    rsi_14: float = 50.0
    rsi_7: float = 50.0
    rsi_21: float = 50.0
    is_overbought: bool = False  # RSI > 70
    is_oversold: bool = False    # RSI < 30
    rsi_divergence: str = ""     # Bullish/Bearish divergence


@dataclass
class MACDIndicator:
    """MACD indicator data."""
    macd_line: float = 0.0
    signal_line: float = 0.0
    histogram: float = 0.0
    is_bullish: bool = False  # MACD > Signal
    crossover: str = ""       # "bullish", "bearish", or ""
    histogram_trend: str = "" # "increasing", "decreasing"


@dataclass
class StochasticIndicator:
    """Stochastic oscillator data."""
    k_line: float = 50.0
    d_line: float = 50.0
    is_overbought: bool = False  # K > 80
    is_oversold: bool = False    # K < 20


@dataclass
class BollingerBands:
    """Bollinger Bands data."""
    upper_band: float = 0.0
    middle_band: float = 0.0
    lower_band: float = 0.0
    bandwidth: float = 0.0
    percent_b: float = 0.5  # Position within bands (0-1)
    squeeze: bool = False   # Low volatility squeeze


@dataclass
class VolumeIndicators:
    """Volume-based indicators."""
    obv: float = 0.0
    obv_trend: str = ""      # "rising", "falling"
    volume_sma_20: float = 0.0
    relative_volume: float = 1.0  # Current vs average
    accumulation_distribution: float = 0.0


@dataclass
class SupportResistance:
    """Support and resistance levels."""
    resistance_1: float = 0.0
    resistance_2: float = 0.0
    support_1: float = 0.0
    support_2: float = 0.0
    pivot_point: float = 0.0
    fibonacci_levels: dict = field(default_factory=dict)


@dataclass
class TechnicalSignal:
    """Individual technical signal."""
    indicator: str
    signal: SignalType
    value: float
    description: str


@dataclass
class TechnicalSummary:
    """Complete technical analysis summary."""
    symbol: str
    current_price: float
    moving_averages: MovingAverages
    rsi: RSIIndicator
    macd: MACDIndicator
    stochastic: StochasticIndicator
    bollinger: BollingerBands
    volume: VolumeIndicators
    support_resistance: SupportResistance
    trend: TrendDirection
    overall_signal: SignalType
    signal_strength: float  # 0-100
    signals: list[TechnicalSignal] = field(default_factory=list)
    analysis_date: str = ""


class TechnicalAnalyzer:
    """
    Technical analysis engine for stocks.

    Calculates comprehensive technical indicators and generates
    trading signals based on multiple factors.
    """

    def __init__(self):
        self.cache = {}

    def analyze(self, symbol: str, period: str = "1y") -> Optional[TechnicalSummary]:
        """
        Perform complete technical analysis on a symbol.

        Args:
            symbol: Stock ticker symbol
            period: Historical data period

        Returns:
            TechnicalSummary with all indicators and signals
        """
        hist = get_ticker_history(symbol, period=period)
        if hist is None or hist.empty or len(hist) < 50:
            return None

        close = hist['Close'].values
        high = hist['High'].values
        low = hist['Low'].values
        volume = hist['Volume'].values if 'Volume' in hist.columns else np.zeros(len(close))

        current_price = close[-1]

        # Calculate all indicators
        ma = self._calculate_moving_averages(close, current_price)
        rsi = self._calculate_rsi(close)
        macd = self._calculate_macd(close)
        stoch = self._calculate_stochastic(high, low, close)
        bb = self._calculate_bollinger_bands(close, current_price)
        vol = self._calculate_volume_indicators(close, volume)
        sr = self._calculate_support_resistance(high, low, close)

        # Generate signals
        signals = self._generate_signals(current_price, ma, rsi, macd, stoch, bb, vol)

        # Determine overall trend and signal
        trend = self._determine_trend(ma, rsi, macd)
        overall_signal, strength = self._calculate_overall_signal(signals)

        return TechnicalSummary(
            symbol=symbol,
            current_price=current_price,
            moving_averages=ma,
            rsi=rsi,
            macd=macd,
            stochastic=stoch,
            bollinger=bb,
            volume=vol,
            support_resistance=sr,
            trend=trend,
            overall_signal=overall_signal,
            signal_strength=strength,
            signals=signals,
            analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M")
        )

    def _calculate_moving_averages(self, close: np.ndarray, current_price: float) -> MovingAverages:
        """Calculate various moving averages."""
        def sma(data, period):
            if len(data) < period:
                return 0
            return np.mean(data[-period:])

        def ema(data, period):
            if len(data) < period:
                return 0
            multiplier = 2 / (period + 1)
            ema_val = data[-period]
            for price in data[-period+1:]:
                ema_val = (price * multiplier) + (ema_val * (1 - multiplier))
            return ema_val

        sma_20 = sma(close, 20)
        sma_50 = sma(close, 50)
        sma_100 = sma(close, 100)
        sma_200 = sma(close, 200) if len(close) >= 200 else 0

        ema_12 = ema(close, 12)
        ema_26 = ema(close, 26)
        ema_50 = ema(close, 50)

        return MovingAverages(
            sma_20=sma_20,
            sma_50=sma_50,
            sma_100=sma_100,
            sma_200=sma_200,
            ema_12=ema_12,
            ema_26=ema_26,
            ema_50=ema_50,
            price_vs_sma_20=(current_price / sma_20 - 1) * 100 if sma_20 > 0 else 0,
            price_vs_sma_50=(current_price / sma_50 - 1) * 100 if sma_50 > 0 else 0,
            price_vs_sma_200=(current_price / sma_200 - 1) * 100 if sma_200 > 0 else 0,
            golden_cross=sma_50 > sma_200 if sma_200 > 0 else False,
            death_cross=sma_50 < sma_200 if sma_200 > 0 else False
        )

    def _calculate_rsi(self, close: np.ndarray, periods: list[int] = [7, 14, 21]) -> RSIIndicator:
        """Calculate RSI for multiple periods."""
        def rsi(data, period):
            if len(data) < period + 1:
                return 50

            deltas = np.diff(data[-period-50:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                return 100
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        rsi_7 = rsi(close, 7)
        rsi_14 = rsi(close, 14)
        rsi_21 = rsi(close, 21)

        return RSIIndicator(
            rsi_14=rsi_14,
            rsi_7=rsi_7,
            rsi_21=rsi_21,
            is_overbought=rsi_14 > 70,
            is_oversold=rsi_14 < 30,
            rsi_divergence=""
        )

    def _calculate_macd(self, close: np.ndarray) -> MACDIndicator:
        """Calculate MACD indicator."""
        def ema(data, period):
            if len(data) < period:
                return np.zeros(len(data))
            result = np.zeros(len(data))
            multiplier = 2 / (period + 1)
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (data[i] * multiplier) + (result[i-1] * (1 - multiplier))
            return result

        ema_12 = ema(close, 12)
        ema_26 = ema(close, 26)
        macd_line = ema_12 - ema_26
        signal_line = ema(macd_line, 9)
        histogram = macd_line - signal_line

        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_hist = histogram[-1]
        prev_hist = histogram[-2] if len(histogram) > 1 else 0

        # Detect crossover
        crossover = ""
        if len(macd_line) > 1:
            if macd_line[-2] < signal_line[-2] and macd_line[-1] > signal_line[-1]:
                crossover = "bullish"
            elif macd_line[-2] > signal_line[-2] and macd_line[-1] < signal_line[-1]:
                crossover = "bearish"

        return MACDIndicator(
            macd_line=current_macd,
            signal_line=current_signal,
            histogram=current_hist,
            is_bullish=current_macd > current_signal,
            crossover=crossover,
            histogram_trend="increasing" if current_hist > prev_hist else "decreasing"
        )

    def _calculate_stochastic(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        k_period: int = 14,
        d_period: int = 3
    ) -> StochasticIndicator:
        """Calculate Stochastic oscillator."""
        if len(close) < k_period:
            return StochasticIndicator()

        # Calculate %K
        lowest_low = np.min(low[-k_period:])
        highest_high = np.max(high[-k_period:])

        if highest_high == lowest_low:
            k = 50
        else:
            k = ((close[-1] - lowest_low) / (highest_high - lowest_low)) * 100

        # Calculate %D (SMA of %K)
        k_values = []
        for i in range(d_period):
            idx = -1 - i
            if abs(idx) <= len(close) - k_period:
                ll = np.min(low[idx-k_period+1:idx+1] if idx != -1 else low[-k_period:])
                hh = np.max(high[idx-k_period+1:idx+1] if idx != -1 else high[-k_period:])
                if hh != ll:
                    k_values.append(((close[idx] - ll) / (hh - ll)) * 100)

        d = np.mean(k_values) if k_values else k

        return StochasticIndicator(
            k_line=k,
            d_line=d,
            is_overbought=k > 80,
            is_oversold=k < 20
        )

    def _calculate_bollinger_bands(
        self,
        close: np.ndarray,
        current_price: float,
        period: int = 20,
        std_dev: float = 2.0
    ) -> BollingerBands:
        """Calculate Bollinger Bands."""
        if len(close) < period:
            return BollingerBands()

        middle = np.mean(close[-period:])
        std = np.std(close[-period:])

        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)

        bandwidth = ((upper - lower) / middle) * 100 if middle > 0 else 0

        # Percent B: position within bands
        if upper != lower:
            percent_b = (current_price - lower) / (upper - lower)
        else:
            percent_b = 0.5

        # Detect squeeze (low volatility)
        avg_bandwidth = np.mean([
            (np.mean(close[i:i+period]) + std_dev * np.std(close[i:i+period]) -
             (np.mean(close[i:i+period]) - std_dev * np.std(close[i:i+period]))) /
            np.mean(close[i:i+period]) * 100
            for i in range(max(0, len(close)-100), len(close)-period, 10)
        ]) if len(close) > period + 10 else bandwidth

        squeeze = bandwidth < avg_bandwidth * 0.7  # 30% below average

        return BollingerBands(
            upper_band=upper,
            middle_band=middle,
            lower_band=lower,
            bandwidth=bandwidth,
            percent_b=percent_b,
            squeeze=squeeze
        )

    def _calculate_volume_indicators(
        self,
        close: np.ndarray,
        volume: np.ndarray
    ) -> VolumeIndicators:
        """Calculate volume-based indicators."""
        if len(close) < 2 or len(volume) < 2:
            return VolumeIndicators()

        # On-Balance Volume (OBV)
        obv = 0
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv += volume[i]
            elif close[i] < close[i-1]:
                obv -= volume[i]

        # OBV trend (compare to 10-day ago)
        obv_10_ago = 0
        if len(close) > 10:
            for i in range(1, len(close)-10):
                if close[i] > close[i-1]:
                    obv_10_ago += volume[i]
                elif close[i] < close[i-1]:
                    obv_10_ago -= volume[i]
        obv_trend = "rising" if obv > obv_10_ago else "falling"

        # Volume SMA
        vol_sma_20 = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)

        # Relative volume
        rel_vol = volume[-1] / vol_sma_20 if vol_sma_20 > 0 else 1

        # Accumulation/Distribution (simplified using price change * volume)
        ad = 0
        for i in range(1, len(close)):
            price_change = close[i] - close[i-1]
            if close[i-1] != 0:
                ad += (price_change / close[i-1]) * volume[i]

        return VolumeIndicators(
            obv=obv,
            obv_trend=obv_trend,
            volume_sma_20=vol_sma_20,
            relative_volume=rel_vol,
            accumulation_distribution=ad
        )

    def _calculate_support_resistance(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> SupportResistance:
        """Calculate support and resistance levels using pivot points."""
        if len(close) < 1:
            return SupportResistance()

        # Use recent high/low/close for pivot calculation
        h = np.max(high[-20:]) if len(high) >= 20 else high[-1]
        l = np.min(low[-20:]) if len(low) >= 20 else low[-1]
        c = close[-1]

        # Classic pivot points
        pivot = (h + l + c) / 3
        r1 = (2 * pivot) - l
        r2 = pivot + (h - l)
        s1 = (2 * pivot) - h
        s2 = pivot - (h - l)

        # Fibonacci levels
        diff = h - l
        fib_levels = {
            '0.0': l,
            '0.236': l + (diff * 0.236),
            '0.382': l + (diff * 0.382),
            '0.5': l + (diff * 0.5),
            '0.618': l + (diff * 0.618),
            '0.786': l + (diff * 0.786),
            '1.0': h
        }

        return SupportResistance(
            resistance_1=r1,
            resistance_2=r2,
            support_1=s1,
            support_2=s2,
            pivot_point=pivot,
            fibonacci_levels=fib_levels
        )

    def _generate_signals(
        self,
        price: float,
        ma: MovingAverages,
        rsi: RSIIndicator,
        macd: MACDIndicator,
        stoch: StochasticIndicator,
        bb: BollingerBands,
        vol: VolumeIndicators
    ) -> list[TechnicalSignal]:
        """Generate trading signals from indicators."""
        signals = []

        # Moving Average signals
        if ma.golden_cross:
            signals.append(TechnicalSignal(
                indicator="Moving Averages",
                signal=SignalType.BUY,
                value=ma.sma_50,
                description="Golden Cross: SMA50 above SMA200"
            ))
        elif ma.death_cross:
            signals.append(TechnicalSignal(
                indicator="Moving Averages",
                signal=SignalType.SELL,
                value=ma.sma_50,
                description="Death Cross: SMA50 below SMA200"
            ))

        if ma.price_vs_sma_200 > 10:
            signals.append(TechnicalSignal(
                indicator="Price vs SMA200",
                signal=SignalType.BUY,
                value=ma.price_vs_sma_200,
                description=f"Price {ma.price_vs_sma_200:.1f}% above SMA200"
            ))
        elif ma.price_vs_sma_200 < -10:
            signals.append(TechnicalSignal(
                indicator="Price vs SMA200",
                signal=SignalType.SELL,
                value=ma.price_vs_sma_200,
                description=f"Price {abs(ma.price_vs_sma_200):.1f}% below SMA200"
            ))

        # RSI signals
        if rsi.is_oversold:
            signals.append(TechnicalSignal(
                indicator="RSI",
                signal=SignalType.STRONG_BUY,
                value=rsi.rsi_14,
                description=f"RSI oversold at {rsi.rsi_14:.1f}"
            ))
        elif rsi.is_overbought:
            signals.append(TechnicalSignal(
                indicator="RSI",
                signal=SignalType.STRONG_SELL,
                value=rsi.rsi_14,
                description=f"RSI overbought at {rsi.rsi_14:.1f}"
            ))
        elif rsi.rsi_14 < 40:
            signals.append(TechnicalSignal(
                indicator="RSI",
                signal=SignalType.BUY,
                value=rsi.rsi_14,
                description=f"RSI showing weakness at {rsi.rsi_14:.1f}"
            ))
        elif rsi.rsi_14 > 60:
            signals.append(TechnicalSignal(
                indicator="RSI",
                signal=SignalType.SELL,
                value=rsi.rsi_14,
                description=f"RSI showing strength at {rsi.rsi_14:.1f}"
            ))

        # MACD signals
        if macd.crossover == "bullish":
            signals.append(TechnicalSignal(
                indicator="MACD",
                signal=SignalType.STRONG_BUY,
                value=macd.macd_line,
                description="Bullish MACD crossover"
            ))
        elif macd.crossover == "bearish":
            signals.append(TechnicalSignal(
                indicator="MACD",
                signal=SignalType.STRONG_SELL,
                value=macd.macd_line,
                description="Bearish MACD crossover"
            ))
        elif macd.is_bullish and macd.histogram_trend == "increasing":
            signals.append(TechnicalSignal(
                indicator="MACD",
                signal=SignalType.BUY,
                value=macd.histogram,
                description="MACD bullish with increasing momentum"
            ))
        elif not macd.is_bullish and macd.histogram_trend == "decreasing":
            signals.append(TechnicalSignal(
                indicator="MACD",
                signal=SignalType.SELL,
                value=macd.histogram,
                description="MACD bearish with decreasing momentum"
            ))

        # Stochastic signals
        if stoch.is_oversold and stoch.k_line > stoch.d_line:
            signals.append(TechnicalSignal(
                indicator="Stochastic",
                signal=SignalType.STRONG_BUY,
                value=stoch.k_line,
                description="Stochastic oversold with bullish crossover"
            ))
        elif stoch.is_overbought and stoch.k_line < stoch.d_line:
            signals.append(TechnicalSignal(
                indicator="Stochastic",
                signal=SignalType.STRONG_SELL,
                value=stoch.k_line,
                description="Stochastic overbought with bearish crossover"
            ))

        # Bollinger Band signals
        if bb.percent_b < 0:
            signals.append(TechnicalSignal(
                indicator="Bollinger Bands",
                signal=SignalType.STRONG_BUY,
                value=bb.percent_b,
                description="Price below lower Bollinger Band"
            ))
        elif bb.percent_b > 1:
            signals.append(TechnicalSignal(
                indicator="Bollinger Bands",
                signal=SignalType.STRONG_SELL,
                value=bb.percent_b,
                description="Price above upper Bollinger Band"
            ))
        elif bb.squeeze:
            signals.append(TechnicalSignal(
                indicator="Bollinger Bands",
                signal=SignalType.HOLD,
                value=bb.bandwidth,
                description="Bollinger Band squeeze - volatility contraction"
            ))

        # Volume signals
        if vol.relative_volume > 2 and vol.obv_trend == "rising":
            signals.append(TechnicalSignal(
                indicator="Volume",
                signal=SignalType.BUY,
                value=vol.relative_volume,
                description="High volume with rising OBV"
            ))
        elif vol.relative_volume > 2 and vol.obv_trend == "falling":
            signals.append(TechnicalSignal(
                indicator="Volume",
                signal=SignalType.SELL,
                value=vol.relative_volume,
                description="High volume with falling OBV"
            ))

        return signals

    def _determine_trend(
        self,
        ma: MovingAverages,
        rsi: RSIIndicator,
        macd: MACDIndicator
    ) -> TrendDirection:
        """Determine overall trend direction."""
        bullish_points = 0
        bearish_points = 0

        # Moving averages
        if ma.golden_cross:
            bullish_points += 2
        elif ma.death_cross:
            bearish_points += 2

        if ma.price_vs_sma_50 > 0:
            bullish_points += 1
        else:
            bearish_points += 1

        if ma.price_vs_sma_200 > 0:
            bullish_points += 1
        else:
            bearish_points += 1

        # RSI
        if rsi.rsi_14 > 50:
            bullish_points += 1
        else:
            bearish_points += 1

        # MACD
        if macd.is_bullish:
            bullish_points += 1
        else:
            bearish_points += 1

        if macd.histogram_trend == "increasing":
            bullish_points += 1
        else:
            bearish_points += 1

        # Determine trend
        net_score = bullish_points - bearish_points

        if net_score >= 5:
            return TrendDirection.STRONG_BULLISH
        elif net_score >= 2:
            return TrendDirection.BULLISH
        elif net_score <= -5:
            return TrendDirection.STRONG_BEARISH
        elif net_score <= -2:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.NEUTRAL

    def _calculate_overall_signal(
        self,
        signals: list[TechnicalSignal]
    ) -> tuple[SignalType, float]:
        """Calculate overall signal from individual signals."""
        if not signals:
            return SignalType.HOLD, 50.0

        # Score each signal
        signal_scores = {
            SignalType.STRONG_BUY: 2,
            SignalType.BUY: 1,
            SignalType.HOLD: 0,
            SignalType.SELL: -1,
            SignalType.STRONG_SELL: -2
        }

        total_score = sum(signal_scores[s.signal] for s in signals)
        max_possible = len(signals) * 2
        min_possible = len(signals) * -2

        # Normalize to 0-100
        if max_possible != min_possible:
            strength = ((total_score - min_possible) / (max_possible - min_possible)) * 100
        else:
            strength = 50

        # Determine signal
        avg_score = total_score / len(signals) if signals else 0

        if avg_score >= 1.5:
            signal = SignalType.STRONG_BUY
        elif avg_score >= 0.5:
            signal = SignalType.BUY
        elif avg_score <= -1.5:
            signal = SignalType.STRONG_SELL
        elif avg_score <= -0.5:
            signal = SignalType.SELL
        else:
            signal = SignalType.HOLD

        return signal, strength


# Singleton instance
_analyzer_instance = None

def get_technical_analyzer() -> TechnicalAnalyzer:
    """Get singleton TechnicalAnalyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = TechnicalAnalyzer()
    return _analyzer_instance
