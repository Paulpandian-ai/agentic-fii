"""Data schemas and models for the stock analysis platform."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Status of an agent's execution."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StockData(BaseModel):
    """Basic stock data model."""

    symbol: str = Field(..., description="Stock ticker symbol")
    name: Optional[str] = Field(None, description="Company name")
    current_price: Optional[float] = Field(None, description="Current stock price")
    previous_close: Optional[float] = Field(None, description="Previous closing price")
    open_price: Optional[float] = Field(None, description="Opening price")
    high: Optional[float] = Field(None, description="Day's high")
    low: Optional[float] = Field(None, description="Day's low")
    volume: Optional[int] = Field(None, description="Trading volume")
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    timestamp: datetime = Field(default_factory=datetime.now)


class FundamentalMetrics(BaseModel):
    """Fundamental analysis metrics."""

    symbol: str
    pe_ratio: Optional[float] = Field(None, description="Price to Earnings ratio")
    forward_pe: Optional[float] = Field(None, description="Forward P/E ratio")
    peg_ratio: Optional[float] = Field(None, description="Price/Earnings to Growth ratio")
    price_to_book: Optional[float] = Field(None, description="Price to Book ratio")
    price_to_sales: Optional[float] = Field(None, description="Price to Sales ratio")

    # Profitability
    profit_margin: Optional[float] = Field(None, description="Profit margin percentage")
    operating_margin: Optional[float] = Field(None, description="Operating margin percentage")
    return_on_equity: Optional[float] = Field(None, description="Return on Equity (ROE)")
    return_on_assets: Optional[float] = Field(None, description="Return on Assets (ROA)")

    # Growth
    revenue_growth: Optional[float] = Field(None, description="Revenue growth rate")
    earnings_growth: Optional[float] = Field(None, description="Earnings growth rate")

    # Financial Health
    debt_to_equity: Optional[float] = Field(None, description="Debt to Equity ratio")
    current_ratio: Optional[float] = Field(None, description="Current ratio")
    quick_ratio: Optional[float] = Field(None, description="Quick ratio")

    # Dividends
    dividend_yield: Optional[float] = Field(None, description="Dividend yield percentage")
    payout_ratio: Optional[float] = Field(None, description="Dividend payout ratio")

    # Earnings
    eps: Optional[float] = Field(None, description="Earnings per share")
    eps_growth: Optional[float] = Field(None, description="EPS growth rate")

    analysis_summary: Optional[str] = Field(None, description="Summary of fundamental analysis")
    score: Optional[float] = Field(None, ge=0, le=100, description="Fundamental score 0-100")


class TechnicalIndicators(BaseModel):
    """Technical analysis indicators."""

    symbol: str

    # Moving Averages
    sma_20: Optional[float] = Field(None, description="20-day Simple Moving Average")
    sma_50: Optional[float] = Field(None, description="50-day Simple Moving Average")
    sma_200: Optional[float] = Field(None, description="200-day Simple Moving Average")
    ema_12: Optional[float] = Field(None, description="12-day Exponential Moving Average")
    ema_26: Optional[float] = Field(None, description="26-day Exponential Moving Average")

    # Momentum Indicators
    rsi: Optional[float] = Field(None, description="Relative Strength Index")
    macd: Optional[float] = Field(None, description="MACD value")
    macd_signal: Optional[float] = Field(None, description="MACD signal line")
    macd_histogram: Optional[float] = Field(None, description="MACD histogram")
    stochastic_k: Optional[float] = Field(None, description="Stochastic %K")
    stochastic_d: Optional[float] = Field(None, description="Stochastic %D")

    # Volatility
    bollinger_upper: Optional[float] = Field(None, description="Bollinger Band upper")
    bollinger_middle: Optional[float] = Field(None, description="Bollinger Band middle")
    bollinger_lower: Optional[float] = Field(None, description="Bollinger Band lower")
    atr: Optional[float] = Field(None, description="Average True Range")

    # Volume
    volume_sma: Optional[float] = Field(None, description="Volume SMA")
    obv: Optional[float] = Field(None, description="On-Balance Volume")

    # Trend
    adx: Optional[float] = Field(None, description="Average Directional Index")
    trend_direction: Optional[str] = Field(None, description="Current trend direction")

    # Signals
    buy_signals: list[str] = Field(default_factory=list, description="Active buy signals")
    sell_signals: list[str] = Field(default_factory=list, description="Active sell signals")

    analysis_summary: Optional[str] = Field(None, description="Summary of technical analysis")
    score: Optional[float] = Field(None, ge=0, le=100, description="Technical score 0-100")


class SentimentData(BaseModel):
    """Sentiment analysis data."""

    symbol: str

    # Overall Sentiment
    overall_sentiment: Optional[str] = Field(None, description="Overall sentiment: bullish/bearish/neutral")
    sentiment_score: Optional[float] = Field(None, ge=-1, le=1, description="Sentiment score -1 to 1")

    # News Sentiment
    news_sentiment: Optional[float] = Field(None, description="News sentiment score")
    news_count: int = Field(0, description="Number of news articles analyzed")
    positive_news_count: int = Field(0, description="Count of positive news")
    negative_news_count: int = Field(0, description="Count of negative news")
    neutral_news_count: int = Field(0, description="Count of neutral news")

    # Key Headlines
    key_headlines: list[str] = Field(default_factory=list, description="Key news headlines")

    # Social Media (if available)
    social_sentiment: Optional[float] = Field(None, description="Social media sentiment")
    social_volume: Optional[int] = Field(None, description="Social media mention volume")

    # Analyst Ratings
    analyst_rating: Optional[str] = Field(None, description="Consensus analyst rating")
    buy_ratings: int = Field(0, description="Number of buy ratings")
    hold_ratings: int = Field(0, description="Number of hold ratings")
    sell_ratings: int = Field(0, description="Number of sell ratings")
    target_price: Optional[float] = Field(None, description="Average analyst target price")

    analysis_summary: Optional[str] = Field(None, description="Summary of sentiment analysis")
    score: Optional[float] = Field(None, ge=0, le=100, description="Sentiment score 0-100")


class RiskMetrics(BaseModel):
    """Risk assessment metrics."""

    symbol: str

    # Volatility Metrics
    volatility_daily: Optional[float] = Field(None, description="Daily volatility")
    volatility_annual: Optional[float] = Field(None, description="Annualized volatility")
    beta: Optional[float] = Field(None, description="Beta coefficient")

    # Risk Ratios
    sharpe_ratio: Optional[float] = Field(None, description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")

    # Drawdown
    max_drawdown: Optional[float] = Field(None, description="Maximum drawdown percentage")
    current_drawdown: Optional[float] = Field(None, description="Current drawdown from peak")

    # Value at Risk
    var_95: Optional[float] = Field(None, description="95% Value at Risk")
    var_99: Optional[float] = Field(None, description="99% Value at Risk")

    # Risk Level
    risk_level: Optional[str] = Field(None, description="Risk level: low/medium/high/very_high")
    risk_factors: list[str] = Field(default_factory=list, description="Identified risk factors")

    analysis_summary: Optional[str] = Field(None, description="Summary of risk assessment")
    score: Optional[float] = Field(None, ge=0, le=100, description="Risk score 0-100 (higher = safer)")


class AgentResult(BaseModel):
    """Result from an individual agent."""

    agent_name: str = Field(..., description="Name of the agent")
    agent_type: str = Field(..., description="Type of analysis performed")
    status: AgentStatus = Field(default=AgentStatus.COMPLETED)
    execution_time: float = Field(0.0, description="Execution time in seconds")
    data: dict[str, Any] = Field(default_factory=dict, description="Agent-specific result data")
    score: Optional[float] = Field(None, ge=0, le=100, description="Agent's analysis score")
    summary: Optional[str] = Field(None, description="Brief summary of findings")
    errors: list[str] = Field(default_factory=list, description="Any errors encountered")
    timestamp: datetime = Field(default_factory=datetime.now)


class AnalysisReport(BaseModel):
    """Complete analysis report from the Master Agent."""

    symbol: str = Field(..., description="Stock ticker symbol")
    company_name: Optional[str] = Field(None, description="Company name")

    # Individual Agent Results
    stock_data: Optional[StockData] = None
    fundamental_analysis: Optional[FundamentalMetrics] = None
    technical_analysis: Optional[TechnicalIndicators] = None
    sentiment_analysis: Optional[SentimentData] = None
    risk_assessment: Optional[RiskMetrics] = None

    # Aggregated Results
    agent_results: list[AgentResult] = Field(default_factory=list)

    # Overall Analysis
    overall_score: Optional[float] = Field(None, ge=0, le=100, description="Weighted overall score")
    recommendation: Optional[str] = Field(None, description="Buy/Hold/Sell recommendation")
    confidence: Optional[float] = Field(None, ge=0, le=100, description="Confidence in recommendation")

    # Summary
    executive_summary: Optional[str] = Field(None, description="Executive summary of analysis")
    key_strengths: list[str] = Field(default_factory=list, description="Key strengths identified")
    key_risks: list[str] = Field(default_factory=list, description="Key risks identified")
    key_catalysts: list[str] = Field(default_factory=list, description="Potential catalysts")

    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    total_execution_time: float = Field(0.0, description="Total analysis time in seconds")
    agents_executed: int = Field(0, description="Number of agents executed")
    successful_agents: int = Field(0, description="Number of successful agent executions")
