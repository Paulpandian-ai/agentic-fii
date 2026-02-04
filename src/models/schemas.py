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
    sector: Optional[str] = Field(None, description="Company sector")
    industry: Optional[str] = Field(None, description="Company industry")
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


# ============== NEW SCHEMAS FOR BRIDGEWATER-STYLE ANALYSIS ==============


class SupplierInfo(BaseModel):
    """Information about a supplier company."""

    symbol: Optional[str] = Field(None, description="Supplier stock symbol if public")
    name: str = Field(..., description="Supplier company name")
    relationship: Optional[str] = Field(None, description="Type of supply relationship")
    revenue_dependency: Optional[float] = Field(None, description="Estimated % of supplier revenue from this company")

    # Financial Health
    market_cap: Optional[float] = Field(None, description="Supplier market cap")
    profit_margin: Optional[float] = Field(None, description="Supplier profit margin")
    debt_to_equity: Optional[float] = Field(None, description="Supplier debt/equity")
    current_ratio: Optional[float] = Field(None, description="Supplier current ratio")

    # Performance
    stock_performance_ytd: Optional[float] = Field(None, description="YTD stock performance")
    revenue_growth: Optional[float] = Field(None, description="Revenue growth rate")

    # Risk Assessment
    financial_health_score: Optional[float] = Field(None, ge=0, le=100, description="Financial health 0-100")
    supply_risk: Optional[str] = Field(None, description="Supply risk level: low/medium/high")
    risk_factors: list[str] = Field(default_factory=list, description="Identified risks")


class SupplyChainAnalysis(BaseModel):
    """Supply chain analysis results."""

    symbol: str
    key_suppliers: list[SupplierInfo] = Field(default_factory=list, description="Key suppliers identified")
    supplier_concentration_risk: Optional[float] = Field(None, description="Concentration risk score")
    geographic_risk: Optional[str] = Field(None, description="Geographic concentration risk")
    supply_chain_resilience: Optional[float] = Field(None, ge=0, le=100, description="Supply chain resilience score")

    # Aggregate Metrics
    avg_supplier_health: Optional[float] = Field(None, description="Average supplier financial health")
    suppliers_at_risk: int = Field(0, description="Number of suppliers with concerning metrics")
    critical_dependencies: list[str] = Field(default_factory=list, description="Critical supply dependencies")

    # Impact Assessment
    supply_disruption_risk: Optional[str] = Field(None, description="Overall supply disruption risk")
    cost_pressure_outlook: Optional[str] = Field(None, description="Input cost pressure outlook")
    supply_chain_threats: list[str] = Field(default_factory=list, description="Identified threats")
    supply_chain_opportunities: list[str] = Field(default_factory=list, description="Identified opportunities")

    analysis_summary: Optional[str] = Field(None, description="Summary of supply chain analysis")
    score: Optional[float] = Field(None, ge=0, le=100, description="Supply chain health score 0-100")


class CustomerInfo(BaseModel):
    """Information about a major customer."""

    symbol: Optional[str] = Field(None, description="Customer stock symbol if public")
    name: str = Field(..., description="Customer company name")
    segment: Optional[str] = Field(None, description="Customer segment/industry")
    revenue_contribution: Optional[float] = Field(None, description="Estimated % of revenue from this customer")

    # Financial Health
    market_cap: Optional[float] = Field(None, description="Customer market cap")
    revenue_growth: Optional[float] = Field(None, description="Customer revenue growth")
    profit_margin: Optional[float] = Field(None, description="Customer profit margin")

    # Performance
    stock_performance_ytd: Optional[float] = Field(None, description="YTD stock performance")
    business_outlook: Optional[str] = Field(None, description="Business outlook: positive/neutral/negative")

    # Risk Assessment
    financial_health_score: Optional[float] = Field(None, ge=0, le=100, description="Financial health 0-100")
    churn_risk: Optional[str] = Field(None, description="Customer churn risk: low/medium/high")


class CustomerAnalysis(BaseModel):
    """Customer analysis results."""

    symbol: str
    key_customers: list[CustomerInfo] = Field(default_factory=list, description="Key customers identified")
    customer_concentration_risk: Optional[float] = Field(None, description="Customer concentration risk")
    revenue_concentration_top5: Optional[float] = Field(None, description="% revenue from top 5 customers")

    # Segment Analysis
    segment_breakdown: dict[str, float] = Field(default_factory=dict, description="Revenue by customer segment")
    fastest_growing_segment: Optional[str] = Field(None, description="Fastest growing customer segment")
    declining_segments: list[str] = Field(default_factory=list, description="Declining customer segments")

    # Health Metrics
    avg_customer_health: Optional[float] = Field(None, description="Average customer financial health")
    customers_at_risk: int = Field(0, description="Number of customers with concerning metrics")
    customer_retention_outlook: Optional[str] = Field(None, description="Customer retention outlook")

    # Impact Assessment
    demand_outlook: Optional[str] = Field(None, description="Overall demand outlook")
    pricing_power: Optional[str] = Field(None, description="Pricing power assessment")
    customer_threats: list[str] = Field(default_factory=list, description="Customer-related threats")
    customer_opportunities: list[str] = Field(default_factory=list, description="Customer-related opportunities")

    analysis_summary: Optional[str] = Field(None, description="Summary of customer analysis")
    score: Optional[float] = Field(None, ge=0, le=100, description="Customer health score 0-100")


class CompetitorInfo(BaseModel):
    """Information about a competitor."""

    symbol: Optional[str] = Field(None, description="Competitor stock symbol")
    name: str = Field(..., description="Competitor company name")
    market_share: Optional[float] = Field(None, description="Estimated market share %")

    # Financials
    market_cap: Optional[float] = Field(None, description="Market capitalization")
    revenue: Optional[float] = Field(None, description="Annual revenue")
    revenue_growth: Optional[float] = Field(None, description="Revenue growth rate")
    profit_margin: Optional[float] = Field(None, description="Profit margin")
    pe_ratio: Optional[float] = Field(None, description="P/E ratio")

    # Stock Performance
    stock_performance_1m: Optional[float] = Field(None, description="1-month stock performance")
    stock_performance_ytd: Optional[float] = Field(None, description="YTD stock performance")

    # Competitive Position
    competitive_advantages: list[str] = Field(default_factory=list, description="Key competitive advantages")
    recent_developments: list[str] = Field(default_factory=list, description="Recent news/developments")
    threat_level: Optional[str] = Field(None, description="Threat level: low/medium/high")


class CompetitiveAnalysis(BaseModel):
    """Competitive landscape analysis."""

    symbol: str
    industry: Optional[str] = Field(None, description="Industry classification")
    market_position: Optional[str] = Field(None, description="Market position: leader/challenger/follower/niche")
    estimated_market_share: Optional[float] = Field(None, description="Estimated market share %")

    # Competitors
    key_competitors: list[CompetitorInfo] = Field(default_factory=list, description="Key competitors")
    competitive_intensity: Optional[str] = Field(None, description="Competitive intensity: low/medium/high")

    # Comparative Analysis
    revenue_rank: Optional[int] = Field(None, description="Revenue rank among competitors")
    margin_rank: Optional[int] = Field(None, description="Profit margin rank among competitors")
    growth_rank: Optional[int] = Field(None, description="Growth rate rank among competitors")
    valuation_rank: Optional[int] = Field(None, description="Valuation rank (P/E) among competitors")

    # Industry Dynamics
    industry_growth_rate: Optional[float] = Field(None, description="Industry growth rate")
    barriers_to_entry: Optional[str] = Field(None, description="Barriers to entry: low/medium/high")
    technological_disruption_risk: Optional[str] = Field(None, description="Tech disruption risk")

    # Porter's Five Forces Assessment
    supplier_power: Optional[str] = Field(None, description="Supplier bargaining power")
    buyer_power: Optional[str] = Field(None, description="Buyer bargaining power")
    threat_of_substitutes: Optional[str] = Field(None, description="Threat of substitutes")
    threat_of_new_entrants: Optional[str] = Field(None, description="Threat of new entrants")
    competitive_rivalry: Optional[str] = Field(None, description="Competitive rivalry intensity")

    # Strategic Assessment
    competitive_advantages: list[str] = Field(default_factory=list, description="Company's competitive advantages")
    competitive_weaknesses: list[str] = Field(default_factory=list, description="Competitive weaknesses")
    emerging_threats: list[str] = Field(default_factory=list, description="Emerging competitive threats")
    strategic_opportunities: list[str] = Field(default_factory=list, description="Strategic opportunities")

    analysis_summary: Optional[str] = Field(None, description="Summary of competitive analysis")
    score: Optional[float] = Field(None, ge=0, le=100, description="Competitive position score 0-100")


class MacroeconomicData(BaseModel):
    """Macroeconomic indicators and analysis."""

    # GDP
    gdp_growth_current: Optional[float] = Field(None, description="Current GDP growth rate")
    gdp_growth_forecast: Optional[float] = Field(None, description="Forecasted GDP growth")
    gdp_trend: Optional[str] = Field(None, description="GDP trend: accelerating/stable/decelerating")

    # Employment
    unemployment_rate: Optional[float] = Field(None, description="Unemployment rate")
    employment_trend: Optional[str] = Field(None, description="Employment trend")
    wage_growth: Optional[float] = Field(None, description="Wage growth rate")

    # Consumer
    consumer_confidence: Optional[float] = Field(None, description="Consumer confidence index")
    consumer_spending_growth: Optional[float] = Field(None, description="Consumer spending growth")
    retail_sales_growth: Optional[float] = Field(None, description="Retail sales growth")

    # Business
    business_confidence: Optional[float] = Field(None, description="Business confidence index")
    pmi_manufacturing: Optional[float] = Field(None, description="Manufacturing PMI")
    pmi_services: Optional[float] = Field(None, description="Services PMI")
    industrial_production: Optional[float] = Field(None, description="Industrial production growth")
    capacity_utilization: Optional[float] = Field(None, description="Capacity utilization rate")

    # Trade
    trade_balance: Optional[float] = Field(None, description="Trade balance")
    export_growth: Optional[float] = Field(None, description="Export growth rate")
    import_growth: Optional[float] = Field(None, description="Import growth rate")

    # Housing
    housing_starts: Optional[float] = Field(None, description="Housing starts (thousands)")
    home_price_growth: Optional[float] = Field(None, description="Home price growth rate")

    # Leading Indicators
    leading_economic_index: Optional[float] = Field(None, description="Leading Economic Index")
    economic_cycle_phase: Optional[str] = Field(None, description="Economic cycle: expansion/peak/contraction/trough")

    # Global Factors
    global_growth_outlook: Optional[str] = Field(None, description="Global growth outlook")
    china_growth: Optional[float] = Field(None, description="China GDP growth")
    europe_growth: Optional[float] = Field(None, description="Europe GDP growth")
    emerging_markets_outlook: Optional[str] = Field(None, description="Emerging markets outlook")

    # Sector-Specific Impact
    sector_outlook: dict[str, str] = Field(default_factory=dict, description="Outlook by sector")

    macro_risks: list[str] = Field(default_factory=list, description="Key macro risks")
    macro_opportunities: list[str] = Field(default_factory=list, description="Macro opportunities")

    analysis_summary: Optional[str] = Field(None, description="Summary of macro analysis")
    score: Optional[float] = Field(None, ge=0, le=100, description="Macro environment score 0-100")


class MonetaryPolicyData(BaseModel):
    """Federal Reserve and monetary policy analysis."""

    # Interest Rates
    fed_funds_rate: Optional[float] = Field(None, description="Current Fed Funds rate")
    fed_funds_target_upper: Optional[float] = Field(None, description="Fed Funds target upper bound")
    fed_funds_target_lower: Optional[float] = Field(None, description="Fed Funds target lower bound")

    # Rate Expectations
    next_meeting_date: Optional[str] = Field(None, description="Next FOMC meeting date")
    rate_hike_probability: Optional[float] = Field(None, description="Probability of rate hike")
    rate_cut_probability: Optional[float] = Field(None, description="Probability of rate cut")
    rate_hold_probability: Optional[float] = Field(None, description="Probability of rate hold")
    expected_rate_12m: Optional[float] = Field(None, description="Expected rate in 12 months")
    rate_direction: Optional[str] = Field(None, description="Rate direction: hawkish/neutral/dovish")

    # Inflation
    cpi_current: Optional[float] = Field(None, description="Current CPI (headline)")
    cpi_core: Optional[float] = Field(None, description="Core CPI (ex food & energy)")
    pce_current: Optional[float] = Field(None, description="Current PCE")
    pce_core: Optional[float] = Field(None, description="Core PCE (Fed's preferred measure)")
    inflation_trend: Optional[str] = Field(None, description="Inflation trend: rising/stable/falling")
    inflation_expectations: Optional[float] = Field(None, description="5-year inflation expectations")

    # Yield Curve
    treasury_2y: Optional[float] = Field(None, description="2-year Treasury yield")
    treasury_5y: Optional[float] = Field(None, description="5-year Treasury yield")
    treasury_10y: Optional[float] = Field(None, description="10-year Treasury yield")
    treasury_30y: Optional[float] = Field(None, description="30-year Treasury yield")
    yield_curve_spread: Optional[float] = Field(None, description="10Y-2Y spread (basis points)")
    yield_curve_status: Optional[str] = Field(None, description="Yield curve: normal/flat/inverted")
    recession_probability: Optional[float] = Field(None, description="Implied recession probability")

    # Quantitative Policy
    fed_balance_sheet: Optional[float] = Field(None, description="Fed balance sheet size (trillions)")
    qt_pace: Optional[str] = Field(None, description="QT pace: accelerating/steady/slowing/paused")

    # Dollar & Liquidity
    dxy_index: Optional[float] = Field(None, description="US Dollar Index (DXY)")
    dollar_trend: Optional[str] = Field(None, description="Dollar trend: strengthening/stable/weakening")
    financial_conditions: Optional[str] = Field(None, description="Financial conditions: tight/neutral/loose")

    # Market Impact Assessment
    rate_sensitivity_impact: Optional[str] = Field(None, description="Stock's rate sensitivity")
    sector_impact: Optional[str] = Field(None, description="Sector-specific monetary policy impact")
    valuation_impact: Optional[str] = Field(None, description="Valuation impact from rates")

    policy_risks: list[str] = Field(default_factory=list, description="Monetary policy risks")
    policy_opportunities: list[str] = Field(default_factory=list, description="Policy opportunities")

    analysis_summary: Optional[str] = Field(None, description="Summary of monetary policy analysis")
    score: Optional[float] = Field(None, ge=0, le=100, description="Monetary environment score 0-100")


class PortfolioStock(BaseModel):
    """Individual stock in portfolio."""

    symbol: str = Field(..., description="Stock ticker")
    name: Optional[str] = Field(None, description="Company name")
    weight: float = Field(..., ge=0, le=1, description="Portfolio weight (0-1)")
    shares: Optional[float] = Field(None, description="Number of shares")
    current_price: Optional[float] = Field(None, description="Current price")
    position_value: Optional[float] = Field(None, description="Position value")

    # Analysis Scores
    overall_score: Optional[float] = Field(None, description="Overall analysis score")
    fundamental_score: Optional[float] = Field(None, description="Fundamental score")
    technical_score: Optional[float] = Field(None, description="Technical score")
    risk_score: Optional[float] = Field(None, description="Risk score")

    # Risk Metrics
    expected_return: Optional[float] = Field(None, description="Expected return")
    volatility: Optional[float] = Field(None, description="Annual volatility")
    beta: Optional[float] = Field(None, description="Beta")
    sharpe_contribution: Optional[float] = Field(None, description="Contribution to portfolio Sharpe")

    # Rationale
    investment_thesis: Optional[str] = Field(None, description="Investment thesis")
    key_risks: list[str] = Field(default_factory=list, description="Key risks")


class PortfolioOptimization(BaseModel):
    """Portfolio optimization results."""

    # Portfolio Composition
    stocks: list[PortfolioStock] = Field(default_factory=list, description="Stocks in portfolio")
    total_stocks: int = Field(0, description="Number of stocks")
    cash_weight: float = Field(0.0, description="Cash allocation weight")

    # Portfolio Metrics
    expected_return: Optional[float] = Field(None, description="Portfolio expected return")
    portfolio_volatility: Optional[float] = Field(None, description="Portfolio volatility")
    sharpe_ratio: Optional[float] = Field(None, description="Portfolio Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Portfolio Sortino ratio")
    max_drawdown: Optional[float] = Field(None, description="Expected max drawdown")

    # Risk Metrics
    var_95: Optional[float] = Field(None, description="95% VaR")
    cvar_95: Optional[float] = Field(None, description="95% CVaR (Expected Shortfall)")
    beta: Optional[float] = Field(None, description="Portfolio beta")
    tracking_error: Optional[float] = Field(None, description="Tracking error vs benchmark")

    # Diversification
    diversification_ratio: Optional[float] = Field(None, description="Diversification ratio")
    concentration_top5: Optional[float] = Field(None, description="Weight in top 5 holdings")
    sector_weights: dict[str, float] = Field(default_factory=dict, description="Sector allocations")
    correlation_avg: Optional[float] = Field(None, description="Average pairwise correlation")

    # Optimization Details
    optimization_method: str = Field("mean_variance", description="Optimization method used")
    risk_free_rate: float = Field(0.05, description="Risk-free rate used")
    target_return: Optional[float] = Field(None, description="Target return if specified")
    risk_tolerance: Optional[str] = Field(None, description="Risk tolerance: conservative/moderate/aggressive")

    # Constraints Applied
    max_position_size: Optional[float] = Field(None, description="Max single position size")
    min_position_size: Optional[float] = Field(None, description="Min position size")
    sector_constraints: dict[str, float] = Field(default_factory=dict, description="Sector constraints")

    # Comparison to Benchmark
    benchmark: str = Field("SPY", description="Benchmark used")
    benchmark_return: Optional[float] = Field(None, description="Benchmark expected return")
    benchmark_volatility: Optional[float] = Field(None, description="Benchmark volatility")
    information_ratio: Optional[float] = Field(None, description="Information ratio")
    alpha: Optional[float] = Field(None, description="Expected alpha")

    # Rebalancing
    rebalance_frequency: Optional[str] = Field(None, description="Recommended rebalance frequency")
    turnover_estimate: Optional[float] = Field(None, description="Estimated annual turnover")

    portfolio_summary: Optional[str] = Field(None, description="Portfolio summary")
    investment_rationale: Optional[str] = Field(None, description="Overall investment rationale")


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
    sector: Optional[str] = Field(None, description="Company sector")
    industry: Optional[str] = Field(None, description="Company industry")

    # Core Agent Results
    stock_data: Optional[StockData] = None
    fundamental_analysis: Optional[FundamentalMetrics] = None
    technical_analysis: Optional[TechnicalIndicators] = None
    sentiment_analysis: Optional[SentimentData] = None
    risk_assessment: Optional[RiskMetrics] = None

    # Ecosystem Agent Results (Bridgewater-style)
    supply_chain_analysis: Optional[SupplyChainAnalysis] = None
    customer_analysis: Optional[CustomerAnalysis] = None
    competitive_analysis: Optional[CompetitiveAnalysis] = None
    macroeconomic_analysis: Optional[MacroeconomicData] = None
    monetary_policy_analysis: Optional[MonetaryPolicyData] = None

    # Aggregated Results
    agent_results: list[AgentResult] = Field(default_factory=list)

    # Overall Analysis
    overall_score: Optional[float] = Field(None, ge=0, le=100, description="Weighted overall score")
    recommendation: Optional[str] = Field(None, description="Buy/Hold/Sell recommendation")
    confidence: Optional[float] = Field(None, ge=0, le=100, description="Confidence in recommendation")
    conviction_level: Optional[str] = Field(None, description="Conviction: low/medium/high/very_high")

    # Position Sizing
    suggested_position_size: Optional[float] = Field(None, description="Suggested portfolio weight")
    position_rationale: Optional[str] = Field(None, description="Position sizing rationale")

    # Summary
    executive_summary: Optional[str] = Field(None, description="Executive summary of analysis")
    investment_thesis: Optional[str] = Field(None, description="Investment thesis")
    key_strengths: list[str] = Field(default_factory=list, description="Key strengths identified")
    key_risks: list[str] = Field(default_factory=list, description="Key risks identified")
    key_catalysts: list[str] = Field(default_factory=list, description="Potential catalysts")
    key_watchpoints: list[str] = Field(default_factory=list, description="Items to monitor")

    # Time Horizons
    short_term_outlook: Optional[str] = Field(None, description="1-3 month outlook")
    medium_term_outlook: Optional[str] = Field(None, description="3-12 month outlook")
    long_term_outlook: Optional[str] = Field(None, description="1-3 year outlook")

    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    total_execution_time: float = Field(0.0, description="Total analysis time in seconds")
    agents_executed: int = Field(0, description="Number of agents executed")
    successful_agents: int = Field(0, description="Number of successful agent executions")


class PortfolioAnalysisReport(BaseModel):
    """Complete portfolio analysis and recommendation report."""

    # Input Parameters
    investment_amount: float = Field(..., description="Total investment amount")
    risk_tolerance: str = Field("moderate", description="Risk tolerance level")
    investment_horizon: str = Field("medium", description="Investment horizon")

    # Individual Stock Analyses
    stock_analyses: list[AnalysisReport] = Field(default_factory=list, description="Individual stock reports")

    # Portfolio Optimization
    optimized_portfolio: Optional[PortfolioOptimization] = None

    # Macro Context
    macro_analysis: Optional[MacroeconomicData] = None
    monetary_analysis: Optional[MonetaryPolicyData] = None

    # Market Regime
    market_regime: Optional[str] = Field(None, description="Current market regime")
    regime_implications: list[str] = Field(default_factory=list, description="Regime implications")

    # Asset Allocation
    recommended_equity_allocation: Optional[float] = Field(None, description="Recommended equity %")
    recommended_bond_allocation: Optional[float] = Field(None, description="Recommended bond %")
    recommended_cash_allocation: Optional[float] = Field(None, description="Recommended cash %")
    recommended_alternatives: Optional[float] = Field(None, description="Recommended alternatives %")

    # Summary
    portfolio_summary: Optional[str] = Field(None, description="Portfolio summary")
    key_themes: list[str] = Field(default_factory=list, description="Key investment themes")
    risk_warnings: list[str] = Field(default_factory=list, description="Risk warnings")
    action_items: list[str] = Field(default_factory=list, description="Recommended actions")

    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    total_execution_time: float = Field(0.0, description="Total analysis time")
