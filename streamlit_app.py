"""
Stock Analysis Multi-Agent Platform - Streamlit UI

A comprehensive web interface for analyzing stocks using multiple specialized agents.
Run with: streamlit run streamlit_app.py
"""

import asyncio
import json
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from src.agents.master_agent import MasterAgent
from src.agents.portfolio_optimizer_agent import PortfolioOptimizerAgent
from src.models.schemas import AnalysisReport, AgentStatus
from src.screener import (
    StockScreener,
    ScreenedStock,
    ScreenerCriteria,
    StockCategory,
    Recommendation,
    get_sp500_symbols,
    get_nasdaq100_symbols,
    get_dow30_symbols,
    get_sector_stocks,
    get_all_major_stocks,
    get_dividend_aristocrats,
    get_all_sectors,
    get_watchlist_manager,
    WatchlistStock,
    # Portfolio Analytics
    get_portfolio_analytics,
    # Technical Analysis
    get_technical_analyzer,
    TrendDirection,
    SignalType,
    # Peer Comparison
    get_peer_analyzer,
    # Events Calendar
    get_events_calendar,
    # Institutional Tracking
    get_institutional_tracker,
)
from src.utils.yfinance_cache import get_ticker_info, get_ticker_history


# Page configuration
st.set_page_config(
    page_title="Stock Analysis Platform - Bridgewater Style",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .score-high { color: #28a745; }
    .score-medium { color: #ffc107; }
    .score-low { color: #dc3545; }
    .recommendation-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .recommendation-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .recommendation-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    /* Tab container styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        padding: 10px 15px;
        border-radius: 12px;
        margin-bottom: 20px;
        flex-wrap: wrap;
        justify-content: center;
    }

    /* Individual tab styling */
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 14px;
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        transition: all 0.3s ease;
        margin: 4px;
    }

    /* Tab hover effect */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e3f2fd;
        border-color: #1976d2;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Active/Selected tab */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1976d2 0%, #1565c0 100%) !important;
        color: white !important;
        border-color: #1565c0 !important;
        box-shadow: 0 4px 12px rgba(25, 118, 210, 0.4);
    }

    /* Tab highlight bar */
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }

    /* Tab panel content */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 20px;
    }

    /* Core agent tabs - first 6 tabs (blue theme) */
    .stTabs [data-baseweb="tab-list"] button:nth-child(-n+6) {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-color: #64b5f6;
    }

    .stTabs [data-baseweb="tab-list"] button:nth-child(-n+6):hover {
        background: linear-gradient(135deg, #bbdefb 0%, #90caf9 100%);
        border-color: #42a5f5;
    }

    /* Ecosystem agent tabs - tabs 7-11 (purple/violet theme) */
    .stTabs [data-baseweb="tab-list"] button:nth-child(n+7):nth-child(-n+11) {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        border-color: #ba68c8;
    }

    .stTabs [data-baseweb="tab-list"] button:nth-child(n+7):nth-child(-n+11):hover {
        background: linear-gradient(135deg, #e1bee7 0%, #ce93d8 100%);
        border-color: #ab47bc;
    }

    /* Tab text styling */
    .stTabs [data-baseweb="tab"] span {
        font-weight: 600 !important;
    }
    .ecosystem-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin: 0.5rem 0;
    }
    .macro-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin: 0.5rem 0;
    }
    .portfolio-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def get_score_color(score: Optional[float]) -> str:
    """Get color based on score value."""
    if score is None:
        return "gray"
    if score >= 65:
        return "green"
    elif score >= 40:
        return "orange"
    return "red"


def get_recommendation_class(recommendation: Optional[str]) -> str:
    """Get CSS class for recommendation."""
    if recommendation is None:
        return ""
    rec_upper = recommendation.upper()
    if "BUY" in rec_upper:
        return "recommendation-buy"
    elif "SELL" in rec_upper:
        return "recommendation-sell"
    return "recommendation-hold"


def create_gauge_chart(value: float, title: str, max_val: float = 100) -> go.Figure:
    """Create a gauge chart for scores."""
    color = get_score_color(value)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': '#ffcccb'},
                {'range': [40, 65], 'color': '#ffffcc'},
                {'range': [65, 100], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def create_price_chart(symbol: str, period: str = "6mo") -> go.Figure:
    """Create a candlestick price chart with volume."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)

    if df.empty:
        return None

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{symbol} Price', 'Volume'),
        row_heights=[0.7, 0.3]
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )

    # Add moving averages
    if len(df) >= 20:
        df['SMA20'] = df['Close'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA20'], name='SMA 20',
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )

    if len(df) >= 50:
        df['SMA50'] = df['Close'].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA50'], name='SMA 50',
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )

    # Volume bars
    colors = ['red' if row['Open'] > row['Close'] else 'green'
              for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )

    fig.update_layout(
        height=500,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])

    return fig


def create_technical_indicators_chart(symbol: str) -> go.Figure:
    """Create technical indicators chart (RSI, MACD)."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="6mo")

    if df.empty or len(df) < 26:
        return None

    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Histogram'] = df['MACD'] - df['Signal']

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('RSI (14)', 'MACD'),
        row_heights=[0.5, 0.5]
    )

    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
        row=1, col=1
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Signal'], name='Signal', line=dict(color='orange')),
        row=2, col=1
    )
    colors = ['green' if val >= 0 else 'red' for val in df['Histogram']]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Histogram'], name='Histogram', marker_color=colors),
        row=2, col=1
    )

    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_agent_scores_chart(report: AnalysisReport) -> go.Figure:
    """Create a radar chart of agent scores."""
    categories = []
    scores = []

    for result in report.agent_results:
        if result.status == AgentStatus.COMPLETED and result.score is not None:
            categories.append(result.agent_type.capitalize())
            scores.append(result.score)

    if not categories:
        return None

    # Close the radar chart
    categories.append(categories[0])
    scores.append(scores[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        name='Agent Scores',
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False,
        height=350,
        margin=dict(l=80, r=80, t=40, b=40)
    )

    return fig


def display_stock_overview(report: AnalysisReport):
    """Display stock overview section."""
    st.subheader("📊 Stock Overview")

    if report.stock_data:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if report.stock_data.current_price:
                price_change = None
                if report.stock_data.previous_close:
                    price_change = report.stock_data.current_price - report.stock_data.previous_close
                    pct_change = (price_change / report.stock_data.previous_close) * 100
                    delta_str = f"{price_change:+.2f} ({pct_change:+.2f}%)"
                else:
                    delta_str = None
                st.metric(
                    "Current Price",
                    f"${report.stock_data.current_price:.2f}",
                    delta=delta_str
                )

        with col2:
            if report.stock_data.market_cap:
                cap = report.stock_data.market_cap
                if cap >= 1e12:
                    cap_str = f"${cap/1e12:.2f}T"
                elif cap >= 1e9:
                    cap_str = f"${cap/1e9:.2f}B"
                else:
                    cap_str = f"${cap/1e6:.2f}M"
                st.metric("Market Cap", cap_str)

        with col3:
            if report.stock_data.volume:
                vol = report.stock_data.volume
                if vol >= 1e6:
                    vol_str = f"{vol/1e6:.2f}M"
                else:
                    vol_str = f"{vol/1e3:.2f}K"
                st.metric("Volume", vol_str)

        with col4:
            if report.stock_data.high and report.stock_data.low:
                st.metric("Day Range", f"${report.stock_data.low:.2f} - ${report.stock_data.high:.2f}")


def display_overall_analysis(report: AnalysisReport):
    """Display overall analysis results."""
    st.subheader("🎯 Overall Analysis")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if report.overall_score is not None:
            fig = create_gauge_chart(report.overall_score, "Overall Score")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Recommendation")
        if report.recommendation:
            rec_class = get_recommendation_class(report.recommendation)
            st.markdown(f'<div class="{rec_class}">{report.recommendation}</div>',
                       unsafe_allow_html=True)

        st.markdown("### Confidence")
        if report.confidence is not None:
            st.progress(report.confidence / 100)
            st.write(f"{report.confidence:.0f}%")

    with col3:
        fig = create_agent_scores_chart(report)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


def display_fundamental_analysis(report: AnalysisReport):
    """Display fundamental analysis tab."""
    if report.fundamental_analysis:
        fa = report.fundamental_analysis

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Valuation Metrics")
            metrics_data = {
                "P/E Ratio": f"{fa.pe_ratio:.2f}" if fa.pe_ratio else "N/A",
                "Forward P/E": f"{fa.forward_pe:.2f}" if fa.forward_pe else "N/A",
                "PEG Ratio": f"{fa.peg_ratio:.2f}" if fa.peg_ratio else "N/A",
                "Price/Book": f"{fa.price_to_book:.2f}" if fa.price_to_book else "N/A",
                "Price/Sales": f"{fa.price_to_sales:.2f}" if fa.price_to_sales else "N/A",
            }
            for metric, value in metrics_data.items():
                st.write(f"**{metric}:** {value}")

            st.markdown("#### Growth Metrics")
            if fa.revenue_growth:
                st.write(f"**Revenue Growth:** {fa.revenue_growth*100:.1f}%")
            if fa.earnings_growth:
                st.write(f"**Earnings Growth:** {fa.earnings_growth*100:.1f}%")

        with col2:
            st.markdown("#### Profitability")
            if fa.profit_margin:
                st.write(f"**Profit Margin:** {fa.profit_margin*100:.1f}%")
            if fa.operating_margin:
                st.write(f"**Operating Margin:** {fa.operating_margin*100:.1f}%")
            if fa.return_on_equity:
                st.write(f"**ROE:** {fa.return_on_equity*100:.1f}%")
            if fa.return_on_assets:
                st.write(f"**ROA:** {fa.return_on_assets*100:.1f}%")

            st.markdown("#### Financial Health")
            if fa.debt_to_equity:
                st.write(f"**Debt/Equity:** {fa.debt_to_equity:.2f}")
            if fa.current_ratio:
                st.write(f"**Current Ratio:** {fa.current_ratio:.2f}")

        if fa.analysis_summary:
            st.info(f"**Summary:** {fa.analysis_summary}")
    else:
        st.warning("Fundamental analysis data not available")


def display_technical_analysis(report: AnalysisReport):
    """Display technical analysis tab."""
    if report.technical_analysis:
        ta = report.technical_analysis

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Trend & Momentum")
            if ta.trend_direction:
                trend_color = "green" if ta.trend_direction == "bullish" else "red" if ta.trend_direction == "bearish" else "gray"
                st.markdown(f"**Trend:** :{trend_color}[{ta.trend_direction.upper()}]")
            if ta.rsi:
                rsi_status = "Overbought" if ta.rsi > 70 else "Oversold" if ta.rsi < 30 else "Neutral"
                st.write(f"**RSI (14):** {ta.rsi:.1f} ({rsi_status})")
            if ta.adx:
                st.write(f"**ADX:** {ta.adx:.1f}")

            st.markdown("#### MACD")
            if ta.macd:
                st.write(f"**MACD:** {ta.macd:.4f}")
            if ta.macd_signal:
                st.write(f"**Signal:** {ta.macd_signal:.4f}")

        with col2:
            st.markdown("#### Moving Averages")
            if ta.sma_20:
                st.write(f"**SMA 20:** ${ta.sma_20:.2f}")
            if ta.sma_50:
                st.write(f"**SMA 50:** ${ta.sma_50:.2f}")
            if ta.sma_200:
                st.write(f"**SMA 200:** ${ta.sma_200:.2f}")

            st.markdown("#### Signals")
            if ta.buy_signals:
                st.success(f"**Buy Signals ({len(ta.buy_signals)}):** {', '.join(ta.buy_signals[:3])}")
            if ta.sell_signals:
                st.error(f"**Sell Signals ({len(ta.sell_signals)}):** {', '.join(ta.sell_signals[:3])}")

        # Technical chart
        st.markdown("#### Technical Indicators Chart")
        tech_chart = create_technical_indicators_chart(report.symbol)
        if tech_chart:
            st.plotly_chart(tech_chart, use_container_width=True)

        if ta.analysis_summary:
            st.info(f"**Summary:** {ta.analysis_summary}")
    else:
        st.warning("Technical analysis data not available")


def display_sentiment_analysis(report: AnalysisReport):
    """Display sentiment analysis tab."""
    if report.sentiment_analysis:
        sa = report.sentiment_analysis

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Overall Sentiment")
            if sa.overall_sentiment:
                sent_color = "green" if sa.overall_sentiment == "bullish" else "red" if sa.overall_sentiment == "bearish" else "gray"
                st.markdown(f"**Sentiment:** :{sent_color}[{sa.overall_sentiment.upper()}]")
            if sa.sentiment_score is not None:
                st.write(f"**Sentiment Score:** {sa.sentiment_score:.2f}")

            st.markdown("#### News Analysis")
            st.write(f"**Articles Analyzed:** {sa.news_count}")

            # News sentiment breakdown
            if sa.news_count > 0:
                news_data = pd.DataFrame({
                    'Type': ['Positive', 'Neutral', 'Negative'],
                    'Count': [sa.positive_news_count, sa.neutral_news_count, sa.negative_news_count]
                })
                fig = px.pie(news_data, values='Count', names='Type',
                            color='Type',
                            color_discrete_map={'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'})
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Analyst Ratings")
            if sa.analyst_rating:
                st.write(f"**Consensus:** {sa.analyst_rating}")
            if sa.target_price:
                st.write(f"**Target Price:** ${sa.target_price:.2f}")
                if report.stock_data and report.stock_data.current_price:
                    upside = ((sa.target_price - report.stock_data.current_price) / report.stock_data.current_price) * 100
                    st.write(f"**Implied Upside:** {upside:+.1f}%")

            # Ratings breakdown
            if sa.buy_ratings or sa.hold_ratings or sa.sell_ratings:
                ratings_data = pd.DataFrame({
                    'Rating': ['Buy', 'Hold', 'Sell'],
                    'Count': [sa.buy_ratings, sa.hold_ratings, sa.sell_ratings]
                })
                fig = px.bar(ratings_data, x='Rating', y='Count',
                            color='Rating',
                            color_discrete_map={'Buy': 'green', 'Hold': 'orange', 'Sell': 'red'})
                fig.update_layout(height=250, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)

        # Key headlines
        if sa.key_headlines:
            st.markdown("#### Recent Headlines")
            for headline in sa.key_headlines[:5]:
                st.write(f"• {headline}")

        if sa.analysis_summary:
            st.info(f"**Summary:** {sa.analysis_summary}")
    else:
        st.warning("Sentiment analysis data not available")


def display_risk_analysis(report: AnalysisReport):
    """Display risk assessment tab."""
    if report.risk_assessment:
        ra = report.risk_assessment

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Risk Level")
            if ra.risk_level:
                risk_colors = {"low": "green", "medium": "orange", "high": "red", "very_high": "red"}
                st.markdown(f"**Risk Level:** :{risk_colors.get(ra.risk_level, 'gray')}[{ra.risk_level.upper()}]")

            st.markdown("#### Volatility Metrics")
            if ra.volatility_daily:
                st.write(f"**Daily Volatility:** {ra.volatility_daily*100:.2f}%")
            if ra.volatility_annual:
                st.write(f"**Annual Volatility:** {ra.volatility_annual*100:.1f}%")
            if ra.beta is not None:
                st.write(f"**Beta:** {ra.beta:.2f}")

            st.markdown("#### Value at Risk")
            if ra.var_95:
                st.write(f"**VaR (95%):** {ra.var_95*100:.2f}%")
            if ra.var_99:
                st.write(f"**VaR (99%):** {ra.var_99*100:.2f}%")

        with col2:
            st.markdown("#### Risk-Adjusted Returns")
            if ra.sharpe_ratio is not None:
                sharpe_color = "green" if ra.sharpe_ratio > 1 else "orange" if ra.sharpe_ratio > 0 else "red"
                st.markdown(f"**Sharpe Ratio:** :{sharpe_color}[{ra.sharpe_ratio:.2f}]")
            if ra.sortino_ratio is not None:
                st.write(f"**Sortino Ratio:** {ra.sortino_ratio:.2f}")

            st.markdown("#### Drawdown")
            if ra.max_drawdown:
                st.write(f"**Max Drawdown:** {ra.max_drawdown*100:.1f}%")
            if ra.current_drawdown:
                st.write(f"**Current Drawdown:** {ra.current_drawdown*100:.1f}%")

        # Risk factors
        if ra.risk_factors:
            st.markdown("#### Identified Risk Factors")
            for factor in ra.risk_factors:
                st.warning(f"⚠️ {factor}")

        if ra.analysis_summary:
            st.info(f"**Summary:** {ra.analysis_summary}")
    else:
        st.warning("Risk assessment data not available")


def display_supply_chain_analysis(report: AnalysisReport):
    """Display supply chain analysis tab."""
    if report.supply_chain_analysis:
        sc = report.supply_chain_analysis

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Supply Chain Health")
            if sc.supply_disruption_risk:
                health_color = "green" if sc.supply_disruption_risk == "low" else "orange" if sc.supply_disruption_risk == "medium" else "red"
                st.markdown(f"**Disruption Risk:** :{health_color}[{sc.supply_disruption_risk.upper()}]")

            if sc.supply_chain_resilience is not None:
                fig = create_gauge_chart(sc.supply_chain_resilience, "Resilience Score")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Risk Metrics")
            if sc.supplier_concentration_risk is not None:
                st.write(f"**Concentration Risk:** {sc.supplier_concentration_risk*100:.1f}%")
            if sc.geographic_risk:
                st.write(f"**Geographic Risk:** {sc.geographic_risk}")
            if sc.suppliers_at_risk > 0:
                st.write(f"**Suppliers at Risk:** {sc.suppliers_at_risk}")

        with col2:
            st.markdown("#### Key Suppliers")
            if sc.key_suppliers:
                for supplier in sc.key_suppliers[:5]:
                    health_score = getattr(supplier, 'financial_health_score', None)
                    # Convert numeric score (0-100) to status icon
                    if health_score is not None:
                        status_icon = "🟢" if health_score >= 70 else "🟡" if health_score >= 40 else "🔴"
                    else:
                        status_icon = "⚪"
                    st.write(f"{status_icon} **{supplier.name}** ({supplier.symbol or 'Private'})")
                    if hasattr(supplier, 'revenue_dependency') and supplier.revenue_dependency:
                        st.caption(f"   Revenue Dependency: {supplier.revenue_dependency*100:.1f}%")

            if sc.critical_dependencies:
                st.markdown("#### Critical Dependencies")
                for dep in sc.critical_dependencies[:3]:
                    st.write(f"• {dep}")

        # Supply chain risks
        if sc.supply_chain_threats:
            st.markdown("#### Identified Supply Chain Threats")
            for threat in sc.supply_chain_threats:
                st.warning(f"⚠️ {threat}")

        if sc.analysis_summary:
            st.info(f"**Summary:** {sc.analysis_summary}")
    else:
        st.warning("Supply chain analysis data not available")


def display_customer_analysis(report: AnalysisReport):
    """Display customer analysis tab."""
    if report.customer_analysis:
        ca = report.customer_analysis

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Customer Base Health")
            if ca.customer_retention_outlook:
                health_color = "green" if ca.customer_retention_outlook == "positive" else "orange" if ca.customer_retention_outlook == "stable" else "red"
                st.markdown(f"**Retention Outlook:** :{health_color}[{ca.customer_retention_outlook.upper()}]")

            if ca.customer_concentration_risk is not None:
                risk_level = "Low" if ca.customer_concentration_risk < 0.3 else "Medium" if ca.customer_concentration_risk < 0.6 else "High"
                st.write(f"**Concentration Risk:** {ca.customer_concentration_risk*100:.1f}% ({risk_level})")

            if ca.revenue_concentration_top5 is not None:
                st.write(f"**Top 5 Customer Revenue:** {ca.revenue_concentration_top5*100:.1f}%")

            if ca.pricing_power:
                st.write(f"**Pricing Power:** {ca.pricing_power}")

            st.markdown("#### Demand Outlook")
            if ca.demand_outlook:
                outlook_color = "green" if ca.demand_outlook == "positive" else "red" if ca.demand_outlook == "negative" else "gray"
                st.markdown(f"**Outlook:** :{outlook_color}[{ca.demand_outlook.upper()}]")

        with col2:
            st.markdown("#### Key Customers/Segments")
            if ca.key_customers:
                for customer in ca.key_customers[:5]:
                    health_score = getattr(customer, 'financial_health_score', None)
                    # Convert numeric score (0-100) to status icon
                    if health_score is not None:
                        status_icon = "🟢" if health_score >= 70 else "🟡" if health_score >= 40 else "🔴"
                    else:
                        status_icon = "⚪"
                    segment = getattr(customer, 'segment', 'N/A') or 'N/A'
                    st.write(f"{status_icon} **{customer.name}** ({segment})")
                    if customer.revenue_contribution:
                        st.caption(f"   Revenue Contribution: {customer.revenue_contribution*100:.1f}%")

            if ca.segment_breakdown:
                st.markdown("#### Segment Breakdown")
                for segment, pct in list(ca.segment_breakdown.items())[:4]:
                    st.write(f"• {segment}: {pct*100:.1f}%")

        # Customer risks
        if ca.customer_threats:
            st.markdown("#### Customer-Related Risks")
            for threat in ca.customer_threats:
                st.warning(f"⚠️ {threat}")

        if ca.analysis_summary:
            st.info(f"**Summary:** {ca.analysis_summary}")
    else:
        st.warning("Customer analysis data not available")


def display_competitive_analysis(report: AnalysisReport):
    """Display competitive analysis tab."""
    if report.competitive_analysis:
        comp = report.competitive_analysis

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Competitive Position")
            if comp.market_position:
                pos_color = "green" if comp.market_position == "leader" else "orange" if comp.market_position == "challenger" else "gray"
                st.markdown(f"**Position:** :{pos_color}[{comp.market_position.upper()}]")

            if comp.estimated_market_share is not None:
                st.write(f"**Market Share:** {comp.estimated_market_share*100:.1f}%")

            if comp.competitive_intensity:
                intensity_color = "red" if comp.competitive_intensity == "high" else "orange" if comp.competitive_intensity == "medium" else "green"
                st.markdown(f"**Competitive Intensity:** :{intensity_color}[{comp.competitive_intensity.upper()}]")

            st.markdown("#### Rankings")
            if comp.revenue_rank:
                st.write(f"**Revenue Rank:** #{comp.revenue_rank}")
            if comp.margin_rank:
                st.write(f"**Margin Rank:** #{comp.margin_rank}")
            if comp.growth_rank:
                st.write(f"**Growth Rank:** #{comp.growth_rank}")

        with col2:
            st.markdown("#### Key Competitors")
            if comp.key_competitors:
                # Create comparison dataframe
                comp_data = []
                for competitor in comp.key_competitors[:5]:
                    mkt_share = getattr(competitor, 'market_share', None)
                    rev_growth = getattr(competitor, 'revenue_growth', None)
                    threat = getattr(competitor, 'threat_level', None)
                    comp_data.append({
                        'Name': competitor.name,
                        'Market Share': f"{mkt_share*100:.1f}%" if mkt_share else "N/A",
                        'Growth': f"{rev_growth*100:.1f}%" if rev_growth else "N/A",
                        'Threat': threat or "N/A"
                    })
                if comp_data:
                    st.dataframe(pd.DataFrame(comp_data), hide_index=True)

            st.markdown("#### Porter's Five Forces")
            # Display individual Porter's Five Forces attributes
            forces_available = any([comp.competitive_rivalry, comp.supplier_power, comp.buyer_power,
                                   comp.threat_of_substitutes, comp.threat_of_new_entrants])
            if forces_available:
                forces_data = []
                if comp.competitive_rivalry:
                    forces_data.append({'Force': 'Competitive Rivalry', 'Level': comp.competitive_rivalry})
                if comp.supplier_power:
                    forces_data.append({'Force': 'Supplier Power', 'Level': comp.supplier_power})
                if comp.buyer_power:
                    forces_data.append({'Force': 'Buyer Power', 'Level': comp.buyer_power})
                if comp.threat_of_substitutes:
                    forces_data.append({'Force': 'Threat of Substitutes', 'Level': comp.threat_of_substitutes})
                if comp.threat_of_new_entrants:
                    forces_data.append({'Force': 'Threat of New Entrants', 'Level': comp.threat_of_new_entrants})
                if forces_data:
                    st.dataframe(pd.DataFrame(forces_data), hide_index=True)

        # Competitive advantages and threats
        col3, col4 = st.columns(2)
        with col3:
            if comp.competitive_advantages:
                st.markdown("#### Competitive Advantages")
                for adv in comp.competitive_advantages:
                    st.success(f"✓ {adv}")
        with col4:
            if comp.emerging_threats:
                st.markdown("#### Emerging Threats")
                for threat in comp.emerging_threats:
                    st.error(f"⚠ {threat}")

        if comp.analysis_summary:
            st.info(f"**Summary:** {comp.analysis_summary}")
    else:
        st.warning("Competitive analysis data not available")


def display_macro_analysis(report: AnalysisReport):
    """Display macroeconomic analysis tab."""
    if report.macroeconomic_analysis:
        macro = report.macroeconomic_analysis

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Economic Environment")
            if macro.economic_cycle_phase:
                cycle_colors = {"expansion": "green", "peak": "orange", "contraction": "red", "trough": "blue"}
                st.markdown(f"**Economic Cycle:** :{cycle_colors.get(macro.economic_cycle_phase, 'gray')}[{macro.economic_cycle_phase.upper()}]")

            if macro.gdp_growth_current is not None:
                st.write(f"**GDP Growth:** {macro.gdp_growth_current*100:.2f}%")
            if macro.unemployment_rate is not None:
                st.write(f"**Unemployment:** {macro.unemployment_rate*100:.1f}%")
            if macro.wage_growth is not None:
                st.write(f"**Wage Growth:** {macro.wage_growth*100:.1f}%")

            st.markdown("#### Consumer & Business")
            if macro.consumer_confidence is not None:
                st.write(f"**Consumer Confidence:** {macro.consumer_confidence:.1f}")
            if macro.business_confidence is not None:
                st.write(f"**Business Confidence:** {macro.business_confidence:.1f}")
            if macro.consumer_spending_growth is not None:
                st.write(f"**Consumer Spending:** {macro.consumer_spending_growth*100:.1f}%")

        with col2:
            st.markdown("#### Manufacturing & Services")
            if macro.pmi_manufacturing is not None:
                pmi_color = "green" if macro.pmi_manufacturing > 50 else "red"
                st.markdown(f"**Manufacturing PMI:** :{pmi_color}[{macro.pmi_manufacturing:.1f}]")
            if macro.pmi_services is not None:
                pmi_color = "green" if macro.pmi_services > 50 else "red"
                st.markdown(f"**Services PMI:** :{pmi_color}[{macro.pmi_services:.1f}]")
            if macro.industrial_production is not None:
                st.write(f"**Industrial Production:** {macro.industrial_production*100:.1f}%")

            st.markdown("#### Global & Trade")
            if macro.global_growth_outlook:
                outlook_color = "green" if macro.global_growth_outlook == "positive" else "red" if macro.global_growth_outlook == "negative" else "gray"
                st.markdown(f"**Global Outlook:** :{outlook_color}[{macro.global_growth_outlook.upper()}]")
            if macro.export_growth is not None:
                st.write(f"**Export Growth:** {macro.export_growth*100:.1f}%")

        # Leading indicators
        if macro.leading_economic_index is not None or macro.gdp_trend:
            st.markdown("#### Leading Indicators")
            col3, col4 = st.columns(2)
            with col3:
                if macro.leading_economic_index is not None:
                    st.write(f"**Leading Economic Index:** {macro.leading_economic_index:.1f}")
            with col4:
                if macro.gdp_trend:
                    st.write(f"**GDP Trend:** {macro.gdp_trend}")

        if hasattr(macro, 'analysis_summary') and macro.analysis_summary:
            st.info(f"**Summary:** {macro.analysis_summary}")
    else:
        st.warning("Macroeconomic analysis data not available")


def display_monetary_analysis(report: AnalysisReport):
    """Display monetary policy analysis tab."""
    if report.monetary_policy_analysis:
        mon = report.monetary_policy_analysis

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Federal Reserve Policy")
            if mon.fed_funds_rate is not None:
                st.write(f"**Fed Funds Rate:** {mon.fed_funds_rate:.2f}%")
            if mon.rate_direction:
                dir_color = "red" if mon.rate_direction == "hawkish" else "green" if mon.rate_direction == "dovish" else "gray"
                st.markdown(f"**Rate Direction:** :{dir_color}[{mon.rate_direction.upper()}]")
            if mon.rate_hike_probability is not None:
                st.write(f"**Rate Hike Probability:** {mon.rate_hike_probability*100:.0f}%")
            if mon.rate_cut_probability is not None:
                st.write(f"**Rate Cut Probability:** {mon.rate_cut_probability*100:.0f}%")

            st.markdown("#### Yield Curve")
            if mon.yield_curve_status:
                yc_color = "red" if mon.yield_curve_status == "inverted" else "green" if mon.yield_curve_status == "normal" else "orange"
                st.markdown(f"**Yield Curve:** :{yc_color}[{mon.yield_curve_status.upper()}]")
            if mon.yield_curve_spread is not None:
                # Spread is in basis points, convert to percentage
                spread_pct = mon.yield_curve_spread / 100
                st.write(f"**10Y-2Y Spread:** {spread_pct:.2f}%")
            if mon.recession_probability is not None:
                recession_color = "green" if mon.recession_probability < 0.3 else "orange" if mon.recession_probability < 0.6 else "red"
                st.markdown(f"**Recession Probability:** :{recession_color}[{mon.recession_probability*100:.0f}%]")

        with col2:
            st.markdown("#### Treasury Yields")
            if mon.treasury_2y is not None:
                st.write(f"**2Y Treasury:** {mon.treasury_2y:.2f}%")
            if mon.treasury_5y is not None:
                st.write(f"**5Y Treasury:** {mon.treasury_5y:.2f}%")
            if mon.treasury_10y is not None:
                st.write(f"**10Y Treasury:** {mon.treasury_10y:.2f}%")
            if mon.treasury_30y is not None:
                st.write(f"**30Y Treasury:** {mon.treasury_30y:.2f}%")

            st.markdown("#### Inflation")
            if mon.cpi_current is not None:
                st.write(f"**CPI (Headline):** {mon.cpi_current:.1f}%")
            if mon.cpi_core is not None:
                st.write(f"**CPI (Core):** {mon.cpi_core:.1f}%")
            if mon.inflation_trend:
                trend_color = "red" if mon.inflation_trend in ["rising", "above_target"] else "green" if mon.inflation_trend == "falling" else "gray"
                st.markdown(f"**Inflation Trend:** :{trend_color}[{mon.inflation_trend.upper()}]")

        # Dollar and liquidity
        col3, col4 = st.columns(2)
        with col3:
            if mon.dxy_index is not None or mon.dollar_trend:
                st.markdown("#### US Dollar")
                if mon.dxy_index is not None:
                    st.write(f"**DXY Index:** {mon.dxy_index:.1f}")
                if mon.dollar_trend:
                    st.write(f"**Dollar Trend:** {mon.dollar_trend}")
        with col4:
            if mon.financial_conditions or mon.qt_pace:
                st.markdown("#### Financial Conditions")
                if mon.financial_conditions:
                    cond_color = "red" if mon.financial_conditions == "tight" else "green" if mon.financial_conditions == "loose" else "gray"
                    st.markdown(f"**Conditions:** :{cond_color}[{mon.financial_conditions.upper()}]")
                if mon.qt_pace:
                    st.write(f"**QT Pace:** {mon.qt_pace}")

        if hasattr(mon, 'analysis_summary') and mon.analysis_summary:
            st.info(f"**Summary:** {mon.analysis_summary}")
    else:
        st.warning("Monetary policy analysis data not available")


def display_agent_results(report: AnalysisReport):
    """Display individual agent results in tabs."""
    st.subheader("📈 Detailed Analysis")

    # Determine which tabs to show based on available data
    tab_names = ["Fundamental", "Technical", "Sentiment", "Risk"]

    # Add ecosystem tabs if data available
    if report.supply_chain_analysis:
        tab_names.append("Supply Chain")
    if report.customer_analysis:
        tab_names.append("Customers")
    if report.competitive_analysis:
        tab_names.append("Competitive")
    if report.macroeconomic_analysis:
        tab_names.append("Macro")
    if report.monetary_policy_analysis:
        tab_names.append("Monetary")

    tabs = st.tabs(tab_names)
    tab_idx = 0

    # Core analysis tabs
    with tabs[tab_idx]:
        display_fundamental_analysis(report)
    tab_idx += 1

    with tabs[tab_idx]:
        display_technical_analysis(report)
    tab_idx += 1

    with tabs[tab_idx]:
        display_sentiment_analysis(report)
    tab_idx += 1

    with tabs[tab_idx]:
        display_risk_analysis(report)
    tab_idx += 1

    # Ecosystem tabs
    if report.supply_chain_analysis:
        with tabs[tab_idx]:
            display_supply_chain_analysis(report)
        tab_idx += 1

    if report.customer_analysis:
        with tabs[tab_idx]:
            display_customer_analysis(report)
        tab_idx += 1

    if report.competitive_analysis:
        with tabs[tab_idx]:
            display_competitive_analysis(report)
        tab_idx += 1

    if report.macroeconomic_analysis:
        with tabs[tab_idx]:
            display_macro_analysis(report)
        tab_idx += 1

    if report.monetary_policy_analysis:
        with tabs[tab_idx]:
            display_monetary_analysis(report)


def display_key_insights(report: AnalysisReport):
    """Display key insights section."""
    st.subheader("🔍 Key Insights")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Strengths")
        if report.key_strengths:
            for strength in report.key_strengths:
                st.success(f"✓ {strength}")
        else:
            st.write("No significant strengths identified")

    with col2:
        st.markdown("#### Risks")
        if report.key_risks:
            for risk in report.key_risks:
                st.error(f"⚠ {risk}")
        else:
            st.write("No significant risks identified")

    with col3:
        st.markdown("#### Catalysts")
        if report.key_catalysts:
            for catalyst in report.key_catalysts:
                st.info(f"→ {catalyst}")
        else:
            st.write("No catalysts identified")


def display_investment_thesis(report: AnalysisReport):
    """Display investment thesis section."""
    if hasattr(report, 'investment_thesis') and report.investment_thesis:
        st.subheader("📝 Investment Thesis")
        st.write(report.investment_thesis)

    # Position sizing suggestion
    if hasattr(report, 'position_sizing') and report.position_sizing:
        st.subheader("💰 Position Sizing Suggestion")
        ps = report.position_sizing
        col1, col2, col3 = st.columns(3)
        with col1:
            if ps.get('suggested_allocation'):
                st.metric("Suggested Allocation", f"{ps['suggested_allocation']*100:.1f}%")
        with col2:
            if ps.get('max_position'):
                st.metric("Max Position", f"{ps['max_position']*100:.1f}%")
        with col3:
            if ps.get('risk_budget'):
                st.metric("Risk Budget", f"{ps['risk_budget']*100:.1f}%")


def display_executive_summary(report: AnalysisReport):
    """Display executive summary."""
    if report.executive_summary:
        st.subheader("📋 Executive Summary")
        st.write(report.executive_summary)


async def run_analysis(symbol: str, agents: list[str], mode: str, analysis_mode: str = "full") -> AnalysisReport:
    """Run the analysis using the master agent."""
    master = MasterAgent(execution_mode=mode, analysis_mode=analysis_mode)

    agents_to_run = agents if agents else None
    report = await master.analyze(symbol, agents_to_run=agents_to_run)

    return report


async def run_portfolio_optimization(symbols: list[str], stock_analyses: dict) -> dict:
    """Run portfolio optimization."""
    optimizer = PortfolioOptimizerAgent()
    result = await optimizer.analyze(symbols, stock_analyses=stock_analyses)
    return result


def display_portfolio_builder_page():
    """Display the portfolio builder page with current holdings input."""
    st.markdown('<h1 class="main-header">📊 Portfolio Builder</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Sharpe Ratio Optimization with Buy/Sell Recommendations</p>',
                unsafe_allow_html=True)

    # Initialize session state for holdings
    if 'holdings_data' not in st.session_state:
        st.session_state.holdings_data = []

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Risk-free rate
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=4.5,
            step=0.1,
            help="Current treasury rate for Sharpe calculation"
        ) / 100

        # Constraints
        st.subheader("Constraints")
        min_weight = st.slider("Min Weight per Stock (%)", 0, 20, 2) / 100
        max_weight = st.slider("Max Weight per Stock (%)", 10, 50, 30) / 100

        st.markdown("---")

        # Additional stocks to consider
        st.subheader("Additional Stocks")
        additional_symbols = st.text_area(
            "Consider These Stocks (optional)",
            value="",
            height=100,
            help="Enter additional stock symbols to consider for buying"
        )
        additional_list = [s.strip().upper() for s in additional_symbols.split('\n') if s.strip()]

        st.markdown("---")
        optimize_button = st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True)
        clear_button = st.button("🗑️ Clear Holdings", use_container_width=True)

        if clear_button:
            st.session_state.holdings_data = []
            if 'portfolio_result' in st.session_state:
                del st.session_state['portfolio_result']
            st.rerun()

    # Main content - Holdings Input
    st.subheader("📝 Enter Your Current Holdings")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Input form for adding holdings
        with st.form("add_holding_form", clear_on_submit=True):
            form_cols = st.columns([2, 2, 2, 1])

            with form_cols[0]:
                new_symbol = st.text_input("Symbol", placeholder="AAPL").upper()
            with form_cols[1]:
                new_shares = st.number_input("Shares", min_value=0.0, step=1.0, value=0.0)
            with form_cols[2]:
                new_cost = st.number_input("Cost Basis ($)", min_value=0.0, step=0.01, value=0.0)
            with form_cols[3]:
                st.write("")  # Spacer
                add_submitted = st.form_submit_button("Add")

            if add_submitted and new_symbol and new_shares > 0:
                st.session_state.holdings_data.append({
                    "symbol": new_symbol,
                    "shares": new_shares,
                    "cost_basis": new_cost if new_cost > 0 else 100.0
                })
                st.rerun()

    with col2:
        # Quick add presets
        st.markdown("**Quick Add Sample Portfolio:**")
        if st.button("Tech Portfolio", key="quick_tech"):
            st.session_state.holdings_data = [
                {"symbol": "AAPL", "shares": 50, "cost_basis": 150.0},
                {"symbol": "MSFT", "shares": 30, "cost_basis": 300.0},
                {"symbol": "GOOGL", "shares": 20, "cost_basis": 140.0},
                {"symbol": "NVDA", "shares": 25, "cost_basis": 400.0},
            ]
            st.rerun()
        if st.button("Diversified Portfolio", key="quick_div"):
            st.session_state.holdings_data = [
                {"symbol": "AAPL", "shares": 40, "cost_basis": 150.0},
                {"symbol": "JPM", "shares": 30, "cost_basis": 150.0},
                {"symbol": "JNJ", "shares": 35, "cost_basis": 160.0},
                {"symbol": "XOM", "shares": 50, "cost_basis": 100.0},
                {"symbol": "PG", "shares": 25, "cost_basis": 150.0},
            ]
            st.rerun()

    # Display current holdings
    if st.session_state.holdings_data:
        st.markdown("---")
        st.subheader("📋 Your Current Holdings")

        holdings_df = pd.DataFrame(st.session_state.holdings_data)
        holdings_df['Total Cost'] = holdings_df['shares'] * holdings_df['cost_basis']
        holdings_df.columns = ['Symbol', 'Shares', 'Cost Basis ($)', 'Total Cost ($)']

        # Display with delete buttons
        for idx, row in holdings_df.iterrows():
            cols = st.columns([2, 2, 2, 2, 1])
            cols[0].write(f"**{row['Symbol']}**")
            cols[1].write(f"{row['Shares']:.0f} shares")
            cols[2].write(f"${row['Cost Basis ($)']:.2f}")
            cols[3].write(f"${row['Total Cost ($)']:,.2f}")
            if cols[4].button("❌", key=f"del_{idx}"):
                st.session_state.holdings_data.pop(idx)
                st.rerun()

        total_invested = holdings_df['Total Cost ($)'].sum()
        st.markdown(f"**Total Invested:** ${total_invested:,.2f}")

    # Run optimization
    if optimize_button:
        holdings = st.session_state.holdings_data
        if not holdings and not additional_list:
            st.error("Please add at least one holding or additional stock to analyze.")
            return

        # Convert holdings to optimizer format
        holdings_dict = {}
        for h in holdings:
            holdings_dict[h['symbol']] = {
                'shares': h['shares'],
                'cost_basis': h['cost_basis']
            }

        with st.spinner("Optimizing portfolio..."):
            try:
                from src.agents.portfolio_optimizer_agent import PortfolioOptimizerAgent

                optimizer = PortfolioOptimizerAgent(risk_free_rate=risk_free_rate)
                result = asyncio.run(optimizer.optimize_portfolio(
                    holdings=holdings_dict,
                    candidate_symbols=additional_list,
                    max_position_pct=max_weight,
                    min_position_pct=min_weight,
                ))

                if result.get('error'):
                    st.error(f"Optimization error: {result['error']}")
                else:
                    st.session_state['portfolio_result'] = result
                    st.rerun()

            except Exception as e:
                st.error(f"Error optimizing portfolio: {str(e)}")
                return

    # Display results
    if 'portfolio_result' in st.session_state:
        display_portfolio_results(st.session_state['portfolio_result'])
    elif not st.session_state.holdings_data:
        st.info("👆 Add your current holdings above or use a sample portfolio, then click **Optimize Portfolio**")


def display_portfolio_results(result: dict):
    """Display portfolio optimization results with recommendations."""
    if result.get('status') != 'success':
        st.error(f"Optimization failed: {result.get('error', 'Unknown error')}")
        return

    st.markdown("---")

    # Current Portfolio Analysis
    current = result.get('current_portfolio', {})
    optimal_metrics = result.get('optimal_metrics', {})
    optimal_weights = result.get('optimal_allocation', {})
    recommendations = result.get('recommendations', {})
    stock_analysis = result.get('stock_analysis', [])

    # Metrics header
    st.subheader("🎯 Portfolio Optimization Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sharpe = optimal_metrics.get('sharpe_ratio', 0)
        st.metric("Optimal Sharpe Ratio", f"{sharpe:.2f}")

    with col2:
        ret = optimal_metrics.get('expected_return', 0)
        st.metric("Expected Return", f"{ret*100:.1f}%")

    with col3:
        vol = optimal_metrics.get('volatility', 0)
        st.metric("Portfolio Volatility", f"{vol*100:.1f}%")

    with col4:
        max_dd = optimal_metrics.get('max_drawdown', 0)
        st.metric("Max Drawdown", f"{max_dd*100:.1f}%")

    st.markdown("---")

    # Current vs Optimal allocation
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Current Holdings")
        positions = current.get('positions', [])
        if positions:
            pos_df = pd.DataFrame(positions)
            display_df = pos_df[['symbol', 'shares', 'current_price', 'market_value', 'weight', 'gain_loss_pct']].copy()
            display_df.columns = ['Symbol', 'Shares', 'Price', 'Value', 'Weight', 'Gain/Loss %']
            display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}" if x else "N/A")
            display_df['Value'] = display_df['Value'].apply(lambda x: f"${x:,.2f}")
            display_df['Weight'] = display_df['Weight'].apply(lambda x: f"{x*100:.1f}%")
            display_df['Gain/Loss %'] = display_df['Gain/Loss %'].apply(lambda x: f"{x:+.1f}%")
            st.dataframe(display_df, hide_index=True, use_container_width=True)

            total_val = current.get('total_value', 0)
            st.markdown(f"**Total Portfolio Value:** ${total_val:,.2f}")
        else:
            st.info("No current holdings")

    with col2:
        st.subheader("🎯 Optimal Allocation")
        if optimal_weights:
            opt_data = []
            for symbol, weight in sorted(optimal_weights.items(), key=lambda x: -x[1]):
                opt_data.append({
                    'Symbol': symbol,
                    'Optimal Weight': f"{weight*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(opt_data), hide_index=True, use_container_width=True)

            # Pie chart
            fig = px.pie(
                values=list(optimal_weights.values()),
                names=list(optimal_weights.keys()),
                title="Optimal Allocation"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Recommendations
    st.subheader("💡 Action Recommendations")

    rec_summary = recommendations.get('summary', '')
    if rec_summary:
        st.info(f"**Summary:** {rec_summary}")

    col1, col2 = st.columns(2)

    with col1:
        # Sell recommendations
        sell_recs = recommendations.get('sell', [])
        if sell_recs:
            st.markdown("### 🔴 SELL")
            for rec in sell_recs:
                st.error(f"""
                **{rec['symbol']}** - Sell all shares
                - Current Weight: {rec['current_weight']*100:.1f}%
                - Shares to Sell: {rec.get('shares_to_sell', 'All')}
                - Reason: {rec['reason']}
                """)

        # Reduce recommendations
        reduce_recs = [r for r in recommendations.get('rebalance', []) if r['action'] == 'REDUCE']
        if reduce_recs:
            st.markdown("### 🟠 REDUCE")
            for rec in reduce_recs:
                st.warning(f"""
                **{rec['symbol']}** - Reduce position
                - Current: {rec['current_weight']*100:.1f}% → Target: {rec['target_weight']*100:.1f}%
                - Shares to Sell: ~{rec.get('shares_to_sell', 0)}
                """)

    with col2:
        # Buy recommendations
        buy_recs = recommendations.get('buy', [])
        if buy_recs:
            st.markdown("### 🟢 BUY (New Positions)")
            for rec in buy_recs:
                st.success(f"""
                **{rec['symbol']}** - Open new position
                - Target Weight: {rec['target_weight']*100:.1f}%
                - Shares to Buy: ~{rec.get('shares_to_buy', 0)}
                - Est. Cost: ${rec.get('estimated_cost', 0):,.2f}
                """)

        # Increase recommendations
        increase_recs = [r for r in recommendations.get('rebalance', []) if r['action'] == 'INCREASE']
        if increase_recs:
            st.markdown("### 🟢 INCREASE")
            for rec in increase_recs:
                st.success(f"""
                **{rec['symbol']}** - Increase position
                - Current: {rec['current_weight']*100:.1f}% → Target: {rec['target_weight']*100:.1f}%
                - Shares to Buy: ~{rec.get('shares_to_buy', 0)}
                """)

    # New stock picks
    new_picks = recommendations.get('new_stock_picks', [])
    if new_picks:
        st.markdown("---")
        st.subheader("🌟 Suggested New Stocks to Consider")
        picks_df = pd.DataFrame(new_picks)
        picks_df = picks_df[['symbol', 'name', 'sharpe_ratio', 'rating', 'reason']]
        picks_df.columns = ['Symbol', 'Name', 'Sharpe', 'Rating', 'Reason']
        st.dataframe(picks_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    # Stock Analysis Details
    st.subheader("📈 Individual Stock Analysis")

    if stock_analysis:
        analysis_df = pd.DataFrame(stock_analysis)
        display_cols = ['symbol', 'name', 'sector', 'current_price', 'annual_return', 'volatility', 'sharpe_ratio', 'momentum', 'rating']
        analysis_df = analysis_df[[c for c in display_cols if c in analysis_df.columns]]

        # Format columns
        if 'current_price' in analysis_df.columns:
            analysis_df['current_price'] = analysis_df['current_price'].apply(lambda x: f"${x:.2f}" if x else "N/A")
        if 'annual_return' in analysis_df.columns:
            analysis_df['annual_return'] = analysis_df['annual_return'].apply(lambda x: f"{x*100:.1f}%")
        if 'volatility' in analysis_df.columns:
            analysis_df['volatility'] = analysis_df['volatility'].apply(lambda x: f"{x*100:.1f}%")

        analysis_df.columns = ['Symbol', 'Name', 'Sector', 'Price', 'Annual Return', 'Volatility', 'Sharpe', 'Momentum', 'Rating']
        st.dataframe(analysis_df, hide_index=True, use_container_width=True)

    # Risk metrics
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⚖️ Risk Metrics")
        metrics_data = {
            'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Daily VaR (95%)', 'Daily VaR (99%)', 'Max Drawdown'],
            'Value': [
                f"{optimal_metrics.get('sharpe_ratio', 0):.2f}",
                f"{optimal_metrics.get('sortino_ratio', 0):.2f}",
                f"{optimal_metrics.get('var_95_daily', 0)*100:.2f}%",
                f"{optimal_metrics.get('var_99_daily', 0)*100:.2f}%",
                f"{optimal_metrics.get('max_drawdown', 0)*100:.1f}%"
            ]
        }
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True, use_container_width=True)

    with col2:
        # Risk-Return scatter
        if stock_analysis:
            st.subheader("📊 Risk-Return Profile")
            scatter_data = []
            for stock in stock_analysis:
                scatter_data.append({
                    'Symbol': stock['symbol'],
                    'Return': stock.get('annual_return', 0) * 100,
                    'Volatility': stock.get('volatility', 0) * 100,
                    'Sharpe': stock.get('sharpe_ratio', 0)
                })

            scatter_df = pd.DataFrame(scatter_data)
            fig = px.scatter(
                scatter_df, x='Volatility', y='Return',
                text='Symbol', color='Sharpe',
                color_continuous_scale='RdYlGn',
                labels={'Volatility': 'Volatility (%)', 'Return': 'Expected Return (%)'}
            )
            fig.update_traces(textposition='top center')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Export
    st.markdown("---")
    st.subheader("📥 Export")

    report_json = json.dumps(result, indent=2, default=str)
    st.download_button(
        label="Download Portfolio Report (JSON)",
        data=report_json,
        file_name=f"portfolio_optimization_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json"
    )


def display_stock_analysis_page():
    """Display the stock analysis page."""
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Bridgewater-Style Multi-Agent Analysis System</p>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Stock symbol input
        symbol = st.text_input(
            "Stock Symbol",
            value="AAPL",
            placeholder="Enter ticker symbol (e.g., AAPL)",
            help="Enter a valid stock ticker symbol"
        ).upper()

        # Analysis mode
        st.subheader("Analysis Mode")
        analysis_mode = st.selectbox(
            "Select Analysis Depth",
            ["core", "ecosystem", "full"],
            index=2,
            format_func=lambda x: {
                "core": "Core (4 agents)",
                "ecosystem": "Ecosystem (9 agents)",
                "full": "Full Analysis (All agents)"
            }[x],
            help="Core: Fundamental, Technical, Sentiment, Risk | Ecosystem: +Supply Chain, Customer, Competitive, Macro, Monetary"
        )

        # Agent selection
        st.subheader("Select Agents")
        all_agents = st.checkbox("Run All Agents", value=True)

        selected_agents = []
        if not all_agents:
            st.markdown("**Core Agents**")
            col1, col2 = st.columns(2)
            with col1:
                if st.checkbox("Fundamental", value=True):
                    selected_agents.append("fundamental")
                if st.checkbox("Technical", value=True):
                    selected_agents.append("technical")
            with col2:
                if st.checkbox("Sentiment", value=True):
                    selected_agents.append("sentiment")
                if st.checkbox("Risk", value=True):
                    selected_agents.append("risk")

            if analysis_mode in ["ecosystem", "full"]:
                st.markdown("**Ecosystem Agents**")
                col3, col4 = st.columns(2)
                with col3:
                    if st.checkbox("Supply Chain", value=True):
                        selected_agents.append("supply_chain")
                    if st.checkbox("Customer", value=True):
                        selected_agents.append("customer")
                    if st.checkbox("Competitive", value=True):
                        selected_agents.append("competitive")
                with col4:
                    if st.checkbox("Macro", value=True):
                        selected_agents.append("macro")
                    if st.checkbox("Monetary", value=True):
                        selected_agents.append("monetary")

        # Execution mode
        st.subheader("Execution Mode")
        mode = st.radio(
            "Agent Execution",
            ["parallel", "sequential"],
            index=0,
            help="Parallel mode runs all agents simultaneously for faster results"
        )

        # Analyze button
        st.markdown("---")
        analyze_button = st.button("🚀 Analyze Stock", type="primary", use_container_width=True)

        # Info
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Core Agents:**
        - Fundamental: Financials & valuation
        - Technical: Price patterns & indicators
        - Sentiment: News & market sentiment
        - Risk: Volatility & risk factors

        **Ecosystem Agents:**
        - Supply Chain: Supplier analysis
        - Customer: Customer base health
        - Competitive: Market position
        - Macro: Economic environment
        - Monetary: Fed policy & rates
        """)

    # Main content
    if analyze_button:
        if not symbol:
            st.error("Please enter a stock symbol")
            return

        # Progress indicator
        with st.spinner(f"Analyzing {symbol}... This may take a moment."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Initializing agents...")
            progress_bar.progress(10)

            try:
                # Run analysis
                agents_to_run = selected_agents if not all_agents and selected_agents else None
                report = asyncio.run(run_analysis(symbol, agents_to_run, mode, analysis_mode))

                progress_bar.progress(100)
                status_text.text("Analysis complete!")

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Store report in session state
                st.session_state['report'] = report
                st.session_state['symbol'] = symbol

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"Error analyzing {symbol}: {str(e)}")
                return

    # Display results if available
    if 'report' in st.session_state:
        report = st.session_state['report']
        symbol = st.session_state['symbol']

        # Company header
        company_name = report.company_name or symbol
        st.markdown(f"## {company_name} ({symbol})")
        st.caption(f"Analysis completed at {report.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Execution time: {report.total_execution_time:.2f}s | "
                  f"Agents: {report.successful_agents}/{report.agents_executed}")

        # Price chart
        st.subheader("📊 Price Chart")
        chart_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        price_chart = create_price_chart(symbol, chart_period)
        if price_chart:
            st.plotly_chart(price_chart, use_container_width=True)

        # Divider
        st.markdown("---")

        # Stock overview
        display_stock_overview(report)

        st.markdown("---")

        # Overall analysis
        display_overall_analysis(report)

        st.markdown("---")

        # Key insights
        display_key_insights(report)

        st.markdown("---")

        # Investment thesis (if available)
        display_investment_thesis(report)

        st.markdown("---")

        # Detailed agent results
        display_agent_results(report)

        st.markdown("---")

        # Executive summary
        display_executive_summary(report)

        # Download option
        st.markdown("---")
        st.subheader("📥 Export")
        col1, col2 = st.columns(2)
        with col1:
            # JSON export
            report_json = json.dumps(report.model_dump(), indent=2, default=str)
            st.download_button(
                label="Download JSON Report",
                data=report_json,
                file_name=f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

    else:
        # Welcome message
        st.info("👈 Enter a stock symbol and click **Analyze Stock** to get started!")

        # Sample stocks
        st.markdown("### Popular Stocks to Analyze")
        sample_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "JPM"]
        cols = st.columns(4)
        for i, stock in enumerate(sample_stocks):
            with cols[i % 4]:
                if st.button(stock, key=f"sample_{stock}"):
                    st.session_state['selected_symbol'] = stock
                    st.rerun()


def display_watchlist_management(screener_results: list = None):
    """Display watchlist management interface."""
    wm = st.session_state.get('watchlist_manager') or get_watchlist_manager()
    st.session_state.watchlist_manager = wm

    st.subheader("⭐ Watchlist Manager")

    # Create new watchlist section
    with st.expander("➕ Create New Watchlist", expanded=False):
        col1, col2 = st.columns([2, 3])
        with col1:
            new_wl_name = st.text_input("Watchlist Name", key="new_watchlist_name")
        with col2:
            new_wl_desc = st.text_input("Description (optional)", key="new_watchlist_desc")

        if st.button("Create Watchlist", type="primary", key="create_watchlist_btn"):
            if new_wl_name:
                if wm.create_watchlist(new_wl_name, new_wl_desc):
                    st.success(f"Created watchlist: {new_wl_name}")
                    st.rerun()
                else:
                    st.error(f"Watchlist '{new_wl_name}' already exists")
            else:
                st.warning("Please enter a watchlist name")

    st.markdown("---")

    # Display existing watchlists
    watchlists = wm.get_all_watchlists()

    if not watchlists:
        st.info("No watchlists yet. Create one above to get started!")
        return

    # Watchlist selector
    wl_names = wm.get_watchlist_names()
    selected_wl_name = st.selectbox(
        "Select Watchlist",
        wl_names,
        key="selected_watchlist"
    )

    if selected_wl_name:
        wl = wm.get_watchlist(selected_wl_name)
        if wl:
            # Watchlist info
            col1, col2, col3 = st.columns([3, 2, 1])
            with col1:
                st.markdown(f"**{wl.name}**")
                if wl.description:
                    st.caption(wl.description)
            with col2:
                st.caption(f"Created: {wl.created_date}")
            with col3:
                if st.button("🗑️ Delete", key=f"delete_wl_{wl.name}"):
                    wm.delete_watchlist(wl.name)
                    st.success(f"Deleted watchlist: {wl.name}")
                    st.rerun()

            st.markdown("---")

            # Add stock from screener results
            if screener_results:
                with st.expander("➕ Add Stock from Screener Results", expanded=False):
                    available_symbols = [s.symbol for s in screener_results]
                    selected_to_add = st.multiselect(
                        "Select stocks to add",
                        available_symbols,
                        key=f"add_stocks_{wl.name}"
                    )
                    if st.button("Add Selected Stocks", key=f"add_btn_{wl.name}"):
                        added_count = 0
                        for sym in selected_to_add:
                            stock_data = next((s for s in screener_results if s.symbol == sym), None)
                            if stock_data:
                                if wm.add_stock_to_watchlist(
                                    wl.name, sym,
                                    name=stock_data.name,
                                    price=stock_data.current_price,
                                    category=stock_data.category.value
                                ):
                                    added_count += 1
                        if added_count > 0:
                            st.success(f"Added {added_count} stocks to {wl.name}")
                            st.rerun()
                        else:
                            st.warning("Stocks already in watchlist or error occurred")

            # Add stock manually
            with st.expander("➕ Add Stock Manually", expanded=False):
                manual_col1, manual_col2 = st.columns(2)
                with manual_col1:
                    manual_symbol = st.text_input("Stock Symbol", key=f"manual_sym_{wl.name}").upper()
                with manual_col2:
                    manual_notes = st.text_input("Notes (optional)", key=f"manual_notes_{wl.name}")

                if st.button("Add Stock", key=f"manual_add_{wl.name}"):
                    if manual_symbol:
                        # Try to get stock info
                        try:
                            info = get_ticker_info(manual_symbol)
                            name = info.get('shortName', '') if info else ''
                            price = info.get('currentPrice') or info.get('regularMarketPrice') if info else None
                        except:
                            name = ''
                            price = None

                        if wm.add_stock_to_watchlist(
                            wl.name, manual_symbol,
                            name=name,
                            price=price,
                            notes=manual_notes
                        ):
                            st.success(f"Added {manual_symbol} to {wl.name}")
                            st.rerun()
                        else:
                            st.warning(f"{manual_symbol} already in watchlist")
                    else:
                        st.warning("Please enter a stock symbol")

            # Display stocks in watchlist
            if wl.stocks:
                st.markdown("##### 📊 Stocks in Watchlist")

                stock_data = []
                for stock in wl.stocks:
                    row = {
                        'Symbol': stock.symbol,
                        'Name': stock.name[:30] + '...' if len(stock.name) > 30 else stock.name,
                        'Added Date': stock.added_date,
                        'Added Price': f"${stock.added_price:.2f}" if stock.added_price else "N/A",
                        'Category': stock.category.title() if stock.category else "—",
                        'Notes': stock.notes[:20] + '...' if len(stock.notes) > 20 else stock.notes,
                    }
                    stock_data.append(row)

                stock_df = pd.DataFrame(stock_data)
                st.dataframe(stock_df, hide_index=True, use_container_width=True)

                # Remove stock section
                st.markdown("##### Remove Stocks")
                stocks_to_remove = st.multiselect(
                    "Select stocks to remove",
                    [s.symbol for s in wl.stocks],
                    key=f"remove_stocks_{wl.name}"
                )
                if st.button("Remove Selected", type="secondary", key=f"remove_btn_{wl.name}"):
                    for sym in stocks_to_remove:
                        wm.remove_stock_from_watchlist(wl.name, sym)
                    if stocks_to_remove:
                        st.success(f"Removed {len(stocks_to_remove)} stocks")
                        st.rerun()
            else:
                st.info("No stocks in this watchlist yet. Add some using the options above!")


def display_stock_screener_page():
    """Display the stock screener page for finding value and growth stocks."""
    st.markdown('<h1 class="main-header">🔍 Stock Screener</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Scan NYSE & NASDAQ for Value and Growth Opportunities</p>',
                unsafe_allow_html=True)

    # Initialize session state
    if 'screener_results' not in st.session_state:
        st.session_state.screener_results = None
    if 'screening_complete' not in st.session_state:
        st.session_state.screening_complete = False
    if 'watchlist_manager' not in st.session_state:
        st.session_state.watchlist_manager = get_watchlist_manager()
    if 'show_watchlist_tab' not in st.session_state:
        st.session_state.show_watchlist_tab = False

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Screening Configuration")

        # Stock Universe Selection
        st.subheader("📊 Stock Universe")
        universe = st.selectbox(
            "Select Universe",
            ["S&P 500", "NASDAQ 100", "Dow Jones 30", "Dividend Aristocrats", "All Major Stocks", "Custom Sector"],
            index=0
        )

        # If custom sector selected, show sector dropdown
        selected_sector = None
        if universe == "Custom Sector":
            selected_sector = st.selectbox(
                "Select Sector",
                get_all_sectors()
            )

        st.markdown("---")

        # Value Criteria
        st.subheader("💰 Value Criteria")
        max_pe = st.slider("Max P/E Ratio", 5, 50, 20)
        max_pb = st.slider("Max P/B Ratio", 0.5, 10.0, 3.0, step=0.5)
        max_peg = st.slider("Max PEG Ratio", 0.5, 3.0, 1.5, step=0.1)

        st.markdown("---")

        # Growth Criteria
        st.subheader("📈 Growth Criteria")
        min_revenue_growth = st.slider("Min Revenue Growth (%)", 0, 50, 10) / 100
        min_earnings_growth = st.slider("Min Earnings Growth (%)", 0, 50, 10) / 100

        st.markdown("---")

        # Quality Criteria
        st.subheader("✨ Quality Criteria")
        min_profit_margin = st.slider("Min Profit Margin (%)", 0, 30, 5) / 100
        min_roe = st.slider("Min ROE (%)", 0, 40, 10) / 100

        st.markdown("---")

        # Market Cap Filter
        st.subheader("📏 Size Filter")
        min_market_cap_b = st.slider("Min Market Cap ($B)", 0.1, 100.0, 1.0, step=0.1)
        min_market_cap = min_market_cap_b * 1e9

        st.markdown("---")

        # Run button
        run_screener = st.button("🚀 Run Screener", type="primary", use_container_width=True)

    # Main content area
    if run_screener:
        # Get symbols based on universe selection
        if universe == "S&P 500":
            symbols = get_sp500_symbols()
        elif universe == "NASDAQ 100":
            symbols = get_nasdaq100_symbols()
        elif universe == "Dow Jones 30":
            symbols = get_dow30_symbols()
        elif universe == "Dividend Aristocrats":
            symbols = get_dividend_aristocrats()
        elif universe == "Custom Sector" and selected_sector:
            symbols = get_sector_stocks(selected_sector)
        else:
            symbols = get_all_major_stocks()

        # Create criteria
        criteria = ScreenerCriteria(
            max_pe_ratio=max_pe,
            max_pb_ratio=max_pb,
            max_peg_ratio=max_peg,
            min_revenue_growth=min_revenue_growth,
            min_earnings_growth=min_earnings_growth,
            min_profit_margin=min_profit_margin,
            min_roe=min_roe,
            min_market_cap=min_market_cap,
        )

        # Run screening
        st.info(f"Screening {len(symbols)} stocks from {universe}...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(current, total, symbol):
            progress_bar.progress(current / total)
            status_text.text(f"Analyzing {symbol}... ({current}/{total})")

        screener = StockScreener()

        # Run async screening
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                screener.screen_stocks(symbols, criteria, update_progress)
            )
        finally:
            loop.close()

        progress_bar.progress(1.0)
        status_text.text(f"Screening complete! Analyzed {len(results)} stocks.")

        st.session_state.screener_results = results
        st.session_state.screener = screener
        st.session_state.screening_complete = True

    # Display results
    if st.session_state.screening_complete and st.session_state.screener_results:
        results = st.session_state.screener_results
        screener = st.session_state.screener

        st.markdown("---")

        # Summary statistics
        summary = screener.get_screening_summary()

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Stocks Screened", summary.get('total_screened', 0))
        with col2:
            st.metric("Value Stocks", summary.get('value_stocks', 0))
        with col3:
            st.metric("Growth Stocks", summary.get('growth_stocks', 0))
        with col4:
            st.metric("Strong Buy", summary.get('strong_buy', 0))
        with col5:
            st.metric("Buy", summary.get('buy', 0))

        st.markdown("---")

        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🏆 Top Recommendations",
            "💰 Value Stocks",
            "📈 Growth Stocks",
            "💵 Dividend Stocks",
            "📋 All Results",
            "⭐ My Watchlists"
        ])

        with tab1:
            st.subheader("Top Stock Recommendations")
            top_stocks = screener.get_top_recommendations(n=20)
            if top_stocks:
                display_screener_results_table(top_stocks, tab_name="top_recommendations")
            else:
                st.info("No stocks met the criteria for a buy recommendation.")

        with tab2:
            st.subheader("Value Stocks")
            value_stocks = screener.get_value_stocks(min_score=50)
            value_stocks = sorted(value_stocks, key=lambda x: x.value_score, reverse=True)[:30]
            if value_stocks:
                display_screener_results_table(value_stocks, show_value_metrics=True, tab_name="value_stocks")
            else:
                st.info("No value stocks found with the current criteria.")

        with tab3:
            st.subheader("Growth Stocks")
            growth_stocks = screener.get_growth_stocks(min_score=50)
            growth_stocks = sorted(growth_stocks, key=lambda x: x.growth_score, reverse=True)[:30]
            if growth_stocks:
                display_screener_results_table(growth_stocks, show_growth_metrics=True, tab_name="growth_stocks")
            else:
                st.info("No growth stocks found with the current criteria.")

        with tab4:
            st.subheader("Dividend Stocks")
            dividend_stocks = screener.get_dividend_stocks(min_yield=0.02)
            dividend_stocks = sorted(dividend_stocks, key=lambda x: x.dividend_yield or 0, reverse=True)[:30]
            if dividend_stocks:
                display_screener_results_table(dividend_stocks, show_dividend_metrics=True, tab_name="dividend_stocks")
            else:
                st.info("No dividend stocks found with the current criteria.")

        with tab5:
            st.subheader("All Screened Stocks")
            all_results = sorted(results, key=lambda x: x.overall_score, reverse=True)
            display_screener_results_table(all_results, tab_name="all_results")

        with tab6:
            display_watchlist_management(results)

        # Sector breakdown chart
        st.markdown("---")
        st.subheader("📊 Sector Analysis")

        col1, col2 = st.columns(2)

        with col1:
            # Sector distribution
            sector_counts = {}
            for stock in results:
                sector = stock.sector if stock.sector != "Unknown" else "Other"
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            if sector_counts:
                sector_df = pd.DataFrame({
                    'Sector': list(sector_counts.keys()),
                    'Count': list(sector_counts.values())
                })
                fig = px.pie(sector_df, values='Count', names='Sector', title='Stocks by Sector')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Recommendation distribution
            rec_counts = {}
            for stock in results:
                rec = stock.recommendation.value
                rec_counts[rec] = rec_counts.get(rec, 0) + 1

            if rec_counts:
                rec_df = pd.DataFrame({
                    'Recommendation': list(rec_counts.keys()),
                    'Count': list(rec_counts.values())
                })
                color_map = {
                    'Strong Buy': 'green',
                    'Buy': 'lightgreen',
                    'Hold': 'yellow',
                    'Sell': 'orange',
                    'Strong Sell': 'red'
                }
                fig = px.bar(rec_df, x='Recommendation', y='Count', title='Recommendation Distribution',
                           color='Recommendation', color_discrete_map=color_map)
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

        # Export results
        st.markdown("---")
        st.subheader("📥 Export Results")

        export_data = []
        for stock in results:
            export_data.append({
                'Symbol': stock.symbol,
                'Name': stock.name,
                'Sector': stock.sector,
                'Price': stock.current_price,
                'P/E': stock.pe_ratio,
                'P/B': stock.pb_ratio,
                'Revenue Growth': stock.revenue_growth,
                'Earnings Growth': stock.earnings_growth,
                'Profit Margin': stock.profit_margin,
                'ROE': stock.roe,
                'Dividend Yield': stock.dividend_yield,
                'Value Score': stock.value_score,
                'Growth Score': stock.growth_score,
                'Quality Score': stock.quality_score,
                'Overall Score': stock.overall_score,
                'Category': stock.category.value,
                'Recommendation': stock.recommendation.value,
            })

        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False)

        st.download_button(
            label="Download Results (CSV)",
            data=csv,
            file_name=f"stock_screener_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    else:
        # Show instructions and watchlist management when no results
        instructions_tab, watchlist_tab = st.tabs(["📖 Getting Started", "⭐ My Watchlists"])

        with instructions_tab:
            st.markdown("""
            ### How to Use the Stock Screener

            1. **Select a Stock Universe** - Choose from S&P 500, NASDAQ 100, Dow 30, or scan all major stocks
            2. **Set Screening Criteria** - Adjust value, growth, and quality filters in the sidebar
            3. **Run the Screener** - Click "Run Screener" to analyze stocks
            4. **Review Results** - Explore recommendations by category (Value, Growth, Dividend)
            5. **Add to Watchlist** - Save interesting stocks to your watchlists
            6. **Export** - Download results as CSV for further analysis

            ---

            ### Stock Categories

            | Category | Description |
            |----------|-------------|
            | **Value** | Low P/E, P/B ratios - undervalued relative to fundamentals |
            | **Growth** | High revenue and earnings growth rates |
            | **Dividend** | Solid dividend yield with sustainable payout |
            | **Quality** | Strong profitability and returns |
            | **Momentum** | Positive price trends |

            ---

            ### Scoring System

            Each stock receives scores (0-100) for:
            - **Value Score** - Based on P/E, P/B, P/S, PEG ratios
            - **Growth Score** - Based on revenue, earnings, EPS growth
            - **Quality Score** - Based on margins, ROE, ROA
            - **Momentum Score** - Based on 1M, 3M, YTD, 1Y performance

            The **Overall Score** is a weighted average used for recommendations.
            """)

        with watchlist_tab:
            display_watchlist_management()


def display_screener_results_table(
    stocks: list[ScreenedStock],
    show_value_metrics: bool = False,
    show_growth_metrics: bool = False,
    show_dividend_metrics: bool = False,
    tab_name: str = "default"
):
    """Display screener results in a formatted table."""
    if not stocks:
        st.info("No stocks to display.")
        return

    data = []
    for stock in stocks:
        row = {
            'Symbol': stock.symbol,
            'Name': stock.name[:25] + '...' if len(stock.name) > 25 else stock.name,
            'Sector': stock.sector[:15] if stock.sector else 'N/A',
            'Price': f"${stock.current_price:.2f}" if stock.current_price else "N/A",
        }

        if show_value_metrics:
            row['P/E'] = f"{stock.pe_ratio:.1f}" if stock.pe_ratio else "N/A"
            row['P/B'] = f"{stock.pb_ratio:.1f}" if stock.pb_ratio else "N/A"
            row['PEG'] = f"{stock.peg_ratio:.1f}" if stock.peg_ratio else "N/A"
            row['Value Score'] = f"{stock.value_score:.0f}"
        elif show_growth_metrics:
            row['Rev Growth'] = f"{stock.revenue_growth*100:.1f}%" if stock.revenue_growth else "N/A"
            row['Earn Growth'] = f"{stock.earnings_growth*100:.1f}%" if stock.earnings_growth else "N/A"
            row['Growth Score'] = f"{stock.growth_score:.0f}"
        elif show_dividend_metrics:
            row['Div Yield'] = f"{stock.dividend_yield*100:.2f}%" if stock.dividend_yield else "N/A"
            row['Payout'] = f"{stock.payout_ratio*100:.0f}%" if stock.payout_ratio else "N/A"
            row['Quality Score'] = f"{stock.quality_score:.0f}"
        else:
            row['Overall'] = f"{stock.overall_score:.0f}"
            row['Value'] = f"{stock.value_score:.0f}"
            row['Growth'] = f"{stock.growth_score:.0f}"
            row['Quality'] = f"{stock.quality_score:.0f}"

        row['Category'] = stock.category.value.title()
        row['Recommendation'] = stock.recommendation.value

        data.append(row)

    df = pd.DataFrame(data)

    # Style the recommendation column
    def style_recommendation(val):
        colors = {
            'Strong Buy': 'background-color: #28a745; color: white',
            'Buy': 'background-color: #90EE90',
            'Hold': 'background-color: #FFD700',
            'Sell': 'background-color: #FFA500',
            'Strong Sell': 'background-color: #dc3545; color: white'
        }
        return colors.get(val, '')

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        height=min(len(df) * 35 + 38, 600)
    )

    # Show detailed info for selected stock
    st.markdown("##### 📝 Stock Details")
    selected_symbol = st.selectbox(
        "Select a stock for detailed analysis",
        [s.symbol for s in stocks],
        key=f"detail_select_{tab_name}"
    )

    if selected_symbol:
        selected_stock = next((s for s in stocks if s.symbol == selected_symbol), None)
        if selected_stock:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**{selected_stock.name}**")
                st.write(f"Sector: {selected_stock.sector}")
                st.write(f"Industry: {selected_stock.industry}")
                if selected_stock.market_cap:
                    cap_str = f"${selected_stock.market_cap/1e9:.1f}B" if selected_stock.market_cap >= 1e9 else f"${selected_stock.market_cap/1e6:.0f}M"
                    st.write(f"Market Cap: {cap_str}")

            with col2:
                st.markdown("**Valuation**")
                st.write(f"P/E: {selected_stock.pe_ratio:.1f}" if selected_stock.pe_ratio else "P/E: N/A")
                st.write(f"P/B: {selected_stock.pb_ratio:.1f}" if selected_stock.pb_ratio else "P/B: N/A")
                st.write(f"P/S: {selected_stock.ps_ratio:.1f}" if selected_stock.ps_ratio else "P/S: N/A")

            with col3:
                st.markdown("**Performance**")
                if selected_stock.performance_1m:
                    color = "green" if selected_stock.performance_1m > 0 else "red"
                    st.markdown(f"1M: :{color}[{selected_stock.performance_1m*100:+.1f}%]")
                if selected_stock.performance_3m:
                    color = "green" if selected_stock.performance_3m > 0 else "red"
                    st.markdown(f"3M: :{color}[{selected_stock.performance_3m*100:+.1f}%]")
                if selected_stock.performance_ytd:
                    color = "green" if selected_stock.performance_ytd > 0 else "red"
                    st.markdown(f"YTD: :{color}[{selected_stock.performance_ytd*100:+.1f}%]")

            # Analysis summary
            if selected_stock.analysis_summary:
                st.info(f"**Analysis:** {selected_stock.analysis_summary}")

            # Add to Watchlist button
            wm = st.session_state.get('watchlist_manager') or get_watchlist_manager()
            watchlist_names = wm.get_watchlist_names()
            if watchlist_names:
                add_col1, add_col2 = st.columns([2, 1])
                with add_col1:
                    target_watchlist = st.selectbox(
                        "Add to Watchlist",
                        watchlist_names,
                        key=f"add_wl_select_{tab_name}"
                    )
                with add_col2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("⭐ Add", key=f"add_wl_btn_{tab_name}"):
                        if wm.add_stock_to_watchlist(
                            target_watchlist,
                            selected_stock.symbol,
                            name=selected_stock.name,
                            price=selected_stock.current_price,
                            category=selected_stock.category.value
                        ):
                            st.success(f"Added {selected_stock.symbol} to {target_watchlist}")
                        else:
                            st.warning(f"{selected_stock.symbol} already in {target_watchlist}")
            else:
                st.caption("💡 Create a watchlist in the 'My Watchlists' tab to save stocks")


def display_asset_manager_tools_page():
    """Display the Asset Manager Tools page with advanced analytics."""
    st.markdown('<h1 class="main-header">🛠️ Asset Manager Tools</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Professional-Grade Analytics for Portfolio Management</p>',
                unsafe_allow_html=True)

    # Tool selection tabs
    tool_tab1, tool_tab2, tool_tab3, tool_tab4, tool_tab5 = st.tabs([
        "📊 Technical Analysis",
        "🔄 Peer Comparison",
        "📅 Events Calendar",
        "🏛️ Institutional Activity",
        "📈 Risk Analytics"
    ])

    with tool_tab1:
        display_technical_analysis_tool()

    with tool_tab2:
        display_peer_comparison_tool()

    with tool_tab3:
        display_events_calendar_tool()

    with tool_tab4:
        display_institutional_activity_tool()

    with tool_tab5:
        display_risk_analytics_tool()


def display_technical_analysis_tool():
    """Display technical analysis tool."""
    st.subheader("📊 Technical Analysis")
    st.markdown("Comprehensive technical indicators and trading signals")

    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", key="ta_symbol").upper()
    with col2:
        period = st.selectbox("Analysis Period", ["6mo", "1y", "2y"], index=1, key="ta_period")

    if st.button("Run Technical Analysis", type="primary", key="ta_run"):
        if symbol:
            with st.spinner(f"Analyzing {symbol}..."):
                analyzer = get_technical_analyzer()
                result = analyzer.analyze(symbol, period=period)

                if result:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"${result.current_price:.2f}")
                    with col2:
                        trend_color = "green" if "BULLISH" in result.trend.value.upper() else "red" if "BEARISH" in result.trend.value.upper() else "gray"
                        st.metric("Trend", result.trend.value)
                    with col3:
                        signal_color = "green" if "BUY" in result.overall_signal.value.upper() else "red" if "SELL" in result.overall_signal.value.upper() else "gray"
                        st.metric("Signal", result.overall_signal.value)
                    with col4:
                        st.metric("Signal Strength", f"{result.signal_strength:.0f}/100")

                    st.markdown("---")

                    # Indicator details in columns
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("##### 📉 Moving Averages")
                        ma = result.moving_averages
                        ma_data = {
                            "Indicator": ["SMA 20", "SMA 50", "SMA 200", "EMA 12", "EMA 26"],
                            "Value": [f"${ma.sma_20:.2f}", f"${ma.sma_50:.2f}", f"${ma.sma_200:.2f}" if ma.sma_200 > 0 else "N/A", f"${ma.ema_12:.2f}", f"${ma.ema_26:.2f}"],
                            "vs Price": [f"{ma.price_vs_sma_20:+.1f}%", f"{ma.price_vs_sma_50:+.1f}%", f"{ma.price_vs_sma_200:+.1f}%" if ma.sma_200 > 0 else "N/A", "-", "-"]
                        }
                        st.dataframe(pd.DataFrame(ma_data), hide_index=True)

                        if ma.golden_cross:
                            st.success("✅ Golden Cross (Bullish)")
                        elif ma.death_cross:
                            st.error("❌ Death Cross (Bearish)")

                        st.markdown("##### 📊 RSI & Stochastic")
                        rsi = result.rsi
                        stoch = result.stochastic
                        osc_data = {
                            "Indicator": ["RSI (14)", "RSI (7)", "Stochastic %K", "Stochastic %D"],
                            "Value": [f"{rsi.rsi_14:.1f}", f"{rsi.rsi_7:.1f}", f"{stoch.k_line:.1f}", f"{stoch.d_line:.1f}"],
                            "Status": [
                                "Overbought" if rsi.is_overbought else "Oversold" if rsi.is_oversold else "Neutral",
                                "-",
                                "Overbought" if stoch.is_overbought else "Oversold" if stoch.is_oversold else "Neutral",
                                "-"
                            ]
                        }
                        st.dataframe(pd.DataFrame(osc_data), hide_index=True)

                    with col2:
                        st.markdown("##### 📈 MACD")
                        macd = result.macd
                        macd_data = {
                            "Component": ["MACD Line", "Signal Line", "Histogram"],
                            "Value": [f"{macd.macd_line:.3f}", f"{macd.signal_line:.3f}", f"{macd.histogram:.3f}"],
                            "Status": [
                                "Bullish" if macd.is_bullish else "Bearish",
                                macd.crossover.title() if macd.crossover else "-",
                                macd.histogram_trend.title()
                            ]
                        }
                        st.dataframe(pd.DataFrame(macd_data), hide_index=True)

                        st.markdown("##### 🎯 Bollinger Bands")
                        bb = result.bollinger
                        bb_data = {
                            "Band": ["Upper", "Middle", "Lower"],
                            "Value": [f"${bb.upper_band:.2f}", f"${bb.middle_band:.2f}", f"${bb.lower_band:.2f}"]
                        }
                        st.dataframe(pd.DataFrame(bb_data), hide_index=True)
                        st.caption(f"Bandwidth: {bb.bandwidth:.1f}% | %B: {bb.percent_b:.2f}")
                        if bb.squeeze:
                            st.warning("⚠️ Volatility Squeeze Detected")

                        st.markdown("##### 🎚️ Support & Resistance")
                        sr = result.support_resistance
                        sr_data = {
                            "Level": ["Resistance 2", "Resistance 1", "Pivot", "Support 1", "Support 2"],
                            "Price": [f"${sr.resistance_2:.2f}", f"${sr.resistance_1:.2f}", f"${sr.pivot_point:.2f}", f"${sr.support_1:.2f}", f"${sr.support_2:.2f}"]
                        }
                        st.dataframe(pd.DataFrame(sr_data), hide_index=True)

                    # Trading Signals
                    st.markdown("---")
                    st.markdown("##### 📋 Trading Signals")
                    if result.signals:
                        signals_data = []
                        for sig in result.signals:
                            signals_data.append({
                                "Indicator": sig.indicator,
                                "Signal": sig.signal.value,
                                "Description": sig.description
                            })
                        st.dataframe(pd.DataFrame(signals_data), hide_index=True, use_container_width=True)
                    else:
                        st.info("No significant signals detected")

                else:
                    st.error(f"Could not analyze {symbol}. Please check the symbol.")


def display_peer_comparison_tool():
    """Display peer comparison tool."""
    st.subheader("🔄 Peer Comparison")
    st.markdown("Compare a stock against its sector/industry peers")

    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", key="peer_symbol").upper()
    with col2:
        max_peers = st.slider("Max Peers", 5, 20, 10, key="peer_count")

    if st.button("Compare to Peers", type="primary", key="peer_run"):
        if symbol:
            with st.spinner(f"Finding peers and analyzing {symbol}..."):
                analyzer = get_peer_analyzer()
                comparison = analyzer.compare(symbol, max_peers=max_peers)

                if comparison.peers:
                    # Summary
                    st.markdown(f"### {comparison.target_name}")
                    st.caption(f"Sector: {comparison.sector} | Industry: {comparison.industry}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Ranking", comparison.overall_ranking)
                    with col2:
                        st.metric("Peers Analyzed", len(comparison.peers))
                    with col3:
                        target = next((p for p in comparison.peers if p.symbol == symbol), None)
                        if target:
                            st.metric("Overall Rank", f"#{target.overall_rank} of {len(comparison.peers)}")

                    st.markdown("---")

                    # Strengths and Weaknesses
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### 💪 Strengths")
                        if comparison.strengths:
                            for s in comparison.strengths:
                                st.success(s)
                        else:
                            st.info("No standout strengths identified")
                    with col2:
                        st.markdown("##### ⚠️ Weaknesses")
                        if comparison.weaknesses:
                            for w in comparison.weaknesses:
                                st.warning(w)
                        else:
                            st.info("No significant weaknesses identified")

                    st.markdown("---")

                    # Peer Comparison Table
                    st.markdown("##### 📊 Peer Comparison Table")
                    peer_data = []
                    for peer in comparison.peers:
                        peer_data.append({
                            "Symbol": peer.symbol,
                            "Name": peer.name[:25] + "..." if len(peer.name) > 25 else peer.name,
                            "Market Cap": f"${peer.market_cap/1e9:.1f}B" if peer.market_cap >= 1e9 else f"${peer.market_cap/1e6:.0f}M",
                            "P/E": f"{peer.pe_ratio:.1f}" if peer.pe_ratio > 0 else "N/A",
                            "P/B": f"{peer.pb_ratio:.1f}" if peer.pb_ratio > 0 else "N/A",
                            "Rev Growth": f"{peer.revenue_growth*100:.1f}%" if peer.revenue_growth else "N/A",
                            "Profit Margin": f"{peer.profit_margin*100:.1f}%" if peer.profit_margin else "N/A",
                            "ROE": f"{peer.roe*100:.1f}%" if peer.roe else "N/A",
                            "1Y Return": f"{peer.return_1y*100:+.1f}%" if peer.return_1y else "N/A",
                            "Overall Rank": f"#{peer.overall_rank}"
                        })

                    peer_df = pd.DataFrame(peer_data)
                    st.dataframe(peer_df, hide_index=True, use_container_width=True)

                    # Valuation Comparison Chart
                    st.markdown("---")
                    st.markdown("##### 📈 Valuation Comparison")

                    chart_data = []
                    for peer in comparison.peers[:10]:
                        if peer.pe_ratio > 0 and peer.pe_ratio < 100:
                            chart_data.append({
                                "Symbol": peer.symbol,
                                "P/E Ratio": peer.pe_ratio,
                                "Is Target": peer.symbol == symbol
                            })

                    if chart_data:
                        chart_df = pd.DataFrame(chart_data)
                        fig = px.bar(chart_df, x="Symbol", y="P/E Ratio",
                                   color="Is Target",
                                   color_discrete_map={True: "green", False: "steelblue"},
                                   title="P/E Ratio Comparison")
                        fig.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.error(f"Could not find peers for {symbol}")


def display_events_calendar_tool():
    """Display events calendar tool."""
    st.subheader("📅 Events Calendar")
    st.markdown("Track earnings, dividends, and analyst actions for your watchlist")

    # Get watchlist symbols
    wm = get_watchlist_manager()
    watchlist_names = wm.get_watchlist_names()

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if watchlist_names:
            source = st.selectbox("Symbol Source", ["Enter Manually", "From Watchlist"], key="event_source")
        else:
            source = "Enter Manually"
            st.caption("Create a watchlist to use saved symbols")

    if source == "Enter Manually":
        with col2:
            symbols_input = st.text_input("Symbols (comma-separated)", value="AAPL,MSFT,GOOGL", key="event_symbols")
            symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    else:
        with col2:
            selected_wl = st.selectbox("Select Watchlist", watchlist_names, key="event_watchlist")
            wl = wm.get_watchlist(selected_wl)
            symbols = [s.symbol for s in wl.stocks] if wl else []

    with col3:
        days_ahead = st.slider("Days Ahead", 7, 90, 30, key="event_days")

    if st.button("Get Events", type="primary", key="event_run"):
        if symbols:
            calendar = get_events_calendar()

            with st.spinner("Fetching events..."):
                summary = calendar.get_calendar_summary(symbols, days_ahead=days_ahead)

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Events", summary.total_events)
                with col2:
                    st.metric("Earnings This Week", summary.earnings_this_week)
                with col3:
                    st.metric("Dividends This Week", summary.dividends_this_week)
                with col4:
                    st.metric("Symbols Tracked", len(symbols))

                st.markdown("---")

                # Earnings Calendar
                st.markdown("##### 📊 Upcoming Earnings")
                if summary.earnings_events:
                    earnings_data = []
                    for e in summary.earnings_events[:20]:
                        earnings_data.append({
                            "Symbol": e.symbol,
                            "Company": e.company_name[:30],
                            "Earnings Date": e.earnings_date,
                            "EPS Estimate": f"${e.eps_estimate:.2f}" if e.eps_estimate else "N/A",
                            "Revenue Est.": f"${e.revenue_estimate/1e9:.2f}B" if e.revenue_estimate >= 1e9 else f"${e.revenue_estimate/1e6:.0f}M" if e.revenue_estimate else "N/A"
                        })
                    st.dataframe(pd.DataFrame(earnings_data), hide_index=True, use_container_width=True)
                else:
                    st.info("No upcoming earnings in the selected period")

                # Dividend Calendar
                st.markdown("##### 💰 Upcoming Dividends")
                if summary.dividend_events:
                    div_data = []
                    for d in summary.dividend_events[:20]:
                        div_data.append({
                            "Symbol": d.symbol,
                            "Company": d.company_name[:30],
                            "Ex-Dividend Date": d.ex_dividend_date,
                            "Amount": f"${d.dividend_amount:.2f}" if d.dividend_amount else "N/A",
                            "Yield": f"{d.dividend_yield*100:.2f}%" if d.dividend_yield else "N/A"
                        })
                    st.dataframe(pd.DataFrame(div_data), hide_index=True, use_container_width=True)
                else:
                    st.info("No upcoming dividends in the selected period")

                # Recent Analyst Actions
                st.markdown("##### 📝 Recent Analyst Actions")
                analyst_actions = calendar.get_recent_analyst_actions(symbols, days_back=30)
                if analyst_actions:
                    analyst_data = []
                    for a in analyst_actions[:15]:
                        analyst_data.append({
                            "Date": a.date,
                            "Symbol": a.symbol,
                            "Firm": a.firm[:20],
                            "Action": a.action,
                            "Old Rating": a.old_rating,
                            "New Rating": a.new_rating
                        })
                    st.dataframe(pd.DataFrame(analyst_data), hide_index=True, use_container_width=True)
                else:
                    st.info("No recent analyst actions")
        else:
            st.warning("Please enter at least one symbol")


def display_institutional_activity_tool():
    """Display institutional and insider activity tool."""
    st.subheader("🏛️ Institutional & Insider Activity")
    st.markdown("Track smart money movements and insider transactions")

    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Enter Stock Symbol", value="AAPL", key="inst_symbol").upper()
    with col2:
        days_back = st.slider("Insider History (Days)", 30, 180, 90, key="inst_days")

    if st.button("Analyze Ownership", type="primary", key="inst_run"):
        if symbol:
            tracker = get_institutional_tracker()

            with st.spinner(f"Analyzing ownership for {symbol}..."):
                summary = tracker.get_ownership_summary(symbol)
                sentiment = tracker.analyze_insider_sentiment(symbol, period_days=days_back)

                # Summary metrics
                st.markdown(f"### {summary.company_name}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Institutional Ownership", f"{summary.institutional_pct*100:.1f}%")
                with col2:
                    st.metric("Insider Ownership", f"{summary.insider_pct*100:.1f}%")
                with col3:
                    st.metric("Institutional Trend", summary.institutional_trend)
                with col4:
                    sentiment_color = "green" if "Buy" in sentiment.sentiment_label else "red" if "Sell" in sentiment.sentiment_label else "gray"
                    st.metric("Insider Sentiment", sentiment.sentiment_label)

                st.markdown("---")

                # Two columns for institutional and insider
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("##### 🏦 Top Institutional Holders")
                    if summary.top_institutional_holders:
                        inst_data = []
                        for h in summary.top_institutional_holders:
                            inst_data.append({
                                "Holder": h.holder_name[:30],
                                "Shares": f"{h.shares:,}",
                                "Value": f"${h.value/1e6:.1f}M" if h.value >= 1e6 else f"${h.value:,.0f}",
                                "% Held": f"{h.pct_held:.2f}%"
                            })
                        st.dataframe(pd.DataFrame(inst_data), hide_index=True, use_container_width=True)
                    else:
                        st.info("No institutional holder data available")

                    st.markdown("##### 📊 Top Fund Holders")
                    if summary.top_fund_holders:
                        fund_data = []
                        for h in summary.top_fund_holders:
                            fund_data.append({
                                "Fund": h.holder_name[:30],
                                "Shares": f"{h.shares:,}",
                                "Value": f"${h.value/1e6:.1f}M" if h.value >= 1e6 else f"${h.value:,.0f}"
                            })
                        st.dataframe(pd.DataFrame(fund_data), hide_index=True, use_container_width=True)
                    else:
                        st.info("No fund holder data available")

                with col2:
                    st.markdown("##### 👤 Insider Activity Summary")

                    # Sentiment gauge
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Sentiment Score", f"{sentiment.sentiment_score:+.0f}")
                    with col_b:
                        st.metric("Net Value", f"${sentiment.net_value:+,.0f}")

                    st.markdown("**Transaction Breakdown:**")
                    trans_data = {
                        "Type": ["Buy Transactions", "Sell Transactions", "Total Buy Value", "Total Sell Value", "Unique Buyers", "Unique Sellers"],
                        "Value": [
                            sentiment.buy_transactions,
                            sentiment.sell_transactions,
                            f"${sentiment.total_buy_value:,.0f}",
                            f"${sentiment.total_sell_value:,.0f}",
                            sentiment.unique_insiders_buying,
                            sentiment.unique_insiders_selling
                        ]
                    }
                    st.dataframe(pd.DataFrame(trans_data), hide_index=True, use_container_width=True)

                    # Recent transactions
                    st.markdown("##### 📋 Recent Insider Transactions")
                    transactions = tracker.get_insider_transactions(symbol, days_back=days_back)
                    if transactions:
                        trans_list = []
                        for t in transactions[:10]:
                            trans_list.append({
                                "Date": t.transaction_date,
                                "Insider": t.insider_name[:20],
                                "Type": t.transaction_type.value,
                                "Shares": f"{t.shares:,}",
                                "Value": f"${t.value:,.0f}"
                            })
                        st.dataframe(pd.DataFrame(trans_list), hide_index=True, use_container_width=True)
                    else:
                        st.info("No recent insider transactions")


def display_risk_analytics_tool():
    """Display portfolio risk analytics tool."""
    st.subheader("📈 Portfolio Risk Analytics")
    st.markdown("Calculate risk metrics, correlations, and benchmark comparisons")

    # Input method
    input_method = st.radio("Input Method", ["Enter Symbols", "From Watchlist"], horizontal=True, key="risk_input")

    if input_method == "Enter Symbols":
        symbols_input = st.text_input("Portfolio Symbols (comma-separated)", value="AAPL,MSFT,GOOGL,AMZN,META", key="risk_symbols")
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]
    else:
        wm = get_watchlist_manager()
        watchlist_names = wm.get_watchlist_names()
        if watchlist_names:
            selected_wl = st.selectbox("Select Watchlist", watchlist_names, key="risk_watchlist")
            wl = wm.get_watchlist(selected_wl)
            symbols = [s.symbol for s in wl.stocks] if wl else []
        else:
            st.warning("No watchlists available. Create one first.")
            symbols = []

    col1, col2 = st.columns(2)
    with col1:
        benchmark = st.selectbox("Benchmark", ["SPY", "QQQ", "IWM", "DIA"], key="risk_benchmark")
    with col2:
        period = st.selectbox("Analysis Period", ["6mo", "1y", "2y", "5y"], index=1, key="risk_period")

    if st.button("Analyze Portfolio Risk", type="primary", key="risk_run"):
        if symbols and len(symbols) >= 2:
            analytics = get_portfolio_analytics()

            with st.spinner("Calculating portfolio metrics..."):
                # Get portfolio returns (equal-weighted)
                all_returns = []
                valid_symbols = []

                for sym in symbols:
                    hist = get_ticker_history(sym, period=period)
                    if hist is not None and not hist.empty and 'Close' in hist.columns:
                        returns = hist['Close'].pct_change().dropna().values
                        if len(returns) > 50:
                            all_returns.append(returns)
                            valid_symbols.append(sym)

                if len(all_returns) >= 2:
                    # Align all return series
                    min_len = min(len(r) for r in all_returns)
                    aligned = [r[-min_len:] for r in all_returns]

                    # Equal-weighted portfolio returns
                    import numpy as np
                    portfolio_returns = np.mean(aligned, axis=0).tolist()

                    # Get benchmark returns
                    bench_hist = get_ticker_history(benchmark, period=period)
                    if bench_hist is not None and not bench_hist.empty:
                        bench_returns = bench_hist['Close'].pct_change().dropna().values[-min_len:].tolist()
                    else:
                        bench_returns = None

                    # Calculate metrics
                    risk_metrics = analytics.calculate_risk_metrics(portfolio_returns, bench_returns)
                    perf_metrics = analytics.calculate_performance_metrics(portfolio_returns)
                    correlation = analytics.calculate_correlation_matrix(valid_symbols, period=period)

                    # Display Risk Metrics
                    st.markdown("---")
                    st.markdown("##### 📊 Risk Metrics")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Annual Volatility", f"{risk_metrics.volatility_annual*100:.1f}%")
                        st.metric("Max Drawdown", f"{risk_metrics.max_drawdown*100:.1f}%")
                    with col2:
                        st.metric("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}")
                        st.metric("Sortino Ratio", f"{risk_metrics.sortino_ratio:.2f}")
                    with col3:
                        st.metric("VaR (95%)", f"{risk_metrics.var_95*100:.2f}%")
                        st.metric("CVaR (95%)", f"{risk_metrics.cvar_95*100:.2f}%")
                    with col4:
                        st.metric("Beta", f"{risk_metrics.beta:.2f}")
                        st.metric("Alpha (Annual)", f"{risk_metrics.alpha*100:.2f}%")

                    # Performance Metrics
                    st.markdown("---")
                    st.markdown("##### 📈 Performance Metrics")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Return", f"{perf_metrics.total_return*100:+.1f}%")
                    with col2:
                        st.metric("Annualized Return", f"{perf_metrics.annualized_return*100:+.1f}%")
                    with col3:
                        st.metric("Best Day", f"{perf_metrics.best_day*100:+.2f}%")
                    with col4:
                        st.metric("Worst Day", f"{perf_metrics.worst_day*100:.2f}%")

                    period_col1, period_col2, period_col3, period_col4 = st.columns(4)
                    with period_col1:
                        st.metric("1 Week", f"{perf_metrics.return_1w*100:+.1f}%")
                    with period_col2:
                        st.metric("1 Month", f"{perf_metrics.return_1m*100:+.1f}%")
                    with period_col3:
                        st.metric("3 Months", f"{perf_metrics.return_3m*100:+.1f}%")
                    with period_col4:
                        st.metric("1 Year", f"{perf_metrics.return_1y*100:+.1f}%")

                    # Benchmark Comparison
                    if bench_returns:
                        benchmark_comp = analytics.compare_to_benchmark(portfolio_returns, benchmark, period)

                        st.markdown("---")
                        st.markdown(f"##### 🎯 Benchmark Comparison (vs {benchmark})")

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Portfolio Return", f"{benchmark_comp.portfolio_return*100:+.1f}%")
                        with col2:
                            st.metric("Benchmark Return", f"{benchmark_comp.benchmark_return*100:+.1f}%")
                        with col3:
                            excess = benchmark_comp.excess_return * 100
                            st.metric("Excess Return", f"{excess:+.1f}%",
                                    delta=f"{'Outperformed' if excess > 0 else 'Underperformed'}")
                        with col4:
                            st.metric("Information Ratio", f"{benchmark_comp.information_ratio:.2f}")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Up Capture", f"{benchmark_comp.up_capture*100:.0f}%")
                        with col2:
                            st.metric("Down Capture", f"{benchmark_comp.down_capture*100:.0f}%")
                        with col3:
                            st.metric("Batting Avg", f"{benchmark_comp.batting_average*100:.0f}%")

                    # Correlation Matrix
                    st.markdown("---")
                    st.markdown("##### 🔗 Correlation Matrix")

                    if correlation.matrix and len(correlation.matrix) > 1:
                        import numpy as np
                        corr_df = pd.DataFrame(
                            correlation.matrix,
                            columns=correlation.symbols,
                            index=correlation.symbols
                        )

                        fig = px.imshow(
                            corr_df,
                            labels=dict(color="Correlation"),
                            x=correlation.symbols,
                            y=correlation.symbols,
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)

                        # High correlation warnings
                        high_corr = []
                        for i in range(len(correlation.symbols)):
                            for j in range(i+1, len(correlation.symbols)):
                                if correlation.matrix[i][j] > 0.8:
                                    high_corr.append(f"{correlation.symbols[i]} & {correlation.symbols[j]}: {correlation.matrix[i][j]:.2f}")

                        if high_corr:
                            st.warning(f"⚠️ High correlations detected: {', '.join(high_corr)}")

                    # Concentration Risk
                    st.markdown("---")
                    st.markdown("##### 🎲 Diversification Analysis")

                    # Assume equal weights for now
                    positions = {s: 100000 / len(valid_symbols) for s in valid_symbols}
                    concentration = analytics.get_concentration_risk(positions)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Positions", concentration.get('total_positions', 0))
                    with col2:
                        st.metric("Effective Positions", f"{concentration.get('effective_positions', 0):.1f}")
                    with col3:
                        div_ratio = concentration.get('diversification_ratio', 0)
                        st.metric("Diversification Ratio", f"{div_ratio*100:.0f}%",
                                help="Higher is better. 100% means equal weights.")

                else:
                    st.error("Not enough valid data for portfolio analysis. Need at least 2 stocks with sufficient history.")
        else:
            st.warning("Please enter at least 2 symbols for portfolio analysis")


def main():
    """Main Streamlit app with navigation."""

    # Navigation
    st.sidebar.title("🏦 Bridgewater Analytics")

    page = st.sidebar.radio(
        "Navigation",
        ["📈 Stock Analysis", "📊 Portfolio Builder", "🔍 Stock Screener", "🛠️ Asset Manager Tools"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    if page == "📈 Stock Analysis":
        display_stock_analysis_page()
    elif page == "📊 Portfolio Builder":
        display_portfolio_builder_page()
    elif page == "🔍 Stock Screener":
        display_stock_screener_page()
    else:
        display_asset_manager_tools_page()


if __name__ == "__main__":
    main()
