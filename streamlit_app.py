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


def main():
    """Main Streamlit app with navigation."""

    # Navigation
    st.sidebar.title("🏦 Bridgewater Analytics")

    page = st.sidebar.radio(
        "Navigation",
        ["📈 Stock Analysis", "📊 Portfolio Builder"],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    if page == "📈 Stock Analysis":
        display_stock_analysis_page()
    else:
        display_portfolio_builder_page()


if __name__ == "__main__":
    main()
