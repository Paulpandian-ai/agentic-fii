"""
Stock Analysis Multi-Agent Platform - Streamlit UI

A comprehensive web interface for analyzing stocks using multiple specialized agents.
Run with: streamlit run streamlit_app.py
"""

import asyncio
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from src.agents.master_agent import MasterAgent
from src.models.schemas import AnalysisReport, AgentStatus


# Page configuration
st.set_page_config(
    page_title="Stock Analysis Platform",
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-top: 10px;
        padding-bottom: 10px;
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


def display_agent_results(report: AnalysisReport):
    """Display individual agent results in tabs."""
    st.subheader("📈 Detailed Analysis")

    tabs = st.tabs(["Fundamental", "Technical", "Sentiment", "Risk"])

    # Fundamental Tab
    with tabs[0]:
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

    # Technical Tab
    with tabs[1]:
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

    # Sentiment Tab
    with tabs[2]:
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

    # Risk Tab
    with tabs[3]:
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


def display_executive_summary(report: AnalysisReport):
    """Display executive summary."""
    if report.executive_summary:
        st.subheader("📋 Executive Summary")
        st.write(report.executive_summary)


async def run_analysis(symbol: str, agents: list[str], mode: str) -> AnalysisReport:
    """Run the analysis using the master agent."""
    master = MasterAgent(execution_mode=mode)

    agents_to_run = agents if agents else None
    report = await master.analyze(symbol, agents_to_run=agents_to_run)

    return report


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: gray;">Multi-Agent System for Comprehensive Stock Analysis</p>',
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

        # Agent selection
        st.subheader("Select Agents")
        all_agents = st.checkbox("Run All Agents", value=True)

        selected_agents = []
        if not all_agents:
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
        This platform uses specialized AI agents to analyze:
        - **Fundamental**: Financial metrics & valuation
        - **Technical**: Price patterns & indicators
        - **Sentiment**: News & market sentiment
        - **Risk**: Volatility & risk factors
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
                report = asyncio.run(run_analysis(symbol, agents_to_run, mode))

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
            import json
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


if __name__ == "__main__":
    main()
