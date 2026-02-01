"""
Stock Analysis Multi-Agent Platform

A multi-agent system for comprehensive stock analysis featuring:
- Master Agent: Orchestrates and coordinates all servant agents
- Fundamental Analysis Agent: Analyzes financial metrics
- Technical Analysis Agent: Analyzes price patterns and indicators
- Sentiment Analysis Agent: Analyzes news and market sentiment
- Risk Assessment Agent: Evaluates risk factors and volatility

Usage:
    python main.py AAPL           # Analyze a single stock
    python main.py AAPL GOOGL MSFT  # Analyze multiple stocks
    python main.py --help         # Show help
"""

import argparse
import asyncio
import json
import sys
from typing import Optional

from loguru import logger

from config.settings import get_settings
from src.agents.master_agent import MasterAgent
from src.models.schemas import AnalysisReport
from src.utils.helpers import format_currency, format_percentage, setup_logging


def print_report(report: AnalysisReport) -> None:
    """Print the analysis report in a formatted way."""
    print("\n" + "=" * 70)
    print(f"  STOCK ANALYSIS REPORT: {report.symbol}")
    if report.company_name:
        print(f"  {report.company_name}")
    print("=" * 70)

    # Stock Data
    if report.stock_data:
        print("\n📊 CURRENT MARKET DATA")
        print("-" * 40)
        if report.stock_data.current_price:
            print(f"  Current Price:    {format_currency(report.stock_data.current_price)}")
        if report.stock_data.previous_close:
            print(f"  Previous Close:   {format_currency(report.stock_data.previous_close)}")
        if report.stock_data.market_cap:
            print(f"  Market Cap:       {format_currency(report.stock_data.market_cap)}")
        if report.stock_data.volume:
            print(f"  Volume:           {report.stock_data.volume:,}")

    # Overall Results
    print("\n🎯 OVERALL ANALYSIS")
    print("-" * 40)
    if report.overall_score is not None:
        score_bar = "█" * int(report.overall_score / 5) + "░" * (20 - int(report.overall_score / 5))
        print(f"  Overall Score:    [{score_bar}] {report.overall_score:.1f}/100")
    if report.recommendation:
        print(f"  Recommendation:   {report.recommendation}")
    if report.confidence is not None:
        print(f"  Confidence:       {report.confidence:.0f}%")

    # Individual Agent Scores
    print("\n📈 ANALYSIS BREAKDOWN")
    print("-" * 40)
    for result in report.agent_results:
        status_icon = "✅" if result.status.value == "completed" else "❌"
        score_str = f"{result.score:.1f}" if result.score is not None else "N/A"
        print(f"  {status_icon} {result.agent_type.capitalize():15} Score: {score_str:>6}")

    # Fundamental Analysis
    if report.fundamental_analysis and report.fundamental_analysis.analysis_summary:
        print("\n💰 FUNDAMENTAL ANALYSIS")
        print("-" * 40)
        fa = report.fundamental_analysis
        if fa.pe_ratio:
            print(f"  P/E Ratio:        {fa.pe_ratio:.2f}")
        if fa.profit_margin:
            print(f"  Profit Margin:    {format_percentage(fa.profit_margin)}")
        if fa.revenue_growth:
            print(f"  Revenue Growth:   {format_percentage(fa.revenue_growth)}")
        if fa.debt_to_equity:
            print(f"  Debt/Equity:      {fa.debt_to_equity:.2f}")
        print(f"\n  Summary: {fa.analysis_summary}")

    # Technical Analysis
    if report.technical_analysis and report.technical_analysis.analysis_summary:
        print("\n📉 TECHNICAL ANALYSIS")
        print("-" * 40)
        ta = report.technical_analysis
        if ta.trend_direction:
            print(f"  Trend:            {ta.trend_direction.upper()}")
        if ta.rsi:
            print(f"  RSI:              {ta.rsi:.1f}")
        if ta.buy_signals:
            print(f"  Buy Signals:      {len(ta.buy_signals)}")
        if ta.sell_signals:
            print(f"  Sell Signals:     {len(ta.sell_signals)}")
        print(f"\n  Summary: {ta.analysis_summary}")

    # Sentiment Analysis
    if report.sentiment_analysis and report.sentiment_analysis.analysis_summary:
        print("\n💬 SENTIMENT ANALYSIS")
        print("-" * 40)
        sa = report.sentiment_analysis
        if sa.overall_sentiment:
            print(f"  Sentiment:        {sa.overall_sentiment.upper()}")
        if sa.news_count > 0:
            print(f"  News Analyzed:    {sa.news_count}")
        if sa.analyst_rating:
            print(f"  Analyst Rating:   {sa.analyst_rating}")
        if sa.target_price:
            print(f"  Target Price:     {format_currency(sa.target_price)}")
        print(f"\n  Summary: {sa.analysis_summary}")

    # Risk Assessment
    if report.risk_assessment and report.risk_assessment.analysis_summary:
        print("\n⚠️  RISK ASSESSMENT")
        print("-" * 40)
        ra = report.risk_assessment
        if ra.risk_level:
            print(f"  Risk Level:       {ra.risk_level.upper()}")
        if ra.volatility_annual:
            print(f"  Annual Volatility:{format_percentage(ra.volatility_annual)}")
        if ra.beta:
            print(f"  Beta:             {ra.beta:.2f}")
        if ra.sharpe_ratio:
            print(f"  Sharpe Ratio:     {ra.sharpe_ratio:.2f}")
        print(f"\n  Summary: {ra.analysis_summary}")

    # Key Insights
    if report.key_strengths or report.key_risks:
        print("\n🔍 KEY INSIGHTS")
        print("-" * 40)
        if report.key_strengths:
            print("  Strengths:")
            for strength in report.key_strengths:
                print(f"    ✓ {strength}")
        if report.key_risks:
            print("  Risks:")
            for risk in report.key_risks:
                print(f"    ⚠ {risk}")
        if report.key_catalysts:
            print("  Potential Catalysts:")
            for catalyst in report.key_catalysts:
                print(f"    → {catalyst}")

    # Executive Summary
    if report.executive_summary:
        print("\n📋 EXECUTIVE SUMMARY")
        print("-" * 40)
        print(f"  {report.executive_summary}")

    # Metadata
    print("\n" + "-" * 40)
    print(f"  Analysis Time: {report.total_execution_time:.2f}s")
    print(f"  Agents Run: {report.successful_agents}/{report.agents_executed}")
    print("=" * 70 + "\n")


async def analyze_stock(
    symbol: str,
    master: MasterAgent,
    output_json: bool = False,
) -> Optional[AnalysisReport]:
    """Analyze a single stock."""
    try:
        report = await master.analyze(symbol)

        if output_json:
            print(json.dumps(report.model_dump(), indent=2, default=str))
        else:
            print_report(report)

        return report

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        print(f"\n❌ Error analyzing {symbol}: {e}\n")
        return None


async def main_async(
    symbols: list[str],
    execution_mode: str = "parallel",
    output_json: bool = False,
) -> None:
    """Main async function to run analysis."""
    settings = get_settings()
    setup_logging(settings.log_level)

    # Create master agent
    master = MasterAgent(
        execution_mode=execution_mode,
        weights=settings.agent_weights,
    )

    print(f"\n🚀 Starting analysis for {len(symbols)} stock(s)...")
    print(f"   Mode: {execution_mode.upper()}")
    print(f"   Agents: {', '.join(master.agents.keys())}")

    # Analyze each stock
    for symbol in symbols:
        await analyze_stock(symbol, master, output_json)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Stock Analysis Multi-Agent Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL              Analyze Apple stock
  python main.py AAPL GOOGL MSFT   Analyze multiple stocks
  python main.py AAPL --json       Output as JSON
  python main.py AAPL --sequential Run agents sequentially
        """,
    )

    parser.add_argument(
        "symbols",
        nargs="+",
        help="Stock ticker symbol(s) to analyze",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["parallel", "sequential"],
        default="parallel",
        help="Agent execution mode (default: parallel)",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--sequential",
        "-s",
        action="store_true",
        help="Run agents sequentially (shorthand for --mode sequential)",
    )

    args = parser.parse_args()

    # Handle sequential shorthand
    execution_mode = "sequential" if args.sequential else args.mode

    # Convert symbols to uppercase
    symbols = [s.upper() for s in args.symbols]

    # Run async main
    asyncio.run(main_async(symbols, execution_mode, args.json))


if __name__ == "__main__":
    main()
