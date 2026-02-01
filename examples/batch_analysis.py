"""
Example showing batch analysis of multiple stocks.

This example demonstrates how to:
1. Analyze multiple stocks
2. Compare and rank stocks
3. Generate a portfolio summary
"""

import asyncio
import sys
from typing import Optional

sys.path.insert(0, '..')

from src.agents.master_agent import MasterAgent
from src.models.schemas import AnalysisReport


async def batch_analysis(symbols: list[str]) -> list[AnalysisReport]:
    """Analyze multiple stocks and return reports."""
    master = MasterAgent(execution_mode="parallel")
    reports = []

    print(f"Analyzing {len(symbols)} stocks...")
    print("-" * 50)

    for i, symbol in enumerate(symbols, 1):
        print(f"\n[{i}/{len(symbols)}] Analyzing {symbol}...")
        try:
            report = await master.analyze(symbol)
            reports.append(report)
            print(f"   Score: {report.overall_score:.1f} | Recommendation: {report.recommendation}")
        except Exception as e:
            print(f"   Error: {e}")

    return reports


def rank_stocks(reports: list[AnalysisReport]) -> list[tuple[str, float, str]]:
    """Rank stocks by overall score."""
    ranked = []
    for report in reports:
        if report.overall_score is not None:
            ranked.append((
                report.symbol,
                report.overall_score,
                report.recommendation or "N/A"
            ))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def generate_summary(reports: list[AnalysisReport]) -> None:
    """Generate a summary of all analyzed stocks."""
    print("\n" + "=" * 60)
    print("  BATCH ANALYSIS SUMMARY")
    print("=" * 60)

    # Rankings
    rankings = rank_stocks(reports)
    print("\n📊 Stock Rankings (by Overall Score):")
    print("-" * 40)
    for i, (symbol, score, rec) in enumerate(rankings, 1):
        bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
        print(f"  {i}. {symbol:6} [{bar}] {score:5.1f}  {rec}")

    # Buy recommendations
    buy_stocks = [r for r in reports if r.recommendation and "BUY" in r.recommendation.upper()]
    if buy_stocks:
        print(f"\n✅ Buy Recommendations ({len(buy_stocks)}):")
        for report in sorted(buy_stocks, key=lambda x: x.overall_score or 0, reverse=True):
            print(f"   • {report.symbol}: {report.recommendation}")

    # Sell recommendations
    sell_stocks = [r for r in reports if r.recommendation and "SELL" in r.recommendation.upper()]
    if sell_stocks:
        print(f"\n❌ Sell Recommendations ({len(sell_stocks)}):")
        for report in sorted(sell_stocks, key=lambda x: x.overall_score or 0):
            print(f"   • {report.symbol}: {report.recommendation}")

    # Statistics
    print("\n📈 Statistics:")
    print("-" * 40)
    scores = [r.overall_score for r in reports if r.overall_score is not None]
    if scores:
        print(f"   Average Score: {sum(scores)/len(scores):.1f}")
        print(f"   Highest:       {max(scores):.1f}")
        print(f"   Lowest:        {min(scores):.1f}")

    # Common strengths and risks
    all_strengths = []
    all_risks = []
    for report in reports:
        all_strengths.extend(report.key_strengths)
        all_risks.extend(report.key_risks)

    if all_strengths:
        from collections import Counter
        top_strengths = Counter(all_strengths).most_common(3)
        print("\n🌟 Common Strengths:")
        for strength, count in top_strengths:
            print(f"   • {strength} ({count} stocks)")

    if all_risks:
        from collections import Counter
        top_risks = Counter(all_risks).most_common(3)
        print("\n⚠️  Common Risks:")
        for risk, count in top_risks:
            print(f"   • {risk} ({count} stocks)")

    print("\n" + "=" * 60)


async def main():
    """Run batch analysis on tech stocks."""
    # Example: Analyze major tech stocks
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]

    print("=" * 60)
    print("  STOCK BATCH ANALYSIS")
    print(f"  Analyzing: {', '.join(symbols)}")
    print("=" * 60)

    reports = await batch_analysis(symbols)
    generate_summary(reports)


if __name__ == "__main__":
    asyncio.run(main())
