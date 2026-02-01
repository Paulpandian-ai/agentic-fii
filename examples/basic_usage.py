"""
Basic usage example for the Stock Analysis Multi-Agent Platform.

This example demonstrates how to:
1. Create a Master Agent
2. Analyze a stock using all servant agents
3. Access the analysis results
"""

import asyncio
import sys
sys.path.insert(0, '..')

from src.agents.master_agent import MasterAgent
from src.utils.helpers import format_currency, format_percentage


async def basic_analysis():
    """Run a basic stock analysis."""
    # Create the master agent with default settings
    master = MasterAgent(execution_mode="parallel")

    # Analyze a stock
    symbol = "AAPL"
    print(f"Analyzing {symbol}...")

    report = await master.analyze(symbol)

    # Print results
    print(f"\n{'='*50}")
    print(f"Analysis Report for {report.symbol}")
    print(f"{'='*50}")

    if report.company_name:
        print(f"Company: {report.company_name}")

    if report.stock_data and report.stock_data.current_price:
        print(f"Current Price: {format_currency(report.stock_data.current_price)}")

    print(f"\nOverall Score: {report.overall_score:.1f}/100")
    print(f"Recommendation: {report.recommendation}")
    print(f"Confidence: {report.confidence:.0f}%")

    print("\n--- Agent Results ---")
    for result in report.agent_results:
        score = f"{result.score:.1f}" if result.score else "N/A"
        print(f"  {result.agent_type}: {score}")

    if report.key_strengths:
        print("\nStrengths:")
        for s in report.key_strengths:
            print(f"  + {s}")

    if report.key_risks:
        print("\nRisks:")
        for r in report.key_risks:
            print(f"  - {r}")

    print(f"\nExecution Time: {report.total_execution_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(basic_analysis())
