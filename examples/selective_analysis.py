"""
Example showing selective agent execution.

This example demonstrates how to:
1. Run only specific agents
2. Use sequential execution
3. Compare results between different analysis modes
"""

import asyncio
import sys

sys.path.insert(0, '..')

from src.agents.master_agent import MasterAgent


async def selective_analysis():
    """Run analysis with only selected agents."""
    master = MasterAgent(execution_mode="parallel")
    symbol = "MSFT"

    print(f"Selective Analysis for {symbol}")
    print("=" * 50)

    # Example 1: Quick technical analysis only
    print("\n1. Quick Technical Analysis:")
    report = await master.analyze(symbol, agents_to_run=["technical"])
    if report.technical_analysis:
        print(f"   Trend: {report.technical_analysis.trend_direction}")
        print(f"   RSI: {report.technical_analysis.rsi:.1f}" if report.technical_analysis.rsi else "   RSI: N/A")
        print(f"   Score: {report.technical_analysis.score:.1f}" if report.technical_analysis.score else "   Score: N/A")
    print(f"   Execution Time: {report.total_execution_time:.2f}s")

    # Example 2: Fundamental + Risk analysis
    print("\n2. Fundamental + Risk Analysis:")
    report = await master.analyze(
        symbol,
        agents_to_run=["fundamental", "risk"]
    )
    print(f"   Overall Score: {report.overall_score:.1f}")
    for result in report.agent_results:
        score = f"{result.score:.1f}" if result.score else "N/A"
        print(f"   {result.agent_type}: {score}")
    print(f"   Execution Time: {report.total_execution_time:.2f}s")

    # Example 3: Full analysis with sequential execution
    print("\n3. Full Sequential Analysis:")
    master_seq = MasterAgent(execution_mode="sequential")
    report = await master_seq.analyze(symbol)
    print(f"   Overall Score: {report.overall_score:.1f}")
    print(f"   Recommendation: {report.recommendation}")
    print(f"   Execution Time: {report.total_execution_time:.2f}s")


async def compare_execution_modes():
    """Compare parallel vs sequential execution."""
    symbol = "GOOGL"

    print(f"\nExecution Mode Comparison for {symbol}")
    print("=" * 50)

    # Parallel execution
    master_parallel = MasterAgent(execution_mode="parallel")
    report_parallel = await master_parallel.analyze(symbol)

    # Sequential execution
    master_sequential = MasterAgent(execution_mode="sequential")
    report_sequential = await master_sequential.analyze(symbol)

    print(f"\nParallel Execution:")
    print(f"   Time: {report_parallel.total_execution_time:.2f}s")
    print(f"   Score: {report_parallel.overall_score:.1f}")

    print(f"\nSequential Execution:")
    print(f"   Time: {report_sequential.total_execution_time:.2f}s")
    print(f"   Score: {report_sequential.overall_score:.1f}")

    speedup = report_sequential.total_execution_time / report_parallel.total_execution_time
    print(f"\nParallel speedup: {speedup:.2f}x faster")


if __name__ == "__main__":
    asyncio.run(selective_analysis())
    asyncio.run(compare_execution_modes())
