"""
Example showing how to create and add custom agents.

This example demonstrates how to:
1. Create a custom agent by extending BaseAgent
2. Add custom agents to the Master Agent
3. Configure custom weights
"""

import asyncio
import sys
from typing import Any

sys.path.insert(0, '..')

from src.agents.base_agent import BaseAgent
from src.agents.master_agent import MasterAgent


class IndustryAnalysisAgent(BaseAgent):
    """
    Custom agent that analyzes industry/sector performance.

    This is an example of how to create a custom servant agent.
    """

    def __init__(self, name: str = "IndustryAgent"):
        super().__init__(name=name, agent_type="industry")

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """Perform industry analysis."""
        import yfinance as yf

        self.log_info(f"Analyzing industry data for {symbol}")

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")

            # In a real implementation, you would:
            # 1. Fetch sector ETF data
            # 2. Compare performance vs sector
            # 3. Analyze industry trends

            analysis = {
                "sector": sector,
                "industry": industry,
                "sector_performance": "Neutral",  # Placeholder
                "industry_outlook": "Stable",  # Placeholder
                "competitive_position": "Average",  # Placeholder
            }

            # Calculate a simple score
            score = 50.0  # Neutral baseline

            summary = (
                f"{symbol} operates in the {industry} industry within the "
                f"{sector} sector. Industry outlook is stable."
            )

            return {
                "analysis": analysis,
                "score": score,
                "summary": summary,
            }

        except Exception as e:
            self.log_warning(f"Error in industry analysis: {e}")
            return {
                "analysis": {},
                "score": None,
                "summary": f"Industry analysis failed: {e}",
            }


class DividendAnalysisAgent(BaseAgent):
    """
    Custom agent that focuses on dividend analysis.
    """

    def __init__(self, name: str = "DividendAgent"):
        super().__init__(name=name, agent_type="dividend")

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """Analyze dividend metrics."""
        import yfinance as yf

        self.log_info(f"Analyzing dividends for {symbol}")

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            dividend_yield = info.get("dividendYield", 0) or 0
            payout_ratio = info.get("payoutRatio", 0) or 0
            dividend_rate = info.get("dividendRate", 0) or 0

            # Get dividend history
            dividends = ticker.dividends
            has_consistent_dividends = len(dividends) > 20

            analysis = {
                "dividend_yield": dividend_yield,
                "payout_ratio": payout_ratio,
                "annual_dividend": dividend_rate,
                "consistent_history": has_consistent_dividends,
                "years_of_dividends": len(set(dividends.index.year)) if len(dividends) > 0 else 0,
            }

            # Calculate dividend score
            score = 50.0
            if dividend_yield > 0.03:
                score += 20
            elif dividend_yield > 0.02:
                score += 10
            elif dividend_yield > 0.01:
                score += 5

            if has_consistent_dividends:
                score += 15

            if 0.3 <= payout_ratio <= 0.6:
                score += 10  # Healthy payout ratio

            score = max(0, min(100, score))

            if dividend_yield > 0:
                summary = (
                    f"Dividend yield of {dividend_yield:.2%} with "
                    f"{'consistent' if has_consistent_dividends else 'inconsistent'} "
                    f"dividend history."
                )
            else:
                summary = f"{symbol} does not currently pay dividends."

            return {
                "analysis": analysis,
                "score": score,
                "summary": summary,
            }

        except Exception as e:
            self.log_warning(f"Error in dividend analysis: {e}")
            return {
                "analysis": {},
                "score": None,
                "summary": f"Dividend analysis failed: {e}",
            }


async def custom_agents_example():
    """Example using custom agents."""
    # Create master agent
    master = MasterAgent(execution_mode="parallel")

    # Add custom agents
    industry_agent = IndustryAnalysisAgent()
    dividend_agent = DividendAnalysisAgent()

    master.add_agent("industry", industry_agent, weight=0.15)
    master.add_agent("dividend", dividend_agent, weight=0.10)

    # Adjust weights for existing agents
    master.set_weights({
        "fundamental": 0.25,
        "technical": 0.20,
        "sentiment": 0.15,
        "risk": 0.15,
        "industry": 0.15,
        "dividend": 0.10,
    })

    # Now analyze with all agents including custom ones
    symbol = "JNJ"  # Johnson & Johnson - good dividend stock example
    print(f"Analyzing {symbol} with custom agents...")

    report = await master.analyze(symbol)

    print(f"\nAnalysis for {report.symbol}")
    print(f"Overall Score: {report.overall_score:.1f}")
    print(f"Recommendation: {report.recommendation}")

    print("\nAll Agent Results:")
    for result in report.agent_results:
        score = f"{result.score:.1f}" if result.score else "N/A"
        print(f"  {result.agent_type:15} Score: {score:>6}")
        if result.summary:
            print(f"    Summary: {result.summary[:80]}...")


if __name__ == "__main__":
    asyncio.run(custom_agents_example())
