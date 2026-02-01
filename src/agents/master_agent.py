"""Master Agent - Orchestrates servant agents and summarizes results."""

import asyncio
import time
from typing import Any, Optional

import yfinance as yf
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.agents.fundamental_agent import FundamentalAnalysisAgent
from src.agents.technical_agent import TechnicalAnalysisAgent
from src.agents.sentiment_agent import SentimentAnalysisAgent
from src.agents.risk_agent import RiskAssessmentAgent
from src.models.schemas import (
    AgentResult,
    AgentStatus,
    AnalysisReport,
    FundamentalMetrics,
    RiskMetrics,
    SentimentData,
    StockData,
    TechnicalIndicators,
)


class MasterAgent:
    """
    Master Agent that orchestrates all servant agents.

    The Master Agent is responsible for:
    - Triggering servant agents (sequentially or in parallel)
    - Collecting and aggregating results
    - Generating the final analysis report
    - Providing investment recommendations

    Attributes:
        agents: Dictionary of servant agents
        execution_mode: "parallel" or "sequential"
        weights: Weights for each analysis type in final score
    """

    def __init__(
        self,
        execution_mode: str = "parallel",
        weights: Optional[dict[str, float]] = None,
    ):
        """
        Initialize the Master Agent.

        Args:
            execution_mode: "parallel" to run agents concurrently,
                          "sequential" to run one at a time
            weights: Custom weights for scoring (default: equal weights)
        """
        self.execution_mode = execution_mode
        self.weights = weights or {
            "fundamental": 0.30,
            "technical": 0.25,
            "sentiment": 0.20,
            "risk": 0.25,
        }

        # Initialize servant agents
        self.agents: dict[str, BaseAgent] = {
            "fundamental": FundamentalAnalysisAgent(),
            "technical": TechnicalAnalysisAgent(),
            "sentiment": SentimentAnalysisAgent(),
            "risk": RiskAssessmentAgent(),
        }

        logger.info(
            f"Master Agent initialized with {len(self.agents)} servant agents "
            f"in {execution_mode} mode"
        )

    async def analyze(
        self,
        symbol: str,
        agents_to_run: Optional[list[str]] = None,
        **kwargs,
    ) -> AnalysisReport:
        """
        Perform comprehensive stock analysis using servant agents.

        Args:
            symbol: Stock ticker symbol to analyze
            agents_to_run: List of specific agents to run (default: all)
            **kwargs: Additional parameters passed to agents

        Returns:
            AnalysisReport containing complete analysis results
        """
        start_time = time.time()
        symbol = symbol.upper()

        logger.info(f"Master Agent starting analysis for {symbol}")

        # Determine which agents to run
        if agents_to_run:
            selected_agents = {
                k: v for k, v in self.agents.items() if k in agents_to_run
            }
        else:
            selected_agents = self.agents

        # Get basic stock data first
        stock_data = await self._fetch_stock_data(symbol)

        # Execute servant agents
        if self.execution_mode == "parallel":
            agent_results = await self._run_agents_parallel(
                symbol, selected_agents, **kwargs
            )
        else:
            agent_results = await self._run_agents_sequential(
                symbol, selected_agents, **kwargs
            )

        # Build the analysis report
        report = self._build_report(symbol, stock_data, agent_results)

        # Calculate overall score and recommendation
        report.overall_score = self._calculate_overall_score(agent_results)
        report.recommendation = self._generate_recommendation(
            report.overall_score, agent_results
        )
        report.confidence = self._calculate_confidence(agent_results)

        # Generate executive summary
        report.executive_summary = self._generate_executive_summary(report)

        # Extract key insights
        report.key_strengths = self._extract_strengths(report)
        report.key_risks = self._extract_risks(report)
        report.key_catalysts = self._extract_catalysts(report)

        # Set metadata
        report.total_execution_time = time.time() - start_time
        report.agents_executed = len(selected_agents)
        report.successful_agents = sum(
            1 for r in agent_results if r.status == AgentStatus.COMPLETED
        )

        logger.info(
            f"Master Agent completed analysis for {symbol} in "
            f"{report.total_execution_time:.2f}s"
        )

        return report

    async def _fetch_stock_data(self, symbol: str) -> StockData:
        """Fetch basic stock data."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return StockData(
                symbol=symbol,
                name=info.get("longName") or info.get("shortName"),
                current_price=info.get("currentPrice") or info.get("regularMarketPrice"),
                previous_close=info.get("previousClose"),
                open_price=info.get("open"),
                high=info.get("dayHigh"),
                low=info.get("dayLow"),
                volume=info.get("volume"),
                market_cap=info.get("marketCap"),
            )
        except Exception as e:
            logger.warning(f"Error fetching stock data for {symbol}: {e}")
            return StockData(symbol=symbol)

    async def _run_agents_parallel(
        self, symbol: str, agents: dict[str, BaseAgent], **kwargs
    ) -> list[AgentResult]:
        """Run all agents in parallel."""
        logger.info(f"Running {len(agents)} agents in parallel")

        tasks = [
            agent.execute(symbol, **kwargs)
            for agent in agents.values()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            agent_name = list(agents.keys())[i]
            if isinstance(result, Exception):
                logger.error(f"Agent {agent_name} failed with exception: {result}")
                processed_results.append(
                    AgentResult(
                        agent_name=agent_name,
                        agent_type=agent_name,
                        status=AgentStatus.FAILED,
                        errors=[str(result)],
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _run_agents_sequential(
        self, symbol: str, agents: dict[str, BaseAgent], **kwargs
    ) -> list[AgentResult]:
        """Run agents one at a time."""
        logger.info(f"Running {len(agents)} agents sequentially")

        results = []
        for name, agent in agents.items():
            logger.info(f"Executing agent: {name}")
            try:
                result = await agent.execute(symbol, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Agent {name} failed: {e}")
                results.append(
                    AgentResult(
                        agent_name=name,
                        agent_type=name,
                        status=AgentStatus.FAILED,
                        errors=[str(e)],
                    )
                )

        return results

    def _build_report(
        self,
        symbol: str,
        stock_data: StockData,
        agent_results: list[AgentResult],
    ) -> AnalysisReport:
        """Build the analysis report from agent results."""
        report = AnalysisReport(
            symbol=symbol,
            company_name=stock_data.name,
            stock_data=stock_data,
            agent_results=agent_results,
        )

        # Extract typed analysis results
        for result in agent_results:
            if result.status != AgentStatus.COMPLETED:
                continue

            if result.agent_type == "fundamental" and "metrics" in result.data:
                report.fundamental_analysis = FundamentalMetrics(
                    **result.data["metrics"]
                )
                if report.fundamental_analysis:
                    report.fundamental_analysis.score = result.score
                    report.fundamental_analysis.analysis_summary = result.summary

            elif result.agent_type == "technical" and "indicators" in result.data:
                report.technical_analysis = TechnicalIndicators(
                    **result.data["indicators"]
                )
                if report.technical_analysis:
                    report.technical_analysis.score = result.score
                    report.technical_analysis.analysis_summary = result.summary

            elif result.agent_type == "sentiment" and "sentiment_data" in result.data:
                report.sentiment_analysis = SentimentData(
                    **result.data["sentiment_data"]
                )
                if report.sentiment_analysis:
                    report.sentiment_analysis.score = result.score
                    report.sentiment_analysis.analysis_summary = result.summary

            elif result.agent_type == "risk" and "risk_metrics" in result.data:
                report.risk_assessment = RiskMetrics(**result.data["risk_metrics"])
                if report.risk_assessment:
                    report.risk_assessment.score = result.score
                    report.risk_assessment.analysis_summary = result.summary

        return report

    def _calculate_overall_score(self, results: list[AgentResult]) -> float:
        """Calculate weighted overall score from agent results."""
        total_weight = 0.0
        weighted_score = 0.0

        for result in results:
            if result.status == AgentStatus.COMPLETED and result.score is not None:
                weight = self.weights.get(result.agent_type, 0.25)
                weighted_score += result.score * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_score / total_weight
        return 50.0  # Default neutral score

    def _generate_recommendation(
        self, overall_score: float, results: list[AgentResult]
    ) -> str:
        """Generate investment recommendation based on analysis."""
        # Check for critical risk factors
        has_high_risk = False
        for result in results:
            if result.agent_type == "risk" and result.score is not None:
                if result.score < 30:
                    has_high_risk = True

        # Generate recommendation
        if overall_score >= 70:
            if has_high_risk:
                return "BUY (with caution due to risk factors)"
            return "STRONG BUY"
        elif overall_score >= 60:
            return "BUY"
        elif overall_score >= 50:
            return "HOLD"
        elif overall_score >= 40:
            return "SELL"
        else:
            return "STRONG SELL"

    def _calculate_confidence(self, results: list[AgentResult]) -> float:
        """Calculate confidence level in the recommendation."""
        successful = sum(1 for r in results if r.status == AgentStatus.COMPLETED)
        total = len(results)

        if total == 0:
            return 0.0

        # Base confidence on successful agents
        base_confidence = (successful / total) * 100

        # Adjust based on score variance
        scores = [r.score for r in results if r.score is not None]
        if len(scores) > 1:
            import statistics
            variance = statistics.stdev(scores)
            # Lower confidence if scores vary widely
            variance_penalty = min(variance / 2, 20)
            base_confidence -= variance_penalty

        return max(0, min(100, base_confidence))

    def _generate_executive_summary(self, report: AnalysisReport) -> str:
        """Generate executive summary of the analysis."""
        parts = []

        # Company intro
        name = report.company_name or report.symbol
        parts.append(f"Analysis of {name} ({report.symbol}).")

        # Current price
        if report.stock_data and report.stock_data.current_price:
            parts.append(f"Current price: ${report.stock_data.current_price:.2f}.")

        # Overall assessment
        if report.overall_score:
            if report.overall_score >= 65:
                parts.append("Overall assessment is positive.")
            elif report.overall_score <= 35:
                parts.append("Overall assessment is negative.")
            else:
                parts.append("Overall assessment is neutral.")

        # Key findings from each agent
        summaries = []
        for result in report.agent_results:
            if result.status == AgentStatus.COMPLETED and result.summary:
                summaries.append(f"{result.agent_type.capitalize()}: {result.summary}")

        if summaries:
            parts.append(" ".join(summaries[:2]))  # Include top 2 summaries

        # Recommendation
        if report.recommendation:
            parts.append(f"Recommendation: {report.recommendation}.")

        if report.confidence:
            parts.append(f"Confidence: {report.confidence:.0f}%.")

        return " ".join(parts)

    def _extract_strengths(self, report: AnalysisReport) -> list[str]:
        """Extract key strengths from the analysis."""
        strengths = []

        # From fundamental analysis
        if report.fundamental_analysis:
            fa = report.fundamental_analysis
            if fa.score and fa.score >= 65:
                strengths.append("Strong fundamental metrics")
            if fa.profit_margin and fa.profit_margin > 0.15:
                strengths.append(f"High profit margin ({fa.profit_margin:.1%})")
            if fa.revenue_growth and fa.revenue_growth > 0.15:
                strengths.append(f"Strong revenue growth ({fa.revenue_growth:.1%})")
            if fa.debt_to_equity and fa.debt_to_equity < 0.5:
                strengths.append("Low debt levels")

        # From technical analysis
        if report.technical_analysis:
            ta = report.technical_analysis
            if ta.trend_direction == "bullish":
                strengths.append("Bullish technical trend")
            if len(ta.buy_signals) > len(ta.sell_signals):
                strengths.append(f"{len(ta.buy_signals)} active buy signals")

        # From sentiment analysis
        if report.sentiment_analysis:
            sa = report.sentiment_analysis
            if sa.overall_sentiment == "bullish":
                strengths.append("Positive market sentiment")
            if sa.analyst_rating and "buy" in sa.analyst_rating.lower():
                strengths.append(f"Analyst consensus: {sa.analyst_rating}")

        # From risk assessment
        if report.risk_assessment:
            ra = report.risk_assessment
            if ra.risk_level == "low":
                strengths.append("Low risk profile")
            if ra.sharpe_ratio and ra.sharpe_ratio > 1:
                strengths.append(f"Good risk-adjusted returns (Sharpe: {ra.sharpe_ratio:.2f})")

        return strengths[:5]  # Return top 5 strengths

    def _extract_risks(self, report: AnalysisReport) -> list[str]:
        """Extract key risks from the analysis."""
        risks = []

        # From fundamental analysis
        if report.fundamental_analysis:
            fa = report.fundamental_analysis
            if fa.pe_ratio and fa.pe_ratio > 35:
                risks.append(f"High valuation (P/E: {fa.pe_ratio:.1f})")
            if fa.debt_to_equity and fa.debt_to_equity > 2:
                risks.append(f"High debt (D/E: {fa.debt_to_equity:.1f})")
            if fa.earnings_growth and fa.earnings_growth < 0:
                risks.append("Declining earnings")

        # From technical analysis
        if report.technical_analysis:
            ta = report.technical_analysis
            if ta.trend_direction == "bearish":
                risks.append("Bearish technical trend")
            if ta.rsi and ta.rsi > 70:
                risks.append("Overbought conditions (RSI > 70)")

        # From sentiment analysis
        if report.sentiment_analysis:
            sa = report.sentiment_analysis
            if sa.overall_sentiment == "bearish":
                risks.append("Negative market sentiment")
            if sa.negative_news_count > sa.positive_news_count:
                risks.append("Negative news sentiment")

        # From risk assessment
        if report.risk_assessment:
            ra = report.risk_assessment
            risks.extend(ra.risk_factors[:3])  # Add top risk factors

        return risks[:5]  # Return top 5 risks

    def _extract_catalysts(self, report: AnalysisReport) -> list[str]:
        """Identify potential catalysts for the stock."""
        catalysts = []

        # Technical catalysts
        if report.technical_analysis:
            ta = report.technical_analysis
            if ta.rsi and ta.rsi < 30:
                catalysts.append("Potential reversal from oversold levels")
            if "Golden cross" in str(ta.buy_signals):
                catalysts.append("Golden cross formation")

        # Fundamental catalysts
        if report.fundamental_analysis:
            fa = report.fundamental_analysis
            if fa.peg_ratio and fa.peg_ratio < 1:
                catalysts.append("Undervalued relative to growth (PEG < 1)")
            if fa.earnings_growth and fa.earnings_growth > 0.20:
                catalysts.append("Strong earnings momentum")

        # Sentiment catalysts
        if report.sentiment_analysis:
            sa = report.sentiment_analysis
            if sa.target_price and report.stock_data and report.stock_data.current_price:
                upside = (sa.target_price - report.stock_data.current_price) / report.stock_data.current_price
                if upside > 0.15:
                    catalysts.append(f"Analyst target implies {upside:.0%} upside")

        return catalysts[:3]  # Return top 3 catalysts

    def add_agent(self, name: str, agent: BaseAgent, weight: float = 0.25) -> None:
        """Add a new servant agent to the master."""
        self.agents[name] = agent
        self.weights[name] = weight
        logger.info(f"Added new agent: {name} with weight {weight}")

    def remove_agent(self, name: str) -> None:
        """Remove a servant agent."""
        if name in self.agents:
            del self.agents[name]
            del self.weights[name]
            logger.info(f"Removed agent: {name}")

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set custom weights for scoring."""
        self.weights = weights
        logger.info(f"Updated weights: {weights}")

    def get_agent_status(self) -> dict[str, AgentStatus]:
        """Get status of all agents."""
        return {name: agent.status for name, agent in self.agents.items()}
