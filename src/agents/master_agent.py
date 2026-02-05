"""Master Agent - Orchestrates servant agents and summarizes results."""

import asyncio
import time
from typing import Any, Optional

from loguru import logger

from src.utils.yfinance_cache import get_ticker_info, get_ticker_history

from src.agents.base_agent import BaseAgent
from src.agents.fundamental_agent import FundamentalAnalysisAgent
from src.agents.technical_agent import TechnicalAnalysisAgent
from src.agents.sentiment_agent import SentimentAnalysisAgent
from src.agents.risk_agent import RiskAssessmentAgent
from src.agents.supply_chain_agent import SupplyChainAgent
from src.agents.customer_agent import CustomerAnalysisAgent
from src.agents.competitive_agent import CompetitiveAnalysisAgent
from src.agents.macro_agent import MacroeconomicAgent
from src.agents.monetary_policy_agent import MonetaryPolicyAgent
from src.models.schemas import (
    AgentResult,
    AgentStatus,
    AnalysisReport,
    CompetitiveAnalysis,
    CustomerAnalysis,
    FundamentalMetrics,
    MacroeconomicData,
    MonetaryPolicyData,
    RiskMetrics,
    SentimentData,
    StockData,
    SupplyChainAnalysis,
    TechnicalIndicators,
)


class MasterAgent:
    """
    Master Agent that orchestrates all servant agents (Bridgewater-style).

    The Master Agent is responsible for:
    - Triggering servant agents (sequentially or in parallel)
    - Collecting and aggregating results
    - Generating the final analysis report
    - Providing investment recommendations
    - Analyzing the complete investment ecosystem

    Agent Categories:
    - Core Analysis: Fundamental, Technical, Sentiment, Risk
    - Ecosystem Analysis: Supply Chain, Customer, Competitive
    - Macro Analysis: Macroeconomic, Monetary Policy

    Attributes:
        agents: Dictionary of servant agents
        execution_mode: "parallel" or "sequential"
        weights: Weights for each analysis type in final score
        analysis_mode: "core", "full", or "ecosystem"
    """

    def __init__(
        self,
        execution_mode: str = "parallel",
        weights: Optional[dict[str, float]] = None,
        analysis_mode: str = "full",
    ):
        """
        Initialize the Master Agent.

        Args:
            execution_mode: "parallel" to run agents concurrently,
                          "sequential" to run one at a time
            weights: Custom weights for scoring (default: equal weights)
            analysis_mode: "core" (basic), "full" (all agents), or "ecosystem" (core + ecosystem)
        """
        self.execution_mode = execution_mode
        self.analysis_mode = analysis_mode

        # Default weights for all agent types
        self.weights = weights or {
            # Core Analysis (40%)
            "fundamental": 0.15,
            "technical": 0.10,
            "sentiment": 0.08,
            "risk": 0.07,
            # Ecosystem Analysis (35%)
            "supply_chain": 0.10,
            "customer": 0.10,
            "competitive": 0.15,
            # Macro Analysis (25%)
            "macro": 0.12,
            "monetary_policy": 0.13,
        }

        # Initialize all agents
        self._init_agents(analysis_mode)

        logger.info(
            f"Master Agent initialized with {len(self.agents)} agents "
            f"in {execution_mode} mode (analysis: {analysis_mode})"
        )

    def _init_agents(self, mode: str) -> None:
        """Initialize agents based on analysis mode."""
        # Core agents (always included)
        self.agents: dict[str, BaseAgent] = {
            "fundamental": FundamentalAnalysisAgent(),
            "technical": TechnicalAnalysisAgent(),
            "sentiment": SentimentAnalysisAgent(),
            "risk": RiskAssessmentAgent(),
        }

        # Ecosystem agents
        if mode in ["full", "ecosystem"]:
            self.agents.update({
                "supply_chain": SupplyChainAgent(),
                "customer": CustomerAnalysisAgent(),
                "competitive": CompetitiveAnalysisAgent(),
            })

        # Macro agents
        if mode == "full":
            self.agents.update({
                "macro": MacroeconomicAgent(),
                "monetary_policy": MonetaryPolicyAgent(),
            })

    async def analyze(
        self,
        symbol: str,
        agents_to_run: Optional[list[str]] = None,
        include_ecosystem: bool = True,
        include_macro: bool = True,
        **kwargs,
    ) -> AnalysisReport:
        """
        Perform comprehensive stock analysis using servant agents.

        Args:
            symbol: Stock ticker symbol to analyze
            agents_to_run: List of specific agents to run (default: all)
            include_ecosystem: Include supply chain, customer, competitive analysis
            include_macro: Include macroeconomic and monetary policy analysis
            **kwargs: Additional parameters passed to agents

        Returns:
            AnalysisReport containing complete analysis results
        """
        start_time = time.time()
        symbol = symbol.upper()

        logger.info(f"Master Agent starting comprehensive analysis for {symbol}")

        # Determine which agents to run
        if agents_to_run:
            selected_agents = {
                k: v for k, v in self.agents.items() if k in agents_to_run
            }
        else:
            selected_agents = {}
            # Core agents always run
            for key in ["fundamental", "technical", "sentiment", "risk"]:
                if key in self.agents:
                    selected_agents[key] = self.agents[key]

            # Ecosystem agents if requested
            if include_ecosystem:
                for key in ["supply_chain", "customer", "competitive"]:
                    if key in self.agents:
                        selected_agents[key] = self.agents[key]

            # Macro agents if requested
            if include_macro:
                for key in ["macro", "monetary_policy"]:
                    if key in self.agents:
                        selected_agents[key] = self.agents[key]

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
            report.overall_score, agent_results, report
        )
        report.confidence = self._calculate_confidence(agent_results)
        report.conviction_level = self._determine_conviction(report.confidence, report.overall_score)

        # Generate position sizing suggestion
        report.suggested_position_size = self._suggest_position_size(report)
        report.position_rationale = self._generate_position_rationale(report)

        # Generate investment thesis
        report.investment_thesis = self._generate_investment_thesis(report)

        # Generate executive summary
        report.executive_summary = self._generate_executive_summary(report)

        # Extract key insights
        report.key_strengths = self._extract_strengths(report)
        report.key_risks = self._extract_risks(report)
        report.key_catalysts = self._extract_catalysts(report)
        report.key_watchpoints = self._extract_watchpoints(report)

        # Generate outlook by time horizon
        report.short_term_outlook = self._generate_short_term_outlook(report)
        report.medium_term_outlook = self._generate_medium_term_outlook(report)
        report.long_term_outlook = self._generate_long_term_outlook(report)

        # Set metadata
        report.total_execution_time = time.time() - start_time
        report.agents_executed = len(selected_agents)
        report.successful_agents = sum(
            1 for r in agent_results if r.status == AgentStatus.COMPLETED
        )

        logger.info(
            f"Master Agent completed analysis for {symbol} in "
            f"{report.total_execution_time:.2f}s "
            f"({report.successful_agents}/{report.agents_executed} agents successful)"
        )

        return report

    async def _fetch_stock_data(self, symbol: str) -> StockData:
        """Fetch basic stock data with caching."""
        try:
            info = get_ticker_info(symbol)

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
                sector=info.get("sector"),
                industry=info.get("industry"),
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
            sector=stock_data.sector,
            industry=stock_data.industry,
            stock_data=stock_data,
            agent_results=agent_results,
        )

        # Extract typed analysis results
        for result in agent_results:
            if result.status != AgentStatus.COMPLETED:
                continue

            # Core Analysis
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

            # Ecosystem Analysis
            elif result.agent_type == "supply_chain" and "supply_chain_data" in result.data:
                report.supply_chain_analysis = SupplyChainAnalysis(
                    **result.data["supply_chain_data"]
                )
                if report.supply_chain_analysis:
                    report.supply_chain_analysis.score = result.score
                    report.supply_chain_analysis.analysis_summary = result.summary

            elif result.agent_type == "customer" and "customer_data" in result.data:
                report.customer_analysis = CustomerAnalysis(
                    **result.data["customer_data"]
                )
                if report.customer_analysis:
                    report.customer_analysis.score = result.score
                    report.customer_analysis.analysis_summary = result.summary

            elif result.agent_type == "competitive" and "competitive_data" in result.data:
                report.competitive_analysis = CompetitiveAnalysis(
                    **result.data["competitive_data"]
                )
                if report.competitive_analysis:
                    report.competitive_analysis.score = result.score
                    report.competitive_analysis.analysis_summary = result.summary

            # Macro Analysis
            elif result.agent_type == "macro" and "macro_data" in result.data:
                report.macroeconomic_analysis = MacroeconomicData(
                    **result.data["macro_data"]
                )
                if report.macroeconomic_analysis:
                    report.macroeconomic_analysis.score = result.score
                    report.macroeconomic_analysis.analysis_summary = result.summary

            elif result.agent_type == "monetary_policy" and "monetary_data" in result.data:
                report.monetary_policy_analysis = MonetaryPolicyData(
                    **result.data["monetary_data"]
                )
                if report.monetary_policy_analysis:
                    report.monetary_policy_analysis.score = result.score
                    report.monetary_policy_analysis.analysis_summary = result.summary

        return report

    def _calculate_overall_score(self, results: list[AgentResult]) -> float:
        """Calculate weighted overall score from agent results."""
        total_weight = 0.0
        weighted_score = 0.0

        for result in results:
            if result.status == AgentStatus.COMPLETED and result.score is not None:
                weight = self.weights.get(result.agent_type, 0.10)
                weighted_score += result.score * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_score / total_weight
        return 50.0  # Default neutral score

    def _generate_recommendation(
        self, overall_score: float, results: list[AgentResult], report: AnalysisReport
    ) -> str:
        """Generate investment recommendation based on comprehensive analysis."""
        # Check for critical risk factors
        has_high_risk = False
        has_macro_headwinds = False
        has_competitive_threat = False

        for result in results:
            if result.agent_type == "risk" and result.score is not None:
                if result.score < 30:
                    has_high_risk = True

            if result.agent_type == "macro" and result.score is not None:
                if result.score < 40:
                    has_macro_headwinds = True

            if result.agent_type == "competitive" and result.score is not None:
                if result.score < 40:
                    has_competitive_threat = True

        # Generate recommendation with context
        if overall_score >= 75:
            if has_high_risk:
                return "STRONG BUY (with risk management)"
            return "STRONG BUY"
        elif overall_score >= 65:
            if has_high_risk:
                return "BUY (with caution)"
            elif has_macro_headwinds:
                return "BUY (monitor macro conditions)"
            return "BUY"
        elif overall_score >= 55:
            if has_competitive_threat:
                return "HOLD (competitive pressure)"
            return "HOLD"
        elif overall_score >= 45:
            return "REDUCE"
        elif overall_score >= 35:
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

        # Bonus for more comprehensive analysis
        if successful >= 8:
            base_confidence += 5
        elif successful >= 6:
            base_confidence += 3

        return max(0, min(100, base_confidence))

    def _determine_conviction(self, confidence: float, score: float) -> str:
        """Determine conviction level."""
        combined = (confidence + score) / 2

        if combined >= 75:
            return "very_high"
        elif combined >= 60:
            return "high"
        elif combined >= 45:
            return "medium"
        else:
            return "low"

    def _suggest_position_size(self, report: AnalysisReport) -> float:
        """Suggest position size based on conviction and risk."""
        base_size = 0.05  # 5% base position

        # Adjust for overall score
        if report.overall_score:
            if report.overall_score >= 75:
                base_size = 0.08
            elif report.overall_score >= 65:
                base_size = 0.06
            elif report.overall_score >= 55:
                base_size = 0.04
            else:
                base_size = 0.02

        # Adjust for risk
        if report.risk_assessment and report.risk_assessment.score:
            if report.risk_assessment.score < 40:
                base_size *= 0.5  # Reduce for high risk
            elif report.risk_assessment.score > 70:
                base_size *= 1.2  # Increase for low risk

        # Adjust for confidence
        if report.confidence:
            confidence_factor = report.confidence / 100
            base_size *= (0.5 + 0.5 * confidence_factor)

        return min(0.15, max(0.01, base_size))  # Cap at 15%, min 1%

    def _generate_position_rationale(self, report: AnalysisReport) -> str:
        """Generate rationale for position sizing."""
        parts = []

        if report.suggested_position_size:
            size_pct = report.suggested_position_size * 100
            if size_pct >= 7:
                parts.append(f"High conviction position ({size_pct:.1f}%).")
            elif size_pct >= 4:
                parts.append(f"Moderate position ({size_pct:.1f}%).")
            else:
                parts.append(f"Small position ({size_pct:.1f}%).")

        if report.risk_assessment and report.risk_assessment.risk_level:
            parts.append(f"Risk level: {report.risk_assessment.risk_level}.")

        if report.confidence:
            parts.append(f"Analysis confidence: {report.confidence:.0f}%.")

        return " ".join(parts)

    def _generate_investment_thesis(self, report: AnalysisReport) -> str:
        """Generate comprehensive investment thesis."""
        parts = []

        # Valuation perspective
        if report.fundamental_analysis:
            fa = report.fundamental_analysis
            if fa.score and fa.score >= 65:
                parts.append("Fundamentals are attractive")
            elif fa.score and fa.score <= 40:
                parts.append("Fundamental concerns exist")

        # Technical perspective
        if report.technical_analysis:
            ta = report.technical_analysis
            if ta.trend_direction == "bullish":
                parts.append("with bullish technical setup")
            elif ta.trend_direction == "bearish":
                parts.append("but technical trend is weak")

        # Competitive position
        if report.competitive_analysis:
            ca = report.competitive_analysis
            if ca.market_position == "leader":
                parts.append("Company is a market leader")
            elif ca.market_position == "challenger":
                parts.append("Company is a strong challenger")

        # Macro context
        if report.macroeconomic_analysis:
            ma = report.macroeconomic_analysis
            if ma.score and ma.score >= 60:
                parts.append("with supportive macro backdrop")
            elif ma.score and ma.score <= 40:
                parts.append("though macro headwinds exist")

        # Risk/reward
        if report.overall_score:
            if report.overall_score >= 65:
                parts.append("Risk/reward appears favorable.")
            elif report.overall_score <= 45:
                parts.append("Risk/reward is unfavorable.")

        return " ".join(parts) if parts else "Analysis complete."

    def _generate_executive_summary(self, report: AnalysisReport) -> str:
        """Generate executive summary of the analysis."""
        parts = []

        # Company intro
        name = report.company_name or report.symbol
        parts.append(f"Analysis of {name} ({report.symbol})")
        if report.sector:
            parts.append(f"in {report.sector}.")
        else:
            parts.append(".")

        # Current price
        if report.stock_data and report.stock_data.current_price:
            parts.append(f"Current price: ${report.stock_data.current_price:.2f}.")

        # Overall assessment
        if report.overall_score:
            if report.overall_score >= 65:
                parts.append("Overall assessment is POSITIVE.")
            elif report.overall_score <= 35:
                parts.append("Overall assessment is NEGATIVE.")
            else:
                parts.append("Overall assessment is NEUTRAL.")

        # Key agent summaries
        summaries = []
        priority_agents = ["fundamental", "competitive", "macro"]
        for result in report.agent_results:
            if result.agent_type in priority_agents:
                if result.status == AgentStatus.COMPLETED and result.summary:
                    summaries.append(result.summary)

        if summaries:
            parts.append(" ".join(summaries[:2]))

        # Recommendation
        if report.recommendation:
            parts.append(f"Recommendation: {report.recommendation}.")

        if report.conviction_level:
            parts.append(f"Conviction: {report.conviction_level.replace('_', ' ')}.")

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

        # From competitive analysis
        if report.competitive_analysis:
            ca = report.competitive_analysis
            if ca.market_position == "leader":
                strengths.append("Market leadership position")
            if ca.competitive_advantages:
                strengths.extend(ca.competitive_advantages[:2])

        # From supply chain analysis
        if report.supply_chain_analysis:
            sca = report.supply_chain_analysis
            if sca.score and sca.score >= 65:
                strengths.append("Resilient supply chain")
            if sca.supply_chain_opportunities:
                strengths.extend(sca.supply_chain_opportunities[:1])

        # From customer analysis
        if report.customer_analysis:
            cua = report.customer_analysis
            if cua.demand_outlook == "positive":
                strengths.append("Strong customer demand")
            if cua.pricing_power == "strong":
                strengths.append("Strong pricing power")

        # From macro analysis
        if report.macroeconomic_analysis:
            ma = report.macroeconomic_analysis
            if ma.macro_opportunities:
                strengths.extend(ma.macro_opportunities[:1])

        return strengths[:7]  # Return top 7 strengths

    def _extract_risks(self, report: AnalysisReport) -> list[str]:
        """Extract key risks from the analysis."""
        risks = []

        # From risk assessment
        if report.risk_assessment:
            ra = report.risk_assessment
            risks.extend(ra.risk_factors[:3])

        # From competitive analysis
        if report.competitive_analysis:
            ca = report.competitive_analysis
            if ca.emerging_threats:
                risks.extend(ca.emerging_threats[:2])
            if ca.competitive_weaknesses:
                risks.extend(ca.competitive_weaknesses[:1])

        # From supply chain analysis
        if report.supply_chain_analysis:
            sca = report.supply_chain_analysis
            if sca.supply_chain_threats:
                risks.extend(sca.supply_chain_threats[:2])

        # From customer analysis
        if report.customer_analysis:
            cua = report.customer_analysis
            if cua.customer_threats:
                risks.extend(cua.customer_threats[:1])

        # From macro analysis
        if report.macroeconomic_analysis:
            ma = report.macroeconomic_analysis
            if ma.macro_risks:
                risks.extend(ma.macro_risks[:2])

        # From monetary policy
        if report.monetary_policy_analysis:
            mpa = report.monetary_policy_analysis
            if mpa.policy_risks:
                risks.extend(mpa.policy_risks[:1])

        return risks[:7]  # Return top 7 risks

    def _extract_catalysts(self, report: AnalysisReport) -> list[str]:
        """Identify potential catalysts for the stock."""
        catalysts = []

        # Technical catalysts
        if report.technical_analysis:
            ta = report.technical_analysis
            if ta.rsi and ta.rsi < 30:
                catalysts.append("Potential reversal from oversold levels")
            if ta.buy_signals:
                catalysts.append(f"Technical buy signals: {', '.join(ta.buy_signals[:2])}")

        # Fundamental catalysts
        if report.fundamental_analysis:
            fa = report.fundamental_analysis
            if fa.peg_ratio and fa.peg_ratio < 1:
                catalysts.append("Undervalued relative to growth (PEG < 1)")

        # Competitive catalysts
        if report.competitive_analysis:
            ca = report.competitive_analysis
            if ca.strategic_opportunities:
                catalysts.extend(ca.strategic_opportunities[:2])

        # Macro catalysts
        if report.monetary_policy_analysis:
            mpa = report.monetary_policy_analysis
            if mpa.policy_opportunities:
                catalysts.extend(mpa.policy_opportunities[:1])

        # Sentiment catalysts
        if report.sentiment_analysis:
            sa = report.sentiment_analysis
            if sa.target_price and report.stock_data and report.stock_data.current_price:
                upside = (sa.target_price - report.stock_data.current_price) / report.stock_data.current_price
                if upside > 0.15:
                    catalysts.append(f"Analyst target implies {upside:.0%} upside")

        return catalysts[:5]

    def _extract_watchpoints(self, report: AnalysisReport) -> list[str]:
        """Identify key items to monitor."""
        watchpoints = []

        # Macro watchpoints
        if report.macroeconomic_analysis:
            ma = report.macroeconomic_analysis
            if ma.economic_cycle_phase:
                watchpoints.append(f"Economic cycle phase: {ma.economic_cycle_phase}")

        # Monetary policy watchpoints
        if report.monetary_policy_analysis:
            mpa = report.monetary_policy_analysis
            if mpa.rate_direction:
                watchpoints.append(f"Fed policy direction: {mpa.rate_direction}")
            if mpa.yield_curve_status == "inverted":
                watchpoints.append("Inverted yield curve - recession signal")

        # Competitive watchpoints
        if report.competitive_analysis:
            ca = report.competitive_analysis
            if ca.technological_disruption_risk == "high":
                watchpoints.append("High technological disruption risk")

        # Supply chain watchpoints
        if report.supply_chain_analysis:
            sca = report.supply_chain_analysis
            if sca.suppliers_at_risk > 0:
                watchpoints.append(f"{sca.suppliers_at_risk} suppliers at elevated risk")

        # Customer watchpoints
        if report.customer_analysis:
            cua = report.customer_analysis
            if cua.customer_concentration_risk and cua.customer_concentration_risk > 70:
                watchpoints.append("High customer concentration")

        return watchpoints[:5]

    def _generate_short_term_outlook(self, report: AnalysisReport) -> str:
        """Generate 1-3 month outlook."""
        signals = []

        if report.technical_analysis:
            ta = report.technical_analysis
            if ta.trend_direction:
                signals.append(f"Technical trend: {ta.trend_direction}")

        if report.sentiment_analysis:
            sa = report.sentiment_analysis
            if sa.overall_sentiment:
                signals.append(f"Sentiment: {sa.overall_sentiment}")

        if signals:
            return "; ".join(signals)
        return "Neutral"

    def _generate_medium_term_outlook(self, report: AnalysisReport) -> str:
        """Generate 3-12 month outlook."""
        signals = []

        if report.fundamental_analysis and report.fundamental_analysis.score:
            if report.fundamental_analysis.score >= 65:
                signals.append("Fundamentals supportive")
            elif report.fundamental_analysis.score <= 40:
                signals.append("Fundamental concerns")

        if report.macroeconomic_analysis and report.macroeconomic_analysis.score:
            if report.macroeconomic_analysis.score >= 60:
                signals.append("Macro environment favorable")
            elif report.macroeconomic_analysis.score <= 40:
                signals.append("Macro headwinds")

        if signals:
            return "; ".join(signals)
        return "Neutral"

    def _generate_long_term_outlook(self, report: AnalysisReport) -> str:
        """Generate 1-3 year outlook."""
        signals = []

        if report.competitive_analysis:
            ca = report.competitive_analysis
            if ca.market_position in ["leader", "challenger"]:
                signals.append(f"Strong competitive position ({ca.market_position})")
            if ca.barriers_to_entry == "high":
                signals.append("Protected by high barriers")

        if report.customer_analysis:
            cua = report.customer_analysis
            if cua.demand_outlook == "positive":
                signals.append("Positive demand trajectory")

        if signals:
            return "; ".join(signals)
        return "Neutral"

    def add_agent(self, name: str, agent: BaseAgent, weight: float = 0.10) -> None:
        """Add a new servant agent to the master."""
        self.agents[name] = agent
        self.weights[name] = weight
        logger.info(f"Added new agent: {name} with weight {weight}")

    def remove_agent(self, name: str) -> None:
        """Remove a servant agent."""
        if name in self.agents:
            del self.agents[name]
            if name in self.weights:
                del self.weights[name]
            logger.info(f"Removed agent: {name}")

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set custom weights for scoring."""
        self.weights = weights
        logger.info(f"Updated weights: {weights}")

    def get_agent_status(self) -> dict[str, AgentStatus]:
        """Get status of all agents."""
        return {name: agent.status for name, agent in self.agents.items()}

    def get_available_agents(self) -> list[str]:
        """Get list of available agents."""
        return list(self.agents.keys())
