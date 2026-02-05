"""Customer Analysis Agent - Analyzes customer performance and concentration."""

from typing import Any, Optional

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.schemas import CustomerInfo, CustomerAnalysis
from src.utils.yfinance_cache import get_ticker_info, get_ticker_history


# Known major customers for select companies
KNOWN_CUSTOMERS = {
    "NVDA": [
        {"name": "Microsoft", "symbol": "MSFT", "segment": "Cloud/AI", "revenue_pct": 15},
        {"name": "Meta", "symbol": "META", "segment": "AI/Data Center", "revenue_pct": 10},
        {"name": "Amazon", "symbol": "AMZN", "segment": "Cloud/AI", "revenue_pct": 10},
        {"name": "Google", "symbol": "GOOGL", "segment": "Cloud/AI", "revenue_pct": 8},
        {"name": "Tesla", "symbol": "TSLA", "segment": "Automotive", "revenue_pct": 5},
    ],
    "TSM": [
        {"name": "Apple", "symbol": "AAPL", "segment": "Consumer Electronics", "revenue_pct": 25},
        {"name": "NVIDIA", "symbol": "NVDA", "segment": "AI/GPUs", "revenue_pct": 15},
        {"name": "AMD", "symbol": "AMD", "segment": "CPUs/GPUs", "revenue_pct": 10},
        {"name": "Qualcomm", "symbol": "QCOM", "segment": "Mobile", "revenue_pct": 8},
        {"name": "Broadcom", "symbol": "AVGO", "segment": "Semiconductors", "revenue_pct": 5},
    ],
    "MSFT": [
        {"name": "Enterprise segment", "symbol": None, "segment": "Enterprise", "revenue_pct": 40},
        {"name": "Government/Public Sector", "symbol": None, "segment": "Government", "revenue_pct": 15},
        {"name": "SMB segment", "symbol": None, "segment": "SMB", "revenue_pct": 25},
        {"name": "Consumer segment", "symbol": None, "segment": "Consumer", "revenue_pct": 20},
    ],
    "AAPL": [
        {"name": "Consumer Direct", "symbol": None, "segment": "Consumer", "revenue_pct": 35},
        {"name": "Enterprise/Business", "symbol": None, "segment": "Enterprise", "revenue_pct": 20},
        {"name": "Carrier Partners", "symbol": None, "segment": "Telecom", "revenue_pct": 25},
        {"name": "Retail Partners", "symbol": None, "segment": "Retail", "revenue_pct": 20},
    ],
}


class CustomerAnalysisAgent(BaseAgent):
    """
    Agent responsible for customer analysis.

    Analyzes:
    - Key customer identification
    - Customer concentration risk
    - Customer financial health
    - Demand outlook
    - Pricing power assessment
    """

    def __init__(self, name: str = "CustomerAgent"):
        super().__init__(name=name, agent_type="customer")

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """
        Analyze customer base for the given stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing customer analysis
        """
        self.log_info(f"Analyzing customer base for {symbol}")

        try:
            # Get company info with caching
            info = get_ticker_info(symbol)
            sector = info.get("sector", "")
            industry = info.get("industry", "")

            # Get known customers or estimate based on business model
            customers = await self._identify_customers(symbol, sector, industry, info)

            # Analyze each customer
            analyzed_customers = []
            for customer in customers:
                customer_analysis = await self._analyze_customer(customer)
                if customer_analysis:
                    analyzed_customers.append(customer_analysis)

            # Build complete analysis
            analysis = self._build_analysis(symbol, analyzed_customers, info)

            # Calculate score
            score = self._calculate_score(analysis)

            # Generate summary
            summary = self._generate_summary(analysis, score)

            return {
                "customer_data": analysis.model_dump(),
                "score": score,
                "summary": summary,
            }

        except Exception as e:
            self.log_warning(f"Error in customer analysis: {str(e)}")
            return {
                "customer_data": CustomerAnalysis(symbol=symbol).model_dump(),
                "score": None,
                "summary": f"Unable to perform customer analysis: {str(e)}",
            }

    async def _identify_customers(
        self, symbol: str, sector: str, industry: str, info: dict
    ) -> list[dict]:
        """Identify key customers for the company."""
        # Check known customers first
        if symbol in KNOWN_CUSTOMERS:
            return KNOWN_CUSTOMERS[symbol]

        # Estimate based on business model
        return self._estimate_customer_segments(sector, industry, info)

    def _estimate_customer_segments(
        self, sector: str, industry: str, info: dict
    ) -> list[dict]:
        """Estimate customer segments based on business model."""
        # B2B vs B2C estimation based on industry
        b2b_industries = [
            "Semiconductors", "Software—Infrastructure", "Information Technology Services",
            "Industrial", "Aerospace & Defense", "Business Equipment & Supplies"
        ]

        b2c_industries = [
            "Consumer Electronics", "Retail", "Restaurants", "Entertainment",
            "Apparel", "Household Products"
        ]

        if industry in b2b_industries:
            return [
                {"name": "Enterprise Customers", "symbol": None, "segment": "Enterprise", "revenue_pct": 50},
                {"name": "Mid-Market", "symbol": None, "segment": "Mid-Market", "revenue_pct": 30},
                {"name": "SMB", "symbol": None, "segment": "SMB", "revenue_pct": 20},
            ]
        elif industry in b2c_industries:
            return [
                {"name": "Direct Consumers", "symbol": None, "segment": "Consumer", "revenue_pct": 60},
                {"name": "Retail Partners", "symbol": None, "segment": "Retail", "revenue_pct": 25},
                {"name": "Online/Digital", "symbol": None, "segment": "Digital", "revenue_pct": 15},
            ]
        else:
            return [
                {"name": "Primary Segment", "symbol": None, "segment": "Primary", "revenue_pct": 50},
                {"name": "Secondary Segment", "symbol": None, "segment": "Secondary", "revenue_pct": 30},
                {"name": "Other", "symbol": None, "segment": "Other", "revenue_pct": 20},
            ]

    async def _analyze_customer(self, customer: dict) -> Optional[CustomerInfo]:
        """Analyze a single customer's health."""
        symbol = customer.get("symbol")
        name = customer.get("name")
        segment = customer.get("segment")
        revenue_pct = customer.get("revenue_pct", 0)

        if not symbol:
            # Return segment-based info for non-public customers
            return CustomerInfo(
                name=name,
                segment=segment,
                revenue_contribution=revenue_pct / 100 if revenue_pct else None,
                business_outlook="neutral",
                churn_risk="low",
            )

        try:
            info = get_ticker_info(symbol)
            hist = get_ticker_history(symbol, period="ytd")

            # Calculate YTD performance
            ytd_performance = None
            if len(hist) > 1:
                ytd_performance = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]

            # Extract metrics
            market_cap = info.get("marketCap")
            revenue_growth = info.get("revenueGrowth")
            profit_margin = info.get("profitMargins")

            # Calculate health score
            health_score = self._calculate_customer_health(
                revenue_growth, profit_margin, ytd_performance
            )

            # Determine business outlook
            outlook = self._determine_outlook(revenue_growth, ytd_performance)

            # Determine churn risk
            churn_risk = self._determine_churn_risk(health_score, revenue_growth)

            return CustomerInfo(
                symbol=symbol,
                name=name,
                segment=segment,
                revenue_contribution=revenue_pct / 100 if revenue_pct else None,
                market_cap=market_cap,
                revenue_growth=revenue_growth,
                profit_margin=profit_margin,
                stock_performance_ytd=ytd_performance,
                financial_health_score=health_score,
                business_outlook=outlook,
                churn_risk=churn_risk,
            )

        except Exception as e:
            logger.warning(f"Error analyzing customer {name}: {e}")
            return CustomerInfo(
                name=name,
                segment=segment,
                revenue_contribution=revenue_pct / 100 if revenue_pct else None,
                business_outlook="neutral",
                churn_risk="unknown",
            )

    def _calculate_customer_health(
        self,
        revenue_growth: Optional[float],
        profit_margin: Optional[float],
        ytd_performance: Optional[float],
    ) -> float:
        """Calculate customer financial health score."""
        score = 60.0

        if revenue_growth is not None:
            if revenue_growth > 0.20:
                score += 15
            elif revenue_growth > 0.10:
                score += 10
            elif revenue_growth > 0:
                score += 5
            elif revenue_growth < -0.10:
                score -= 15
            elif revenue_growth < 0:
                score -= 5

        if profit_margin is not None:
            if profit_margin > 0.20:
                score += 10
            elif profit_margin > 0.10:
                score += 5
            elif profit_margin < 0:
                score -= 15

        if ytd_performance is not None:
            if ytd_performance > 0.20:
                score += 10
            elif ytd_performance > 0:
                score += 5
            elif ytd_performance < -0.20:
                score -= 10

        return max(0, min(100, score))

    def _determine_outlook(
        self, revenue_growth: Optional[float], ytd_performance: Optional[float]
    ) -> str:
        """Determine customer business outlook."""
        positive_signals = 0
        negative_signals = 0

        if revenue_growth is not None:
            if revenue_growth > 0.10:
                positive_signals += 1
            elif revenue_growth < 0:
                negative_signals += 1

        if ytd_performance is not None:
            if ytd_performance > 0.10:
                positive_signals += 1
            elif ytd_performance < -0.10:
                negative_signals += 1

        if positive_signals > negative_signals:
            return "positive"
        elif negative_signals > positive_signals:
            return "negative"
        return "neutral"

    def _determine_churn_risk(
        self, health_score: float, revenue_growth: Optional[float]
    ) -> str:
        """Determine customer churn risk."""
        if health_score < 40:
            return "high"
        elif health_score < 55:
            return "medium"
        return "low"

    def _build_analysis(
        self, symbol: str, customers: list[CustomerInfo], info: dict
    ) -> CustomerAnalysis:
        """Build complete customer analysis."""
        analysis = CustomerAnalysis(symbol=symbol, key_customers=customers)

        if not customers:
            return analysis

        # Calculate concentration risk
        revenue_contributions = [
            c.revenue_contribution for c in customers
            if c.revenue_contribution is not None
        ]
        if revenue_contributions:
            top_5 = sorted(revenue_contributions, reverse=True)[:5]
            analysis.revenue_concentration_top5 = sum(top_5)

            # Concentration risk score (higher concentration = higher risk)
            if analysis.revenue_concentration_top5 > 0.6:
                analysis.customer_concentration_risk = 80
            elif analysis.revenue_concentration_top5 > 0.4:
                analysis.customer_concentration_risk = 60
            else:
                analysis.customer_concentration_risk = 40

        # Build segment breakdown
        segment_revenue = {}
        for c in customers:
            if c.segment and c.revenue_contribution:
                segment_revenue[c.segment] = segment_revenue.get(c.segment, 0) + c.revenue_contribution
        analysis.segment_breakdown = segment_revenue

        # Find fastest growing segment
        segment_growth = {}
        for c in customers:
            if c.segment and c.revenue_growth is not None:
                if c.segment not in segment_growth:
                    segment_growth[c.segment] = []
                segment_growth[c.segment].append(c.revenue_growth)

        if segment_growth:
            avg_growth = {seg: sum(g)/len(g) for seg, g in segment_growth.items()}
            analysis.fastest_growing_segment = max(avg_growth, key=avg_growth.get)
            analysis.declining_segments = [seg for seg, g in avg_growth.items() if g < 0]

        # Average customer health
        health_scores = [c.financial_health_score for c in customers if c.financial_health_score]
        if health_scores:
            analysis.avg_customer_health = sum(health_scores) / len(health_scores)

        # Count at-risk customers
        analysis.customers_at_risk = sum(
            1 for c in customers if c.churn_risk in ["high", "unknown"]
        )

        # Determine demand outlook
        positive_outlooks = sum(1 for c in customers if c.business_outlook == "positive")
        negative_outlooks = sum(1 for c in customers if c.business_outlook == "negative")

        if positive_outlooks > negative_outlooks + 1:
            analysis.demand_outlook = "positive"
        elif negative_outlooks > positive_outlooks + 1:
            analysis.demand_outlook = "negative"
        else:
            analysis.demand_outlook = "neutral"

        # Assess pricing power
        analysis.pricing_power = self._assess_pricing_power(analysis, customers)

        # Retention outlook
        if analysis.customers_at_risk == 0:
            analysis.customer_retention_outlook = "stable"
        elif analysis.customers_at_risk <= 1:
            analysis.customer_retention_outlook = "moderate"
        else:
            analysis.customer_retention_outlook = "concerning"

        # Identify threats and opportunities
        analysis.customer_threats = self._identify_threats(analysis, customers)
        analysis.customer_opportunities = self._identify_opportunities(analysis, customers)

        return analysis

    def _assess_pricing_power(
        self, analysis: CustomerAnalysis, customers: list[CustomerInfo]
    ) -> str:
        """Assess company's pricing power with customers."""
        # Factors that increase pricing power:
        # - Low customer concentration
        # - Healthy customer base
        # - Growing demand

        score = 0

        if analysis.customer_concentration_risk:
            if analysis.customer_concentration_risk < 50:
                score += 1
            elif analysis.customer_concentration_risk > 70:
                score -= 1

        if analysis.avg_customer_health:
            if analysis.avg_customer_health > 65:
                score += 1
            elif analysis.avg_customer_health < 50:
                score -= 1

        if analysis.demand_outlook == "positive":
            score += 1
        elif analysis.demand_outlook == "negative":
            score -= 1

        if score >= 2:
            return "strong"
        elif score <= -2:
            return "weak"
        return "moderate"

    def _identify_threats(
        self, analysis: CustomerAnalysis, customers: list[CustomerInfo]
    ) -> list[str]:
        """Identify customer-related threats."""
        threats = []

        if analysis.customer_concentration_risk and analysis.customer_concentration_risk > 70:
            threats.append("High customer concentration risk")

        if analysis.customers_at_risk > 1:
            threats.append(f"{analysis.customers_at_risk} major customers showing stress")

        if analysis.demand_outlook == "negative":
            threats.append("Weakening demand from customer base")

        if analysis.declining_segments:
            threats.append(f"Declining segments: {', '.join(analysis.declining_segments[:2])}")

        if analysis.pricing_power == "weak":
            threats.append("Limited pricing power with customers")

        return threats[:4]

    def _identify_opportunities(
        self, analysis: CustomerAnalysis, customers: list[CustomerInfo]
    ) -> list[str]:
        """Identify customer-related opportunities."""
        opportunities = []

        if analysis.fastest_growing_segment:
            opportunities.append(f"Strong growth in {analysis.fastest_growing_segment} segment")

        if analysis.demand_outlook == "positive":
            opportunities.append("Healthy demand trends from customer base")

        if analysis.pricing_power == "strong":
            opportunities.append("Strong pricing power enables margin expansion")

        healthy_customers = sum(
            1 for c in customers
            if c.financial_health_score and c.financial_health_score > 70
        )
        if healthy_customers > len(customers) / 2:
            opportunities.append("Financially healthy customer base")

        return opportunities[:3]

    def _calculate_score(self, analysis: CustomerAnalysis) -> float:
        """Calculate overall customer score."""
        score = 60.0

        # Customer health
        if analysis.avg_customer_health:
            if analysis.avg_customer_health > 70:
                score += 15
            elif analysis.avg_customer_health > 60:
                score += 5
            elif analysis.avg_customer_health < 50:
                score -= 15

        # Concentration risk
        if analysis.customer_concentration_risk:
            if analysis.customer_concentration_risk < 50:
                score += 10
            elif analysis.customer_concentration_risk > 70:
                score -= 15

        # Demand outlook
        if analysis.demand_outlook == "positive":
            score += 10
        elif analysis.demand_outlook == "negative":
            score -= 10

        # Pricing power
        if analysis.pricing_power == "strong":
            score += 10
        elif analysis.pricing_power == "weak":
            score -= 10

        # Customers at risk
        if analysis.customers_at_risk > 2:
            score -= 10
        elif analysis.customers_at_risk == 0:
            score += 5

        return max(0, min(100, score))

    def _generate_summary(self, analysis: CustomerAnalysis, score: float) -> str:
        """Generate analysis summary."""
        parts = []

        num_customers = len(analysis.key_customers)
        parts.append(f"Analyzed {num_customers} key customers/segments.")

        if analysis.demand_outlook:
            parts.append(f"Demand outlook is {analysis.demand_outlook}.")

        if analysis.customer_concentration_risk:
            if analysis.customer_concentration_risk > 70:
                parts.append("High customer concentration presents risk.")
            elif analysis.customer_concentration_risk < 50:
                parts.append("Diversified customer base reduces risk.")

        if analysis.pricing_power:
            parts.append(f"Pricing power is {analysis.pricing_power}.")

        if analysis.fastest_growing_segment:
            parts.append(f"Strongest growth in {analysis.fastest_growing_segment}.")

        return " ".join(parts)
