"""Competitive Analysis Agent - Analyzes competitive landscape and positioning."""

from typing import Any, Optional

import yfinance as yf
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.schemas import CompetitorInfo, CompetitiveAnalysis


# Industry peer groups
INDUSTRY_PEERS = {
    "Semiconductors": ["NVDA", "AMD", "INTC", "QCOM", "AVGO", "TXN", "MU", "AMAT", "LRCX", "ASML"],
    "Software—Infrastructure": ["MSFT", "ORCL", "CRM", "NOW", "ADBE", "INTU", "IBM"],
    "Internet Content & Information": ["GOOGL", "META", "SNAP", "PINS", "TWTR"],
    "Consumer Electronics": ["AAPL", "SONY", "SSNLF", "HPQ", "DELL"],
    "Auto Manufacturers": ["TSLA", "F", "GM", "TM", "HMC", "RIVN", "LCID"],
    "Internet Retail": ["AMZN", "BABA", "JD", "EBAY", "ETSY", "SHOP"],
    "Drug Manufacturers": ["JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "GILD"],
    "Banks—Diversified": ["JPM", "BAC", "WFC", "C", "GS", "MS"],
    "Aerospace & Defense": ["BA", "LMT", "RTX", "NOC", "GD"],
}


class CompetitiveAnalysisAgent(BaseAgent):
    """
    Agent responsible for competitive landscape analysis.

    Analyzes:
    - Key competitors identification
    - Market position assessment
    - Comparative financial metrics
    - Porter's Five Forces
    - Competitive advantages and threats
    """

    def __init__(self, name: str = "CompetitiveAgent"):
        super().__init__(name=name, agent_type="competitive")

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """
        Analyze competitive landscape for the given stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing competitive analysis
        """
        self.log_info(f"Analyzing competitive landscape for {symbol}")

        try:
            # Get company info
            ticker = yf.Ticker(symbol)
            info = ticker.info
            industry = info.get("industry", "")
            sector = info.get("sector", "")

            # Identify competitors
            competitors = await self._identify_competitors(symbol, industry, sector)

            # Get company metrics for comparison
            company_metrics = self._extract_company_metrics(info, ticker)

            # Analyze each competitor
            analyzed_competitors = []
            all_metrics = [company_metrics]  # Include target company

            for comp_symbol in competitors:
                if comp_symbol != symbol:
                    competitor_analysis = await self._analyze_competitor(comp_symbol)
                    if competitor_analysis:
                        analyzed_competitors.append(competitor_analysis)
                        all_metrics.append({
                            "symbol": comp_symbol,
                            "revenue": competitor_analysis.revenue,
                            "profit_margin": competitor_analysis.profit_margin,
                            "revenue_growth": competitor_analysis.revenue_growth,
                            "pe_ratio": competitor_analysis.pe_ratio,
                            "market_cap": competitor_analysis.market_cap,
                        })

            # Build complete analysis
            analysis = self._build_analysis(
                symbol, info, company_metrics, analyzed_competitors, all_metrics
            )

            # Calculate score
            score = self._calculate_score(analysis)

            # Generate summary
            summary = self._generate_summary(analysis, score)

            return {
                "competitive_data": analysis.model_dump(),
                "score": score,
                "summary": summary,
            }

        except Exception as e:
            self.log_warning(f"Error in competitive analysis: {str(e)}")
            return {
                "competitive_data": CompetitiveAnalysis(symbol=symbol).model_dump(),
                "score": None,
                "summary": f"Unable to perform competitive analysis: {str(e)}",
            }

    async def _identify_competitors(
        self, symbol: str, industry: str, sector: str
    ) -> list[str]:
        """Identify key competitors."""
        # Check industry peer groups
        for ind, peers in INDUSTRY_PEERS.items():
            if ind.lower() in industry.lower() or industry.lower() in ind.lower():
                return peers[:8]  # Return top 8 peers

        # Try to find peers by sector
        sector_peers = {
            "Technology": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD"],
            "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
            "Financial Services": ["JPM", "BAC", "GS", "MS", "BRK-B"],
            "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
            "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA"],
            "Industrials": ["HON", "UPS", "CAT", "GE", "BA"],
            "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
        }

        return sector_peers.get(sector, [])[:8]

    def _extract_company_metrics(self, info: dict, ticker) -> dict:
        """Extract key metrics for the target company."""
        hist = ticker.history(period="1mo")
        perf_1m = None
        if len(hist) > 1:
            perf_1m = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]

        hist_ytd = ticker.history(period="ytd")
        perf_ytd = None
        if len(hist_ytd) > 1:
            perf_ytd = (hist_ytd['Close'].iloc[-1] - hist_ytd['Close'].iloc[0]) / hist_ytd['Close'].iloc[0]

        return {
            "symbol": info.get("symbol"),
            "revenue": info.get("totalRevenue"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "pe_ratio": info.get("trailingPE"),
            "market_cap": info.get("marketCap"),
            "performance_1m": perf_1m,
            "performance_ytd": perf_ytd,
        }

    async def _analyze_competitor(self, symbol: str) -> Optional[CompetitorInfo]:
        """Analyze a single competitor."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get performance data
            hist_1m = ticker.history(period="1mo")
            perf_1m = None
            if len(hist_1m) > 1:
                perf_1m = (hist_1m['Close'].iloc[-1] - hist_1m['Close'].iloc[0]) / hist_1m['Close'].iloc[0]

            hist_ytd = ticker.history(period="ytd")
            perf_ytd = None
            if len(hist_ytd) > 1:
                perf_ytd = (hist_ytd['Close'].iloc[-1] - hist_ytd['Close'].iloc[0]) / hist_ytd['Close'].iloc[0]

            return CompetitorInfo(
                symbol=symbol,
                name=info.get("longName") or info.get("shortName") or symbol,
                market_cap=info.get("marketCap"),
                revenue=info.get("totalRevenue"),
                revenue_growth=info.get("revenueGrowth"),
                profit_margin=info.get("profitMargins"),
                pe_ratio=info.get("trailingPE"),
                stock_performance_1m=perf_1m,
                stock_performance_ytd=perf_ytd,
            )

        except Exception as e:
            logger.warning(f"Error analyzing competitor {symbol}: {e}")
            return None

    def _build_analysis(
        self,
        symbol: str,
        info: dict,
        company_metrics: dict,
        competitors: list[CompetitorInfo],
        all_metrics: list[dict],
    ) -> CompetitiveAnalysis:
        """Build complete competitive analysis."""
        analysis = CompetitiveAnalysis(
            symbol=symbol,
            industry=info.get("industry"),
            key_competitors=competitors,
        )

        # Calculate rankings
        analysis.revenue_rank = self._calculate_rank(
            all_metrics, "revenue", company_metrics["symbol"], higher_is_better=True
        )
        analysis.margin_rank = self._calculate_rank(
            all_metrics, "profit_margin", company_metrics["symbol"], higher_is_better=True
        )
        analysis.growth_rank = self._calculate_rank(
            all_metrics, "revenue_growth", company_metrics["symbol"], higher_is_better=True
        )
        analysis.valuation_rank = self._calculate_rank(
            all_metrics, "pe_ratio", company_metrics["symbol"], higher_is_better=False
        )

        # Determine market position
        analysis.market_position = self._determine_market_position(
            company_metrics, all_metrics
        )

        # Estimate market share based on revenue
        total_revenue = sum(m.get("revenue") or 0 for m in all_metrics)
        if total_revenue > 0 and company_metrics.get("revenue"):
            analysis.estimated_market_share = (company_metrics["revenue"] / total_revenue) * 100

        # Assess competitive intensity
        analysis.competitive_intensity = self._assess_competitive_intensity(
            competitors, company_metrics
        )

        # Calculate industry growth rate (average of competitors)
        growth_rates = [m.get("revenue_growth") for m in all_metrics if m.get("revenue_growth")]
        if growth_rates:
            analysis.industry_growth_rate = sum(growth_rates) / len(growth_rates)

        # Porter's Five Forces (simplified assessment)
        analysis.supplier_power = self._assess_supplier_power(info)
        analysis.buyer_power = self._assess_buyer_power(info)
        analysis.threat_of_substitutes = self._assess_substitutes_threat(info)
        analysis.threat_of_new_entrants = self._assess_entry_threat(info, analysis)
        analysis.competitive_rivalry = analysis.competitive_intensity

        # Barriers to entry assessment
        analysis.barriers_to_entry = self._assess_barriers_to_entry(info, company_metrics)

        # Tech disruption risk
        analysis.technological_disruption_risk = self._assess_tech_disruption(info)

        # Identify competitive advantages and weaknesses
        analysis.competitive_advantages = self._identify_advantages(
            company_metrics, all_metrics, analysis
        )
        analysis.competitive_weaknesses = self._identify_weaknesses(
            company_metrics, all_metrics, analysis
        )
        analysis.emerging_threats = self._identify_threats(competitors, analysis)
        analysis.strategic_opportunities = self._identify_opportunities(analysis)

        # Assess threat level for each competitor
        for comp in analysis.key_competitors:
            comp.threat_level = self._assess_competitor_threat(comp, company_metrics)

        return analysis

    def _calculate_rank(
        self, metrics: list[dict], field: str, target_symbol: str, higher_is_better: bool
    ) -> Optional[int]:
        """Calculate company's rank among competitors for a metric."""
        values = [(m.get("symbol"), m.get(field)) for m in metrics if m.get(field) is not None]
        if not values:
            return None

        values.sort(key=lambda x: x[1], reverse=higher_is_better)
        for i, (sym, _) in enumerate(values, 1):
            if sym == target_symbol:
                return i
        return None

    def _determine_market_position(
        self, company_metrics: dict, all_metrics: list[dict]
    ) -> str:
        """Determine company's market position."""
        market_caps = [(m.get("symbol"), m.get("market_cap") or 0) for m in all_metrics]
        market_caps.sort(key=lambda x: x[1], reverse=True)

        position = None
        for i, (sym, _) in enumerate(market_caps):
            if sym == company_metrics["symbol"]:
                position = i + 1
                break

        if position == 1:
            return "leader"
        elif position and position <= 3:
            return "challenger"
        elif position and position <= len(market_caps) // 2:
            return "follower"
        else:
            return "niche"

    def _assess_competitive_intensity(
        self, competitors: list[CompetitorInfo], company_metrics: dict
    ) -> str:
        """Assess competitive intensity in the industry."""
        if len(competitors) < 3:
            return "low"
        elif len(competitors) < 6:
            return "medium"
        else:
            return "high"

    def _assess_supplier_power(self, info: dict) -> str:
        """Assess supplier bargaining power."""
        industry = info.get("industry", "").lower()

        # Industries with high supplier power
        high_power = ["semiconductor", "auto", "aerospace", "pharmaceutical"]
        if any(ind in industry for ind in high_power):
            return "high"
        return "medium"

    def _assess_buyer_power(self, info: dict) -> str:
        """Assess buyer bargaining power."""
        industry = info.get("industry", "").lower()

        # B2B industries typically have higher buyer power
        b2b = ["software", "enterprise", "industrial", "equipment"]
        if any(ind in industry for ind in b2b):
            return "high"
        return "medium"

    def _assess_substitutes_threat(self, info: dict) -> str:
        """Assess threat of substitutes."""
        industry = info.get("industry", "").lower()

        # Industries with high substitute threat
        high_threat = ["retail", "restaurant", "media", "entertainment"]
        if any(ind in industry for ind in high_threat):
            return "high"
        return "medium"

    def _assess_entry_threat(self, info: dict, analysis: CompetitiveAnalysis) -> str:
        """Assess threat of new entrants."""
        if analysis.barriers_to_entry == "high":
            return "low"
        elif analysis.barriers_to_entry == "medium":
            return "medium"
        return "high"

    def _assess_barriers_to_entry(self, info: dict, company_metrics: dict) -> str:
        """Assess barriers to entry in the industry."""
        market_cap = company_metrics.get("market_cap") or 0
        industry = info.get("industry", "").lower()

        # Capital-intensive industries
        high_capital = ["semiconductor", "telecom", "utilities", "aerospace", "auto"]
        if any(ind in industry for ind in high_capital):
            return "high"

        # Technology/network effects
        network_effects = ["software", "social", "platform"]
        if any(ind in industry for ind in network_effects):
            return "high"

        if market_cap > 100e9:  # $100B+ suggests significant barriers
            return "high"
        elif market_cap > 10e9:
            return "medium"
        return "low"

    def _assess_tech_disruption(self, info: dict) -> str:
        """Assess technological disruption risk."""
        industry = info.get("industry", "").lower()

        # High disruption risk industries
        high_risk = ["retail", "media", "bank", "auto", "energy"]
        if any(ind in industry for ind in high_risk):
            return "high"

        # Low disruption risk
        low_risk = ["utility", "defense", "healthcare"]
        if any(ind in industry for ind in low_risk):
            return "low"

        return "medium"

    def _identify_advantages(
        self, company: dict, all_metrics: list[dict], analysis: CompetitiveAnalysis
    ) -> list[str]:
        """Identify competitive advantages."""
        advantages = []

        if analysis.market_position == "leader":
            advantages.append("Market leadership position")

        if analysis.revenue_rank == 1:
            advantages.append("Largest revenue in peer group")

        if analysis.margin_rank and analysis.margin_rank <= 2:
            advantages.append("Superior profitability vs peers")

        if analysis.growth_rank and analysis.growth_rank <= 2:
            advantages.append("Faster growth than competitors")

        if analysis.barriers_to_entry == "high":
            advantages.append("High barriers to entry protect market")

        if company.get("profit_margin") and company["profit_margin"] > 0.20:
            advantages.append("Strong pricing power (>20% margins)")

        return advantages[:5]

    def _identify_weaknesses(
        self, company: dict, all_metrics: list[dict], analysis: CompetitiveAnalysis
    ) -> list[str]:
        """Identify competitive weaknesses."""
        weaknesses = []

        num_peers = len(all_metrics)

        if analysis.revenue_rank and analysis.revenue_rank > num_peers // 2:
            weaknesses.append("Below-average scale vs competitors")

        if analysis.margin_rank and analysis.margin_rank > num_peers // 2:
            weaknesses.append("Lower margins than competitors")

        if analysis.growth_rank and analysis.growth_rank > num_peers // 2:
            weaknesses.append("Slower growth vs industry peers")

        if analysis.valuation_rank and analysis.valuation_rank > num_peers // 2:
            weaknesses.append("Premium valuation vs peers")

        if analysis.competitive_intensity == "high":
            weaknesses.append("Intense competitive pressure")

        return weaknesses[:4]

    def _identify_threats(
        self, competitors: list[CompetitorInfo], analysis: CompetitiveAnalysis
    ) -> list[str]:
        """Identify emerging competitive threats."""
        threats = []

        # Fast-growing competitors
        fast_growers = [
            c for c in competitors
            if c.revenue_growth and c.revenue_growth > 0.20
        ]
        if fast_growers:
            names = [c.name for c in fast_growers[:2]]
            threats.append(f"Fast-growing competitors: {', '.join(names)}")

        # Strong recent performers
        strong_performers = [
            c for c in competitors
            if c.stock_performance_ytd and c.stock_performance_ytd > 0.30
        ]
        if strong_performers:
            threats.append("Competitors gaining market favor")

        if analysis.technological_disruption_risk == "high":
            threats.append("Industry faces technological disruption risk")

        if analysis.threat_of_new_entrants == "high":
            threats.append("Low barriers may attract new entrants")

        return threats[:4]

    def _identify_opportunities(self, analysis: CompetitiveAnalysis) -> list[str]:
        """Identify strategic opportunities."""
        opportunities = []

        if analysis.industry_growth_rate and analysis.industry_growth_rate > 0.10:
            opportunities.append(f"Industry growing at {analysis.industry_growth_rate:.0%}")

        if analysis.market_position == "leader":
            opportunities.append("Market leadership enables pricing power")

        if analysis.barriers_to_entry == "high":
            opportunities.append("High barriers protect from new competition")

        if analysis.growth_rank and analysis.growth_rank <= 2:
            opportunities.append("Gaining market share vs competitors")

        return opportunities[:3]

    def _assess_competitor_threat(
        self, competitor: CompetitorInfo, company_metrics: dict
    ) -> str:
        """Assess threat level from a specific competitor."""
        threat_score = 0

        # Size comparison
        if competitor.market_cap and company_metrics.get("market_cap"):
            if competitor.market_cap > company_metrics["market_cap"]:
                threat_score += 1

        # Growth comparison
        if competitor.revenue_growth and company_metrics.get("revenue_growth"):
            if competitor.revenue_growth > company_metrics["revenue_growth"]:
                threat_score += 1

        # Margin comparison
        if competitor.profit_margin and company_metrics.get("profit_margin"):
            if competitor.profit_margin > company_metrics["profit_margin"]:
                threat_score += 1

        # Stock performance
        if competitor.stock_performance_ytd and competitor.stock_performance_ytd > 0.20:
            threat_score += 1

        if threat_score >= 3:
            return "high"
        elif threat_score >= 2:
            return "medium"
        return "low"

    def _calculate_score(self, analysis: CompetitiveAnalysis) -> float:
        """Calculate competitive position score."""
        score = 60.0

        # Market position
        position_scores = {"leader": 20, "challenger": 10, "follower": 0, "niche": -5}
        score += position_scores.get(analysis.market_position, 0)

        # Rankings (lower is better)
        for rank, weight in [
            (analysis.revenue_rank, 3),
            (analysis.margin_rank, 3),
            (analysis.growth_rank, 4),
        ]:
            if rank:
                if rank == 1:
                    score += weight * 3
                elif rank == 2:
                    score += weight * 2
                elif rank == 3:
                    score += weight
                elif rank > 5:
                    score -= weight

        # Competitive intensity
        if analysis.competitive_intensity == "low":
            score += 10
        elif analysis.competitive_intensity == "high":
            score -= 5

        # Barriers to entry
        if analysis.barriers_to_entry == "high":
            score += 10
        elif analysis.barriers_to_entry == "low":
            score -= 5

        # Tech disruption risk
        if analysis.technological_disruption_risk == "high":
            score -= 10
        elif analysis.technological_disruption_risk == "low":
            score += 5

        return max(0, min(100, score))

    def _generate_summary(self, analysis: CompetitiveAnalysis, score: float) -> str:
        """Generate analysis summary."""
        parts = []

        if analysis.market_position:
            parts.append(f"Company is a market {analysis.market_position}.")

        if analysis.key_competitors:
            parts.append(f"Analyzed {len(analysis.key_competitors)} key competitors.")

        rankings = []
        if analysis.revenue_rank:
            rankings.append(f"#{analysis.revenue_rank} in revenue")
        if analysis.growth_rank:
            rankings.append(f"#{analysis.growth_rank} in growth")
        if rankings:
            parts.append(f"Ranks {', '.join(rankings)}.")

        if analysis.competitive_intensity:
            parts.append(f"Competitive intensity is {analysis.competitive_intensity}.")

        if analysis.barriers_to_entry == "high":
            parts.append("High barriers protect market position.")

        if analysis.competitive_advantages:
            parts.append(f"Key advantage: {analysis.competitive_advantages[0]}.")

        return " ".join(parts)
