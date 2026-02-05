"""Supply Chain Analysis Agent - Analyzes supplier performance and risks."""

from typing import Any, Optional

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.schemas import SupplierInfo, SupplyChainAnalysis
from src.utils.yfinance_cache import get_ticker_info, get_ticker_history


# Known supplier relationships for major companies (would be expanded with real data sources)
KNOWN_SUPPLIERS = {
    "AAPL": [
        {"name": "Taiwan Semiconductor", "symbol": "TSM", "relationship": "Chip manufacturing"},
        {"name": "Foxconn", "symbol": "HNHPF", "relationship": "Assembly"},
        {"name": "Samsung Electronics", "symbol": "SSNLF", "relationship": "Displays/Memory"},
        {"name": "Qualcomm", "symbol": "QCOM", "relationship": "Modems"},
        {"name": "Broadcom", "symbol": "AVGO", "relationship": "Wireless components"},
        {"name": "Texas Instruments", "symbol": "TXN", "relationship": "Analog chips"},
        {"name": "Corning", "symbol": "GLW", "relationship": "Glass"},
    ],
    "TSLA": [
        {"name": "Panasonic", "symbol": "PCRFY", "relationship": "Battery cells"},
        {"name": "CATL", "symbol": "300750.SZ", "relationship": "Battery cells"},
        {"name": "LG Energy Solution", "symbol": "373220.KS", "relationship": "Battery cells"},
        {"name": "Taiwan Semiconductor", "symbol": "TSM", "relationship": "Chip manufacturing"},
        {"name": "NVIDIA", "symbol": "NVDA", "relationship": "AI chips"},
        {"name": "Samsung SDI", "symbol": "006400.KS", "relationship": "Battery materials"},
    ],
    "MSFT": [
        {"name": "Intel", "symbol": "INTC", "relationship": "Processors"},
        {"name": "AMD", "symbol": "AMD", "relationship": "Processors/GPUs"},
        {"name": "NVIDIA", "symbol": "NVDA", "relationship": "GPUs/AI"},
        {"name": "Taiwan Semiconductor", "symbol": "TSM", "relationship": "Chip manufacturing"},
        {"name": "Samsung", "symbol": "SSNLF", "relationship": "Memory"},
    ],
    "NVDA": [
        {"name": "Taiwan Semiconductor", "symbol": "TSM", "relationship": "Chip manufacturing"},
        {"name": "Samsung Electronics", "symbol": "SSNLF", "relationship": "Memory"},
        {"name": "SK Hynix", "symbol": "000660.KS", "relationship": "HBM memory"},
        {"name": "Micron", "symbol": "MU", "relationship": "Memory"},
        {"name": "ASML", "symbol": "ASML", "relationship": "Lithography equipment"},
    ],
    "AMZN": [
        {"name": "Intel", "symbol": "INTC", "relationship": "Server processors"},
        {"name": "AMD", "symbol": "AMD", "relationship": "Server processors"},
        {"name": "NVIDIA", "symbol": "NVDA", "relationship": "GPUs"},
        {"name": "Packaging Corp", "symbol": "PKG", "relationship": "Packaging"},
        {"name": "UPS", "symbol": "UPS", "relationship": "Logistics"},
        {"name": "FedEx", "symbol": "FDX", "relationship": "Logistics"},
    ],
    "GOOGL": [
        {"name": "Taiwan Semiconductor", "symbol": "TSM", "relationship": "Chip manufacturing"},
        {"name": "Samsung", "symbol": "SSNLF", "relationship": "Hardware"},
        {"name": "NVIDIA", "symbol": "NVDA", "relationship": "AI chips"},
        {"name": "Intel", "symbol": "INTC", "relationship": "Processors"},
        {"name": "Broadcom", "symbol": "AVGO", "relationship": "Networking"},
    ],
}


class SupplyChainAgent(BaseAgent):
    """
    Agent responsible for supply chain analysis.

    Analyzes:
    - Key supplier identification
    - Supplier financial health
    - Supply concentration risk
    - Geographic supply chain risks
    - Cost pressure assessment
    """

    def __init__(self, name: str = "SupplyChainAgent"):
        super().__init__(name=name, agent_type="supply_chain")

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """
        Analyze supply chain for the given stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing supply chain analysis
        """
        self.log_info(f"Analyzing supply chain for {symbol}")

        try:
            # Get company info with caching
            info = get_ticker_info(symbol)
            sector = info.get("sector", "")
            industry = info.get("industry", "")

            # Get known suppliers or identify based on industry
            suppliers = await self._identify_suppliers(symbol, sector, industry)

            # Analyze each supplier
            analyzed_suppliers = []
            for supplier in suppliers:
                supplier_analysis = await self._analyze_supplier(supplier)
                if supplier_analysis:
                    analyzed_suppliers.append(supplier_analysis)

            # Calculate aggregate metrics
            analysis = self._build_analysis(symbol, analyzed_suppliers)

            # Calculate score
            score = self._calculate_score(analysis)

            # Generate summary
            summary = self._generate_summary(analysis, score)

            return {
                "supply_chain_data": analysis.model_dump(),
                "score": score,
                "summary": summary,
            }

        except Exception as e:
            self.log_warning(f"Error in supply chain analysis: {str(e)}")
            return {
                "supply_chain_data": SupplyChainAnalysis(symbol=symbol).model_dump(),
                "score": None,
                "summary": f"Unable to perform supply chain analysis: {str(e)}",
            }

    async def _identify_suppliers(
        self, symbol: str, sector: str, industry: str
    ) -> list[dict]:
        """Identify key suppliers for the company."""
        # Check known suppliers first
        if symbol in KNOWN_SUPPLIERS:
            return KNOWN_SUPPLIERS[symbol]

        # For unknown companies, identify based on sector
        sector_suppliers = self._get_sector_suppliers(sector, industry)
        return sector_suppliers

    def _get_sector_suppliers(self, sector: str, industry: str) -> list[dict]:
        """Get typical suppliers for a sector."""
        sector_map = {
            "Technology": [
                {"name": "Taiwan Semiconductor", "symbol": "TSM", "relationship": "Semiconductors"},
                {"name": "Samsung", "symbol": "SSNLF", "relationship": "Components"},
                {"name": "Intel", "symbol": "INTC", "relationship": "Processors"},
            ],
            "Consumer Cyclical": [
                {"name": "China-based manufacturers", "symbol": None, "relationship": "Manufacturing"},
            ],
            "Healthcare": [
                {"name": "Thermo Fisher", "symbol": "TMO", "relationship": "Lab equipment"},
                {"name": "Danaher", "symbol": "DHR", "relationship": "Diagnostics"},
            ],
            "Financial Services": [
                {"name": "Microsoft", "symbol": "MSFT", "relationship": "Software/Cloud"},
                {"name": "Oracle", "symbol": "ORCL", "relationship": "Database"},
            ],
            "Industrials": [
                {"name": "3M", "symbol": "MMM", "relationship": "Materials"},
                {"name": "Honeywell", "symbol": "HON", "relationship": "Components"},
            ],
        }
        return sector_map.get(sector, [])

    async def _analyze_supplier(self, supplier: dict) -> Optional[SupplierInfo]:
        """Analyze a single supplier's financial health."""
        symbol = supplier.get("symbol")
        name = supplier.get("name")
        relationship = supplier.get("relationship")

        if not symbol:
            # Return basic info for non-public suppliers
            return SupplierInfo(
                name=name,
                relationship=relationship,
                supply_risk="medium",
                risk_factors=["Private company - limited financial visibility"],
            )

        try:
            info = get_ticker_info(symbol)
            hist = get_ticker_history(symbol, period="ytd")

            # Calculate YTD performance
            ytd_performance = None
            if len(hist) > 1:
                ytd_performance = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]

            # Extract financial metrics
            profit_margin = info.get("profitMargins")
            debt_to_equity = info.get("debtToEquity")
            current_ratio = info.get("currentRatio")
            revenue_growth = info.get("revenueGrowth")
            market_cap = info.get("marketCap")

            # Calculate financial health score
            health_score = self._calculate_supplier_health(
                profit_margin, debt_to_equity, current_ratio, revenue_growth
            )

            # Determine supply risk
            supply_risk = self._determine_supply_risk(health_score, debt_to_equity, profit_margin)

            # Identify risk factors
            risk_factors = self._identify_supplier_risks(info, health_score)

            return SupplierInfo(
                symbol=symbol,
                name=name,
                relationship=relationship,
                market_cap=market_cap,
                profit_margin=profit_margin,
                debt_to_equity=debt_to_equity / 100 if debt_to_equity else None,
                current_ratio=current_ratio,
                stock_performance_ytd=ytd_performance,
                revenue_growth=revenue_growth,
                financial_health_score=health_score,
                supply_risk=supply_risk,
                risk_factors=risk_factors,
            )

        except Exception as e:
            logger.warning(f"Error analyzing supplier {name}: {e}")
            return SupplierInfo(
                name=name,
                symbol=symbol,
                relationship=relationship,
                supply_risk="unknown",
                risk_factors=[f"Unable to analyze: {str(e)}"],
            )

    def _calculate_supplier_health(
        self,
        profit_margin: Optional[float],
        debt_to_equity: Optional[float],
        current_ratio: Optional[float],
        revenue_growth: Optional[float],
    ) -> float:
        """Calculate supplier financial health score (0-100)."""
        score = 60.0  # Start at neutral

        # Profit margin (up to +/- 15)
        if profit_margin is not None:
            if profit_margin > 0.20:
                score += 15
            elif profit_margin > 0.10:
                score += 10
            elif profit_margin > 0.05:
                score += 5
            elif profit_margin < 0:
                score -= 15
            elif profit_margin < 0.03:
                score -= 5

        # Debt to equity (up to +/- 15)
        if debt_to_equity is not None:
            de_ratio = debt_to_equity / 100  # Convert from percentage
            if de_ratio < 0.3:
                score += 15
            elif de_ratio < 0.5:
                score += 10
            elif de_ratio < 1.0:
                score += 5
            elif de_ratio > 2.0:
                score -= 15
            elif de_ratio > 1.5:
                score -= 10

        # Current ratio (up to +/- 10)
        if current_ratio is not None:
            if current_ratio > 2.0:
                score += 10
            elif current_ratio > 1.5:
                score += 5
            elif current_ratio < 1.0:
                score -= 10
            elif current_ratio < 1.2:
                score -= 5

        # Revenue growth (up to +/- 10)
        if revenue_growth is not None:
            if revenue_growth > 0.20:
                score += 10
            elif revenue_growth > 0.10:
                score += 5
            elif revenue_growth < -0.10:
                score -= 10
            elif revenue_growth < 0:
                score -= 5

        return max(0, min(100, score))

    def _determine_supply_risk(
        self,
        health_score: float,
        debt_to_equity: Optional[float],
        profit_margin: Optional[float],
    ) -> str:
        """Determine supply risk level."""
        if health_score >= 70:
            return "low"
        elif health_score >= 50:
            return "medium"
        else:
            return "high"

    def _identify_supplier_risks(self, info: dict, health_score: float) -> list[str]:
        """Identify specific risk factors for a supplier."""
        risks = []

        if health_score < 50:
            risks.append("Weak financial health")

        debt_to_equity = info.get("debtToEquity")
        if debt_to_equity and debt_to_equity > 150:
            risks.append("High debt levels")

        profit_margin = info.get("profitMargins")
        if profit_margin and profit_margin < 0:
            risks.append("Negative profitability")

        current_ratio = info.get("currentRatio")
        if current_ratio and current_ratio < 1.0:
            risks.append("Liquidity concerns")

        revenue_growth = info.get("revenueGrowth")
        if revenue_growth and revenue_growth < -0.10:
            risks.append("Declining revenue")

        # Geographic risk
        country = info.get("country", "")
        if country in ["China", "Taiwan"]:
            risks.append(f"Geopolitical risk ({country})")

        return risks

    def _build_analysis(
        self, symbol: str, suppliers: list[SupplierInfo]
    ) -> SupplyChainAnalysis:
        """Build the complete supply chain analysis."""
        analysis = SupplyChainAnalysis(symbol=symbol, key_suppliers=suppliers)

        if not suppliers:
            return analysis

        # Calculate aggregate metrics
        health_scores = [s.financial_health_score for s in suppliers if s.financial_health_score]
        if health_scores:
            analysis.avg_supplier_health = sum(health_scores) / len(health_scores)

        # Count suppliers at risk
        analysis.suppliers_at_risk = sum(
            1 for s in suppliers if s.supply_risk in ["high", "unknown"]
        )

        # Calculate concentration risk (simplified)
        analysis.supplier_concentration_risk = 100 / len(suppliers) if suppliers else 100

        # Identify critical dependencies
        high_risk_suppliers = [s.name for s in suppliers if s.supply_risk == "high"]
        if high_risk_suppliers:
            analysis.critical_dependencies = high_risk_suppliers

        # Geographic risk assessment
        geo_risks = []
        for s in suppliers:
            if s.risk_factors:
                geo_risks.extend([r for r in s.risk_factors if "Geopolitical" in r])

        if len(geo_risks) > len(suppliers) / 2:
            analysis.geographic_risk = "high"
        elif geo_risks:
            analysis.geographic_risk = "medium"
        else:
            analysis.geographic_risk = "low"

        # Supply disruption risk
        if analysis.suppliers_at_risk > len(suppliers) / 2:
            analysis.supply_disruption_risk = "high"
        elif analysis.suppliers_at_risk > 0:
            analysis.supply_disruption_risk = "medium"
        else:
            analysis.supply_disruption_risk = "low"

        # Cost pressure outlook
        declining_suppliers = sum(
            1 for s in suppliers
            if s.revenue_growth and s.revenue_growth < 0
        )
        if declining_suppliers > len(suppliers) / 2:
            analysis.cost_pressure_outlook = "favorable"  # Suppliers struggling = pricing power
        else:
            analysis.cost_pressure_outlook = "neutral"

        # Identify threats and opportunities
        analysis.supply_chain_threats = self._identify_threats(suppliers, analysis)
        analysis.supply_chain_opportunities = self._identify_opportunities(suppliers, analysis)

        # Calculate resilience score
        analysis.supply_chain_resilience = self._calculate_resilience(analysis)

        return analysis

    def _identify_threats(
        self, suppliers: list[SupplierInfo], analysis: SupplyChainAnalysis
    ) -> list[str]:
        """Identify supply chain threats."""
        threats = []

        if analysis.geographic_risk == "high":
            threats.append("High geographic concentration in risk regions")

        if analysis.suppliers_at_risk > 2:
            threats.append(f"{analysis.suppliers_at_risk} suppliers showing financial stress")

        if analysis.supplier_concentration_risk and analysis.supplier_concentration_risk > 30:
            threats.append("High supplier concentration risk")

        # Check for single points of failure
        critical_suppliers = [s for s in suppliers if "Chip" in str(s.relationship) or "manufacturing" in str(s.relationship).lower()]
        if len(critical_suppliers) == 1:
            threats.append(f"Single source dependency: {critical_suppliers[0].name}")

        return threats[:5]

    def _identify_opportunities(
        self, suppliers: list[SupplierInfo], analysis: SupplyChainAnalysis
    ) -> list[str]:
        """Identify supply chain opportunities."""
        opportunities = []

        strong_suppliers = [s for s in suppliers if s.financial_health_score and s.financial_health_score > 70]
        if len(strong_suppliers) > len(suppliers) / 2:
            opportunities.append("Strong supplier ecosystem provides stability")

        if analysis.cost_pressure_outlook == "favorable":
            opportunities.append("Potential for favorable supplier negotiations")

        growing_suppliers = [s for s in suppliers if s.revenue_growth and s.revenue_growth > 0.10]
        if growing_suppliers:
            opportunities.append("Growing suppliers investing in capacity")

        return opportunities[:3]

    def _calculate_resilience(self, analysis: SupplyChainAnalysis) -> float:
        """Calculate supply chain resilience score."""
        score = 60.0

        # Supplier health impact
        if analysis.avg_supplier_health:
            if analysis.avg_supplier_health > 70:
                score += 15
            elif analysis.avg_supplier_health > 60:
                score += 5
            elif analysis.avg_supplier_health < 50:
                score -= 15

        # Concentration risk
        if analysis.supplier_concentration_risk:
            if analysis.supplier_concentration_risk < 20:
                score += 10
            elif analysis.supplier_concentration_risk > 40:
                score -= 10

        # Geographic risk
        if analysis.geographic_risk == "low":
            score += 10
        elif analysis.geographic_risk == "high":
            score -= 15

        # Suppliers at risk
        if analysis.suppliers_at_risk == 0:
            score += 10
        elif analysis.suppliers_at_risk > 2:
            score -= 10

        return max(0, min(100, score))

    def _calculate_score(self, analysis: SupplyChainAnalysis) -> float:
        """Calculate overall supply chain score."""
        if analysis.supply_chain_resilience:
            return analysis.supply_chain_resilience
        return 50.0

    def _generate_summary(self, analysis: SupplyChainAnalysis, score: float) -> str:
        """Generate analysis summary."""
        parts = []

        num_suppliers = len(analysis.key_suppliers)
        parts.append(f"Identified {num_suppliers} key suppliers.")

        if analysis.avg_supplier_health:
            if analysis.avg_supplier_health > 65:
                parts.append("Supplier ecosystem is financially healthy.")
            elif analysis.avg_supplier_health < 50:
                parts.append("Supplier ecosystem shows financial stress.")

        if analysis.suppliers_at_risk > 0:
            parts.append(f"{analysis.suppliers_at_risk} suppliers flagged for elevated risk.")

        if analysis.geographic_risk == "high":
            parts.append("Geographic concentration creates geopolitical risk.")

        if analysis.supply_disruption_risk == "high":
            parts.append("Supply disruption risk is elevated.")
        elif analysis.supply_disruption_risk == "low":
            parts.append("Supply chain appears resilient.")

        return " ".join(parts)
