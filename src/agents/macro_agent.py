"""Macroeconomic Analysis Agent - Analyzes macro trends and their impact."""

from typing import Any, Optional

import yfinance as yf
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.schemas import MacroeconomicData


# Sector sensitivity to economic factors
SECTOR_SENSITIVITY = {
    "Technology": {
        "gdp": "high",
        "interest_rates": "high",
        "consumer_spending": "medium",
        "business_investment": "high",
    },
    "Consumer Cyclical": {
        "gdp": "high",
        "interest_rates": "medium",
        "consumer_spending": "very_high",
        "employment": "high",
    },
    "Financial Services": {
        "gdp": "high",
        "interest_rates": "very_high",
        "credit_conditions": "very_high",
        "housing": "high",
    },
    "Healthcare": {
        "gdp": "low",
        "interest_rates": "low",
        "demographics": "high",
        "regulation": "high",
    },
    "Consumer Defensive": {
        "gdp": "low",
        "interest_rates": "low",
        "consumer_spending": "low",
        "inflation": "medium",
    },
    "Energy": {
        "gdp": "medium",
        "interest_rates": "low",
        "oil_prices": "very_high",
        "global_demand": "high",
    },
    "Industrials": {
        "gdp": "high",
        "interest_rates": "medium",
        "business_investment": "very_high",
        "trade": "high",
    },
    "Real Estate": {
        "gdp": "medium",
        "interest_rates": "very_high",
        "housing": "very_high",
        "employment": "medium",
    },
    "Utilities": {
        "gdp": "low",
        "interest_rates": "high",
        "regulation": "high",
        "weather": "medium",
    },
    "Communication Services": {
        "gdp": "medium",
        "interest_rates": "medium",
        "consumer_spending": "medium",
        "advertising": "high",
    },
    "Basic Materials": {
        "gdp": "high",
        "interest_rates": "medium",
        "global_demand": "high",
        "commodities": "very_high",
    },
}


class MacroeconomicAgent(BaseAgent):
    """
    Agent responsible for macroeconomic analysis.

    Analyzes:
    - GDP growth and trends
    - Employment and labor market
    - Consumer and business confidence
    - PMI and industrial production
    - Leading economic indicators
    - Global economic conditions
    - Sector-specific macro impact
    """

    def __init__(self, name: str = "MacroAgent"):
        super().__init__(name=name, agent_type="macro")

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """
        Analyze macroeconomic conditions and their impact on the stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing macroeconomic analysis
        """
        self.log_info(f"Analyzing macroeconomic conditions for {symbol}")

        try:
            # Get company info for sector-specific analysis
            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get("sector", "")

            # Fetch macroeconomic data
            macro_data = await self._fetch_macro_data()

            # Assess sector-specific impact
            sector_outlook = self._assess_sector_impact(sector, macro_data)
            macro_data.sector_outlook = sector_outlook

            # Determine economic cycle phase
            macro_data.economic_cycle_phase = self._determine_cycle_phase(macro_data)

            # Identify macro risks and opportunities
            macro_data.macro_risks = self._identify_risks(macro_data, sector)
            macro_data.macro_opportunities = self._identify_opportunities(macro_data, sector)

            # Calculate score
            score = self._calculate_score(macro_data, sector)

            # Generate summary
            summary = self._generate_summary(macro_data, sector, score)

            return {
                "macro_data": macro_data.model_dump(),
                "score": score,
                "summary": summary,
            }

        except Exception as e:
            self.log_warning(f"Error in macro analysis: {str(e)}")
            return {
                "macro_data": MacroeconomicData().model_dump(),
                "score": None,
                "summary": f"Unable to perform macro analysis: {str(e)}",
            }

    async def _fetch_macro_data(self) -> MacroeconomicData:
        """Fetch macroeconomic indicators."""
        macro = MacroeconomicData()

        try:
            # Use market ETFs as proxies for macro conditions
            # SPY for overall market, XLF for financials (rate sensitivity), etc.

            # Get market data as economic proxy
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1y")

            if len(spy_hist) > 200:
                # Calculate trend (proxy for GDP growth expectation)
                recent_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-60] - 1)
                if recent_return > 0.10:
                    macro.gdp_trend = "accelerating"
                elif recent_return > 0:
                    macro.gdp_trend = "stable"
                else:
                    macro.gdp_trend = "decelerating"

            # Get Treasury yields for rate environment
            tlt = yf.Ticker("TLT")  # Long-term treasury ETF
            ief = yf.Ticker("IEF")  # Intermediate treasury ETF
            shy = yf.Ticker("SHY")  # Short-term treasury ETF

            # Consumer discretionary vs staples as consumer confidence proxy
            xly = yf.Ticker("XLY")  # Consumer Discretionary
            xlp = yf.Ticker("XLP")  # Consumer Staples

            xly_hist = xly.history(period="3mo")
            xlp_hist = xlp.history(period="3mo")

            if len(xly_hist) > 20 and len(xlp_hist) > 20:
                xly_return = xly_hist['Close'].iloc[-1] / xly_hist['Close'].iloc[0] - 1
                xlp_return = xlp_hist['Close'].iloc[-1] / xlp_hist['Close'].iloc[0] - 1

                # Discretionary outperforming staples = strong consumer
                if xly_return > xlp_return + 0.05:
                    macro.consumer_confidence = 110  # Above average
                elif xly_return < xlp_return - 0.05:
                    macro.consumer_confidence = 90  # Below average
                else:
                    macro.consumer_confidence = 100  # Neutral

            # Industrial production proxy via XLI
            xli = yf.Ticker("XLI")
            xli_hist = xli.history(period="3mo")
            if len(xli_hist) > 20:
                xli_return = xli_hist['Close'].iloc[-1] / xli_hist['Close'].iloc[0] - 1
                macro.industrial_production = xli_return

            # PMI proxy (using industrial sector momentum)
            if macro.industrial_production:
                if macro.industrial_production > 0.05:
                    macro.pmi_manufacturing = 55  # Expansion
                elif macro.industrial_production > 0:
                    macro.pmi_manufacturing = 51  # Slight expansion
                elif macro.industrial_production > -0.05:
                    macro.pmi_manufacturing = 49  # Slight contraction
                else:
                    macro.pmi_manufacturing = 45  # Contraction

            # Housing market proxy via XHB
            xhb = yf.Ticker("XHB")
            xhb_hist = xhb.history(period="3mo")
            if len(xhb_hist) > 20:
                xhb_return = xhb_hist['Close'].iloc[-1] / xhb_hist['Close'].iloc[0] - 1
                macro.home_price_growth = xhb_return

            # Employment proxy (consumer discretionary as jobs proxy)
            macro.employment_trend = "stable"
            if macro.consumer_confidence:
                if macro.consumer_confidence > 105:
                    macro.employment_trend = "improving"
                elif macro.consumer_confidence < 95:
                    macro.employment_trend = "weakening"

            # Emerging markets proxy via EEM
            eem = yf.Ticker("EEM")
            eem_hist = eem.history(period="3mo")
            if len(eem_hist) > 20:
                eem_return = eem_hist['Close'].iloc[-1] / eem_hist['Close'].iloc[0] - 1
                if eem_return > 0.05:
                    macro.emerging_markets_outlook = "positive"
                elif eem_return < -0.05:
                    macro.emerging_markets_outlook = "negative"
                else:
                    macro.emerging_markets_outlook = "neutral"

            # Global growth proxy
            if macro.emerging_markets_outlook == "positive" and macro.gdp_trend in ["accelerating", "stable"]:
                macro.global_growth_outlook = "positive"
            elif macro.emerging_markets_outlook == "negative" and macro.gdp_trend == "decelerating":
                macro.global_growth_outlook = "negative"
            else:
                macro.global_growth_outlook = "neutral"

            # Set some reasonable estimates for current conditions
            macro.gdp_growth_current = 0.025  # ~2.5% estimate
            macro.unemployment_rate = 0.039  # ~3.9% estimate

        except Exception as e:
            logger.warning(f"Error fetching macro data: {e}")

        return macro

    def _assess_sector_impact(
        self, sector: str, macro: MacroeconomicData
    ) -> dict[str, str]:
        """Assess macro impact on specific sector."""
        outlook = {}

        sensitivity = SECTOR_SENSITIVITY.get(sector, {})

        # GDP impact
        if macro.gdp_trend:
            gdp_sens = sensitivity.get("gdp", "medium")
            if macro.gdp_trend == "accelerating":
                if gdp_sens in ["high", "very_high"]:
                    outlook["gdp_impact"] = "very_positive"
                else:
                    outlook["gdp_impact"] = "positive"
            elif macro.gdp_trend == "decelerating":
                if gdp_sens in ["high", "very_high"]:
                    outlook["gdp_impact"] = "negative"
                else:
                    outlook["gdp_impact"] = "slightly_negative"
            else:
                outlook["gdp_impact"] = "neutral"

        # Consumer spending impact
        if macro.consumer_confidence:
            cons_sens = sensitivity.get("consumer_spending", "medium")
            if macro.consumer_confidence > 105:
                if cons_sens in ["high", "very_high"]:
                    outlook["consumer_impact"] = "positive"
                else:
                    outlook["consumer_impact"] = "slightly_positive"
            elif macro.consumer_confidence < 95:
                if cons_sens in ["high", "very_high"]:
                    outlook["consumer_impact"] = "negative"
                else:
                    outlook["consumer_impact"] = "slightly_negative"

        # PMI/Industrial impact
        if macro.pmi_manufacturing:
            if macro.pmi_manufacturing > 52:
                outlook["manufacturing_impact"] = "positive"
            elif macro.pmi_manufacturing < 48:
                outlook["manufacturing_impact"] = "negative"
            else:
                outlook["manufacturing_impact"] = "neutral"

        # Overall sector outlook
        positive_impacts = sum(1 for v in outlook.values() if "positive" in v)
        negative_impacts = sum(1 for v in outlook.values() if "negative" in v)

        if positive_impacts > negative_impacts + 1:
            outlook["overall"] = "favorable"
        elif negative_impacts > positive_impacts + 1:
            outlook["overall"] = "unfavorable"
        else:
            outlook["overall"] = "neutral"

        return outlook

    def _determine_cycle_phase(self, macro: MacroeconomicData) -> str:
        """Determine current economic cycle phase."""
        expansion_signals = 0
        contraction_signals = 0

        if macro.gdp_trend == "accelerating":
            expansion_signals += 2
        elif macro.gdp_trend == "decelerating":
            contraction_signals += 2
        else:
            expansion_signals += 1

        if macro.pmi_manufacturing:
            if macro.pmi_manufacturing > 52:
                expansion_signals += 1
            elif macro.pmi_manufacturing < 48:
                contraction_signals += 1

        if macro.consumer_confidence:
            if macro.consumer_confidence > 105:
                expansion_signals += 1
            elif macro.consumer_confidence < 95:
                contraction_signals += 1

        if macro.employment_trend == "improving":
            expansion_signals += 1
        elif macro.employment_trend == "weakening":
            contraction_signals += 1

        # Determine phase
        if expansion_signals >= 4:
            return "expansion"
        elif contraction_signals >= 4:
            return "contraction"
        elif expansion_signals > contraction_signals:
            return "late_expansion"
        elif contraction_signals > expansion_signals:
            return "early_contraction"
        else:
            return "transition"

    def _identify_risks(
        self, macro: MacroeconomicData, sector: str
    ) -> list[str]:
        """Identify macroeconomic risks."""
        risks = []

        if macro.gdp_trend == "decelerating":
            risks.append("Economic growth is decelerating")

        if macro.pmi_manufacturing and macro.pmi_manufacturing < 50:
            risks.append("Manufacturing sector in contraction")

        if macro.consumer_confidence and macro.consumer_confidence < 95:
            risks.append("Consumer confidence below historical average")

        if macro.employment_trend == "weakening":
            risks.append("Labor market showing signs of weakness")

        if macro.economic_cycle_phase in ["late_expansion", "early_contraction", "contraction"]:
            risks.append(f"Economic cycle in {macro.economic_cycle_phase.replace('_', ' ')} phase")

        if macro.global_growth_outlook == "negative":
            risks.append("Global growth outlook is negative")

        if macro.emerging_markets_outlook == "negative":
            risks.append("Emerging markets weakness may impact global demand")

        # Sector-specific risks
        sensitivity = SECTOR_SENSITIVITY.get(sector, {})
        if sensitivity.get("gdp") in ["high", "very_high"] and macro.gdp_trend == "decelerating":
            risks.append(f"{sector} sector highly sensitive to GDP slowdown")

        return risks[:5]

    def _identify_opportunities(
        self, macro: MacroeconomicData, sector: str
    ) -> list[str]:
        """Identify macroeconomic opportunities."""
        opportunities = []

        if macro.gdp_trend == "accelerating":
            opportunities.append("Accelerating economic growth supports earnings")

        if macro.pmi_manufacturing and macro.pmi_manufacturing > 52:
            opportunities.append("Manufacturing expansion indicates strong demand")

        if macro.consumer_confidence and macro.consumer_confidence > 105:
            opportunities.append("Strong consumer confidence supports spending")

        if macro.employment_trend == "improving":
            opportunities.append("Improving labor market supports consumer health")

        if macro.economic_cycle_phase == "expansion":
            opportunities.append("Economic expansion phase favors risk assets")

        if macro.global_growth_outlook == "positive":
            opportunities.append("Positive global growth outlook")

        if macro.emerging_markets_outlook == "positive":
            opportunities.append("Emerging markets growth provides tailwind")

        # Sector-specific opportunities
        sensitivity = SECTOR_SENSITIVITY.get(sector, {})
        if sensitivity.get("business_investment") in ["high", "very_high"]:
            if macro.pmi_manufacturing and macro.pmi_manufacturing > 50:
                opportunities.append(f"{sector} benefits from business investment cycle")

        return opportunities[:5]

    def _calculate_score(self, macro: MacroeconomicData, sector: str) -> float:
        """Calculate macroeconomic environment score."""
        score = 50.0

        # GDP trend
        if macro.gdp_trend == "accelerating":
            score += 15
        elif macro.gdp_trend == "stable":
            score += 5
        elif macro.gdp_trend == "decelerating":
            score -= 10

        # PMI
        if macro.pmi_manufacturing:
            if macro.pmi_manufacturing > 54:
                score += 10
            elif macro.pmi_manufacturing > 52:
                score += 5
            elif macro.pmi_manufacturing < 48:
                score -= 10
            elif macro.pmi_manufacturing < 50:
                score -= 5

        # Consumer confidence
        if macro.consumer_confidence:
            if macro.consumer_confidence > 110:
                score += 10
            elif macro.consumer_confidence > 100:
                score += 5
            elif macro.consumer_confidence < 90:
                score -= 10
            elif macro.consumer_confidence < 100:
                score -= 5

        # Employment
        if macro.employment_trend == "improving":
            score += 5
        elif macro.employment_trend == "weakening":
            score -= 5

        # Economic cycle
        cycle_scores = {
            "expansion": 15,
            "late_expansion": 5,
            "transition": 0,
            "early_contraction": -10,
            "contraction": -15,
        }
        score += cycle_scores.get(macro.economic_cycle_phase, 0)

        # Global outlook
        if macro.global_growth_outlook == "positive":
            score += 5
        elif macro.global_growth_outlook == "negative":
            score -= 5

        # Sector-specific adjustment
        if macro.sector_outlook:
            overall = macro.sector_outlook.get("overall", "neutral")
            if overall == "favorable":
                score += 10
            elif overall == "unfavorable":
                score -= 10

        return max(0, min(100, score))

    def _generate_summary(
        self, macro: MacroeconomicData, sector: str, score: float
    ) -> str:
        """Generate analysis summary."""
        parts = []

        if macro.economic_cycle_phase:
            phase = macro.economic_cycle_phase.replace("_", " ")
            parts.append(f"Economy is in {phase} phase.")

        if macro.gdp_trend:
            parts.append(f"GDP growth is {macro.gdp_trend}.")

        if macro.pmi_manufacturing:
            if macro.pmi_manufacturing > 50:
                parts.append("Manufacturing sector expanding.")
            else:
                parts.append("Manufacturing sector contracting.")

        if macro.sector_outlook:
            overall = macro.sector_outlook.get("overall", "neutral")
            parts.append(f"Macro environment is {overall} for {sector}.")

        if macro.global_growth_outlook:
            parts.append(f"Global growth outlook: {macro.global_growth_outlook}.")

        return " ".join(parts)
