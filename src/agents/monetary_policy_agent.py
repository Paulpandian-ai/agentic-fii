"""Monetary Policy Agent - Analyzes Fed policy, inflation, and yield curve."""

from typing import Any, Optional

import numpy as np
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.schemas import MonetaryPolicyData
from src.utils.yfinance_cache import get_ticker_info, get_ticker_history


# Sector rate sensitivity
RATE_SENSITIVITY = {
    "Technology": "high",  # Growth stocks are rate sensitive
    "Financial Services": "very_high",  # Banks benefit from higher rates
    "Real Estate": "very_high",  # REITs very sensitive to rates
    "Utilities": "high",  # Dividend stocks compete with bonds
    "Consumer Discretionary": "medium",
    "Consumer Staples": "low",
    "Healthcare": "low",
    "Energy": "low",
    "Industrials": "medium",
    "Materials": "medium",
    "Communication Services": "medium",
}


class MonetaryPolicyAgent(BaseAgent):
    """
    Agent responsible for monetary policy analysis.

    Analyzes:
    - Federal Reserve policy stance
    - Interest rate environment and expectations
    - Inflation trends (CPI, PCE)
    - Yield curve analysis
    - Dollar strength
    - Impact on specific stocks/sectors
    """

    def __init__(self, name: str = "MonetaryPolicyAgent"):
        super().__init__(name=name, agent_type="monetary_policy")

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """
        Analyze monetary policy conditions and their impact.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing monetary policy analysis
        """
        self.log_info(f"Analyzing monetary policy impact for {symbol}")

        try:
            # Get company info for sector-specific analysis with caching
            info = get_ticker_info(symbol)
            sector = info.get("sector", "")

            # Fetch monetary policy data
            policy_data = await self._fetch_policy_data()

            # Assess sector-specific impact
            policy_data.rate_sensitivity_impact = self._assess_rate_sensitivity(
                sector, policy_data
            )
            policy_data.sector_impact = self._assess_sector_impact(sector, policy_data)
            policy_data.valuation_impact = self._assess_valuation_impact(
                info, policy_data
            )

            # Identify risks and opportunities
            policy_data.policy_risks = self._identify_risks(policy_data, sector)
            policy_data.policy_opportunities = self._identify_opportunities(
                policy_data, sector
            )

            # Calculate score
            score = self._calculate_score(policy_data, sector)

            # Generate summary
            summary = self._generate_summary(policy_data, sector, score)

            return {
                "monetary_data": policy_data.model_dump(),
                "score": score,
                "summary": summary,
            }

        except Exception as e:
            self.log_warning(f"Error in monetary policy analysis: {str(e)}")
            return {
                "monetary_data": MonetaryPolicyData().model_dump(),
                "score": None,
                "summary": f"Unable to perform monetary policy analysis: {str(e)}",
            }

    async def _fetch_policy_data(self) -> MonetaryPolicyData:
        """Fetch monetary policy indicators from market data."""
        policy = MonetaryPolicyData()

        try:
            # Treasury yields from ETFs as proxies with caching
            # TLT = 20+ year, IEF = 7-10 year, SHY = 1-3 year
            shy_hist = get_ticker_history("SHY", period="1mo")
            ief_hist = get_ticker_history("IEF", period="1mo")
            tlt_hist = get_ticker_history("TLT", period="1mo")

            # Use current Fed Funds rate estimate
            policy.fed_funds_rate = 5.25  # Current approximate rate
            policy.fed_funds_target_upper = 5.50
            policy.fed_funds_target_lower = 5.25

            # Estimate Treasury yields from ETF returns
            # Higher ETF prices = lower yields
            if len(shy_hist) > 0:
                shy_change = (shy_hist['Close'].iloc[-1] / shy_hist['Close'].iloc[0] - 1)
                # Approximate 2-year yield (inverse relationship with SHY)
                policy.treasury_2y = 4.5 - shy_change * 50  # Rough approximation

            if len(ief_hist) > 0:
                ief_change = (ief_hist['Close'].iloc[-1] / ief_hist['Close'].iloc[0] - 1)
                policy.treasury_5y = 4.2 - ief_change * 70
                policy.treasury_10y = 4.3 - ief_change * 80

            if len(tlt_hist) > 0:
                tlt_change = (tlt_hist['Close'].iloc[-1] / tlt_hist['Close'].iloc[0] - 1)
                policy.treasury_30y = 4.5 - tlt_change * 100

            # Calculate yield curve spread
            if policy.treasury_10y and policy.treasury_2y:
                policy.yield_curve_spread = (policy.treasury_10y - policy.treasury_2y) * 100

                # Determine yield curve status
                if policy.yield_curve_spread < -20:
                    policy.yield_curve_status = "inverted"
                    policy.recession_probability = 0.40
                elif policy.yield_curve_spread < 0:
                    policy.yield_curve_status = "flat"
                    policy.recession_probability = 0.25
                elif policy.yield_curve_spread < 50:
                    policy.yield_curve_status = "flat"
                    policy.recession_probability = 0.15
                else:
                    policy.yield_curve_status = "normal"
                    policy.recession_probability = 0.10

            # Inflation estimates from TIPS ETF (TIP vs nominal)
            tip_hist = get_ticker_history("TIP", period="3mo")
            if len(tip_hist) > 0 and len(ief_hist) > 0:
                tip_return = tip_hist['Close'].iloc[-1] / tip_hist['Close'].iloc[0] - 1
                ief_return = ief_hist['Close'].iloc[-1] / ief_hist['Close'].iloc[0] - 1
                # TIPS outperformance suggests inflation concerns
                inflation_signal = tip_return - ief_return
                policy.inflation_expectations = 2.5 + inflation_signal * 20

            # Current inflation estimates (would come from economic data API)
            policy.cpi_current = 3.2  # Approximate
            policy.cpi_core = 3.8  # Approximate
            policy.pce_current = 2.8  # Approximate
            policy.pce_core = 2.9  # Approximate (Fed's target is 2%)

            # Determine inflation trend
            if policy.cpi_core and policy.pce_core:
                avg_core = (policy.cpi_core + policy.pce_core) / 2
                if avg_core > 3.5:
                    policy.inflation_trend = "elevated"
                elif avg_core > 2.5:
                    policy.inflation_trend = "above_target"
                elif avg_core > 2.0:
                    policy.inflation_trend = "near_target"
                else:
                    policy.inflation_trend = "below_target"

            # Dollar index from UUP ETF with caching
            uup_hist = get_ticker_history("UUP", period="3mo")
            if len(uup_hist) > 0:
                uup_return = uup_hist['Close'].iloc[-1] / uup_hist['Close'].iloc[0] - 1
                policy.dxy_index = 104 + uup_return * 100  # Rough approximation

                if uup_return > 0.03:
                    policy.dollar_trend = "strengthening"
                elif uup_return < -0.03:
                    policy.dollar_trend = "weakening"
                else:
                    policy.dollar_trend = "stable"

            # Rate expectations based on market signals
            policy.rate_direction = self._determine_rate_direction(policy)
            policy.rate_hike_probability = self._estimate_rate_probabilities(policy, "hike")
            policy.rate_cut_probability = self._estimate_rate_probabilities(policy, "cut")
            policy.rate_hold_probability = 1 - policy.rate_hike_probability - policy.rate_cut_probability

            # Financial conditions assessment
            policy.financial_conditions = self._assess_financial_conditions(policy)

        except Exception as e:
            logger.warning(f"Error fetching policy data: {e}")

        return policy

    def _determine_rate_direction(self, policy: MonetaryPolicyData) -> str:
        """Determine Fed's likely rate direction."""
        hawkish_signals = 0
        dovish_signals = 0

        # Inflation above target = hawkish
        if policy.inflation_trend in ["elevated", "above_target"]:
            hawkish_signals += 2
        elif policy.inflation_trend in ["below_target"]:
            dovish_signals += 2

        # Inverted curve suggests rates too high
        if policy.yield_curve_status == "inverted":
            dovish_signals += 1
        elif policy.yield_curve_status == "normal":
            hawkish_signals += 1

        # Dollar strength
        if policy.dollar_trend == "strengthening":
            hawkish_signals += 1
        elif policy.dollar_trend == "weakening":
            dovish_signals += 1

        if hawkish_signals > dovish_signals + 1:
            return "hawkish"
        elif dovish_signals > hawkish_signals + 1:
            return "dovish"
        return "neutral"

    def _estimate_rate_probabilities(
        self, policy: MonetaryPolicyData, direction: str
    ) -> float:
        """Estimate probability of rate change."""
        base_prob = 0.1  # Base probability

        if direction == "hike":
            if policy.inflation_trend == "elevated":
                base_prob += 0.3
            elif policy.inflation_trend == "above_target":
                base_prob += 0.15
            if policy.rate_direction == "hawkish":
                base_prob += 0.1
        elif direction == "cut":
            if policy.yield_curve_status == "inverted":
                base_prob += 0.2
            if policy.inflation_trend in ["near_target", "below_target"]:
                base_prob += 0.15
            if policy.rate_direction == "dovish":
                base_prob += 0.15

        return min(0.8, max(0.05, base_prob))

    def _assess_financial_conditions(self, policy: MonetaryPolicyData) -> str:
        """Assess overall financial conditions."""
        tight_signals = 0
        loose_signals = 0

        # High rates = tight
        if policy.fed_funds_rate and policy.fed_funds_rate > 5.0:
            tight_signals += 2
        elif policy.fed_funds_rate and policy.fed_funds_rate < 2.0:
            loose_signals += 2

        # Inverted curve = tight
        if policy.yield_curve_status == "inverted":
            tight_signals += 1

        # Strong dollar = tight
        if policy.dollar_trend == "strengthening":
            tight_signals += 1
        elif policy.dollar_trend == "weakening":
            loose_signals += 1

        if tight_signals > loose_signals + 1:
            return "tight"
        elif loose_signals > tight_signals + 1:
            return "loose"
        return "neutral"

    def _assess_rate_sensitivity(
        self, sector: str, policy: MonetaryPolicyData
    ) -> str:
        """Assess stock's sensitivity to interest rates."""
        sensitivity = RATE_SENSITIVITY.get(sector, "medium")

        # Adjust based on current rate environment
        if policy.financial_conditions == "tight":
            if sensitivity == "very_high":
                return "highly_negative"
            elif sensitivity == "high":
                return "negative"
            else:
                return "slightly_negative"
        elif policy.financial_conditions == "loose":
            if sensitivity == "very_high":
                return "highly_positive"
            elif sensitivity == "high":
                return "positive"
            else:
                return "slightly_positive"

        return "neutral"

    def _assess_sector_impact(
        self, sector: str, policy: MonetaryPolicyData
    ) -> str:
        """Assess monetary policy impact on sector."""
        sensitivity = RATE_SENSITIVITY.get(sector, "medium")

        # Financial Services benefit from higher rates
        if sector == "Financial Services":
            if policy.fed_funds_rate and policy.fed_funds_rate > 4.0:
                return "positive (banks benefit from rate spread)"
            else:
                return "neutral"

        # Real Estate hurt by high rates
        if sector == "Real Estate":
            if policy.fed_funds_rate and policy.fed_funds_rate > 4.0:
                return "negative (high rates pressure REITs)"
            else:
                return "positive"

        # Growth stocks (Technology) hurt by high rates
        if sector == "Technology":
            if policy.fed_funds_rate and policy.fed_funds_rate > 4.0:
                return "negative (high rates compress growth valuations)"
            elif policy.rate_direction == "dovish":
                return "positive (rate cuts support growth stocks)"
            return "neutral"

        # Defensive sectors less affected
        if sector in ["Consumer Staples", "Healthcare", "Utilities"]:
            return "limited impact (defensive sector)"

        return "moderate impact"

    def _assess_valuation_impact(
        self, info: dict, policy: MonetaryPolicyData
    ) -> str:
        """Assess impact of rates on stock valuation."""
        pe_ratio = info.get("trailingPE")

        if not pe_ratio:
            return "uncertain"

        # High PE stocks more sensitive to rate changes
        if pe_ratio > 40:
            if policy.financial_conditions == "tight":
                return "negative (high multiples vulnerable to rate pressure)"
            elif policy.rate_direction == "dovish":
                return "positive (rate cuts support high-growth valuations)"
            return "elevated risk from valuation"

        elif pe_ratio > 25:
            if policy.financial_conditions == "tight":
                return "moderately negative"
            return "moderate sensitivity"

        else:
            return "limited valuation impact (reasonable multiple)"

    def _identify_risks(
        self, policy: MonetaryPolicyData, sector: str
    ) -> list[str]:
        """Identify monetary policy risks."""
        risks = []

        if policy.inflation_trend in ["elevated", "above_target"]:
            risks.append("Inflation remains above Fed's 2% target")

        if policy.rate_direction == "hawkish":
            risks.append("Fed may maintain restrictive policy longer")

        if policy.yield_curve_status == "inverted":
            risks.append(f"Inverted yield curve signals recession risk ({policy.recession_probability:.0%} probability)")

        if policy.financial_conditions == "tight":
            risks.append("Tight financial conditions may constrain growth")

        if policy.dollar_trend == "strengthening":
            risks.append("Strong dollar headwind for international earnings")

        sensitivity = RATE_SENSITIVITY.get(sector, "medium")
        if sensitivity in ["high", "very_high"] and policy.fed_funds_rate and policy.fed_funds_rate > 4.0:
            risks.append(f"{sector} sector highly sensitive to current rate levels")

        return risks[:5]

    def _identify_opportunities(
        self, policy: MonetaryPolicyData, sector: str
    ) -> list[str]:
        """Identify monetary policy opportunities."""
        opportunities = []

        if policy.rate_direction == "dovish":
            opportunities.append("Potential rate cuts would support asset prices")

        if policy.inflation_trend in ["near_target", "below_target"]:
            opportunities.append("Inflation normalizing may allow policy easing")

        if policy.rate_cut_probability and policy.rate_cut_probability > 0.3:
            opportunities.append(f"Market pricing {policy.rate_cut_probability:.0%} chance of rate cut")

        # Sector-specific opportunities
        if sector == "Financial Services" and policy.fed_funds_rate and policy.fed_funds_rate > 4.0:
            opportunities.append("Banks benefit from higher net interest margins")

        if sector == "Technology" and policy.rate_direction == "dovish":
            opportunities.append("Tech valuations supported by rate cut expectations")

        if policy.dollar_trend == "weakening":
            opportunities.append("Weakening dollar supports international revenue")

        if policy.yield_curve_status == "normal":
            opportunities.append("Normal yield curve supports economic expansion")

        return opportunities[:4]

    def _calculate_score(self, policy: MonetaryPolicyData, sector: str) -> float:
        """Calculate monetary environment score."""
        score = 50.0

        # Inflation impact
        if policy.inflation_trend == "near_target":
            score += 15
        elif policy.inflation_trend == "below_target":
            score += 10
        elif policy.inflation_trend == "above_target":
            score -= 5
        elif policy.inflation_trend == "elevated":
            score -= 15

        # Rate direction
        if policy.rate_direction == "dovish":
            score += 10
        elif policy.rate_direction == "hawkish":
            score -= 5

        # Yield curve
        if policy.yield_curve_status == "normal":
            score += 10
        elif policy.yield_curve_status == "inverted":
            score -= 15
        elif policy.yield_curve_status == "flat":
            score -= 5

        # Financial conditions
        if policy.financial_conditions == "loose":
            score += 10
        elif policy.financial_conditions == "tight":
            score -= 10

        # Sector-specific adjustment
        sensitivity = RATE_SENSITIVITY.get(sector, "medium")
        if policy.financial_conditions == "tight":
            if sensitivity == "very_high":
                score -= 15
            elif sensitivity == "high":
                score -= 10
        elif policy.financial_conditions == "loose":
            if sensitivity in ["high", "very_high"]:
                score += 5

        # Recession probability
        if policy.recession_probability:
            if policy.recession_probability > 0.3:
                score -= 10
            elif policy.recession_probability > 0.2:
                score -= 5

        return max(0, min(100, score))

    def _generate_summary(
        self, policy: MonetaryPolicyData, sector: str, score: float
    ) -> str:
        """Generate analysis summary."""
        parts = []

        if policy.fed_funds_rate:
            parts.append(f"Fed Funds rate at {policy.fed_funds_rate:.2f}%.")

        if policy.rate_direction:
            parts.append(f"Policy stance is {policy.rate_direction}.")

        if policy.inflation_trend:
            trend = policy.inflation_trend.replace("_", " ")
            parts.append(f"Inflation is {trend}.")

        if policy.yield_curve_status:
            parts.append(f"Yield curve is {policy.yield_curve_status}.")

        if policy.sector_impact:
            parts.append(f"Sector impact: {policy.sector_impact}.")

        if policy.recession_probability and policy.recession_probability > 0.2:
            parts.append(f"Recession probability: {policy.recession_probability:.0%}.")

        return " ".join(parts)
