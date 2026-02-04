"""Stock Analysis Agents Module."""

from .base_agent import BaseAgent
from .master_agent import MasterAgent

# Core Analysis Agents
from .fundamental_agent import FundamentalAnalysisAgent
from .technical_agent import TechnicalAnalysisAgent
from .sentiment_agent import SentimentAnalysisAgent
from .risk_agent import RiskAssessmentAgent

# Ecosystem Analysis Agents (Bridgewater-style)
from .supply_chain_agent import SupplyChainAgent
from .customer_agent import CustomerAnalysisAgent
from .competitive_agent import CompetitiveAnalysisAgent
from .macro_agent import MacroeconomicAgent
from .monetary_policy_agent import MonetaryPolicyAgent

# Portfolio Optimization
from .portfolio_optimizer_agent import PortfolioOptimizerAgent

__all__ = [
    # Base
    "BaseAgent",
    "MasterAgent",
    # Core Analysis
    "FundamentalAnalysisAgent",
    "TechnicalAnalysisAgent",
    "SentimentAnalysisAgent",
    "RiskAssessmentAgent",
    # Ecosystem Analysis
    "SupplyChainAgent",
    "CustomerAnalysisAgent",
    "CompetitiveAnalysisAgent",
    "MacroeconomicAgent",
    "MonetaryPolicyAgent",
    # Portfolio
    "PortfolioOptimizerAgent",
]
