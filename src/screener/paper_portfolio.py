"""
Paper Portfolio Manager - Track virtual portfolios for testing investment strategies.

This module provides functionality to:
- Create paper portfolios from stock recommendations
- Track portfolio performance over time
- Compare multiple portfolios
- Calculate returns, gains/losses, and other metrics
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional
from pathlib import Path

from src.utils.yfinance_cache import get_ticker_info, get_ticker_history


@dataclass
class PaperHolding:
    """A single stock holding in a paper portfolio."""
    symbol: str
    shares: float
    purchase_price: float
    purchase_date: str  # ISO format date string
    notes: str = ""

    @property
    def cost_basis(self) -> float:
        """Total cost of this holding."""
        return self.shares * self.purchase_price

    def get_current_price(self) -> float:
        """Get current market price."""
        try:
            info = get_ticker_info(self.symbol)
            return info.get('currentPrice') or info.get('regularMarketPrice') or self.purchase_price
        except:
            return self.purchase_price

    def get_current_value(self) -> float:
        """Get current market value of holding."""
        return self.shares * self.get_current_price()

    def get_gain_loss(self) -> float:
        """Get unrealized gain/loss."""
        return self.get_current_value() - self.cost_basis

    def get_return_pct(self) -> float:
        """Get return percentage."""
        if self.cost_basis == 0:
            return 0.0
        return ((self.get_current_value() - self.cost_basis) / self.cost_basis) * 100


@dataclass
class PortfolioSnapshot:
    """A point-in-time snapshot of portfolio value."""
    date: str  # ISO format
    total_value: float
    total_cost: float
    return_pct: float
    holdings_values: dict  # symbol -> value


@dataclass
class PaperPortfolio:
    """A paper trading portfolio."""
    name: str
    created_date: str  # ISO format
    initial_investment: float
    holdings: list[PaperHolding] = field(default_factory=list)
    snapshots: list[PortfolioSnapshot] = field(default_factory=list)
    description: str = ""
    strategy: str = ""  # e.g., "Growth", "Value", "Balanced"

    def add_holding(self, symbol: str, shares: float, purchase_price: float,
                    purchase_date: str = None, notes: str = "") -> None:
        """Add a stock holding to the portfolio."""
        if purchase_date is None:
            purchase_date = datetime.now().strftime("%Y-%m-%d")

        # Check if we already have this symbol
        for holding in self.holdings:
            if holding.symbol == symbol:
                # Average the price for the combined position
                total_shares = holding.shares + shares
                total_cost = (holding.shares * holding.purchase_price) + (shares * purchase_price)
                holding.purchase_price = total_cost / total_shares if total_shares > 0 else 0
                holding.shares = total_shares
                return

        self.holdings.append(PaperHolding(
            symbol=symbol,
            shares=shares,
            purchase_price=purchase_price,
            purchase_date=purchase_date,
            notes=notes
        ))

    def remove_holding(self, symbol: str) -> bool:
        """Remove a holding from the portfolio."""
        for i, holding in enumerate(self.holdings):
            if holding.symbol == symbol:
                self.holdings.pop(i)
                return True
        return False

    def get_total_cost(self) -> float:
        """Get total cost basis of all holdings."""
        return sum(h.cost_basis for h in self.holdings)

    def get_total_value(self) -> float:
        """Get current total market value."""
        return sum(h.get_current_value() for h in self.holdings)

    def get_cash_remaining(self) -> float:
        """Get uninvested cash."""
        return self.initial_investment - self.get_total_cost()

    def get_total_return(self) -> float:
        """Get total return amount."""
        return self.get_total_value() - self.get_total_cost()

    def get_total_return_pct(self) -> float:
        """Get total return percentage."""
        cost = self.get_total_cost()
        if cost == 0:
            return 0.0
        return (self.get_total_return() / cost) * 100

    def get_portfolio_return_pct(self) -> float:
        """Get return on initial investment."""
        if self.initial_investment == 0:
            return 0.0
        current_value = self.get_total_value() + self.get_cash_remaining()
        return ((current_value - self.initial_investment) / self.initial_investment) * 100

    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a snapshot of current portfolio state."""
        holdings_values = {}
        for h in self.holdings:
            holdings_values[h.symbol] = h.get_current_value()

        snapshot = PortfolioSnapshot(
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_value=self.get_total_value(),
            total_cost=self.get_total_cost(),
            return_pct=self.get_total_return_pct(),
            holdings_values=holdings_values
        )
        self.snapshots.append(snapshot)
        return snapshot

    def get_holdings_summary(self) -> list[dict]:
        """Get summary of all holdings with current values."""
        summary = []
        for h in self.holdings:
            current_price = h.get_current_price()
            current_value = h.shares * current_price
            gain_loss = current_value - h.cost_basis
            return_pct = ((current_value - h.cost_basis) / h.cost_basis * 100) if h.cost_basis > 0 else 0

            summary.append({
                'symbol': h.symbol,
                'shares': h.shares,
                'purchase_price': h.purchase_price,
                'current_price': current_price,
                'cost_basis': h.cost_basis,
                'current_value': current_value,
                'gain_loss': gain_loss,
                'return_pct': return_pct,
                'purchase_date': h.purchase_date,
                'weight': 0  # Will be calculated below
            })

        # Calculate weights
        total_value = sum(s['current_value'] for s in summary)
        if total_value > 0:
            for s in summary:
                s['weight'] = (s['current_value'] / total_value) * 100

        return summary

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'created_date': self.created_date,
            'initial_investment': self.initial_investment,
            'description': self.description,
            'strategy': self.strategy,
            'holdings': [asdict(h) for h in self.holdings],
            'snapshots': [asdict(s) for s in self.snapshots]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PaperPortfolio':
        """Create from dictionary."""
        portfolio = cls(
            name=data['name'],
            created_date=data['created_date'],
            initial_investment=data['initial_investment'],
            description=data.get('description', ''),
            strategy=data.get('strategy', '')
        )

        for h_data in data.get('holdings', []):
            portfolio.holdings.append(PaperHolding(**h_data))

        for s_data in data.get('snapshots', []):
            portfolio.snapshots.append(PortfolioSnapshot(**s_data))

        return portfolio


class PaperPortfolioManager:
    """Manager for multiple paper portfolios with persistence."""

    def __init__(self, storage_path: str = None):
        """Initialize the manager."""
        if storage_path is None:
            storage_path = os.path.join(os.path.expanduser("~"), ".truenorth_paper_portfolios.json")
        self.storage_path = storage_path
        self.portfolios: dict[str, PaperPortfolio] = {}
        self._load()

    def _load(self) -> None:
        """Load portfolios from storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for name, p_data in data.items():
                        self.portfolios[name] = PaperPortfolio.from_dict(p_data)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading portfolios: {e}")
                self.portfolios = {}

    def _save(self) -> None:
        """Save portfolios to storage."""
        data = {name: p.to_dict() for name, p in self.portfolios.items()}
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def create_portfolio(self, name: str, initial_investment: float,
                        description: str = "", strategy: str = "") -> Optional[PaperPortfolio]:
        """Create a new paper portfolio."""
        if name in self.portfolios:
            return None  # Already exists

        portfolio = PaperPortfolio(
            name=name,
            created_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            initial_investment=initial_investment,
            description=description,
            strategy=strategy
        )
        self.portfolios[name] = portfolio
        self._save()
        return portfolio

    def create_from_recommendations(self, name: str, stocks: list[dict],
                                    initial_investment: float,
                                    description: str = "",
                                    strategy: str = "AI Recommended") -> Optional[PaperPortfolio]:
        """
        Create a portfolio from investment manager recommendations.

        Args:
            name: Portfolio name
            stocks: List of dicts with 'symbol', 'shares', 'current_price' keys
            initial_investment: Total investment amount
            description: Portfolio description
            strategy: Investment strategy label
        """
        portfolio = self.create_portfolio(name, initial_investment, description, strategy)
        if portfolio is None:
            return None

        purchase_date = datetime.now().strftime("%Y-%m-%d")

        for stock in stocks:
            if stock.get('shares', 0) > 0:
                portfolio.add_holding(
                    symbol=stock['symbol'],
                    shares=stock['shares'],
                    purchase_price=stock.get('current_price', stock.get('purchase_price', 0)),
                    purchase_date=purchase_date,
                    notes=f"Weight: {stock.get('weight', 0):.1f}%"
                )

        # Take initial snapshot
        portfolio.take_snapshot()
        self._save()
        return portfolio

    def get_portfolio(self, name: str) -> Optional[PaperPortfolio]:
        """Get a portfolio by name."""
        return self.portfolios.get(name)

    def get_all_portfolios(self) -> list[PaperPortfolio]:
        """Get all portfolios."""
        return list(self.portfolios.values())

    def delete_portfolio(self, name: str) -> bool:
        """Delete a portfolio."""
        if name in self.portfolios:
            del self.portfolios[name]
            self._save()
            return True
        return False

    def rename_portfolio(self, old_name: str, new_name: str) -> bool:
        """Rename a portfolio."""
        if old_name not in self.portfolios or new_name in self.portfolios:
            return False

        portfolio = self.portfolios.pop(old_name)
        portfolio.name = new_name
        self.portfolios[new_name] = portfolio
        self._save()
        return True

    def update_snapshots(self) -> None:
        """Take snapshots for all portfolios."""
        for portfolio in self.portfolios.values():
            portfolio.take_snapshot()
        self._save()

    def get_portfolio_comparison(self) -> list[dict]:
        """Get comparison data for all portfolios."""
        comparison = []
        for portfolio in self.portfolios.values():
            comparison.append({
                'name': portfolio.name,
                'created_date': portfolio.created_date,
                'initial_investment': portfolio.initial_investment,
                'total_value': portfolio.get_total_value(),
                'cash_remaining': portfolio.get_cash_remaining(),
                'total_return': portfolio.get_total_return(),
                'return_pct': portfolio.get_total_return_pct(),
                'portfolio_return_pct': portfolio.get_portfolio_return_pct(),
                'num_holdings': len(portfolio.holdings),
                'strategy': portfolio.strategy
            })
        return comparison


# Singleton instance
_paper_portfolio_manager: Optional[PaperPortfolioManager] = None


def get_paper_portfolio_manager() -> PaperPortfolioManager:
    """Get the singleton paper portfolio manager instance."""
    global _paper_portfolio_manager
    if _paper_portfolio_manager is None:
        _paper_portfolio_manager = PaperPortfolioManager()
    return _paper_portfolio_manager
