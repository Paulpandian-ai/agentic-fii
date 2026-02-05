"""Portfolio Optimizer Agent - Builds optimal portfolios using Sharpe ratio optimization."""

from typing import Any, Optional
import numpy as np
import pandas as pd
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.utils.yfinance_cache import get_ticker_info, get_ticker_history


class PortfolioOptimizerAgent(BaseAgent):
    """
    Agent responsible for portfolio optimization using Sharpe ratio.

    Features:
    - Analyzes current holdings
    - Optimizes for maximum Sharpe ratio
    - Provides buy/sell recommendations
    - Suggests optimal weightage
    """

    def __init__(
        self,
        name: str = "PortfolioOptimizer",
        risk_free_rate: float = 0.045,
    ):
        super().__init__(name=name, agent_type="portfolio")
        self.risk_free_rate = risk_free_rate

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """Main analyze method - optimizes a portfolio of stocks."""
        return await self.optimize_portfolio(**kwargs)

    async def optimize_portfolio(
        self,
        holdings: dict[str, dict] = None,  # {symbol: {shares: x, cost_basis: y}}
        candidate_symbols: list[str] = None,
        risk_free_rate: float = None,
        max_position_pct: float = 0.30,
        min_position_pct: float = 0.02,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Optimize portfolio based on current holdings.

        Args:
            holdings: Current holdings {symbol: {shares: N, cost_basis: price}}
            candidate_symbols: Additional symbols to consider for buying
            risk_free_rate: Risk-free rate for Sharpe calculation
            max_position_pct: Maximum position size (0.30 = 30%)
            min_position_pct: Minimum position size (0.02 = 2%)
        """
        if risk_free_rate:
            self.risk_free_rate = risk_free_rate

        holdings = holdings or {}
        candidate_symbols = candidate_symbols or []

        # Combine all symbols
        all_symbols = list(set(list(holdings.keys()) + candidate_symbols))

        if not all_symbols:
            return {"error": "No symbols provided for optimization"}

        try:
            # Get historical data and current prices
            stock_data = self._get_stock_data(all_symbols)

            if stock_data['returns'].empty:
                return {"error": "Unable to fetch stock data"}

            # Calculate current portfolio value and weights
            current_analysis = self._analyze_current_holdings(
                holdings, stock_data['prices']
            )

            # Run optimization
            optimal_weights = self._optimize_for_sharpe(
                stock_data['returns'],
                max_position_pct,
                min_position_pct,
            )

            # Calculate portfolio metrics for optimal portfolio
            optimal_metrics = self._calculate_portfolio_metrics(
                optimal_weights,
                stock_data['returns'],
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                current_analysis,
                optimal_weights,
                stock_data,
            )

            # Build response
            result = {
                "status": "success",
                "current_portfolio": current_analysis,
                "optimal_allocation": optimal_weights,
                "optimal_metrics": optimal_metrics,
                "recommendations": recommendations,
                "stock_analysis": self._analyze_individual_stocks(
                    all_symbols, stock_data
                ),
            }

            return result

        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {"error": str(e)}

    def _get_stock_data(self, symbols: list[str], period: str = "1y") -> dict:
        """Fetch historical data for stocks."""
        try:
            # Download price data
            data = yf.download(
                symbols,
                period=period,
                progress=False,
                auto_adjust=True,
            )

            if data.empty:
                return {"returns": pd.DataFrame(), "prices": {}}

            # Handle single vs multiple symbols
            if len(symbols) == 1:
                close_prices = data["Close"].to_frame(name=symbols[0])
            else:
                close_prices = data["Close"]

            # Calculate daily returns
            returns = close_prices.pct_change().dropna()

            # Get current prices and info with caching
            current_prices = {}
            stock_info = {}
            for symbol in symbols:
                try:
                    info = get_ticker_info(symbol)
                    price = info.get("currentPrice") or info.get("regularMarketPrice")
                    if price:
                        current_prices[symbol] = price
                    elif symbol in close_prices.columns:
                        current_prices[symbol] = close_prices[symbol].iloc[-1]
                    stock_info[symbol] = {
                        "name": info.get("shortName", symbol),
                        "sector": info.get("sector", "Unknown"),
                        "industry": info.get("industry", "Unknown"),
                    }
                except:
                    if symbol in close_prices.columns:
                        current_prices[symbol] = close_prices[symbol].iloc[-1]
                    stock_info[symbol] = {
                        "name": symbol,
                        "sector": "Unknown",
                        "industry": "Unknown",
                    }

            return {
                "returns": returns,
                "prices": current_prices,
                "info": stock_info,
                "close_history": close_prices,
            }

        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return {"returns": pd.DataFrame(), "prices": {}}

    def _analyze_current_holdings(
        self,
        holdings: dict[str, dict],
        current_prices: dict[str, float],
    ) -> dict:
        """Analyze current portfolio holdings."""
        if not holdings:
            return {
                "total_value": 0,
                "positions": [],
                "weights": {},
            }

        positions = []
        total_value = 0

        for symbol, holding in holdings.items():
            shares = holding.get("shares", 0)
            cost_basis = holding.get("cost_basis", 0)
            current_price = current_prices.get(symbol, cost_basis)

            market_value = shares * current_price
            total_cost = shares * cost_basis
            gain_loss = market_value - total_cost
            gain_loss_pct = (gain_loss / total_cost * 100) if total_cost > 0 else 0

            positions.append({
                "symbol": symbol,
                "shares": shares,
                "cost_basis": cost_basis,
                "current_price": current_price,
                "market_value": market_value,
                "gain_loss": gain_loss,
                "gain_loss_pct": gain_loss_pct,
            })

            total_value += market_value

        # Calculate current weights
        weights = {}
        for pos in positions:
            if total_value > 0:
                pos["weight"] = pos["market_value"] / total_value
                weights[pos["symbol"]] = pos["weight"]
            else:
                pos["weight"] = 0
                weights[pos["symbol"]] = 0

        return {
            "total_value": total_value,
            "positions": positions,
            "weights": weights,
        }

    def _optimize_for_sharpe(
        self,
        returns: pd.DataFrame,
        max_weight: float,
        min_weight: float,
    ) -> dict[str, float]:
        """Optimize portfolio weights for maximum Sharpe ratio."""
        if returns.empty:
            return {}

        symbols = list(returns.columns)
        n_assets = len(symbols)

        # Annualized metrics
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Monte Carlo optimization
        n_portfolios = 10000
        best_sharpe = -np.inf
        best_weights = np.ones(n_assets) / n_assets

        np.random.seed(42)

        for _ in range(n_portfolios):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()

            # Calculate portfolio return and volatility
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))

            # Sharpe ratio
            if port_vol > 0:
                sharpe = (port_return - self.risk_free_rate) / port_vol
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = weights.copy()

        # Create weights dictionary
        optimal_weights = {}
        for i, symbol in enumerate(symbols):
            if best_weights[i] >= min_weight:
                optimal_weights[symbol] = round(float(best_weights[i]), 4)

        # Normalize
        total = sum(optimal_weights.values())
        if total > 0:
            optimal_weights = {k: round(v/total, 4) for k, v in optimal_weights.items()}

        return optimal_weights

    def _calculate_portfolio_metrics(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
    ) -> dict:
        """Calculate portfolio risk/return metrics."""
        if not weights or returns.empty:
            return {}

        symbols = [s for s in weights.keys() if s in returns.columns]
        if not symbols:
            return {}

        weight_array = np.array([weights[s] for s in symbols])
        weight_array = weight_array / weight_array.sum()

        portfolio_returns = returns[symbols]
        mean_returns = portfolio_returns.mean() * 252
        cov_matrix = portfolio_returns.cov() * 252

        # Portfolio return and volatility
        port_return = float(np.dot(weight_array, mean_returns))
        port_vol = float(np.sqrt(np.dot(weight_array.T, np.dot(cov_matrix.values, weight_array))))

        # Sharpe ratio
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0

        # Portfolio daily returns for VaR
        port_daily_returns = portfolio_returns.dot(weight_array)

        # Value at Risk
        var_95 = float(-np.percentile(port_daily_returns, 5))
        var_99 = float(-np.percentile(port_daily_returns, 1))

        # Max Drawdown
        cumulative = (1 + port_daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_dd = float(abs(drawdowns.min()))

        # Sortino ratio
        downside_returns = port_daily_returns[port_daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (port_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0

        return {
            "expected_return": round(port_return, 4),
            "volatility": round(port_vol, 4),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "var_95_daily": round(var_95, 4),
            "var_99_daily": round(var_99, 4),
            "max_drawdown": round(max_dd, 4),
        }

    def _analyze_individual_stocks(
        self,
        symbols: list[str],
        stock_data: dict,
    ) -> list[dict]:
        """Analyze individual stocks for recommendations."""
        results = []
        returns = stock_data.get("returns", pd.DataFrame())
        prices = stock_data.get("prices", {})
        info = stock_data.get("info", {})

        for symbol in symbols:
            if symbol not in returns.columns:
                continue

            stock_returns = returns[symbol]

            # Calculate metrics
            annual_return = float(stock_returns.mean() * 252)
            annual_vol = float(stock_returns.std() * np.sqrt(252))
            sharpe = (annual_return - self.risk_free_rate) / annual_vol if annual_vol > 0 else 0

            # Trend analysis
            if len(stock_returns) >= 20:
                recent_return = float(stock_returns.tail(20).mean() * 252)
                momentum = "positive" if recent_return > annual_return else "negative"
            else:
                momentum = "neutral"

            # Rating based on Sharpe
            if sharpe > 1.0:
                rating = "Strong Buy"
            elif sharpe > 0.5:
                rating = "Buy"
            elif sharpe > 0:
                rating = "Hold"
            elif sharpe > -0.5:
                rating = "Reduce"
            else:
                rating = "Sell"

            results.append({
                "symbol": symbol,
                "name": info.get(symbol, {}).get("name", symbol),
                "sector": info.get(symbol, {}).get("sector", "Unknown"),
                "current_price": prices.get(symbol),
                "annual_return": round(annual_return, 4),
                "volatility": round(annual_vol, 4),
                "sharpe_ratio": round(sharpe, 2),
                "momentum": momentum,
                "rating": rating,
            })

        return sorted(results, key=lambda x: x["sharpe_ratio"], reverse=True)

    def _generate_recommendations(
        self,
        current_analysis: dict,
        optimal_weights: dict[str, float],
        stock_data: dict,
    ) -> dict:
        """Generate buy/sell recommendations."""
        current_weights = current_analysis.get("weights", {})
        total_value = current_analysis.get("total_value", 0)
        prices = stock_data.get("prices", {})

        buy_recommendations = []
        sell_recommendations = []
        rebalance_recommendations = []

        # All symbols in consideration
        all_symbols = set(list(current_weights.keys()) + list(optimal_weights.keys()))

        for symbol in all_symbols:
            current_wt = current_weights.get(symbol, 0)
            optimal_wt = optimal_weights.get(symbol, 0)
            weight_diff = optimal_wt - current_wt

            current_price = prices.get(symbol, 0)

            if abs(weight_diff) < 0.01:  # Less than 1% difference
                continue

            if current_wt == 0 and optimal_wt > 0:
                # New position to buy
                target_value = total_value * optimal_wt if total_value > 0 else optimal_wt * 10000
                shares_to_buy = int(target_value / current_price) if current_price > 0 else 0

                buy_recommendations.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "reason": f"Add new position at {optimal_wt*100:.1f}% allocation",
                    "target_weight": optimal_wt,
                    "current_weight": 0,
                    "shares_to_buy": shares_to_buy,
                    "estimated_cost": shares_to_buy * current_price if current_price else 0,
                })

            elif optimal_wt == 0 and current_wt > 0:
                # Sell entire position
                pos = next((p for p in current_analysis.get("positions", [])
                           if p["symbol"] == symbol), None)

                sell_recommendations.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "reason": "Remove from portfolio - low risk-adjusted returns",
                    "target_weight": 0,
                    "current_weight": current_wt,
                    "shares_to_sell": pos["shares"] if pos else 0,
                    "estimated_proceeds": pos["market_value"] if pos else 0,
                })

            elif weight_diff > 0.02:
                # Increase position
                additional_value = total_value * weight_diff if total_value > 0 else weight_diff * 10000
                shares_to_buy = int(additional_value / current_price) if current_price > 0 else 0

                rebalance_recommendations.append({
                    "symbol": symbol,
                    "action": "INCREASE",
                    "reason": f"Increase from {current_wt*100:.1f}% to {optimal_wt*100:.1f}%",
                    "target_weight": optimal_wt,
                    "current_weight": current_wt,
                    "weight_change": weight_diff,
                    "shares_to_buy": shares_to_buy,
                })

            elif weight_diff < -0.02:
                # Decrease position
                reduce_value = total_value * abs(weight_diff) if total_value > 0 else abs(weight_diff) * 10000
                shares_to_sell = int(reduce_value / current_price) if current_price > 0 else 0

                rebalance_recommendations.append({
                    "symbol": symbol,
                    "action": "REDUCE",
                    "reason": f"Reduce from {current_wt*100:.1f}% to {optimal_wt*100:.1f}%",
                    "target_weight": optimal_wt,
                    "current_weight": current_wt,
                    "weight_change": weight_diff,
                    "shares_to_sell": shares_to_sell,
                })

        # Add high-conviction new stock recommendations
        stock_analysis = self._analyze_individual_stocks(
            list(optimal_weights.keys()), stock_data
        )

        new_stock_picks = []
        for stock in stock_analysis:
            if stock["symbol"] not in current_weights and stock["sharpe_ratio"] > 0.5:
                new_stock_picks.append({
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "sharpe_ratio": stock["sharpe_ratio"],
                    "rating": stock["rating"],
                    "reason": f"High Sharpe ratio ({stock['sharpe_ratio']:.2f}), {stock['momentum']} momentum",
                })

        return {
            "buy": buy_recommendations,
            "sell": sell_recommendations,
            "rebalance": rebalance_recommendations,
            "new_stock_picks": new_stock_picks[:5],  # Top 5
            "summary": self._generate_recommendation_summary(
                buy_recommendations, sell_recommendations, rebalance_recommendations
            ),
        }

    def _generate_recommendation_summary(
        self,
        buy_recs: list,
        sell_recs: list,
        rebalance_recs: list,
    ) -> str:
        """Generate a summary of recommendations."""
        parts = []

        if buy_recs:
            symbols = [r["symbol"] for r in buy_recs]
            parts.append(f"BUY: {', '.join(symbols)}")

        if sell_recs:
            symbols = [r["symbol"] for r in sell_recs]
            parts.append(f"SELL: {', '.join(symbols)}")

        increase = [r for r in rebalance_recs if r["action"] == "INCREASE"]
        decrease = [r for r in rebalance_recs if r["action"] == "REDUCE"]

        if increase:
            symbols = [r["symbol"] for r in increase]
            parts.append(f"INCREASE: {', '.join(symbols)}")

        if decrease:
            symbols = [r["symbol"] for r in decrease]
            parts.append(f"REDUCE: {', '.join(symbols)}")

        if not parts:
            return "Portfolio is optimally balanced. No changes recommended."

        return " | ".join(parts)
