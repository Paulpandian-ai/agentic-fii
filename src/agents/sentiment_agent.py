"""Sentiment Analysis Agent - Analyzes news and market sentiment."""

from typing import Any, Optional

import yfinance as yf
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.schemas import SentimentData


class SentimentAnalysisAgent(BaseAgent):
    """
    Agent responsible for sentiment analysis of stocks.

    Analyzes sentiment from:
    - News articles and headlines
    - Analyst ratings and recommendations
    - Price targets
    - Recent company announcements
    """

    def __init__(self, name: str = "SentimentAgent"):
        super().__init__(name=name, agent_type="sentiment")

    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """
        Perform sentiment analysis on the given stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dictionary containing sentiment data and analysis
        """
        self.log_info(f"Analyzing sentiment for {symbol}")

        try:
            ticker = yf.Ticker(symbol)

            # Gather sentiment data from multiple sources
            sentiment_data = SentimentData(symbol=symbol)

            # Get news sentiment
            news_result = self._analyze_news(ticker)
            sentiment_data.news_sentiment = news_result["sentiment"]
            sentiment_data.news_count = news_result["count"]
            sentiment_data.positive_news_count = news_result["positive"]
            sentiment_data.negative_news_count = news_result["negative"]
            sentiment_data.neutral_news_count = news_result["neutral"]
            sentiment_data.key_headlines = news_result["headlines"]

            # Get analyst recommendations
            analyst_result = self._analyze_recommendations(ticker)
            sentiment_data.analyst_rating = analyst_result["rating"]
            sentiment_data.buy_ratings = analyst_result["buy"]
            sentiment_data.hold_ratings = analyst_result["hold"]
            sentiment_data.sell_ratings = analyst_result["sell"]
            sentiment_data.target_price = analyst_result["target_price"]

            # Calculate overall sentiment
            overall = self._calculate_overall_sentiment(sentiment_data)
            sentiment_data.overall_sentiment = overall["sentiment"]
            sentiment_data.sentiment_score = overall["score"]

            # Calculate sentiment score (0-100)
            score = self._calculate_score(sentiment_data)

            # Generate summary
            summary = self._generate_summary(sentiment_data, score)

            return {
                "sentiment_data": sentiment_data.model_dump(),
                "score": score,
                "summary": summary,
            }

        except Exception as e:
            self.log_warning(f"Error in sentiment analysis: {str(e)}")
            return {
                "sentiment_data": SentimentData(symbol=symbol).model_dump(),
                "score": None,
                "summary": f"Unable to perform sentiment analysis: {str(e)}",
            }

    def _analyze_news(self, ticker) -> dict[str, Any]:
        """Analyze news articles for sentiment."""
        try:
            news = ticker.news
            if not news:
                return {
                    "sentiment": None,
                    "count": 0,
                    "positive": 0,
                    "negative": 0,
                    "neutral": 0,
                    "headlines": [],
                }

            positive = 0
            negative = 0
            neutral = 0
            headlines = []

            # Simple keyword-based sentiment (can be enhanced with NLP)
            positive_words = [
                "surge", "gain", "rise", "jump", "soar", "rally", "beat",
                "exceed", "strong", "growth", "profit", "upgrade", "buy",
                "bullish", "record", "breakthrough", "success", "positive",
            ]
            negative_words = [
                "fall", "drop", "decline", "plunge", "crash", "miss", "weak",
                "loss", "cut", "downgrade", "sell", "bearish", "warning",
                "concern", "risk", "lawsuit", "investigation", "negative",
            ]

            for article in news[:15]:  # Analyze up to 15 recent articles
                title = article.get("title", "").lower()
                headlines.append(article.get("title", ""))

                pos_count = sum(1 for word in positive_words if word in title)
                neg_count = sum(1 for word in negative_words if word in title)

                if pos_count > neg_count:
                    positive += 1
                elif neg_count > pos_count:
                    negative += 1
                else:
                    neutral += 1

            total = positive + negative + neutral
            if total > 0:
                # Calculate weighted sentiment (-1 to 1)
                sentiment = (positive - negative) / total
            else:
                sentiment = 0

            return {
                "sentiment": sentiment,
                "count": len(news[:15]),
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "headlines": headlines[:5],  # Return top 5 headlines
            }

        except Exception as e:
            self.log_warning(f"Error analyzing news: {str(e)}")
            return {
                "sentiment": None,
                "count": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "headlines": [],
            }

    def _analyze_recommendations(self, ticker) -> dict[str, Any]:
        """Analyze analyst recommendations."""
        try:
            info = ticker.info

            # Get analyst recommendations
            buy = info.get("numberOfAnalystOpinions", 0) if info.get("recommendationKey") in ["buy", "strongBuy"] else 0
            hold = info.get("numberOfAnalystOpinions", 0) if info.get("recommendationKey") == "hold" else 0
            sell = info.get("numberOfAnalystOpinions", 0) if info.get("recommendationKey") in ["sell", "strongSell"] else 0

            # Try to get more detailed recommendations
            try:
                recommendations = ticker.recommendations
                if recommendations is not None and len(recommendations) > 0:
                    recent = recommendations.tail(30)  # Last 30 recommendations
                    if "To Grade" in recent.columns:
                        grades = recent["To Grade"].str.lower()
                        buy = sum(grades.isin(["buy", "strong buy", "outperform", "overweight"]))
                        hold = sum(grades.isin(["hold", "neutral", "equal-weight", "market perform"]))
                        sell = sum(grades.isin(["sell", "underperform", "underweight"]))
            except Exception:
                pass

            # Determine consensus rating
            total = buy + hold + sell
            if total > 0:
                if buy > hold + sell:
                    rating = "Buy"
                elif sell > buy + hold:
                    rating = "Sell"
                else:
                    rating = "Hold"
            else:
                rating = info.get("recommendationKey", "N/A")
                if rating:
                    rating = rating.replace("_", " ").title()

            # Get target price
            target_price = info.get("targetMeanPrice")

            return {
                "rating": rating,
                "buy": buy,
                "hold": hold,
                "sell": sell,
                "target_price": target_price,
            }

        except Exception as e:
            self.log_warning(f"Error analyzing recommendations: {str(e)}")
            return {
                "rating": None,
                "buy": 0,
                "hold": 0,
                "sell": 0,
                "target_price": None,
            }

    def _calculate_overall_sentiment(self, data: SentimentData) -> dict[str, Any]:
        """Calculate overall sentiment from all sources."""
        scores = []

        # News sentiment contribution
        if data.news_sentiment is not None:
            scores.append(data.news_sentiment)

        # Analyst rating contribution
        total_ratings = data.buy_ratings + data.hold_ratings + data.sell_ratings
        if total_ratings > 0:
            analyst_score = (data.buy_ratings - data.sell_ratings) / total_ratings
            scores.append(analyst_score)

        if scores:
            avg_score = sum(scores) / len(scores)
        else:
            avg_score = 0

        # Determine sentiment label
        if avg_score > 0.2:
            sentiment = "bullish"
        elif avg_score < -0.2:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "score": avg_score,
        }

    def _calculate_score(self, data: SentimentData) -> float:
        """Calculate sentiment score from 0-100."""
        score = 50.0  # Start neutral

        # News sentiment contribution (up to 25 points)
        if data.news_sentiment is not None:
            score += data.news_sentiment * 25

        # Analyst rating contribution (up to 25 points)
        total_ratings = data.buy_ratings + data.hold_ratings + data.sell_ratings
        if total_ratings > 0:
            analyst_score = (data.buy_ratings - data.sell_ratings) / total_ratings
            score += analyst_score * 25

        # Overall sentiment contribution
        if data.overall_sentiment == "bullish":
            score += 10
        elif data.overall_sentiment == "bearish":
            score -= 10

        return max(0, min(100, score))

    def _generate_summary(self, data: SentimentData, score: float) -> str:
        """Generate a summary of the sentiment analysis."""
        parts = []

        # Overall sentiment
        if data.overall_sentiment:
            parts.append(f"Overall market sentiment is {data.overall_sentiment}.")

        # News analysis
        if data.news_count > 0:
            parts.append(
                f"Analyzed {data.news_count} recent news articles: "
                f"{data.positive_news_count} positive, "
                f"{data.negative_news_count} negative, "
                f"{data.neutral_news_count} neutral."
            )

        # Analyst ratings
        total_ratings = data.buy_ratings + data.hold_ratings + data.sell_ratings
        if total_ratings > 0:
            parts.append(
                f"Analyst consensus: {data.analyst_rating} "
                f"({data.buy_ratings} buy, {data.hold_ratings} hold, {data.sell_ratings} sell)."
            )

        # Target price
        if data.target_price:
            parts.append(f"Average analyst target price: ${data.target_price:.2f}.")

        # Score interpretation
        if score >= 65:
            parts.append("Sentiment outlook is positive.")
        elif score <= 35:
            parts.append("Sentiment outlook is negative.")
        else:
            parts.append("Sentiment outlook is neutral.")

        return " ".join(parts)
