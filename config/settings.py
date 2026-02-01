"""Application settings and configuration."""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = Field(default="Stock Analysis Platform", description="Application name")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # Agent Configuration
    execution_mode: str = Field(
        default="parallel",
        description="Agent execution mode: 'parallel' or 'sequential'",
    )
    default_lookback_days: int = Field(
        default=252,
        description="Default lookback period for historical analysis",
    )

    # Weights for overall scoring
    weight_fundamental: float = Field(default=0.30, description="Weight for fundamental analysis")
    weight_technical: float = Field(default=0.25, description="Weight for technical analysis")
    weight_sentiment: float = Field(default=0.20, description="Weight for sentiment analysis")
    weight_risk: float = Field(default=0.25, description="Weight for risk assessment")

    # Risk Parameters
    risk_free_rate: float = Field(default=0.05, description="Risk-free rate for calculations")

    # API Keys (optional, for extended functionality)
    alpha_vantage_api_key: Optional[str] = Field(
        default=None,
        description="Alpha Vantage API key for additional data",
    )
    news_api_key: Optional[str] = Field(
        default=None,
        description="News API key for news sentiment",
    )

    # API Server (optional)
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def agent_weights(self) -> dict[str, float]:
        """Get agent weights as a dictionary."""
        return {
            "fundamental": self.weight_fundamental,
            "technical": self.weight_technical,
            "sentiment": self.weight_sentiment,
            "risk": self.weight_risk,
        }


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
