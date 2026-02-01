"""Helper utility functions."""

import sys
from typing import Optional

from loguru import logger


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with loguru."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    logger.add(
        "logs/stock_analysis.log",
        rotation="10 MB",
        retention="7 days",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )


def format_currency(value: Optional[float], symbol: str = "$") -> str:
    """Format a number as currency."""
    if value is None:
        return "N/A"
    if value >= 1_000_000_000_000:
        return f"{symbol}{value / 1_000_000_000_000:.2f}T"
    elif value >= 1_000_000_000:
        return f"{symbol}{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{symbol}{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{symbol}{value / 1_000:.2f}K"
    else:
        return f"{symbol}{value:.2f}"


def format_percentage(value: Optional[float], decimals: int = 2) -> str:
    """Format a decimal as percentage."""
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_large_number(value: Optional[float]) -> str:
    """Format a large number with abbreviations."""
    if value is None:
        return "N/A"
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    elif value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.2f}K"
    else:
        return f"{value:.2f}"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """Calculate percentage change between two values."""
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / old_value


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))
