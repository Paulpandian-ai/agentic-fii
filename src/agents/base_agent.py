"""Base agent class that all servant agents inherit from."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from loguru import logger

from src.models.schemas import AgentResult, AgentStatus


class BaseAgent(ABC):
    """
    Abstract base class for all analysis agents.

    Each servant agent must implement the `analyze` method to perform
    its specific analysis function. The base class provides common
    functionality like logging, timing, and error handling.
    """

    def __init__(self, name: str, agent_type: str):
        """
        Initialize the base agent.

        Args:
            name: Unique name for this agent instance
            agent_type: Type of analysis this agent performs
        """
        self.name = name
        self.agent_type = agent_type
        self.status = AgentStatus.IDLE
        self._start_time: Optional[float] = None
        self._errors: list[str] = []

        logger.info(f"Initialized {self.agent_type} agent: {self.name}")

    @property
    def is_running(self) -> bool:
        """Check if the agent is currently running."""
        return self.status == AgentStatus.RUNNING

    @abstractmethod
    async def analyze(self, symbol: str, **kwargs) -> dict[str, Any]:
        """
        Perform the agent's specific analysis.

        This method must be implemented by each servant agent to perform
        its unique analysis function.

        Args:
            symbol: Stock ticker symbol to analyze
            **kwargs: Additional parameters specific to each agent

        Returns:
            Dictionary containing analysis results
        """
        pass

    async def execute(self, symbol: str, **kwargs) -> AgentResult:
        """
        Execute the agent's analysis with timing and error handling.

        This is the main entry point for running an agent. It handles:
        - Status management
        - Execution timing
        - Error catching and logging
        - Result packaging

        Args:
            symbol: Stock ticker symbol to analyze
            **kwargs: Additional parameters passed to analyze()

        Returns:
            AgentResult containing the analysis results and metadata
        """
        self.status = AgentStatus.RUNNING
        self._start_time = time.time()
        self._errors = []

        logger.info(f"[{self.name}] Starting analysis for {symbol}")

        try:
            # Run the specific analysis
            result_data = await self.analyze(symbol, **kwargs)

            execution_time = time.time() - self._start_time
            self.status = AgentStatus.COMPLETED

            # Extract score and summary if present
            score = result_data.pop("score", None)
            summary = result_data.pop("summary", None)

            logger.info(
                f"[{self.name}] Completed analysis for {symbol} "
                f"in {execution_time:.2f}s"
            )

            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                status=self.status,
                execution_time=execution_time,
                data=result_data,
                score=score,
                summary=summary,
                errors=self._errors,
            )

        except Exception as e:
            execution_time = time.time() - self._start_time
            self.status = AgentStatus.FAILED
            error_msg = f"Error in {self.name}: {str(e)}"
            self._errors.append(error_msg)

            logger.error(f"[{self.name}] {error_msg}")

            return AgentResult(
                agent_name=self.name,
                agent_type=self.agent_type,
                status=self.status,
                execution_time=execution_time,
                data={},
                score=None,
                summary=f"Analysis failed: {str(e)}",
                errors=self._errors,
            )

    def log_warning(self, message: str) -> None:
        """Log a warning message and add to errors list."""
        logger.warning(f"[{self.name}] {message}")
        self._errors.append(f"Warning: {message}")

    def log_info(self, message: str) -> None:
        """Log an info message."""
        logger.info(f"[{self.name}] {message}")

    def reset(self) -> None:
        """Reset the agent to idle state."""
        self.status = AgentStatus.IDLE
        self._start_time = None
        self._errors = []

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', status={self.status})"
