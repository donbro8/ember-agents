"""Abstract base agent for Ember Bio agent framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator


class Agent(ABC):
    """Base class for all Ember Bio agents.

    Subclasses must implement the async ``run`` method, which yields
    markdown-formatted strings as the agent produces output.
    """

    @abstractmethod
    async def run(self, query: str) -> AsyncGenerator[str, None]:
        """Execute the agent with the given query.

        Args:
            query: Natural-language query to process.

        Yields:
            Markdown-formatted output fragments.
        """
        ...  # pragma: no cover
