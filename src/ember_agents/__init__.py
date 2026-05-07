"""Ember Bio agent framework.

Provides the base Agent class, factory utilities, and built-in agents.
"""

from ember_agents.base import Agent
from ember_agents.discovery import DiscoveryAgent
from ember_agents.factory import AgentFactory, get_agent
from ember_agents.search import SearchAgent

__all__ = [
    "Agent",
    "AgentFactory",
    "BiosimilarAgent",
    "DiscoveryAgent",
    "SearchAgent",
    "get_agent",
]


def __getattr__(name: str):
    """Load optional agent exports lazily so independent packages can import."""
    if name == "BiosimilarAgent":
        from ember_agents.biosimilar import BiosimilarAgent

        return BiosimilarAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
