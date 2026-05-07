"""Ember Bio agent framework.

Provides the base Agent class, factory utilities, and built-in agents.
"""

from ember_agents.base import Agent
from ember_agents.biosimilar import BiosimilarAgent
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
