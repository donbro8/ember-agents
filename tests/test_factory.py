"""Tests for agent factory and registry."""

import pytest

from ember_agents import AgentFactory, get_agent
from ember_agents.base import Agent


class TestGetAgent:
    """Tests for the get_agent function."""

    def test_get_discovery_agent(self):
        """get_agent('discovery') returns a DiscoveryAgent instance."""
        agent = get_agent("discovery")
        assert isinstance(agent, Agent)
        assert type(agent).__name__ == "DiscoveryAgent"

    def test_get_nonexistent_agent_raises(self):
        """get_agent with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown agent"):
            get_agent("nonexistent")


class TestAgentFactory:
    """Tests for the AgentFactory class."""

    def test_list_agents_includes_discovery(self):
        """AgentFactory.list_agents() includes 'discovery'."""
        agents = AgentFactory.list_agents()
        assert "discovery" in agents
