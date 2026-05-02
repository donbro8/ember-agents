"""Agent registry and factory for Ember Bio agents."""

from __future__ import annotations

from typing import Callable

from ember_agents.base import Agent

_REGISTRY: dict[str, type[Agent]] = {}


def register_agent(name: str) -> Callable[[type[Agent]], type[Agent]]:
    """Decorator that registers an agent class under *name*."""

    def decorator(cls: type[Agent]) -> type[Agent]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_agent(name: str) -> Agent:
    """Instantiate and return the agent registered under *name*.

    Raises:
        ValueError: If no agent is registered with that name.
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown agent: {name!r}")
    return cls()


class AgentFactory:
    """Convenience wrapper around the agent registry."""

    @staticmethod
    def list_agents() -> list[str]:
        """Return the names of all registered agents."""
        return sorted(_REGISTRY)
