"""Search pipeline package for dynamic drug discovery search.

Exports the full pipeline:
  Interpret → Classify → Gate → Fetch → Match

and the top-level SearchAgent orchestrator.
"""

from ember_agents.search.agent import SearchAgent
from ember_agents.search.classify import ClassificationOrchestrator
from ember_agents.search.fetch import FetchOrchestrator
from ember_agents.search.gate import SearchGate
from ember_agents.search.interpret import IntentExtractor
from ember_agents.search.match import MatchScorer

__all__ = [
    "ClassificationOrchestrator",
    "FetchOrchestrator",
    "IntentExtractor",
    "MatchScorer",
    "SearchAgent",
    "SearchGate",
]
