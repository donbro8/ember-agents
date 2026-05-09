"""Search pipeline package for dynamic drug discovery search.

Exports the pipeline components:
  Interpret → Classify → Gate → Fetch → Match

The unified agent is EmberAgent (ember_agents.agent), not SearchAgent.
SearchAgent is retained as internal infrastructure but not publicly exported.
"""

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
    "SearchGate",
]
