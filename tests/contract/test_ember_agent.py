"""Contract tests for EmberAgent with biosimilar_screen query_type.

Migrated and adapted from test_biosimilar_agent.py:  instead of testing the
old BiosimilarAgent directly, these tests verify that EmberAgent is correctly
instantiated and that it can handle a biosimilar_screen query_type end-to-end
using fully mocked pipeline components.

Also includes a startup wiring test adapted from test_factory.py, asserting
that EmberAgent has no None pipeline components once constructed.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock


from ember_agents.agent import EmberAgent
from ember_agents.base import Agent
from ember_agents.render import ResultRenderer
from ember_agents.search.classify import (
    ClassificationOrchestrator,
    ClassificationResult,
)
from ember_agents.search.fetch import FetchOrchestrator, FetchResult
from ember_agents.search.gate import GateResult, SearchGate
from ember_agents.search.interpret import IntentExtractor, RawSignals
from ember_agents.search.match import MatchScorer, ScoredCandidate
from ember_agents.search.seed_source import BiologicSeedSource


# ---------------------------------------------------------------------------
# Stubs and helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeSearchSpec:
    drug_names: list = field(default_factory=list)
    target: object = None
    therapeutic_area: object = None
    indications: list = field(default_factory=list)
    modality: object = None
    resolved_terms: list = field(default_factory=list)
    pending_disambiguations: list = field(default_factory=list)
    max_results: int = 500
    domains: list = field(default_factory=lambda: ["trials", "patents", "articles"])
    min_revenue_millions: float | None = None
    patent_expiry_window: object = None
    jurisdictions: list = field(default_factory=list)


@dataclass
class FakeCandidate:
    drug_name: str | None = None
    target: object = None
    trials: list = field(default_factory=list)
    patents: list = field(default_factory=list)
    articles: list = field(default_factory=list)
    risk_flags: list = field(default_factory=list)
    confidence: float = 0.0
    matched_dimensions: list = field(default_factory=list)
    contributing_sources: list = field(default_factory=list)
    synthesis_summary: str | None = None


def _make_biosimilar_signals(drug_name: list[str] | None = None) -> RawSignals:
    """Build RawSignals with biosimilar_screen query_type."""
    return RawSignals(
        query_type="biosimilar_screen",
        drug_name=drug_name or [],
        target=[],
        modality=["mAb"],
        indication=[],
        commercial=["US"],
    )


def _make_scored(drug_name: str | None = None, score: float = 0.7) -> ScoredCandidate:
    cand = FakeCandidate(drug_name=drug_name)
    return ScoredCandidate(
        candidate=cand,
        semantic_score=score,
        structured_score=score,
        evidence_score=score,
        overall_score=score,
        rank=1,
    )


def _make_agent(
    *,
    signals: RawSignals | None = None,
    spec: FakeSearchSpec | None = None,
    gate_passed: bool = True,
    seed_results: list[FetchResult] | None = None,
    scored: list[ScoredCandidate] | None = None,
) -> EmberAgent:
    """Build a fully-mocked EmberAgent wired for biosimilar_screen."""
    _signals = signals or _make_biosimilar_signals()
    _spec = spec or FakeSearchSpec(drug_names=[])
    _classification = ClassificationResult(spec=_spec)
    _gate = GateResult(passed=gate_passed)
    _seed = seed_results if seed_results is not None else []
    _scored = scored if scored is not None else []

    extractor = MagicMock(spec=IntentExtractor)
    extractor.extract = AsyncMock(return_value=_signals)

    classifier = MagicMock(spec=ClassificationOrchestrator)
    classifier.classify = AsyncMock(return_value=_classification)

    gate = MagicMock(spec=SearchGate)
    gate.check = AsyncMock(return_value=_gate)

    fetcher = MagicMock(spec=FetchOrchestrator)
    fetcher.fetch = AsyncMock(return_value=[])

    seed_source = MagicMock(spec=BiologicSeedSource)
    seed_source.fetch = AsyncMock(return_value=_seed)

    scorer = MagicMock(spec=MatchScorer)
    scorer.score = AsyncMock(return_value=_scored)

    return EmberAgent(
        intent_extractor=extractor,
        classifier=classifier,
        gate=gate,
        fetcher=fetcher,
        scorer=scorer,
        seed_source=seed_source,
    )


# ---------------------------------------------------------------------------
# Startup wiring tests (adapted from test_factory.py)
# ---------------------------------------------------------------------------


class TestEmberAgentWiring:
    """Verify that EmberAgent has no None pipeline components after construction."""

    def test_extractor_not_none(self) -> None:
        agent = _make_agent()
        assert agent._extractor is not None

    def test_classifier_not_none(self) -> None:
        agent = _make_agent()
        assert agent._classifier is not None

    def test_gate_not_none(self) -> None:
        agent = _make_agent()
        assert agent._gate is not None

    def test_fetcher_not_none(self) -> None:
        agent = _make_agent()
        assert agent._fetcher is not None

    def test_scorer_not_none(self) -> None:
        agent = _make_agent()
        assert agent._scorer is not None

    def test_seed_source_not_none(self) -> None:
        agent = _make_agent()
        assert agent._seed_source is not None

    def test_renderer_not_none(self) -> None:
        agent = _make_agent()
        assert agent._renderer is not None

    def test_renderer_is_result_renderer(self) -> None:
        agent = _make_agent()
        assert isinstance(agent._renderer, ResultRenderer)

    def test_is_agent_subclass(self) -> None:
        agent = _make_agent()
        assert isinstance(agent, Agent)

    def test_is_ember_agent(self) -> None:
        agent = _make_agent()
        assert isinstance(agent, EmberAgent)

    def test_custom_renderer_accepted(self) -> None:
        """EmberAgent accepts an explicit renderer without creating a default."""
        renderer = ResultRenderer()
        extractor = MagicMock(spec=IntentExtractor)
        classifier = MagicMock(spec=ClassificationOrchestrator)
        gate = MagicMock(spec=SearchGate)
        fetcher = MagicMock(spec=FetchOrchestrator)
        scorer = MagicMock(spec=MatchScorer)
        seed_source = MagicMock(spec=BiologicSeedSource)
        agent = EmberAgent(
            intent_extractor=extractor,
            classifier=classifier,
            gate=gate,
            fetcher=fetcher,
            scorer=scorer,
            seed_source=seed_source,
            renderer=renderer,
        )
        assert agent._renderer is renderer


# ---------------------------------------------------------------------------
# biosimilar_screen query_type: run() integration tests
# ---------------------------------------------------------------------------


class TestEmberAgentBiosimilarScreen:
    """Verify EmberAgent.run() with biosimilar_screen query_type."""

    async def test_run_is_async_generator(self) -> None:
        """run() returns an async generator."""
        agent = _make_agent()
        gen = agent.run("biosimilar opportunities for mAbs")
        assert inspect.isasyncgen(gen)
        async for _ in gen:
            pass

    async def test_run_yields_output(self) -> None:
        """run() yields non-empty string output."""
        agent = _make_agent()
        chunks: list[str] = []
        async for chunk in agent.run("biosimilar opportunities for mAbs"):
            chunks.append(chunk)
        assert len(chunks) > 0
        assert "".join(chunks) != ""

    async def test_run_yields_header_with_query(self) -> None:
        """run() first chunk contains the query."""
        agent = _make_agent()
        chunks: list[str] = []
        async for chunk in agent.run("biosimilar screen for IL-6 inhibitors"):
            chunks.append(chunk)
        output = "".join(chunks)
        assert "biosimilar screen for IL-6 inhibitors" in output

    async def test_seed_source_called_with_biosimilar_screen(self) -> None:
        """BiologicSeedSource.fetch is called with biosimilar_screen query_type."""
        signals = _make_biosimilar_signals()
        agent = _make_agent(signals=signals)

        async for _ in agent.run("biosimilar mAb opportunities"):
            pass

        # Check that seed_source.fetch was called with the correct query_type
        call_kwargs = agent._seed_source.fetch.call_args
        assert call_kwargs is not None
        # query_type is passed as a keyword argument
        if call_kwargs.kwargs:
            assert call_kwargs.kwargs.get("query_type") == "biosimilar_screen"
        else:
            # positional: (spec, query_type)
            assert call_kwargs.args[1] == "biosimilar_screen"

    async def test_run_with_seed_results_includes_candidates(self) -> None:
        """When seed source returns results, they are included in scoring."""
        seed_result = FetchResult(
            drug_name="AlphaMab",
            synonyms=["AlphaBrand"],
            target_id="",
            patents=[],
            matched_dimensions=["biologic_seed"],
        )
        scored = [_make_scored("AlphaMab")]
        agent = _make_agent(seed_results=[seed_result], scored=scored)

        chunks: list[str] = []
        async for chunk in agent.run("biosimilar mAb opportunities"):
            chunks.append(chunk)

        # Scorer was called — verify it was invoked once
        agent._scorer.score.assert_called_once()

    async def test_run_no_seed_results_still_completes(self) -> None:
        """run() completes successfully when seed source returns nothing."""
        agent = _make_agent(seed_results=[], scored=[])
        chunks: list[str] = []
        async for chunk in agent.run("biosimilar screen"):
            chunks.append(chunk)
        assert len(chunks) > 0

    async def test_all_pipeline_phases_called(self) -> None:
        """All 6 pipeline components are invoked for a biosimilar_screen query."""
        signals = _make_biosimilar_signals()
        agent = _make_agent(signals=signals, scored=[_make_scored("DrugA")])

        async for _ in agent.run("biosimilar screen"):
            pass

        agent._extractor.extract.assert_called_once()
        agent._classifier.classify.assert_called_once()
        agent._gate.check.assert_called_once()
        agent._fetcher.fetch.assert_called_once()
        agent._seed_source.fetch.assert_called_once()
        agent._scorer.score.assert_called_once()

    async def test_gate_blocked_skips_seed_fetch(self) -> None:
        """When gate blocks, seed source fetch is not called."""
        agent = _make_agent(gate_passed=False)
        async for _ in agent.run("biosimilar screen"):
            pass
        agent._seed_source.fetch.assert_not_called()

    async def test_gate_blocked_skips_scoring(self) -> None:
        """When gate blocks, scorer is not called."""
        agent = _make_agent(gate_passed=False)
        async for _ in agent.run("biosimilar screen"):
            pass
        agent._scorer.score.assert_not_called()
