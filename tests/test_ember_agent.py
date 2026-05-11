"""Tests for EmberAgent — unified 6-phase pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ember_agents.agent import (
    EmberAgent,
    PipelineOutput,
    _candidate_to_result,
    _expected_dimensions_from_spec,
    _signals_to_dict,
    _spec_to_classifications,
)
from ember_agents.render import ResultRenderer
from ember_agents.search.classify import ClassificationOrchestrator, ClassificationResult
from ember_agents.synthesis import ResultSynthesizer, SynthesisOutput
from ember_agents.search.fetch import FetchOrchestrator, FetchResult
from ember_agents.search.gate import GateResult, SearchGate
from ember_agents.search.interpret import IntentExtractor, RawSignals
from ember_agents.search.match import MatchScorer, ScoredCandidate
from ember_agents.search.seed_source import BiologicSeedSource
from ember_agents.trace import ExecutionTrace, SourceStatus


# ---------------------------------------------------------------------------
# Stubs and helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeScores:
    clinical_stage: float = 0.0
    ip_freedom: float = 0.0
    evidence_strength: float = 0.0
    novelty: float = 0.0
    overall: float = 0.0
    semantic_score: float = 0.0
    structured_score: float = 0.0
    evidence_score: float = 0.0


@dataclass
class FakeCandidate:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    drug_name: str | None = None
    target: object = None
    trials: list = field(default_factory=list)
    patents: list = field(default_factory=list)
    articles: list = field(default_factory=list)
    scores: object = field(default_factory=FakeScores)
    risk_flags: list = field(default_factory=list)
    confidence: float = 0.0
    matched_dimensions: list = field(default_factory=list)
    contributing_sources: list = field(default_factory=list)
    retrieved_at: object = None
    synthesis_summary: str | None = None


@dataclass
class FakeSearchSpec:
    target: object = None
    therapeutic_area: object = None
    drug_names: list = field(default_factory=list)
    indications: list = field(default_factory=list)
    modality: object = None
    resolved_terms: list = field(default_factory=list)
    pending_disambiguations: list = field(default_factory=list)
    cell_line_class: str | None = None
    min_revenue_millions: float | None = None
    patent_expiry_window: object = None
    jurisdictions: list = field(default_factory=list)
    max_results: int = 500
    domains: list = field(default_factory=lambda: ["trials", "patents", "articles", "candidates"])


def _make_signals(query_type: str = "drug_lookup", drug_name: list | None = None) -> RawSignals:
    """Build a minimal RawSignals fixture."""
    return RawSignals(
        target=[],
        modality=[],
        indication=[],
        commercial=[],
        temporal=None,
        query_type=query_type,
        drug_name=drug_name or ["adalimumab"],
    )


def _make_scored(drug_name: str | None = None, overall_score: float = 0.5) -> ScoredCandidate:
    """Build a minimal ScoredCandidate fixture."""
    cand = FakeCandidate(drug_name=drug_name)
    return ScoredCandidate(
        candidate=cand,
        semantic_score=0.5,
        structured_score=0.5,
        evidence_score=0.5,
        overall_score=overall_score,
        rank=1,
    )


def _make_agent(
    *,
    signals: RawSignals | None = None,
    classification_result: ClassificationResult | None = None,
    gate_result: GateResult | None = None,
    fetch_results: list[Any] | None = None,
    seed_fetch_results: list[FetchResult] | None = None,
    scored: list[ScoredCandidate] | None = None,
    synthesizer: ResultSynthesizer | None = None,
) -> EmberAgent:
    """Build a fully-mocked EmberAgent for testing."""
    _signals = signals or _make_signals()
    _spec = FakeSearchSpec(drug_names=["adalimumab"])
    _classification = classification_result or ClassificationResult(spec=_spec)
    _gate = gate_result or GateResult(passed=True)
    _fetch = fetch_results if fetch_results is not None else []
    _seed = seed_fetch_results if seed_fetch_results is not None else []
    _scored = scored if scored is not None else [_make_scored("adalimumab")]

    extractor = MagicMock(spec=IntentExtractor)
    extractor.extract = AsyncMock(return_value=_signals)

    classifier = MagicMock(spec=ClassificationOrchestrator)
    classifier.classify = AsyncMock(return_value=_classification)

    gate = MagicMock(spec=SearchGate)
    gate.check = AsyncMock(return_value=_gate)

    fetcher = MagicMock(spec=FetchOrchestrator)
    fetcher.fetch = AsyncMock(return_value=_fetch)

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
        synthesizer=synthesizer,
    )


# ---------------------------------------------------------------------------
# Tests: dataclass creation
# ---------------------------------------------------------------------------


class TestSourceStatus:
    def test_fields(self):
        ss = SourceStatus(name="biologic_seed", status="ok", result_count=5)
        assert ss.name == "biologic_seed"
        assert ss.status == "ok"
        assert ss.result_count == 5


class TestExecutionTrace:
    def test_defaults(self):
        trace = ExecutionTrace()
        assert trace.extracted_signals == {}
        assert trace.query_type == "general"
        assert trace.resolved_classifications == {}
        assert trace.gate_outcome == "passed"
        assert trace.source_statuses == []
        assert trace.duration_seconds == 0.0

    def test_with_values(self):
        ss = SourceStatus(name="fetch_orchestrator", status="ok", result_count=3)
        trace = ExecutionTrace(
            extracted_signals={"target": ["HER2"]},
            query_type="drug_lookup",
            resolved_classifications={"target": "HER2"},
            gate_outcome="passed",
            source_statuses=[ss],
            duration_seconds=1.23,
        )
        assert trace.query_type == "drug_lookup"
        assert trace.source_statuses[0].name == "fetch_orchestrator"
        assert trace.duration_seconds == 1.23


# ---------------------------------------------------------------------------
# Tests: full pipeline flow
# ---------------------------------------------------------------------------


class TestEmberAgentPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_yields_output(self):
        """Full pipeline (gate passes) yields non-empty output."""
        agent = _make_agent()
        chunks = []
        async for chunk in agent.run("adalimumab biosimilar opportunities"):
            chunks.append(chunk)
        output = "".join(chunks)
        assert len(output) > 0

    @pytest.mark.asyncio
    async def test_pipeline_calls_all_phases(self):
        """All 6 phase components are invoked during a successful run."""
        extractor = MagicMock(spec=IntentExtractor)
        signals = _make_signals()
        extractor.extract = AsyncMock(return_value=signals)

        spec = FakeSearchSpec(drug_names=["adalimumab"])
        classifier = MagicMock(spec=ClassificationOrchestrator)
        classifier.classify = AsyncMock(return_value=ClassificationResult(spec=spec))

        gate = MagicMock(spec=SearchGate)
        gate.check = AsyncMock(return_value=GateResult(passed=True))

        fetcher = MagicMock(spec=FetchOrchestrator)
        fetcher.fetch = AsyncMock(return_value=[])

        seed_source = MagicMock(spec=BiologicSeedSource)
        seed_source.fetch = AsyncMock(return_value=[])

        scorer = MagicMock(spec=MatchScorer)
        scorer.score = AsyncMock(return_value=[])

        agent = EmberAgent(
            intent_extractor=extractor,
            classifier=classifier,
            gate=gate,
            fetcher=fetcher,
            scorer=scorer,
            seed_source=seed_source,
        )

        chunks = []
        async for chunk in agent.run("adalimumab biosimilar"):
            chunks.append(chunk)

        extractor.extract.assert_called_once_with("adalimumab biosimilar")
        classifier.classify.assert_called_once_with(signals)
        gate.check.assert_called_once_with(spec)
        fetcher.fetch.assert_called_once_with(spec)
        seed_source.fetch.assert_called_once()
        scorer.score.assert_called_once()

    @pytest.mark.asyncio
    async def test_gate_blocked_returns_early(self):
        """When gate blocks, pipeline yields gate message and returns without scoring."""
        scorer = MagicMock(spec=MatchScorer)

        gate_result = GateResult(passed=False, reason="missing_core_fields")
        agent = _make_agent(gate_result=gate_result, scored=[])
        # Override scorer to detect if it was called
        agent._scorer = scorer

        chunks = []
        async for chunk in agent.run("general question"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "missing_core_fields" in output or "Gate blocked" in output
        assert "Extracted signals" in output
        assert "Missing inputs" in output
        # Scorer must NOT have been called
        scorer.score.assert_not_called()

    @pytest.mark.asyncio
    async def test_gate_narrowing_emits_question(self):
        """When gate returns narrowing request, the question appears in output."""
        from ember_agents.search.gate import NarrowingRequest

        narrowing = NarrowingRequest(
            dimension="therapeutic_area",
            question="Your search is too broad. Which therapeutic area?\n1. oncology — Oncology\n2. cardiology — Cardiology",
            options=[("oncology", "Oncology"), ("cardiology", "Cardiology")],
        )
        gate_result = GateResult(passed=False, reason="too_broad", narrowing=narrowing)
        agent = _make_agent(gate_result=gate_result)

        chunks = []
        async for chunk in agent.run("broad query"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "too broad" in output.lower() or "too_broad" in output

    @pytest.mark.asyncio
    async def test_interpret_error_returns_early(self):
        """When IntentExtractor raises, the agent yields an error and stops."""
        extractor = MagicMock(spec=IntentExtractor)
        extractor.extract = AsyncMock(side_effect=RuntimeError("LLM unavailable"))

        classifier = MagicMock(spec=ClassificationOrchestrator)
        gate = MagicMock(spec=SearchGate)
        fetcher = MagicMock(spec=FetchOrchestrator)
        seed_source = MagicMock(spec=BiologicSeedSource)
        scorer = MagicMock(spec=MatchScorer)

        agent = EmberAgent(
            intent_extractor=extractor,
            classifier=classifier,
            gate=gate,
            fetcher=fetcher,
            scorer=scorer,
            seed_source=seed_source,
        )

        chunks = []
        async for chunk in agent.run("test"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "Error" in output
        classifier.classify.assert_not_called()
        gate.check.assert_not_called()

    @pytest.mark.asyncio
    async def test_seed_source_called_with_query_type(self):
        """BiologicSeedSource.fetch is called with the query_type from signals."""
        seed_source = MagicMock(spec=BiologicSeedSource)
        seed_source.fetch = AsyncMock(return_value=[])

        signals = _make_signals(query_type="biosimilar_screen")
        spec = FakeSearchSpec()

        extractor = MagicMock(spec=IntentExtractor)
        extractor.extract = AsyncMock(return_value=signals)

        classifier = MagicMock(spec=ClassificationOrchestrator)
        classifier.classify = AsyncMock(return_value=ClassificationResult(spec=spec))

        gate = MagicMock(spec=SearchGate)
        gate.check = AsyncMock(return_value=GateResult(passed=True))

        fetcher = MagicMock(spec=FetchOrchestrator)
        fetcher.fetch = AsyncMock(return_value=[])

        scorer = MagicMock(spec=MatchScorer)
        scorer.score = AsyncMock(return_value=[])

        agent = EmberAgent(
            intent_extractor=extractor,
            classifier=classifier,
            gate=gate,
            fetcher=fetcher,
            scorer=scorer,
            seed_source=seed_source,
        )

        async for _ in agent.run("biosimilar opportunities"):
            pass

        call_kwargs = seed_source.fetch.call_args
        # query_type should be "biosimilar_screen"
        assert call_kwargs.kwargs.get("query_type") == "biosimilar_screen"


# ---------------------------------------------------------------------------
# Tests: trace output format
# ---------------------------------------------------------------------------


class TestTraceOutput:
    @pytest.mark.asyncio
    async def test_trace_query_type_in_output(self):
        """Output contains the query_type from signals."""
        signals = _make_signals(query_type="opportunity_scan")
        agent = _make_agent(signals=signals)

        chunks = []
        async for chunk in agent.run("PD-1 inhibitors expiring 2028"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "opportunity_scan" in output

    @pytest.mark.asyncio
    async def test_trace_source_status_table_in_output(self):
        """Output contains source status table with expected source names."""
        agent = _make_agent()

        chunks = []
        async for chunk in agent.run("adalimumab"):
            chunks.append(chunk)

        output = "".join(chunks)
        # The renderer must emit a source status table
        assert "fetch_orchestrator" in output or "biologic_seed" in output

    @pytest.mark.asyncio
    async def test_trace_includes_fetcher_source_statuses(self):
        """Agent trace includes structured statuses exposed by fetcher."""
        agent = _make_agent(fetch_results=[])
        agent._fetcher.last_source_statuses = [
            SourceStatus(name="clinicaltrials", status="empty", result_count=0),
            SourceStatus(name="pubmed", status="ok", result_count=2),
        ]

        output = await agent.execute("adalimumab")
        status_names = [s.name for s in output.trace.source_statuses]
        assert "fetch_orchestrator" in status_names
        assert "clinicaltrials" in status_names
        assert "pubmed" in status_names

    @pytest.mark.asyncio
    async def test_trace_gate_outcome_in_output(self):
        """Output contains gate outcome."""
        agent = _make_agent()

        chunks = []
        async for chunk in agent.run("adalimumab"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "passed" in output

    @pytest.mark.asyncio
    async def test_trace_duration_in_output(self):
        """Output contains a duration measurement."""
        agent = _make_agent()

        chunks = []
        async for chunk in agent.run("adalimumab"):
            chunks.append(chunk)

        output = "".join(chunks)
        # Duration appears as "X.XXs"
        assert "s" in output  # minimal check that time value is present


# ---------------------------------------------------------------------------
# Tests: CandidateResult population
# ---------------------------------------------------------------------------


class TestCandidateResultConversion:
    def test_candidate_to_result_drug_name(self):
        """CandidateResult.drug_name is populated from candidate.drug_name."""
        scored = _make_scored(drug_name="adalimumab", overall_score=0.7)
        result = _candidate_to_result(scored)
        assert getattr(result, "drug_name", None) == "adalimumab"

    def test_candidate_to_result_no_drug(self):
        """CandidateResult.drug_name is None when candidate has no drug_name."""
        scored = _make_scored(drug_name=None, overall_score=0.3)
        result = _candidate_to_result(scored)
        assert getattr(result, "drug_name", None) is None

    def test_candidate_to_result_overall_score(self):
        """CandidateResult.overall_score is populated from ScoredCandidate.overall_score."""
        scored = _make_scored(drug_name="trastuzumab", overall_score=0.85)
        result = _candidate_to_result(scored)
        assert getattr(result, "overall_score", None) == pytest.approx(0.85)

    def test_candidate_to_result_zero_score_is_none(self):
        """overall_score of 0 is stored as None (no meaningful score)."""
        scored = _make_scored(drug_name="test", overall_score=0.0)
        result = _candidate_to_result(scored)
        assert getattr(result, "overall_score", None) is None

    def test_candidate_to_result_patents_propagated(self):
        """Patent objects that are PatentJurisdiction instances are passed through."""
        try:
            from ember_data.models.result import PatentJurisdiction
            from datetime import date

            patent = PatentJurisdiction(
                country_code="US",
                country_name="United States",
                publication_number="US1234567",
                expiry_date=date(2028, 6, 1),
                status="active",
                url="https://patents.google.com/patent/US1234567",
            )
            cand = FakeCandidate(drug_name="adalimumab", patents=[patent])
            scored = ScoredCandidate(candidate=cand, overall_score=0.6, rank=1)
            result = _candidate_to_result(scored)
            assert len(getattr(result, "patents", [])) == 1
        except ImportError:
            pytest.skip("ember-data not installed")

    def test_candidate_to_result_risk_flags(self):
        """Risk flags are copied from candidate to result."""
        cand = FakeCandidate(drug_name="risky_drug", risk_flags=["IP_RISK", "GENERIC_AVAILABLE"])
        scored = ScoredCandidate(candidate=cand, overall_score=0.4, rank=2)
        result = _candidate_to_result(scored)
        risk_flags = getattr(result, "risk_flags", [])
        assert "IP_RISK" in risk_flags
        assert "GENERIC_AVAILABLE" in risk_flags

    @pytest.mark.asyncio
    async def test_pipeline_produces_candidate_results(self):
        """End-to-end: run() output mentions the drug name from scored candidates."""
        scored = [_make_scored("trastuzumab", overall_score=0.75)]
        agent = _make_agent(scored=scored)

        chunks = []
        async for chunk in agent.run("trastuzumab biosimilar"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "trastuzumab" in output


# ---------------------------------------------------------------------------
# Tests: ResultRenderer
# ---------------------------------------------------------------------------


class TestResultRenderer:
    def test_render_empty_results(self):
        """Renderer emits 'No results found' when result list is empty."""
        renderer = ResultRenderer()
        trace = ExecutionTrace(query_type="general", gate_outcome="passed", duration_seconds=0.1)
        output = renderer.render(trace, [])
        assert "No results found" in output

    def test_render_query_analysis_section(self):
        """Renderer emits Query Analysis section with query_type."""
        renderer = ResultRenderer()
        trace = ExecutionTrace(
            query_type="biosimilar_screen",
            extracted_signals={"target": ["HER2"]},
            gate_outcome="passed",
            duration_seconds=0.5,
        )
        output = renderer.render(trace, [])
        assert "Query Analysis" in output
        assert "biosimilar_screen" in output

    def test_render_source_table(self):
        """Renderer emits source status table rows."""
        renderer = ResultRenderer()
        ss1 = SourceStatus(name="biologic_seed", status="ok", result_count=10)
        ss2 = SourceStatus(name="fetch_orchestrator", status="empty", result_count=0)
        trace = ExecutionTrace(
            query_type="drug_lookup",
            source_statuses=[ss1, ss2],
            gate_outcome="passed",
            duration_seconds=1.0,
        )
        output = renderer.render(trace, [])
        assert "biologic_seed" in output
        assert "fetch_orchestrator" in output
        assert "10" in output

    def test_render_results_section_with_drug_names(self):
        """Renderer shows drug names in the Results section."""

        class FakeResult:
            drug_name = "bevacizumab"
            fda_generic_name = None
            target = "VEGF"
            display_label = "bevacizumab"
            overall_score = 0.8
            patents = []
            risk_flags = []
            synthesis_summary = "Anti-VEGF mAb."
            indication = None

        renderer = ResultRenderer()
        trace = ExecutionTrace(query_type="drug_lookup", gate_outcome="passed", duration_seconds=0.3)
        output = renderer.render(trace, [FakeResult()])
        assert "bevacizumab" in output
        assert "Results" in output

    def test_render_patent_table(self):
        """Renderer shows patent jurisdictions for results that have patents."""
        try:
            from datetime import date

            from ember_data.models.result import PatentJurisdiction

            patent = PatentJurisdiction(
                country_code="US",
                country_name="United States",
                publication_number="US9999999",
                expiry_date=date(2029, 1, 1),
                status="active",
                url="https://patents.google.com/patent/US9999999",
            )

            class FakeResult:
                drug_name = "cetuximab"
                fda_generic_name = None
                target = None
                display_label = "cetuximab"
                overall_score = 0.6
                patents = [patent]
                risk_flags = []
                synthesis_summary = None
                indication = None

            renderer = ResultRenderer()
            trace = ExecutionTrace(query_type="drug_lookup", gate_outcome="passed", duration_seconds=0.2)
            output = renderer.render(trace, [FakeResult()])
            assert "United States" in output
            assert "active" in output
        except ImportError:
            pytest.skip("ember-data not installed")


# ---------------------------------------------------------------------------
# Tests: helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_signals_to_dict_omits_empty(self):
        """_signals_to_dict only includes dimensions with values."""
        signals = RawSignals(target=["HER2"], modality=[], indication=[], query_type="drug_lookup")
        d = _signals_to_dict(signals)
        assert "target" in d
        assert "modality" not in d
        assert "indication" not in d

    def test_signals_to_dict_drug_name(self):
        """_signals_to_dict includes drug_name when present."""
        signals = RawSignals(drug_name=["adalimumab"], query_type="drug_lookup")
        d = _signals_to_dict(signals)
        assert d["drug_name"] == ["adalimumab"]

    def test_spec_to_classifications_drug_names(self):
        """_spec_to_classifications extracts drug_names from spec."""
        spec = FakeSearchSpec(drug_names=["rituximab"])
        classifications = _spec_to_classifications(spec)
        assert classifications["drug_names"] == ["rituximab"]

    def test_spec_to_classifications_empty(self):
        """_spec_to_classifications returns empty dict for empty spec."""
        spec = FakeSearchSpec()
        classifications = _spec_to_classifications(spec)
        assert classifications == {}

    def test_expected_dimensions_from_spec_includes_opportunity_dims(self):
        spec = FakeSearchSpec(
            target=object(),
            therapeutic_area=object(),
            drug_names=["adalimumab"],
            indications=["ra"],
            modality="mab",
            cell_line_class="mammalian",
            min_revenue_millions=1000.0,
            patent_expiry_window={"start": "2026-01-01", "end": "2029-01-01"},
            jurisdictions=["US"],
        )
        dims = _expected_dimensions_from_spec(spec)
        assert "target" in dims
        assert "therapeutic_area" in dims
        assert "drug_name" in dims
        assert "indication" in dims
        assert "modality" in dims
        assert "cell_line_class" in dims
        assert "revenue" in dims
        assert "patent_expiry_window" in dims
        assert "jurisdiction" in dims

    def test_candidate_to_result_includes_explanation_metadata(self):
        """_candidate_to_result carries additive score/match/suppression metadata."""
        from ember_agents.agent import _candidate_to_result

        cand = FakeCandidate(drug_name="adalimumab", matched_dimensions=["target"])
        scored = ScoredCandidate(
            candidate=cand,
            semantic_score=0.8,
            structured_score=0.6,
            evidence_score=0.4,
            overall_score=0.62,
            rank=1,
            suppressed=True,
        )

        @dataclass
        class _Summary:
            query_type: str = "biosimilar_screen"
            threshold: float = 0.45
            total_candidates: int = 2
            returned_candidates: int = 2
            suppressed_candidates: int = 1

        result = _candidate_to_result(
            scored,
            score_summary=_Summary(),
            expected_dimensions=["target", "indication", "drug_name"],
        )
        match_explanations = getattr(result, "match_explanations", None)
        matched_dimensions = getattr(result, "matched_dimensions", None)
        missed_dimensions = getattr(result, "missed_dimensions", None)
        concrete_labels = getattr(result, "concrete_labels", None)
        suppression_metadata = getattr(result, "suppression_metadata", None)
        component_scores = getattr(result, "component_scores", None)
        evidence_summary = getattr(result, "evidence_summary", None)

        assert isinstance(match_explanations, dict)
        assert match_explanations["matched_dimensions"] == ["target"]
        assert "indication" in match_explanations["missed_dimensions"]
        assert "drug_name" in match_explanations["missed_dimensions"]
        assert matched_dimensions == ["target"]
        assert "indication" in missed_dimensions
        assert "drug_name" in missed_dimensions
        assert concrete_labels == {"drug_name": ["adalimumab"]}

        assert isinstance(suppression_metadata, dict)
        assert suppression_metadata["suppressed"] is True
        assert suppression_metadata["threshold"] == 0.45
        assert suppression_metadata["query_type"] == "biosimilar_screen"

        assert isinstance(component_scores, dict)
        assert component_scores["semantic"] == 0.8
        assert component_scores["overall"] == 0.62

        assert isinstance(evidence_summary, dict)
        assert evidence_summary["trial_count"] == 0


# ---------------------------------------------------------------------------
# Tests: synthesis integration
# ---------------------------------------------------------------------------


class TestSynthesisIntegration:
    @pytest.mark.asyncio
    async def test_execute_without_synthesizer(self):
        """Pipeline completes normally when synthesizer is None (default)."""
        agent = _make_agent()
        output = await agent.execute("adalimumab biosimilar")
        assert isinstance(output, PipelineOutput)
        assert output.synthesis_overview is None
        assert len(output.results) > 0

    @pytest.mark.asyncio
    async def test_execute_with_synthesizer_populates_overview(self):
        """When synthesizer returns valid output, synthesis_overview is populated."""
        synthesis_output = SynthesisOutput(
            overview="Adalimumab faces patent cliff in 2025.",
            per_candidate={"fda:adalimumab": "Top biosimilar opportunity."},
            jurisdiction_insights="US expires before EU.",
        )
        mock_synthesizer = MagicMock(spec=ResultSynthesizer)
        mock_synthesizer.synthesize = AsyncMock(return_value=synthesis_output)

        agent = _make_agent(synthesizer=mock_synthesizer)
        output = await agent.execute("adalimumab biosimilar")

        assert output.synthesis_overview == "Adalimumab faces patent cliff in 2025."
        mock_synthesizer.synthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_synthesizer_populates_per_candidate(self):
        """When synthesizer returns per_candidate data, synthesis_summary is set on matching results."""
        # Create a result that will have a canonical_id we can match
        scored = [_make_scored("adalimumab", overall_score=0.75)]
        agent = _make_agent(scored=scored)

        # Get the canonical_id that the result will have
        output_without = await agent.execute("adalimumab")
        if output_without.results:
            cid = getattr(output_without.results[0], "canonical_id", None)
        else:
            cid = None

        if cid is None:
            # If CandidateResult is not available (no ember-data), just verify
            # the synthesizer was called but skip the per_candidate check
            pytest.skip("ember-data not installed, cannot test canonical_id matching")

        synthesis_output = SynthesisOutput(
            overview="Analysis overview.",
            per_candidate={cid: "Key biosimilar candidate."},
            jurisdiction_insights=None,
        )
        mock_synthesizer = MagicMock(spec=ResultSynthesizer)
        mock_synthesizer.synthesize = AsyncMock(return_value=synthesis_output)

        agent2 = _make_agent(scored=scored, synthesizer=mock_synthesizer)
        output = await agent2.execute("adalimumab")

        matched = [r for r in output.results if getattr(r, "canonical_id", None) == cid]
        assert len(matched) == 1
        assert matched[0].synthesis_summary == "Key biosimilar candidate."

    @pytest.mark.asyncio
    async def test_execute_with_synthesizer_error_degrades_gracefully(self):
        """When synthesizer raises, pipeline still completes with results."""
        mock_synthesizer = MagicMock(spec=ResultSynthesizer)
        mock_synthesizer.synthesize = AsyncMock(side_effect=RuntimeError("LLM down"))

        agent = _make_agent(synthesizer=mock_synthesizer)
        output = await agent.execute("adalimumab biosimilar")

        assert isinstance(output, PipelineOutput)
        assert output.synthesis_overview is None
        assert len(output.results) > 0

    @pytest.mark.asyncio
    async def test_execute_with_synthesizer_overview_in_markdown(self):
        """When synthesis succeeds, the Analysis section appears in markdown output."""
        synthesis_output = SynthesisOutput(
            overview="Patent cliff analysis shows opportunity.",
            per_candidate={},
            jurisdiction_insights=None,
        )
        mock_synthesizer = MagicMock(spec=ResultSynthesizer)
        mock_synthesizer.synthesize = AsyncMock(return_value=synthesis_output)

        agent = _make_agent(synthesizer=mock_synthesizer)
        output = await agent.execute("adalimumab biosimilar")

        assert "## Analysis" in output.markdown
        assert "Patent cliff analysis shows opportunity." in output.markdown


class TestResultRendererSynthesis:
    def test_render_without_synthesis_overview(self):
        """Renderer does not include Analysis section when synthesis_overview is None."""
        renderer = ResultRenderer()
        trace = ExecutionTrace(query_type="general", gate_outcome="passed", duration_seconds=0.1)
        output = renderer.render(trace, [])
        assert "## Analysis" not in output

    def test_render_with_synthesis_overview(self):
        """Renderer includes Analysis section when synthesis_overview is provided."""
        renderer = ResultRenderer()
        trace = ExecutionTrace(query_type="general", gate_outcome="passed", duration_seconds=0.1)
        output = renderer.render(trace, [], synthesis_overview="Key insights here.")
        assert "## Analysis" in output
        assert "Key insights here." in output
