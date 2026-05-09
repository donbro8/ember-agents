"""EmberAgent: unified 6-phase pipeline merging Discovery, Search, and Biosimilar agents.

Pipeline phases:
  1. Interpret   — LLM-based signal extraction (IntentExtractor)
  2. Classify    — Resolver-based canonical classification (ClassificationOrchestrator)
  3. Gate        — Pre-search validation and breadth control (SearchGate)
  4. Fetch       — Parallel data source dispatch (FetchOrchestrator + BiologicSeedSource)
  5. Score       — Multi-signal ranking (MatchScorer)
  6. Render      — Structured markdown output (ResultRenderer)
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from typing import Any

from ember_agents.base import Agent
from ember_agents.render import ResultRenderer
from ember_agents.search.classify import ClassificationOrchestrator
from ember_agents.search.fetch import FetchOrchestrator
from ember_agents.search.gate import SearchGate
from ember_agents.search.interpret import IntentExtractor, RawSignals
from ember_agents.search.match import MatchScorer
from ember_agents.search.seed_source import BiologicSeedSource
from ember_agents.trace import ExecutionTrace, SourceStatus

try:
    from ember_data.models.result import CandidateResult, PatentJurisdiction
except ImportError:  # pragma: no cover — ember-data not installed in this env
    CandidateResult = None  # type: ignore[assignment,misc]
    PatentJurisdiction = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_target_label(target: Any) -> str | None:
    """Return a display label from a target object or None."""
    if target is None:
        return None
    if isinstance(target, str):
        return target
    for attr in ("label", "name", "identifier", "gene_name", "value"):
        val = getattr(target, attr, None)
        if val:
            return str(val)
    return None


def _candidate_to_result(scored: Any) -> Any:
    """Convert a ScoredCandidate to a CandidateResult.

    Falls back to a plain dict when ``ember_data.models.result`` is not
    installed, so the pipeline degrades gracefully in test environments.
    """
    cand = scored.candidate

    drug_name: str | None = getattr(cand, "drug_name", None)
    target_label: str | None = _extract_target_label(getattr(cand, "target", None))

    # Collect patent jurisdictions — each patent object may already be a
    # PatentJurisdiction (from BiologicSeedSource) or a raw Patent model.
    patent_jurisdictions: list[Any] = []
    for p in getattr(cand, "patents", []) or []:
        if PatentJurisdiction is not None and isinstance(p, PatentJurisdiction):
            patent_jurisdictions.append(p)

    # Source provenance names
    sources: list[str] = []
    for prov in getattr(cand, "contributing_sources", []) or []:
        name = getattr(prov, "source_name", None) or getattr(prov, "source_url", None)
        if name and name not in sources:
            sources.append(str(name))

    # Synthesis summary
    synthesis: str | None = getattr(cand, "synthesis_summary", None)

    # Risk flags
    risk_flags: list[str] = [str(f) for f in (getattr(cand, "risk_flags", []) or [])]

    overall_score: float | None = scored.overall_score if scored.overall_score > 0 else None

    if CandidateResult is not None:
        return CandidateResult(
            drug_name=drug_name,
            target=target_label,
            patents=patent_jurisdictions,
            overall_score=overall_score,
            sources_contributed=sources,
            risk_flags=risk_flags,
            synthesis_summary=synthesis,
        )

    # Fallback: a simple namespace so the renderer can still call getattr()
    class _FallbackResult:
        pass

    r = _FallbackResult()
    r.drug_name = drug_name  # type: ignore[attr-defined]
    r.fda_generic_name = None  # type: ignore[attr-defined]
    r.target = target_label  # type: ignore[attr-defined]
    r.display_label = drug_name or target_label or ""  # type: ignore[attr-defined]
    r.patents = patent_jurisdictions  # type: ignore[attr-defined]
    r.overall_score = overall_score  # type: ignore[attr-defined]
    r.sources_contributed = sources  # type: ignore[attr-defined]
    r.risk_flags = risk_flags  # type: ignore[attr-defined]
    r.synthesis_summary = synthesis  # type: ignore[attr-defined]
    r.indication = None  # type: ignore[attr-defined]
    return r


def _signals_to_dict(signals: RawSignals) -> dict:
    """Convert RawSignals to a plain dict for the ExecutionTrace."""
    result: dict = {}
    if signals.target:
        result["target"] = signals.target
    if signals.modality:
        result["modality"] = signals.modality
    if signals.indication:
        result["indication"] = signals.indication
    if signals.commercial:
        result["commercial"] = signals.commercial
    if signals.temporal is not None:
        result["temporal"] = str(signals.temporal)
    if signals.drug_name:
        result["drug_name"] = signals.drug_name
    return result


def _spec_to_classifications(spec: Any) -> dict:
    """Extract resolved classifications from a SearchSpec into a plain dict."""
    classifications: dict = {}

    target = getattr(spec, "target", None)
    if target is not None:
        label = _extract_target_label(target)
        if label:
            classifications["target"] = label

    ta = getattr(spec, "therapeutic_area", None)
    if ta is not None:
        label = _extract_target_label(ta)
        if label:
            classifications["therapeutic_area"] = label

    drug_names: list = getattr(spec, "drug_names", []) or []
    if drug_names:
        classifications["drug_names"] = list(drug_names)

    indications: list = getattr(spec, "indications", []) or []
    if indications:
        ind_labels = []
        for ind in indications:
            lbl = _extract_target_label(ind) or str(ind)
            ind_labels.append(lbl)
        if ind_labels:
            classifications["indications"] = ind_labels

    return classifications


# ---------------------------------------------------------------------------
# EmberAgent
# ---------------------------------------------------------------------------


class EmberAgent(Agent):
    """Unified agent implementing the 6-phase Interpret → Classify → Gate → Fetch → Score → Render pipeline.

    All dependencies are required — this constructor does not accept None for
    any pipeline component.  Callers are responsible for constructing and
    wiring their dependencies before instantiating EmberAgent.

    Parameters
    ----------
    intent_extractor:
        Extracts raw dimension-tagged signals from natural-language queries.
    classifier:
        Resolves raw signals into canonical SearchSpec identifiers.
    gate:
        Validates the SearchSpec before fetching (breadth control,
        disambiguation, and narrowing).
    fetcher:
        Dispatches parallel API calls across configured data sources and
        merges Candidate objects.  Must include a BiologicSeedSource.
    scorer:
        Scores and ranks Candidates using semantic + structured + evidence
        signals.
    seed_source:
        BiologicSeedSource instance used alongside *fetcher* to include
        biologic reference data in Fetch results.
    renderer:
        Formats ExecutionTrace + CandidateResult list into markdown output.
        When *None*, a default :class:`ResultRenderer` is created.
    """

    def __init__(
        self,
        *,
        intent_extractor: IntentExtractor,
        classifier: ClassificationOrchestrator,
        gate: SearchGate,
        fetcher: FetchOrchestrator,
        scorer: MatchScorer,
        seed_source: BiologicSeedSource,
        renderer: ResultRenderer | None = None,
    ) -> None:
        self._extractor = intent_extractor
        self._classifier = classifier
        self._gate = gate
        self._fetcher = fetcher
        self._scorer = scorer
        self._seed_source = seed_source
        self._renderer = renderer or ResultRenderer()

    # ------------------------------------------------------------------
    # Agent ABC implementation
    # ------------------------------------------------------------------

    async def run(self, query: str) -> AsyncGenerator[str, None]:  # type: ignore[override]
        """Execute the 6-phase pipeline and yield streaming markdown output.

        Yields
        ------
        str
            Markdown-formatted fragments: first the execution trace summary,
            then the ranked results.

        Args:
            query: Natural-language search query from the user.
        """
        start_time = time.monotonic()

        yield f"# Ember Search: {query}\n\n"

        # ----------------------------------------------------------------
        # Phase 1: Interpret
        # ----------------------------------------------------------------
        yield "_Phase 1: Extracting search signals…_\n\n"
        try:
            signals: RawSignals = await self._extractor.extract(query)
        except Exception as exc:  # noqa: BLE001
            yield f"> **Error** during signal extraction: {exc}\n"
            return

        # ----------------------------------------------------------------
        # Phase 2: Classify
        # ----------------------------------------------------------------
        yield "_Phase 2: Classifying signals…_\n\n"
        try:
            classification_result = await self._classifier.classify(signals)
        except Exception as exc:  # noqa: BLE001
            yield f"> **Error** during classification: {exc}\n"
            return

        spec = classification_result.spec

        # ----------------------------------------------------------------
        # Phase 3: Gate
        # ----------------------------------------------------------------
        yield "_Phase 3: Validating search spec…_\n\n"
        try:
            gate_result = await self._gate.check(spec)
        except Exception as exc:  # noqa: BLE001
            yield f"> **Error** during gate check: {exc}\n"
            return

        gate_outcome = "passed" if gate_result.passed else (gate_result.reason or "failed")

        if not gate_result.passed:
            if gate_result.narrowing:
                yield f"> **Search too broad** — {gate_result.narrowing.question}\n"
            else:
                yield f"> **Gate blocked:** {gate_outcome}\n"
            # Still emit trace so caller can inspect what happened
            duration = time.monotonic() - start_time
            trace = ExecutionTrace(
                extracted_signals=_signals_to_dict(signals),
                query_type=signals.query_type,
                resolved_classifications=_spec_to_classifications(spec),
                gate_outcome=gate_outcome,
                source_statuses=[],
                duration_seconds=duration,
            )
            yield self._renderer.render(trace, [])
            return

        # ----------------------------------------------------------------
        # Phase 4: Fetch
        # ----------------------------------------------------------------
        yield "_Phase 4: Fetching from data sources…_\n\n"
        source_statuses: list[SourceStatus] = []

        # Primary fetch via FetchOrchestrator
        try:
            candidates: list[Any] = await self._fetcher.fetch(spec)
            orch_count = len(candidates)
            source_statuses.append(
                SourceStatus(name="fetch_orchestrator", status="ok", result_count=orch_count)
            )
        except Exception as exc:  # noqa: BLE001
            yield f"> **Warning** — fetch orchestrator error: {exc}\n"
            candidates = []
            source_statuses.append(
                SourceStatus(name="fetch_orchestrator", status="error", result_count=0)
            )

        # BiologicSeedSource fetch (always runs alongside FetchOrchestrator)
        try:
            seed_results = await self._seed_source.fetch(spec, query_type=signals.query_type)
            seed_count = len(seed_results)
            source_statuses.append(
                SourceStatus(name="biologic_seed", status="ok" if seed_count else "empty", result_count=seed_count)
            )
            # Merge seed results by converting FetchResult → Candidate stubs
            # The seed FetchResults carry patent data; wrap them to look like
            # Candidates so the MatchScorer can score them.
            if seed_results:
                candidates = candidates + [_fetch_result_to_candidate_stub(r) for r in seed_results]
        except Exception as exc:  # noqa: BLE001
            yield f"> **Warning** — biologic seed source error: {exc}\n"
            source_statuses.append(
                SourceStatus(name="biologic_seed", status="error", result_count=0)
            )

        # ----------------------------------------------------------------
        # Phase 5: Score
        # ----------------------------------------------------------------
        yield "_Phase 5: Scoring candidates…_\n\n"
        try:
            scored_candidates = await self._scorer.score(
                candidates,
                spec,
                query_text=query,
                query_type=signals.query_type,
            )
        except Exception as exc:  # noqa: BLE001
            yield f"> **Error** during scoring: {exc}\n"
            scored_candidates = []

        # Convert ScoredCandidates to CandidateResult objects
        candidate_results: list[Any] = [_candidate_to_result(sc) for sc in scored_candidates]

        # ----------------------------------------------------------------
        # Phase 6: Render
        # ----------------------------------------------------------------
        duration = time.monotonic() - start_time
        trace = ExecutionTrace(
            extracted_signals=_signals_to_dict(signals),
            query_type=signals.query_type,
            resolved_classifications=_spec_to_classifications(spec),
            gate_outcome=gate_outcome,
            source_statuses=source_statuses,
            duration_seconds=duration,
        )

        yield self._renderer.render(trace, candidate_results)


# ---------------------------------------------------------------------------
# Helpers for seed result integration
# ---------------------------------------------------------------------------


def _fetch_result_to_candidate_stub(fetch_result: Any) -> Any:
    """Wrap a FetchResult as a minimal Candidate-compatible object for scoring.

    Preserves drug_name, patents, and matched_dimensions so the MatchScorer
    can evaluate structured and evidence signals.
    """
    from dataclasses import dataclass, field as dc_field

    @dataclass
    class _CandidateStub:
        id: str = ""
        drug_name: str | None = None
        target: object = None
        trials: list = dc_field(default_factory=list)
        patents: list = dc_field(default_factory=list)
        articles: list = dc_field(default_factory=list)
        scores: object = None
        risk_flags: list = dc_field(default_factory=list)
        confidence: float = 0.0
        matched_dimensions: list = dc_field(default_factory=list)
        contributing_sources: list = dc_field(default_factory=list)
        retrieved_at: object = None
        synthesis_summary: str | None = None

    import uuid

    return _CandidateStub(
        id=str(uuid.uuid4()),
        drug_name=fetch_result.drug_name or None,
        patents=list(fetch_result.patents),
        matched_dimensions=list(fetch_result.matched_dimensions),
        contributing_sources=list(fetch_result.provenance),
    )
