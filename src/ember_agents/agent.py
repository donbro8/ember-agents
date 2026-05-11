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
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field as dc_field
from typing import Any

from ember_agents.base import Agent
from ember_agents.render import ResultRenderer
from ember_agents.search.classify import ClassificationOrchestrator
from ember_agents.synthesis import ResultSynthesizer
from ember_agents.search.fetch import FetchOrchestrator
from ember_agents.search.gate import SearchGate
from ember_agents.search.interpret import IntentExtractor, RawSignals
from ember_agents.search.match import MatchScorer
from ember_agents.search.seed_source import BiologicSeedSource
from ember_agents.trace import ExecutionTrace, SourceStatus

try:
    from ember_data.models.result import (
        ArticleSummary,
        CandidateResult,
        PatentJurisdiction,
        TrialSummary,
    )
except ImportError:  # pragma: no cover — ember-data not installed in this env
    CandidateResult = None  # type: ignore[assignment,misc]
    PatentJurisdiction = None  # type: ignore[assignment,misc]
    TrialSummary = None  # type: ignore[assignment,misc]
    ArticleSummary = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE_ORDER: dict[str, int] = {
    "Phase I": 1,
    "I": 1,
    "Phase 1": 1,
    "Phase II": 2,
    "II": 2,
    "Phase 2": 2,
    "Phase III": 3,
    "III": 3,
    "Phase 3": 3,
    "Phase IV": 4,
    "IV": 4,
    "Phase 4": 4,
}


# ---------------------------------------------------------------------------
# PipelineOutput dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineOutput:
    """Structured output of a completed EmberAgent pipeline run.

    Attributes:
        markdown: Full markdown-formatted output string.
        results: Ordered list of CandidateResult objects (highest-ranked first).
        trace: ExecutionTrace capturing pipeline metadata.
        query_type: Detected query type (e.g. "biosimilar_screen", "name_lookup").
        run_id: UUID4 string uniquely identifying this pipeline execution.
    """

    markdown: str
    results: list
    trace: Any  # ExecutionTrace
    query_type: str
    run_id: str
    synthesis_overview: str | None = None


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


def _expected_dimensions_from_spec(spec: Any) -> list[str]:
    """Return active scoring dimensions inferred from a SearchSpec-like object."""
    expected: list[str] = []
    if getattr(spec, "target", None) is not None:
        expected.append("target")
    if getattr(spec, "drug_names", None):
        expected.append("drug_name")
    if getattr(spec, "indications", None):
        expected.append("indication")
    if getattr(spec, "therapeutic_area", None) is not None:
        expected.append("therapeutic_area")
    if getattr(spec, "modality", None) is not None:
        expected.append("modality")
    if getattr(spec, "cell_line_class", None) is not None:
        expected.append("cell_line_class")
    if getattr(spec, "min_revenue_millions", None) is not None:
        expected.append("revenue")
    if getattr(spec, "patent_expiry_window", None) is not None:
        expected.append("patent_expiry_window")
    if getattr(spec, "jurisdictions", None):
        expected.append("jurisdiction")
    return expected


def _attach_additive_metadata(result_obj: Any, metadata: dict[str, Any]) -> Any:
    """Attach additive metadata while remaining compatible with strict models."""
    obj = result_obj
    for key, value in metadata.items():
        try:
            setattr(obj, key, value)
            continue
        except Exception:  # noqa: BLE001
            pass
        model_copy = getattr(obj, "model_copy", None)
        if callable(model_copy):
            try:
                obj = model_copy(update={key: value})
            except Exception:  # noqa: BLE001
                pass
    return obj


def _candidate_to_result(
    scored: Any,
    *,
    score_summary: Any | None = None,
    expected_dimensions: list[str] | None = None,
) -> Any:
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

    # Source provenance names and URLs
    sources: list[str] = []
    source_urls: list[str] = []
    for prov in getattr(cand, "contributing_sources", []) or []:
        name = getattr(prov, "source_name", None) or getattr(prov, "source_url", None)
        if name and name not in sources:
            sources.append(str(name))
        url = getattr(prov, "source_url", None)
        if url and str(url) not in source_urls:
            source_urls.append(str(url))

    # Synthesis summary
    synthesis: str | None = getattr(cand, "synthesis_summary", None)

    # Risk flags
    risk_flags: list[str] = [str(f) for f in (getattr(cand, "risk_flags", []) or [])]

    overall_score: float | None = (
        scored.overall_score if scored.overall_score > 0 else None
    )

    # Score breakdown
    structured_score: float | None = getattr(scored, "structured_score", None)
    semantic_score: float | None = getattr(scored, "semantic_score", None)
    evidence_score: float | None = getattr(scored, "evidence_score", None)

    # Build TrialSummary list
    trials_list: list[Any] = []
    if TrialSummary is not None:
        for t in getattr(cand, "trials", []) or []:
            try:
                trials_list.append(
                    TrialSummary(
                        nct_id=getattr(t, "nct_id", "") or "",
                        phase=str(getattr(t, "phase", "") or ""),
                        status=str(getattr(t, "status", "") or ""),
                        indication=(getattr(t, "conditions", None) or [None])[0]
                        if getattr(t, "conditions", None)
                        else None,
                        sponsor=getattr(t, "sponsor", None),
                        url=None,
                    )
                )
            except Exception:  # noqa: BLE001
                pass

    # Build ArticleSummary list
    articles_list: list[Any] = []
    if ArticleSummary is not None:
        for a in getattr(cand, "articles", []) or []:
            try:
                pub_date = getattr(a, "pub_date", None)
                year: int | None = pub_date.year if pub_date is not None else None
                articles_list.append(
                    ArticleSummary(
                        pmid=getattr(a, "pmid", None),
                        title=getattr(a, "title", "") or "",
                        journal=getattr(a, "journal", None),
                        year=year,
                        doi=getattr(a, "doi", None),
                        url=None,
                    )
                )
            except Exception:  # noqa: BLE001
                pass

    # Identity fields
    brand_names: list[str] = list(getattr(cand, "brand_names", []) or [])
    originator: str | None = getattr(cand, "originator", None) or None
    modality: str | None = getattr(cand, "modality", None) or None
    category: str | None = getattr(cand, "category", None) or None
    target_aliases: list[str] = list(getattr(cand, "target_aliases", []) or [])

    # Commercial fields
    annual_revenue_usd_millions: float | None = getattr(
        cand, "annual_revenue_usd_millions", None
    )
    revenue_year: int | None = getattr(cand, "revenue_year", None)
    biosimilar_competitors: list[str] = list(
        getattr(cand, "biosimilar_competitors", []) or []
    )
    has_approved_biosimilar: bool = (
        getattr(cand, "has_approved_biosimilar", False) or False
    )

    # FDA fields
    fda_generic_name: str | None = getattr(cand, "fda_generic_name", None) or None
    fda_brand_name: str | None = getattr(cand, "fda_brand_name", None) or None
    fda_manufacturer: str | None = getattr(cand, "fda_manufacturer", None) or None
    fda_therapeutic_area: str | None = (
        getattr(cand, "fda_therapeutic_area", None) or None
    )

    # Indications
    indication: list[str] = list(getattr(cand, "indications", []) or [])

    # Computed counts
    trial_count: int = len(trials_list)
    article_count: int = len(articles_list)
    biosimilar_competitor_count: int = len(biosimilar_competitors)

    # Compute latest_trial_phase
    latest_trial_phase: str | None = None
    best_phase_order: int = 0
    for t in getattr(cand, "trials", []) or []:
        phase_str = str(getattr(t, "phase", "") or "")
        phase_num = PHASE_ORDER.get(phase_str, 0)
        if phase_num > best_phase_order:
            best_phase_order = phase_num
            latest_trial_phase = phase_str

    matched_dimensions: list[str] = [
        str(d) for d in (getattr(cand, "matched_dimensions", []) or [])
    ]
    matched_set = {d.lower() for d in matched_dimensions}
    expected_dims = [d for d in (expected_dimensions or []) if d]
    missed_dimensions = [d for d in expected_dims if d.lower() not in matched_set]
    concrete_labels: dict[str, list[str]] = {}
    if target_label:
        concrete_labels["target"] = [target_label]
    if drug_name:
        concrete_labels["drug_name"] = [drug_name]
    if indication:
        concrete_labels["indication"] = [str(v) for v in indication if v]
    therapeutic_area = getattr(cand, "therapeutic_area", None)
    therapeutic_area_label = _extract_target_label(therapeutic_area)
    if therapeutic_area_label:
        concrete_labels["therapeutic_area"] = [therapeutic_area_label]
    if modality:
        concrete_labels["modality"] = [modality]

    threshold = getattr(score_summary, "threshold", None)
    query_type = getattr(score_summary, "query_type", None)
    suppression_metadata = {
        "suppressed": bool(getattr(scored, "suppressed", False)),
        "threshold": threshold,
        "query_type": query_type,
        "suppressed_candidates": getattr(score_summary, "suppressed_candidates", None),
        "total_candidates": getattr(score_summary, "total_candidates", None),
    }
    component_scores = {
        "semantic": semantic_score,
        "structured": structured_score,
        "evidence": evidence_score,
        "overall": overall_score,
    }
    evidence_summary = {
        "trial_count": trial_count,
        "article_count": article_count,
        "patent_count": len(patent_jurisdictions),
        "latest_trial_phase": latest_trial_phase,
    }
    match_explanations = {
        "matched_dimensions": matched_dimensions,
        "missed_dimensions": missed_dimensions,
    }

    if CandidateResult is not None:
        result_obj = CandidateResult(
            drug_name=drug_name,
            fda_generic_name=fda_generic_name,
            target=target_label,
            patents=patent_jurisdictions,
            overall_score=overall_score,
            sources_contributed=sources,
            source_urls=source_urls,
            risk_flags=risk_flags,
            synthesis_summary=synthesis,
            structured_score=structured_score,
            semantic_score=semantic_score,
            evidence_score=evidence_score,
            trials=trials_list,
            trial_count=trial_count,
            latest_trial_phase=latest_trial_phase,
            articles=articles_list,
            article_count=article_count,
            brand_names=brand_names,
            originator=originator,
            modality=modality,
            category=category,
            target_aliases=target_aliases,
            annual_revenue_usd_millions=annual_revenue_usd_millions,
            revenue_year=revenue_year,
            biosimilar_competitors=biosimilar_competitors,
            biosimilar_competitor_count=biosimilar_competitor_count,
            has_approved_biosimilar=has_approved_biosimilar,
            fda_brand_name=fda_brand_name,
            fda_manufacturer=fda_manufacturer,
            fda_therapeutic_area=fda_therapeutic_area,
            indication=indication,
        )
        return _attach_additive_metadata(
            result_obj,
            {
                "component_scores": component_scores,
                "suppression_metadata": suppression_metadata,
                "evidence_summary": evidence_summary,
                "matched_dimensions": matched_dimensions,
                "missed_dimensions": missed_dimensions,
                "concrete_labels": concrete_labels,
                "match_explanations": match_explanations,
            },
        )

    # Fallback: a simple namespace so the renderer can still call getattr()
    class _FallbackResult:
        pass

    r = _FallbackResult()
    r.drug_name = drug_name  # type: ignore[attr-defined]
    r.fda_generic_name = fda_generic_name  # type: ignore[attr-defined]
    r.target = target_label  # type: ignore[attr-defined]
    r.display_label = drug_name or target_label or ""  # type: ignore[attr-defined]
    r.patents = patent_jurisdictions  # type: ignore[attr-defined]
    r.overall_score = overall_score  # type: ignore[attr-defined]
    r.sources_contributed = sources  # type: ignore[attr-defined]
    r.source_urls = source_urls  # type: ignore[attr-defined]
    r.risk_flags = risk_flags  # type: ignore[attr-defined]
    r.synthesis_summary = synthesis  # type: ignore[attr-defined]
    r.indication = indication  # type: ignore[attr-defined]
    r.structured_score = structured_score  # type: ignore[attr-defined]
    r.semantic_score = semantic_score  # type: ignore[attr-defined]
    r.evidence_score = evidence_score  # type: ignore[attr-defined]
    r.trials = trials_list  # type: ignore[attr-defined]
    r.trial_count = trial_count  # type: ignore[attr-defined]
    r.latest_trial_phase = latest_trial_phase  # type: ignore[attr-defined]
    r.articles = articles_list  # type: ignore[attr-defined]
    r.article_count = article_count  # type: ignore[attr-defined]
    r.brand_names = brand_names  # type: ignore[attr-defined]
    r.originator = originator  # type: ignore[attr-defined]
    r.modality = modality  # type: ignore[attr-defined]
    r.category = category  # type: ignore[attr-defined]
    r.target_aliases = target_aliases  # type: ignore[attr-defined]
    r.annual_revenue_usd_millions = annual_revenue_usd_millions  # type: ignore[attr-defined]
    r.revenue_year = revenue_year  # type: ignore[attr-defined]
    r.biosimilar_competitors = biosimilar_competitors  # type: ignore[attr-defined]
    r.biosimilar_competitor_count = biosimilar_competitor_count  # type: ignore[attr-defined]
    r.has_approved_biosimilar = has_approved_biosimilar  # type: ignore[attr-defined]
    r.fda_brand_name = fda_brand_name  # type: ignore[attr-defined]
    r.fda_manufacturer = fda_manufacturer  # type: ignore[attr-defined]
    r.fda_therapeutic_area = fda_therapeutic_area  # type: ignore[attr-defined]
    r.component_scores = component_scores  # type: ignore[attr-defined]
    r.suppression_metadata = suppression_metadata  # type: ignore[attr-defined]
    r.evidence_summary = evidence_summary  # type: ignore[attr-defined]
    r.matched_dimensions = matched_dimensions  # type: ignore[attr-defined]
    r.missed_dimensions = missed_dimensions  # type: ignore[attr-defined]
    r.concrete_labels = concrete_labels  # type: ignore[attr-defined]
    r.match_explanations = match_explanations  # type: ignore[attr-defined]
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
    if signals.cell_line:
        result["cell_line"] = signals.cell_line
    if signals.jurisdiction:
        result["jurisdiction"] = signals.jurisdiction
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


def _gate_blocked_diagnostics(gate_reason: str, signals: RawSignals, spec: Any) -> str:
    signal_dict = _signals_to_dict(signals)
    lines = [f"> **Gate blocked:** {gate_reason}"]
    if signal_dict:
        lines.append(f"> Extracted signals: {signal_dict}")

    if gate_reason == "missing_core_fields":
        hints: list[str] = []
        if not getattr(spec, "drug_names", None):
            hints.append("drug name (e.g. adalimumab)")
        if getattr(spec, "target", None) is None:
            hints.append("target (e.g. PD-1)")
        if not getattr(spec, "indications", None):
            hints.append("indication (e.g. NSCLC)")
        if getattr(spec, "modality", None) is None:
            hints.append("modality (e.g. mAb)")
        if getattr(spec, "min_revenue_millions", None) is None:
            hints.append("commercial threshold (e.g. revenue > 1B)")
        if getattr(spec, "patent_expiry_window", None) is None:
            hints.append("patent window (e.g. after 2025 before 2028)")
        if not getattr(spec, "jurisdictions", None):
            hints.append("jurisdiction (e.g. US or EU)")
        if not getattr(spec, "cell_line_class", None):
            hints.append("cell-line class (e.g. mammalian)")
        if hints:
            lines.append("> Missing inputs: " + "; ".join(hints[:4]) + ".")
    elif gate_reason == "pending_disambiguations":
        lines.append(
            "> Missing input: choose one of the disambiguation options and retry."
        )

    return "\n".join(lines) + "\n"


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
        synthesizer: ResultSynthesizer | None = None,
    ) -> None:
        self._extractor = intent_extractor
        self._classifier = classifier
        self._gate = gate
        self._fetcher = fetcher
        self._scorer = scorer
        self._seed_source = seed_source
        self._renderer = renderer or ResultRenderer()
        self._synthesizer = synthesizer

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

        gate_outcome = (
            "passed" if gate_result.passed else (gate_result.reason or "failed")
        )

        if not gate_result.passed:
            if gate_result.narrowing:
                yield f"> **Search too broad** — {gate_result.narrowing.question}\n"
            else:
                yield _gate_blocked_diagnostics(gate_outcome, signals, spec)
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
                SourceStatus(
                    name="fetch_orchestrator", status="ok", result_count=orch_count
                )
            )
            source_statuses.extend(_fetcher_source_statuses(self._fetcher))
        except Exception as exc:  # noqa: BLE001
            yield f"> **Warning** — fetch orchestrator error: {exc}\n"
            candidates = []
            source_statuses.append(
                SourceStatus(name="fetch_orchestrator", status="error", result_count=0)
            )

        # BiologicSeedSource fetch (always runs alongside FetchOrchestrator)
        try:
            seed_results = await self._seed_source.fetch(
                spec, query_type=signals.query_type
            )
            seed_count = len(seed_results)
            source_statuses.append(
                SourceStatus(
                    name="biologic_seed",
                    status="ok" if seed_count else "empty",
                    result_count=seed_count,
                )
            )
            # Merge seed results by converting FetchResult → Candidate stubs
            # The seed FetchResults carry patent data; wrap them to look like
            # Candidates so the MatchScorer can score them.
            if seed_results:
                candidates = candidates + [
                    _fetch_result_to_candidate_stub(r) for r in seed_results
                ]
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
        score_summary = getattr(self._scorer, "last_score_summary", None)
        expected_dimensions = _expected_dimensions_from_spec(spec)
        candidate_results: list[Any] = [
            _candidate_to_result(
                sc,
                score_summary=score_summary,
                expected_dimensions=expected_dimensions,
            )
            for sc in scored_candidates
        ]

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

        yield self._renderer.render(trace, candidate_results, synthesis_overview=None)

    # ------------------------------------------------------------------
    # Non-streaming execute() API
    # ------------------------------------------------------------------

    async def execute(self, query: str) -> PipelineOutput:
        """Execute the full pipeline and return a structured PipelineOutput.

        Unlike :meth:`run`, this method collects all streaming fragments into a
        single markdown string and returns the full list of CandidateResult
        objects alongside the ExecutionTrace and pipeline metadata.

        Parameters
        ----------
        query:
            Natural-language search query from the user.

        Returns
        -------
        PipelineOutput
            Contains markdown output, ordered candidate results, execution
            trace, detected query_type, and a unique run_id.
        """
        run_id = str(uuid.uuid4())

        start_time = time.monotonic()

        # ----------------------------------------------------------------
        # Phase 1: Interpret
        # ----------------------------------------------------------------
        try:
            signals: RawSignals = await self._extractor.extract(query)
        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - start_time
            trace = ExecutionTrace(
                extracted_signals={},
                query_type="unknown",
                resolved_classifications={},
                gate_outcome="error",
                source_statuses=[],
                duration_seconds=duration,
            )
            markdown = f"# Ember Search: {query}\n\n> **Error** during signal extraction: {exc}\n"
            return PipelineOutput(
                markdown=markdown,
                results=[],
                trace=trace,
                query_type="unknown",
                run_id=run_id,
            )

        # ----------------------------------------------------------------
        # Phase 2: Classify
        # ----------------------------------------------------------------
        try:
            classification_result = await self._classifier.classify(signals)
        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - start_time
            trace = ExecutionTrace(
                extracted_signals=_signals_to_dict(signals),
                query_type=signals.query_type,
                resolved_classifications={},
                gate_outcome="error",
                source_statuses=[],
                duration_seconds=duration,
            )
            markdown = (
                f"# Ember Search: {query}\n\n> **Error** during classification: {exc}\n"
            )
            return PipelineOutput(
                markdown=markdown,
                results=[],
                trace=trace,
                query_type=signals.query_type,
                run_id=run_id,
            )

        spec = classification_result.spec

        # ----------------------------------------------------------------
        # Phase 3: Gate
        # ----------------------------------------------------------------
        try:
            gate_result = await self._gate.check(spec)
        except Exception as exc:  # noqa: BLE001
            duration = time.monotonic() - start_time
            trace = ExecutionTrace(
                extracted_signals=_signals_to_dict(signals),
                query_type=signals.query_type,
                resolved_classifications=_spec_to_classifications(spec),
                gate_outcome="error",
                source_statuses=[],
                duration_seconds=duration,
            )
            markdown = (
                f"# Ember Search: {query}\n\n> **Error** during gate check: {exc}\n"
            )
            return PipelineOutput(
                markdown=markdown,
                results=[],
                trace=trace,
                query_type=signals.query_type,
                run_id=run_id,
            )

        gate_outcome = (
            "passed" if gate_result.passed else (gate_result.reason or "failed")
        )

        if not gate_result.passed:
            duration = time.monotonic() - start_time
            trace = ExecutionTrace(
                extracted_signals=_signals_to_dict(signals),
                query_type=signals.query_type,
                resolved_classifications=_spec_to_classifications(spec),
                gate_outcome=gate_outcome,
                source_statuses=[],
                duration_seconds=duration,
            )
            rendered = self._renderer.render(trace, [])
            detail = (
                f"> **Search too broad** — {gate_result.narrowing.question}\n\n"
                if gate_result.narrowing is not None
                else _gate_blocked_diagnostics(gate_outcome, signals, spec) + "\n"
            )
            markdown = f"# Ember Search: {query}\n\n{detail}{rendered}"
            return PipelineOutput(
                markdown=markdown,
                results=[],
                trace=trace,
                query_type=signals.query_type,
                run_id=run_id,
            )

        # ----------------------------------------------------------------
        # Phase 4: Fetch
        # ----------------------------------------------------------------
        source_statuses: list[SourceStatus] = []

        try:
            candidates: list[Any] = await self._fetcher.fetch(spec)
            source_statuses.append(
                SourceStatus(
                    name="fetch_orchestrator", status="ok", result_count=len(candidates)
                )
            )
            source_statuses.extend(_fetcher_source_statuses(self._fetcher))
        except Exception:  # noqa: BLE001
            candidates = []
            source_statuses.append(
                SourceStatus(name="fetch_orchestrator", status="error", result_count=0)
            )

        try:
            seed_results = await self._seed_source.fetch(
                spec, query_type=signals.query_type
            )
            seed_count = len(seed_results)
            source_statuses.append(
                SourceStatus(
                    name="biologic_seed",
                    status="ok" if seed_count else "empty",
                    result_count=seed_count,
                )
            )
            if seed_results:
                candidates = candidates + [
                    _fetch_result_to_candidate_stub(r) for r in seed_results
                ]
        except Exception:  # noqa: BLE001
            source_statuses.append(
                SourceStatus(name="biologic_seed", status="error", result_count=0)
            )

        # ----------------------------------------------------------------
        # Phase 5: Score
        # ----------------------------------------------------------------
        try:
            scored_candidates = await self._scorer.score(
                candidates,
                spec,
                query_text=query,
                query_type=signals.query_type,
            )
        except Exception:  # noqa: BLE001
            scored_candidates = []

        candidate_results: list[Any] = [
            _candidate_to_result(sc) for sc in scored_candidates
        ]

        # ----------------------------------------------------------------
        # Phase 5.5: Synthesize (optional)
        # ----------------------------------------------------------------
        synthesis_overview: str | None = None
        if self._synthesizer is not None:
            try:
                synthesis_output = await self._synthesizer.synthesize(
                    query,
                    signals.query_type,
                    candidate_results,
                    None,
                )
                synthesis_overview = synthesis_output.overview
                for cr in candidate_results:
                    cid = getattr(cr, "canonical_id", None)
                    if cid and cid in synthesis_output.per_candidate:
                        cr.synthesis_summary = synthesis_output.per_candidate[cid]
            except Exception:  # noqa: BLE001
                pass  # graceful degradation — synthesis is optional

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

        markdown = self._renderer.render(
            trace, candidate_results, synthesis_overview=synthesis_overview
        )

        return PipelineOutput(
            markdown=markdown,
            results=candidate_results,
            trace=trace,
            query_type=signals.query_type,
            run_id=run_id,
            synthesis_overview=synthesis_overview,
        )


# ---------------------------------------------------------------------------
# Helpers for seed result integration
# ---------------------------------------------------------------------------


def _fetch_result_to_candidate_stub(fetch_result: Any) -> Any:
    """Wrap a FetchResult as a minimal Candidate-compatible object for scoring.

    Preserves drug_name, patents, and matched_dimensions so the MatchScorer
    can evaluate structured and evidence signals.  Also forwards all new
    FetchResult identity, commercial, FDA, and indication fields.
    """

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
        # Identity enrichment
        brand_names: list = dc_field(default_factory=list)
        originator: str | None = None
        modality: str | None = None
        category: str | None = None
        target_aliases: list = dc_field(default_factory=list)
        # Commercial
        annual_revenue_usd_millions: float | None = None
        revenue_year: int | None = None
        biosimilar_competitors: list = dc_field(default_factory=list)
        has_approved_biosimilar: bool = False
        # Indications
        indications: list = dc_field(default_factory=list)
        # FDA
        fda_generic_name: str | None = None
        fda_brand_name: str | None = None
        fda_manufacturer: str | None = None
        fda_therapeutic_area: str | None = None

    return _CandidateStub(
        id=str(uuid.uuid4()),
        drug_name=fetch_result.drug_name or None,
        patents=list(fetch_result.patents),
        matched_dimensions=list(fetch_result.matched_dimensions),
        contributing_sources=list(fetch_result.provenance),
        brand_names=list(getattr(fetch_result, "brand_names", []) or []),
        originator=getattr(fetch_result, "originator", None) or None,
        modality=getattr(fetch_result, "modality", None) or None,
        category=getattr(fetch_result, "category", None) or None,
        target_aliases=list(getattr(fetch_result, "target_aliases", []) or []),
        annual_revenue_usd_millions=getattr(
            fetch_result, "annual_revenue_usd_millions", None
        ),
        revenue_year=getattr(fetch_result, "revenue_year", None),
        biosimilar_competitors=list(
            getattr(fetch_result, "biosimilar_competitors", []) or []
        ),
        has_approved_biosimilar=getattr(fetch_result, "has_approved_biosimilar", False)
        or False,
        indications=list(getattr(fetch_result, "indications", []) or []),
        fda_generic_name=getattr(fetch_result, "fda_generic_name", None) or None,
        fda_brand_name=getattr(fetch_result, "fda_brand_name", None) or None,
        fda_manufacturer=getattr(fetch_result, "fda_manufacturer", None) or None,
        fda_therapeutic_area=getattr(fetch_result, "fda_therapeutic_area", None)
        or None,
    )


def _fetcher_source_statuses(fetcher: Any) -> list[SourceStatus]:
    """Extract structured per-source statuses from a fetcher if available."""
    raw = getattr(fetcher, "last_source_statuses", None)
    if not isinstance(raw, list):
        return []
    statuses: list[SourceStatus] = []
    for item in raw:
        if isinstance(item, SourceStatus):
            statuses.append(item)
            continue
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        status = item.get("status")
        result_count = item.get("result_count")
        if (
            isinstance(name, str)
            and isinstance(status, str)
            and isinstance(result_count, int)
        ):
            statuses.append(
                SourceStatus(name=name, status=status, result_count=result_count)
            )
    return statuses
