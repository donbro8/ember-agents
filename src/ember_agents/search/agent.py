"""SearchAgent: top-level orchestrator for the dynamic drug discovery search pipeline.

Orchestrates the full Interpret → Classify → Gate → Fetch → Match pipeline and
yields streaming markdown output with ranked candidates, matched dimensions,
contributing sources, provenance, and a synthesis summary.

Handles the Gate's closed-option disambiguation/narrowing loop: when the Gate
returns a disambiguation or narrowing request, the agent surfaces the question to
the caller and re-runs Classify → Gate with the user's selection.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from ember_agents.base import Agent
from ember_agents.factory import register_agent
from ember_agents.search.classify import ClassificationOrchestrator, DisambiguationRequest
from ember_agents.search.fetch import FetchOrchestrator
from ember_agents.search.gate import GateResult, NarrowingRequest, SearchGate
from ember_agents.search.interpret import IntentExtractor, RawSignals
from ember_agents.search.match import MatchScorer, ScoredCandidate, semantic_scoring_available
from ember_agents.search.urls import drug_label_url, patent_url, pubmed_url, trial_url, uniprot_url

# Maximum number of disambiguation/narrowing loops before giving up.
_MAX_GATE_LOOPS = 5


def _format_disambiguations(requests: list[DisambiguationRequest]) -> str:
    """Render disambiguation questions as a markdown block."""
    lines: list[str] = [
        "> **Disambiguation required** — please answer the following before the search can proceed:\n"
    ]
    for req in requests:
        lines.append(f"**{req.dimension}** (`{req.raw_term}`):\n\n{req.question}\n")
    return "\n".join(lines)


def _format_narrowing(narrowing: NarrowingRequest) -> str:
    """Render a narrowing question as a markdown block."""
    return f"> **Search too broad** — {narrowing.question}\n"


def _extract_source_name(provenance: Any) -> str:
    """Extract the display name from a provenance object."""
    name = getattr(provenance, "source_name", None)
    if name:
        return str(name)
    url = getattr(provenance, "source_url", None)
    if url:
        return str(url)
    return "Unknown source"


def _extract_source_url(provenance: Any) -> str | None:
    """Extract the URL from a provenance object, if available."""
    return getattr(provenance, "source_url", None) or None


_MAX_RENDERED_CANDIDATES = 10


def _confidence_label(overall_score: float) -> str:
    """Return a human-readable confidence label for the given overall score."""
    if overall_score >= 0.5:
        return "Strong"
    if overall_score >= 0.2:
        return "Moderate"
    return "Weak"


def _candidate_label(sc: ScoredCandidate) -> str:
    """Build a display label for a scored candidate."""
    cand = sc.candidate
    drug_name = getattr(cand, "drug_name", None)
    target = getattr(cand, "target", None)
    target_label: str | None = None
    if target is not None:
        for attr in ("label", "name", "identifier", "value"):
            val = getattr(target, attr, None)
            if val:
                target_label = str(val)
                break

    parts: list[str] = []
    if drug_name:
        parts.append(drug_name)
    if target_label and target_label != drug_name:
        parts.append(f"target: {target_label}")

    return " / ".join(parts) if parts else f"Candidate {getattr(cand, 'id', '?')}"


def _resolve_dimension_value(
    dim: str, candidate: Any, signals: RawSignals | None
) -> str:
    """Resolve the actual matched value for a dimension.

    Tries candidate attributes first, then falls back to query signals.
    Returns the dimension name as a last resort.
    """
    dim_lower = dim.lower()

    if dim_lower == "target":
        # Try candidate's target object
        target = getattr(candidate, "target", None)
        if target is not None:
            for attr in ("label", "name", "identifier", "gene_name", "value"):
                val = getattr(target, attr, None)
                if val:
                    return str(val)
        # Fall back to query signals
        if signals and signals.target:
            return signals.target[0]

    elif dim_lower == "indication":
        # Fall back to query signals
        if signals and signals.indication:
            return signals.indication[0]

    elif dim_lower == "modality":
        if signals and signals.modality:
            return signals.modality[0]

    # Try drug_name for any dimension as a generic fallback
    drug = getattr(candidate, "drug_name", None)
    if dim_lower == "drug" and drug:
        return str(drug)

    return dim


@register_agent("search")
class SearchAgent(Agent):
    """Orchestrates the Interpret → Classify → Gate → Fetch → Match pipeline.

    Yields streaming markdown output including:
    - A ranked list of drug candidates with scores
    - Matched search dimensions per candidate
    - Contributing data sources with provenance links
    - A synthesis summary

    Handles closed-option disambiguation and narrowing loops from the Gate phase
    by surfacing questions to the caller via markdown output and awaiting
    user-provided selections in subsequent ``run()`` calls (or inline during a
    session via the ``_handle_gate_loop`` internal flow).
    """

    def __init__(
        self,
        *,
        intent_extractor: IntentExtractor | None = None,
        classifier: ClassificationOrchestrator | None = None,
        gate: SearchGate | None = None,
        fetcher: FetchOrchestrator | None = None,
        scorer: MatchScorer | None = None,
    ) -> None:
        """Initialise SearchAgent with optional pre-built pipeline components.

        When a component is *None*, a default instance is created if possible.
        Components that require external configuration (e.g. resolvers for the
        ClassificationOrchestrator) are left as None when not supplied, and the
        pipeline degrades gracefully.

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
            Dispatches parallel API calls and merges Candidate objects.
        scorer:
            Scores and ranks Candidates using semantic + structured + evidence
            signals.
        """
        self._extractor = intent_extractor or IntentExtractor()
        self._classifier = classifier  # May be None — caller must supply resolvers
        self._gate = gate  # May be None — gate is skipped when absent
        self._fetcher = fetcher or FetchOrchestrator()
        self._scorer = scorer or MatchScorer()

    # ------------------------------------------------------------------
    # Agent ABC implementation
    # ------------------------------------------------------------------

    async def run(self, query: str) -> AsyncGenerator[str, None]:  # type: ignore[override]
        """Run the full search pipeline and yield streaming markdown.

        Pipeline stages
        ---------------
        1. **Interpret** — extract raw dimension-tagged signals from *query*.
        2. **Classify** — resolve signals to canonical SearchSpec identifiers.
           If disambiguation is required, yield questions and return early.
        3. **Gate** — validate spec breadth; if too broad, yield narrowing
           question and return early.  Loops up to ``_MAX_GATE_LOOPS`` times
           when gate feedback is available inline.
        4. **Fetch** — parallel dispatch to all configured data sources.
        5. **Match** — score and rank candidates.
        6. **Render** — yield ranked results as streaming markdown.

        Args:
            query: Natural-language search query from the user.

        Yields:
            Markdown-formatted output fragments.
        """
        yield f"# Search Results: {query}\n\n"

        # ----------------------------------------------------------------
        # Stage 1: Interpret
        # ----------------------------------------------------------------
        yield "_Extracting search signals…_\n\n"
        try:
            signals: RawSignals = await self._extractor.extract(query)
        except Exception as exc:  # noqa: BLE001
            yield f"> **Error** during signal extraction: {exc}\n"
            return

        yield self._render_signals(signals)

        if not semantic_scoring_available():
            yield "_Semantic similarity scoring is not available in this environment. Results are ranked by structural match and evidence depth._\n\n"

        # ----------------------------------------------------------------
        # Stage 2: Classify (optional — requires resolvers)
        # ----------------------------------------------------------------
        spec: Any = None
        if self._classifier is not None:
            yield "_Classifying signals…_\n\n"
            try:
                classification = await self._classifier.classify(signals)
            except Exception as exc:  # noqa: BLE001
                yield f"> **Error** during classification: {exc}\n"
                return

            if classification.disambiguations:
                yield _format_disambiguations(classification.disambiguations)
                yield (
                    "\n_Search paused — please resolve the above questions and retry._\n"
                )
                return

            spec = classification.spec
        else:
            # No classifier — build a minimal spec-like object from raw signals
            spec = _signals_to_minimal_spec(signals)

        # ----------------------------------------------------------------
        # Stage 3: Gate (optional — requires estimator + narrowing provider)
        # ----------------------------------------------------------------
        if self._gate is not None:
            gate_result: GateResult | None = None
            for _loop in range(_MAX_GATE_LOOPS):
                try:
                    gate_result = await self._gate.check(spec)
                except Exception as exc:  # noqa: BLE001
                    yield f"> **Error** during gate check: {exc}\n"
                    return

                if gate_result.passed:
                    break

                reason = gate_result.reason or "unknown"

                if reason == "missing_core_fields":
                    yield (
                        "> **Cannot proceed** — the query does not contain enough "
                        "information to run a search. Please add a drug name, target, "
                        "indication, or therapeutic area.\n"
                    )
                    return

                if reason == "pending_disambiguations":
                    yield (
                        "> **Cannot proceed** — there are unresolved disambiguations. "
                        "Please resolve them before retrying.\n"
                    )
                    return

                if reason == "too_broad" and gate_result.narrowing is not None:
                    yield _format_narrowing(gate_result.narrowing)
                    yield (
                        "\n_Search paused — please select one of the options above "
                        "to narrow your search and retry._\n"
                    )
                    return

                # Unexpected gate failure reason — abort
                yield f"> **Gate check failed** with reason: `{reason}`\n"
                return

            if gate_result is not None and not gate_result.passed:
                yield "> **Gate check did not pass after maximum retries.**\n"
                return

        # ----------------------------------------------------------------
        # Stage 4: Fetch
        # ----------------------------------------------------------------
        yield "_Fetching candidates from data sources…_\n\n"
        try:
            candidates: list[Any] = await self._fetcher.fetch(spec)
        except Exception as exc:  # noqa: BLE001
            yield f"> **Error** during fetch: {exc}\n"
            return

        if not candidates:
            yield "_No candidates found for the given search criteria._\n"
            return

        yield f"_Retrieved **{len(candidates)}** candidate(s). Scoring…_\n\n"

        # ----------------------------------------------------------------
        # Stage 5: Match / Score
        # ----------------------------------------------------------------
        try:
            scored_candidates: list[ScoredCandidate] = await self._scorer.score(
                candidates, spec, query_text=query
            )
        except Exception as exc:  # noqa: BLE001
            yield f"> **Error** during scoring: {exc}\n"
            return

        # ----------------------------------------------------------------
        # Stage 6: Render results
        # ----------------------------------------------------------------
        yield f"_Found **{len(scored_candidates)}** candidates matching your query. Results are ranked by how closely each candidate matches your search criteria (target, indication, modality) and the depth of supporting evidence (trials, patents, articles)._\n\n"

        async for fragment in self._render_results(scored_candidates, query, signals=signals):
            yield fragment

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _render_signals(self, signals: RawSignals) -> str:
        """Render extracted signals as a compact markdown summary."""
        parts: list[str] = ["**Extracted signals:**\n"]
        if signals.target:
            parts.append(f"- Target: {', '.join(signals.target)}")
        if signals.modality:
            parts.append(f"- Modality: {', '.join(signals.modality)}")
        if signals.indication:
            parts.append(f"- Indication: {', '.join(signals.indication)}")
        if signals.commercial:
            parts.append(f"- Commercial: {', '.join(signals.commercial)}")
        if signals.temporal is not None:
            temporal_parts: list[str] = []
            if signals.temporal.after:
                temporal_parts.append(f"after {signals.temporal.after}")
            if signals.temporal.before:
                temporal_parts.append(f"before {signals.temporal.before}")
            if signals.temporal.not_expired:
                temporal_parts.append("not expired")
            if temporal_parts:
                parts.append(f"- Temporal: {', '.join(temporal_parts)}")
        parts.append("\n")
        return "\n".join(parts)

    async def _render_results(
        self,
        scored: list[ScoredCandidate],
        query: str,
        signals: RawSignals | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield streaming markdown for the ranked results.

        Renders at most the top ``_MAX_RENDERED_CANDIDATES`` candidates.  When
        more candidates exist, a summary line indicates how many were omitted
        and the score threshold of the last rendered candidate.
        """
        total = len(scored)
        show = scored[:_MAX_RENDERED_CANDIDATES]
        remaining = total - len(show)

        if remaining > 0:
            yield f"## Ranked Candidates (showing top {len(show)} of {total})\n\n"
        else:
            yield f"## Ranked Candidates ({total} total)\n\n"

        for sc in show:
            async for fragment in self._render_candidate(sc, signals=signals):
                yield fragment

        if remaining > 0:
            threshold = show[-1].overall_score if show else 0.0
            yield "---\n\n"
            yield (
                f"_{remaining} additional candidate{'s' if remaining != 1 else ''} "
                f"scored below {threshold:.2f}. "
                f"Results ranked by structural match and evidence depth._\n\n"
            )

        yield "---\n\n"

        # Synthesis summary
        yield "## Synthesis Summary\n\n"
        yield self._build_synthesis_summary(scored, query)

    async def _render_candidate(
        self, sc: ScoredCandidate, *, signals: RawSignals | None = None
    ) -> AsyncGenerator[str, None]:
        """Yield a compact, human-readable markdown block for a single scored candidate."""
        label = _candidate_label(sc)
        yield f"### {sc.rank}. {label}\n"

        # Confidence label
        confidence = _confidence_label(sc.overall_score)

        # Matched dimensions with actual values and checkmarks
        dims = list(getattr(sc.candidate, "matched_dimensions", []) or [])
        dim_parts: list[str] = []
        for dim in dims:
            value = _resolve_dimension_value(dim, sc.candidate, signals)
            dim_parts.append(f"{dim.capitalize()}: {value} \u2713")

        # Evidence counts (only non-zero)
        n_trials = len(list(getattr(sc.candidate, "trials", []) or []))
        n_patents = len(list(getattr(sc.candidate, "patents", []) or []))
        n_articles = len(list(getattr(sc.candidate, "articles", []) or []))
        evidence_parts: list[str] = []
        if n_trials:
            evidence_parts.append(f"{n_trials} trial{'s' if n_trials != 1 else ''}")
        if n_patents:
            evidence_parts.append(f"{n_patents} patent{'s' if n_patents != 1 else ''}")
        if n_articles:
            evidence_parts.append(f"{n_articles} article{'s' if n_articles != 1 else ''}")

        # Build the one-liner
        detail_parts: list[str] = dim_parts
        if evidence_parts:
            detail_parts = detail_parts + [", ".join(evidence_parts)]
        detail_str = " | ".join(detail_parts) if detail_parts else ""

        match_line = f"**Match: {confidence}** ({sc.overall_score:.2f})"
        if detail_str:
            match_line += f" \u2014 {detail_str}"
        yield f"{match_line}\n"

        # Per-candidate verified source links
        links: list[str] = []

        # Check trials for NCT IDs
        for trial in list(getattr(sc.candidate, "trials", []) or [])[:1]:
            nct = getattr(trial, "nct_id", None)
            url = trial_url(nct) if nct else None
            if url:
                links.append(f"[ClinicalTrials.gov]({url})")

        # Check articles for PMIDs
        for article in list(getattr(sc.candidate, "articles", []) or [])[:1]:
            pmid = getattr(article, "pmid", None)
            doi = getattr(article, "doi", None)
            url = pubmed_url(pmid, doi)
            if url:
                links.append(f"[PubMed]({url})")

        # Check target for UniProt
        target = getattr(sc.candidate, "target", None)
        if target:
            acc = (
                getattr(target, "uniprot_id", None)
                or getattr(target, "accession", None)
                or getattr(target, "identifier", None)
            )
            url = uniprot_url(acc) if acc else None
            if url:
                links.append(f"[UniProt]({url})")

        # Check patents
        for pat in list(getattr(sc.candidate, "patents", []) or [])[:1]:
            pub_num = getattr(pat, "publication_number", None)
            url = patent_url(pub_num) if pub_num else None
            if url:
                links.append(f"[Google Patents]({url})")

        # Cap at 3 links
        links = links[:3]
        if links:
            separator = " \u00b7 "
            yield f"**Sources:** {separator.join(links)}\n"

        # Synthesis summary on the candidate itself (if any)
        synthesis = getattr(sc.candidate, "synthesis_summary", None)
        if synthesis:
            yield f"\n> {synthesis}\n"

        yield "\n"

    def _build_synthesis_summary(
        self, scored: list[ScoredCandidate], query: str
    ) -> str:
        """Build an overall synthesis summary paragraph."""
        if not scored:
            return "_No results to summarise._\n"

        top = scored[0]
        top_label = _candidate_label(top)

        n_candidates = len(scored)
        high_confidence = [sc for sc in scored if sc.overall_score >= 0.6]
        moderate = [sc for sc in scored if 0.3 <= sc.overall_score < 0.6]

        lines: list[str] = []
        lines.append(
            f"A total of **{n_candidates}** candidate(s) were identified for the "
            f"query *\"{query}\"*. "
        )

        lines.append(
            f"The top-ranked result is **{top_label}** "
            f"(overall score: {top.overall_score:.3f}). "
        )

        if high_confidence:
            lines.append(
                f"**{len(high_confidence)}** candidate(s) scored ≥ 0.6 (high confidence). "
            )
        if moderate:
            lines.append(
                f"**{len(moderate)}** candidate(s) scored between 0.3 and 0.6 (moderate confidence). "
            )

        # Dominant source types
        all_sources: set[str] = set()
        for sc in scored:
            for prov in list(getattr(sc.candidate, "contributing_sources", []) or []):
                all_sources.add(_extract_source_name(prov))
        if all_sources:
            lines.append(
                f"Data was drawn from {len(all_sources)} unique source(s): "
                f"{', '.join(sorted(all_sources))}."
            )

        # Low confidence advisory
        if top.overall_score < 0.3:
            lines.append(
                "Results with low confidence may not be directly relevant to "
                "your query. Try refining your search with more specific terms."
            )

        return " ".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Minimal spec builder — used when no ClassificationOrchestrator is provided
# ---------------------------------------------------------------------------


def _signals_to_minimal_spec(signals: RawSignals) -> Any:
    """Convert RawSignals into a minimal spec-like object for the Fetch and Match phases.

    This is a best-effort conversion used when no ClassificationOrchestrator
    is configured.  The resulting object exposes the attributes that Fetch and
    Match phases access via ``getattr``.
    """
    from dataclasses import dataclass, field

    @dataclass
    class _MinimalSpec:
        target: object = None
        therapeutic_area: object = None
        drug_names: list = field(default_factory=list)
        indications: list = field(default_factory=list)
        modality: object = None
        resolved_terms: list = field(default_factory=list)
        pending_disambiguations: list = field(default_factory=list)
        patent_expiry_window: object = None
        approval_date_range: object = None
        jurisdictions: list = field(default_factory=list)
        max_results: int = 50
        domains: list = field(
            default_factory=lambda: ["trials", "patents", "articles", "candidates"]
        )
        query: str = ""

    # Represent target signals as a lightweight label object
    target_obj = None
    if signals.target:

        class _LabelObj:
            def __init__(self, label: str) -> None:
                self.label = label
                self.identifier = label
                self.value = label

        target_obj = _LabelObj(signals.target[0])

    # Represent indication signals as lightweight label objects
    indication_objs: list[Any] = []
    for ind in signals.indication:

        class _IndObj:
            def __init__(self, label: str) -> None:
                self.label = label
                self.identifier = label
                self.value = label

        indication_objs.append(_IndObj(ind))

    return _MinimalSpec(
        target=target_obj,
        indications=indication_objs,
        drug_names=[],
        query="",
    )
