"""Result synthesis: LLM-generated interpretation of top pipeline candidates.

Takes top N candidates from a pipeline run and produces a structured
interpretation via Gemini Flash, including per-candidate insights and
jurisdiction-level patent cliff analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from ember_shared import settings as _settings
except ImportError:  # pragma: no cover
    _settings = None

_MAX_CANDIDATES = 10
_MAX_CHANGE_CONTEXT_RESULTS = 5

_CHANGE_SUMMARY_PROMPT = """\
You are a pharmaceutical-intelligence analyst summarising changes between \
two pipeline runs for a patent-landscape watch.

Given the watch name, the original query, a list of detected changes, and \
the current top results for context, produce a 2-4 sentence natural-language \
summary.

Focus on:
- What's new (new candidates entering the landscape)
- What moved (score changes, rank shifts)
- Patent timing implications (new filings, approaching expiries)
- Competitive landscape shifts

Be concise and actionable.  Do NOT return JSON — return plain English only.
"""

_SYSTEM_PROMPT = """\
You are a pharmaceutical-intelligence analyst.

Given a search query, its classification, and a list of top-scoring candidates,
produce a JSON object with exactly three keys:

1. "overview" — 2-3 sentences explaining why the top results are interesting,
   highlighting patent cliff timing, competitive density, and revenue exposure.
2. "per_candidate" — a JSON object mapping each candidate's canonical_id to a
   single sentence interpreting that candidate's strategic significance.
3. "jurisdiction_insights" — a string summarising patent cliff timing across
   jurisdictions (e.g. US vs EU expiry gaps), or null if insufficient data.

Emphasise:
- Patent cliff timing per jurisdiction
- Competitive density (biosimilar competitor count vs revenue)
- Revenue-to-competition ratio
- Trial phase alignment with expiry windows
- Whether expiry dates are approximate or authoritative

Return ONLY the JSON object.  No markdown fences, no explanation outside JSON.
"""


@dataclass
class SynthesisOutput:
    """Structured output from the result synthesizer."""

    overview: str = ""
    per_candidate: dict[str, str] = field(default_factory=dict)
    jurisdiction_insights: str | None = None


@dataclass
class OpportunityHighlight:
    """A top opportunity identified across watches."""

    display_label: str
    reason: str
    watch_name: str


@dataclass
class WatchDigestSection:
    """Summary section for a single watch within a digest."""

    watch_name: str
    summary: str
    change_count: int
    highlight: str | None = None


@dataclass
class WatchDigestInput:
    """Input data for a single watch to the digest generator."""

    watch_name: str
    query: str
    changes: list
    change_summary: str | None
    latest_results: list


@dataclass
class DigestOutput:
    """Structured output from the digest generator."""

    period_start: date
    period_end: date
    summary: str
    per_watch: list[WatchDigestSection] = field(default_factory=list)
    top_opportunities: list[OpportunityHighlight] = field(default_factory=list)
    stable_watches: list[str] = field(default_factory=list)


def _build_candidate_block(candidate: Any) -> dict[str, Any]:
    """Extract relevant fields from a candidate for the prompt context."""
    fields = [
        "canonical_id",
        "drug_name",
        "target",
        "indication",
        "overall_score",
        "earliest_patent_expiry",
        "earliest_expiry_jurisdiction",
        "annual_revenue_usd_millions",
        "biosimilar_competitor_count",
        "trial_count",
        "latest_trial_phase",
        "sources_contributed",
    ]
    block: dict[str, Any] = {}
    for f in fields:
        val = getattr(candidate, f, None)
        if val is None and isinstance(candidate, dict):
            val = candidate.get(f)
        block[f] = val
    return block


class ResultSynthesizer:
    """Produces LLM-generated interpretation of top pipeline candidates."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self._model_name = model_name
        genai_kwargs: dict[str, Any] = {"model": model_name}
        if _settings is not None and _settings.GOOGLE_API_KEY:
            genai_kwargs["google_api_key"] = _settings.GOOGLE_API_KEY
        self._llm: Any = ChatGoogleGenerativeAI(**genai_kwargs)

    async def synthesize(
        self,
        query: str,
        query_type: str,
        results: list,
        trace: Any,
    ) -> SynthesisOutput:
        """Synthesize an interpretation of top candidates.

        Args:
            query: The original search query.
            query_type: Classification of the query (e.g. opportunity_scan).
            results: List of candidate results from the pipeline.
            trace: Execution trace for observability (currently unused).

        Returns:
            SynthesisOutput with overview, per-candidate insights, and
            jurisdiction analysis.  Returns a safe default on any failure.
        """
        try:
            top = results[:_MAX_CANDIDATES]
            if not top:
                return SynthesisOutput(
                    overview="No candidates to synthesize.",
                    per_candidate={},
                    jurisdiction_insights=None,
                )

            candidate_blocks = [_build_candidate_block(c) for c in top]
            user_content = (
                f"Query: {query}\n"
                f"Query type: {query_type}\n\n"
                f"Top candidates:\n{json.dumps(candidate_blocks, indent=2, default=str)}"
            )

            messages = [
                ("system", _SYSTEM_PROMPT),
                ("human", user_content),
            ]
            response = await self._llm.ainvoke(messages)

            raw_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            # Strip markdown fences if present
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[: -len("```")]
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            return SynthesisOutput(
                overview=parsed.get("overview", ""),
                per_candidate=parsed.get("per_candidate", {}),
                jurisdiction_insights=parsed.get("jurisdiction_insights"),
            )
        except Exception:  # noqa: BLE001
            return SynthesisOutput(
                overview="Synthesis unavailable.",
                per_candidate={},
                jurisdiction_insights=None,
            )


class ChangeSummarizer:
    """Produces LLM-generated summaries of changes between pipeline runs."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self._model_name = model_name
        genai_kwargs: dict[str, Any] = {"model": model_name}
        if _settings is not None and _settings.GOOGLE_API_KEY:
            genai_kwargs["google_api_key"] = _settings.GOOGLE_API_KEY
        self._llm: Any = ChatGoogleGenerativeAI(**genai_kwargs)

    async def summarize_changes(
        self,
        watch_name: str,
        query: str,
        changes: list,
        current_results: list,
    ) -> str:
        """Summarize detected changes in natural language.

        Args:
            watch_name: Name of the watch that detected changes.
            query: The original search query for context.
            changes: List of ChangeEntry objects from the change detector.
            current_results: Current top candidate results for context.

        Returns:
            A natural-language summary string.  Returns a plaintext fallback
            on any LLM failure, and ``"No changes detected."`` when the
            changes list is empty.
        """
        if not changes:
            return "No changes detected."

        try:
            change_lines = []
            for c in changes:
                line = f"- {c.change_type}: {c.display_label} ({c.canonical_id})"
                if c.detail:
                    line += f" — {c.detail}"
                change_lines.append(line)

            top_results = current_results[:_MAX_CHANGE_CONTEXT_RESULTS]
            result_blocks = [_build_candidate_block(r) for r in top_results]

            user_content = (
                f"Watch: {watch_name}\n"
                f"Query: {query}\n\n"
                f"Changes detected:\n" + "\n".join(change_lines) + "\n\n"
                f"Current top results:\n{json.dumps(result_blocks, indent=2, default=str)}"
            )

            messages = [
                ("system", _CHANGE_SUMMARY_PROMPT),
                ("human", user_content),
            ]
            response = await self._llm.ainvoke(messages)
            raw_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            return raw_text.strip()
        except Exception:  # noqa: BLE001
            # Graceful degradation: plaintext fallback
            parts = [f"{c.change_type}: {c.display_label}" for c in changes]
            return "Changes detected: " + ", ".join(parts)


_DIGEST_PROMPT = """\
You are a pharmaceutical-intelligence analyst producing a periodic digest \
that spans multiple patent-landscape watches.

Given per-watch summaries and top results, produce a JSON object with exactly \
two keys:

1. "summary" — 3-5 sentences providing a cross-watch overview.  Emphasise:
   - Cross-watch patterns (same drug or target appearing in multiple watches)
   - Jurisdiction convergence (expiry windows aligning across regions)
   - Competitive landscape shifts (new entrants, biosimilar density changes)
   - Actionable opportunities arising from the combined picture

2. "top_opportunities" — a JSON array of up to 3 objects, each with:
   - "display_label": candidate or drug name
   - "reason": one sentence explaining why this is a top opportunity
   - "watch_name": which watch surfaced this opportunity

Return ONLY the JSON object.  No markdown fences, no explanation outside JSON.
"""


class DigestGenerator:
    """Aggregates changes across watches and produces a structured digest."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self._model_name = model_name
        genai_kwargs: dict[str, Any] = {"model": model_name}
        if _settings is not None and _settings.GOOGLE_API_KEY:
            genai_kwargs["google_api_key"] = _settings.GOOGLE_API_KEY
        self._llm: Any = ChatGoogleGenerativeAI(**genai_kwargs)

    async def generate_digest(
        self,
        watches: list[WatchDigestInput],
        period_days: int = 7,
    ) -> DigestOutput:
        """Generate a cross-watch digest for the given period.

        Args:
            watches: List of watch inputs with changes and results.
            period_days: Number of days the digest covers (default 7).

        Returns:
            DigestOutput with cross-watch summary, per-watch sections,
            top opportunities, and stable watch names.
        """
        period_end = date.today()
        period_start = period_end - timedelta(days=period_days)

        active: list[WatchDigestInput] = []
        stable: list[str] = []
        for w in watches:
            if w.changes:
                active.append(w)
            else:
                stable.append(w.watch_name)

        # Build per-watch sections for active watches
        per_watch: list[WatchDigestSection] = []
        for w in active:
            summary = w.change_summary or f"{len(w.changes)} changes detected"
            per_watch.append(
                WatchDigestSection(
                    watch_name=w.watch_name,
                    summary=summary,
                    change_count=len(w.changes),
                )
            )

        if not active:
            return DigestOutput(
                period_start=period_start,
                period_end=period_end,
                summary="No changes across any watch.",
                per_watch=[],
                top_opportunities=[],
                stable_watches=stable,
            )

        try:
            # Build context for the LLM
            watch_sections = []
            for w in active:
                section = f"## {w.watch_name}\nQuery: {w.query}\n"
                section += f"Changes: {len(w.changes)}\n"
                if w.change_summary:
                    section += f"Summary: {w.change_summary}\n"
                top_results = w.latest_results[:_MAX_CHANGE_CONTEXT_RESULTS]
                if top_results:
                    result_blocks = [_build_candidate_block(r) for r in top_results]
                    section += (
                        f"Top results:\n"
                        f"{json.dumps(result_blocks, indent=2, default=str)}\n"
                    )
                watch_sections.append(section)

            user_content = f"Period: {period_start} to {period_end}\n\n" + "\n".join(
                watch_sections
            )

            messages = [
                ("system", _DIGEST_PROMPT),
                ("human", user_content),
            ]
            response = await self._llm.ainvoke(messages)

            raw_text = (
                response.content if hasattr(response, "content") else str(response)
            )
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
                if cleaned.endswith("```"):
                    cleaned = cleaned[: -len("```")]
                cleaned = cleaned.strip()

            parsed = json.loads(cleaned)

            llm_summary = parsed.get("summary", f"{len(active)} watches had changes.")
            top_opps = [
                OpportunityHighlight(
                    display_label=o.get("display_label", ""),
                    reason=o.get("reason", ""),
                    watch_name=o.get("watch_name", ""),
                )
                for o in parsed.get("top_opportunities", [])
            ]

            return DigestOutput(
                period_start=period_start,
                period_end=period_end,
                summary=llm_summary,
                per_watch=per_watch,
                top_opportunities=top_opps,
                stable_watches=stable,
            )
        except Exception:  # noqa: BLE001
            return DigestOutput(
                period_start=period_start,
                period_end=period_end,
                summary=f"{len(active)} watches had changes.",
                per_watch=per_watch,
                top_opportunities=[],
                stable_watches=stable,
            )
