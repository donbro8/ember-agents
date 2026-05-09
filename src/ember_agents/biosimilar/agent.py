"""BiosimilarAgent implementation."""

from __future__ import annotations

from collections import Counter
from collections.abc import AsyncGenerator
from datetime import date

from ember_data import BigQueryClient
from ember_shared import settings

from ember_agents.base import Agent
from ember_agents.biosimilar.pipeline import (
    stage1_hard_filter,
    stage2_enrich,
    stage3_deep_enrich,
)
from ember_agents.biosimilar.tools import load_biologic_seed
from ember_agents.factory import register_agent

_DEFAULT_PATENT_CUTOFF = date(2028, 12, 31)
_DEFAULT_MIN_REVENUE = 1.0


def _format_revenue(millions: float) -> str:
    """Format revenue in millions as a human-readable string."""
    if millions >= 1000:
        return f"${millions / 1000:.1f}B"
    return f"${millions:.0f}M"


def _format_expiry(expiry: date, jurisdiction: str = "") -> str:
    """Format a patent expiry date as a relative description with jurisdiction."""
    today = date.today()
    if expiry < today:
        if expiry.year == today.year:
            month_name = expiry.strftime("%b")
            return f"Expired ({month_name} {expiry.year})"
        return f"Expired ({expiry.year})"
    years = (expiry - today).days / 365.25
    if years < 1:
        return f"<1 year ({jurisdiction})" if jurisdiction else "<1 year"
    j_label = f" ({jurisdiction})" if jurisdiction else ""
    return f"~{years:.0f} years{j_label}"


@register_agent("biosimilar")
class BiosimilarAgent(Agent):
    """Agent that screens biologics for biosimilar development opportunity."""

    async def run(self, query: str) -> AsyncGenerator[str, None]:
        """Run the biosimilar screening pipeline and yield markdown output.

        Args:
            query: Natural-language query (used for context in the report header).

        Yields:
            Markdown-formatted sections: header, filter summary, ranked table,
            patent details.
        """
        yield f"# Biosimilar Candidate Screening: {query}\n\n"

        # --- Stage 1: Load and hard-filter seed data ---
        all_entries = load_biologic_seed()
        filtered = stage1_hard_filter(
            entries=all_entries,
            patent_expiry_cutoff=_DEFAULT_PATENT_CUTOFF,
            min_revenue_millions=_DEFAULT_MIN_REVENUE,
        )

        yield "## Filter Summary\n\n"
        yield f"- **Total seed entries:** {len(all_entries)}\n"
        yield f"- **After Stage 1 (hard filter):** {len(filtered)}\n"
        yield (
            f"  - Cell line class: all biologics (no filter)\n"
            f"  - Min annual revenue: ${_DEFAULT_MIN_REVENUE:.0f}M\n"
            f"  - Patent expiry cutoff: {_DEFAULT_PATENT_CUTOFF.isoformat()}\n"
        )
        yield "\n"

        # TASK-043: Context header with category breakdown
        cat_counts = Counter(getattr(e, "category", "unknown") for e in filtered)
        n_categories = len(cat_counts)
        yield f"_Screening **{len(all_entries)}** biologics across **{n_categories}** categories for biosimilar development opportunity based on patent expiry (before {_DEFAULT_PATENT_CUTOFF.isoformat()}) and minimum revenue (${_DEFAULT_MIN_REVENUE:.0f}M)._\n\n"

        # TASK-045: Category breakdown
        yield "**Category breakdown:** "
        parts = [f"{cat}: {count}" for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])]
        yield " | ".join(parts) + "\n\n"

        # --- Stage 2: Enrich and rank ---
        candidates = stage2_enrich(filtered)
        yield f"- **After Stage 2 (enrichment & ranking):** {len(candidates)}\n\n"

        # --- Stage 3: Deep-enrich with BigQuery patents ---
        try:
            bq_client = BigQueryClient(project_id=settings.GCP_PROJECT_ID)
            candidates = await stage3_deep_enrich(candidates, bq_client=bq_client)
            yield f"- **Patent data fetched for:** {min(len(candidates), 50)} candidates\n\n"
        except Exception:
            yield "- **Stage 3 (patent lookup):** skipped — BigQuery unavailable\n\n"

        # --- Ranked candidate table ---
        _TABLE_LIMIT = 25
        yield "## Ranked Candidates\n\n"
        if candidates:
            yield (
                "| Rank | Drug | Category | Originator | Revenue | Earliest Expiry "
                "| Biosimilars |\n"
            )
            yield "|------|------|----------|------------|---------|-----------------|-------------|\n"
            for c in candidates[:_TABLE_LIMIT]:
                biosim_count = c.competitive_landscape.count
                rev = _format_revenue(c.annual_revenue_usd_millions)
                exp = _format_expiry(c.earliest_expiry, c.earliest_expiry_jurisdiction)
                yield (
                    f"| {c.rank} | {c.drug_name} | {c.category} | {c.originator} "
                    f"| {rev} "
                    f"| {exp} "
                    f"| {biosim_count} |\n"
                )
            total = len(candidates)
            if total > _TABLE_LIMIT:
                remaining = total - _TABLE_LIMIT
                yield f"\n_Showing top {_TABLE_LIMIT} of {total} candidates. {remaining} additional candidates not shown._\n"
        else:
            yield "_No candidates passed the filter criteria._\n"
        yield "\n"

        # --- Key Takeaways ---
        if candidates:
            yield "## Key Takeaways\n\n"

            # Top 2 overall by revenue (already sorted by rank)
            top_overall = candidates[:2]
            seen_names: set[str] = {c.drug_name for c in top_overall}
            seen_categories: set[str] = {c.category for c in top_overall}

            # Top candidate per remaining category
            top_by_category: list = []
            for c in candidates:
                if c.category not in seen_categories and c.drug_name not in seen_names:
                    top_by_category.append(c)
                    seen_categories.add(c.category)
                    seen_names.add(c.drug_name)
                    if len(top_overall) + len(top_by_category) >= 7:
                        break

            def _render_takeaway(c):  # noqa: ANN001, ANN202
                rev = _format_revenue(c.annual_revenue_usd_millions)
                exp = _format_expiry(c.earliest_expiry, c.earliest_expiry_jurisdiction)
                line = f"- **{c.drug_name}** ({c.category}) — {rev} revenue, expiry: {exp}\n"
                if c.patent_expiries:
                    parts = [f"{j}: {d.year}" for j, d in sorted(c.patent_expiries.items())]
                    line += f"  Patent expiries: {' | '.join(parts)}\n"
                return line

            yield "**Top overall:**\n"
            for c in top_overall:
                yield _render_takeaway(c)

            if top_by_category:
                yield "\n**Top by category:**\n"
                for c in top_by_category:
                    yield _render_takeaway(c)

            yield "\n"

        # --- Patent details (only if any candidates have patent data) ---
        candidates_with_patents = [c for c in candidates if c.patents]
        if candidates_with_patents:
            yield "## Patent Details\n\n"
            for c in candidates_with_patents[:10]:
                yield f"### {c.drug_name}\n\n"
                for pat in c.patents:
                    yield f"- **{pat.publication_number}** — {pat.title}\n"
                    yield f"  - Assignee: {pat.assignee}\n"
                    yield f"  - Filed: {pat.filing_date.isoformat()}\n"
                    if pat.grant_date:
                        yield f"  - Granted: {pat.grant_date.isoformat()}\n"
                if c.patent_expiries:
                    parts = [f"{j}: {d.year}" for j, d in sorted(c.patent_expiries.items())]
                    yield f"  - Jurisdictions: {' | '.join(parts)}\n"
                yield "\n"
