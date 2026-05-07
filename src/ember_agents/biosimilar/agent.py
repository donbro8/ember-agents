"""BiosimilarAgent implementation."""

from __future__ import annotations

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
from ember_agents.biosimilar.tools import load_mab_seed
from ember_agents.factory import register_agent

_DEFAULT_PATENT_CUTOFF = date(2028, 12, 31)
_DEFAULT_MIN_REVENUE = 1.0


@register_agent("biosimilar")
class BiosimilarAgent(Agent):
    """Agent that screens mAb drugs for biosimilar development opportunity."""

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
        all_entries = load_mab_seed()
        filtered = stage1_hard_filter(
            entries=all_entries,
            patent_expiry_cutoff=_DEFAULT_PATENT_CUTOFF,
            min_revenue_millions=_DEFAULT_MIN_REVENUE,
        )

        yield "## Filter Summary\n\n"
        yield f"- **Total seed entries:** {len(all_entries)}\n"
        yield f"- **After Stage 1 (hard filter):** {len(filtered)}\n"
        yield (
            f"  - Cell line class: mammalian\n"
            f"  - Min annual revenue: ${_DEFAULT_MIN_REVENUE:.0f}M\n"
            f"  - Patent expiry cutoff: {_DEFAULT_PATENT_CUTOFF.isoformat()}\n"
        )
        yield "\n"

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
        yield "## Ranked Candidates\n\n"
        if candidates:
            yield (
                "| Rank | Drug | Originator | Revenue ($M) | Earliest Expiry "
                "| Biosimilars |\n"
            )
            yield "|------|------|------------|--------------|-----------------|-------------|\n"
            for c in candidates:
                biosim_count = c.competitive_landscape.count
                yield (
                    f"| {c.rank} | {c.drug_name} | {c.originator} "
                    f"| {c.annual_revenue_usd_millions:.1f} "
                    f"| {c.earliest_expiry.isoformat()} "
                    f"| {biosim_count} |\n"
                )
        else:
            yield "_No candidates passed the filter criteria._\n"
        yield "\n"

        # --- Patent details ---
        yield "## Patent Details\n\n"
        candidates_with_patents = [c for c in candidates if c.patents]
        if candidates_with_patents:
            for c in candidates_with_patents:
                yield f"### {c.drug_name}\n\n"
                for pat in c.patents:
                    yield f"- **{pat.publication_number}** — {pat.title}\n"
                    yield f"  - Assignee: {pat.assignee}\n"
                    yield f"  - Filed: {pat.filing_date.isoformat()}\n"
                    if pat.grant_date:
                        yield f"  - Granted: {pat.grant_date.isoformat()}\n"
                yield "\n"
        else:
            yield "_No patent data retrieved._\n"
