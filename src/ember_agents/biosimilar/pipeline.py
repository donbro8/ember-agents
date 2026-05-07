"""3-stage biosimilar candidate filter pipeline."""

from __future__ import annotations

from datetime import date

from ember_data.models.biosimilar import BiosimilarCandidate, CompetitiveLandscape
from ember_data.seed.schema import MabEntry

from ember_agents.biosimilar.tools import search_drug_patents

_RANK_CAP = 1000


def stage1_hard_filter(
    entries: list[MabEntry],
    patent_expiry_cutoff: date,
    min_revenue_millions: float,
    cell_line_class: str = "mammalian",
    jurisdictions: list[str] | None = None,
) -> list[MabEntry]:
    """Filter mAb entries by hard criteria.

    Keeps entries that satisfy ALL of:
    - cell_line_class matches the requested class
    - annual_revenue_usd_millions >= min_revenue_millions
    - patent_expiry <= patent_expiry_cutoff for at least one of the requested jurisdictions

    Args:
        entries: Seed dataset entries to filter.
        patent_expiry_cutoff: Latest acceptable patent expiry date.
        min_revenue_millions: Minimum annual revenue in USD millions.
        cell_line_class: Required cell-line class (default "mammalian").
        jurisdictions: List of jurisdiction codes to check (default ["us", "eu"]).

    Returns:
        Filtered list of MabEntry instances.
    """
    if jurisdictions is None:
        jurisdictions = ["us", "eu"]

    result: list[MabEntry] = []
    for entry in entries:
        # Filter: cell line class
        if entry.cell_line_class != cell_line_class:
            continue

        # Filter: revenue threshold
        if entry.annual_revenue_usd_millions < min_revenue_millions:
            continue

        # Filter: patent expiry — at least one jurisdiction must have a non-null
        # expiry date on or before the cutoff
        expiry_dates: list[date] = []
        if "us" in jurisdictions and entry.patent_expiry_us is not None:
            expiry_dates.append(entry.patent_expiry_us)
        if "eu" in jurisdictions and entry.patent_expiry_eu is not None:
            expiry_dates.append(entry.patent_expiry_eu)

        if not expiry_dates:
            # No expiry data for any requested jurisdiction — skip
            continue

        if not any(d <= patent_expiry_cutoff for d in expiry_dates):
            continue

        result.append(entry)

    return result


def stage2_enrich(entries: list[MabEntry]) -> list[BiosimilarCandidate]:
    """Convert MabEntry list to ranked BiosimilarCandidate list.

    Converts each MabEntry to a BiosimilarCandidate, computing:
    - earliest_expiry from the minimum of non-null US/EU patent expiry dates
    - CompetitiveLandscape from the biosimilar_competitors list

    Sorts results by annual_revenue_usd_millions descending, assigns 1-based
    rank, and caps the list at 1000 entries.

    Args:
        entries: Filtered MabEntry instances from stage1.

    Returns:
        Ranked list of BiosimilarCandidate instances (max 1000).
    """
    candidates: list[BiosimilarCandidate] = []

    for entry in entries:
        # Compute earliest expiry across available jurisdictions
        expiry_dates = [
            d
            for d in (entry.patent_expiry_us, entry.patent_expiry_eu)
            if d is not None
        ]
        earliest = min(expiry_dates) if expiry_dates else None

        if earliest is None:
            # Should not happen after stage1, but guard defensively
            continue

        landscape = CompetitiveLandscape(
            approved_biosimilars=list(entry.biosimilar_competitors),
            count=len(entry.biosimilar_competitors),
        )

        candidate = BiosimilarCandidate(
            drug_name=entry.drug_name,
            brand_names=list(entry.brand_names),
            originator=entry.originator,
            target_antigen=entry.target_antigen,
            modality=entry.modality,
            cell_line=entry.cell_line,
            cell_line_class=entry.cell_line_class,
            indications=list(entry.indications),
            annual_revenue_usd_millions=entry.annual_revenue_usd_millions,
            revenue_year=entry.revenue_year,
            earliest_expiry=earliest,
            patent_expiry_us=entry.patent_expiry_us,
            patent_expiry_eu=entry.patent_expiry_eu,
            key_patent_numbers=list(entry.key_patent_numbers),
            competitive_landscape=landscape,
            has_approved_biosimilar=entry.has_approved_biosimilar,
            notes=entry.notes,
        )
        candidates.append(candidate)

    # Sort by revenue descending
    candidates.sort(key=lambda c: c.annual_revenue_usd_millions, reverse=True)

    # Assign rank and cap
    ranked: list[BiosimilarCandidate] = []
    for i, candidate in enumerate(candidates[:_RANK_CAP], start=1):
        candidate.rank = i
        ranked.append(candidate)

    return ranked


async def stage3_deep_enrich(
    candidates: list[BiosimilarCandidate],
    bq_client,
    max_candidates: int = 50,
) -> list[BiosimilarCandidate]:
    """Attach BigQuery patent data to each candidate.

    Queries BigQuery for patents related to each candidate's drug name and
    originator, then attaches the resulting Patent objects to the candidate.
    No LLM calls are made in v1.

    Args:
        candidates: Ranked candidates from stage2.
        bq_client: An initialised BigQueryClient instance.
        max_candidates: Maximum number of candidates to enrich (default 50).

    Returns:
        The same candidate list with patents populated for the top N.
    """
    for candidate in candidates[:max_candidates]:
        patents = search_drug_patents(
            drug_name=candidate.drug_name,
            originator=candidate.originator,
            bq_client=bq_client,
        )
        candidate.patents = patents

    return candidates
