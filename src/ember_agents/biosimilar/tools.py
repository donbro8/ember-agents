"""Data-access helpers for the Biosimilar Agent."""

from __future__ import annotations

from ember_data.models.patent import Patent
from ember_data.seed import load_biologic_reference
from ember_data.seed.schema import BiologicEntry


def load_biologic_seed() -> list[BiologicEntry]:
    """Load the curated biologic reference seed dataset.

    Returns:
        List of validated BiologicEntry instances.
    """
    return load_biologic_reference()


# Backward-compatible alias
load_mab_seed = load_biologic_seed


def search_drug_patents(drug_name: str, originator: str, bq_client) -> list[Patent]:
    """Query BigQuery for patents related to a drug name and originator.

    Args:
        drug_name: The INN or brand name of the reference biologic.
        originator: The originator company name.
        bq_client: An initialised BigQueryClient instance.

    Returns:
        List of Patent objects matching the query.
    """
    query = f"{drug_name} {originator}"
    rows = bq_client.search_patents(query)
    patents: list[Patent] = []
    for row in rows:
        try:
            patent = Patent(
                publication_number=row.get("publication_number", ""),
                title=row.get("title", ""),
                abstract=row.get("abstract", ""),
                claims=row.get("claims", []),
                assignee=row.get("assignee", originator),
                filing_date=row.get("filing_date"),
                grant_date=row.get("grant_date"),
                cited_by_count=row.get("cited_by_count", 0),
                relevant_targets=row.get("relevant_targets", []),
            )
            patents.append(patent)
        except Exception:
            # Skip malformed rows
            continue
    return patents
