"""Data-access helpers for the Discovery Agent."""

from __future__ import annotations

from ember_data import BigQueryClient

_DEFAULT_PROJECT = "ember-bio"


def search_patents(
    query: str,
    limit: int = 20,
    *,
    project_id: str = _DEFAULT_PROJECT,
) -> list[dict]:
    """Search patents via BigQuery.

    Args:
        query: Search terms for patent title/abstract.
        limit: Maximum number of results.
        project_id: GCP project to bill queries against.

    Returns:
        List of patent result rows as dictionaries.
    """
    client = BigQueryClient(project_id=project_id)
    return client.search_patents(query, limit=limit)


def search_fda_events(
    drug_name: str,
    limit: int = 20,
    *,
    project_id: str = _DEFAULT_PROJECT,
) -> list[dict]:
    """Search FDA adverse-event reports via BigQuery.

    Args:
        drug_name: Drug name to search for.
        limit: Maximum number of results.
        project_id: GCP project to bill queries against.

    Returns:
        List of FDA event rows as dictionaries.
    """
    client = BigQueryClient(project_id=project_id)
    return client.query_fda_drug_events(drug_name, limit=limit)
