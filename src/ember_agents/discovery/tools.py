"""Data-access helpers for the Discovery Agent."""

from __future__ import annotations

from ember_data import BigQueryClient
from ember_shared import settings


def search_patents(query: str, limit: int = 20) -> list[dict]:
    client = BigQueryClient(project_id=settings.GCP_PROJECT_ID)
    return client.search_patents(query, limit=limit)


def search_fda_events(drug_name: str, limit: int = 20) -> list[dict]:
    client = BigQueryClient(project_id=settings.GCP_PROJECT_ID)
    return client.query_fda_drug_events(drug_name, limit=limit)
