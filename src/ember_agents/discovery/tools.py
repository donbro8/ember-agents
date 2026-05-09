"""Data-access helpers for the Discovery Agent."""

from __future__ import annotations

import re

from ember_data import BigQueryClient
from ember_shared import settings

_STRIP_PATTERNS = [
    r"(?i)^(?:find|show\s*me|show|get|list|search|look\s*up|what\s+are|what)\s+",
    r"(?i)^(?:the\s+)?(?:patents?|fda\s+(?:adverse\s+)?events?|adverse\s+events?|safety\s+data)\s+(?:and\s+(?:fda\s+)?(?:adverse\s+)?events?\s+)?(?:for|of|about|on|related\s+to|exist\s+for)\s+",
    r"(?i)^(?:about|for|on|related\s+to)\s+",
    r"(?i)\s*\?+$",
]


def _extract_drug_name(query: str) -> str:
    """Extract the drug/search term from a natural language query."""
    term = query.strip()
    for pattern in _STRIP_PATTERNS:
        term = re.sub(pattern, "", term).strip()
    return term if term else query.strip()


def search_patents(query: str, limit: int = 20) -> list[dict]:
    client = BigQueryClient(project_id=settings.GCP_PROJECT_ID)
    term = _extract_drug_name(query)
    return client.search_patents(term, limit=limit)


def search_fda_events(drug_name: str, limit: int = 20) -> list[dict]:
    client = BigQueryClient(project_id=settings.GCP_PROJECT_ID)
    term = _extract_drug_name(drug_name)
    return client.query_fda_drug_events(term, limit=limit)
