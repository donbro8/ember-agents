"""Contract test helpers — shared fixtures and utilities.

Provides the ``load_fixture`` helper that loads a recorded JSON response
from ``tests/contract/fixtures/`` by name, without any network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict | list:
    """Load a recorded JSON fixture by filename (without extension).

    Parameters
    ----------
    name:
        Fixture file stem, e.g. ``"bigquery_patents_adalimumab"``.
        The ``.json`` extension is appended automatically.

    Returns
    -------
    Parsed JSON as a ``dict`` or ``list``, depending on the fixture.

    Raises
    ------
    FileNotFoundError
        When no fixture file with the given name exists.
    """
    fixture_path = _FIXTURES_DIR / f"{name}.json"
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Fixture '{name}' not found at {fixture_path}. "
            f"Available fixtures: {[p.stem for p in _FIXTURES_DIR.glob('*.json')]}"
        )
    return json.loads(fixture_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Shared pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def bigquery_patents_adalimumab() -> list:
    """Recorded BigQuery patent rows for adalimumab."""
    return load_fixture("bigquery_patents_adalimumab")  # type: ignore[return-value]


@pytest.fixture()
def bigquery_fda() -> list:
    """Recorded BigQuery FDA rows for adalimumab and biosimilars."""
    return load_fixture("bigquery_fda")  # type: ignore[return-value]


@pytest.fixture()
def clinicaltrials() -> dict:
    """Recorded ClinicalTrials.gov response."""
    return load_fixture("clinicaltrials")  # type: ignore[return-value]


@pytest.fixture()
def pubmed() -> dict:
    """Recorded PubMed response."""
    return load_fixture("pubmed")  # type: ignore[return-value]


@pytest.fixture()
def uniprot() -> dict:
    """Recorded UniProt response."""
    return load_fixture("uniprot")  # type: ignore[return-value]


@pytest.fixture()
def biologic_seed() -> list:
    """Recorded biologic seed entries."""
    return load_fixture("biologic_seed")  # type: ignore[return-value]
