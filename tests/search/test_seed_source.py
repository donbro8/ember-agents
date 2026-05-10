"""Tests for BiologicSeedSource versioned format loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ember_agents.search.seed_source import BiologicSeedSource


@pytest.fixture()
def versioned_seed_file(tmp_path: Path) -> Path:
    """Create a temporary versioned seed JSON file."""
    data = {
        "data_version": 1,
        "last_enrichment_run": None,
        "entry_count": 2,
        "entries": [
            {
                "drug_name": "test-mab-a",
                "originator": "TestCo",
                "target_antigen": "CD20",
                "modality": "mAb",
                "category": "mAb",
                "cell_line": "CHO",
                "cell_line_class": "mammalian",
                "annual_revenue_usd_millions": 5000.0,
                "revenue_year": 2025,
                "patent_expiries": {"US": "2027-06-01", "EU": "2027-06-01"},
                "last_updated": None,
                "source": "manual",
            },
            {
                "drug_name": "test-mab-b",
                "originator": "OtherCo",
                "target_antigen": "HER2",
                "modality": "mAb",
                "category": "mAb",
                "cell_line": "CHO",
                "cell_line_class": "mammalian",
                "annual_revenue_usd_millions": 3000.0,
                "revenue_year": 2025,
                "patent_expiries": {"US": "2028-01-01", "EU": "2028-01-01"},
                "last_updated": None,
                "source": "manual",
            },
        ],
    }
    p = tmp_path / "biologic_reference.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture()
def legacy_seed_file(tmp_path: Path) -> Path:
    """Create a temporary legacy (bare-array) seed JSON file."""
    data = [
        {
            "drug_name": "legacy-mab",
            "originator": "LegacyCo",
            "target_antigen": "TNF-alpha",
            "modality": "mAb",
            "category": "mAb",
            "cell_line": "CHO",
            "cell_line_class": "mammalian",
            "annual_revenue_usd_millions": 1000.0,
            "revenue_year": 2024,
            "patent_expiries": {"US": "2026-01-01", "EU": "2026-01-01"},
        },
    ]
    p = tmp_path / "biologic_reference.json"
    p.write_text(json.dumps(data))
    return p


def test_load_versioned_format(versioned_seed_file: Path):
    source = BiologicSeedSource(versioned_seed_file)
    assert len(source._entries) == 2
    assert source._entries[0].drug_name == "test-mab-a"


def test_load_legacy_format(legacy_seed_file: Path):
    source = BiologicSeedSource(legacy_seed_file)
    assert len(source._entries) == 1
    assert source._entries[0].drug_name == "legacy-mab"


def test_invalid_versioned_format_missing_entries(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"data_version": 1}))
    with pytest.raises(ValueError, match="Expected 'entries' key"):
        BiologicSeedSource(p)


def test_invalid_format_not_list_or_dict(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps("just a string"))
    with pytest.raises(ValueError, match="Expected a JSON array or versioned object"):
        BiologicSeedSource(p)
