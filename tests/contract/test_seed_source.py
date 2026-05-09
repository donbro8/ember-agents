"""Contract tests for BiologicSeedSource — pipeline stage behaviour.

Migrated and adapted from test_biosimilar_pipeline.py:  instead of testing the
old stage1_hard_filter / stage2_enrich / stage3_deep_enrich functions directly,
these tests verify that BiologicSeedSource applies the same hard-filter logic in
biosimilar_screen mode and that name-matched fetch works for other query types.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ember_agents.search.fetch import FetchResult
from ember_agents.search.seed_source import BiologicSeedSource, _apply_biosimilar_filter


# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------

_CUTOFF = date(2028, 12, 31)
_MIN_REVENUE = 1.0


def _make_seed_entry(
    *,
    drug_name: str = "TestDrug",
    annual_revenue_usd_millions: float = 500.0,
    patent_expiry_us: str | None = "2026-06-01",
    patent_expiry_eu: str | None = "2027-01-01",
    cell_line_class: str = "mammalian",
    brand_names: list[str] | None = None,
    target_antigen: str = "TNF",
) -> dict:
    """Return a minimal biologic seed entry dict."""
    expiries: dict[str, str] = {}
    if patent_expiry_us:
        expiries["US"] = patent_expiry_us
    if patent_expiry_eu:
        expiries["EU"] = patent_expiry_eu

    return {
        "drug_name": drug_name,
        "brand_names": brand_names or [],
        "originator": "TestCo",
        "target_antigen": target_antigen,
        "modality": "mAb",
        "category": "mAb",
        "cell_line": "CHO",
        "cell_line_class": cell_line_class,
        "indications": ["Test Indication"],
        "annual_revenue_usd_millions": annual_revenue_usd_millions,
        "revenue_year": 2023,
        "patent_expiries": expiries,
        "key_patent_numbers": [],
        "biosimilar_competitors": [],
        "has_approved_biosimilar": False,
        "notes": "",
    }


def _make_multi_entry_seed() -> list[dict]:
    """Return a seed dataset with entries that should pass/fail hard filters."""
    return [
        # PASS: high revenue, US expiry within cutoff
        _make_seed_entry(
            drug_name="DrugA",
            annual_revenue_usd_millions=2000.0,
            patent_expiry_us="2026-03-01",
            patent_expiry_eu="2030-01-01",
        ),
        # PASS: meets revenue floor, EU expiry within cutoff
        _make_seed_entry(
            drug_name="DrugB",
            annual_revenue_usd_millions=1.0,
            patent_expiry_us=None,
            patent_expiry_eu="2027-12-31",
        ),
        # FAIL: revenue below threshold
        _make_seed_entry(
            drug_name="DrugD",
            annual_revenue_usd_millions=0.5,
            patent_expiry_us="2025-01-01",
        ),
        # FAIL: both expiry dates beyond the cutoff
        _make_seed_entry(
            drug_name="DrugE",
            annual_revenue_usd_millions=3000.0,
            patent_expiry_us="2030-01-01",
            patent_expiry_eu="2031-01-01",
        ),
        # FAIL: no expiry data at all
        _make_seed_entry(
            drug_name="DrugF",
            annual_revenue_usd_millions=800.0,
            patent_expiry_us=None,
            patent_expiry_eu=None,
        ),
    ]


@pytest.fixture()
def multi_seed_file(tmp_path: Path) -> Path:
    data = _make_multi_entry_seed()
    p = tmp_path / "biologic_reference.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture()
def multi_source(multi_seed_file: Path) -> BiologicSeedSource:
    return BiologicSeedSource(multi_seed_file)


def _make_spec(**kwargs: Any) -> SimpleNamespace:
    """Build a minimal duck-typed SearchSpec."""
    defaults: dict[str, Any] = {
        "drug_names": [],
        "brand_names": [],
        "target": None,
        "min_revenue_millions": None,
        "patent_expiry_window": None,
        "jurisdictions": [],
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Stage 1 equivalent: hard filter logic in _apply_biosimilar_filter
# ---------------------------------------------------------------------------


class TestBiosimilarFilterPassingEntries:
    """Tests that qualifying entries pass the hard filter — mirrors Stage 1 tests."""

    def _make_namespace_entry(self, **kwargs: Any) -> SimpleNamespace:
        """Build a SimpleNamespace entry compatible with _apply_biosimilar_filter."""
        data = _make_seed_entry(**kwargs)
        ns = SimpleNamespace(**data)
        # Build patent_expiries dict of date objects
        expiries: dict[str, date] = {}
        for k, v in data.get("patent_expiries", {}).items():
            expiries[k.upper()] = date.fromisoformat(v)
        ns.patent_expiries = expiries
        return ns

    def test_passing_entries_included(self) -> None:
        entries = [
            self._make_namespace_entry(
                drug_name="DrugA",
                annual_revenue_usd_millions=2000.0,
                patent_expiry_us="2026-03-01",
                patent_expiry_eu="2030-01-01",
            ),
            self._make_namespace_entry(
                drug_name="DrugB",
                annual_revenue_usd_millions=1.0,
                patent_expiry_us=None,
                patent_expiry_eu="2027-12-31",
            ),
        ]
        result = _apply_biosimilar_filter(
            entries,
            min_revenue_millions=1.0,
            patent_expiry_cutoff=_CUTOFF,
        )
        names = {e.drug_name for e in result}
        assert "DrugA" in names
        assert "DrugB" in names

    def test_revenue_below_threshold_fails(self) -> None:
        entries = [
            self._make_namespace_entry(
                drug_name="LowRevenue",
                annual_revenue_usd_millions=0.5,
                patent_expiry_us="2025-01-01",
            )
        ]
        result = _apply_biosimilar_filter(
            entries, min_revenue_millions=1.0, patent_expiry_cutoff=_CUTOFF
        )
        assert result == []

    def test_expiry_beyond_cutoff_fails(self) -> None:
        entries = [
            self._make_namespace_entry(
                drug_name="TooFar",
                annual_revenue_usd_millions=3000.0,
                patent_expiry_us="2030-01-01",
                patent_expiry_eu="2031-01-01",
            )
        ]
        result = _apply_biosimilar_filter(
            entries, min_revenue_millions=1.0, patent_expiry_cutoff=_CUTOFF
        )
        assert result == []

    def test_no_expiry_data_fails(self) -> None:
        entries = [
            self._make_namespace_entry(
                drug_name="NoExpiry",
                annual_revenue_usd_millions=800.0,
                patent_expiry_us=None,
                patent_expiry_eu=None,
            )
        ]
        result = _apply_biosimilar_filter(
            entries, min_revenue_millions=1.0, patent_expiry_cutoff=_CUTOFF
        )
        assert result == []

    def test_exact_revenue_threshold_passes(self) -> None:
        entries = [
            self._make_namespace_entry(
                drug_name="Exact",
                annual_revenue_usd_millions=1.0,
                patent_expiry_us="2026-01-01",
            )
        ]
        result = _apply_biosimilar_filter(
            entries, min_revenue_millions=1.0, patent_expiry_cutoff=_CUTOFF
        )
        assert len(result) == 1

    def test_exact_expiry_cutoff_passes(self) -> None:
        entries = [
            self._make_namespace_entry(
                drug_name="OnCutoff",
                annual_revenue_usd_millions=500.0,
                patent_expiry_us="2028-12-31",
            )
        ]
        result = _apply_biosimilar_filter(
            entries, min_revenue_millions=1.0, patent_expiry_cutoff=_CUTOFF
        )
        assert len(result) == 1

    def test_empty_input_returns_empty(self) -> None:
        result = _apply_biosimilar_filter(
            [], min_revenue_millions=1.0, patent_expiry_cutoff=_CUTOFF
        )
        assert result == []

    def test_jurisdiction_filter_us_only(self) -> None:
        entries = [
            self._make_namespace_entry(
                drug_name="EUOnly",
                annual_revenue_usd_millions=500.0,
                patent_expiry_us=None,
                patent_expiry_eu="2026-01-01",
            )
        ]
        # Only US jurisdiction requested — entry has no US expiry, should fail
        result = _apply_biosimilar_filter(
            entries,
            min_revenue_millions=1.0,
            patent_expiry_cutoff=_CUTOFF,
            jurisdictions=["US"],
        )
        assert result == []

    def test_jurisdiction_filter_eu_only(self) -> None:
        entries = [
            self._make_namespace_entry(
                drug_name="EUOnly",
                annual_revenue_usd_millions=500.0,
                patent_expiry_us=None,
                patent_expiry_eu="2026-01-01",
            )
        ]
        result = _apply_biosimilar_filter(
            entries,
            min_revenue_millions=1.0,
            patent_expiry_cutoff=_CUTOFF,
            jurisdictions=["EU"],
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Stage 2 equivalent: biosimilar_screen mode returns enriched FetchResults
# ---------------------------------------------------------------------------


class TestBiosimilarScreenMode:
    """Verify BiologicSeedSource.fetch with query_type='biosimilar_screen'."""

    async def test_screen_mode_returns_passing_entries(
        self, multi_source: BiologicSeedSource
    ) -> None:
        spec = _make_spec()
        results = await multi_source.fetch(spec, query_type="biosimilar_screen")
        names = {r.drug_name for r in results}
        assert "DrugA" in names
        assert "DrugB" in names

    async def test_screen_mode_excludes_failing_entries(
        self, multi_source: BiologicSeedSource
    ) -> None:
        spec = _make_spec()
        results = await multi_source.fetch(spec, query_type="biosimilar_screen")
        names = {r.drug_name for r in results}
        assert "DrugD" not in names  # revenue too low
        assert "DrugE" not in names  # expiry beyond cutoff
        assert "DrugF" not in names  # no expiry data

    async def test_screen_mode_returns_fetch_results(
        self, multi_source: BiologicSeedSource
    ) -> None:
        spec = _make_spec()
        results = await multi_source.fetch(spec, query_type="biosimilar_screen")
        assert all(isinstance(r, FetchResult) for r in results)

    async def test_screen_mode_results_have_drug_name(
        self, multi_source: BiologicSeedSource
    ) -> None:
        spec = _make_spec()
        results = await multi_source.fetch(spec, query_type="biosimilar_screen")
        for r in results:
            assert r.drug_name is not None
            assert r.drug_name != ""

    async def test_screen_mode_results_tagged_biologic_seed(
        self, multi_source: BiologicSeedSource
    ) -> None:
        spec = _make_spec()
        results = await multi_source.fetch(spec, query_type="biosimilar_screen")
        for r in results:
            assert "biologic_seed" in r.matched_dimensions

    async def test_screen_mode_empty_seed_returns_empty(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("[]", encoding="utf-8")
        source = BiologicSeedSource(empty_file)
        spec = _make_spec()
        results = await source.fetch(spec, query_type="biosimilar_screen")
        assert results == []

    async def test_screen_mode_custom_min_revenue(self, tmp_path: Path) -> None:
        """Custom min_revenue_millions in spec filters appropriately."""
        data = [
            _make_seed_entry(
                drug_name="HighRevDrug",
                annual_revenue_usd_millions=5000.0,
                patent_expiry_us="2026-01-01",
            ),
            _make_seed_entry(
                drug_name="LowRevDrug",
                annual_revenue_usd_millions=50.0,
                patent_expiry_us="2026-01-01",
            ),
        ]
        seed_file = tmp_path / "custom.json"
        seed_file.write_text(json.dumps(data), encoding="utf-8")
        source = BiologicSeedSource(seed_file)
        spec = _make_spec(min_revenue_millions=1000.0)
        results = await source.fetch(spec, query_type="biosimilar_screen")
        names = {r.drug_name for r in results}
        assert "HighRevDrug" in names
        assert "LowRevDrug" not in names


# ---------------------------------------------------------------------------
# Stage 3 equivalent: name-based fetch returns enriched results
# ---------------------------------------------------------------------------


class TestNameLookupMode:
    """Verify BiologicSeedSource name-matching fetch for non-biosimilar queries."""

    async def test_name_lookup_returns_matching_entry(
        self, multi_source: BiologicSeedSource
    ) -> None:
        spec = _make_spec(drug_names=["DrugA"])
        results = await multi_source.fetch(spec, query_type="drug_lookup")
        assert len(results) == 1
        assert results[0].drug_name == "DrugA"

    async def test_name_lookup_no_match_returns_empty(
        self, multi_source: BiologicSeedSource
    ) -> None:
        spec = _make_spec(drug_names=["UnknownDrug"])
        results = await multi_source.fetch(spec, query_type="drug_lookup")
        assert results == []

    async def test_name_lookup_empty_spec_returns_empty(
        self, multi_source: BiologicSeedSource
    ) -> None:
        spec = _make_spec()
        results = await multi_source.fetch(spec, query_type="drug_lookup")
        assert results == []

    async def test_name_lookup_case_insensitive(
        self, multi_source: BiologicSeedSource
    ) -> None:
        spec = _make_spec(drug_names=["druga"])
        results = await multi_source.fetch(spec, query_type="drug_lookup")
        assert len(results) == 1
        assert results[0].drug_name == "DrugA"

    async def test_general_query_type_uses_name_matching(
        self, multi_source: BiologicSeedSource
    ) -> None:
        """Non-biosimilar_screen query_type falls back to name matching."""
        spec = _make_spec(drug_names=["DrugB"])
        results = await multi_source.fetch(spec, query_type="general")
        assert len(results) == 1
        assert results[0].drug_name == "DrugB"
