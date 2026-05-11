"""Tests for BiologicSeedSource."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ember_agents.search.seed_source import BiologicSeedSource
from ember_agents.search.fetch import FetchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TODAY = date.today()

# A future date that falls within the default cutoff window.
_NEAR_EXPIRY = date(2027, 6, 1)
# A far-future date that falls OUTSIDE the default cutoff window.
_FAR_EXPIRY = date(2035, 1, 1)


def _make_seed_data() -> list[dict]:
    """Return a small, self-contained seed dataset for tests."""
    return [
        {
            "drug_name": "AlphaMab",
            "brand_names": ["AlphaBrand", "AlphaTrade"],
            "originator": "AlphaCo",
            "target_antigen": "IL-6",
            "modality": "mAb",
            "category": "mAb",
            "cell_line": "CHO",
            "cell_line_class": "mammalian",
            "indications": ["rheumatoid arthritis"],
            "annual_revenue_usd_millions": 3000.0,
            "revenue_year": 2024,
            "patent_expiries": {
                "US": _NEAR_EXPIRY.isoformat(),
                "EU": "2030-01-01",
            },
            "key_patent_numbers": ["US9999901"],
            "biosimilar_competitors": ["BioSim1"],
            "has_approved_biosimilar": True,
            "notes": "",
        },
        {
            "drug_name": "BetaMab",
            "brand_names": ["BetaBrand"],
            "originator": "BetaCo",
            "target_antigen": "TNF-alpha",
            "modality": "mAb",
            "category": "mAb",
            "cell_line": "CHO",
            "cell_line_class": "mammalian",
            "indications": ["psoriasis"],
            "annual_revenue_usd_millions": 800.0,
            "revenue_year": 2024,
            "patent_expiries": {
                "US": _NEAR_EXPIRY.isoformat(),
            },
            "key_patent_numbers": ["US9999902"],
            "biosimilar_competitors": [],
            "has_approved_biosimilar": False,
            "notes": "",
        },
        {
            "drug_name": "EpsilonProtein",
            "brand_names": ["EpsilonBrand"],
            "originator": "EpsilonCo",
            "target_antigen": "IL-17",
            "modality": "protein",
            "category": "fusion",
            "cell_line": "E. coli",
            "cell_line_class": "non-mammalian",
            "indications": ["psoriasis"],
            "annual_revenue_usd_millions": 1800.0,
            "revenue_year": 2024,
            "patent_expiries": {
                "US": _NEAR_EXPIRY.isoformat(),
            },
            "key_patent_numbers": ["US9999905"],
            "biosimilar_competitors": [],
            "has_approved_biosimilar": False,
            "notes": "",
        },
        {
            # Should be excluded from biosimilar_screen due to low revenue
            "drug_name": "GammaMab",
            "brand_names": [],
            "originator": "GammaCo",
            "target_antigen": "VEGF",
            "modality": "mAb",
            "category": "mAb",
            "cell_line": "CHO",
            "cell_line_class": "mammalian",
            "indications": ["cancer"],
            "annual_revenue_usd_millions": 0.1,
            "revenue_year": 2024,
            "patent_expiries": {
                "US": _NEAR_EXPIRY.isoformat(),
            },
            "key_patent_numbers": [],
            "biosimilar_competitors": [],
            "has_approved_biosimilar": False,
            "notes": "",
        },
        {
            # Should be excluded from biosimilar_screen: patent expiry too far out
            "drug_name": "DeltaMab",
            "brand_names": ["DeltaBrand"],
            "originator": "DeltaCo",
            "target_antigen": "PD-1",
            "modality": "mAb",
            "category": "mAb",
            "cell_line": "CHO",
            "cell_line_class": "mammalian",
            "indications": ["melanoma"],
            "annual_revenue_usd_millions": 5000.0,
            "revenue_year": 2024,
            "patent_expiries": {
                "US": _FAR_EXPIRY.isoformat(),
                "EU": _FAR_EXPIRY.isoformat(),
            },
            "key_patent_numbers": ["US9999904"],
            "biosimilar_competitors": [],
            "has_approved_biosimilar": False,
            "notes": "",
        },
    ]


@pytest.fixture()
def seed_file(tmp_path: Path) -> Path:
    """Write fixture seed data to a temp JSON file and return its path."""
    data = _make_seed_data()
    p = tmp_path / "biologic_reference.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture()
def source(seed_file: Path) -> BiologicSeedSource:
    """Return a BiologicSeedSource loaded from the fixture file."""
    return BiologicSeedSource(seed_file)


@pytest.fixture()
def reference_seed_source() -> BiologicSeedSource:
    """Source backed by ember-data biologic reference for realistic filter assertions."""
    seed_path = (
        Path(__file__).resolve().parents[2]
        / "ember-data"
        / "src"
        / "ember_data"
        / "seed"
        / "biologic_reference.json"
    )
    if not seed_path.exists():
        pytest.skip(f"Missing seed fixture at {seed_path}")
    return BiologicSeedSource(seed_path)


def _make_spec(**kwargs: Any) -> SimpleNamespace:
    """Build a minimal duck-typed SearchSpec."""
    defaults = {
        "drug_names": [],
        "brand_names": [],
        "target": None,
        "min_revenue_millions": None,
        "patent_expiry_window": None,
        "jurisdictions": [],
        "modality_categories": [],
        "categories": [],
        "cell_line_classes": [],
        "cell_line_class": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
# Constructor / loading tests
# ---------------------------------------------------------------------------


class TestBiologicSeedSourceInit:
    def test_loads_entries(self, source: BiologicSeedSource) -> None:
        assert len(source._entries) == 5

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            BiologicSeedSource(tmp_path / "nonexistent.json")

    def test_dict_without_entries_key_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text('{"key": "value"}', encoding="utf-8")
        with pytest.raises(ValueError, match="Expected 'entries' key with a list"):
            BiologicSeedSource(bad)


# ---------------------------------------------------------------------------
# Name-matching tests
# ---------------------------------------------------------------------------


class TestFetchByName:
    @pytest.mark.asyncio
    async def test_match_by_drug_name(self, source: BiologicSeedSource) -> None:
        spec = _make_spec(drug_names=["AlphaMab"])
        results = await source.fetch(spec, query_type="name_lookup")
        assert len(results) == 1
        assert results[0].drug_name == "AlphaMab"

    @pytest.mark.asyncio
    async def test_match_case_insensitive(self, source: BiologicSeedSource) -> None:
        spec = _make_spec(drug_names=["ALPHAMAB"])
        results = await source.fetch(spec, query_type="name_lookup")
        assert len(results) == 1
        assert results[0].drug_name == "AlphaMab"

    @pytest.mark.asyncio
    async def test_match_by_brand_name_in_drug_names(
        self, source: BiologicSeedSource
    ) -> None:
        # brand name passed through drug_names field
        spec = _make_spec(drug_names=["AlphaBrand"])
        results = await source.fetch(spec, query_type="name_lookup")
        assert len(results) == 1
        assert results[0].drug_name == "AlphaMab"

    @pytest.mark.asyncio
    async def test_match_by_brand_names_field(self, source: BiologicSeedSource) -> None:
        spec = _make_spec(brand_names=["BetaBrand"])
        results = await source.fetch(spec, query_type="name_lookup")
        assert len(results) == 1
        assert results[0].drug_name == "BetaMab"

    @pytest.mark.asyncio
    async def test_match_by_target_string(self, source: BiologicSeedSource) -> None:
        spec = _make_spec(target="TNF-alpha")
        results = await source.fetch(spec, query_type="name_lookup")
        assert len(results) == 1
        assert results[0].drug_name == "BetaMab"

    @pytest.mark.asyncio
    async def test_match_by_target_object_label(
        self, source: BiologicSeedSource
    ) -> None:
        target_obj = SimpleNamespace(label="IL-6", synonyms=[])
        spec = _make_spec(target=target_obj)
        results = await source.fetch(spec, query_type="name_lookup")
        assert len(results) == 1
        assert results[0].drug_name == "AlphaMab"

    @pytest.mark.asyncio
    async def test_match_by_target_synonyms(self, source: BiologicSeedSource) -> None:
        target_obj = SimpleNamespace(label="interleukin-6", synonyms=["IL-6"])
        spec = _make_spec(target=target_obj)
        results = await source.fetch(spec, query_type="name_lookup")
        assert len(results) == 1
        assert results[0].drug_name == "AlphaMab"

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self, source: BiologicSeedSource) -> None:
        spec = _make_spec(drug_names=["CompletelyUnknownDrug"])
        results = await source.fetch(spec, query_type="name_lookup")
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_spec_returns_empty(self, source: BiologicSeedSource) -> None:
        spec = _make_spec()
        results = await source.fetch(spec, query_type="name_lookup")
        assert results == []

    @pytest.mark.asyncio
    async def test_multiple_matches(self, source: BiologicSeedSource) -> None:
        # Both AlphaMab and BetaMab have "mAb" as a kind of name;
        # use drug_names that match two entries.
        spec = _make_spec(drug_names=["AlphaMab", "BetaMab"])
        results = await source.fetch(spec, query_type="name_lookup")
        names = {r.drug_name for r in results}
        assert names == {"AlphaMab", "BetaMab"}


# ---------------------------------------------------------------------------
# FetchResult structure tests
# ---------------------------------------------------------------------------


class TestFetchResultStructure:
    @pytest.mark.asyncio
    async def test_returns_fetch_result_instances(
        self, source: BiologicSeedSource
    ) -> None:
        spec = _make_spec(drug_names=["AlphaMab"])
        results = await source.fetch(spec, query_type="name_lookup")
        assert all(isinstance(r, FetchResult) for r in results)

    @pytest.mark.asyncio
    async def test_synonyms_contain_brand_names(
        self, source: BiologicSeedSource
    ) -> None:
        spec = _make_spec(drug_names=["AlphaMab"])
        results = await source.fetch(spec, query_type="name_lookup")
        assert "AlphaBrand" in results[0].synonyms
        assert "AlphaTrade" in results[0].synonyms

    @pytest.mark.asyncio
    async def test_matched_dimensions_tagged(self, source: BiologicSeedSource) -> None:
        spec = _make_spec(drug_names=["AlphaMab"])
        results = await source.fetch(spec, query_type="name_lookup")
        assert "biologic_seed" in results[0].matched_dimensions

    @pytest.mark.asyncio
    async def test_patents_populated_as_list(self, source: BiologicSeedSource) -> None:
        spec = _make_spec(drug_names=["AlphaMab"])
        results = await source.fetch(spec, query_type="name_lookup")
        # AlphaMab has US and EU patent_expiries so there must be at least one entry
        assert isinstance(results[0].patents, list)
        assert len(results[0].patents) >= 1

    @pytest.mark.asyncio
    async def test_patent_jurisdiction_fields(self, source: BiologicSeedSource) -> None:
        """PatentJurisdiction objects must carry country_code, status, and url."""
        spec = _make_spec(drug_names=["AlphaMab"])
        results = await source.fetch(spec, query_type="name_lookup")
        for pj in results[0].patents:
            assert hasattr(pj, "country_code")
            assert hasattr(pj, "status")
            assert hasattr(pj, "url")
            assert pj.url.startswith("https://patents.google.com")


# ---------------------------------------------------------------------------
# biosimilar_screen filter tests
# ---------------------------------------------------------------------------


class TestBiosimilarScreenFilter:
    @pytest.mark.asyncio
    async def test_excludes_low_revenue(self, source: BiologicSeedSource) -> None:
        spec = _make_spec(min_revenue_millions=1.0)
        results = await source.fetch(spec, query_type="biosimilar_screen")
        names = {r.drug_name for r in results}
        assert "GammaMab" not in names  # annual_revenue = 0.1M

    @pytest.mark.asyncio
    async def test_excludes_far_expiry(self, source: BiologicSeedSource) -> None:
        spec = _make_spec()
        results = await source.fetch(spec, query_type="biosimilar_screen")
        names = {r.drug_name for r in results}
        assert "DeltaMab" not in names  # both US/EU expiry at 2035

    @pytest.mark.asyncio
    async def test_includes_entries_within_cutoff(
        self, source: BiologicSeedSource
    ) -> None:
        spec = _make_spec(min_revenue_millions=1.0)
        results = await source.fetch(spec, query_type="biosimilar_screen")
        names = {r.drug_name for r in results}
        assert "AlphaMab" in names
        assert "BetaMab" in names

    @pytest.mark.asyncio
    async def test_no_name_filter_applied(self, source: BiologicSeedSource) -> None:
        """biosimilar_screen should NOT restrict by drug name even if spec is empty."""
        spec = _make_spec(drug_names=[], min_revenue_millions=1.0)
        results = await source.fetch(spec, query_type="biosimilar_screen")
        # At least 2 entries should survive (AlphaMab and BetaMab pass hard filters)
        assert len(results) >= 2

    @pytest.mark.asyncio
    async def test_custom_revenue_threshold(self, source: BiologicSeedSource) -> None:
        spec = _make_spec(min_revenue_millions=1000.0)
        results = await source.fetch(spec, query_type="biosimilar_screen")
        names = {r.drug_name for r in results}
        # Only AlphaMab (3000M) passes the 1000M floor; BetaMab (800M) does not
        assert "AlphaMab" in names
        assert "BetaMab" not in names

    @pytest.mark.asyncio
    async def test_custom_expiry_cutoff(self, source: BiologicSeedSource) -> None:
        """A tight cutoff that excludes even _NEAR_EXPIRY should return empty."""
        # Set cutoff to before _NEAR_EXPIRY
        cutoff = date(_NEAR_EXPIRY.year - 1, 1, 1)
        expiry_window = SimpleNamespace(end=cutoff, start=None)
        spec = _make_spec(patent_expiry_window=expiry_window)
        results = await source.fetch(spec, query_type="biosimilar_screen")
        assert results == []

    @pytest.mark.asyncio
    async def test_jurisdictions_filter(self, source: BiologicSeedSource) -> None:
        """Only EU jurisdiction — AlphaMab has EU expiry within cutoff; BetaMab does not."""
        # AlphaMab: EU expiry = 2030-01-01 (> default cutoff of 2028-12-31)
        # BetaMab: no EU expiry
        # With default cutoff, neither EU expiry passes; let's use a wide cutoff.
        cutoff = date(2035, 12, 31)
        expiry_window = SimpleNamespace(end=cutoff, start=None)
        # Jurisdiction objects as plain strings
        spec = _make_spec(
            patent_expiry_window=expiry_window,
            jurisdictions=["EU"],
            min_revenue_millions=1.0,
        )
        results = await source.fetch(spec, query_type="biosimilar_screen")
        names = {r.drug_name for r in results}
        # AlphaMab has EU expiry 2030 <= 2035 cutoff → included
        assert "AlphaMab" in names
        # BetaMab has no EU entry → excluded
        assert "BetaMab" not in names

    @pytest.mark.asyncio
    async def test_patents_in_biosimilar_results(
        self, source: BiologicSeedSource
    ) -> None:
        spec = _make_spec(min_revenue_millions=1.0)
        results = await source.fetch(spec, query_type="biosimilar_screen")
        for r in results:
            assert isinstance(r.patents, list)
            assert len(r.patents) >= 1

    @pytest.mark.asyncio
    async def test_filters_modality_category_and_cell_line(
        self, source: BiologicSeedSource
    ) -> None:
        expiry_window = SimpleNamespace(start=date(2025, 1, 1), end=date(2028, 12, 31))
        spec = _make_spec(
            min_revenue_millions=1.0,
            patent_expiry_window=expiry_window,
            modality_categories=["mab"],
            categories=["mab"],
            cell_line_class="mammalian",
        )
        results = await source.fetch(spec, query_type="biosimilar_screen")
        names = {r.drug_name for r in results}
        assert "AlphaMab" in names
        assert "EpsilonProtein" not in names

    @pytest.mark.asyncio
    async def test_modality_alias_monoclonal_antibody_matches_mab(
        self, source: BiologicSeedSource
    ) -> None:
        expiry_window = SimpleNamespace(start=date(2025, 1, 1), end=date(2028, 12, 31))
        spec_mab = _make_spec(
            min_revenue_millions=1.0,
            patent_expiry_window=expiry_window,
            modality="mAb",
            categories=["mAb"],
            cell_line_class="mammalian",
        )
        spec_alias = _make_spec(
            min_revenue_millions=1.0,
            patent_expiry_window=expiry_window,
            modality="monoclonal_antibody",
            categories=["mAb"],
            cell_line_class="mammalian",
        )

        results_mab = await source.fetch(spec_mab, query_type="biosimilar_screen")
        results_alias = await source.fetch(spec_alias, query_type="biosimilar_screen")

        names_mab = {r.drug_name for r in results_mab}
        names_alias = {r.drug_name for r in results_alias}
        assert names_alias == names_mab


class TestOpportunityScanAttributeFilters:
    @pytest.mark.asyncio
    async def test_attribute_only_opportunity_scan_returns_seed_candidates(
        self, reference_seed_source: BiologicSeedSource
    ) -> None:
        expiry_window = SimpleNamespace(start=date(2025, 1, 1), end=date(2028, 12, 31))
        spec = _make_spec(
            min_revenue_millions=1000.0,
            patent_expiry_window=expiry_window,
            modality="monoclonal_antibody",
            categories=["mAb"],
            cell_line_class="mammalian",
        )

        results = await reference_seed_source.fetch(spec, query_type="opportunity_scan")
        assert len(results) >= 5

    @pytest.mark.asyncio
    async def test_non_attribute_opportunity_scan_without_names_stays_empty(
        self, source: BiologicSeedSource
    ) -> None:
        spec = _make_spec()
        results = await source.fetch(spec, query_type="opportunity_scan")
        assert results == []
