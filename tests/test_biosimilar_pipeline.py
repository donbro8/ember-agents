"""Tests for the biosimilar 3-stage filter pipeline."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from ember_data.seed.schema import BiologicEntry
from ember_agents.biosimilar.pipeline import (
    stage1_hard_filter,
    stage2_enrich,
    stage3_deep_enrich,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_entry(**kwargs) -> BiologicEntry:
    """Build a minimal BiologicEntry with sensible defaults."""
    defaults = dict(
        drug_name="TestDrug",
        originator="TestCo",
        target_antigen="TNF",
        modality="mAb",
        cell_line="CHO",
        cell_line_class="mammalian",
        annual_revenue_usd_millions=500.0,
        revenue_year=2023,
        patent_expiry_us=date(2026, 6, 1),
        patent_expiry_eu=date(2027, 1, 1),
        has_approved_biosimilar=False,
        biosimilar_competitors=[],
    )
    defaults.update(kwargs)
    return BiologicEntry(**defaults)


@pytest.fixture()
def seed_entries() -> list[BiologicEntry]:
    """A mix of entries that should pass and fail stage1 filters."""
    return [
        # PASS: mammalian, high revenue, US expiry within cutoff
        _make_entry(
            drug_name="DrugA",
            cell_line_class="mammalian",
            annual_revenue_usd_millions=2000.0,
            patent_expiry_us=date(2026, 3, 1),
            patent_expiry_eu=date(2030, 1, 1),
        ),
        # PASS: mammalian, meets revenue floor, EU expiry within cutoff
        _make_entry(
            drug_name="DrugB",
            cell_line_class="mammalian",
            annual_revenue_usd_millions=1.0,
            patent_expiry_us=None,
            patent_expiry_eu=date(2027, 12, 31),
        ),
        # FAIL: wrong cell_line_class
        _make_entry(
            drug_name="DrugC",
            cell_line_class="microbial",
            annual_revenue_usd_millions=5000.0,
            patent_expiry_us=date(2025, 1, 1),
        ),
        # FAIL: revenue below threshold
        _make_entry(
            drug_name="DrugD",
            cell_line_class="mammalian",
            annual_revenue_usd_millions=0.5,
            patent_expiry_us=date(2025, 1, 1),
        ),
        # FAIL: both expiry dates beyond the cutoff
        _make_entry(
            drug_name="DrugE",
            cell_line_class="mammalian",
            annual_revenue_usd_millions=3000.0,
            patent_expiry_us=date(2030, 1, 1),
            patent_expiry_eu=date(2031, 1, 1),
        ),
        # FAIL: no expiry data at all for requested jurisdictions
        _make_entry(
            drug_name="DrugF",
            cell_line_class="mammalian",
            annual_revenue_usd_millions=800.0,
            patent_expiry_us=None,
            patent_expiry_eu=None,
        ),
    ]


# ---------------------------------------------------------------------------
# Stage 1 tests
# ---------------------------------------------------------------------------

class TestStage1HardFilter:
    """Tests for stage1_hard_filter."""

    def test_passing_entries_included(self, seed_entries):
        result = stage1_hard_filter(
            entries=seed_entries,
            patent_expiry_cutoff=date(2028, 12, 31),
            min_revenue_millions=1.0,
            cell_line_class="mammalian",
        )
        names = {c.drug_name for c in result}
        assert "DrugA" in names
        assert "DrugB" in names

    def test_failing_entries_excluded(self, seed_entries):
        result = stage1_hard_filter(
            entries=seed_entries,
            patent_expiry_cutoff=date(2028, 12, 31),
            min_revenue_millions=1.0,
            cell_line_class="mammalian",
        )
        names = {c.drug_name for c in result}
        assert "DrugC" not in names  # wrong cell_line_class
        assert "DrugD" not in names  # revenue too low
        assert "DrugE" not in names  # both expiry dates beyond cutoff
        assert "DrugF" not in names  # no expiry data

    def test_exact_revenue_threshold_passes(self):
        entry = _make_entry(annual_revenue_usd_millions=1.0, patent_expiry_us=date(2026, 1, 1))
        result = stage1_hard_filter([entry], date(2028, 12, 31), min_revenue_millions=1.0)
        assert len(result) == 1

    def test_exact_expiry_cutoff_passes(self):
        cutoff = date(2028, 12, 31)
        entry = _make_entry(patent_expiry_us=cutoff)
        result = stage1_hard_filter([entry], cutoff, min_revenue_millions=1.0)
        assert len(result) == 1

    def test_empty_input(self):
        result = stage1_hard_filter([], date(2028, 12, 31), min_revenue_millions=1.0)
        assert result == []

    def test_custom_cell_line_class(self):
        mammalian = _make_entry(drug_name="M", cell_line_class="mammalian", patent_expiry_us=date(2025, 1, 1))
        microbial = _make_entry(drug_name="B", cell_line_class="microbial", patent_expiry_us=date(2025, 1, 1))
        result = stage1_hard_filter(
            [mammalian, microbial],
            date(2028, 12, 31),
            min_revenue_millions=1.0,
            cell_line_class="microbial",
        )
        assert len(result) == 1
        assert result[0].drug_name == "B"

    def test_cell_line_class_none_skips_filter(self):
        """When cell_line_class is None, entries of any class pass."""
        mammalian = _make_entry(drug_name="M", cell_line_class="mammalian", patent_expiry_us=date(2025, 1, 1))
        microbial = _make_entry(drug_name="B", cell_line_class="microbial", patent_expiry_us=date(2025, 1, 1))
        insect = _make_entry(drug_name="I", cell_line_class="insect", patent_expiry_us=date(2025, 1, 1))
        result = stage1_hard_filter(
            [mammalian, microbial, insect],
            date(2028, 12, 31),
            min_revenue_millions=1.0,
            cell_line_class=None,
        )
        names = {e.drug_name for e in result}
        assert names == {"M", "B", "I"}

    def test_jurisdiction_filter_us_only(self):
        # Entry has only EU expiry within cutoff
        entry = _make_entry(
            patent_expiry_us=None,
            patent_expiry_eu=date(2026, 1, 1),
        )
        # Restricting to "US" only should exclude it (no US expiry data)
        result = stage1_hard_filter(
            [entry],
            date(2028, 12, 31),
            min_revenue_millions=1.0,
            jurisdictions=["US"],
        )
        assert result == []

    def test_jurisdiction_filter_eu_only(self):
        entry = _make_entry(
            patent_expiry_us=None,
            patent_expiry_eu=date(2026, 1, 1),
        )
        result = stage1_hard_filter(
            [entry],
            date(2028, 12, 31),
            min_revenue_millions=1.0,
            jurisdictions=["EU"],
        )
        assert len(result) == 1

    def test_multi_region_jp_cn_in_patent_expiries(self):
        """Entries with JP/CN in patent_expiries are correctly filtered."""
        entry = _make_entry(
            drug_name="AsiaOnly",
            patent_expiry_us=None,
            patent_expiry_eu=None,
            patent_expiries={"JP": date(2026, 3, 1), "CN": date(2027, 6, 1)},
        )
        # Default jurisdictions (US, EU) should exclude this entry
        result = stage1_hard_filter(
            [entry], date(2028, 12, 31), min_revenue_millions=1.0
        )
        assert result == []

        # But requesting JP should include it
        result = stage1_hard_filter(
            [entry], date(2028, 12, 31), min_revenue_millions=1.0,
            jurisdictions=["JP"],
        )
        assert len(result) == 1
        assert result[0].drug_name == "AsiaOnly"


# ---------------------------------------------------------------------------
# Stage 2 tests
# ---------------------------------------------------------------------------

class TestStage2Enrich:
    """Tests for stage2_enrich."""

    def _make_entries_for_enrichment(self) -> list[BiologicEntry]:
        return [
            _make_entry(
                drug_name="Low",
                annual_revenue_usd_millions=100.0,
                patent_expiry_us=date(2026, 1, 1),
                patent_expiry_eu=date(2027, 1, 1),
                biosimilar_competitors=["BioA"],
                has_approved_biosimilar=True,
            ),
            _make_entry(
                drug_name="High",
                annual_revenue_usd_millions=5000.0,
                patent_expiry_us=date(2025, 6, 1),
                patent_expiry_eu=None,
                biosimilar_competitors=["BioB", "BioC"],
                has_approved_biosimilar=True,
            ),
            _make_entry(
                drug_name="Mid",
                annual_revenue_usd_millions=1500.0,
                patent_expiry_us=None,
                patent_expiry_eu=date(2028, 3, 1),
                biosimilar_competitors=[],
                has_approved_biosimilar=False,
            ),
        ]

    def test_sorted_by_revenue_descending(self):
        entries = self._make_entries_for_enrichment()
        candidates = stage2_enrich(entries)
        revenues = [c.annual_revenue_usd_millions for c in candidates]
        assert revenues == sorted(revenues, reverse=True)

    def test_rank_assigned_1_based(self):
        entries = self._make_entries_for_enrichment()
        candidates = stage2_enrich(entries)
        assert candidates[0].rank == 1
        assert candidates[1].rank == 2
        assert candidates[2].rank == 3

    def test_earliest_expiry_is_minimum(self):
        entries = self._make_entries_for_enrichment()
        candidates = stage2_enrich(entries)
        # "High" has only US expiry 2025-06-01
        high = next(c for c in candidates if c.drug_name == "High")
        assert high.earliest_expiry == date(2025, 6, 1)
        # "Low" has US 2026-01-01 and EU 2027-01-01; min is 2026-01-01
        low = next(c for c in candidates if c.drug_name == "Low")
        assert low.earliest_expiry == date(2026, 1, 1)

    def test_competitive_landscape_populated(self):
        entries = self._make_entries_for_enrichment()
        candidates = stage2_enrich(entries)
        high = next(c for c in candidates if c.drug_name == "High")
        assert high.competitive_landscape.count == 2
        assert "BioB" in high.competitive_landscape.approved_biosimilars

    def test_cap_at_1000(self):
        # Build 1100 entries that all pass, ensure output is capped at 1000
        many = [
            _make_entry(
                drug_name=f"Drug{i}",
                annual_revenue_usd_millions=float(i),
                patent_expiry_us=date(2026, 1, 1),
            )
            for i in range(1100)
        ]
        candidates = stage2_enrich(many)
        assert len(candidates) == 1000

    def test_empty_input(self):
        assert stage2_enrich([]) == []

    def test_returns_biosimilar_candidates(self):
        from ember_data.models.biosimilar import BiosimilarCandidate
        entries = self._make_entries_for_enrichment()
        candidates = stage2_enrich(entries)
        assert all(isinstance(c, BiosimilarCandidate) for c in candidates)

    def test_earliest_expiry_jurisdiction_set(self):
        entries = self._make_entries_for_enrichment()
        candidates = stage2_enrich(entries)
        # "High" has only US expiry
        high = next(c for c in candidates if c.drug_name == "High")
        assert high.earliest_expiry_jurisdiction == "US"
        # "Low" has US 2026-01-01 and EU 2027-01-01; earliest is US
        low = next(c for c in candidates if c.drug_name == "Low")
        assert low.earliest_expiry_jurisdiction == "US"
        assert low.earliest_expiry == date(2026, 1, 1)

    def test_multi_region_earliest_expiry_from_jp(self):
        """JP patent earlier than US/EU should be selected as earliest."""
        entry = _make_entry(
            drug_name="MultiRegion",
            annual_revenue_usd_millions=2000.0,
            patent_expiry_us=date(2028, 1, 1),
            patent_expiry_eu=date(2029, 1, 1),
            patent_expiries={
                "US": date(2028, 1, 1),
                "EU": date(2029, 1, 1),
                "JP": date(2025, 6, 1),
                "CN": date(2027, 3, 1),
            },
        )
        candidates = stage2_enrich([entry])
        assert len(candidates) == 1
        c = candidates[0]
        assert c.earliest_expiry == date(2025, 6, 1)
        assert c.earliest_expiry_jurisdiction == "JP"
        assert c.patent_expiries == {
            "US": date(2028, 1, 1),
            "EU": date(2029, 1, 1),
            "JP": date(2025, 6, 1),
            "CN": date(2027, 3, 1),
        }

    def test_patent_expiries_dict_populated(self):
        """Enriched candidates carry the full patent_expiries dict."""
        entry = _make_entry(
            drug_name="DictCheck",
            annual_revenue_usd_millions=500.0,
            patent_expiry_us=date(2026, 1, 1),
            patent_expiry_eu=date(2027, 1, 1),
        )
        candidates = stage2_enrich([entry])
        assert len(candidates) == 1
        assert "US" in candidates[0].patent_expiries
        assert "EU" in candidates[0].patent_expiries


# ---------------------------------------------------------------------------
# Stage 3 tests
# ---------------------------------------------------------------------------

class TestStage3DeepEnrich:
    """Tests for stage3_deep_enrich."""

    def _make_candidates(self, n: int = 3):
        entries = [
            _make_entry(
                drug_name=f"Drug{i}",
                annual_revenue_usd_millions=float(1000 - i * 100),
                patent_expiry_us=date(2026, 1, 1),
                originator=f"Company{i}",
            )
            for i in range(n)
        ]
        return stage2_enrich(entries)

    def _make_bq_mock(self) -> MagicMock:
        bq = MagicMock()
        bq.search_patents.return_value = [
            {
                "publication_number": "US12345",
                "title": "Antibody Composition",
                "abstract": "A novel antibody...",
                "claims": ["Claim 1"],
                "assignee": "TestCo",
                "filing_date": date(2010, 1, 1),
                "grant_date": date(2012, 6, 1),
                "cited_by_count": 5,
                "relevant_targets": ["TNF"],
            }
        ]
        return bq

    async def test_patents_attached_to_candidates(self):
        candidates = self._make_candidates(3)
        bq = self._make_bq_mock()
        result = await stage3_deep_enrich(candidates, bq_client=bq, max_candidates=3)
        for c in result:
            assert len(c.patents) == 1
            assert c.patents[0].publication_number == "US12345"

    async def test_max_candidates_limits_bq_calls(self):
        candidates = self._make_candidates(5)
        bq = self._make_bq_mock()
        await stage3_deep_enrich(candidates, bq_client=bq, max_candidates=2)
        assert bq.search_patents.call_count == 2

    async def test_beyond_max_candidates_not_enriched(self):
        candidates = self._make_candidates(5)
        bq = self._make_bq_mock()
        result = await stage3_deep_enrich(candidates, bq_client=bq, max_candidates=2)
        # First 2 enriched, remaining 3 have empty patents
        assert len(result[0].patents) == 1
        assert result[2].patents == []

    async def test_empty_candidates(self):
        bq = self._make_bq_mock()
        result = await stage3_deep_enrich([], bq_client=bq)
        assert result == []
        bq.search_patents.assert_not_called()

    async def test_bq_returns_empty(self):
        candidates = self._make_candidates(2)
        bq = MagicMock()
        bq.search_patents.return_value = []
        result = await stage3_deep_enrich(candidates, bq_client=bq, max_candidates=2)
        for c in result:
            assert c.patents == []
