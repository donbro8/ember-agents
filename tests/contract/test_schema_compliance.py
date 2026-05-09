"""Tier 1 contract tests — schema compliance without network access.

All tests use recorded JSON fixtures from ``tests/contract/fixtures/`` and
validate that data parsed from those fixtures conforms to the expected schema
contracts defined in ``ember_data.models.result``.

Covers:
1. BigQuery patent fixtures → PatentJurisdiction with correct fields
2. Seed fixtures → PatentJurisdiction with expiry_date_approximate=False
3. canonical_id stable across adalimumab / Humira when fda_generic_name present
4. result_type is DRUG when drug_name present, TARGET when only target
5. display_label always populated
6. Scoring weights shift correctly per query_type
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from ember_agents.search.match import _QUERY_TYPE_WEIGHTS
from ember_data.models.result import (
    CandidateResult,
    PatentJurisdiction,
    ResultType,
)

from tests.contract.conftest import load_fixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JURISDICTION_NAMES: dict[str, str] = {
    "US": "United States",
    "EP": "European Patent Office",
    "JP": "Japan",
    "CN": "China",
    "KR": "South Korea",
    "CA": "Canada",
    "AU": "Australia",
    "GB": "United Kingdom",
    "IN": "India",
    "BR": "Brazil",
}


def _parse_patent_row_from_bigquery(row: dict[str, Any]) -> PatentJurisdiction:
    """Construct a PatentJurisdiction from a BigQuery patent fixture row.

    BigQuery returns rows with fields like publication_number, country_code,
    filing_date, grant_date, expiry_date, title, status.  The expiry date is
    derived from the patent term so it is treated as *approximate*.
    """
    country_code = row["country_code"]
    country_name = _JURISDICTION_NAMES.get(country_code, country_code)
    pub_number = row["publication_number"]

    return PatentJurisdiction(
        country_code=country_code,
        country_name=country_name,
        publication_number=pub_number,
        filing_date=date.fromisoformat(row["filing_date"]) if row.get("filing_date") else None,
        grant_date=date.fromisoformat(row["grant_date"]) if row.get("grant_date") else None,
        expiry_date=date.fromisoformat(row["expiry_date"]) if row.get("expiry_date") else None,
        expiry_date_approximate=True,  # BigQuery-derived expiry is always approximate
        status=row["status"],
        title=row.get("title"),
        url=f"https://patents.google.com/patent/{pub_number}",
    )


def _parse_patent_row_from_seed(
    country_code: str,
    expiry_str: str,
    pub_number: str,
) -> PatentJurisdiction:
    """Construct a PatentJurisdiction from a biologic seed entry.

    Seed entries carry explicit expiry dates recorded from authoritative sources,
    so expiry_date_approximate is False.
    """
    country_name = _JURISDICTION_NAMES.get(country_code, country_code)
    return PatentJurisdiction(
        country_code=country_code,
        country_name=country_name,
        publication_number=pub_number,
        expiry_date=date.fromisoformat(expiry_str),
        expiry_date_approximate=False,
        status="expired" if date.fromisoformat(expiry_str) < date.today() else "active",
        url=f"https://patents.google.com/patent/{pub_number}",
    )


# ---------------------------------------------------------------------------
# 1. BigQuery patent fixtures → PatentJurisdiction schema
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestBigQueryPatentFixtures:
    """BigQuery patent rows produce well-formed PatentJurisdiction objects."""

    def test_all_rows_parse_without_error(self) -> None:
        rows = load_fixture("bigquery_patents_adalimumab")
        assert isinstance(rows, list)
        for row in rows:
            pj = _parse_patent_row_from_bigquery(row)
            assert isinstance(pj, PatentJurisdiction)

    def test_country_code_populated(self) -> None:
        rows = load_fixture("bigquery_patents_adalimumab")
        for row in rows:
            pj = _parse_patent_row_from_bigquery(row)
            assert pj.country_code, f"country_code empty for {row['publication_number']}"
            assert len(pj.country_code) >= 2

    def test_country_name_populated(self) -> None:
        rows = load_fixture("bigquery_patents_adalimumab")
        for row in rows:
            pj = _parse_patent_row_from_bigquery(row)
            assert pj.country_name, f"country_name empty for {row['publication_number']}"

    def test_expiry_date_present(self) -> None:
        rows = load_fixture("bigquery_patents_adalimumab")
        for row in rows:
            pj = _parse_patent_row_from_bigquery(row)
            assert pj.expiry_date is not None, (
                f"expiry_date missing for {row['publication_number']}"
            )
            assert isinstance(pj.expiry_date, date)

    def test_expiry_date_approximate_is_true(self) -> None:
        """BigQuery-derived expiry dates are always approximate."""
        rows = load_fixture("bigquery_patents_adalimumab")
        for row in rows:
            pj = _parse_patent_row_from_bigquery(row)
            assert pj.expiry_date_approximate is True, (
                f"expiry_date_approximate should be True for BigQuery row "
                f"{row['publication_number']}"
            )

    def test_status_is_valid_literal(self) -> None:
        rows = load_fixture("bigquery_patents_adalimumab")
        valid_statuses = {"active", "expired", "pending"}
        for row in rows:
            pj = _parse_patent_row_from_bigquery(row)
            assert pj.status in valid_statuses, (
                f"Invalid status '{pj.status}' for {row['publication_number']}"
            )

    def test_url_starts_with_google_patents(self) -> None:
        rows = load_fixture("bigquery_patents_adalimumab")
        for row in rows:
            pj = _parse_patent_row_from_bigquery(row)
            assert pj.url.startswith("https://patents.google.com/patent/"), (
                f"URL does not start with expected prefix: {pj.url}"
            )

    def test_url_contains_publication_number(self) -> None:
        rows = load_fixture("bigquery_patents_adalimumab")
        for row in rows:
            pj = _parse_patent_row_from_bigquery(row)
            assert row["publication_number"] in pj.url

    def test_us_patent_country_name(self) -> None:
        rows = load_fixture("bigquery_patents_adalimumab")
        us_rows = [r for r in rows if r["country_code"] == "US"]
        assert us_rows, "Expected at least one US patent row"
        for row in us_rows:
            pj = _parse_patent_row_from_bigquery(row)
            assert pj.country_name == "United States"

    def test_ep_patent_country_name(self) -> None:
        rows = load_fixture("bigquery_patents_adalimumab")
        ep_rows = [r for r in rows if r["country_code"] == "EP"]
        assert ep_rows, "Expected at least one EP patent row"
        for row in ep_rows:
            pj = _parse_patent_row_from_bigquery(row)
            assert pj.country_name == "European Patent Office"


# ---------------------------------------------------------------------------
# 2. Seed fixtures → PatentJurisdiction with expiry_date_approximate=False
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestSeedPatentFixtures:
    """Biologic seed entries produce PatentJurisdiction with exact expiry dates."""

    def test_seed_patents_parse_without_error(self) -> None:
        entries = load_fixture("biologic_seed")
        assert isinstance(entries, list)
        for entry in entries:
            for country_code, expiry_str in entry.get("patent_expiries", {}).items():
                pub_numbers = entry.get("key_patent_numbers", [])
                pub_number = pub_numbers[0] if pub_numbers else f"{country_code}-UNKNOWN"
                pj = _parse_patent_row_from_seed(country_code, expiry_str, pub_number)
                assert isinstance(pj, PatentJurisdiction)

    def test_expiry_date_approximate_is_false(self) -> None:
        """Seed-derived expiry dates are exact (sourced from authoritative records)."""
        entries = load_fixture("biologic_seed")
        for entry in entries:
            for country_code, expiry_str in entry.get("patent_expiries", {}).items():
                pub_numbers = entry.get("key_patent_numbers", [])
                pub_number = pub_numbers[0] if pub_numbers else f"{country_code}-UNKNOWN"
                pj = _parse_patent_row_from_seed(country_code, expiry_str, pub_number)
                assert pj.expiry_date_approximate is False, (
                    f"Seed entry '{entry['drug_name']}' {country_code} should have "
                    f"expiry_date_approximate=False"
                )

    def test_seed_country_code_populated(self) -> None:
        entries = load_fixture("biologic_seed")
        for entry in entries:
            for country_code, expiry_str in entry.get("patent_expiries", {}).items():
                pub_number = entry.get("key_patent_numbers", [country_code])[0]
                pj = _parse_patent_row_from_seed(country_code, expiry_str, pub_number)
                assert pj.country_code == country_code

    def test_seed_expiry_date_is_date_type(self) -> None:
        entries = load_fixture("biologic_seed")
        for entry in entries:
            for country_code, expiry_str in entry.get("patent_expiries", {}).items():
                pub_number = entry.get("key_patent_numbers", [country_code])[0]
                pj = _parse_patent_row_from_seed(country_code, expiry_str, pub_number)
                assert isinstance(pj.expiry_date, date)

    def test_seed_url_starts_with_google_patents(self) -> None:
        entries = load_fixture("biologic_seed")
        for entry in entries:
            for country_code, expiry_str in entry.get("patent_expiries", {}).items():
                pub_number = entry.get("key_patent_numbers", [country_code])[0]
                pj = _parse_patent_row_from_seed(country_code, expiry_str, pub_number)
                assert pj.url.startswith("https://patents.google.com/patent/")


# ---------------------------------------------------------------------------
# 3. canonical_id stability: adalimumab / Humira collapse when fda_generic_name set
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestCanonicalIdStability:
    """canonical_id is stable across equivalent drug representations."""

    def test_adalimumab_and_humira_same_canonical_id(self) -> None:
        """Both brand name and generic name resolve to the same canonical_id."""
        fda_rows = load_fixture("bigquery_fda")
        # Find the originator row with generic_name = adalimumab
        originator = next(r for r in fda_rows if r.get("generic_name") == "adalimumab")

        result_generic = CandidateResult(
            drug_name=originator["brand_name"],
            fda_generic_name=originator["generic_name"],
            target="TNF-alpha",
            sources_contributed=["bigquery_fda"],
        )
        result_brand = CandidateResult(
            drug_name="Humira",
            fda_generic_name="adalimumab",
            target="TNF-alpha",
            sources_contributed=["bigquery_fda"],
        )
        assert result_generic.canonical_id == result_brand.canonical_id

    def test_canonical_id_is_normalized_generic_name(self) -> None:
        """When fda_generic_name is set, canonical_id equals the normalized name."""
        result = CandidateResult(
            drug_name="HUMIRA",
            fda_generic_name="Adalimumab",
            sources_contributed=["bigquery_fda"],
        )
        # _normalize strips whitespace and lowercases
        assert result.canonical_id == "adalimumab"

    def test_canonical_id_stable_across_multiple_instances(self) -> None:
        """Same inputs always produce the same canonical_id."""
        kwargs = {
            "drug_name": "Humira",
            "fda_generic_name": "adalimumab",
            "sources_contributed": ["bigquery_fda"],
        }
        id1 = CandidateResult(**kwargs).canonical_id
        id2 = CandidateResult(**kwargs).canonical_id
        assert id1 == id2

    def test_canonical_id_differs_for_different_generics(self) -> None:
        """Different fda_generic_name values yield different canonical_ids."""
        result_a = CandidateResult(
            fda_generic_name="adalimumab",
            sources_contributed=["bigquery_fda"],
        )
        result_b = CandidateResult(
            fda_generic_name="trastuzumab",
            sources_contributed=["bigquery_fda"],
        )
        assert result_a.canonical_id != result_b.canonical_id

    def test_seed_adalimumab_canonical_id(self) -> None:
        """Biologic seed entry for adalimumab maps to a stable canonical_id."""
        entries = load_fixture("biologic_seed")
        adalimumab = next(e for e in entries if e["drug_name"] == "adalimumab")

        result = CandidateResult(
            drug_name=adalimumab["drug_name"],
            fda_generic_name=adalimumab["drug_name"],  # seed drug_name is the generic name
            target=adalimumab["target_antigen"],
            sources_contributed=["biologic_seed"],
        )
        assert result.canonical_id == "adalimumab"


# ---------------------------------------------------------------------------
# 4. result_type: DRUG when drug_name present, TARGET when only target
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestResultType:
    """CandidateResult.result_type is derived correctly from input fields."""

    def test_drug_name_present_gives_drug_type(self) -> None:
        fda_rows = load_fixture("bigquery_fda")
        for row in fda_rows:
            result = CandidateResult(
                drug_name=row["brand_name"],
                fda_generic_name=row["generic_name"],
                sources_contributed=["bigquery_fda"],
            )
            assert result.result_type == ResultType.DRUG, (
                f"Expected DRUG for {row['brand_name']}, got {result.result_type}"
            )

    def test_only_target_gives_target_type(self) -> None:
        uniprot_data = load_fixture("uniprot")
        protein = uniprot_data["results"][0]
        gene_name = protein["genes"][0]["geneName"]["value"]

        result = CandidateResult(
            target=gene_name,
            sources_contributed=["uniprot"],
        )
        assert result.result_type == ResultType.TARGET

    def test_drug_name_without_fda_name_still_drug(self) -> None:
        result = CandidateResult(
            drug_name="adalimumab",
            sources_contributed=["biologic_seed"],
        )
        assert result.result_type == ResultType.DRUG

    def test_fda_generic_name_alone_gives_drug(self) -> None:
        result = CandidateResult(
            fda_generic_name="adalimumab",
            sources_contributed=["bigquery_fda"],
        )
        assert result.result_type == ResultType.DRUG

    def test_no_drug_no_target_gives_target(self) -> None:
        """Edge case: no drug or target fields → defaults to TARGET."""
        result = CandidateResult(sources_contributed=["unknown"])
        assert result.result_type == ResultType.TARGET

    def test_uniprot_target_is_target_type(self) -> None:
        uniprot_data = load_fixture("uniprot")
        protein = uniprot_data["results"][0]
        result = CandidateResult(
            target=protein["uniProtkbId"],
            sources_contributed=["uniprot"],
        )
        assert result.result_type == ResultType.TARGET


# ---------------------------------------------------------------------------
# 5. display_label always populated
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestDisplayLabel:
    """display_label is always non-empty for any combination of input fields."""

    def test_drug_with_fda_name_uses_fda_name(self) -> None:
        result = CandidateResult(
            drug_name="Humira",
            fda_generic_name="adalimumab",
            sources_contributed=["bigquery_fda"],
        )
        assert result.display_label == "adalimumab"

    def test_drug_without_fda_name_uses_drug_name(self) -> None:
        result = CandidateResult(
            drug_name="Humira",
            sources_contributed=["biologic_seed"],
        )
        assert result.display_label == "Humira"

    def test_target_only_uses_target(self) -> None:
        uniprot_data = load_fixture("uniprot")
        protein = uniprot_data["results"][0]
        gene_name = protein["genes"][0]["geneName"]["value"]

        result = CandidateResult(
            target=gene_name,
            sources_contributed=["uniprot"],
        )
        assert result.display_label == gene_name
        assert result.display_label != ""

    def test_all_fda_rows_have_display_label(self) -> None:
        fda_rows = load_fixture("bigquery_fda")
        for row in fda_rows:
            result = CandidateResult(
                drug_name=row["brand_name"],
                fda_generic_name=row["generic_name"],
                sources_contributed=["bigquery_fda"],
            )
            assert result.display_label, (
                f"display_label is empty for FDA row {row['brand_name']}"
            )

    def test_all_seed_entries_have_display_label(self) -> None:
        entries = load_fixture("biologic_seed")
        for entry in entries:
            result = CandidateResult(
                drug_name=entry["drug_name"],
                fda_generic_name=entry["drug_name"],
                target=entry.get("target_antigen"),
                sources_contributed=["biologic_seed"],
            )
            assert result.display_label, (
                f"display_label is empty for seed entry {entry['drug_name']}"
            )

    def test_display_label_not_empty_string(self) -> None:
        """display_label should never be an empty string for populated entries."""
        result = CandidateResult(
            drug_name="trastuzumab",
            sources_contributed=["biologic_seed"],
        )
        assert result.display_label != ""


# ---------------------------------------------------------------------------
# 6. Scoring weights shift correctly per query_type
# ---------------------------------------------------------------------------


@pytest.mark.contract
class TestScoringWeights:
    """_QUERY_TYPE_WEIGHTS defines correct weight tuples per query type."""

    def test_biosimilar_screen_weights_defined(self) -> None:
        assert "biosimilar_screen" in _QUERY_TYPE_WEIGHTS

    def test_drug_lookup_weights_defined(self) -> None:
        assert "drug_lookup" in _QUERY_TYPE_WEIGHTS

    def test_opportunity_scan_weights_defined(self) -> None:
        assert "opportunity_scan" in _QUERY_TYPE_WEIGHTS

    def test_general_weights_defined(self) -> None:
        assert "general" in _QUERY_TYPE_WEIGHTS

    def test_all_weight_tuples_sum_to_one(self) -> None:
        """Each weight triple (structured, evidence, semantic) sums to 1.0."""
        for query_type, weights in _QUERY_TYPE_WEIGHTS.items():
            total = sum(weights)
            assert abs(total - 1.0) < 1e-9, (
                f"Weights for '{query_type}' sum to {total}, expected 1.0"
            )

    def test_all_weights_non_negative(self) -> None:
        for query_type, weights in _QUERY_TYPE_WEIGHTS.items():
            for w in weights:
                assert w >= 0, f"Negative weight {w} in '{query_type}'"

    def test_biosimilar_screen_higher_structured_than_drug_lookup(self) -> None:
        """biosimilar_screen emphasises structured signal over drug_lookup."""
        bs_structured = _QUERY_TYPE_WEIGHTS["biosimilar_screen"][0]
        dl_structured = _QUERY_TYPE_WEIGHTS["drug_lookup"][0]
        assert bs_structured > dl_structured, (
            f"Expected biosimilar_screen structured ({bs_structured}) > "
            f"drug_lookup structured ({dl_structured})"
        )

    def test_drug_lookup_higher_evidence_than_biosimilar_screen(self) -> None:
        """drug_lookup emphasises evidence signal more than biosimilar_screen."""
        bs_evidence = _QUERY_TYPE_WEIGHTS["biosimilar_screen"][1]
        dl_evidence = _QUERY_TYPE_WEIGHTS["drug_lookup"][1]
        assert dl_evidence > bs_evidence, (
            f"Expected drug_lookup evidence ({dl_evidence}) > "
            f"biosimilar_screen evidence ({bs_evidence})"
        )

    def test_opportunity_scan_weights_are_tuple_of_three(self) -> None:
        weights = _QUERY_TYPE_WEIGHTS["opportunity_scan"]
        assert len(weights) == 3

    def test_weight_tuples_contain_floats(self) -> None:
        for query_type, weights in _QUERY_TYPE_WEIGHTS.items():
            for w in weights:
                assert isinstance(w, float), (
                    f"Weight {w} in '{query_type}' is not float (got {type(w).__name__})"
                )
