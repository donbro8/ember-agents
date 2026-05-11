"""Tests for the URL builder utility functions."""

from __future__ import annotations


from ember_agents.search.urls import (
    drug_label_url,
    patent_url,
    pubmed_url,
    trial_url,
    uniprot_url,
)


# -- trial_url ---------------------------------------------------------------


class TestTrialUrl:
    def test_valid_nct_id(self) -> None:
        assert (
            trial_url("NCT12345678") == "https://clinicaltrials.gov/study/NCT12345678"
        )

    def test_empty_string_returns_none(self) -> None:
        assert trial_url("") is None

    def test_none_like_empty(self) -> None:
        # Passing empty string (None would be a type error, but empty is falsy)
        assert trial_url("") is None


# -- pubmed_url ---------------------------------------------------------------


class TestPubmedUrl:
    def test_pmid_string(self) -> None:
        assert pubmed_url(pmid="12345") == "https://pubmed.ncbi.nlm.nih.gov/12345"

    def test_pmid_int(self) -> None:
        assert pubmed_url(pmid=12345) == "https://pubmed.ncbi.nlm.nih.gov/12345"

    def test_doi_fallback(self) -> None:
        assert pubmed_url(doi="10.1234/example") == "https://doi.org/10.1234/example"

    def test_pmid_preferred_over_doi(self) -> None:
        result = pubmed_url(pmid="99999", doi="10.1234/example")
        assert result == "https://pubmed.ncbi.nlm.nih.gov/99999"

    def test_no_args_returns_none(self) -> None:
        assert pubmed_url() is None

    def test_none_pmid_none_doi(self) -> None:
        assert pubmed_url(pmid=None, doi=None) is None


# -- uniprot_url --------------------------------------------------------------


class TestUniprotUrl:
    def test_valid_accession(self) -> None:
        assert uniprot_url("P12345") == "https://www.uniprot.org/uniprotkb/P12345"

    def test_empty_string_returns_none(self) -> None:
        assert uniprot_url("") is None


# -- patent_url ---------------------------------------------------------------


class TestPatentUrl:
    def test_valid_publication_number(self) -> None:
        assert (
            patent_url("US1234567B2") == "https://patents.google.com/patent/US1234567B2"
        )

    def test_empty_string_returns_none(self) -> None:
        assert patent_url("") is None


# -- drug_label_url -----------------------------------------------------------


class TestDrugLabelUrl:
    def test_simple_drug_name(self) -> None:
        result = drug_label_url("aspirin")
        assert result == (
            "https://dailymed.nlm.nih.gov/dailymed/search.cfm"
            "?labeltype=all&query=aspirin"
        )

    def test_special_characters_encoded(self) -> None:
        result = drug_label_url("drug name/test & more")
        assert result is not None
        # Spaces and special chars should be percent-encoded
        assert "drug%20name" in result or "drug+name" in result
        assert "%26" in result  # & → %26

    def test_empty_string_returns_none(self) -> None:
        assert drug_label_url("") is None
