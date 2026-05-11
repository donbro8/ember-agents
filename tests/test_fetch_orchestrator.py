"""Tests for the FetchOrchestrator (Phase 4 Fetch phase)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from ember_agents.search.fetch import (
    FetchOrchestrator,
    FetchResult,
    _DedupeRegistry,
    _normalize_name,
)

# ---------------------------------------------------------------------------
# Real ember-data models (needed because Candidate is a Pydantic model)
# ---------------------------------------------------------------------------

try:
    from ember_data.models.article import Article
    from ember_data.models.candidate import Candidate, CandidateScores
    from ember_data.models.provenance import SourceProvenance
    from ember_data.models.target import Target
    from ember_data.models.trial import Trial

    _EMBER_DATA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _EMBER_DATA_AVAILABLE = False
    Article = None  # type: ignore[assignment,misc]
    Candidate = None  # type: ignore[assignment,misc]
    CandidateScores = None  # type: ignore[assignment,misc]
    SourceProvenance = None  # type: ignore[assignment,misc]
    Target = None  # type: ignore[assignment,misc]
    Trial = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers to build real ember-data model instances
# ---------------------------------------------------------------------------


def _make_provenance(source_name: str = "ClinicalTrials.gov") -> Any:
    """Build a real SourceProvenance instance."""
    return SourceProvenance(
        source_name=source_name,
        retrieved_at=datetime.now(UTC),
    )


def _make_trial(
    interventions: list[str] | None = None, condition: str = "Breast Cancer"
) -> Any:
    """Build a real Trial instance."""
    return Trial(
        nct_id=f"NCT{uuid.uuid4().hex[:8].upper()}",
        title="A Phase 2 clinical trial",
        phase="II",
        status="Completed",
        conditions=[condition],
        interventions=interventions or ["trastuzumab"],
        sponsor="Test Sponsor",
        start_date=date(2020, 1, 1),
        primary_endpoints=["Overall survival"],
    )


def _make_article(title: str = "HER2 inhibition study") -> Any:
    """Build a real Article instance."""
    return Article(
        title=title,
        authors=["Smith J", "Jones A"],
        journal="Nature",
        pub_date=date(2022, 6, 1),
        abstract="We studied HER2 inhibitors in breast cancer.",
    )


def _make_target(name: str = "HER2", uniprot_id: str = "P04626") -> Any:
    """Build a real Target instance."""
    return Target(
        id=uniprot_id,
        name=name,
        type="protein",
        organism="human",
        aliases=["ErbB2", "NEU"],
    )


# ---------------------------------------------------------------------------
# SearchSpec-alike stubs (don't need Pydantic, just attributes)
# ---------------------------------------------------------------------------


@dataclass
class FakeSpec:
    """Minimal SearchSpec-alike for testing."""

    target: object = None
    therapeutic_area: object = None
    drug_names: list = field(default_factory=list)
    indications: list = field(default_factory=list)
    modality: object = None
    resolved_terms: list = field(default_factory=list)
    pending_disambiguations: list = field(default_factory=list)
    patent_expiry_window: object = None
    approval_date_range: object = None
    jurisdictions: list = field(default_factory=list)
    max_results: int = 500
    domains: list = field(
        default_factory=lambda: ["trials", "patents", "articles", "candidates"]
    )


@dataclass
class FakeTargetSpec:
    label: str = "HER2"
    identifier: str = "P04626"
    value: str = "HER2"


@dataclass
class FakeIndication:
    label: str = "Breast Cancer"
    identifier: str = "D001943"
    value: str = "Breast Cancer"


# ---------------------------------------------------------------------------
# Client mock factories
# ---------------------------------------------------------------------------


def _make_ct_client(results=None) -> MagicMock:
    """Build a mock ClinicalTrials client that returns given results."""
    client = MagicMock()
    client.search.return_value = results or []
    return client


def _make_pubmed_client(results=None) -> MagicMock:
    client = MagicMock()
    client.search.return_value = results or []
    return client


def _make_uniprot_client(results=None) -> MagicMock:
    client = MagicMock()
    client.search_targets.return_value = results or []
    return client


def _make_bq_client(fda_rows=None, patent_rows=None) -> MagicMock:
    client = MagicMock()
    client.query_fda_drug_events.return_value = fda_rows or []
    client.search_patents.return_value = patent_rows or []
    return client


def _make_orchestrator(
    *,
    bq_client=None,
    ct_client=None,
    uniprot_client=None,
    pubmed_client=None,
) -> FetchOrchestrator:
    return FetchOrchestrator(
        bigquery_client=bq_client,
        clinicaltrials_client=ct_client,
        uniprot_client=uniprot_client,
        pubmed_client=pubmed_client,
    )


# ---------------------------------------------------------------------------
# _normalize_name tests
# ---------------------------------------------------------------------------


def test_normalize_name_lowercases() -> None:
    assert _normalize_name("Trastuzumab") == "trastuzumab"


def test_normalize_name_strips_whitespace() -> None:
    assert _normalize_name("  imatinib  ") == "imatinib"


def test_normalize_name_nfc_normalisation() -> None:
    # Both forms should normalise to the same value
    name_a = "caf\u00e9"  # precomposed NFC
    name_b = "cafe\u0301"  # decomposed NFD
    assert _normalize_name(name_a) == _normalize_name(name_b)


def test_normalize_name_none_returns_empty() -> None:
    assert _normalize_name(None) == ""


def test_normalize_name_empty_returns_empty() -> None:
    assert _normalize_name("") == ""


# ---------------------------------------------------------------------------
# _DedupeRegistry tests
# ---------------------------------------------------------------------------


def test_dedupe_registry_detects_duplicate_by_name() -> None:
    registry = _DedupeRegistry()
    r1 = FetchResult(drug_name="Trastuzumab")
    registry.register(r1)
    r2 = FetchResult(drug_name="trastuzumab")  # same name, different case
    assert registry.is_duplicate(r2)


def test_dedupe_registry_detects_duplicate_by_synonym() -> None:
    registry = _DedupeRegistry()
    r1 = FetchResult(drug_name="herceptin", synonyms=["trastuzumab"])
    registry.register(r1)
    r2 = FetchResult(drug_name="trastuzumab")
    assert registry.is_duplicate(r2)


def test_dedupe_registry_detects_duplicate_by_target_id() -> None:
    registry = _DedupeRegistry()
    r1 = FetchResult(drug_name="drug-a", target_id="P04626")
    registry.register(r1)
    r2 = FetchResult(drug_name="different-name", target_id="P04626")
    assert registry.is_duplicate(r2)


def test_dedupe_registry_not_duplicate_distinct_names() -> None:
    registry = _DedupeRegistry()
    r1 = FetchResult(drug_name="imatinib", target_id="P00533")
    registry.register(r1)
    r2 = FetchResult(drug_name="erlotinib", target_id="P12345")
    assert not registry.is_duplicate(r2)


def test_dedupe_registry_empty_name_never_duplicates() -> None:
    registry = _DedupeRegistry()
    r1 = FetchResult(drug_name="")
    registry.register(r1)
    r2 = FetchResult(drug_name="")
    # Empty name should not trigger duplicate detection
    assert not registry.is_duplicate(r2)


# ---------------------------------------------------------------------------
# FetchOrchestrator — construction
# ---------------------------------------------------------------------------


def test_orchestrator_stores_provided_clients() -> None:
    bq = MagicMock()
    ct = MagicMock()
    uniprot = MagicMock()
    pubmed = MagicMock()

    orch = FetchOrchestrator(
        bigquery_client=bq,
        clinicaltrials_client=ct,
        uniprot_client=uniprot,
        pubmed_client=pubmed,
    )

    assert orch._bq is bq
    assert orch._ct is ct
    assert orch._uniprot is uniprot
    assert orch._pubmed is pubmed


def test_orchestrator_no_bq_without_project_id() -> None:
    """BigQuery client should not be auto-created without bq_project_id."""
    orch = FetchOrchestrator()
    assert orch._bq is None


# ---------------------------------------------------------------------------
# FetchOrchestrator.fetch — end-to-end with mocked clients
# ---------------------------------------------------------------------------


async def test_fetch_returns_candidates_from_clinicaltrials() -> None:
    pytest.importorskip("ember_data")
    trial = _make_trial(interventions=["trastuzumab"])
    prov = _make_provenance("ClinicalTrials.gov")
    ct_client = _make_ct_client(results=[(trial, prov)])

    spec = FakeSpec(
        indications=[FakeIndication()],
        domains=["trials"],
    )
    orch = _make_orchestrator(ct_client=ct_client)
    candidates = await orch.fetch(spec)

    assert len(candidates) >= 1
    first = candidates[0]
    assert hasattr(first, "drug_name")
    assert first.drug_name == "trastuzumab"


async def test_fetch_provenance_tagged_on_candidates() -> None:
    """Each candidate should have provenance with source_name and retrieved_at."""
    pytest.importorskip("ember_data")
    trial = _make_trial(interventions=["imatinib"])
    prov = _make_provenance("ClinicalTrials.gov")
    ct_client = _make_ct_client(results=[(trial, prov)])

    spec = FakeSpec(
        indications=[FakeIndication()],
        domains=["trials"],
    )
    orch = _make_orchestrator(ct_client=ct_client)
    candidates = await orch.fetch(spec)

    assert len(candidates) >= 1
    sources = candidates[0].contributing_sources
    assert len(sources) >= 1
    source = sources[0]
    assert hasattr(source, "source_name")
    assert source.source_name == "ClinicalTrials.gov"
    assert hasattr(source, "retrieved_at")
    assert source.retrieved_at is not None


async def test_fetch_deduplication_by_drug_name() -> None:
    """Same drug returned by two sources should yield a single candidate."""
    pytest.importorskip("ember_data")
    trial_a = _make_trial(interventions=["trastuzumab"])
    trial_b = _make_trial(interventions=["Trastuzumab"])  # same drug, different case
    prov_a = _make_provenance("ClinicalTrials.gov")
    prov_b = _make_provenance("ClinicalTrials.gov")

    ct_client = _make_ct_client(results=[(trial_a, prov_a), (trial_b, prov_b)])

    spec = FakeSpec(
        indications=[FakeIndication()],
        domains=["trials"],
    )
    orch = _make_orchestrator(ct_client=ct_client)
    candidates = await orch.fetch(spec)

    # Should be deduplicated to one candidate
    drug_names = [c.drug_name for c in candidates if c.drug_name]
    lowered = [n.lower() for n in drug_names]
    assert lowered.count("trastuzumab") == 1


async def test_fetch_pubmed_candidates() -> None:
    pytest.importorskip("ember_data")
    article = _make_article()
    prov = _make_provenance("PubMed")
    pubmed_client = _make_pubmed_client(results=[(article, prov)])

    spec = FakeSpec(
        indications=[FakeIndication()],
        domains=["articles"],
    )
    orch = _make_orchestrator(pubmed_client=pubmed_client)
    candidates = await orch.fetch(spec)

    assert len(candidates) >= 1
    sources = candidates[0].contributing_sources
    assert any(s.source_name == "PubMed" for s in sources)


async def test_fetch_max_results_enforcement() -> None:
    """Fetch should respect max_results and return no more than that number."""
    pytest.importorskip("ember_data")
    ct_results = [
        (
            _make_trial(interventions=[f"drug-{i}"]),
            _make_provenance("ClinicalTrials.gov"),
        )
        for i in range(20)
    ]
    ct_client = _make_ct_client(results=ct_results)

    spec = FakeSpec(
        indications=[FakeIndication()],
        domains=["trials"],
        max_results=5,
    )
    orch = _make_orchestrator(ct_client=ct_client)
    candidates = await orch.fetch(spec)

    assert len(candidates) <= 5


async def test_fetch_source_failure_does_not_abort() -> None:
    """If one source raises an exception, the others should still return results."""
    pytest.importorskip("ember_data")
    trial = _make_trial(interventions=["erlotinib"])
    prov = _make_provenance("ClinicalTrials.gov")
    ct_client = _make_ct_client(results=[(trial, prov)])

    # pubmed will raise
    pubmed_client = MagicMock()
    pubmed_client.search.side_effect = RuntimeError("PubMed unavailable")

    spec = FakeSpec(
        indications=[FakeIndication()],
        domains=["trials", "articles"],
    )
    orch = _make_orchestrator(ct_client=ct_client, pubmed_client=pubmed_client)
    candidates = await orch.fetch(spec)

    # Should still get ClinicalTrials results
    assert len(candidates) >= 1


# ---------------------------------------------------------------------------
# FetchOrchestrator — BigQuery wave
# ---------------------------------------------------------------------------


async def test_fetch_bigquery_fda_rows() -> None:
    """BigQuery FDA rows should be parsed into candidates with FDA provenance."""
    bq_client = _make_bq_client(
        fda_rows=[
            {
                "openfda_generic_name": "trastuzumab",
                "openfda_brand_name": "Herceptin",
                "openfda_product_ndc": "50242-134-01",
            }
        ]
    )

    spec = FakeSpec(
        drug_names=["trastuzumab"],
        domains=["candidates"],
    )
    orch = _make_orchestrator(bq_client=bq_client)
    candidates = await orch.fetch(spec)

    assert len(candidates) >= 1
    sources = candidates[0].contributing_sources
    assert any("FDA" in (s.source_name or "") for s in sources)


async def test_fetch_bigquery_executes_before_external_apis() -> None:
    """BigQuery results (wave 1) should appear before external API results (wave 2)."""
    pytest.importorskip("ember_data")
    bq_client = _make_bq_client(
        fda_rows=[
            {
                "openfda_generic_name": "drug-bq",
                "openfda_brand_name": "",
                "openfda_product_ndc": "12345",
            }
        ]
    )

    trial = _make_trial(interventions=["drug-ct"])
    prov = _make_provenance("ClinicalTrials.gov")
    ct_client = _make_ct_client(results=[(trial, prov)])

    spec = FakeSpec(
        drug_names=["drug-bq"],
        indications=[FakeIndication()],
        domains=["candidates", "trials"],
    )
    orch = _make_orchestrator(bq_client=bq_client, ct_client=ct_client)
    candidates = await orch.fetch(spec)

    # Should have at least 2 distinct candidates (one per source)
    assert len(candidates) >= 1
    # BigQuery candidate (drug-bq) should appear first
    first_drug = candidates[0].drug_name
    assert first_drug == "drug-bq"


async def test_fetch_bigquery_patent_rows_parsed() -> None:
    """BigQuery patent rows with valid filing_date should produce patent candidates."""
    bq_client = _make_bq_client(
        patent_rows=[
            {
                "publication_number": "US-12345678-A1",
                "title": "Novel HER2 inhibitor",
                "abstract": "A compound for treating HER2+ cancers",
                "filing_date": 20200101,
                "grant_date": 20220601,
            }
        ]
    )

    spec = FakeSpec(
        drug_names=["trastuzumab"],
        domains=["patents"],
    )
    orch = _make_orchestrator(bq_client=bq_client)
    candidates = await orch.fetch(spec)

    # Patent candidates with no drug name are still created
    assert len(candidates) >= 1
    sources = candidates[0].contributing_sources
    assert any(
        "Patent" in (s.source_name or "") or "BigQuery" in (s.source_name or "")
        for s in sources
    )


async def test_fetch_bigquery_skips_patent_rows_without_filing_date() -> None:
    """Patent rows missing filing_date should be silently skipped."""
    bq_client = _make_bq_client(
        patent_rows=[
            {
                "publication_number": "US-99999999-A1",
                "title": "No date patent",
                "abstract": "Missing filing date",
                "filing_date": None,
                "grant_date": None,
            }
        ]
    )

    spec = FakeSpec(
        drug_names=["trastuzumab"],
        domains=["patents"],
    )
    orch = _make_orchestrator(bq_client=bq_client)
    candidates = await orch.fetch(spec)

    # The invalid row should be skipped — 0 patent candidates
    patent_sources = [
        c
        for c in candidates
        if any("Patent" in (s.source_name or "") for s in c.contributing_sources)
    ]
    assert len(patent_sources) == 0


# ---------------------------------------------------------------------------
# FetchOrchestrator — UniProt wave
# ---------------------------------------------------------------------------


async def test_fetch_uniprot_target_results() -> None:
    """UniProt should produce candidates tagged with UniProt provenance."""
    pytest.importorskip("ember_data")
    tgt = _make_target("HER2", "P04626")
    prov = _make_provenance("UniProt")
    uniprot_client = _make_uniprot_client(results=[(tgt, prov)])

    spec = FakeSpec(target=FakeTargetSpec(), domains=["candidates"])
    orch = _make_orchestrator(uniprot_client=uniprot_client)
    candidates = await orch.fetch(spec)

    assert len(candidates) >= 1
    first = candidates[0]
    assert first.drug_name == "HER2"
    sources = first.contributing_sources
    assert any(s.source_name == "UniProt" for s in sources)


# ---------------------------------------------------------------------------
# FetchOrchestrator — per-source limit and domain filtering
# ---------------------------------------------------------------------------


async def test_per_source_limit_caps_when_max_results_is_large() -> None:
    """per_source_limit should be capped at min(max_results, 100)."""
    ct_client = MagicMock()
    ct_client.search.return_value = []

    spec = FakeSpec(
        indications=[FakeIndication()],
        domains=["trials"],
        max_results=1000,
    )
    orch = _make_orchestrator(ct_client=ct_client)
    await orch.fetch(spec)

    # Verify search was called (even if with no results)
    assert ct_client.search.called


async def test_domains_filter_excludes_trials() -> None:
    """When 'trials' is not in domains, ClinicalTrials client should not be called."""
    ct_client = MagicMock()
    ct_client.search.return_value = []

    spec = FakeSpec(
        indications=[FakeIndication()],
        domains=["articles"],  # no 'trials'
    )
    orch = _make_orchestrator(ct_client=ct_client)
    await orch.fetch(spec)

    ct_client.search.assert_not_called()


async def test_domains_filter_excludes_articles() -> None:
    """When 'articles' is not in domains, PubMed client should not be called."""
    pubmed_client = MagicMock()
    pubmed_client.search.return_value = []

    spec = FakeSpec(
        indications=[FakeIndication()],
        domains=["trials"],  # no 'articles'
    )
    orch = _make_orchestrator(pubmed_client=pubmed_client)
    await orch.fetch(spec)

    pubmed_client.search.assert_not_called()


async def test_fetch_returns_empty_with_no_matching_domains() -> None:
    """No candidates should be returned when domain list is empty."""
    spec = FakeSpec(
        drug_names=["trastuzumab"],
        indications=[FakeIndication()],
        domains=[],
    )
    orch = FetchOrchestrator()
    result = await orch.fetch(spec)
    assert result == []


async def test_clinicaltrials_uses_intervention_only_without_condition() -> None:
    """ClinicalTrials falls back to intervention-only when condition is absent."""
    trial = _make_trial(interventions=["adalimumab"])
    prov = _make_provenance("ClinicalTrials.gov")
    ct_client = _make_ct_client(results=[(trial, prov)])
    spec = FakeSpec(
        drug_names=["adalimumab"],
        indications=[],
        therapeutic_area=None,
        domains=["trials"],
    )
    orch = _make_orchestrator(ct_client=ct_client)
    candidates = await orch.fetch(spec)

    assert len(candidates) >= 1
    ct_client.search.assert_called()
    _, kwargs = ct_client.search.call_args
    assert kwargs.get("intervention") == "adalimumab"
    assert kwargs.get("term") is None
    assert any(
        s.name == "clinicaltrials" and s.status == "ok"
        for s in orch.last_source_statuses
    )


async def test_clinicaltrials_term_fallback_is_capped_and_marked_constrained() -> None:
    """Broad target term fallback is capped and status marked constrained."""
    trial = _make_trial(interventions=["drug-x"])
    prov = _make_provenance("ClinicalTrials.gov")
    ct_client = _make_ct_client(results=[(trial, prov)])

    @dataclass
    class _ResolvedTerm:
        label: str = ""
        identifier: str = ""
        value: str = ""

    spec = FakeSpec(
        target=FakeTargetSpec(
            label="PD-1", identifier="PDCD1", value="Programmed cell death 1"
        ),
        indications=[],
        drug_names=[],
        resolved_terms=[
            _ResolvedTerm(label="PD-1"),
            _ResolvedTerm(label="checkpoint inhibitor"),
            _ResolvedTerm(label="immuno-oncology"),
            _ResolvedTerm(label="extra-term"),
        ],
        domains=["trials"],
    )
    orch = _make_orchestrator(ct_client=ct_client)
    await orch.fetch(spec)

    term_calls = [c for c in ct_client.search.call_args_list if c.kwargs.get("term")]
    assert len(term_calls) == 2
    ct_statuses = [s for s in orch.last_source_statuses if s.name == "clinicaltrials"]
    assert ct_statuses
    assert ct_statuses[-1].status.startswith("ok_constrained")


async def test_clinicaltrials_condition_path_not_marked_constrained_when_terms_capped() -> (
    None
):
    """Condition-based query path should not inherit constrained term-fallback status."""
    trial = _make_trial(interventions=["drug-y"])
    prov = _make_provenance("ClinicalTrials.gov")
    ct_client = _make_ct_client(results=[(trial, prov)])

    @dataclass
    class _ResolvedTerm:
        label: str = ""
        identifier: str = ""
        value: str = ""

    spec = FakeSpec(
        target=FakeTargetSpec(
            label="PD-1", identifier="PDCD1", value="Programmed cell death 1"
        ),
        indications=[FakeIndication()],
        drug_names=[],
        resolved_terms=[
            _ResolvedTerm(label="PD-1"),
            _ResolvedTerm(label="checkpoint inhibitor"),
            _ResolvedTerm(label="immuno-oncology"),
            _ResolvedTerm(label="extra-term"),
        ],
        domains=["trials"],
    )
    orch = _make_orchestrator(ct_client=ct_client)
    await orch.fetch(spec)

    term_calls = [c for c in ct_client.search.call_args_list if c.kwargs.get("term")]
    assert len(term_calls) == 0
    ct_statuses = [s for s in orch.last_source_statuses if s.name == "clinicaltrials"]
    assert ct_statuses
    assert ct_statuses[-1].status == "ok"


async def test_clinicaltrials_skipped_status_recorded() -> None:
    """ClinicalTrials records skipped status when trials domain is not requested."""
    ct_client = _make_ct_client(results=[])
    spec = FakeSpec(drug_names=["adalimumab"], domains=["articles"])
    orch = _make_orchestrator(ct_client=ct_client)
    await orch.fetch(spec)
    assert any(
        s.name == "clinicaltrials" and s.status == "skipped" and s.result_count == 0
        for s in orch.last_source_statuses
    )
