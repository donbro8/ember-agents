"""Shared fixtures for Tier 2 integration tests.

All fixtures produce fully-mocked EmberAgent instances that return realistic
CandidateResult objects without any network access, making them safe for CI.

Investigation protocol:
  - If a test unexpectedly fails field completeness checks, add an INVESTIGATE
    comment identifying the field and the query_type that produced the low rate.
  - completeness_report.py can be run standalone to generate population rates
    for all reference queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from ember_agents.agent import EmberAgent
from ember_agents.search.classify import ClassificationOrchestrator, ClassificationResult
from ember_agents.search.fetch import FetchOrchestrator
from ember_agents.search.gate import GateResult, SearchGate
from ember_agents.search.interpret import IntentExtractor, RawSignals
from ember_agents.search.match import MatchScorer, ScoredCandidate
from ember_agents.search.seed_source import BiologicSeedSource

try:
    from ember_data.models.result import CandidateResult, PatentJurisdiction, ResultType
except ImportError:  # pragma: no cover
    CandidateResult = None  # type: ignore[assignment,misc]
    PatentJurisdiction = None  # type: ignore[assignment,misc]
    ResultType = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class FakeSearchSpec:
    drug_names: list = field(default_factory=list)
    target: object = None
    therapeutic_area: object = None
    indications: list = field(default_factory=list)
    modality: object = None
    resolved_terms: list = field(default_factory=list)
    pending_disambiguations: list = field(default_factory=list)
    max_results: int = 500
    domains: list = field(default_factory=lambda: ["trials", "patents", "articles"])
    min_revenue_millions: float | None = None
    patent_expiry_window: object = None
    jurisdictions: list = field(default_factory=list)


@dataclass
class FakeCandidate:
    drug_name: str | None = None
    target: object = None
    trials: list = field(default_factory=list)
    patents: list = field(default_factory=list)
    articles: list = field(default_factory=list)
    risk_flags: list = field(default_factory=list)
    confidence: float = 0.0
    matched_dimensions: list = field(default_factory=list)
    contributing_sources: list = field(default_factory=list)
    synthesis_summary: str | None = None


# ---------------------------------------------------------------------------
# Realistic mock CandidateResult builders
# ---------------------------------------------------------------------------


def _make_patent_jurisdiction(
    country_code: str = "US",
    country_name: str = "United States",
    expiry_year: int = 2028,
) -> "PatentJurisdiction":
    """Build a realistic PatentJurisdiction for testing."""
    pub_number = f"{country_code}12345678"
    return PatentJurisdiction(
        country_code=country_code,
        country_name=country_name,
        publication_number=pub_number,
        filing_date=date(2010, 3, 15),
        grant_date=date(2012, 6, 20),
        expiry_date=date(expiry_year, 3, 15),
        expiry_date_approximate=False,
        status="active",
        title=f"Composition for {country_code} jurisdiction",
        url=f"https://patents.google.com/patent/{pub_number}",
    )


def _make_drug_result(
    drug_name: str,
    score: float = 0.85,
    sources: list[str] | None = None,
    include_patents: bool = True,
    indication: list[str] | None = None,
) -> "CandidateResult":
    """Build a realistic DRUG CandidateResult."""
    pjs = []
    if include_patents:
        pjs = [
            _make_patent_jurisdiction("US", "United States", 2028),
            _make_patent_jurisdiction("EP", "European Patent Office", 2027),
        ]
    return CandidateResult(
        drug_name=drug_name,
        patents=pjs,
        overall_score=score,
        sources_contributed=sources or ["bigquery_patents", "bigquery_fda"],
        source_urls=[
            "https://patents.google.com/patent/US12345678",
            f"https://www.fda.gov/drugs/drug-approvals-and-databases/{drug_name.lower()}",
        ],
        indication=indication,
        mechanism_of_action="Monoclonal antibody",
        clinical_stage="Approved",
    )


def _make_target_result(
    target: str,
    score: float = 0.75,
    sources: list[str] | None = None,
) -> "CandidateResult":
    """Build a realistic TARGET CandidateResult."""
    return CandidateResult(
        target=target,
        overall_score=score,
        sources_contributed=sources or ["uniprot", "pubmed"],
        source_urls=[
            f"https://www.uniprot.org/uniprot/{target.upper()}",
            f"https://pubmed.ncbi.nlm.nih.gov/?term={target}",
        ],
    )


# ---------------------------------------------------------------------------
# Per-query realistic result sets
# ---------------------------------------------------------------------------

QUERY_RESULTS: dict[str, list] = {}


def _build_results() -> None:
    """Populate QUERY_RESULTS with realistic CandidateResult objects."""
    QUERY_RESULTS["adalimumab"] = [
        _make_drug_result(
            "adalimumab",
            score=0.95,
            sources=["bigquery_patents", "bigquery_fda", "clinicaltrials"],
            indication=["Rheumatoid arthritis"],
        ),
        _make_drug_result(
            "adalimumab-atto",
            score=0.82,
            sources=["bigquery_fda"],
            indication=["Rheumatoid arthritis"],
        ),
        _make_drug_result(
            "adalimumab-adbm",
            score=0.79,
            sources=["bigquery_fda"],
            indication=["Rheumatoid arthritis"],
        ),
    ]

    QUERY_RESULTS["PD-1 inhibitors with patents expiring before 2028"] = [
        _make_drug_result(
            "pembrolizumab",
            score=0.91,
            sources=["bigquery_patents", "clinicaltrials"],
            indication=["Non-small cell lung cancer"],
        ),
        _make_drug_result(
            "nivolumab",
            score=0.88,
            sources=["bigquery_patents", "clinicaltrials"],
            indication=["Melanoma"],
        ),
        _make_drug_result(
            "cemiplimab",
            score=0.72,
            sources=["bigquery_patents"],
            indication=["Cutaneous squamous cell carcinoma"],
        ),
    ]

    QUERY_RESULTS["biosimilar opportunities for biologics with revenue over 1 billion"] = [
        _make_drug_result(
            "trastuzumab",
            score=0.93,
            sources=["bigquery_fda", "bigquery_patents"],
            indication=["HER2-positive breast cancer"],
        ),
        _make_drug_result(
            "bevacizumab",
            score=0.89,
            sources=["bigquery_fda", "bigquery_patents"],
            indication=["Colorectal cancer"],
        ),
        _make_drug_result(
            "rituximab",
            score=0.87,
            sources=["bigquery_fda"],
            indication=["Non-Hodgkin lymphoma"],
        ),
        _make_drug_result(
            "infliximab",
            score=0.84,
            sources=["bigquery_fda", "bigquery_patents"],
            indication=["Crohn's disease"],
        ),
    ]

    QUERY_RESULTS["HER2 targeting antibodies for breast cancer"] = [
        _make_drug_result(
            "trastuzumab",
            score=0.96,
            sources=["bigquery_patents", "clinicaltrials", "pubmed"],
            indication=["HER2-positive breast cancer"],
        ),
        _make_drug_result(
            "pertuzumab",
            score=0.90,
            sources=["bigquery_patents", "clinicaltrials"],
            indication=["HER2-positive breast cancer"],
        ),
        _make_target_result(
            "HER2",
            score=0.78,
            sources=["uniprot", "pubmed"],
        ),
    ]

    QUERY_RESULTS["etanercept patent landscape"] = [
        _make_drug_result(
            "etanercept",
            score=0.97,
            sources=["bigquery_patents", "bigquery_fda"],
            indication=["Rheumatoid arthritis"],
        ),
        _make_drug_result(
            "etanercept-szzs",
            score=0.81,
            sources=["bigquery_fda"],
            indication=["Rheumatoid arthritis"],
        ),
    ]

    QUERY_RESULTS["VEGF pathway inhibitors"] = [
        _make_drug_result(
            "bevacizumab",
            score=0.92,
            sources=["bigquery_patents", "pubmed"],
            indication=["Colorectal cancer"],
        ),
        _make_target_result(
            "VEGF-A",
            score=0.80,
            sources=["uniprot", "pubmed"],
        ),
        _make_drug_result(
            "ranibizumab",
            score=0.76,
            sources=["bigquery_patents"],
            indication=["Wet age-related macular degeneration"],
        ),
    ]


_build_results()


# ---------------------------------------------------------------------------
# ScoredCandidate wrappers
# ---------------------------------------------------------------------------


def _results_to_scored(results: list) -> list[ScoredCandidate]:
    """Wrap CandidateResult objects as ScoredCandidate-compatible stubs."""
    scored = []
    for i, r in enumerate(results):
        score = r.overall_score if r.overall_score is not None else 0.7
        cand = FakeCandidate(
            drug_name=r.drug_name,
            patents=r.patents,
        )
        sc = ScoredCandidate(
            candidate=cand,
            semantic_score=score,
            structured_score=score,
            evidence_score=score,
            overall_score=score,
            rank=i + 1,
        )
        scored.append(sc)
    return scored


# ---------------------------------------------------------------------------
# Agent factory fixture
# ---------------------------------------------------------------------------


def make_mocked_agent(query: str, query_type: str = "drug_lookup") -> EmberAgent:
    """Build a fully-mocked EmberAgent that returns realistic results for *query*."""
    results = QUERY_RESULTS.get(query, [])
    scored = _results_to_scored(results)

    signals = RawSignals(query_type=query_type)
    spec = FakeSearchSpec()
    classification = ClassificationResult(spec=spec)
    gate = GateResult(passed=True)

    extractor = MagicMock(spec=IntentExtractor)
    extractor.extract = AsyncMock(return_value=signals)

    classifier = MagicMock(spec=ClassificationOrchestrator)
    classifier.classify = AsyncMock(return_value=classification)

    gate_mock = MagicMock(spec=SearchGate)
    gate_mock.check = AsyncMock(return_value=gate)

    fetcher = MagicMock(spec=FetchOrchestrator)
    fetcher.fetch = AsyncMock(return_value=[])

    seed_source = MagicMock(spec=BiologicSeedSource)
    seed_source.fetch = AsyncMock(return_value=[])

    scorer = MagicMock(spec=MatchScorer)
    scorer.score = AsyncMock(return_value=scored)

    return EmberAgent(
        intent_extractor=extractor,
        classifier=classifier,
        gate=gate_mock,
        fetcher=fetcher,
        scorer=scorer,
        seed_source=seed_source,
    )


@pytest.fixture()
def mocked_agent_factory():
    """Fixture that returns the make_mocked_agent factory function."""
    return make_mocked_agent


@pytest.fixture()
def all_query_results() -> dict[str, list]:
    """Fixture exposing all reference query result sets."""
    return QUERY_RESULTS
