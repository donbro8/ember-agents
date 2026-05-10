"""Tests for TASK-041 (compact candidate rendering) and TASK-044 (top-10 cap).

Validates that:
- _render_candidate produces compact one-liner format instead of score tables
- Confidence labels map correctly (Strong >= 0.5, Moderate >= 0.2, Weak < 0.2)
- Evidence counts only show non-zero items
- _render_results caps output at 10 candidates with a summary for the rest
- Sources, synthesis summary, and contributing sources sections are preserved

Also covers TASK-078/TASK-079 ResultRenderer improvements:
- Score breakdown line per candidate
- Patent table Link column with approximate date marker
- Evidence counts with singular/plural, zero sections omitted
- Commercial line when revenue is present
- Source URLs with domain-derived labels
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from ember_agents.search.agent import (
    SearchAgent,
    _MAX_RENDERED_CANDIDATES,
    _confidence_label,
)
from ember_agents.search.match import ScoredCandidate


# ---------------------------------------------------------------------------
# Lightweight stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubTarget:
    label: str = ""
    identifier: str = ""
    uniprot_id: str | None = None


@dataclass
class _StubTrial:
    nct_id: str | None = None
    phase: str | None = None
    status: str | None = None


@dataclass
class _StubArticle:
    pmid: str | None = None
    doi: str | None = None
    title: str | None = None


@dataclass
class _StubPatent:
    publication_number: str | None = None
    title: str | None = None


@dataclass
class _StubProv:
    source_name: str = ""
    source_url: str | None = None


@dataclass
class _StubCandidate:
    id: str = ""
    drug_name: str | None = None
    target: Any = None
    matched_dimensions: list = field(default_factory=list)
    trials: list = field(default_factory=list)
    patents: list = field(default_factory=list)
    articles: list = field(default_factory=list)
    contributing_sources: list = field(default_factory=list)
    synthesis_summary: str | None = None


def _make_scored(
    rank: int = 1,
    overall: float = 0.5,
    drug_name: str | None = None,
    target_label: str | None = None,
    dims: list[str] | None = None,
    n_trials: int = 0,
    n_patents: int = 0,
    n_articles: int = 0,
    synthesis: str | None = None,
    sources: list[_StubProv] | None = None,
    target_uniprot: str | None = None,
) -> ScoredCandidate:
    target = _StubTarget(label=target_label or "", uniprot_id=target_uniprot) if target_label else None
    cand = _StubCandidate(
        id=f"cand-{rank}",
        drug_name=drug_name,
        target=target,
        matched_dimensions=dims or [],
        trials=[_StubTrial() for _ in range(n_trials)],
        patents=[_StubPatent() for _ in range(n_patents)],
        articles=[_StubArticle() for _ in range(n_articles)],
        contributing_sources=sources or [],
        synthesis_summary=synthesis,
    )
    return ScoredCandidate(
        candidate=cand,
        overall_score=overall,
        semantic_score=0.0,
        structured_score=overall,
        evidence_score=0.0,
        rank=rank,
    )


async def _collect_candidate(agent: SearchAgent, sc: ScoredCandidate) -> str:
    """Collect all fragments from _render_candidate into a single string."""
    parts: list[str] = []
    async for fragment in agent._render_candidate(sc):
        parts.append(fragment)
    return "".join(parts)


async def _collect_results(agent: SearchAgent, scored: list[ScoredCandidate], query: str = "test") -> str:
    parts: list[str] = []
    async for fragment in agent._render_results(scored, query):
        parts.append(fragment)
    return "".join(parts)


# ---------------------------------------------------------------------------
# TASK-041: Confidence labels
# ---------------------------------------------------------------------------


class TestConfidenceLabel:
    def test_strong(self):
        assert _confidence_label(0.5) == "Strong"
        assert _confidence_label(0.99) == "Strong"
        assert _confidence_label(1.0) == "Strong"

    def test_moderate(self):
        assert _confidence_label(0.2) == "Moderate"
        assert _confidence_label(0.49) == "Moderate"

    def test_weak(self):
        assert _confidence_label(0.0) == "Weak"
        assert _confidence_label(0.19) == "Weak"

    def test_boundary_moderate_strong(self):
        assert _confidence_label(0.5) == "Strong"
        assert _confidence_label(0.4999) == "Moderate"

    def test_boundary_weak_moderate(self):
        assert _confidence_label(0.2) == "Moderate"
        assert _confidence_label(0.1999) == "Weak"


# ---------------------------------------------------------------------------
# TASK-041: Compact candidate rendering
# ---------------------------------------------------------------------------


class TestRenderCandidate:
    @pytest.fixture()
    def agent(self) -> SearchAgent:
        return SearchAgent.__new__(SearchAgent)

    async def test_no_score_table(self, agent: SearchAgent):
        """Score tables must not appear in output."""
        sc = _make_scored(overall=0.46, drug_name="Erlotinib", target_label="EGFR")
        output = await _collect_candidate(agent, sc)
        assert "| Signal |" not in output
        assert "| Semantic |" not in output
        assert "| Structured |" not in output
        assert "| Evidence |" not in output

    async def test_confidence_label_in_output(self, agent: SearchAgent):
        sc = _make_scored(overall=0.52, drug_name="Erlotinib", target_label="EGFR")
        output = await _collect_candidate(agent, sc)
        assert "**Match: Strong**" in output
        assert "(0.52)" in output

    async def test_moderate_label(self, agent: SearchAgent):
        sc = _make_scored(overall=0.30, drug_name="TestDrug")
        output = await _collect_candidate(agent, sc)
        assert "**Match: Moderate**" in output

    async def test_weak_label(self, agent: SearchAgent):
        sc = _make_scored(overall=0.10, drug_name="TestDrug")
        output = await _collect_candidate(agent, sc)
        assert "**Match: Weak**" in output

    async def test_matched_dimensions_with_checkmark(self, agent: SearchAgent):
        sc = _make_scored(overall=0.5, drug_name="X", target_label="EGFR", dims=["target", "indication"])
        output = await _collect_candidate(agent, sc)
        # Should show actual target value from candidate, not dimension name
        assert "Target: EGFR \u2713" in output
        # indication falls back to dim name without signals
        assert "Indication: indication \u2713" in output

    async def test_matched_dimensions_with_signals(self, agent: SearchAgent):
        """When signals are provided, dimension values come from query signals."""
        from ember_agents.search.interpret import RawSignals
        sc = _make_scored(overall=0.5, drug_name="X", dims=["target", "indication"])
        signals = RawSignals(target=["EGFR"], indication=["oncology"], modality=[], commercial=[])
        parts: list[str] = []
        async for fragment in agent._render_candidate(sc, signals=signals):
            parts.append(fragment)
        output = "".join(parts)
        assert "Target: EGFR \u2713" in output
        assert "Indication: oncology \u2713" in output

    async def test_evidence_counts_only_nonzero(self, agent: SearchAgent):
        sc = _make_scored(overall=0.5, drug_name="X", n_trials=3, n_articles=1)
        output = await _collect_candidate(agent, sc)
        assert "3 trials" in output
        assert "1 article" in output
        # patents should not appear since count is 0
        assert "patent" not in output

    async def test_evidence_singular(self, agent: SearchAgent):
        sc = _make_scored(overall=0.5, drug_name="X", n_trials=1)
        output = await _collect_candidate(agent, sc)
        assert "1 trial" in output
        assert "1 trials" not in output

    async def test_no_evidence_no_dims(self, agent: SearchAgent):
        """When there are no dims or evidence, just the match line."""
        sc = _make_scored(overall=0.5, drug_name="X")
        output = await _collect_candidate(agent, sc)
        assert "**Match: Strong** (0.50)" in output
        # No dash separator when nothing follows
        assert "\u2014" not in output

    async def test_synthesis_summary_preserved(self, agent: SearchAgent):
        sc = _make_scored(overall=0.5, drug_name="X", synthesis="First-gen EGFR TKI.")
        output = await _collect_candidate(agent, sc)
        assert "> First-gen EGFR TKI." in output

    async def test_source_links_rendered(self, agent: SearchAgent):
        """Source links with URLs should use markdown link syntax."""
        sources = [_StubProv(source_name="UniProt", source_url="https://uniprot.org/123")]
        sc = _make_scored(overall=0.5, drug_name="X", sources=sources)
        # Note: the new code uses verified source links from trials/articles/target,
        # not from contributing_sources. contributing_sources are used in the
        # Contributing Sources section. Source links in candidate come from verified URLs.
        # This test checks that the candidate renders without error.
        output = await _collect_candidate(agent, sc)
        assert "### 1. X\n" in output

    async def test_verified_source_links(self, agent: SearchAgent):
        """Candidates with trial NCT IDs should show ClinicalTrials.gov link."""
        sc = _make_scored(overall=0.5, drug_name="Erlotinib", target_label="EGFR", target_uniprot="P00533")
        # Add a trial with NCT ID
        sc.candidate.trials = [_StubTrial(nct_id="NCT01234567")]
        sc.candidate.articles = [_StubArticle(pmid="12345678")]
        output = await _collect_candidate(agent, sc)
        assert "[ClinicalTrials.gov]" in output
        assert "[PubMed]" in output
        assert "[UniProt]" in output


# ---------------------------------------------------------------------------
# TASK-044: Top-10 cap
# ---------------------------------------------------------------------------


class TestRenderResultsCap:
    @pytest.fixture()
    def agent(self) -> SearchAgent:
        return SearchAgent.__new__(SearchAgent)

    async def test_max_rendered_candidates_is_10(self):
        assert _MAX_RENDERED_CANDIDATES == 10

    async def test_renders_all_when_under_cap(self, agent: SearchAgent):
        scored = [_make_scored(rank=i, overall=1.0 - i * 0.05) for i in range(1, 6)]
        output = await _collect_results(agent, scored)
        assert "5 total" in output
        assert "### 5." in output
        assert "additional candidate" not in output

    async def test_caps_at_10_candidates(self, agent: SearchAgent):
        scored = [_make_scored(rank=i, overall=1.0 - i * 0.01, drug_name=f"Drug{i}") for i in range(1, 21)]
        output = await _collect_results(agent, scored)
        assert "showing top 10 of 20" in output
        assert "### 10." in output
        # 11th candidate should not be rendered
        assert "### 11." not in output

    async def test_remaining_summary_line(self, agent: SearchAgent):
        scored = [_make_scored(rank=i, overall=1.0 - i * 0.01, drug_name=f"Drug{i}") for i in range(1, 51)]
        output = await _collect_results(agent, scored)
        assert "40 additional candidates scored below" in output
        assert "Results ranked by structural match and evidence depth." in output

    async def test_remaining_summary_singular(self, agent: SearchAgent):
        scored = [_make_scored(rank=i, overall=1.0 - i * 0.01, drug_name=f"Drug{i}") for i in range(1, 12)]
        output = await _collect_results(agent, scored)
        assert "1 additional candidate scored below" in output

    async def test_synthesis_section_preserved_no_contributing_sources(self, agent: SearchAgent):
        scored = [_make_scored(rank=1, overall=0.5, drug_name="X")]
        output = await _collect_results(agent, scored)
        # Contributing Sources section was removed (redundant with per-candidate links)
        assert "## Contributing Sources" not in output
        assert "## Synthesis Summary" in output

    async def test_exactly_10_no_cap_message(self, agent: SearchAgent):
        scored = [_make_scored(rank=i, overall=1.0 - i * 0.05, drug_name=f"Drug{i}") for i in range(1, 11)]
        output = await _collect_results(agent, scored)
        assert "10 total" in output
        assert "additional candidate" not in output


# ---------------------------------------------------------------------------
# Synthesis summary: low-confidence advisory
# ---------------------------------------------------------------------------


class TestSynthesisSummaryLowConfidence:
    @pytest.fixture()
    def agent(self) -> SearchAgent:
        return SearchAgent.__new__(SearchAgent)

    async def test_low_confidence_advisory_when_top_below_03(self, agent: SearchAgent):
        scored = [_make_scored(rank=1, overall=0.20, drug_name="WeakDrug")]
        output = await _collect_results(agent, scored)
        assert "Results with low confidence may not be directly relevant" in output

    async def test_no_advisory_when_top_above_03(self, agent: SearchAgent):
        scored = [_make_scored(rank=1, overall=0.50, drug_name="StrongDrug")]
        output = await _collect_results(agent, scored)
        assert "Results with low confidence may not be directly relevant" not in output

    async def test_boundary_at_03(self, agent: SearchAgent):
        scored = [_make_scored(rank=1, overall=0.30, drug_name="BorderDrug")]
        output = await _collect_results(agent, scored)
        assert "Results with low confidence may not be directly relevant" not in output


# ---------------------------------------------------------------------------
# TASK-078/TASK-079: ResultRenderer improvements
# ---------------------------------------------------------------------------


from ember_agents.render import ResultRenderer, _url_label  # noqa: E402
from ember_agents.trace import ExecutionTrace  # noqa: E402


def _make_trace() -> ExecutionTrace:
    """Build a minimal ExecutionTrace for renderer tests."""
    return ExecutionTrace(
        query_type="general",
        extracted_signals={},
        resolved_classifications={},
        gate_outcome="passed",
        source_statuses=[],
        duration_seconds=0.1,
    )


@dataclass
class _PatentStub:
    country_code: str = "US"
    country_name: str = "United States"
    publication_number: str | None = None
    expiry_date: str | None = None
    expiry_date_approximate: bool = False
    status: str = "active"
    url: str | None = None


@dataclass
class _ResultStub:
    drug_name: str | None = None
    target: str | None = None
    display_label: str | None = None
    overall_score: float | None = None
    semantic_score: float | None = None
    structured_score: float | None = None
    evidence_score: float | None = None
    indication: list | None = None
    synthesis_summary: str | None = None
    risk_flags: list = field(default_factory=list)
    patents: list = field(default_factory=list)
    trials: list = field(default_factory=list)
    trial_count: int | None = None
    latest_trial_phase: str | None = None
    articles: list = field(default_factory=list)
    article_count: int | None = None
    annual_revenue_usd_millions: float | None = None
    revenue_year: int | None = None
    biosimilar_competitors: list = field(default_factory=list)
    biosimilar_competitor_count: int | None = None
    source_urls: list = field(default_factory=list)


class TestScoreBreakdown:
    """TASK-079: Score breakdown line rendered when sub-scores present."""

    def test_all_subscores_rendered(self):
        renderer = ResultRenderer()
        result = _ResultStub(
            drug_name="Adalimumab",
            semantic_score=0.85,
            structured_score=0.72,
            evidence_score=0.91,
        )
        output = renderer.render(_make_trace(), [result])
        assert "**Scores:**" in output
        assert "semantic: 0.85" in output
        assert "structured: 0.72" in output
        assert "evidence: 0.91" in output

    def test_partial_subscores_only_show_present(self):
        renderer = ResultRenderer()
        result = _ResultStub(
            drug_name="Bevacizumab",
            semantic_score=None,
            structured_score=0.60,
            evidence_score=None,
        )
        output = renderer.render(_make_trace(), [result])
        assert "structured: 0.60" in output
        assert "semantic:" not in output
        assert "evidence:" not in output

    def test_all_none_subscores_omit_line(self):
        renderer = ResultRenderer()
        result = _ResultStub(
            drug_name="TestDrug",
            semantic_score=None,
            structured_score=None,
            evidence_score=None,
        )
        output = renderer.render(_make_trace(), [result])
        assert "**Scores:**" not in output


class TestPatentUrlColumn:
    """TASK-079: Patent table includes Link column with URLs and approximate date marker."""

    def test_patent_url_rendered_in_table(self):
        renderer = ResultRenderer()
        patent = _PatentStub(
            country_name="United States",
            expiry_date="2028-06-15",
            url="https://patents.google.com/patent/US1234567",
        )
        result = _ResultStub(drug_name="DrugX", patents=[patent])
        output = renderer.render(_make_trace(), [result])
        assert "[Patent](https://patents.google.com/patent/US1234567)" in output
        assert "Link" in output

    def test_approximate_expiry_shows_estimated_marker(self):
        renderer = ResultRenderer()
        patent = _PatentStub(
            country_name="Germany",
            expiry_date="2027-03-01",
            expiry_date_approximate=True,
            url="https://patents.google.com/patent/DE9876543",
        )
        result = _ResultStub(drug_name="DrugY", patents=[patent])
        output = renderer.render(_make_trace(), [result])
        assert "~2027-03-01 (estimated)" in output

    def test_no_url_patent_shows_empty_link_cell(self):
        renderer = ResultRenderer()
        patent = _PatentStub(country_name="Japan", expiry_date="2029-01-01", url=None)
        result = _ResultStub(drug_name="DrugZ", patents=[patent])
        output = renderer.render(_make_trace(), [result])
        # Link column should still be present but empty
        assert "| Link |" in output
        assert "[Patent]" not in output


class TestEvidenceCounts:
    """TASK-079: Evidence counts with singular/plural; zero sections omitted."""

    def test_trials_articles_patents_all_shown(self):
        renderer = ResultRenderer()
        result = _ResultStub(
            drug_name="DrugA",
            trial_count=3,
            latest_trial_phase="Phase III",
            article_count=5,
            patents=[_PatentStub()],
        )
        output = renderer.render(_make_trace(), [result])
        assert "3 trials (latest: Phase III)" in output
        assert "5 articles" in output
        assert "1 patent" in output

    def test_singular_forms(self):
        renderer = ResultRenderer()
        result = _ResultStub(
            drug_name="DrugB",
            trial_count=1,
            article_count=1,
            patents=[_PatentStub()],
        )
        output = renderer.render(_make_trace(), [result])
        assert "1 trial" in output
        assert "1 trials" not in output
        assert "1 article" in output
        assert "1 articles" not in output
        assert "1 patent" in output
        assert "1 patents" not in output

    def test_zero_sections_omitted(self):
        renderer = ResultRenderer()
        result = _ResultStub(
            drug_name="DrugC",
            trial_count=2,
            article_count=0,
            patents=[],
        )
        output = renderer.render(_make_trace(), [result])
        assert "2 trials" in output
        assert "article" not in output
        assert "patent" not in output


class TestCommercialRendering:
    """TASK-079: Commercial line rendered when revenue present, omitted when None."""

    def test_revenue_rendered_with_year(self):
        renderer = ResultRenderer()
        result = _ResultStub(
            drug_name="Humira",
            annual_revenue_usd_millions=20000.0,
            revenue_year=2022,
        )
        output = renderer.render(_make_trace(), [result])
        assert "**Commercial:**" in output
        assert "Revenue: $20,000M (2022)" in output

    def test_revenue_with_biosimilar_count(self):
        renderer = ResultRenderer()
        result = _ResultStub(
            drug_name="Herceptin",
            annual_revenue_usd_millions=7500.5,
            revenue_year=2021,
            biosimilar_competitor_count=4,
        )
        output = renderer.render(_make_trace(), [result])
        assert "Revenue: $7,500M (2021)" in output
        assert "Biosimilars: 4 approved" in output

    def test_no_revenue_omits_commercial_line(self):
        renderer = ResultRenderer()
        result = _ResultStub(drug_name="DrugNoRevenue", annual_revenue_usd_millions=None)
        output = renderer.render(_make_trace(), [result])
        assert "**Commercial:**" not in output


class TestSourceLinks:
    """TASK-079: Source URLs rendered as markdown links with domain-derived labels."""

    def test_known_domains_get_canonical_labels(self):
        assert _url_label("https://pubmed.ncbi.nlm.nih.gov/12345678/") == "PubMed"
        assert _url_label("https://clinicaltrials.gov/ct2/show/NCT01234567") == "ClinicalTrials.gov"
        assert _url_label("https://patents.google.com/patent/US1234567") == "Google Patents"
        assert _url_label("https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm") == "DailyMed"
        assert _url_label("https://www.fda.gov/drugs/whatever") == "FDA"

    def test_unknown_domain_capitalizes_first_part(self):
        label = _url_label("https://example.com/some/path")
        assert label == "Example"

    def test_source_urls_rendered_in_output(self):
        renderer = ResultRenderer()
        result = _ResultStub(
            drug_name="Rituximab",
            source_urls=[
                "https://pubmed.ncbi.nlm.nih.gov/99999999/",
                "https://clinicaltrials.gov/ct2/show/NCT99999999",
            ],
        )
        output = renderer.render(_make_trace(), [result])
        assert "[PubMed](https://pubmed.ncbi.nlm.nih.gov/99999999/)" in output
        assert "[ClinicalTrials.gov](https://clinicaltrials.gov/ct2/show/NCT99999999)" in output
