"""Tests for ResultSynthesizer with mocked LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ember_agents.synthesis import (
    ChangeSummarizer,
    DigestGenerator,
    DigestOutput,
    OpportunityHighlight,
    ResultSynthesizer,
    SynthesisOutput,
    WatchDigestInput,
    _MAX_CANDIDATES,
    _MAX_CHANGE_CONTEXT_RESULTS,
)


@dataclass
class FakeCandidate:
    """Minimal candidate stub for testing."""

    canonical_id: str = "CAND-001"
    drug_name: str = "testamab"
    target: str = "PD-1"
    indication: str = "NSCLC"
    overall_score: float = 0.85
    earliest_patent_expiry: str = "2028-06-15"
    earliest_expiry_jurisdiction: str = "US"
    annual_revenue_usd_millions: float = 5000.0
    biosimilar_competitor_count: int = 3
    trial_count: int = 12
    latest_trial_phase: str = "Phase 3"
    sources_contributed: int = 4


def _make_candidates(n: int) -> list[FakeCandidate]:
    return [
        FakeCandidate(canonical_id=f"CAND-{i:03d}", drug_name=f"drug-{i}")
        for i in range(n)
    ]


def _mock_llm_response(overview: str, per_candidate: dict, jurisdiction_insights: str | None = None) -> MagicMock:
    """Create a mock LLM response with JSON content."""
    payload = {
        "overview": overview,
        "per_candidate": per_candidate,
        "jurisdiction_insights": jurisdiction_insights,
    }
    resp = MagicMock()
    resp.content = json.dumps(payload)
    return resp


@pytest.fixture
def synthesizer() -> ResultSynthesizer:
    with patch("ember_agents.synthesis.ChatGoogleGenerativeAI"):
        return ResultSynthesizer()


@pytest.mark.asyncio
async def test_synthesize_returns_populated_output(synthesizer: ResultSynthesizer) -> None:
    """synthesize() should return a populated SynthesisOutput from mocked LLM."""
    candidates = _make_candidates(3)
    per_cand = {c.canonical_id: f"{c.drug_name} is notable." for c in candidates}
    mock_resp = _mock_llm_response(
        overview="These candidates target PD-1 with upcoming patent cliffs.",
        per_candidate=per_cand,
        jurisdiction_insights="US patents expire 6 months before EU.",
    )
    synthesizer._llm.ainvoke = AsyncMock(return_value=mock_resp)

    result = await synthesizer.synthesize(
        query="PD-1 inhibitors expiring before 2029",
        query_type="opportunity_scan",
        results=candidates,
        trace=None,
    )

    assert isinstance(result, SynthesisOutput)
    assert "PD-1" in result.overview
    assert len(result.per_candidate) == 3
    assert result.jurisdiction_insights is not None
    assert "US" in result.jurisdiction_insights


@pytest.mark.asyncio
async def test_graceful_degradation_on_llm_error(synthesizer: ResultSynthesizer) -> None:
    """synthesize() should return safe defaults when LLM raises."""
    synthesizer._llm.ainvoke = AsyncMock(side_effect=RuntimeError("API unavailable"))

    result = await synthesizer.synthesize(
        query="test query",
        query_type="general",
        results=_make_candidates(2),
        trace=None,
    )

    assert result.overview == "Synthesis unavailable."
    assert result.per_candidate == {}
    assert result.jurisdiction_insights is None


@pytest.mark.asyncio
async def test_only_top_10_candidates_in_prompt(synthesizer: ResultSynthesizer) -> None:
    """synthesize() should limit prompt context to top 10 candidates."""
    candidates = _make_candidates(15)
    per_cand = {f"CAND-{i:03d}": "notable" for i in range(10)}
    mock_resp = _mock_llm_response(overview="Top 10 analysed.", per_candidate=per_cand)
    synthesizer._llm.ainvoke = AsyncMock(return_value=mock_resp)

    result = await synthesizer.synthesize(
        query="broad scan",
        query_type="opportunity_scan",
        results=candidates,
        trace=None,
    )

    # Verify the LLM was called with at most 10 candidates in the prompt
    call_args = synthesizer._llm.ainvoke.call_args
    messages = call_args[0][0]
    human_msg = messages[1][1]
    parsed_candidates = json.loads(human_msg.split("Top candidates:\n")[1])
    assert len(parsed_candidates) == _MAX_CANDIDATES
    assert isinstance(result, SynthesisOutput)


@pytest.mark.asyncio
async def test_empty_results_list(synthesizer: ResultSynthesizer) -> None:
    """synthesize() should handle empty results without calling the LLM."""
    synthesizer._llm.ainvoke = AsyncMock()

    result = await synthesizer.synthesize(
        query="no results query",
        query_type="general",
        results=[],
        trace=None,
    )

    assert result.overview == "No candidates to synthesize."
    assert result.per_candidate == {}
    assert result.jurisdiction_insights is None
    synthesizer._llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_handles_markdown_fenced_response(synthesizer: ResultSynthesizer) -> None:
    """synthesize() should strip markdown fences from LLM response."""
    payload = json.dumps({
        "overview": "Fenced response.",
        "per_candidate": {"CAND-000": "interesting"},
        "jurisdiction_insights": None,
    })
    resp = MagicMock()
    resp.content = f"```json\n{payload}\n```"
    synthesizer._llm.ainvoke = AsyncMock(return_value=resp)

    result = await synthesizer.synthesize(
        query="test",
        query_type="general",
        results=_make_candidates(1),
        trace=None,
    )

    assert result.overview == "Fenced response."
    assert result.per_candidate == {"CAND-000": "interesting"}


@pytest.mark.asyncio
async def test_malformed_json_degrades_gracefully(synthesizer: ResultSynthesizer) -> None:
    """synthesize() should degrade gracefully on malformed JSON from LLM."""
    resp = MagicMock()
    resp.content = "not valid json at all"
    synthesizer._llm.ainvoke = AsyncMock(return_value=resp)

    result = await synthesizer.synthesize(
        query="test",
        query_type="general",
        results=_make_candidates(1),
        trace=None,
    )

    assert result.overview == "Synthesis unavailable."
    assert result.per_candidate == {}


# ---------------------------------------------------------------------------
# ChangeSummarizer tests
# ---------------------------------------------------------------------------

@dataclass
class FakeChangeEntry:
    """Minimal ChangeEntry stub for testing."""

    change_id: str = "CHG-001"
    watch_id: str = "W-001"
    run_id: str = "R-002"
    previous_run_id: str = "R-001"
    change_type: str = "new_candidate"
    canonical_id: str = "CAND-100"
    display_label: str = "Adalimumab"
    detail: str | None = None
    created_at: str = "2026-05-10T00:00:00Z"


def _make_changes(n: int) -> list[FakeChangeEntry]:
    labels = ["Adalimumab", "Pembrolizumab", "Nivolumab", "Trastuzumab", "Bevacizumab"]
    types = ["new_candidate", "score_change", "rank_change", "new_candidate", "removed"]
    return [
        FakeChangeEntry(
            change_id=f"CHG-{i:03d}",
            canonical_id=f"CAND-{i:03d}",
            display_label=labels[i % len(labels)],
            change_type=types[i % len(types)],
            detail=f"+0.{i}" if types[i % len(types)] == "score_change" else None,
        )
        for i in range(n)
    ]


@pytest.fixture
def change_summarizer() -> ChangeSummarizer:
    with patch("ember_agents.synthesis.ChatGoogleGenerativeAI"):
        return ChangeSummarizer()


@pytest.mark.asyncio
async def test_summarize_changes_returns_llm_summary(change_summarizer: ChangeSummarizer) -> None:
    """summarize_changes() should return the LLM-generated summary."""
    changes = _make_changes(2)
    resp = MagicMock()
    resp.content = "2 changes in your PD-1 watch. Adalimumab is a new entrant."
    change_summarizer._llm.ainvoke = AsyncMock(return_value=resp)

    result = await change_summarizer.summarize_changes(
        watch_name="PD-1 Watch",
        query="PD-1 inhibitors",
        changes=changes,
        current_results=_make_candidates(3),
    )

    assert result == "2 changes in your PD-1 watch. Adalimumab is a new entrant."
    change_summarizer._llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_summarize_changes_graceful_degradation(change_summarizer: ChangeSummarizer) -> None:
    """summarize_changes() should return plaintext fallback when LLM raises."""
    changes = _make_changes(2)
    change_summarizer._llm.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))

    result = await change_summarizer.summarize_changes(
        watch_name="PD-1 Watch",
        query="PD-1 inhibitors",
        changes=changes,
        current_results=_make_candidates(1),
    )

    assert result.startswith("Changes detected:")
    assert "Adalimumab" in result
    assert "Pembrolizumab" in result


@pytest.mark.asyncio
async def test_summarize_changes_empty_list(change_summarizer: ChangeSummarizer) -> None:
    """summarize_changes() should return 'No changes' without calling LLM."""
    change_summarizer._llm.ainvoke = AsyncMock()

    result = await change_summarizer.summarize_changes(
        watch_name="PD-1 Watch",
        query="PD-1 inhibitors",
        changes=[],
        current_results=_make_candidates(3),
    )

    assert result == "No changes detected."
    change_summarizer._llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_summarize_changes_limits_results_to_top_5(change_summarizer: ChangeSummarizer) -> None:
    """summarize_changes() should only include top 5 current_results in prompt."""
    changes = _make_changes(1)
    resp = MagicMock()
    resp.content = "Summary with limited context."
    change_summarizer._llm.ainvoke = AsyncMock(return_value=resp)

    result = await change_summarizer.summarize_changes(
        watch_name="Test Watch",
        query="broad scan",
        changes=changes,
        current_results=_make_candidates(10),
    )

    assert isinstance(result, str)
    # Verify prompt only contains 5 results
    call_args = change_summarizer._llm.ainvoke.call_args
    messages = call_args[0][0]
    human_msg = messages[1][1]
    parsed_results = json.loads(human_msg.split("Current top results:\n")[1])
    assert len(parsed_results) == _MAX_CHANGE_CONTEXT_RESULTS


# ---------------------------------------------------------------------------
# DigestGenerator tests
# ---------------------------------------------------------------------------


def _make_watch_input(
    name: str,
    n_changes: int = 0,
    change_summary: str | None = None,
) -> WatchDigestInput:
    """Create a WatchDigestInput with optional changes."""
    changes = _make_changes(n_changes) if n_changes > 0 else []
    return WatchDigestInput(
        watch_name=name,
        query=f"{name} query",
        changes=changes,
        change_summary=change_summary,
        latest_results=_make_candidates(3),
    )


def _mock_digest_response(
    summary: str,
    top_opportunities: list[dict] | None = None,
) -> MagicMock:
    """Create a mock LLM response for digest generation."""
    payload = {
        "summary": summary,
        "top_opportunities": top_opportunities or [],
    }
    resp = MagicMock()
    resp.content = json.dumps(payload)
    return resp


@pytest.fixture
def digest_generator() -> DigestGenerator:
    with patch("ember_agents.synthesis.ChatGoogleGenerativeAI"):
        return DigestGenerator()


@pytest.mark.asyncio
async def test_generate_digest_with_active_watches(
    digest_generator: DigestGenerator,
) -> None:
    """generate_digest() should return a complete DigestOutput from mocked LLM."""
    watches = [
        _make_watch_input("PD-1 Watch", n_changes=3, change_summary="3 new candidates entered."),
        _make_watch_input("HER2 Watch", n_changes=1, change_summary="Score shift detected."),
        _make_watch_input("Stable Watch"),
    ]
    mock_resp = _mock_digest_response(
        summary="Cross-watch analysis shows PD-1 and HER2 convergence.",
        top_opportunities=[
            {
                "display_label": "Adalimumab",
                "reason": "Appears in both PD-1 and HER2 watches.",
                "watch_name": "PD-1 Watch",
            },
        ],
    )
    digest_generator._llm.ainvoke = AsyncMock(return_value=mock_resp)

    result = await digest_generator.generate_digest(watches, period_days=7)

    assert isinstance(result, DigestOutput)
    assert "convergence" in result.summary
    assert len(result.per_watch) == 2
    assert result.per_watch[0].watch_name == "PD-1 Watch"
    assert result.per_watch[0].change_count == 3
    assert result.per_watch[1].watch_name == "HER2 Watch"
    assert len(result.top_opportunities) == 1
    assert isinstance(result.top_opportunities[0], OpportunityHighlight)
    assert result.top_opportunities[0].display_label == "Adalimumab"
    assert result.stable_watches == ["Stable Watch"]
    digest_generator._llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_generate_digest_graceful_degradation(
    digest_generator: DigestGenerator,
) -> None:
    """generate_digest() should return statistical fallback when LLM raises."""
    watches = [
        _make_watch_input("PD-1 Watch", n_changes=2, change_summary="2 changes."),
        _make_watch_input("HER2 Watch", n_changes=1),
    ]
    digest_generator._llm.ainvoke = AsyncMock(side_effect=RuntimeError("API down"))

    result = await digest_generator.generate_digest(watches)

    assert isinstance(result, DigestOutput)
    assert "2 watches had changes" in result.summary
    assert len(result.per_watch) == 2
    assert result.top_opportunities == []


@pytest.mark.asyncio
async def test_generate_digest_all_stable(
    digest_generator: DigestGenerator,
) -> None:
    """generate_digest() should skip LLM call when all watches are stable."""
    watches = [
        _make_watch_input("Watch A"),
        _make_watch_input("Watch B"),
    ]
    digest_generator._llm.ainvoke = AsyncMock()

    result = await digest_generator.generate_digest(watches)

    assert isinstance(result, DigestOutput)
    assert result.summary == "No changes across any watch."
    assert result.per_watch == []
    assert result.top_opportunities == []
    assert set(result.stable_watches) == {"Watch A", "Watch B"}
    digest_generator._llm.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_generate_digest_empty_watches(
    digest_generator: DigestGenerator,
) -> None:
    """generate_digest() should handle empty watches list without LLM call."""
    digest_generator._llm.ainvoke = AsyncMock()

    result = await digest_generator.generate_digest([])

    assert isinstance(result, DigestOutput)
    assert result.summary == "No changes across any watch."
    assert result.per_watch == []
    assert result.top_opportunities == []
    assert result.stable_watches == []
    digest_generator._llm.ainvoke.assert_not_called()
