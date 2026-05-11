"""Tier 2 integration tests: 6 reference queries with field completeness assertions.

All tests are marked ``@pytest.mark.integration`` and use a fully-mocked
EmberAgent pipeline — safe for CI without network access.

Field completeness tiers:
  ALWAYS_REQUIRED: result_type, canonical_id, display_label, overall_score,
                   sources_contributed, source_urls
  REQUIRED_FOR_DRUG: drug_name (on ResultType.DRUG results only)
  PATENT_FIELDS: country_code, country_name, expiry_date, expiry_date_approximate,
                 url, status (on every PatentJurisdiction object)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from ember_data.models.result import CandidateResult, PatentJurisdiction, ResultType

from tests.integration.conftest import QUERY_RESULTS, make_mocked_agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALWAYS_REQUIRED = [
    "result_type",
    "canonical_id",
    "display_label",
    "overall_score",
    "sources_contributed",
    "source_urls",
]

REQUIRED_FOR_DRUG = ["drug_name"]

PATENT_FIELDS = [
    "country_code",
    "country_name",
    "expiry_date",
    "expiry_date_approximate",
    "url",
    "status",
]

# Map query → expected query_type for source coverage checks
QUERY_TYPE_MAP: dict[str, str] = {
    "adalimumab": "drug_lookup",
    "PD-1 inhibitors with patents expiring before 2028": "opportunity_scan",
    "biosimilar opportunities for biologics with revenue over 1 billion": "biosimilar_screen",
    "HER2 targeting antibodies for breast cancer": "opportunity_scan",
    "etanercept patent landscape": "drug_lookup",
    "VEGF pathway inhibitors": "general",
}

# Minimum expected source coverage per query_type
MIN_SOURCE_COVERAGE: dict[str, list[str]] = {
    "drug_lookup": ["bigquery_patents", "bigquery_fda"],
    "opportunity_scan": ["bigquery_patents"],
    "biosimilar_screen": ["bigquery_fda"],
    "general": [],
}

REFERENCE_QUERIES = list(QUERY_TYPE_MAP.keys())

BASELINES_PATH = Path(__file__).parent / "baselines.json"
REGRESSION_THRESHOLD_PP = 10  # percentage-point drop that triggers failure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_results(query: str) -> list[CandidateResult]:
    """Return the pre-built CandidateResult list for *query*."""
    return QUERY_RESULTS.get(query, [])


def _field_populated(result: CandidateResult, field_name: str) -> bool:
    """Return True if *field_name* is populated (non-None, non-empty) on *result*."""
    val = getattr(result, field_name, None)
    if val is None:
        return False
    if isinstance(val, (list, str)) and len(val) == 0:
        return False
    return True


def _compute_population_rates(results: list[CandidateResult]) -> dict[str, float]:
    """Return per-field population rates (0.0–1.0) across *results*."""
    if not results:
        return {}

    rates: dict[str, float] = {}
    all_fields = (
        ALWAYS_REQUIRED + REQUIRED_FOR_DRUG + ["indication", "mechanism_of_action"]
    )

    for fname in all_fields:
        populated = sum(1 for r in results if _field_populated(r, fname))
        rates[fname] = populated / len(results)

    # Patent field rates — computed over all PatentJurisdiction objects
    all_pjs: list[PatentJurisdiction] = []
    for r in results:
        all_pjs.extend(r.patents)

    for fname in PATENT_FIELDS:
        if not all_pjs:
            rates[f"patent.{fname}"] = 1.0  # vacuously complete
            continue
        populated = sum(1 for pj in all_pjs if getattr(pj, fname, None) is not None)
        rates[f"patent.{fname}"] = populated / len(all_pjs)

    return rates


def _load_baselines() -> dict[str, dict[str, float]]:
    """Load baseline population rates from baselines.json."""
    if not BASELINES_PATH.exists():
        return {}
    return json.loads(BASELINES_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Always-required field assertions (parametrized per query)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("query", REFERENCE_QUERIES)
def test_query_returns_candidates(query: str) -> None:
    """Each reference query must return at least one candidate result."""
    results = _collect_results(query)
    assert len(results) > 0, f"Query returned no candidates: {query!r}"


@pytest.mark.integration
def test_dir007_biosimilar_query_returns_at_least_five_candidates() -> None:
    """DIR-007 fixture query should return >= 5 qualifying candidates."""
    query = "biosimilar opportunities for biologics with revenue over 1 billion"
    results = _collect_results(query)
    assert len(results) >= 5, (
        f"Expected >=5 candidates for {query!r}, got {len(results)}"
    )


@pytest.mark.integration
@pytest.mark.parametrize("query", REFERENCE_QUERIES)
def test_always_required_fields_populated(query: str) -> None:
    """ALWAYS_REQUIRED fields must be populated on every result for every query."""
    results = _collect_results(query)
    assert results, f"No results to check for {query!r}"

    failures: list[str] = []
    for i, result in enumerate(results):
        for fname in ALWAYS_REQUIRED:
            if not _field_populated(result, fname):
                failures.append(
                    f"result[{i}] ({result.display_label!r}): field {fname!r} not populated"
                )

    if failures:
        msg = f"Query {query!r} — missing required fields:\n" + "\n".join(failures)
        pytest.fail(msg)


@pytest.mark.integration
@pytest.mark.parametrize("query", REFERENCE_QUERIES)
def test_drug_results_have_drug_name(query: str) -> None:
    """DRUG results must have drug_name populated."""
    results = _collect_results(query)
    failures: list[str] = []

    for i, result in enumerate(results):
        if result.result_type == ResultType.DRUG:
            if not _field_populated(result, "drug_name"):
                failures.append(
                    f"result[{i}] (canonical_id={result.canonical_id!r}): "
                    f"DRUG result missing drug_name"
                )

    if failures:
        msg = f"Query {query!r} — DRUG results missing drug_name:\n" + "\n".join(
            failures
        )
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Patent jurisdiction completeness
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("query", REFERENCE_QUERIES)
def test_patent_jurisdiction_field_completeness(query: str) -> None:
    """Every PatentJurisdiction must have all PATENT_FIELDS populated."""
    results = _collect_results(query)
    failures: list[str] = []

    for i, result in enumerate(results):
        for j, pj in enumerate(result.patents):
            for fname in PATENT_FIELDS:
                val = getattr(pj, fname, None)
                # expiry_date_approximate is a bool — None is not valid; False is valid
                if fname == "expiry_date_approximate":
                    if val is None:
                        failures.append(
                            f"result[{i}].patents[{j}] ({pj.country_code}): "
                            f"{fname!r} is None"
                        )
                elif val is None or (isinstance(val, str) and not val):
                    failures.append(
                        f"result[{i}].patents[{j}] ({pj.country_code}): "
                        f"{fname!r} not populated"
                    )

    if failures:
        msg = f"Query {query!r} — patent jurisdiction field gaps:\n" + "\n".join(
            failures
        )
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Source coverage per query_type
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("query", REFERENCE_QUERIES)
def test_source_coverage_per_query_type(query: str) -> None:
    """Expected sources must appear across all results for the query_type."""
    query_type = QUERY_TYPE_MAP[query]
    required_sources = MIN_SOURCE_COVERAGE.get(query_type, [])

    if not required_sources:
        pytest.skip(f"No source coverage requirements for query_type={query_type!r}")

    results = _collect_results(query)
    assert results, f"No results to check coverage for {query!r}"

    all_sources: set[str] = set()
    for result in results:
        all_sources.update(result.sources_contributed)

    missing = [s for s in required_sources if s not in all_sources]
    if missing:
        pytest.fail(
            f"Query {query!r} (type={query_type!r}) missing sources: {missing}. "
            f"Found: {sorted(all_sources)}"
        )


# ---------------------------------------------------------------------------
# Regression checks against baselines.json
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("query", REFERENCE_QUERIES)
def test_completeness_regression(query: str) -> None:
    """Population rates must not drop more than REGRESSION_THRESHOLD_PP below baseline."""
    baselines = _load_baselines()
    if not baselines or query not in baselines:
        pytest.skip(
            f"No baseline data for query {query!r} — run completeness_report.py first"
        )

    query_baseline = baselines[query]
    results = _collect_results(query)
    current_rates = _compute_population_rates(results)

    failures: list[str] = []
    for fname, baseline_rate in query_baseline.items():
        current = current_rates.get(fname, 0.0)
        drop_pp = (baseline_rate - current) * 100
        if drop_pp > REGRESSION_THRESHOLD_PP:
            failures.append(
                f"  {fname}: baseline={baseline_rate:.1%} current={current:.1%} "
                f"drop={drop_pp:.1f}pp (threshold={REGRESSION_THRESHOLD_PP}pp)"
            )
            logger.warning(
                "INVESTIGATE: field %r for query %r dropped %.1fpp below baseline",
                fname,
                query,
                drop_pp,
            )

    if failures:
        msg = f"Query {query!r} — completeness regression detected:\n" + "\n".join(
            failures
        )
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# EmberAgent pipeline integration (mocked, validates output structure)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize(
    "query,query_type",
    [(q, qt) for q, qt in QUERY_TYPE_MAP.items()],
)
async def test_agent_run_yields_output(query: str, query_type: str) -> None:
    """EmberAgent.run() must yield non-empty markdown output for each reference query."""
    agent = make_mocked_agent(query, query_type=query_type)
    chunks: list[str] = []
    async for chunk in agent.run(query):
        chunks.append(chunk)

    output = "".join(chunks)
    assert output.strip(), f"EmberAgent produced no output for {query!r}"
    assert query in output, f"Query string not in agent output for {query!r}"


@pytest.mark.integration
@pytest.mark.parametrize(
    "query,query_type",
    [(q, qt) for q, qt in QUERY_TYPE_MAP.items()],
)
async def test_agent_pipeline_phases_all_invoked(query: str, query_type: str) -> None:
    """All 6 pipeline phases must be called for each reference query."""
    agent = make_mocked_agent(query, query_type=query_type)

    async for _ in agent.run(query):
        pass

    agent._extractor.extract.assert_called_once()
    agent._classifier.classify.assert_called_once()
    agent._gate.check.assert_called_once()
    agent._fetcher.fetch.assert_called_once()
    agent._seed_source.fetch.assert_called_once()
    agent._scorer.score.assert_called_once()
