"""Completeness report generator for Tier 2 integration tests.

Generates field population percentages across all 6 reference queries and
flags fields with low population rates as INVESTIGATE candidates.

Can be run standalone:
    uv run python tests/integration/completeness_report.py

Or imported for programmatic use:
    from tests.integration.completeness_report import generate_report, save_baselines
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).parents[3] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

try:
    from ember_data.models.result import CandidateResult, PatentJurisdiction
except ImportError as exc:
    print(f"ERROR: cannot import ember_data.models.result — {exc}", file=sys.stderr)
    sys.exit(1)

# Import reference data from conftest (works when run from package root)
_TESTS_ROOT = Path(__file__).parents[2]
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from tests.integration.conftest import QUERY_RESULTS  # noqa: E402

BASELINES_PATH = Path(__file__).parent / "baselines.json"

# Fields checked for completeness
ALWAYS_REQUIRED = [
    "result_type",
    "canonical_id",
    "display_label",
    "overall_score",
    "sources_contributed",
    "source_urls",
]
REQUIRED_FOR_DRUG = ["drug_name"]
OPTIONAL_ENRICHMENT = ["indication", "mechanism_of_action", "clinical_stage", "synthesis_summary"]
PATENT_FIELDS = [
    "country_code",
    "country_name",
    "expiry_date",
    "expiry_date_approximate",
    "url",
    "status",
]

# Population rate below this fraction triggers an INVESTIGATE flag
INVESTIGATE_THRESHOLD = 0.80


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _field_populated(result: CandidateResult, field_name: str) -> bool:
    """Return True if *field_name* is populated (non-None, non-empty)."""
    val = getattr(result, field_name, None)
    if val is None:
        return False
    if isinstance(val, (list, str)) and len(val) == 0:
        return False
    return True


def compute_rates(results: list[CandidateResult]) -> dict[str, float]:
    """Return per-field population rates (0.0–1.0) for *results*.

    Patent fields are prefixed with ``patent.`` and computed over all
    PatentJurisdiction objects found in the result set.
    """
    if not results:
        return {}

    rates: dict[str, float] = {}
    all_fields = ALWAYS_REQUIRED + REQUIRED_FOR_DRUG + OPTIONAL_ENRICHMENT

    for fname in all_fields:
        populated = sum(1 for r in results if _field_populated(r, fname))
        rates[fname] = populated / len(results)

    # Patent-level fields
    all_pjs: list[PatentJurisdiction] = []
    for r in results:
        all_pjs.extend(r.patents)

    for fname in PATENT_FIELDS:
        if not all_pjs:
            rates[f"patent.{fname}"] = 1.0
            continue
        populated = sum(1 for pj in all_pjs if getattr(pj, fname, None) is not None)
        rates[f"patent.{fname}"] = populated / len(all_pjs)

    return rates


def generate_report(
    query_results: dict[str, list] | None = None,
    investigate_threshold: float = INVESTIGATE_THRESHOLD,
) -> dict[str, dict[str, float]]:
    """Generate population rates for all reference queries.

    Parameters
    ----------
    query_results:
        Mapping of query string → list of CandidateResult objects.
        Defaults to the pre-built QUERY_RESULTS from conftest.
    investigate_threshold:
        Fields with population rate below this value are flagged INVESTIGATE.

    Returns
    -------
    dict mapping query → {field_name: rate} for all checked fields.
    """
    if query_results is None:
        query_results = QUERY_RESULTS

    report: dict[str, dict[str, float]] = {}

    print("=" * 70)
    print("EMBER INTEGRATION TEST — FIELD COMPLETENESS REPORT")
    print("=" * 70)

    for query, results in query_results.items():
        print(f"\nQuery: {query!r}")
        print(f"  Results: {len(results)}")

        if not results:
            print("  WARNING: No results — skipping field analysis")
            report[query] = {}
            continue

        rates = compute_rates(results)
        report[query] = rates

        # Segregate by tier for display
        required_fields = ALWAYS_REQUIRED + REQUIRED_FOR_DRUG
        optional_fields = OPTIONAL_ENRICHMENT
        patent_keys = [k for k in rates if k.startswith("patent.")]

        print("\n  Required fields:")
        for fname in required_fields:
            rate = rates.get(fname, 0.0)
            flag = "  INVESTIGATE" if rate < investigate_threshold else ""
            bar = _progress_bar(rate)
            print(f"    {fname:<30} {bar} {rate:>6.1%}{flag}")

        print("\n  Optional enrichment:")
        for fname in optional_fields:
            rate = rates.get(fname, 0.0)
            flag = "  INVESTIGATE" if rate < investigate_threshold else ""
            bar = _progress_bar(rate)
            print(f"    {fname:<30} {bar} {rate:>6.1%}{flag}")

        if patent_keys:
            print("\n  Patent jurisdiction fields:")
            for key in patent_keys:
                rate = rates[key]
                short_key = key.replace("patent.", "")
                flag = "  INVESTIGATE" if rate < investigate_threshold else ""
                bar = _progress_bar(rate)
                print(f"    {short_key:<30} {bar} {rate:>6.1%}{flag}")

    print("\n" + "=" * 70)
    return report


def _progress_bar(rate: float, width: int = 20) -> str:
    """Return a text progress bar for *rate* (0.0–1.0)."""
    filled = round(rate * width)
    empty = width - filled
    return f"[{'#' * filled}{'.' * empty}]"


def save_baselines(
    report: dict[str, dict[str, float]],
    path: Path = BASELINES_PATH,
) -> None:
    """Write *report* to *path* as the new baselines.json."""
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nBaselines saved to: {path}")


def check_regressions(
    current: dict[str, dict[str, float]],
    baselines: dict[str, dict[str, float]],
    threshold_pp: float = 10.0,
) -> list[str]:
    """Compare current rates against baselines.

    Returns a list of regression strings (empty = no regressions).
    A regression is when a field's population rate drops more than
    *threshold_pp* percentage points below its baseline.
    """
    regressions: list[str] = []
    for query, baseline_rates in baselines.items():
        current_rates = current.get(query, {})
        for fname, baseline_rate in baseline_rates.items():
            current_rate = current_rates.get(fname, 0.0)
            drop_pp = (baseline_rate - current_rate) * 100
            if drop_pp > threshold_pp:
                regressions.append(
                    f"REGRESSION [{query!r}] {fname}: "
                    f"baseline={baseline_rate:.1%} current={current_rate:.1%} "
                    f"drop={drop_pp:.1f}pp"
                )
    return regressions


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate field completeness report for integration test reference queries."
    )
    parser.add_argument(
        "--save-baselines",
        action="store_true",
        help="Save current rates as new baselines.json",
    )
    parser.add_argument(
        "--check-regressions",
        action="store_true",
        help="Compare against existing baselines.json and report regressions",
    )
    parser.add_argument(
        "--investigate-threshold",
        type=float,
        default=INVESTIGATE_THRESHOLD,
        help=f"Rate below which fields are flagged INVESTIGATE (default: {INVESTIGATE_THRESHOLD})",
    )
    args = parser.parse_args()

    report = generate_report(investigate_threshold=args.investigate_threshold)

    if args.save_baselines:
        save_baselines(report)

    if args.check_regressions:
        if not BASELINES_PATH.exists():
            print(
                f"\nERROR: baselines.json not found at {BASELINES_PATH}. "
                "Run with --save-baselines first.",
                file=sys.stderr,
            )
            sys.exit(1)

        existing = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
        regressions = check_regressions(report, existing)

        if regressions:
            print("\n" + "=" * 70)
            print("REGRESSIONS DETECTED:")
            for reg in regressions:
                print(f"  {reg}")
            print("=" * 70)
            sys.exit(1)
        else:
            print("\nNo regressions detected — all field rates within threshold.")
