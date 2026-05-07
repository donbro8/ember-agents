"""Gate phase: pre-search validation and breadth control.

Validates a SearchSpec before executing a search:
1. Ensures at least one core field (target, therapeutic_area, indications, drug_names)
    is populated.
2. Checks that pending disambiguations have been resolved.
3. Estimates the result count and, if it exceeds SearchSpec.max_results, identifies the
    broadest search dimension and returns resolver-derived closed-option narrowing choices.

All user-facing prompts emitted by this module are closed-option — never open-ended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

try:
    from ember_data import SearchSpec
except ImportError:  # pragma: no cover — ember-data not installed in this env
    from ember_agents.search.classify import SearchSpec  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Ordered from broadest to narrowest: we report the first one that is set.
_DIMENSION_PRIORITY: list[str] = [
    "therapeutic_area",
    "indications",
    "target",
    "drug_names",
]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ResultEstimator(Protocol):
    """Estimates the number of results a SearchSpec would return."""

    async def estimate(self, spec: SearchSpec) -> int: ...


@runtime_checkable
class NarrowingProvider(Protocol):
    """Returns closed-option narrowing choices for a given search dimension."""

    async def get_options(
        self, dimension: str, spec: SearchSpec
    ) -> list[tuple[str, str]]: ...


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NarrowingRequest:
    """A closed-option request for the user to narrow a broad search dimension.

    Attributes:
        dimension: The search dimension that is too broad (e.g. "therapeutic_area").
        question: Pre-formatted question string with numbered options.
        options: Ordered list of (value, label) tuples matching the numbered options.
    """

    dimension: str
    question: str
    options: list[tuple[str, str]]


@dataclass
class GateResult:
    """Result of the gate check.

    Attributes:
        passed: True when the spec is valid and the estimated result count is
            within the limit.
        reason: Machine-readable failure reason if passed is False.
            One of "missing_core_fields", "pending_disambiguations", "too_broad".
        narrowing: Present when reason is "too_broad"; contains a closed-option
            request for the user to constrain the broadest dimension.
    """

    passed: bool
    reason: str | None = None
    narrowing: NarrowingRequest | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_core_fields(spec: SearchSpec) -> bool:
    """Return True if at least one core search field is populated."""
    if getattr(spec, "target", None) is not None:
        return True
    if getattr(spec, "therapeutic_area", None) is not None:
        return True
    indications = getattr(spec, "indications", None)
    if indications:
        return True
    drug_names = getattr(spec, "drug_names", None)
    if drug_names:
        return True
    return False


def _identify_broadest_dimension(spec: SearchSpec) -> str:
    """Return the dimension that most broadens the search (highest priority set field)."""
    for dim in _DIMENSION_PRIORITY:
        val = getattr(spec, dim, None)
        if val is not None and val != []:
            return dim
    return _DIMENSION_PRIORITY[0]


def _build_narrowing_question(dimension: str, options: list[tuple[str, str]]) -> str:
    """Build a closed-option question string for narrowing a search dimension."""
    _LABELS = {
        "therapeutic_area": "therapeutic area",
        "indications": "indication",
        "target": "biological target",
        "drug_names": "drug name",
    }
    dim_label = _LABELS.get(dimension, dimension)
    lines = [f"Your search is too broad. Which {dim_label} would you like to focus on?"]
    for i, (value, label) in enumerate(options, start=1):
        lines.append(f"{i}. {value} — {label}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SearchGate
# ---------------------------------------------------------------------------


class SearchGate:
    """Validates a SearchSpec before executing a search.

    Checks (in order):
    1. Pending disambiguations — if any remain the gate blocks.
    2. Core fields — at least one of target / therapeutic_area / indications /
       drug_names must be set.
    3. Result estimate — if the estimated count exceeds ``SearchSpec.max_results``,
       identifies the broadest dimension and returns closed-option narrowing choices.
    """

    def __init__(
        self,
        estimator: ResultEstimator,
        narrowing_provider: NarrowingProvider,
    ) -> None:
        self._estimator = estimator
        self._narrowing_provider = narrowing_provider

    async def check(self, spec: SearchSpec) -> GateResult:
        """Run all gate checks against the spec.

        Args:
            spec: The SearchSpec produced by the Classify phase.

        Returns:
            GateResult with ``passed=True`` when all checks pass, or a failing
            result with a reason and optional narrowing request.
        """
        # 1. Pending disambiguations must be resolved first.
        pending = getattr(spec, "pending_disambiguations", [])
        if pending:
            return GateResult(passed=False, reason="pending_disambiguations")

        # 2. At least one core field is required.
        if not _has_core_fields(spec):
            return GateResult(passed=False, reason="missing_core_fields")

        # 3. Estimate result count and check against max_results.
        max_results: int | None = getattr(spec, "max_results", None)
        if max_results is not None:
            count = await self._estimator.estimate(spec)
            if count > max_results:
                broadest = _identify_broadest_dimension(spec)
                options = await self._narrowing_provider.get_options(broadest, spec)
                question = _build_narrowing_question(broadest, options)
                return GateResult(
                    passed=False,
                    reason="too_broad",
                    narrowing=NarrowingRequest(
                        dimension=broadest,
                        question=question,
                        options=options,
                    ),
                )

        return GateResult(passed=True)
