"""Tests for the search Gate phase."""

from __future__ import annotations

from dataclasses import dataclass, field


from ember_agents.search.classify import ResolvedTerm, SearchSpec
from ember_agents.search.gate import NarrowingRequest, SearchGate

try:
    from ember_data import DisambiguationRequest as EmberDisambiguationRequest
except ImportError:  # pragma: no cover — ember-data not installed
    EmberDisambiguationRequest = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Fake infrastructure
# ---------------------------------------------------------------------------


@dataclass
class FakeEstimator:
    """Returns a fixed estimate regardless of the spec."""

    count: int = 0

    async def estimate(self, spec: SearchSpec) -> int:
        return self.count


@dataclass
class FakeNarrowingProvider:
    """Returns a fixed set of options for any dimension."""

    options: list[tuple[str, str]] = field(default_factory=list)

    async def get_options(self, dimension: str, spec: SearchSpec) -> list[tuple[str, str]]:
        return self.options


def _make_gate(
    estimate: int = 0,
    narrowing_options: list[tuple[str, str]] | None = None,
) -> SearchGate:
    return SearchGate(
        estimator=FakeEstimator(count=estimate),
        narrowing_provider=FakeNarrowingProvider(
            options=narrowing_options or [("TA001", "Oncology"), ("TA002", "Immunology")]
        ),
    )


def _spec(**kwargs) -> SearchSpec:
    """Build a SearchSpec stub with sensible defaults."""
    defaults: dict = {
        "target": None,
        "therapeutic_area": None,
        "drug_names": [],
        "indications": [],
        "modality": None,
        "resolved_terms": [],
        "pending_disambiguations": [],
    }
    defaults.update(kwargs)
    return SearchSpec(**defaults)


def _resolved_term(
    taxonomy: str = "target",
    identifier: str = "P04626",
    label: str = "HER2",
    confidence: float = 0.95,
) -> ResolvedTerm:
    """Build a ResolvedTerm using the real ember_data model (or fallback stub)."""
    return ResolvedTerm(taxonomy=taxonomy, identifier=identifier, label=label, confidence=confidence)


def _pending_disambiguation(
    raw_input: str = "BRAF",
    rationale: str = "Which biological target did you mean?",
) -> object:
    """Build a pending DisambiguationRequest for pending_disambiguations.

    Uses the real ember_data.DisambiguationRequest when available, otherwise a
    plain object that is truthy (sufficient for gate truthiness check).
    """
    option = _resolved_term(identifier="P15056", label=raw_input)
    if EmberDisambiguationRequest is not None:
        return EmberDisambiguationRequest(
            raw_input=raw_input,
            options=[option],
            rationale=rationale,
        )
    # Fallback: a simple truthy object with the expected attributes.
    @dataclass
    class _FallbackDisambiguation:
        raw_input: str
        rationale: str

    return _FallbackDisambiguation(raw_input=raw_input, rationale=rationale)


# ---------------------------------------------------------------------------
# Gate pass
# ---------------------------------------------------------------------------


async def test_gate_passes_with_target_set() -> None:
    """Gate should pass when target is populated and estimate is within limit."""
    gate = _make_gate(estimate=10)
    spec = _spec(target=_resolved_term(taxonomy="target", identifier="P04626", label="HER2"), max_results=100)
    result = await gate.check(spec)

    assert result.passed is True
    assert result.reason is None
    assert result.narrowing is None


async def test_gate_passes_with_therapeutic_area_within_limit() -> None:
    """Gate should pass when therapeutic_area is set and estimate <= max_results."""
    gate = _make_gate(estimate=50)
    spec = _spec(
        therapeutic_area=_resolved_term(taxonomy="indication_atc", identifier="L01XC", label="Antineoplastics", confidence=0.9),
        max_results=50,
    )
    result = await gate.check(spec)

    assert result.passed is True


async def test_gate_passes_with_indications_set() -> None:
    """Gate should pass when indications list is non-empty."""
    gate = _make_gate(estimate=5)
    spec = _spec(
        indications=[_resolved_term(taxonomy="indication", identifier="D001943", label="Breast Neoplasms")],
        max_results=20,
    )
    result = await gate.check(spec)

    assert result.passed is True


async def test_gate_passes_with_drug_names_set() -> None:
    """Gate should pass when drug_names is non-empty."""
    gate = _make_gate(estimate=3)
    spec = _spec(drug_names=["trastuzumab"], max_results=20)
    result = await gate.check(spec)

    assert result.passed is True


async def test_gate_honors_default_max_results() -> None:
    """Gate should pass when estimate is within the default max_results=500."""
    gate = _make_gate(estimate=499)
    spec = _spec(target=_resolved_term(taxonomy="target", identifier="P04626", label="HER2"))
    result = await gate.check(spec)

    assert result.passed is True


async def test_gate_fails_too_broad_at_default_max_results() -> None:
    """Gate should fail too_broad when estimate exceeds the default max_results=500."""
    gate = _make_gate(estimate=99999)
    spec = _spec(target=_resolved_term(taxonomy="target", identifier="P04626", label="HER2"))
    result = await gate.check(spec)

    assert result.passed is False
    assert result.reason == "too_broad"


# ---------------------------------------------------------------------------
# Missing core fields
# ---------------------------------------------------------------------------


async def test_gate_fails_when_no_core_fields() -> None:
    """Gate should fail with 'missing_core_fields' when all core fields are empty."""
    gate = _make_gate()
    spec = _spec()  # all core fields are None/empty
    result = await gate.check(spec)

    assert result.passed is False
    assert result.reason == "missing_core_fields"
    assert result.narrowing is None


async def test_gate_fails_missing_core_fields_even_with_modality() -> None:
    """Modality alone does not satisfy the core-field requirement."""
    gate = _make_gate()
    spec = _spec(modality="mab")
    result = await gate.check(spec)

    assert result.passed is False
    assert result.reason == "missing_core_fields"


# ---------------------------------------------------------------------------
# Pending disambiguations (low-confidence / within-0.1 ambiguity)
# ---------------------------------------------------------------------------


async def test_gate_blocked_by_pending_disambiguation() -> None:
    """Gate should fail when spec has unresolved disambiguation items."""
    gate = _make_gate()
    spec = _spec(
        pending_disambiguations=[_pending_disambiguation(raw_input="BRAF", rationale="Which biological target did you mean?")],
    )
    result = await gate.check(spec)

    assert result.passed is False
    assert result.reason == "pending_disambiguations"


async def test_gate_blocked_by_low_confidence_disambiguation() -> None:
    """Low-confidence candidates produce pending_disambiguations that block the gate."""
    gate = _make_gate()
    spec = _spec(
        pending_disambiguations=[
            _pending_disambiguation(
                raw_input="her2",
                rationale="Which biological target did you mean?\n1. P04626 — HER2",
            )
        ],
    )
    result = await gate.check(spec)

    assert result.passed is False
    assert result.reason == "pending_disambiguations"


async def test_gate_blocked_by_within_margin_ambiguity() -> None:
    """Two candidates within the 0.1 ambiguity margin leave a pending disambiguation."""
    gate = _make_gate()
    spec = _spec(
        pending_disambiguations=[
            _pending_disambiguation(
                raw_input="HER2",
                rationale=(
                    "Which biological target did you mean?\n"
                    "1. P04626 — HER2\n"
                    "2. P15056 — B-Raf"
                ),
            )
        ],
    )
    result = await gate.check(spec)

    assert result.passed is False
    assert result.reason == "pending_disambiguations"


# ---------------------------------------------------------------------------
# Too broad
# ---------------------------------------------------------------------------


async def test_gate_fails_too_broad_returns_narrowing() -> None:
    """When estimate exceeds max_results, gate returns a narrowing request."""
    gate = _make_gate(
        estimate=500,
        narrowing_options=[
            ("P04626", "HER2 / ErbB2"),
            ("P15056", "BRAF"),
            ("Q15116", "PD-1"),
        ],
    )
    spec = _spec(
        therapeutic_area=_resolved_term(taxonomy="indication_atc", identifier="L01XC", label="Antineoplastics", confidence=0.9),
        max_results=100,
    )
    result = await gate.check(spec)

    assert result.passed is False
    assert result.reason == "too_broad"
    assert isinstance(result.narrowing, NarrowingRequest)
    assert result.narrowing.dimension == "therapeutic_area"
    assert len(result.narrowing.options) == 3


async def test_gate_too_broad_broadest_dimension_is_therapeutic_area() -> None:
    """When both therapeutic_area and indications are set, therapeutic_area is broadest."""
    gate = _make_gate(estimate=200)
    spec = _spec(
        therapeutic_area=_resolved_term(taxonomy="indication_atc", identifier="L01XC", label="Antineoplastics", confidence=0.9),
        indications=[_resolved_term(taxonomy="indication", identifier="D001943", label="Breast Neoplasms")],
        max_results=50,
    )
    result = await gate.check(spec)

    assert result.passed is False
    assert result.reason == "too_broad"
    assert result.narrowing is not None
    assert result.narrowing.dimension == "therapeutic_area"


async def test_gate_too_broad_broadest_dimension_is_indications_without_therapeutic_area() -> None:
    """When only indications is set (no therapeutic_area), indications is the broadest."""
    gate = _make_gate(estimate=300)
    spec = _spec(
        indications=[_resolved_term(taxonomy="indication", identifier="D001943", label="Breast Neoplasms")],
        max_results=50,
    )
    result = await gate.check(spec)

    assert result.passed is False
    assert result.reason == "too_broad"
    assert result.narrowing is not None
    assert result.narrowing.dimension == "indications"


# ---------------------------------------------------------------------------
# Closed-option narrowing question format
# ---------------------------------------------------------------------------


async def test_narrowing_question_is_closed_option() -> None:
    """Narrowing question must be closed-option with numbered choices."""
    gate = _make_gate(
        estimate=999,
        narrowing_options=[
            ("P04626", "HER2 / ErbB2"),
            ("Q15116", "PD-1"),
        ],
    )
    spec = _spec(
        therapeutic_area=_resolved_term(taxonomy="indication_atc", identifier="L01XC", label="Antineoplastics", confidence=0.9),
        max_results=10,
    )
    result = await gate.check(spec)

    assert result.narrowing is not None
    q = result.narrowing.question
    # Must contain numbered items
    assert "1." in q
    assert "2." in q
    # Must include option values and labels
    assert "P04626" in q
    assert "HER2 / ErbB2" in q
    assert "Q15116" in q
    assert "PD-1" in q
    # Must NOT be open-ended (no "?" without options would signal open-ended)
    # Validate it's closed by confirming options list is populated
    assert len(result.narrowing.options) == 2


async def test_narrowing_question_mentions_dimension() -> None:
    """The narrowing question should mention the dimension being constrained."""
    gate = _make_gate(
        estimate=500,
        narrowing_options=[("P04626", "HER2")],
    )
    spec = _spec(
        therapeutic_area=_resolved_term(taxonomy="indication_atc", identifier="L01XC", label="Antineoplastics", confidence=0.9),
        max_results=50,
    )
    result = await gate.check(spec)

    assert result.narrowing is not None
    assert "therapeutic area" in result.narrowing.question.lower()
