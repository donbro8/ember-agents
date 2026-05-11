"""Tests for the search Classify phase."""

from __future__ import annotations

from dataclasses import dataclass, field

from ember_agents.search.classify import (
    AMBIGUITY_MARGIN,
    DISAMBIGUATION_THRESHOLD,
    ClassificationOrchestrator,
    ClassificationResult,
)
from ember_agents.search.interpret import RawSignals


# ---------------------------------------------------------------------------
# Fake resolver infrastructure
# ---------------------------------------------------------------------------


@dataclass
class FakeCandidate:
    id: str
    label: str
    confidence: float


@dataclass
class FakeResolverResult:
    candidates: list[FakeCandidate]


@dataclass
class FakeResolver:
    """Configurable fake resolver for testing."""

    results: dict[str, list[FakeCandidate]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    async def resolve(self, term: str) -> FakeResolverResult:
        self.calls.append(term)
        return FakeResolverResult(candidates=self.results.get(term, []))


def _noop_resolver() -> FakeResolver:
    """Return a resolver that records calls but returns no candidates."""
    return FakeResolver()


def _make_orchestrator(
    *,
    uniprot: FakeResolver | None = None,
    modality: FakeResolver | None = None,
    mesh: FakeResolver | None = None,
    atc: FakeResolver | None = None,
) -> ClassificationOrchestrator:
    return ClassificationOrchestrator(
        uniprot_resolver=uniprot or _noop_resolver(),
        modality_resolver=modality or _noop_resolver(),
        mesh_resolver=mesh or _noop_resolver(),
        atc_resolver=atc or _noop_resolver(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_clear_winner_populates_search_spec() -> None:
    """A single high-confidence candidate should populate spec.target as a ResolvedTerm."""
    uniprot = FakeResolver(
        results={"HER2": [FakeCandidate(id="P04626", label="HER2", confidence=0.95)]}
    )
    orchestrator = _make_orchestrator(uniprot=uniprot)

    signals = RawSignals(target=["HER2"])
    result = await orchestrator.classify(signals)

    assert isinstance(result, ClassificationResult)
    assert result.disambiguations == []
    assert result.spec.target is not None
    assert result.spec.target.identifier == "P04626"
    assert any(rt.identifier == "P04626" for rt in result.spec.resolved_terms)


async def test_low_confidence_triggers_disambiguation() -> None:
    """A top candidate below DISAMBIGUATION_THRESHOLD should trigger a disambiguation request."""
    uniprot = FakeResolver(
        results={
            "BRAF": [
                FakeCandidate(id="P15056", label="B-Raf", confidence=0.7),
            ]
        }
    )
    orchestrator = _make_orchestrator(uniprot=uniprot)

    signals = RawSignals(target=["BRAF"])
    result = await orchestrator.classify(signals)

    assert len(result.disambiguations) == 1
    req = result.disambiguations[0]
    assert req.dimension == "target"
    assert req.raw_term == "BRAF"
    assert result.spec.target is None


async def test_within_margin_ambiguity_triggers_disambiguation() -> None:
    """Two candidates within AMBIGUITY_MARGIN of each other should trigger disambiguation."""
    uniprot = FakeResolver(
        results={
            "HER2": [
                FakeCandidate(id="P04626", label="HER2", confidence=0.85),
                FakeCandidate(id="P15056", label="B-Raf", confidence=0.82),
            ]
        }
    )
    orchestrator = _make_orchestrator(uniprot=uniprot)

    signals = RawSignals(target=["HER2"])
    result = await orchestrator.classify(signals)

    assert len(result.disambiguations) == 1
    req = result.disambiguations[0]
    assert req.dimension == "target"
    assert len(req.options) == 2
    assert result.spec.target is None


async def test_confidence_exactly_at_threshold_is_not_ambiguous() -> None:
    """A single candidate with confidence exactly at DISAMBIGUATION_THRESHOLD (0.8) is a clear winner."""
    assert DISAMBIGUATION_THRESHOLD == 0.8

    uniprot = FakeResolver(
        results={
            "CD19": [FakeCandidate(id="P15391", label="CD19", confidence=0.8)]
        }
    )
    orchestrator = _make_orchestrator(uniprot=uniprot)

    signals = RawSignals(target=["CD19"])
    result = await orchestrator.classify(signals)

    assert result.disambiguations == []
    assert result.spec.target is not None
    assert result.spec.target.identifier == "P15391"


async def test_closed_option_question_format() -> None:
    """DisambiguationRequest.question should include numbered options with id and label."""
    uniprot = FakeResolver(
        results={
            "HER2": [
                FakeCandidate(id="P04626", label="HER2", confidence=0.6),
                FakeCandidate(id="P15056", label="B-Raf", confidence=0.55),
            ]
        }
    )
    orchestrator = _make_orchestrator(uniprot=uniprot)

    signals = RawSignals(target=["HER2"])
    result = await orchestrator.classify(signals)

    assert len(result.disambiguations) == 1
    req = result.disambiguations[0]

    assert "Which biological target did you mean?" in req.question
    assert "1." in req.question
    assert "P04626" in req.question
    assert "HER2" in req.question
    assert "2." in req.question
    assert "P15056" in req.question
    assert "B-Raf" in req.question


async def test_empty_signal_skips_resolver_call() -> None:
    """If signals.target is empty, the uniprot_resolver should never be called."""
    uniprot = FakeResolver()
    orchestrator = _make_orchestrator(uniprot=uniprot)

    signals = RawSignals()  # all lists are empty
    result = await orchestrator.classify(signals)

    assert uniprot.calls == []
    assert result.disambiguations == []
    assert result.spec.target is None


async def test_modality_signals_go_to_resolved_terms() -> None:
    """Resolved modality terms go into resolved_terms; modality field stays None if id is not a valid ModalityClassification."""
    modality = FakeResolver(
        results={
            "mAb": [FakeCandidate(id="MAB001", label="Monoclonal Antibody", confidence=0.95)]
        }
    )
    orchestrator = _make_orchestrator(modality=modality)

    signals = RawSignals(modality=["mAb"])
    result = await orchestrator.classify(signals)

    assert result.disambiguations == []
    assert any(rt.identifier == "MAB001" for rt in result.spec.resolved_terms)
    # MAB001 is not a valid ModalityClassification enum value
    assert result.spec.modality is None


async def test_multiple_terms_in_dimension() -> None:
    """Multiple terms in a dimension should each be resolved independently."""
    uniprot = FakeResolver(
        results={
            "HER2": [FakeCandidate(id="P04626", label="HER2", confidence=0.95)],
            "PD-1": [FakeCandidate(id="Q15116", label="PD-1", confidence=0.92)],
        }
    )
    orchestrator = _make_orchestrator(uniprot=uniprot)

    signals = RawSignals(target=["HER2", "PD-1"])
    result = await orchestrator.classify(signals)

    assert result.disambiguations == []
    resolved_ids = {rt.identifier for rt in result.spec.resolved_terms}
    assert resolved_ids == {"P04626", "Q15116"}
    # First resolved target is set on spec.target
    assert result.spec.target is not None
    assert result.spec.target.identifier == "P04626"


async def test_constants_have_correct_values() -> None:
    """Constants should match the specified values."""
    assert DISAMBIGUATION_THRESHOLD == 0.8
    assert AMBIGUITY_MARGIN == 0.1


async def test_disambiguation_dimension_modality() -> None:
    """Ambiguous modality terms should use the modality dimension and prefix."""
    modality = FakeResolver(
        results={
            "antibody": [
                FakeCandidate(id="MAB", label="Monoclonal Antibody", confidence=0.75),
                FakeCandidate(id="BISP", label="Bispecific", confidence=0.72),
            ]
        }
    )
    orchestrator = _make_orchestrator(modality=modality)

    signals = RawSignals(modality=["antibody"])
    result = await orchestrator.classify(signals)

    assert len(result.disambiguations) == 1
    req = result.disambiguations[0]
    assert req.dimension == "modality"
    assert "Which drug modality did you mean?" in req.question


async def test_indication_resolves_through_both_mesh_and_atc() -> None:
    """Indication terms should be resolved by both mesh_resolver and atc_resolver."""
    mesh = FakeResolver(
        results={
            "breast cancer": [
                FakeCandidate(id="D001943", label="Breast Neoplasms", confidence=0.95)
            ]
        }
    )
    atc = FakeResolver(
        results={
            "breast cancer": [
                FakeCandidate(
                    id="L01XC",
                    label="Monoclonal antibodies (antineoplastic)",
                    confidence=0.88,
                )
            ]
        }
    )
    orchestrator = _make_orchestrator(mesh=mesh, atc=atc)

    signals = RawSignals(indication=["breast cancer"])
    result = await orchestrator.classify(signals)

    assert result.disambiguations == []
    assert len(result.spec.indications) == 1
    assert result.spec.indications[0].identifier == "D001943"
    assert result.spec.therapeutic_area is not None
    assert result.spec.therapeutic_area.identifier == "L01XC"


async def test_dir007_attribute_signals_mapped_into_spec() -> None:
    orchestrator = _make_orchestrator()
    signals = RawSignals(
        modality=["mAb"],
        cell_line=["mammalian cell line"],
        jurisdiction=["US"],
        commercial=["high-value", "revenue > 1B"],
        temporal={"after": "2025-01-01", "before": "2028-01-01"},
    )
    result = await orchestrator.classify(signals)

    assert getattr(result.spec, "min_revenue_millions", None) == 1000.0
    assert getattr(result.spec, "jurisdictions", None) == ["US"]
    assert getattr(result.spec, "query_type", None) == "general"
    assert getattr(result.spec, "cell_line_class", None) == "mammalian"
    window = getattr(result.spec, "patent_expiry_window", None)
    assert window is not None
    assert str(getattr(window, "start")) == "2025-01-01"
    assert str(getattr(window, "end")) == "2028-01-01"


async def test_dir007_cell_line_non_mammalian_normalizes_to_non_mammalian() -> None:
    orchestrator = _make_orchestrator()
    signals = RawSignals(cell_line=["nonmammalian cell line"])

    result = await orchestrator.classify(signals)

    assert getattr(result.spec, "cell_line_class", None) == "non-mammalian"


async def test_dir007_cell_line_mammalian_still_normalizes_to_mammalian() -> None:
    orchestrator = _make_orchestrator()
    signals = RawSignals(cell_line=["mammalian cell line"])

    result = await orchestrator.classify(signals)

    assert getattr(result.spec, "cell_line_class", None) == "mammalian"
