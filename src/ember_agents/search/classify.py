"""Classify phase: resolver orchestration for search signal classification.

Converts RawSignals (dimension-tagged strings) into a canonical SearchSpec by
running each dimension through the appropriate resolver.  Dimensions that cannot
be resolved unambiguously produce DisambiguationRequest objects that the caller
must surface to the user before retrying.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ember_agents.search.interpret import RawSignals

try:
    from ember_data import SearchSpec, ResolvedTerm

    try:
        from ember_data import DisambiguationRequest as _EmberDisambiguationRequest
    except ImportError:
        _EmberDisambiguationRequest = None  # type: ignore[assignment,misc]

    try:
        from ember_data import ModalityClassification as _ModalityClassification
    except ImportError:
        _ModalityClassification = None  # type: ignore[assignment,misc]

except ImportError:  # pragma: no cover — ember-data not installed in this env

    @dataclass  # type: ignore[no-redef]
    class ResolvedTerm:  # type: ignore[no-redef]
        """Fallback stub for ember_data.ResolvedTerm."""

        taxonomy: str = ""
        identifier: str = ""
        label: str = ""
        confidence: float = 0.0

    @dataclass  # type: ignore[no-redef]
    class SearchSpec:  # type: ignore[no-redef]
        """Fallback stub when ember-data is not installed.

        Fields mirror the real ember_data.SearchSpec to ensure the classify
        phase populates only fields that exist on the actual model.
        """

        target: object = None
        therapeutic_area: object = None
        drug_names: list = field(default_factory=list)
        indications: list = field(default_factory=list)
        modality: object = None
        resolved_terms: list = field(default_factory=list)
        pending_disambiguations: list = field(default_factory=list)

    _EmberDisambiguationRequest = None  # type: ignore[assignment,misc]
    _ModalityClassification = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISAMBIGUATION_THRESHOLD = 0.8
AMBIGUITY_MARGIN = 0.1


# ---------------------------------------------------------------------------
# Resolver Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ResolverCandidate(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def label(self) -> str: ...

    @property
    def confidence(self) -> float: ...


@runtime_checkable
class ResolverResult(Protocol):
    @property
    def candidates(self) -> list[ResolverCandidate]: ...


class Resolver(Protocol):
    async def resolve(self, term: str) -> ResolverResult: ...


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DisambiguationRequest:
    """A request for the user to disambiguate a term in a specific dimension.

    Attributes:
        dimension: The signal dimension (e.g. "target", "modality", "indication").
        raw_term: The original term extracted from the natural-language query.
        question: A closed-option question pre-formatted with numbered options.
        options: Ordered list of (id, label) tuples matching the numbered options.
    """

    dimension: str
    raw_term: str
    question: str
    options: list[tuple[str, str]]


@dataclass
class ClassificationResult:
    """Result of the classify phase.

    Attributes:
        spec: Populated SearchSpec with resolved canonical IDs.
        disambiguations: Any terms that could not be resolved automatically and
            require user input before the search can proceed.
    """

    spec: SearchSpec
    disambiguations: list[DisambiguationRequest] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_question(prefix: str, candidates: list[ResolverCandidate]) -> str:
    """Build a closed-option question string from a list of candidates."""
    lines = [prefix]
    for i, candidate in enumerate(candidates, start=1):
        lines.append(f"{i}. {candidate.id} — {candidate.label}")
    return "\n".join(lines)


def _needs_disambiguation(candidates: list[ResolverCandidate]) -> bool:
    """Return True when candidates cannot be resolved to a single clear winner.

    Disambiguation is needed when:
    - The top candidate's confidence is below DISAMBIGUATION_THRESHOLD, OR
    - Any other candidate's confidence is within AMBIGUITY_MARGIN of the top.
    """
    if not candidates:
        return False

    top = candidates[0]

    if top.confidence < DISAMBIGUATION_THRESHOLD:
        return True

    for other in candidates[1:]:
        if top.confidence - other.confidence < AMBIGUITY_MARGIN:
            return True

    return False


def _to_resolved_term(taxonomy: str, candidate: ResolverCandidate) -> ResolvedTerm:
    """Convert a resolver candidate into a ResolvedTerm."""
    return ResolvedTerm(
        taxonomy=taxonomy,
        identifier=candidate.id,
        label=candidate.label,
        confidence=candidate.confidence,
    )


# ---------------------------------------------------------------------------
# ClassificationOrchestrator
# ---------------------------------------------------------------------------


class ClassificationOrchestrator:
    """Orchestrates resolver calls to classify RawSignals into a SearchSpec.

    Each signal dimension is routed to the appropriate resolver.  Resolved
    terms are written into SearchSpec fields:

    - ``target`` signals → resolved via uniprot_resolver → ``SearchSpec.target``
      (first clear winner as ResolvedTerm; all go into ``resolved_terms``)
    - ``indication`` signals → resolved via mesh_resolver → ``SearchSpec.indications``
    - ``indication`` signals → resolved via atc_resolver → ``SearchSpec.therapeutic_area``
      (first clear winner as ResolvedTerm; all go into ``resolved_terms``)
    - ``modality`` signals → resolved via modality_resolver → ``SearchSpec.resolved_terms``
      only; ``SearchSpec.modality`` is set only if the identifier is a valid
      ``ModalityClassification`` value.

    Terms that cannot be resolved unambiguously produce
    :class:`DisambiguationRequest` objects for the caller to surface to the user.
    """

    def __init__(
        self,
        *,
        uniprot_resolver: Resolver,
        modality_resolver: Resolver,
        mesh_resolver: Resolver,
        atc_resolver: Resolver,
    ) -> None:
        self._uniprot_resolver = uniprot_resolver
        self._modality_resolver = modality_resolver
        self._mesh_resolver = mesh_resolver
        self._atc_resolver = atc_resolver

    async def classify(self, signals: RawSignals) -> ClassificationResult:
        """Classify raw signals into a SearchSpec with optional disambiguation.

        Args:
            signals: RawSignals produced by the Interpret phase.

        Returns:
            ClassificationResult containing a SearchSpec and zero or more
            DisambiguationRequest objects for terms that need user input.
        """
        all_resolved: list[ResolvedTerm] = []
        disambiguations: list[DisambiguationRequest] = []
        pending_spec: list = []

        target_term: ResolvedTerm | None = None
        indications: list[ResolvedTerm] = []
        therapeutic_area_term: ResolvedTerm | None = None
        modality_value = None

        # --- target (uniprot) ---
        for term in signals.target:
            result = await self._uniprot_resolver.resolve(term)
            candidates = result.candidates
            if not candidates:
                continue
            if _needs_disambiguation(candidates):
                options_rt = [_to_resolved_term("target", c) for c in candidates]
                question = _build_question("Which biological target did you mean?", candidates)
                disambiguations.append(
                    DisambiguationRequest(
                        dimension="target",
                        raw_term=term,
                        question=question,
                        options=[(c.id, c.label) for c in candidates],
                    )
                )
                if _EmberDisambiguationRequest is not None:
                    pending_spec.append(
                        _EmberDisambiguationRequest(
                            raw_input=term, options=options_rt, rationale=question
                        )
                    )
            else:
                rt = _to_resolved_term("target", candidates[0])
                all_resolved.append(rt)
                if target_term is None:
                    target_term = rt

        # --- indication (mesh) ---
        for term in signals.indication:
            result = await self._mesh_resolver.resolve(term)
            candidates = result.candidates
            if not candidates:
                continue
            if _needs_disambiguation(candidates):
                options_rt = [_to_resolved_term("indication", c) for c in candidates]
                question = _build_question(
                    "Which indication (disease) did you mean?", candidates
                )
                disambiguations.append(
                    DisambiguationRequest(
                        dimension="indication",
                        raw_term=term,
                        question=question,
                        options=[(c.id, c.label) for c in candidates],
                    )
                )
                if _EmberDisambiguationRequest is not None:
                    pending_spec.append(
                        _EmberDisambiguationRequest(
                            raw_input=term, options=options_rt, rationale=question
                        )
                    )
            else:
                rt = _to_resolved_term("indication", candidates[0])
                all_resolved.append(rt)
                indications.append(rt)

        # --- indication (atc) ---
        for term in signals.indication:
            result = await self._atc_resolver.resolve(term)
            candidates = result.candidates
            if not candidates:
                continue
            if _needs_disambiguation(candidates):
                options_rt = [_to_resolved_term("indication_atc", c) for c in candidates]
                question = _build_question("Which ATC classification did you mean?", candidates)
                disambiguations.append(
                    DisambiguationRequest(
                        dimension="indication_atc",
                        raw_term=term,
                        question=question,
                        options=[(c.id, c.label) for c in candidates],
                    )
                )
                if _EmberDisambiguationRequest is not None:
                    pending_spec.append(
                        _EmberDisambiguationRequest(
                            raw_input=term, options=options_rt, rationale=question
                        )
                    )
            else:
                rt = _to_resolved_term("indication_atc", candidates[0])
                all_resolved.append(rt)
                if therapeutic_area_term is None:
                    therapeutic_area_term = rt

        # --- modality ---
        for term in signals.modality:
            result = await self._modality_resolver.resolve(term)
            candidates = result.candidates
            if not candidates:
                continue
            if _needs_disambiguation(candidates):
                options_rt = [_to_resolved_term("modality", c) for c in candidates]
                question = _build_question("Which drug modality did you mean?", candidates)
                disambiguations.append(
                    DisambiguationRequest(
                        dimension="modality",
                        raw_term=term,
                        question=question,
                        options=[(c.id, c.label) for c in candidates],
                    )
                )
                if _EmberDisambiguationRequest is not None:
                    pending_spec.append(
                        _EmberDisambiguationRequest(
                            raw_input=term, options=options_rt, rationale=question
                        )
                    )
            else:
                rt = _to_resolved_term("modality", candidates[0])
                all_resolved.append(rt)
                if _ModalityClassification is not None and modality_value is None:
                    try:
                        modality_value = _ModalityClassification(candidates[0].id)
                    except (ValueError, KeyError):
                        pass

        spec = SearchSpec(
            target=target_term,
            therapeutic_area=therapeutic_area_term,
            drug_names=[],
            indications=indications,
            modality=modality_value,
            resolved_terms=all_resolved,
            pending_disambiguations=pending_spec,
        )

        return ClassificationResult(spec=spec, disambiguations=disambiguations)
