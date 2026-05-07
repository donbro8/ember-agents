"""Match phase: semantic embedding, retrieval, and 3-signal scoring.

Takes the list of Candidate objects produced by the Fetch phase and ranks
them against the original query and resolved SearchSpec using three signals:

- **Semantic** (weight 0.4): Cosine similarity from ChromaDB embedding lookup.
- **Structured** (weight 0.4): Overlap between candidate fields and SearchSpec
  resolved dimensions (target, indications, drug_names, therapeutic_area).
- **Evidence** (weight 0.2): Depth of supporting evidence (trials, patents,
  articles) attached to each candidate.

A session-scoped in-memory ChromaDB collection is created for each call to
:meth:`MatchScorer.score`; it is never persisted to disk.

ATC synonyms and UniProt aliases are used during structured scoring to bridge
alternative name forms, so that e.g. ``"HER2"`` matches ``"ERBB2"`` entries.

This is phase-5 of the dynamic search pipeline.
"""

from __future__ import annotations

import hashlib
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# ember-data imports — graceful degradation when package unavailable
# ---------------------------------------------------------------------------

try:
    from ember_data.classification.spec import ResolvedTerm, SearchSpec
    from ember_data.models.candidate import Candidate, CandidateScores

    try:
        from ember_data.classification.atc_resolver import ATCResolver
    except ImportError:
        ATCResolver = None  # type: ignore[assignment,misc]

    try:
        from ember_data.classification.uniprot_resolver import UniProtResolver
    except ImportError:
        UniProtResolver = None  # type: ignore[assignment,misc]

except ImportError:  # pragma: no cover — ember-data not installed in this env

    @dataclass  # type: ignore[no-redef]
    class ResolvedTerm:  # type: ignore[no-redef]
        """Fallback stub for ember_data.classification.spec.ResolvedTerm."""

        taxonomy: str = ""
        identifier: str = ""
        label: str = ""
        confidence: float = 0.0
        synonyms: list = field(default_factory=list)
        parent_id: object = None

    @dataclass  # type: ignore[no-redef]
    class SearchSpec:  # type: ignore[no-redef]
        """Fallback stub when ember-data is not installed."""

        query: str = ""
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
        domains: list = field(default_factory=lambda: ["trials", "patents", "articles", "candidates"])

    @dataclass  # type: ignore[no-redef]
    class CandidateScores:  # type: ignore[no-redef]
        """Fallback stub for ember_data.models.candidate.CandidateScores."""

        clinical_stage: float = 0.0
        ip_freedom: float = 0.0
        evidence_strength: float = 0.0
        novelty: float = 0.0
        overall: float = 0.0
        semantic_score: object = None
        structured_score: object = None
        evidence_score: object = None

    @dataclass  # type: ignore[no-redef]
    class Candidate:  # type: ignore[no-redef]
        """Fallback stub for ember_data.models.candidate.Candidate."""

        id: str = ""
        target: object = None
        drug_name: str | None = None
        trials: list = field(default_factory=list)
        patents: list = field(default_factory=list)
        articles: list = field(default_factory=list)
        scores: object = None
        risk_flags: list = field(default_factory=list)
        confidence: float = 0.0
        matched_dimensions: list = field(default_factory=list)
        contributing_sources: list = field(default_factory=list)
        retrieved_at: object = None
        synthesis_summary: str | None = None

    ATCResolver = None  # type: ignore[assignment,misc]
    UniProtResolver = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# ChromaDB — graceful degradation when package unavailable
# ---------------------------------------------------------------------------

try:
    import chromadb

    _CHROMA_AVAILABLE = True
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    _CHROMA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

_WEIGHT_SEMANTIC: float = 0.4
_WEIGHT_STRUCTURED: float = 0.4
_WEIGHT_EVIDENCE: float = 0.2

# Maximum evidence items considered for full evidence score normalisation
_MAX_TRIALS: int = 10
_MAX_PATENTS: int = 10
_MAX_ARTICLES: int = 10


# ---------------------------------------------------------------------------
# ScoredCandidate — annotated output type
# ---------------------------------------------------------------------------


@dataclass
class ScoredCandidate:
    """A candidate annotated with match-phase scoring signals.

    Attributes:
        candidate: The original Candidate object from the Fetch phase.
        semantic_score: Cosine-similarity-derived score in [0, 1].
        structured_score: Dimension-overlap score in [0, 1].
        evidence_score: Evidence-depth score in [0, 1].
        overall_score: Weighted composite of the three signals.
        rank: 1-based rank after sorting (populated by :meth:`MatchScorer.score`).
    """

    candidate: Any
    semantic_score: float = 0.0
    structured_score: float = 0.0
    evidence_score: float = 0.0
    overall_score: float = 0.0
    rank: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(text: str | None) -> str:
    """Normalise text to lowercase NFC for comparison."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text).lower().strip()


def _collection_name(search_id: str) -> str:
    """Derive a valid ChromaDB collection name from a search session ID."""
    # ChromaDB collection names must be 3–63 chars, alphanumeric + hyphens.
    digest = hashlib.sha1(search_id.encode()).hexdigest()[:16]
    return f"ember-match-{digest}"


def _build_candidate_document(candidate: Any) -> str:
    """Build a free-text document string from a Candidate for embedding.

    Concatenates drug name, target identifiers, indication labels, patent
    status, trial phases, and article abstracts into a single string that
    captures all semantically relevant information for the embedding model.
    """
    parts: list[str] = []

    # Drug name
    drug_name = getattr(candidate, "drug_name", None)
    if drug_name:
        parts.append(f"drug: {drug_name}")

    # Target
    target = getattr(candidate, "target", None)
    if target is not None:
        for attr in ("label", "name", "identifier", "value", "gene_name"):
            val = getattr(target, attr, None)
            if val:
                parts.append(f"target: {val}")
                break
        # Also include target synonyms / aliases
        aliases = getattr(target, "aliases", None) or getattr(target, "synonyms", None)
        if aliases:
            for alias in list(aliases)[:5]:
                if alias:
                    parts.append(f"alias: {alias}")

    # Indications from matched_dimensions
    matched_dims = getattr(candidate, "matched_dimensions", []) or []
    if matched_dims:
        parts.append(f"dimensions: {' '.join(matched_dims)}")

    # Trials — include phase and status
    for trial in list(getattr(candidate, "trials", []) or [])[:_MAX_TRIALS]:
        phase = getattr(trial, "phase", None)
        status = getattr(trial, "status", None)
        condition = getattr(trial, "condition", None) or getattr(trial, "conditions", None)
        if condition and isinstance(condition, list):
            condition = " ".join(str(c) for c in condition[:3])
        bits = [str(x) for x in [phase, status, condition] if x]
        if bits:
            parts.append(f"trial: {' '.join(bits)}")

    # Patents — include title and abstract snippet
    for patent in list(getattr(candidate, "patents", []) or [])[:_MAX_PATENTS]:
        title = getattr(patent, "title", None)
        abstract = getattr(patent, "abstract", None)
        if title:
            parts.append(f"patent title: {title}")
        if abstract:
            parts.append(f"patent abstract: {str(abstract)[:300]}")

    # Articles — include title and abstract
    for article in list(getattr(candidate, "articles", []) or [])[:_MAX_ARTICLES]:
        title = getattr(article, "title", None)
        abstract = getattr(article, "abstract", None)
        if title:
            parts.append(f"article title: {title}")
        if abstract:
            parts.append(f"article abstract: {str(abstract)[:300]}")

    return "\n".join(parts) if parts else (drug_name or getattr(candidate, "id", "unknown"))


def _build_query_document(query_text: str, spec: Any) -> str:
    """Build a query document string combining free text and resolved SearchSpec fields.

    The resulting string is embedded as the query vector against which all
    candidate documents are compared.
    """
    parts: list[str] = []

    if query_text:
        parts.append(query_text)

    # Target
    target = getattr(spec, "target", None)
    if target is not None:
        for attr in ("label", "identifier", "value"):
            val = getattr(target, attr, None)
            if val:
                parts.append(f"target: {val}")
                break

    # Therapeutic area
    ta = getattr(spec, "therapeutic_area", None)
    if ta is not None:
        for attr in ("label", "identifier", "value"):
            val = getattr(ta, attr, None)
            if val:
                parts.append(f"therapeutic area: {val}")
                break

    # Drug names
    for name in list(getattr(spec, "drug_names", []) or []):
        if name:
            parts.append(f"drug: {name}")

    # Indications
    for ind in list(getattr(spec, "indications", []) or []):
        for attr in ("label", "identifier", "value"):
            val = getattr(ind, attr, None)
            if val:
                parts.append(f"indication: {val}")
                break

    # Modality
    modality = getattr(spec, "modality", None)
    if modality is not None:
        for attr in ("label", "value", "name"):
            val = getattr(modality, attr, None)
            if val:
                parts.append(f"modality: {val}")
                break

    # Resolved terms
    for term in list(getattr(spec, "resolved_terms", []) or []):
        label = getattr(term, "label", None)
        if label:
            parts.append(label)

    return "\n".join(parts) if parts else query_text or ""


# ---------------------------------------------------------------------------
# Synonym tables — built once per MatchScorer instance
# ---------------------------------------------------------------------------


def _build_atc_synonym_table(atc_resolver: Any) -> dict[str, str]:
    """Extract the flat synonym→canonical mapping from an ATCResolver instance.

    Returns a dict mapping lowercase synonym text to its canonical lowercase
    drug/class name.  Falls back to empty dict when resolver is unavailable or
    the internal ``_synonyms`` attribute is absent.
    """
    if atc_resolver is None:
        return {}
    raw: dict[str, str] = getattr(atc_resolver, "_synonyms", {}) or {}
    nodes: dict[str, Any] = getattr(atc_resolver, "_nodes", {}) or {}
    result: dict[str, str] = {}
    for synonym_lower, code in raw.items():
        node = nodes.get(code, {})
        canonical = node.get("name", "").lower() if node else ""
        if canonical:
            result[synonym_lower] = canonical
            result[canonical] = canonical  # identity mapping
    return result


def _build_uniprot_alias_table() -> dict[str, str]:
    """Return the static UniProt alias map as lowercase → lowercase canonical."""
    # Import the module-level alias table if available; otherwise return empty.
    try:
        from ember_data.classification.uniprot_resolver import _ALIAS_MAP  # type: ignore[attr-defined]

        return {k.lower(): v.lower() for k, v in _ALIAS_MAP.items()}
    except ImportError:
        return {}


def _resolve_name_via_synonyms(
    name: str,
    atc_table: dict[str, str],
    uniprot_table: dict[str, str],
) -> str:
    """Return the canonical form of *name* after synonym resolution.

    Checks ATC synonyms first, then UniProt aliases.  Returns the input
    name unchanged when no mapping is found.
    """
    key = _normalize(name)
    if key in atc_table:
        return atc_table[key]
    if key in uniprot_table:
        return uniprot_table[key]
    return key


# ---------------------------------------------------------------------------
# Scoring signals
# ---------------------------------------------------------------------------


def _score_structured(
    candidate: Any,
    spec: Any,
    atc_table: dict[str, str],
    uniprot_table: dict[str, str],
) -> float:
    """Compute the structured matching score for *candidate* against *spec*.

    Counts how many SearchSpec dimensions (target, indications, drug_names,
    therapeutic_area) are present in the candidate's fields.  Each matching
    dimension contributes equally; the score is normalised to [0, 1].

    Synonym resolution uses ATC synonyms and UniProt aliases so that
    alternative name forms are treated as matches.
    """
    signals: list[float] = []

    # --- Drug name match ---
    drug_name_norm = _normalize(getattr(candidate, "drug_name", None))
    if drug_name_norm:
        canonical_drug = _resolve_name_via_synonyms(drug_name_norm, atc_table, uniprot_table)
    else:
        canonical_drug = ""

    spec_drug_names: list[str] = [
        _resolve_name_via_synonyms(_normalize(n), atc_table, uniprot_table)
        for n in (getattr(spec, "drug_names", []) or [])
        if n
    ]
    if spec_drug_names:
        hit = 1.0 if canonical_drug and canonical_drug in spec_drug_names else 0.0
        signals.append(hit)

    # --- Target match ---
    spec_target = getattr(spec, "target", None)
    if spec_target is not None:
        spec_target_labels: set[str] = set()
        for attr in ("label", "identifier", "value"):
            val = getattr(spec_target, attr, None)
            if val:
                spec_target_labels.add(_resolve_name_via_synonyms(_normalize(val), atc_table, uniprot_table))

        cand_target = getattr(candidate, "target", None)
        cand_target_hit = 0.0
        if cand_target is not None:
            for attr in ("label", "name", "identifier", "gene_name"):
                val = getattr(cand_target, attr, None)
                if val:
                    norm = _resolve_name_via_synonyms(_normalize(val), atc_table, uniprot_table)
                    if norm in spec_target_labels:
                        cand_target_hit = 1.0
                        break
            # Also check aliases / synonyms on the candidate target
            if cand_target_hit == 0.0:
                aliases = getattr(cand_target, "aliases", None) or getattr(cand_target, "synonyms", None)
                if aliases:
                    for alias in list(aliases):
                        norm = _resolve_name_via_synonyms(_normalize(alias), atc_table, uniprot_table)
                        if norm in spec_target_labels:
                            cand_target_hit = 1.0
                            break
        signals.append(cand_target_hit)

    # --- Indication match ---
    spec_indications: list[str] = []
    for ind in list(getattr(spec, "indications", []) or []):
        for attr in ("label", "identifier", "value"):
            val = getattr(ind, attr, None)
            if val:
                spec_indications.append(_normalize(val))
                break

    if spec_indications:
        cand_dims = {_normalize(d) for d in (getattr(candidate, "matched_dimensions", []) or [])}
        ind_hit = 0.0
        for ind_label in spec_indications:
            if ind_label in cand_dims or any(ind_label in dim for dim in cand_dims):
                ind_hit = 1.0
                break
        # Also check trial condition fields
        if ind_hit == 0.0:
            for trial in list(getattr(candidate, "trials", []) or []):
                cond = getattr(trial, "condition", None) or getattr(trial, "conditions", None)
                if cond:
                    cond_str = " ".join(str(c) for c in cond) if isinstance(cond, list) else str(cond)
                    cond_norm = _normalize(cond_str)
                    for ind_label in spec_indications:
                        if ind_label in cond_norm:
                            ind_hit = 1.0
                            break
                if ind_hit == 1.0:
                    break
        signals.append(ind_hit)

    # --- Therapeutic area match ---
    spec_ta = getattr(spec, "therapeutic_area", None)
    if spec_ta is not None:
        spec_ta_labels: set[str] = set()
        for attr in ("label", "identifier", "value"):
            val = getattr(spec_ta, attr, None)
            if val:
                spec_ta_labels.add(_resolve_name_via_synonyms(_normalize(val), atc_table, uniprot_table))

        ta_hit = 0.0
        if canonical_drug and spec_ta_labels:
            for ta_label in spec_ta_labels:
                if ta_label in canonical_drug or canonical_drug in ta_label:
                    ta_hit = 1.0
                    break
        signals.append(ta_hit)

    if not signals:
        return 0.0

    return sum(signals) / len(signals)


def _score_evidence(candidate: Any) -> float:
    """Compute an evidence-depth score based on the number of attached data objects.

    Scales each evidence type against a reasonable maximum and combines them
    with equal weights.  The result is clamped to [0, 1].
    """
    n_trials = min(len(list(getattr(candidate, "trials", []) or [])), _MAX_TRIALS)
    n_patents = min(len(list(getattr(candidate, "patents", []) or [])), _MAX_PATENTS)
    n_articles = min(len(list(getattr(candidate, "articles", []) or [])), _MAX_ARTICLES)

    trials_score = n_trials / _MAX_TRIALS
    patents_score = n_patents / _MAX_PATENTS
    articles_score = n_articles / _MAX_ARTICLES

    return (trials_score + patents_score + articles_score) / 3.0


def _compute_overall(semantic: float, structured: float, evidence: float) -> float:
    """Weighted combination of the three scoring signals."""
    return (
        _WEIGHT_SEMANTIC * semantic
        + _WEIGHT_STRUCTURED * structured
        + _WEIGHT_EVIDENCE * evidence
    )


# ---------------------------------------------------------------------------
# ChromaDB session collection
# ---------------------------------------------------------------------------


def _create_ephemeral_collection(session_id: str) -> Any:
    """Create a new in-memory ChromaDB collection for this search session.

    Returns the collection object or None when ChromaDB is not available.
    The collection uses ChromaDB's default embedding function.
    """
    if not _CHROMA_AVAILABLE or chromadb is None:
        return None

    try:
        client = chromadb.EphemeralClient()
        name = _collection_name(session_id)
        collection = client.create_collection(name=name)
        return collection
    except Exception:  # noqa: BLE001
        return None


def _embed_candidates(collection: Any, candidates: list[Any]) -> bool:
    """Embed all candidate documents into *collection*.

    Returns True when at least one document was successfully added.
    Documents without any meaningful text are skipped.
    """
    if collection is None:
        return False

    documents: list[str] = []
    ids: list[str] = []

    for candidate in candidates:
        doc = _build_candidate_document(candidate)
        if not doc:
            continue
        cand_id = str(getattr(candidate, "id", None) or uuid.uuid4())
        documents.append(doc)
        ids.append(cand_id)

    if not documents:
        return False

    try:
        collection.add(documents=documents, ids=ids)
        return True
    except Exception:  # noqa: BLE001
        return False


def _query_collection(
    collection: Any,
    query_text: str,
    n_results: int,
) -> dict[str, list[Any]]:
    """Query *collection* with *query_text* and return the raw ChromaDB result dict.

    Returns a dict with empty lists when the query fails or ChromaDB is unavailable.
    """
    if collection is None or not query_text:
        return {"ids": [[]], "distances": [[]], "documents": [[]]}

    try:
        result = collection.query(
            query_texts=[query_text],
            n_results=min(n_results, collection.count()),
        )
        return result  # type: ignore[return-value]
    except Exception:  # noqa: BLE001
        return {"ids": [[]], "distances": [[]], "documents": [[]]}


# ---------------------------------------------------------------------------
# MatchScorer
# ---------------------------------------------------------------------------


class MatchScorer:
    """Scores and ranks Candidates against a query using a 3-signal model.

    Signal breakdown
    ----------------
    - **Semantic** (0.4): ChromaDB cosine similarity between an embedded
      query document and each candidate document.  When ChromaDB is
      unavailable, this signal defaults to 0 for all candidates.
    - **Structured** (0.4): Overlap between candidate fields and the
      resolved SearchSpec dimensions (target, drug names, indications,
      therapeutic area).  ATC synonyms and UniProt aliases are applied
      during comparison so alternative name forms are treated as matches.
    - **Evidence** (0.2): Depth of supporting evidence (number of attached
      Trial, Patent, and Article objects), normalised to [0, 1].

    The three signals are combined into a weighted ``overall_score`` that
    drives final ranking.  Scores are also written back to the candidate's
    ``scores`` object (``semantic_score``, ``structured_score``,
    ``evidence_score``).

    Parameters
    ----------
    atc_resolver:
        Optional pre-built :class:`~ember_data.classification.atc_resolver.ATCResolver`
        instance.  When *None*, an attempt is made to instantiate one
        automatically.  Falls back gracefully when ember-data is unavailable.
    uniprot_resolver:
        Optional pre-built :class:`~ember_data.classification.uniprot_resolver.UniProtResolver`
        instance.  When *None*, an attempt is made to instantiate one
        automatically.  Falls back gracefully when ember-data is unavailable.
    """

    def __init__(
        self,
        *,
        atc_resolver: Any = None,
        uniprot_resolver: Any = None,
    ) -> None:
        # Build synonym tables from resolvers
        if atc_resolver is not None:
            self._atc = atc_resolver
        elif ATCResolver is not None:
            try:
                self._atc: Any = ATCResolver()
            except Exception:  # noqa: BLE001
                self._atc = None
        else:
            self._atc = None

        if uniprot_resolver is not None:
            self._uniprot = uniprot_resolver
        else:
            # UniProtResolver makes network calls; do not auto-instantiate
            # to avoid unexpected latency — caller must supply explicitly.
            self._uniprot = None

        self._atc_table: dict[str, str] = _build_atc_synonym_table(self._atc)
        self._uniprot_table: dict[str, str] = _build_uniprot_alias_table()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score(
        self,
        candidates: list[Any],
        spec: Any,
        *,
        query_text: str = "",
    ) -> list[ScoredCandidate]:
        """Score and rank *candidates* against *spec* and *query_text*.

        Parameters
        ----------
        candidates:
            List of :class:`~ember_data.models.candidate.Candidate` objects
            returned by the Fetch phase.
        spec:
            The resolved :class:`~ember_data.classification.spec.SearchSpec`
            from the Classify phase.
        query_text:
            The user's original free-text query string.  Combined with
            resolved SearchSpec fields to build the embedding query vector.

        Returns
        -------
        List of :class:`ScoredCandidate` objects sorted by ``overall_score``
        descending, with 1-based ``rank`` assigned.
        """
        if not candidates:
            return []

        # Resolve query text from spec if not provided
        if not query_text:
            query_text = getattr(spec, "query", "") or ""

        # Build combined query document
        query_doc = _build_query_document(query_text, spec)

        # Session ID — stable per call, not persisted
        session_id = str(uuid.uuid4())

        # --- Semantic signal via ChromaDB ---
        semantic_map: dict[str, float] = {}
        collection = _create_ephemeral_collection(session_id)

        if collection is not None and _embed_candidates(collection, candidates):
            n_results = max(1, collection.count())
            chroma_result = _query_collection(collection, query_doc, n_results)

            ids_list: list[str] = (chroma_result.get("ids") or [[]])[0]
            distances_list: list[float] = (chroma_result.get("distances") or [[]])[0]

            for cand_id, distance in zip(ids_list, distances_list):
                # ChromaDB L2 distance → similarity in [0, 1]
                # distance=0 is perfect match; we cap at 2 (max L2 for unit vecs)
                similarity = max(0.0, 1.0 - float(distance) / 2.0)
                semantic_map[str(cand_id)] = round(similarity, 6)

        # --- Build scored candidates ---
        scored: list[ScoredCandidate] = []
        for candidate in candidates:
            cand_id = str(getattr(candidate, "id", None) or "")

            semantic = semantic_map.get(cand_id, 0.0)
            structured = _score_structured(
                candidate, spec, self._atc_table, self._uniprot_table
            )
            evidence = _score_evidence(candidate)
            overall = _compute_overall(semantic, structured, evidence)

            # Write scores back to candidate's scores object
            self._write_scores(candidate, semantic, structured, evidence)

            scored.append(
                ScoredCandidate(
                    candidate=candidate,
                    semantic_score=round(semantic, 6),
                    structured_score=round(structured, 6),
                    evidence_score=round(evidence, 6),
                    overall_score=round(overall, 6),
                )
            )

        # Sort by overall score descending, then by candidate id for determinism
        scored.sort(key=lambda sc: (-sc.overall_score, str(getattr(sc.candidate, "id", ""))))

        # Assign 1-based ranks
        for i, sc in enumerate(scored, start=1):
            sc.rank = i

        return scored

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_scores(
        self,
        candidate: Any,
        semantic: float,
        structured: float,
        evidence: float,
    ) -> None:
        """Write match-phase scores into the candidate's scores object.

        Handles both Pydantic CandidateScores (immutable fields require
        object replacement) and fallback dataclass stubs (direct attribute
        assignment).  Silently no-ops when the scores attribute is absent
        or incompatible.
        """
        scores = getattr(candidate, "scores", None)
        if scores is None:
            return

        # Attempt Pydantic model_copy with overrides
        try:
            updated = scores.model_copy(
                update={
                    "semantic_score": semantic,
                    "structured_score": structured,
                    "evidence_score": evidence,
                }
            )
            candidate.scores = updated
            return
        except (AttributeError, TypeError, ValueError):
            pass

        # Fallback: direct attribute assignment (dataclass stubs)
        try:
            scores.semantic_score = semantic
            scores.structured_score = structured
            scores.evidence_score = evidence
        except (AttributeError, TypeError):
            pass
