"""Fetch phase: parallel API dispatch and result merging.

Translates a populated SearchSpec into API calls across BigQuery FDA labels,
BigQuery patents, ClinicalTrials.gov, UniProt, and PubMed.  Independent sources
are executed concurrently while preserving BigQuery-first cost ordering.  Results
are merged into dynamic Candidate objects, deduplicated, and tagged with
source provenance.

This is phase-4 of the dynamic search pipeline; result ranking and scoring
happens in the Match phase (TASK-025).
"""

from __future__ import annotations

import asyncio
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

try:
    from ember_data.classification.spec import SearchSpec
    from ember_data.models.article import Article
    from ember_data.models.candidate import Candidate, CandidateScores
    from ember_data.models.patent import Patent
    from ember_data.models.provenance import SourceProvenance
    from ember_data.models.target import Target
    from ember_data.models.trial import Trial

    try:
        from ember_data.clients.clinicaltrials import ClinicalTrialsClient
    except ImportError:
        ClinicalTrialsClient = None  # type: ignore[assignment,misc]

    try:
        from ember_data.clients.pubmed import PubMedClient
    except ImportError:
        PubMedClient = None  # type: ignore[assignment,misc]

    try:
        from ember_data.clients.uniprot import UniProtClient
    except ImportError:
        UniProtClient = None  # type: ignore[assignment,misc]

    try:
        from ember_data.bigquery.client import BigQueryClient
    except ImportError:
        BigQueryClient = None  # type: ignore[assignment,misc]

except ImportError:  # pragma: no cover — ember-data not installed in this env

    @dataclass  # type: ignore[no-redef]
    class SearchSpec:  # type: ignore[no-redef]
        """Fallback stub when ember-data is not installed."""

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
    class SourceProvenance:  # type: ignore[no-redef]
        """Fallback stub for ember_data.models.provenance.SourceProvenance."""

        source_name: str = ""
        source_id: str | None = None
        source_url: str | None = None
        retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))
        matched_fields: list = field(default_factory=list)

    @dataclass  # type: ignore[no-redef]
    class CandidateScores:  # type: ignore[no-redef]
        """Fallback stub for ember_data.models.candidate.CandidateScores."""

        clinical_stage: float = 0.0
        ip_freedom: float = 0.0
        evidence_strength: float = 0.0
        novelty: float = 0.0
        overall: float = 0.0

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
        retrieved_at: datetime | None = None
        synthesis_summary: str | None = None

    Target = None  # type: ignore[assignment,misc]
    Trial = None  # type: ignore[assignment,misc]
    Patent = None  # type: ignore[assignment,misc]
    Article = None  # type: ignore[assignment,misc]

    ClinicalTrialsClient = None  # type: ignore[assignment,misc]
    PubMedClient = None  # type: ignore[assignment,misc]
    UniProtClient = None  # type: ignore[assignment,misc]
    BigQueryClient = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NULL_SCORES_KWARGS: dict[str, Any] = {
    "clinical_stage": 0.0,
    "ip_freedom": 0.0,
    "evidence_strength": 0.0,
    "novelty": 0.0,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_name(name: str | None) -> str:
    """Normalise a drug name or synonym for deduplication.

    Applies NFC normalisation, lowercases, and strips whitespace so that
    trivial differences in encoding or casing do not create duplicate
    candidates.
    """
    if not name:
        return ""
    normalised = unicodedata.normalize("NFC", name).lower().strip()
    return normalised


def _candidate_id() -> str:
    """Generate a deterministic-enough unique ID for a new Candidate."""
    return str(uuid.uuid4())


def _make_scores() -> Any:
    """Build a zeroed CandidateScores instance regardless of import status."""
    if CandidateScores is not None and not isinstance(CandidateScores, type(None)):
        try:
            return CandidateScores(**_NULL_SCORES_KWARGS)
        except Exception:  # noqa: BLE001
            pass
    # Fallback dataclass stub
    return _FallbackScores()


class _FallbackScores:
    """Minimal scores placeholder used when Pydantic model is unavailable."""

    clinical_stage: float = 0.0
    ip_freedom: float = 0.0
    evidence_strength: float = 0.0
    novelty: float = 0.0
    overall: float = 0.0


# ---------------------------------------------------------------------------
# FetchResult — intermediate container returned by each source fetcher
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    """Intermediate result container for a single source's fetch.

    Attributes:
        drug_name: Primary drug or gene name identified by the source.
        synonyms: Alternative names / aliases extracted from the result.
        target_id: UniProt accession or other canonical target identifier, if any.
        target: Target object if retrieved from UniProt.
        trials: Clinical trial records from this source.
        patents: Patent records from this source.
        articles: Article records from this source.
        provenance: Source provenance for every data element from this result.
        matched_dimensions: Which SearchSpec dimensions this result matched.
    """

    drug_name: str = ""
    synonyms: list[str] = field(default_factory=list)
    target_id: str = ""
    target: object = None
    trials: list = field(default_factory=list)
    patents: list = field(default_factory=list)
    articles: list = field(default_factory=list)
    provenance: list = field(default_factory=list)
    matched_dimensions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Deduplication registry
# ---------------------------------------------------------------------------


class _DedupeRegistry:
    """Tracks seen keys to detect duplicate Candidate entries.

    A candidate is considered a duplicate if any of its drug names, synonyms,
    or target identifiers have already been recorded.
    """

    def __init__(self) -> None:
        self._seen_names: set[str] = set()
        self._seen_target_ids: set[str] = set()

    def is_duplicate(self, result: FetchResult) -> bool:
        """Return True if *result* duplicates an already-seen candidate."""
        norm_name = _normalize_name(result.drug_name)
        if norm_name and norm_name in self._seen_names:
            return True

        for syn in result.synonyms:
            norm_syn = _normalize_name(syn)
            if norm_syn and norm_syn in self._seen_names:
                return True

        if result.target_id and result.target_id in self._seen_target_ids:
            return True

        return False

    def register(self, result: FetchResult) -> None:
        """Record the identifiers in *result* so future duplicates are detected."""
        norm_name = _normalize_name(result.drug_name)
        if norm_name:
            self._seen_names.add(norm_name)

        for syn in result.synonyms:
            norm_syn = _normalize_name(syn)
            if norm_syn:
                self._seen_names.add(norm_syn)

        if result.target_id:
            self._seen_target_ids.add(result.target_id)


# ---------------------------------------------------------------------------
# FetchOrchestrator
# ---------------------------------------------------------------------------


class FetchOrchestrator:
    """Orchestrates parallel data fetching across all configured sources.

    Execution order
    ---------------
    BigQuery sources (FDA labels, patents) are dispatched first to preserve
    cost-visibility ordering — BigQuery scan costs are proportional to data
    scanned, so callers may want to inspect partial results before waiting for
    external API calls.  External API calls (ClinicalTrials.gov, UniProt,
    PubMed) are dispatched concurrently in a second wave via ``asyncio.gather``.

    Deduplication
    -------------
    Results are deduplicated by normalised drug name, synonym, and UniProt
    target ID.  The first occurrence (BigQuery sources take priority) is kept;
    later duplicates are discarded.

    Provenance
    ----------
    Every Candidate's ``contributing_sources`` list and each data element's
    provenance record the source name, source ID/URL, and ``retrieved_at``
    timestamp.
    """

    def __init__(
        self,
        *,
        bigquery_client: Any = None,
        clinicaltrials_client: Any = None,
        uniprot_client: Any = None,
        pubmed_client: Any = None,
        bq_project_id: str | None = None,
    ) -> None:
        """Initialise FetchOrchestrator with optional pre-built clients.

        Parameters
        ----------
        bigquery_client:
            Optional pre-built BigQueryClient.  When None and ``bq_project_id``
            is set, a new client is instantiated automatically.
        clinicaltrials_client:
            Optional pre-built ClinicalTrialsClient.  Instantiated automatically
            when None (if the package is available).
        uniprot_client:
            Optional pre-built UniProtClient.  Instantiated automatically when
            None (if the package is available).
        pubmed_client:
            Optional pre-built PubMedClient.  Instantiated automatically when
            None (if the package is available).
        bq_project_id:
            GCP project ID used when auto-instantiating BigQueryClient.
        """
        # BigQuery client — auto-instantiate when bq_project_id is provided
        if bigquery_client is not None:
            self._bq = bigquery_client
        elif bq_project_id and BigQueryClient is not None:
            try:
                self._bq = BigQueryClient(bq_project_id)
            except Exception:  # noqa: BLE001
                self._bq = None
        else:
            self._bq = None

        # External API clients — auto-instantiate when available
        if clinicaltrials_client is not None:
            self._ct = clinicaltrials_client
        elif ClinicalTrialsClient is not None:
            try:
                self._ct: Any = ClinicalTrialsClient()
            except Exception:  # noqa: BLE001
                self._ct = None
        else:
            self._ct = None

        if uniprot_client is not None:
            self._uniprot = uniprot_client
        elif UniProtClient is not None:
            try:
                self._uniprot: Any = UniProtClient()
            except Exception:  # noqa: BLE001
                self._uniprot = None
        else:
            self._uniprot = None

        if pubmed_client is not None:
            self._pubmed = pubmed_client
        elif PubMedClient is not None:
            try:
                self._pubmed: Any = PubMedClient()
            except Exception:  # noqa: BLE001
                self._pubmed = None
        else:
            self._pubmed = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(self, spec: SearchSpec) -> list[Any]:
        """Fetch candidates from all sources for the given SearchSpec.

        Parameters
        ----------
        spec:
            Populated SearchSpec produced by the Classify / Gate phases.

        Returns
        -------
        List of Candidate objects, deduplicated and tagged with provenance.
        The list is bounded by ``spec.max_results`` across all sources
        combined (applied after deduplication).
        """
        max_results: int = getattr(spec, "max_results", 500)
        per_source_limit = min(max_results, 100)

        raw_results: list[FetchResult] = []

        # --- Wave 1: BigQuery (serial to preserve cost ordering) ---
        bq_results = await self._fetch_bigquery(spec, per_source_limit)
        raw_results.extend(bq_results)

        # --- Wave 2: External APIs (concurrent) ---
        ext_tasks = [
            self._fetch_clinicaltrials(spec, per_source_limit),
            self._fetch_uniprot(spec, per_source_limit),
            self._fetch_pubmed(spec, per_source_limit),
        ]
        ext_results_lists = await asyncio.gather(*ext_tasks, return_exceptions=True)
        for result in ext_results_lists:
            if isinstance(result, list):
                raw_results.extend(result)
            # Exceptions from individual sources are silently dropped so that
            # partial results are still returned.

        # --- Deduplicate and merge ---
        candidates = self._merge(raw_results)

        # --- Enforce max_results ---
        return candidates[:max_results]

    # ------------------------------------------------------------------
    # Source-specific fetchers
    # ------------------------------------------------------------------

    async def _fetch_bigquery(
        self, spec: SearchSpec, limit: int
    ) -> list[FetchResult]:
        """Fetch from BigQuery FDA labels and patents (serial, BigQuery-first)."""
        results: list[FetchResult] = []

        if self._bq is None:
            return results

        loop = asyncio.get_event_loop()

        # FDA drug labels
        domains = getattr(spec, "domains", ["trials", "patents", "articles", "candidates"])
        if "candidates" in domains or "trials" in domains:
            drug_names: list[str] = list(getattr(spec, "drug_names", []) or [])
            # Also try label of resolved target
            target = getattr(spec, "target", None)
            if target is not None:
                label = getattr(target, "label", None) or getattr(target, "value", None)
                if label and label not in drug_names:
                    drug_names.append(str(label))

            if drug_names:
                ta = getattr(spec, "therapeutic_area", None)
                ta_str: str | None = None
                if ta is not None:
                    ta_str = getattr(ta, "label", None) or getattr(ta, "value", None)
                    if ta_str:
                        ta_str = str(ta_str)

                try:
                    fda_rows: list[dict] = await loop.run_in_executor(
                        None,
                        lambda: self._bq.query_fda_drug_events(
                            drug_names, ta_str, limit
                        ),
                    )
                    results.extend(self._parse_fda_rows(fda_rows))
                except Exception:  # noqa: BLE001
                    pass  # Source failure does not abort the overall fetch

        # Patents
        if "patents" in domains:
            patent_query = self._build_patent_query(spec)
            if patent_query:
                date_window = getattr(spec, "patent_expiry_window", None)
                jurisdictions_raw = getattr(spec, "jurisdictions", []) or []
                jurisdictions = [
                    getattr(j, "value", j) for j in jurisdictions_raw
                ]

                try:
                    patent_rows: list[dict] = await loop.run_in_executor(
                        None,
                        lambda: self._bq.search_patents(
                            patent_query,
                            limit,
                            date_window,
                            jurisdictions or None,
                        ),
                    )
                    results.extend(self._parse_patent_rows(patent_rows))
                except Exception:  # noqa: BLE001
                    pass

        return results

    async def _fetch_clinicaltrials(
        self, spec: SearchSpec, limit: int
    ) -> list[FetchResult]:
        """Fetch from ClinicalTrials.gov."""
        results: list[FetchResult] = []

        domains = getattr(spec, "domains", ["trials", "patents", "articles", "candidates"])
        if "trials" not in domains:
            return results

        if self._ct is None:
            return results

        condition = self._extract_condition_label(spec)
        if not condition:
            return results

        intervention: str | None = None
        drug_names: list[str] = list(getattr(spec, "drug_names", []) or [])
        if drug_names:
            intervention = drug_names[0]

        approval_window = getattr(spec, "approval_date_range", None)
        from_date = getattr(approval_window, "start", None) if approval_window else None
        to_date = getattr(approval_window, "end", None) if approval_window else None

        loop = asyncio.get_event_loop()
        try:
            ct_results: list[tuple] = await loop.run_in_executor(
                None,
                lambda: self._ct.search(
                    condition,
                    intervention=intervention,
                    from_date=from_date,
                    to_date=to_date,
                    max_results=limit,
                ),
            )
        except Exception:  # noqa: BLE001
            return results

        for trial, provenance in ct_results:
            fetch_result = FetchResult(
                drug_name=_extract_first(getattr(trial, "interventions", [])),
                synonyms=list(getattr(trial, "interventions", []))[1:],
                target_id="",
                trials=[trial],
                provenance=[provenance],
                matched_dimensions=["indication"],
            )
            results.append(fetch_result)

        return results

    async def _fetch_uniprot(
        self, spec: SearchSpec, limit: int
    ) -> list[FetchResult]:
        """Fetch target records from UniProt."""
        results: list[FetchResult] = []

        if self._uniprot is None:
            return results

        target = getattr(spec, "target", None)
        if target is None:
            return results

        query = (
            getattr(target, "identifier", None)
            or getattr(target, "label", None)
            or getattr(target, "value", None)
        )
        if not query:
            return results

        loop = asyncio.get_event_loop()
        try:
            uniprot_results: list[tuple] = await loop.run_in_executor(
                None,
                lambda: self._uniprot.search_targets(str(query), max_results=limit),
            )
        except Exception:  # noqa: BLE001
            return results

        for tgt, provenance in uniprot_results:
            aliases: list[str] = list(getattr(tgt, "aliases", []) or [])
            fetch_result = FetchResult(
                drug_name=getattr(tgt, "name", "") or "",
                synonyms=aliases,
                target_id=getattr(tgt, "uniprot_id", "") or getattr(tgt, "id", "") or "",
                target=tgt,
                provenance=[provenance],
                matched_dimensions=["target"],
            )
            results.append(fetch_result)

        return results

    async def _fetch_pubmed(
        self, spec: SearchSpec, limit: int
    ) -> list[FetchResult]:
        """Fetch article records from PubMed."""
        results: list[FetchResult] = []

        domains = getattr(spec, "domains", ["trials", "patents", "articles", "candidates"])
        if "articles" not in domains:
            return results

        if self._pubmed is None:
            return results

        mesh_term: str | None = None
        target_query: str | None = None

        # Extract MeSH term from first indication
        indications = getattr(spec, "indications", []) or []
        if indications:
            first_ind = indications[0]
            mesh_term = (
                getattr(first_ind, "identifier", None)
                or getattr(first_ind, "label", None)
                or getattr(first_ind, "value", None)
            )
            if mesh_term:
                mesh_term = str(mesh_term)

        # Extract target query
        target = getattr(spec, "target", None)
        if target is not None:
            target_query = (
                getattr(target, "label", None)
                or getattr(target, "identifier", None)
                or getattr(target, "value", None)
            )
            if target_query:
                target_query = str(target_query)

        if not mesh_term and not target_query:
            return results

        approval_window = getattr(spec, "approval_date_range", None)
        from_date = getattr(approval_window, "start", None) if approval_window else None
        to_date = getattr(approval_window, "end", None) if approval_window else None

        loop = asyncio.get_event_loop()
        try:
            pubmed_results: list[tuple] = await loop.run_in_executor(
                None,
                lambda: self._pubmed.search(
                    mesh_term=mesh_term,
                    target=target_query,
                    from_date=from_date,
                    to_date=to_date,
                    max_results=limit,
                ),
            )
        except Exception:  # noqa: BLE001
            return results

        for article, provenance in pubmed_results:
            fetch_result = FetchResult(
                drug_name="",
                synonyms=[],
                target_id="",
                articles=[article],
                provenance=[provenance],
                matched_dimensions=["indication"] if mesh_term else ["target"],
            )
            results.append(fetch_result)

        return results

    # ------------------------------------------------------------------
    # Row parsers for BigQuery results
    # ------------------------------------------------------------------

    def _parse_fda_rows(self, rows: list[dict]) -> list[FetchResult]:
        """Convert BigQuery FDA drug-label rows into FetchResult objects."""
        results: list[FetchResult] = []
        now = datetime.now(UTC)

        for row in rows:
            generic_name: str = row.get("openfda_generic_name") or ""
            brand_name: str = row.get("openfda_brand_name") or ""
            product_ndc: str = row.get("openfda_product_ndc") or ""

            # Primary name preference: generic > brand
            drug_name = generic_name or brand_name
            synonyms: list[str] = []
            if brand_name and brand_name != drug_name:
                synonyms.append(brand_name)
            if generic_name and generic_name != drug_name:
                synonyms.append(generic_name)

            provenance = SourceProvenance(
                source_name="FDA OpenFDA",
                source_id=product_ndc or None,
                source_url=(
                    f"https://api.fda.gov/drug/label.json?search=openfda.product_ndc:{product_ndc}"
                    if product_ndc
                    else None
                ),
                retrieved_at=now,
                matched_fields=["drug_name"],
            )

            results.append(
                FetchResult(
                    drug_name=drug_name,
                    synonyms=synonyms,
                    target_id="",
                    provenance=[provenance],
                    matched_dimensions=["drug_name"],
                )
            )

        return results

    def _parse_patent_rows(self, rows: list[dict]) -> list[FetchResult]:
        """Convert BigQuery patent rows into FetchResult objects."""
        results: list[FetchResult] = []
        now = datetime.now(UTC)

        for row in rows:
            pub_number: str = str(row.get("publication_number") or "")
            title_str: str = str(row.get("title") or "")
            abstract_str: str = str(row.get("abstract") or "")
            filing_date_raw = row.get("filing_date")
            grant_date_raw = row.get("grant_date")

            filing_date = _parse_int_date(filing_date_raw)
            if filing_date is None:
                continue  # Patent model requires filing_date

            grant_date = _parse_int_date(grant_date_raw)

            if Patent is not None:
                try:
                    patent = Patent(
                        publication_number=pub_number,
                        title=title_str,
                        abstract=abstract_str,
                        claims=[],
                        assignee="",
                        filing_date=filing_date,
                        grant_date=grant_date,
                    )
                except Exception:  # noqa: BLE001
                    patent = None  # type: ignore[assignment]
            else:
                patent = None  # type: ignore[assignment]

            provenance = SourceProvenance(
                source_name="Google Patents (BigQuery)",
                source_id=pub_number or None,
                source_url=(
                    f"https://patents.google.com/patent/{pub_number}"
                    if pub_number
                    else None
                ),
                retrieved_at=now,
                matched_fields=["title", "abstract"],
            )

            patents_list = [patent] if patent is not None else []
            results.append(
                FetchResult(
                    drug_name="",
                    synonyms=[],
                    target_id="",
                    patents=patents_list,
                    provenance=[provenance],
                    matched_dimensions=["patent"],
                )
            )

        return results

    # ------------------------------------------------------------------
    # Merge and deduplication
    # ------------------------------------------------------------------

    def _merge(self, raw_results: list[FetchResult]) -> list[Any]:
        """Merge FetchResult objects into deduplicated Candidate instances.

        Strategy
        --------
        - Results are grouped by normalised drug name / target ID.
        - The first occurrence wins (BigQuery results arrive first so are
          preferred as the primary entry for a given drug).
        - Subsequent results with the same key contribute additional trials,
          patents, articles, and provenance entries to the existing Candidate.
        - Results with no drug name and no target ID each form their own
          Candidate (e.g. patent-only or article-only entries).
        """
        registry = _DedupeRegistry()
        # candidate_key → Candidate accumulator dicts
        accumulators: dict[str, dict[str, Any]] = {}
        # Ordered list of keys to preserve first-seen ordering
        ordered_keys: list[str] = []

        now = datetime.now(UTC)

        for fetch_result in raw_results:
            dedup_key = self._dedup_key(fetch_result)

            if dedup_key and dedup_key in accumulators:
                # Merge into existing accumulator
                acc = accumulators[dedup_key]
                acc["trials"].extend(fetch_result.trials)
                acc["patents"].extend(fetch_result.patents)
                acc["articles"].extend(fetch_result.articles)
                acc["contributing_sources"].extend(fetch_result.provenance)
                for dim in fetch_result.matched_dimensions:
                    if dim not in acc["matched_dimensions"]:
                        acc["matched_dimensions"].append(dim)
                continue

            # Determine whether this is truly a duplicate (overlapping name/synonym)
            if registry.is_duplicate(fetch_result):
                # Find the existing accumulator that owns these identifiers
                # and merge into it — the dedup_key differs but names overlap.
                matched_acc_key = self._find_matching_key(
                    fetch_result, accumulators
                )
                if matched_acc_key:
                    acc = accumulators[matched_acc_key]
                    acc["trials"].extend(fetch_result.trials)
                    acc["patents"].extend(fetch_result.patents)
                    acc["articles"].extend(fetch_result.articles)
                    acc["contributing_sources"].extend(fetch_result.provenance)
                    for dim in fetch_result.matched_dimensions:
                        if dim not in acc["matched_dimensions"]:
                            acc["matched_dimensions"].append(dim)
                    continue

            # New candidate — register and create accumulator
            registry.register(fetch_result)
            key = dedup_key or _candidate_id()
            accumulators[key] = {
                "id": _candidate_id(),
                "drug_name": fetch_result.drug_name or None,
                "target": fetch_result.target,
                "trials": list(fetch_result.trials),
                "patents": list(fetch_result.patents),
                "articles": list(fetch_result.articles),
                "contributing_sources": list(fetch_result.provenance),
                "matched_dimensions": list(fetch_result.matched_dimensions),
                "retrieved_at": now,
            }
            ordered_keys.append(key)

        candidates: list[Any] = []
        for key in ordered_keys:
            acc = accumulators[key]
            try:
                candidate = Candidate(
                    id=acc["id"],
                    target=acc["target"],
                    drug_name=acc["drug_name"],
                    trials=acc["trials"],
                    patents=acc["patents"],
                    articles=acc["articles"],
                    scores=_make_scores(),
                    confidence=0.0,
                    matched_dimensions=acc["matched_dimensions"],
                    contributing_sources=acc["contributing_sources"],
                    retrieved_at=acc["retrieved_at"],
                )
            except Exception:  # noqa: BLE001
                # If Candidate instantiation fails (e.g. schema mismatch),
                # skip this entry rather than aborting the whole merge.
                continue
            candidates.append(candidate)

        return candidates

    def _dedup_key(self, result: FetchResult) -> str:
        """Compute a stable deduplication key for a FetchResult.

        Prefers target_id (canonical), then normalised drug_name.
        Returns empty string for results with neither.
        """
        if result.target_id:
            return f"target:{result.target_id}"
        norm = _normalize_name(result.drug_name)
        if norm:
            return f"drug:{norm}"
        return ""

    def _find_matching_key(
        self, result: FetchResult, accumulators: dict[str, dict[str, Any]]
    ) -> str | None:
        """Find the accumulator key whose drug name / synonyms overlap with result."""
        norm_name = _normalize_name(result.drug_name)
        norm_syns = {_normalize_name(s) for s in result.synonyms if s}

        for key, acc in accumulators.items():
            existing_name = _normalize_name(acc.get("drug_name") or "")
            if norm_name and existing_name and norm_name == existing_name:
                return key
            if existing_name and existing_name in norm_syns:
                return key
            if norm_name and norm_name == existing_name:
                return key
        return None

    # ------------------------------------------------------------------
    # Query / label extraction helpers
    # ------------------------------------------------------------------

    def _build_patent_query(self, spec: SearchSpec) -> str:
        """Build a free-text patent search query from SearchSpec dimensions."""
        terms: list[str] = []

        # Drug names
        for name in getattr(spec, "drug_names", []) or []:
            if name:
                terms.append(name)

        # Target label
        target = getattr(spec, "target", None)
        if target is not None:
            label = (
                getattr(target, "label", None)
                or getattr(target, "identifier", None)
                or getattr(target, "value", None)
            )
            if label:
                terms.append(str(label))

        # Indication labels
        for ind in getattr(spec, "indications", []) or []:
            label = (
                getattr(ind, "label", None)
                or getattr(ind, "identifier", None)
                or getattr(ind, "value", None)
            )
            if label:
                terms.append(str(label))

        return " ".join(terms)

    def _extract_condition_label(self, spec: SearchSpec) -> str:
        """Extract the primary condition/indication label for ClinicalTrials search."""
        indications = getattr(spec, "indications", []) or []
        if indications:
            first = indications[0]
            label = (
                getattr(first, "label", None)
                or getattr(first, "identifier", None)
                or getattr(first, "value", None)
            )
            if label:
                return str(label)

        # Fall back to therapeutic area
        ta = getattr(spec, "therapeutic_area", None)
        if ta is not None:
            label = (
                getattr(ta, "label", None)
                or getattr(ta, "value", None)
            )
            if label:
                return str(label)

        return ""


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _extract_first(items: list[str]) -> str:
    """Return the first non-empty string from *items*, or empty string."""
    for item in items:
        if item:
            return item
    return ""


def _parse_int_date(value: Any) -> Any:
    """Parse a BigQuery INT64 date in YYYYMMDD format to a date object.

    Returns None when the value is absent or cannot be parsed.
    """
    if value is None:
        return None
    try:
        from datetime import date as _date

        raw = int(value)
        year = raw // 10000
        month = (raw % 10000) // 100
        day = raw % 100
        return _date(year, month, day)
    except (ValueError, TypeError):
        return None
