"""BiologicSeedSource: FetchOrchestrator source backed by biologic_reference.json."""

from __future__ import annotations

import json
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

from ember_agents.search.fetch import FetchResult

try:
    from ember_data.seed.convert import seed_entry_to_patent_jurisdictions
    from ember_data.seed.schema import BiologicEntry

    _EMBER_DATA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _EMBER_DATA_AVAILABLE = False
    seed_entry_to_patent_jurisdictions = None  # type: ignore[assignment]
    BiologicEntry = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MIN_REVENUE: float = 1.0
_DEFAULT_PATENT_EXPIRY_CUTOFF: date = date(2028, 12, 31)

# Dimension tag used in FetchResult.matched_dimensions
_SOURCE_NAME = "biologic_seed"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _norm(text: str | None) -> str:
    """NFC-normalise, lowercase, and strip *text* for case-insensitive comparison."""
    if not text:
        return ""
    return unicodedata.normalize("NFC", text).lower().strip()


def _entry_names(entry: Any) -> list[str]:
    """Return all normalised name tokens for a BiologicEntry.

    Covers drug_name, brand_names, and target_antigen so that a spec can
    match via any of these fields.
    """
    names: list[str] = []
    if entry.drug_name:
        names.append(_norm(entry.drug_name))
    for bn in getattr(entry, "brand_names", []):
        if bn:
            names.append(_norm(bn))
    if entry.target_antigen:
        names.append(_norm(entry.target_antigen))
    return names


def _spec_query_tokens(spec: Any) -> set[str]:
    """Extract normalised search tokens from a SearchSpec (or compatible object).

    Includes drug_names, brand_names (if present), and the target field.
    """
    tokens: set[str] = set()

    for name in getattr(spec, "drug_names", []) or []:
        t = _norm(name)
        if t:
            tokens.add(t)

    for name in getattr(spec, "brand_names", []) or []:
        t = _norm(name)
        if t:
            tokens.add(t)

    target = getattr(spec, "target", None)
    if target is not None:
        # SearchSpec.target can be a TargetClassification enum, a ResolvedTerm, or a str.
        if isinstance(target, str):
            t = _norm(target)
            if t:
                tokens.add(t)
        else:
            label = getattr(target, "label", None) or getattr(target, "value", None)
            if label:
                t = _norm(str(label))
                if t:
                    tokens.add(t)
            for syn in getattr(target, "synonyms", []) or []:
                t = _norm(syn)
                if t:
                    tokens.add(t)

    return tokens


def _matches_spec(entry: Any, query_tokens: set[str]) -> bool:
    """Return True when any entry name token intersects with *query_tokens*."""
    if not query_tokens:
        return False
    entry_tokens = set(_entry_names(entry))
    return bool(entry_tokens & query_tokens)


def _apply_biosimilar_filter(
    entries: list[Any],
    *,
    min_revenue_millions: float,
    patent_expiry_cutoff: date,
    jurisdictions: list[str] | None = None,
) -> list[Any]:
    """Apply hard filters used in biosimilar_screen mode.

    Mirrors the logic of ``stage1_hard_filter`` in the biosimilar pipeline:
    - annual_revenue_usd_millions >= min_revenue_millions
    - At least one jurisdiction has a patent expiry <= patent_expiry_cutoff

    Cell-line class is not filtered here — that concern belongs to the agent.
    """
    if jurisdictions is None:
        jurisdictions = ["US", "EU"]

    result: list[Any] = []
    for entry in entries:
        revenue = entry.annual_revenue_usd_millions
        if revenue is None or revenue < min_revenue_millions:
            continue

        expiry_dates: list[date] = []
        for j in jurisdictions:
            d = entry.patent_expiries.get(j.upper())
            if d is not None:
                expiry_dates.append(d)

        if not expiry_dates:
            continue

        if not any(d <= patent_expiry_cutoff for d in expiry_dates):
            continue

        result.append(entry)

    return result


def _entry_to_fetch_result(entry: Any) -> FetchResult:
    """Convert a BiologicEntry into a FetchResult with PatentJurisdiction data."""
    patents: list[Any] = []
    if _EMBER_DATA_AVAILABLE and seed_entry_to_patent_jurisdictions is not None:
        patents = seed_entry_to_patent_jurisdictions(entry)

    brand_names: list[str] = list(getattr(entry, "brand_names", []) or [])

    return FetchResult(
        drug_name=entry.drug_name,
        synonyms=brand_names,
        target_id="",
        patents=patents,
        matched_dimensions=[_SOURCE_NAME],
        brand_names=brand_names,
        originator=getattr(entry, "originator", "") or "",
        modality=getattr(entry, "modality", "") or "",
        category=getattr(entry, "category", "") or "",
        annual_revenue_usd_millions=getattr(entry, "annual_revenue_usd_millions", None),
        revenue_year=getattr(entry, "revenue_year", None),
        biosimilar_competitors=list(getattr(entry, "biosimilar_competitors", []) or []),
        has_approved_biosimilar=getattr(entry, "has_approved_biosimilar", False) or False,
        indications=list(getattr(entry, "indications", []) or []),
    )


# ---------------------------------------------------------------------------
# BiologicSeedSource
# ---------------------------------------------------------------------------


class BiologicSeedSource:
    """FetchOrchestrator-compatible source backed by biologic_reference.json.

    Usage::

        source = BiologicSeedSource("/path/to/biologic_reference.json")
        results = await source.fetch(spec, query_type="name_lookup")

    Name matching
    -------------
    Each entry is checked against the spec's drug_names, brand_names, and
    target (label + synonyms).  If *no* query tokens are present in the spec,
    an empty list is returned for non-biosimilar queries.

    biosimilar_screen mode
    ----------------------
    When *query_type* is ``"biosimilar_screen"``, the source returns the full
    filtered dataset (all entries that pass the hard revenue + patent-expiry
    filters) rather than restricting to name-matched entries.  Filter
    parameters are read from the spec when available:

    - ``spec.min_revenue_millions`` → minimum annual revenue
    - ``spec.patent_expiry_window.end`` → patent expiry cutoff date
    - ``spec.jurisdictions`` → list of jurisdiction codes (e.g. "US", "EU")
    """

    def __init__(self, seed_path: str | Path) -> None:
        """Load and parse *seed_path* (path to biologic_reference.json).

        Args:
            seed_path: Filesystem path to the JSON seed file.

        Raises:
            FileNotFoundError: If *seed_path* does not exist.
            ValueError: If the JSON is not a list or entries fail validation.
        """
        self._seed_path = Path(seed_path)
        self._entries: list[Any] = self._load(self._seed_path)

    # ------------------------------------------------------------------
    # Internal loader
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> list[Any]:
        """Parse the seed JSON file and return a list of BiologicEntry objects."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            entries = raw.get("entries")
            if not isinstance(entries, list):
                raise ValueError(f"Expected 'entries' key with a list in {path}")
            raw = entries
        elif not isinstance(raw, list):
            raise ValueError(f"Expected a JSON array or versioned object in {path}, got {type(raw).__name__}")

        if _EMBER_DATA_AVAILABLE and BiologicEntry is not None:
            return [BiologicEntry.model_validate(item) for item in raw]

        # Fallback: return raw dicts wrapped in a simple namespace so attribute
        # access works in _entry_names / filters.
        return [_DictEntry(item) for item in raw]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fetch(self, spec: Any, query_type: str) -> list[FetchResult]:
        """Return matching FetchResult objects for *spec*.

        Args:
            spec: A SearchSpec (or duck-typed equivalent) carrying search
                  dimensions: drug_names, brand_names, target, etc.
            query_type: Query mode selector.  ``"biosimilar_screen"`` applies
                        hard filters and returns the full filtered set.  All
                        other values use name matching only.

        Returns:
            A list of FetchResult objects, each with patents populated as
            PatentJurisdiction entries.
        """
        if query_type == "biosimilar_screen":
            return self._fetch_biosimilar_screen(spec)

        return self._fetch_by_name(spec)

    # ------------------------------------------------------------------
    # Private fetch strategies
    # ------------------------------------------------------------------

    def _fetch_by_name(self, spec: Any) -> list[FetchResult]:
        """Return entries whose names match the spec's query tokens."""
        query_tokens = _spec_query_tokens(spec)
        if not query_tokens:
            return []

        results: list[FetchResult] = []
        for entry in self._entries:
            if _matches_spec(entry, query_tokens):
                results.append(_entry_to_fetch_result(entry))

        return results

    def _fetch_biosimilar_screen(self, spec: Any) -> list[FetchResult]:
        """Apply hard filters and return the full qualifying set."""
        min_revenue = (
            float(spec.min_revenue_millions)
            if getattr(spec, "min_revenue_millions", None) is not None
            else _DEFAULT_MIN_REVENUE
        )

        expiry_cutoff = _DEFAULT_PATENT_EXPIRY_CUTOFF
        patent_expiry_window = getattr(spec, "patent_expiry_window", None)
        if patent_expiry_window is not None:
            end = getattr(patent_expiry_window, "end", None)
            if end is not None:
                expiry_cutoff = end

        jurisdictions: list[str] | None = None
        raw_jurisdictions = getattr(spec, "jurisdictions", None)
        if raw_jurisdictions:
            # Jurisdiction may be an enum with a .value or a plain string.
            jurisdictions = [
                (j.value if hasattr(j, "value") else str(j)).upper()
                for j in raw_jurisdictions
            ]

        filtered = _apply_biosimilar_filter(
            self._entries,
            min_revenue_millions=min_revenue,
            patent_expiry_cutoff=expiry_cutoff,
            jurisdictions=jurisdictions,
        )

        return [_entry_to_fetch_result(entry) for entry in filtered]


# ---------------------------------------------------------------------------
# Fallback entry wrapper (used when ember_data is not installed)
# ---------------------------------------------------------------------------


class _DictEntry:
    """Attribute-access wrapper around a raw seed-entry dict."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        try:
            return self._data[name]
        except KeyError:
            return None

    @property
    def patent_expiries(self) -> dict[str, date]:
        raw = self._data.get("patent_expiries", {})
        result: dict[str, date] = {}
        for k, v in raw.items():
            if isinstance(v, date):
                result[k] = v
            elif isinstance(v, str):
                try:
                    result[k] = date.fromisoformat(v)
                except ValueError:
                    pass
        return result
