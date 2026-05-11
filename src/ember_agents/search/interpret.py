"""Interpret phase: LLM-based signal extraction from natural-language queries.

Converts a free-text search query into raw dimension-tagged signals (RawSignals).
This is the first stage of the dynamic search pipeline; classification of raw
signals into canonical IDs happens in the Classify phase (TASK-022).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

try:
    from ember_shared import settings as _settings
except ImportError:  # pragma: no cover
    _settings = None

_SYSTEM_PROMPT_TEMPLATE = """\
You are a pharmaceutical-intelligence signal extractor.
Today's date is {today}.

Your task is to parse a natural-language search query and extract structured
signals across five dimensions plus a query classification.  Return ONLY a JSON
object that matches the schema described below.  Do not include explanation or
prose outside the JSON.

## Dimensions

1. **target** – Biological targets or antigens mentioned (e.g. "PD-1", "VEGF",
   "HER2", "CD19").  Extract verbatim or lightly normalised terms.

2. **modality** – Drug format or modality (e.g. "mAb", "bispecific", "ADC",
   "small molecule", "CAR-T", "siRNA").  Extract verbatim or lightly normalised.

3. **indication** – Disease, condition, or therapeutic area (e.g. "NSCLC",
   "breast cancer", "type 2 diabetes", "haematology").  Extract verbatim or
   lightly normalised.

4. **cell_line** – Cell line constraints (e.g. "CHO", "mammalian cell line",
   "non-mammalian", "HEK293"). Extract verbatim or lightly normalised.

5. **jurisdiction** – Patent/commercial jurisdiction constraints (e.g. "US",
   "EU", "Japan", "global ex-US"). Extract jurisdiction tokens.

6. **temporal** – Patent or exclusivity date window.  Map to a JSON object with
   three optional fields:
   - `before`: upper-bound date as "YYYY-MM-DD" (e.g. "before 2028" → "2028-01-01")
   - `after`:  lower-bound date as "YYYY-MM-DD" (e.g. "after 2020" → "2020-01-01")
   - `not_expired`: true if the query requests only patents/exclusivity still in
     force (e.g. "not expired yet", "still active").
   "between now and 2028" should produce after = today's date, before = 2028-01-01.
   If no temporal information is present, set temporal to null.

7. **commercial** – Market, geography, or revenue constraints (e.g. "US", "EU",
   "revenue > 1B", "top-selling").  Extract verbatim or lightly normalised.

## Query classification

Set `query_type` to one of the following values based on the primary intent of
the query:

- `drug_lookup` – The query names one or more specific drugs (by INN or brand
  name) and is primarily asking about those drugs.  Example: "adalimumab",
  "Humira patent expiry", "trastuzumab biosimilar".
- `opportunity_scan` – The query asks for a broad set of patent or market
  opportunities, often with date or exclusivity filters but without naming
  specific drugs.  Example: "PD-1 inhibitors expiring 2028", "mAbs losing
  exclusivity before 2030".
- `biosimilar_screen` – The query explicitly asks about biosimilar opportunities
  or filters.  Example: "biosimilar opportunities revenue >1B", "biosimilar
  candidates for monoclonal antibodies".
- `general` – Everything else that does not fit the above categories.  Example:
  "VEGF pathway", "what is a bispecific antibody".

Use the most specific matching category.  If a query names a drug AND asks about
biosimilar opportunities, prefer `biosimilar_screen`.

## Drug name extraction

Set `drug_name` to a list of drug names (INN or brand names) explicitly
mentioned in the query.  Extract only names that appear in the query; do not
infer names from targets or indications.  Return an empty list `[]` when no
drug names are present.

## Output rules

- Return empty lists `[]` for dimensions with no signal.
- Return `null` (JSON) for `temporal` when no date/expiry signal is present.
- Do not invent signals that are not present in the query.
"""


class TemporalSignal(BaseModel):
    """DateWindow-compatible temporal signal extracted from NL query."""

    before: date | None = None
    after: date | None = None
    not_expired: bool = False


class RawSignals(BaseModel):
    """Raw search signals extracted from NL query, tagged by dimension."""

    target: list[str] = Field(default_factory=list)
    modality: list[str] = Field(default_factory=list)
    indication: list[str] = Field(default_factory=list)
    cell_line: list[str] = Field(default_factory=list)
    jurisdiction: list[str] = Field(default_factory=list)
    temporal: TemporalSignal | None = None
    commercial: list[str] = Field(default_factory=list)
    query_type: str = Field(
        default="general",
        description=(
            "Query classification: opportunity_scan | drug_lookup | "
            "biosimilar_screen | general"
        ),
    )
    drug_name: list[str] = Field(
        default_factory=list,
        description="Drug names (INN or brand names) mentioned in the query.",
    )


class IntentExtractor:
    """Extracts raw search signals from natural-language queries using LLM structured output."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self._model_name = model_name
        genai_kwargs: dict[str, Any] = {"model": model_name}
        if _settings is not None and _settings.GOOGLE_API_KEY:
            genai_kwargs["google_api_key"] = _settings.GOOGLE_API_KEY
        self._llm: Any = ChatGoogleGenerativeAI(**genai_kwargs).with_structured_output(
            RawSignals
        )

    async def extract(self, query: str) -> RawSignals:
        """Extract dimension-tagged signals from a natural-language query.

        Args:
            query: Free-text search query from the user.

        Returns:
            RawSignals with populated dimension fields.
        """
        today = date.today().isoformat()
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(today=today)
        messages = [
            ("system", system_prompt),
            ("human", query),
        ]
        result = await self._llm.ainvoke(messages)
        return result
