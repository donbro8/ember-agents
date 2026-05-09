"""Discovery Agent implementation."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from ember_agents.base import Agent
from ember_agents.discovery.tools import (
    _extract_drug_name,
    search_fda_events,
    search_patents,
)
from ember_agents.factory import register_agent
from ember_agents.search.urls import patent_url


@register_agent("discovery")
class DiscoveryAgent(Agent):
    """Agent that searches patents and FDA adverse-event data."""

    async def run(self, query: str) -> AsyncGenerator[str, None]:
        """Run a discovery search and yield markdown output.

        Args:
            query: Natural-language query (used for both patent and FDA search).

        Yields:
            Markdown-formatted sections: header, patents, FDA events, sources.
        """
        extracted_term = _extract_drug_name(query)
        yield f"# Discovery Report: {query}\n\n"
        yield f"> Searching for: **{extracted_term}**\n\n"

        # --- Patents ---
        try:
            patents = search_patents(query)
        except Exception as exc:
            patents = []
            yield f"> **Warning:** Could not query patents: {exc}\n\n"
        yield "## Patents\n\n"
        if patents:
            for pat in patents:
                title = pat.get("title", "Untitled")
                filing_date = pat.get("filing_date", "N/A")
                country = pat.get("country_code_from_pub", pat.get("country_code", ""))
                pub_number = pat.get("publication_number", "")
                url = patent_url(pub_number) if pub_number else None
                if url:
                    yield f"- [**{title}**]({url}) ({country} \u00b7 filed {filing_date})\n"
                else:
                    yield f"- **{title}** ({country} \u00b7 filed {filing_date})\n"
        else:
            yield "_No patent results found._\n"
        yield "\n"

        # --- FDA Drug Labels ---
        try:
            fda_events = search_fda_events(query)
        except Exception as exc:
            fda_events = []
            yield f"> **Warning:** Could not query FDA drug labels: {exc}\n\n"
        yield "## FDA Drug Labels\n\n"
        if fda_events:
            for evt in fda_events:
                brand = evt.get("openfda_brand_name", "Unknown brand")
                generic = evt.get("openfda_generic_name", "Unknown generic")
                manufacturer = evt.get("openfda_manufacturer_name", "Unknown manufacturer")
                yield f"- {brand} ({generic}) \u2014 {manufacturer}\n"
        else:
            yield "_No FDA drug label results found._\n"
        yield "\n"

        # --- Sources ---
        yield "## Sources\n\n"
        yield "- Google Patents Public Dataset (BigQuery)\n"
        yield "- FDA Adverse Event Reporting System (BigQuery)\n"
