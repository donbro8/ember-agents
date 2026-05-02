"""Discovery Agent implementation."""

from __future__ import annotations

from collections.abc import AsyncGenerator

from ember_agents.base import Agent
from ember_agents.discovery.tools import search_fda_events, search_patents
from ember_agents.factory import register_agent


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
        yield f"# Discovery Report: {query}\n\n"

        # --- Patents ---
        patents = search_patents(query)
        yield "## Patents\n\n"
        if patents:
            for pat in patents:
                title = pat.get("title", "Untitled")
                pub_date = pat.get("publication_date", "N/A")
                yield f"- **{title}** (published {pub_date})\n"
        else:
            yield "_No patent results found._\n"
        yield "\n"

        # --- FDA Adverse Events ---
        fda_events = search_fda_events(query)
        yield "## FDA Adverse Events\n\n"
        if fda_events:
            for evt in fda_events:
                reaction = evt.get("reaction", "Unknown reaction")
                severity = evt.get("severity", "N/A")
                yield f"- {reaction} (severity: {severity})\n"
        else:
            yield "_No FDA adverse-event results found._\n"
        yield "\n"

        # --- Sources ---
        yield "## Sources\n\n"
        yield "- Google Patents Public Dataset (BigQuery)\n"
        yield "- FDA Adverse Event Reporting System (BigQuery)\n"
