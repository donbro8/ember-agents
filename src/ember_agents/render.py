"""Result rendering: formats ExecutionTrace + CandidateResult list into markdown."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ember_agents.trace import ExecutionTrace

if TYPE_CHECKING:
    pass


class ResultRenderer:
    """Formats an ExecutionTrace and a list of CandidateResult objects into markdown.

    Sections rendered (in order):

    1. **Query analysis** — extracted signals and query classification.
    2. **Source status table** — per-source fetch outcome and result counts.
    3. **Results** — ranked candidates with patent jurisdiction details.
    """

    def render(
        self,
        trace: ExecutionTrace,
        results: list[Any],
    ) -> str:
        """Render *trace* and *results* as a markdown string.

        Args:
            trace: Execution trace from the EmberAgent pipeline.
            results: List of CandidateResult objects to display.

        Returns:
            Markdown-formatted string suitable for streaming to the caller.
        """
        sections: list[str] = []

        sections.append(self._render_query_analysis(trace))
        sections.append(self._render_source_table(trace))
        sections.append(self._render_results(results))

        return "\n\n".join(s for s in sections if s)

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _render_query_analysis(self, trace: ExecutionTrace) -> str:
        """Render the query analysis section."""
        lines: list[str] = ["## Query Analysis"]

        lines.append(f"**Query type:** `{trace.query_type}`")

        if trace.extracted_signals:
            lines.append("\n**Extracted signals:**")
            for dim, values in trace.extracted_signals.items():
                if values:
                    if isinstance(values, list):
                        value_str = ", ".join(str(v) for v in values)
                    else:
                        value_str = str(values)
                    lines.append(f"- **{dim}:** {value_str}")

        if trace.resolved_classifications:
            lines.append("\n**Resolved classifications:**")
            for dim, val in trace.resolved_classifications.items():
                lines.append(f"- **{dim}:** {val}")

        lines.append(f"\n**Gate outcome:** {trace.gate_outcome}")
        lines.append(f"**Duration:** {trace.duration_seconds:.2f}s")

        return "\n".join(lines)

    def _render_source_table(self, trace: ExecutionTrace) -> str:
        """Render the per-source status table."""
        if not trace.source_statuses:
            return ""

        lines: list[str] = ["## Source Status"]
        lines.append("| Source | Status | Results |")
        lines.append("|---|---|---|")

        for src in trace.source_statuses:
            lines.append(f"| {src.name} | {src.status} | {src.result_count} |")

        return "\n".join(lines)

    def _render_results(self, results: list[Any]) -> str:
        """Render the ranked results section."""
        if not results:
            return "## Results\n\n_No results found._"

        lines: list[str] = [f"## Results ({len(results)} found)"]

        for i, result in enumerate(results, start=1):
            label = getattr(result, "display_label", None) or getattr(result, "drug_name", None) or "Unknown"
            overall_score = getattr(result, "overall_score", None)
            score_str = f" — score: {overall_score:.3f}" if overall_score is not None else ""
            lines.append(f"\n### {i}. {label}{score_str}")

            target = getattr(result, "target", None)
            if target:
                lines.append(f"**Target:** {target}")

            indication = getattr(result, "indication", None)
            if indication:
                lines.append(f"**Indication:** {indication}")

            synthesis = getattr(result, "synthesis_summary", None)
            if synthesis:
                lines.append(f"**Summary:** {synthesis}")

            risk_flags = getattr(result, "risk_flags", [])
            if risk_flags:
                lines.append(f"**Risk flags:** {', '.join(risk_flags)}")

            # Patent jurisdictions
            patents = getattr(result, "patents", [])
            if patents:
                lines.append("\n**Patent jurisdictions:**")
                lines.append("| Jurisdiction | Expiry | Status |")
                lines.append("|---|---|---|")
                for patent in patents:
                    country = getattr(patent, "country_name", None) or getattr(patent, "country_code", "")
                    expiry = getattr(patent, "expiry_date", None)
                    expiry_str = str(expiry) if expiry else "N/A"
                    status = getattr(patent, "status", "unknown")
                    lines.append(f"| {country} | {expiry_str} | {status} |")

        return "\n".join(lines)
