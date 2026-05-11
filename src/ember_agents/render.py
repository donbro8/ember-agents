"""Result rendering: formats ExecutionTrace + CandidateResult list into markdown."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from ember_agents.trace import ExecutionTrace

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# URL label helper
# ---------------------------------------------------------------------------

_KNOWN_DOMAIN_LABELS: dict[str, str] = {
    "pubmed.ncbi.nlm.nih.gov": "PubMed",
    "clinicaltrials.gov": "ClinicalTrials.gov",
    "patents.google.com": "Google Patents",
    "dailymed.nlm.nih.gov": "DailyMed",
    "fda.gov": "FDA",
}

_DIMENSION_LABELS: dict[str, str] = {
    "target": "Target",
    "drug_name": "Drug name",
    "indication": "Indication",
    "therapeutic_area": "Therapeutic area",
    "modality": "Modality",
}


def _url_label(url: str) -> str:
    """Derive a human-readable label from a URL.

    Maps known domains to canonical labels and falls back to capitalizing
    the first part of the domain for unknown domains.

    Args:
        url: The URL to derive a label for.

    Returns:
        A human-readable label string.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
    except Exception:  # noqa: BLE001
        return url

    # Strip leading "www."
    if hostname.startswith("www."):
        hostname = hostname[4:]

    if hostname in _KNOWN_DOMAIN_LABELS:
        return _KNOWN_DOMAIN_LABELS[hostname]

    # Check if any known domain is a suffix (e.g. subdomain of fda.gov)
    for domain, label in _KNOWN_DOMAIN_LABELS.items():
        if hostname == domain or hostname.endswith("." + domain):
            return label

    # Default: capitalize first part of domain
    first_part = hostname.split(".")[0] if hostname else url
    return first_part.capitalize() if first_part else url


def _friendly_dimension(dim: str) -> str:
    key = (dim or "").strip().lower()
    return _DIMENSION_LABELS.get(key, dim.replace("_", " ").capitalize())


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
        *,
        synthesis_overview: str | None = None,
    ) -> str:
        """Render *trace* and *results* as a markdown string.

        Args:
            trace: Execution trace from the EmberAgent pipeline.
            results: List of CandidateResult objects to display.
            synthesis_overview: Optional overview text from ResultSynthesizer.

        Returns:
            Markdown-formatted string suitable for streaming to the caller.
        """
        sections: list[str] = []

        sections.append(self._render_query_analysis(trace))
        sections.append(self._render_source_table(trace))
        sections.append(self._render_results(results))
        if synthesis_overview is not None:
            sections.append(self._render_synthesis(synthesis_overview))

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

    def _render_results(self, results: list[Any], max_results: int = 20) -> str:
        """Render the ranked results section.

        Filters out results with no identifying information (no display_label,
        drug_name, or target) and limits output to *max_results*.
        """
        if not results:
            return "## Results\n\n_No results found._"

        # Filter: keep only results with at least one identifying field
        displayable = [
            r
            for r in results
            if (
                getattr(r, "display_label", None)
                or getattr(r, "drug_name", None)
                or getattr(r, "target", None)
            )
        ]

        total = len(results)
        shown = displayable[:max_results]

        if not shown:
            return f"## Results\n\n_{total} results found but none had identifying information._"

        header = f"## Results ({len(shown)} shown"
        if total > len(shown):
            header += f", {total} total"
        header += ")"
        lines: list[str] = [header]

        for i, result in enumerate(shown, start=1):
            label = (
                getattr(result, "display_label", None)
                or getattr(result, "drug_name", None)
                or "Unknown"
            )
            overall_score = getattr(result, "overall_score", None)
            score_str = (
                f" — score: {overall_score:.3f}" if overall_score is not None else ""
            )
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

            # Score breakdown line
            semantic_score = getattr(result, "semantic_score", None)
            structured_score = getattr(result, "structured_score", None)
            evidence_score = getattr(result, "evidence_score", None)
            score_parts: list[str] = []
            if semantic_score is not None:
                score_parts.append(f"semantic: {semantic_score:.2f}")
            if structured_score is not None:
                score_parts.append(f"structured: {structured_score:.2f}")
            if evidence_score is not None:
                score_parts.append(f"evidence: {evidence_score:.2f}")
            if score_parts:
                lines.append(f"**Scores:** {', '.join(score_parts)}")

            matched = getattr(result, "matched_dimensions", None)
            missed = getattr(result, "missed_dimensions", None)
            match_explanations = getattr(result, "match_explanations", None)
            if not matched and isinstance(match_explanations, dict):
                matched = match_explanations.get("matched_dimensions") or []
            if not missed and isinstance(match_explanations, dict):
                missed = match_explanations.get("missed_dimensions") or []
            concrete_labels = getattr(result, "concrete_labels", None) or {}
            if matched:
                matched_text: list[str] = []
                for d in matched:
                    dim = str(d)
                    labels = (
                        concrete_labels.get(dim)
                        if isinstance(concrete_labels, dict)
                        else None
                    )
                    if labels:
                        joined = ", ".join(str(v) for v in labels if v)
                        matched_text.append(f"{_friendly_dimension(dim)} ({joined})")
                    else:
                        matched_text.append(_friendly_dimension(dim))
                lines.append(f"**Matched dimensions:** {', '.join(matched_text)}")
            if missed:
                missed_labels = ", ".join(_friendly_dimension(str(d)) for d in missed)
                lines.append(f"**Missed dimensions:** {missed_labels}")

            suppression_metadata = getattr(result, "suppression_metadata", None)
            if isinstance(suppression_metadata, dict):
                threshold = suppression_metadata.get("threshold")
                suppressed = suppression_metadata.get("suppressed")
                query_type = suppression_metadata.get("query_type")
                meta_parts: list[str] = []
                if isinstance(threshold, int | float):
                    meta_parts.append(f"threshold: {float(threshold):.2f}")
                if isinstance(suppressed, bool):
                    meta_parts.append(f"suppressed: {'yes' if suppressed else 'no'}")
                if query_type:
                    meta_parts.append(f"query type: {query_type}")
                if meta_parts:
                    lines.append(f"**Suppression:** {', '.join(meta_parts)}")

            # Evidence counts line
            trial_count = getattr(result, "trial_count", None)
            if trial_count is None:
                trials = getattr(result, "trials", None)
                trial_count = len(trials) if trials is not None else 0

            article_count = getattr(result, "article_count", None)
            if article_count is None:
                articles = getattr(result, "articles", None)
                article_count = len(articles) if articles is not None else 0

            patents = getattr(result, "patents", []) or []
            patent_count = len(patents)

            evidence_parts: list[str] = []
            if trial_count:
                latest_phase = getattr(result, "latest_trial_phase", None)
                trial_str = f"{trial_count} trial{'s' if trial_count != 1 else ''}"
                if latest_phase:
                    trial_str += f" (latest: {latest_phase})"
                evidence_parts.append(trial_str)
            if article_count:
                evidence_parts.append(
                    f"{article_count} article{'s' if article_count != 1 else ''}"
                )
            if patent_count:
                evidence_parts.append(
                    f"{patent_count} patent{'s' if patent_count != 1 else ''}"
                )
            if evidence_parts:
                lines.append(" · ".join(evidence_parts))

            evidence_summary = getattr(result, "evidence_summary", None)
            if isinstance(evidence_summary, dict):
                summary_parts: list[str] = []
                if evidence_summary.get("trial_count") is not None:
                    summary_parts.append(
                        f"trials: {evidence_summary.get('trial_count')}"
                    )
                if evidence_summary.get("article_count") is not None:
                    summary_parts.append(
                        f"articles: {evidence_summary.get('article_count')}"
                    )
                if evidence_summary.get("patent_count") is not None:
                    summary_parts.append(
                        f"patents: {evidence_summary.get('patent_count')}"
                    )
                latest_phase = evidence_summary.get("latest_trial_phase")
                if latest_phase:
                    summary_parts.append(f"latest trial phase: {latest_phase}")
                if summary_parts:
                    lines.append(f"**Evidence summary:** {', '.join(summary_parts)}")

            # Commercial line
            annual_revenue = getattr(result, "annual_revenue_usd_millions", None)
            if annual_revenue is not None and annual_revenue > 0:
                revenue_year = getattr(result, "revenue_year", None)
                revenue_int = int(annual_revenue)
                revenue_formatted = f"${revenue_int:,}M"
                if revenue_year:
                    revenue_formatted += f" ({revenue_year})"
                biosimilar_count = getattr(result, "biosimilar_competitor_count", None)
                if biosimilar_count is None:
                    biosimilar_competitors = getattr(
                        result, "biosimilar_competitors", None
                    )
                    biosimilar_count = (
                        len(biosimilar_competitors) if biosimilar_competitors else 0
                    )
                commercial_parts = [f"Revenue: {revenue_formatted}"]
                if biosimilar_count:
                    commercial_parts.append(f"Biosimilars: {biosimilar_count} approved")
                lines.append(f"**Commercial:** {' · '.join(commercial_parts)}")

            # Patent jurisdictions
            if patents:
                lines.append("\n**Patent jurisdictions:**")
                lines.append("| Jurisdiction | Expiry | Status | Link |")
                lines.append("|---|---|---|---|")
                for patent in patents:
                    country = getattr(patent, "country_name", None) or getattr(
                        patent, "country_code", ""
                    )
                    expiry = getattr(patent, "expiry_date", None)
                    expiry_approximate = getattr(
                        patent, "expiry_date_approximate", False
                    )
                    if expiry:
                        expiry_str = str(expiry)
                        if expiry_approximate:
                            expiry_str = f"~{expiry_str} (estimated)"
                    else:
                        expiry_str = "N/A"
                    status = getattr(patent, "status", "unknown")
                    patent_url = getattr(patent, "url", None)
                    link_str = f"[Patent]({patent_url})" if patent_url else ""
                    lines.append(
                        f"| {country} | {expiry_str} | {status} | {link_str} |"
                    )

            # Source URLs
            source_urls = getattr(result, "source_urls", None) or []
            if source_urls:
                url_links = [f"[{_url_label(url)}]({url})" for url in source_urls]
                lines.append(f"**Sources:** {' · '.join(url_links)}")

        return "\n".join(lines)

    def _render_synthesis(self, overview: str) -> str:
        """Render the synthesis analysis section."""
        return f"## Analysis\n\n{overview}"
