"""Tests for the DiscoveryAgent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ember_agents.discovery.agent import DiscoveryAgent


@pytest.fixture()
def mock_bq():
    """Patch BigQueryClient in discovery.tools so no real BQ calls are made."""
    with patch("ember_agents.discovery.tools.BigQueryClient") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


class TestDiscoveryAgent:
    """Tests for DiscoveryAgent.run()."""

    async def test_run_with_results(self, mock_bq: MagicMock):
        """run() yields markdown containing patent and FDA event data."""
        mock_bq.search_patents.return_value = [
            {"title": "Gene Therapy Patent", "publication_date": "2024-01-15"},
        ]
        mock_bq.query_fda_drug_events.return_value = [
            {"reaction": "Headache", "severity": "Mild"},
        ]

        agent = DiscoveryAgent()
        chunks: list[str] = []
        async for chunk in agent.run("CRISPR"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "# Discovery Report: CRISPR" in output
        assert "Gene Therapy Patent" in output
        assert "2024-01-15" in output
        assert "Headache" in output
        assert "Mild" in output
        assert "## Sources" in output

    async def test_run_empty_results(self, mock_bq: MagicMock):
        """run() handles empty results gracefully."""
        mock_bq.search_patents.return_value = []
        mock_bq.query_fda_drug_events.return_value = []

        agent = DiscoveryAgent()
        chunks: list[str] = []
        async for chunk in agent.run("nonexistent-compound"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "No patent results found" in output
        assert "No FDA adverse-event results found" in output
