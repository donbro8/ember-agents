"""Tests for the BiosimilarAgent."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ember_agents import get_agent
from ember_agents.base import Agent
from ember_agents.biosimilar.agent import BiosimilarAgent
from ember_data.seed.schema import MabEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seed_entries() -> list[MabEntry]:
    """Return a small seed dataset with one passing and one failing entry."""
    return [
        MabEntry(
            drug_name="AlphaMab",
            brand_names=["AlphaBrand"],
            originator="AlphaCo",
            target_antigen="IL-6",
            modality="mAb",
            cell_line="CHO",
            cell_line_class="mammalian",
            indications=["Rheumatoid Arthritis"],
            annual_revenue_usd_millions=3000.0,
            revenue_year=2023,
            patent_expiry_us=date(2026, 1, 1),
            patent_expiry_eu=date(2027, 6, 1),
            key_patent_numbers=["US9999999"],
            biosimilar_competitors=["BioSim1", "BioSim2"],
            has_approved_biosimilar=True,
        ),
        # This entry will fail stage1: wrong cell_line_class
        MabEntry(
            drug_name="BetaMab",
            originator="BetaCo",
            target_antigen="VEGF",
            modality="mAb",
            cell_line="E. coli",
            cell_line_class="microbial",
            annual_revenue_usd_millions=2000.0,
            revenue_year=2023,
            patent_expiry_us=date(2025, 1, 1),
            has_approved_biosimilar=False,
        ),
    ]


@pytest.fixture()
def mock_bq():
    """Patch BigQueryClient in biosimilar.agent so no real BQ calls are made."""
    with patch("ember_agents.biosimilar.agent.BigQueryClient") as mock_cls:
        client = MagicMock()
        client.search_patents.return_value = [
            {
                "publication_number": "US8888888",
                "title": "Anti-IL-6 Antibody",
                "abstract": "An antibody that binds IL-6.",
                "claims": ["Claim 1", "Claim 2"],
                "assignee": "AlphaCo",
                "filing_date": date(2008, 3, 15),
                "grant_date": date(2011, 1, 20),
                "cited_by_count": 42,
                "relevant_targets": ["IL-6"],
            }
        ]
        mock_cls.return_value = client
        yield client


@pytest.fixture()
def mock_seed():
    """Patch load_mab_seed in biosimilar.agent to return controlled data."""
    with patch("ember_agents.biosimilar.agent.load_mab_seed") as mock_fn:
        mock_fn.return_value = _make_seed_entries()
        yield mock_fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBiosimilarAgentRegistration:
    """Tests for agent registry integration."""

    def test_registered_as_biosimilar(self):
        """get_agent('biosimilar') returns a BiosimilarAgent instance."""
        agent = get_agent("biosimilar")
        assert isinstance(agent, Agent)
        assert isinstance(agent, BiosimilarAgent)

    def test_agent_name_in_list(self):
        """'biosimilar' is listed in AgentFactory.list_agents()."""
        from ember_agents import AgentFactory
        assert "biosimilar" in AgentFactory.list_agents()


class TestBiosimilarAgentRun:
    """Tests for BiosimilarAgent.run()."""

    async def test_run_yields_header(self, mock_bq, mock_seed):
        """run() yields a markdown header containing the query."""
        agent = BiosimilarAgent()
        chunks: list[str] = []
        async for chunk in agent.run("IL-6 inhibitors"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "# Biosimilar Candidate Screening: IL-6 inhibitors" in output

    async def test_run_yields_filter_summary(self, mock_bq, mock_seed):
        """run() yields filter summary section."""
        agent = BiosimilarAgent()
        chunks: list[str] = []
        async for chunk in agent.run("test query"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "## Filter Summary" in output
        assert "Total seed entries" in output
        assert "After Stage 1" in output
        assert "After Stage 2" in output

    async def test_run_yields_ranked_table(self, mock_bq, mock_seed):
        """run() yields ranked candidates table with the passing entry."""
        agent = BiosimilarAgent()
        chunks: list[str] = []
        async for chunk in agent.run("test query"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "## Ranked Candidates" in output
        assert "AlphaMab" in output
        # Failing entry should not appear
        assert "BetaMab" not in output

    async def test_run_yields_patent_details(self, mock_bq, mock_seed):
        """run() yields patent details section with BQ-retrieved data."""
        agent = BiosimilarAgent()
        chunks: list[str] = []
        async for chunk in agent.run("test query"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "## Patent Details" in output
        assert "US8888888" in output
        assert "Anti-IL-6 Antibody" in output

    async def test_run_no_candidates(self, mock_bq):
        """run() handles the case where no entries pass stage1 filters."""
        with patch("ember_agents.biosimilar.agent.load_mab_seed") as mock_fn:
            # Provide only microbial entries which will fail the mammalian filter
            mock_fn.return_value = [
                MabEntry(
                    drug_name="OnlyMicrobial",
                    originator="MicrobCo",
                    target_antigen="X",
                    modality="mAb",
                    cell_line="E. coli",
                    cell_line_class="microbial",
                    annual_revenue_usd_millions=500.0,
                    revenue_year=2023,
                    patent_expiry_us=date(2025, 1, 1),
                    has_approved_biosimilar=False,
                )
            ]

            agent = BiosimilarAgent()
            chunks: list[str] = []
            async for chunk in agent.run("test"):
                chunks.append(chunk)

            output = "".join(chunks)
            assert "No candidates passed" in output

    async def test_run_is_async_generator(self, mock_bq, mock_seed):
        """run() is an async generator."""
        import inspect
        agent = BiosimilarAgent()
        gen = agent.run("test")
        assert inspect.isasyncgen(gen)
        # Consume it
        async for _ in gen:
            pass

    async def test_bq_client_instantiated(self, mock_bq, mock_seed):
        """BigQueryClient is instantiated during run()."""
        from ember_agents.biosimilar.agent import BiosimilarAgent as _Agent
        with patch("ember_agents.biosimilar.agent.BigQueryClient") as mock_cls:
            mock_cls.return_value = mock_bq
            agent = _Agent()
            async for _ in agent.run("test"):
                pass
            mock_cls.assert_called_once()
