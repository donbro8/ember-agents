"""Tests for the BiosimilarAgent."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from ember_agents import get_agent
from ember_agents.base import Agent
from ember_agents.biosimilar.agent import BiosimilarAgent
from ember_data.seed.schema import BiologicEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_seed_entries() -> list[BiologicEntry]:
    """Return a small seed dataset with one passing and one failing entry."""
    return [
        BiologicEntry(
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
        BiologicEntry(
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
    """Patch load_biologic_seed in biosimilar.agent to return controlled data."""
    with patch("ember_agents.biosimilar.agent.load_biologic_seed") as mock_fn:
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
        assert "all biologics" in output

    async def test_run_yields_ranked_table(self, mock_bq, mock_seed):
        """run() yields ranked candidates table with passing entries."""
        agent = BiosimilarAgent()
        chunks: list[str] = []
        async for chunk in agent.run("test query"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "## Ranked Candidates" in output
        assert "AlphaMab" in output
        # With cell_line_class=None (no filter), BetaMab also passes
        assert "BetaMab" in output

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
        with patch("ember_agents.biosimilar.agent.load_biologic_seed") as mock_fn:
            # Provide an entry with no patent expiry data — will fail stage1
            mock_fn.return_value = [
                BiologicEntry(
                    drug_name="NoExpiry",
                    originator="NoCo",
                    target_antigen="X",
                    modality="mAb",
                    cell_line="CHO",
                    cell_line_class="mammalian",
                    annual_revenue_usd_millions=500.0,
                    revenue_year=2023,
                    patent_expiry_us=None,
                    patent_expiry_eu=None,
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


class TestBiosimilarAgentFormatting:
    """Tests for TASK-043 context headers and TASK-045 formatting improvements."""

    async def test_context_header_present(self, mock_bq, mock_seed):
        """run() yields the screening context header after filter summary."""
        agent = BiosimilarAgent()
        chunks: list[str] = []
        async for chunk in agent.run("test query"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "Screening **2** biologics across" in output
        assert "categories for biosimilar development opportunity" in output

    async def test_category_breakdown_present(self, mock_bq, mock_seed):
        """run() yields a category breakdown line."""
        agent = BiosimilarAgent()
        chunks: list[str] = []
        async for chunk in agent.run("test query"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "**Category breakdown:**" in output
        assert "mAb:" in output

    async def test_ranked_table_has_category_column(self, mock_bq, mock_seed):
        """run() includes Category column in the ranked table header."""
        agent = BiosimilarAgent()
        chunks: list[str] = []
        async for chunk in agent.run("test query"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "| Rank | Drug | Category |" in output

    async def test_revenue_formatted_as_billions(self, mock_bq, mock_seed):
        """Revenue >= 1000M is formatted as $X.XB."""
        agent = BiosimilarAgent()
        chunks: list[str] = []
        async for chunk in agent.run("test query"):
            chunks.append(chunk)

        output = "".join(chunks)
        # AlphaMab has 3000M revenue -> $3.0B
        assert "$3.0B" in output

    async def test_key_takeaways_section(self, mock_bq, mock_seed):
        """run() yields a Key Takeaways section."""
        agent = BiosimilarAgent()
        chunks: list[str] = []
        async for chunk in agent.run("test query"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "## Key Takeaways" in output
        assert "**AlphaMab**" in output

    async def test_patent_jurisdiction_breakdown(self, mock_bq, mock_seed):
        """run() yields jurisdiction breakdown in patent details."""
        agent = BiosimilarAgent()
        chunks: list[str] = []
        async for chunk in agent.run("test query"):
            chunks.append(chunk)

        output = "".join(chunks)
        # AlphaMab has US and EU patents
        assert "Jurisdictions:" in output


class TestFormatHelpers:
    """Tests for _format_revenue and _format_expiry helpers."""

    def test_format_revenue_billions(self):
        from ember_agents.biosimilar.agent import _format_revenue
        assert _format_revenue(3000.0) == "$3.0B"
        assert _format_revenue(1000.0) == "$1.0B"
        assert _format_revenue(21237.0) == "$21.2B"

    def test_format_revenue_millions(self):
        from ember_agents.biosimilar.agent import _format_revenue
        assert _format_revenue(500.0) == "$500M"
        assert _format_revenue(999.0) == "$999M"

    def test_format_expiry_expired(self):
        from ember_agents.biosimilar.agent import _format_expiry
        result = _format_expiry(date(2020, 1, 1))
        assert "Expired" in result
        assert "2020" in result

    def test_format_expiry_future_with_jurisdiction(self):
        from ember_agents.biosimilar.agent import _format_expiry
        future = date(2030, 6, 1)
        result = _format_expiry(future, "US")
        assert "years" in result
        assert "(US)" in result

    def test_format_expiry_no_jurisdiction(self):
        from ember_agents.biosimilar.agent import _format_expiry
        future = date(2030, 6, 1)
        result = _format_expiry(future, "")
        assert "years" in result
        assert "(" not in result


class TestBiosimilarAgentWithSeedData:
    """Tests that the agent works with the real seed dataset."""

    async def test_agent_returns_results_from_seed(self, mock_bq):
        """Agent loads seed data and returns candidates."""
        agent = BiosimilarAgent()
        chunks: list[str] = []
        async for chunk in agent.run("full pipeline test"):
            chunks.append(chunk)

        output = "".join(chunks)
        assert "## Ranked Candidates" in output
        # The 150-entry seed should produce many candidates
        assert "After Stage 1" in output
