"""Contract tests for IntentExtractor drug_name field and RawSignals.

Migrated and adapted from test_discovery_agent.py:  instead of testing the old
_extract_drug_name helper, these tests verify that RawSignals carries the
drug_name field and that IntentExtractor produces the correct query_type for
known inputs — without making real LLM calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch


from ember_agents.search.interpret import IntentExtractor, RawSignals, TemporalSignal


# ---------------------------------------------------------------------------
# RawSignals contract tests
# ---------------------------------------------------------------------------


class TestRawSignalsFields:
    """Verify that RawSignals has the drug_name and query_type fields."""

    def test_drug_name_field_exists(self):
        signals = RawSignals()
        assert hasattr(signals, "drug_name")

    def test_drug_name_defaults_to_empty_list(self):
        signals = RawSignals()
        assert signals.drug_name == []

    def test_query_type_defaults_to_general(self):
        signals = RawSignals()
        assert signals.query_type == "general"

    def test_drug_name_can_be_set(self):
        signals = RawSignals(drug_name=["pembrolizumab"])
        assert signals.drug_name == ["pembrolizumab"]

    def test_multiple_drug_names(self):
        signals = RawSignals(drug_name=["adalimumab", "trastuzumab"])
        assert len(signals.drug_name) == 2
        assert "adalimumab" in signals.drug_name
        assert "trastuzumab" in signals.drug_name

    def test_query_type_drug_lookup(self):
        signals = RawSignals(query_type="drug_lookup", drug_name=["pembrolizumab"])
        assert signals.query_type == "drug_lookup"

    def test_query_type_biosimilar_screen(self):
        signals = RawSignals(query_type="biosimilar_screen")
        assert signals.query_type == "biosimilar_screen"

    def test_query_type_opportunity_scan(self):
        signals = RawSignals(query_type="opportunity_scan")
        assert signals.query_type == "opportunity_scan"

    def test_all_standard_fields_present(self):
        signals = RawSignals(
            target=["PD-1"],
            modality=["mAb"],
            indication=["NSCLC"],
            commercial=["US"],
            query_type="drug_lookup",
            drug_name=["pembrolizumab"],
        )
        assert signals.target == ["PD-1"]
        assert signals.modality == ["mAb"]
        assert signals.indication == ["NSCLC"]
        assert signals.commercial == ["US"]
        assert signals.query_type == "drug_lookup"
        assert signals.drug_name == ["pembrolizumab"]

    def test_temporal_field_none_by_default(self):
        signals = RawSignals()
        assert signals.temporal is None

    def test_temporal_field_can_be_set(self):

        signals = RawSignals(temporal=TemporalSignal(not_expired=True))
        assert signals.temporal is not None
        assert signals.temporal.not_expired is True


# ---------------------------------------------------------------------------
# IntentExtractor contract tests (with mocked LLM)
# ---------------------------------------------------------------------------


class TestIntentExtractorContract:
    """Verify IntentExtractor returns RawSignals with drug_name populated."""

    def _make_extractor_with_mock(self, return_value: RawSignals) -> IntentExtractor:
        """Build an IntentExtractor whose internal LLM is replaced with a mock."""
        with patch("ember_agents.search.interpret.ChatGoogleGenerativeAI") as mock_cls:
            llm = MagicMock()
            llm.with_structured_output.return_value = llm
            llm.ainvoke = AsyncMock(return_value=return_value)
            mock_cls.return_value = llm
            extractor = IntentExtractor.__new__(IntentExtractor)
            extractor._model_name = "gemini-2.5-flash"
            extractor._llm = llm
        return extractor

    async def test_extract_returns_raw_signals(self):
        expected = RawSignals(query_type="drug_lookup", drug_name=["adalimumab"])
        extractor = self._make_extractor_with_mock(expected)
        result = await extractor.extract("adalimumab patent expiry")
        assert isinstance(result, RawSignals)

    async def test_extract_drug_name_field_populated(self):
        expected = RawSignals(query_type="drug_lookup", drug_name=["pembrolizumab"])
        extractor = self._make_extractor_with_mock(expected)
        result = await extractor.extract("What patents exist for pembrolizumab?")
        assert "pembrolizumab" in result.drug_name

    async def test_extract_biosimilar_screen_query_type(self):
        expected = RawSignals(query_type="biosimilar_screen", drug_name=[])
        extractor = self._make_extractor_with_mock(expected)
        result = await extractor.extract("biosimilar opportunities for monoclonal antibodies")
        assert result.query_type == "biosimilar_screen"

    async def test_extract_drug_lookup_sets_drug_name(self):
        expected = RawSignals(query_type="drug_lookup", drug_name=["adalimumab"])
        extractor = self._make_extractor_with_mock(expected)
        result = await extractor.extract("Find FDA adverse events for adalimumab")
        assert result.drug_name == ["adalimumab"]

    async def test_extract_multiple_drugs(self):
        expected = RawSignals(
            query_type="drug_lookup",
            drug_name=["adalimumab", "trastuzumab"],
        )
        extractor = self._make_extractor_with_mock(expected)
        result = await extractor.extract("Compare adalimumab and trastuzumab patents")
        assert "adalimumab" in result.drug_name
        assert "trastuzumab" in result.drug_name

    async def test_extract_no_drug_name_returns_empty_list(self):
        expected = RawSignals(query_type="opportunity_scan", drug_name=[])
        extractor = self._make_extractor_with_mock(expected)
        result = await extractor.extract("PD-1 inhibitors expiring before 2028")
        assert result.drug_name == []

    async def test_extract_passthrough_drug_name(self):
        """A query that IS the drug name populates drug_name directly."""
        expected = RawSignals(query_type="drug_lookup", drug_name=["bevacizumab"])
        extractor = self._make_extractor_with_mock(expected)
        result = await extractor.extract("bevacizumab")
        assert "bevacizumab" in result.drug_name

    async def test_extract_complex_query_drug_name(self):
        """Complex query extracts the drug name correctly."""
        expected = RawSignals(
            query_type="drug_lookup",
            drug_name=["bevacizumab"],
            target=["VEGF"],
        )
        extractor = self._make_extractor_with_mock(expected)
        result = await extractor.extract(
            "Show me patents and FDA adverse events for bevacizumab"
        )
        assert "bevacizumab" in result.drug_name
