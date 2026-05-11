"""Tests for the search Interpret phase."""

from datetime import date

import pytest

from ember_agents.search.interpret import IntentExtractor, RawSignals, TemporalSignal


class _FakeStructuredLLM:
    def __init__(self, response: RawSignals) -> None:
        self.response = response
        self.messages = None

    async def ainvoke(self, messages):
        self.messages = messages
        return self.response


class _FakeChatGoogleGenerativeAI:
    def __init__(self, response: RawSignals, calls: list[dict]) -> None:
        self.response = response
        self.calls = calls

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self

    def with_structured_output(self, schema):
        self.calls.append({"schema": schema})
        return _FakeStructuredLLM(self.response)


def test_raw_signals_defaults_are_dimension_tagged() -> None:
    signals = RawSignals()

    assert signals.target == []
    assert signals.modality == []
    assert signals.indication == []
    assert signals.temporal is None
    assert signals.commercial == []
    assert signals.query_type == "general"
    assert signals.drug_name == []


def test_temporal_signal_accepts_date_window_compatible_fields() -> None:
    signal = TemporalSignal(
        before=date(2028, 1, 1),
        after=date(2026, 5, 7),
        not_expired=True,
    )

    assert signal.before == date(2028, 1, 1)
    assert signal.after == date(2026, 5, 7)
    assert signal.not_expired is True


@pytest.mark.asyncio
async def test_intent_extractor_uses_structured_output(monkeypatch) -> None:
    response = RawSignals(
        target=["HER2"],
        modality=["mAb"],
        indication=["breast cancer"],
        temporal=TemporalSignal(before=date(2028, 1, 1), not_expired=True),
        commercial=["US"],
    )
    calls: list[dict] = []
    fake_chat = _FakeChatGoogleGenerativeAI(response, calls)
    monkeypatch.setattr(
        "ember_agents.search.interpret.ChatGoogleGenerativeAI", fake_chat
    )

    extractor = IntentExtractor(model_name="test-model")
    signals = await extractor.extract(
        "Find US mAbs targeting HER2 in breast cancer with patents before 2028 "
        "that are not expired yet"
    )

    assert signals == response
    assert calls[0]["model"] == "test-model"
    assert calls[1] == {"schema": RawSignals}


@pytest.mark.asyncio
async def test_prompt_includes_temporal_mapping_rules(monkeypatch) -> None:
    response = RawSignals()
    calls: list[dict] = []
    fake_chat = _FakeChatGoogleGenerativeAI(response, calls)
    monkeypatch.setattr(
        "ember_agents.search.interpret.ChatGoogleGenerativeAI", fake_chat
    )

    extractor = IntentExtractor()
    llm = extractor._llm
    await extractor.extract("between now and 2028")

    system_prompt = llm.messages[0][1]
    assert "before 2028" in system_prompt
    assert "between now and 2028" in system_prompt
    assert "not expired yet" in system_prompt
    assert date.today().isoformat() in system_prompt


@pytest.mark.asyncio
async def test_query_type_drug_lookup(monkeypatch) -> None:
    response = RawSignals(query_type="drug_lookup", drug_name=["adalimumab"])
    calls: list[dict] = []
    fake_chat = _FakeChatGoogleGenerativeAI(response, calls)
    monkeypatch.setattr(
        "ember_agents.search.interpret.ChatGoogleGenerativeAI", fake_chat
    )

    extractor = IntentExtractor()
    signals = await extractor.extract("adalimumab")

    assert signals.query_type == "drug_lookup"
    assert "adalimumab" in signals.drug_name


@pytest.mark.asyncio
async def test_query_type_opportunity_scan(monkeypatch) -> None:
    response = RawSignals(query_type="opportunity_scan")
    calls: list[dict] = []
    fake_chat = _FakeChatGoogleGenerativeAI(response, calls)
    monkeypatch.setattr(
        "ember_agents.search.interpret.ChatGoogleGenerativeAI", fake_chat
    )

    extractor = IntentExtractor()
    signals = await extractor.extract("PD-1 inhibitors expiring 2028")

    assert signals.query_type == "opportunity_scan"


@pytest.mark.asyncio
async def test_query_type_biosimilar_screen(monkeypatch) -> None:
    response = RawSignals(query_type="biosimilar_screen")
    calls: list[dict] = []
    fake_chat = _FakeChatGoogleGenerativeAI(response, calls)
    monkeypatch.setattr(
        "ember_agents.search.interpret.ChatGoogleGenerativeAI", fake_chat
    )

    extractor = IntentExtractor()
    signals = await extractor.extract("biosimilar opportunities revenue >1B")

    assert signals.query_type == "biosimilar_screen"


@pytest.mark.asyncio
async def test_query_type_general(monkeypatch) -> None:
    response = RawSignals(query_type="general")
    calls: list[dict] = []
    fake_chat = _FakeChatGoogleGenerativeAI(response, calls)
    monkeypatch.setattr(
        "ember_agents.search.interpret.ChatGoogleGenerativeAI", fake_chat
    )

    extractor = IntentExtractor()
    signals = await extractor.extract("VEGF pathway")

    assert signals.query_type == "general"


@pytest.mark.asyncio
async def test_prompt_includes_query_type_and_drug_name_rules(monkeypatch) -> None:
    response = RawSignals()
    calls: list[dict] = []
    fake_chat = _FakeChatGoogleGenerativeAI(response, calls)
    monkeypatch.setattr(
        "ember_agents.search.interpret.ChatGoogleGenerativeAI", fake_chat
    )

    extractor = IntentExtractor()
    llm = extractor._llm
    await extractor.extract("adalimumab biosimilar")

    system_prompt = llm.messages[0][1]
    assert "query_type" in system_prompt
    assert "drug_lookup" in system_prompt
    assert "opportunity_scan" in system_prompt
    assert "biosimilar_screen" in system_prompt
    assert "drug_name" in system_prompt
