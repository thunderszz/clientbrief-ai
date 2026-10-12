"""Tests for the provider-agnostic LLM client (no network: the transport is faked)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from services import llm_client
from services.llm_client import (
    LLMSettings,
    Tier,
    collect_llm_calls,
    current_settings,
    structured_extract,
    use_llm_settings,
)


class Answer(BaseModel):
    value: str


class _StatusError(Exception):
    """Stands in for SDK errors that carry an HTTP status in ``.code``."""

    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


@pytest.fixture
def no_sleep(monkeypatch):
    delays: list[float] = []
    monkeypatch.setattr(llm_client, "_sleep", delays.append)
    return delays


def _fake_transport(monkeypatch, responses):
    """Install a transport that pops scripted responses and records each call."""
    calls: list[dict] = []
    script = list(responses)

    def transport(settings, model, system_prompt, user_prompt, schema, temperature):
        calls.append({"model": model, "user_prompt": user_prompt, "settings": settings})
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(llm_client, "_transport", transport)
    return calls


def _extract(tier: Tier = Tier.FAST):
    return structured_extract(
        system_prompt="sys", user_prompt="question", response_model=Answer, tier=tier
    )


def test_success_records_call(monkeypatch, no_sleep):
    calls = _fake_transport(monkeypatch, ['{"value": "ok"}'])

    with collect_llm_calls() as records:
        result = _extract()

    assert result == Answer(value="ok")
    assert calls[0]["model"] == "gemini-3.5-flash-lite"
    assert len(records) == 1
    assert records[0]["model"] == "gemini-3.5-flash-lite"
    assert records[0]["tier"] == "fast"
    assert records[0]["ok"] is True
    assert records[0]["fallback"] is False
    assert records[0]["attempts"] == 1


def test_reasoning_tier_uses_reasoning_model(monkeypatch, no_sleep):
    calls = _fake_transport(monkeypatch, ['{"value": "ok"}'])

    _extract(Tier.REASONING)

    assert calls[0]["model"] == "gemini-3.6-flash"


def test_retry_then_success(monkeypatch, no_sleep):
    calls = _fake_transport(monkeypatch, [_StatusError(503), '{"value": "ok"}'])

    with collect_llm_calls() as records:
        result = _extract()

    assert result == Answer(value="ok")
    assert [c["model"] for c in calls] == ["gemini-3.5-flash-lite"] * 2
    assert no_sleep == [1.0]
    assert records[0]["attempts"] == 2
    assert records[0]["fallback"] is False


def test_fallback_to_other_tier(monkeypatch, no_sleep):
    busy = [_StatusError(429)] * 3
    calls = _fake_transport(monkeypatch, [*busy, '{"value": "from fallback"}'])

    with collect_llm_calls() as records:
        result = _extract(Tier.REASONING)

    assert result == Answer(value="from fallback")
    assert [c["model"] for c in calls] == ["gemini-3.6-flash"] * 3 + ["gemini-3.5-flash-lite"]
    assert records[0]["model"] == "gemini-3.5-flash-lite"
    assert records[0]["fallback"] is True
    assert records[0]["ok"] is True


def test_all_attempts_exhausted_returns_none(monkeypatch, no_sleep):
    _fake_transport(monkeypatch, [_StatusError(503)] * 5)

    with collect_llm_calls() as records:
        result = _extract()

    assert result is None
    assert records[0]["ok"] is False


def test_repair_retry_then_none(monkeypatch, no_sleep):
    calls = _fake_transport(monkeypatch, ['{"wrong": 1}', "not json at all"])

    with collect_llm_calls() as records:
        result = _extract()

    assert result is None
    assert len(calls) == 2
    assert "failed validation" in calls[1]["user_prompt"]
    assert records[0]["ok"] is False


def test_repair_retry_succeeds(monkeypatch, no_sleep):
    _fake_transport(monkeypatch, ['{"wrong": 1}', '{"value": "fixed"}'])

    assert _extract() == Answer(value="fixed")


def test_non_retryable_returns_none(monkeypatch, no_sleep):
    calls = _fake_transport(monkeypatch, [_StatusError(400)])

    assert _extract() is None
    assert len(calls) == 1
    assert no_sleep == []


def test_contextvar_override_beats_env(monkeypatch, no_sleep):
    calls = _fake_transport(monkeypatch, ['{"value": "ok"}'])
    byok = LLMSettings(
        provider="openai_compatible",
        api_key="visitor-key",
        model_fast="llama-fast",
        model_reasoning="llama-big",
        base_url="https://api.example.test/v1",
    )

    with use_llm_settings(byok):
        assert current_settings() == byok
        _extract()

    assert calls[0]["settings"].api_key == "visitor-key"
    assert calls[0]["model"] == "llama-fast"
    assert current_settings().api_key == "test-gemini-key"


def test_use_llm_settings_none_is_noop():
    with use_llm_settings(None):
        assert current_settings().provider == "gemini"


def test_env_model_overrides(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_FAST", "custom-fast")
    monkeypatch.setenv("LLM_MODEL_REASONING", "custom-reasoning")

    settings = current_settings()

    assert settings.model_fast == "custom-fast"
    assert settings.model_reasoning == "custom-reasoning"


def test_openai_compatible_requires_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.test/v1")

    with pytest.raises(EnvironmentError, match="LLM_MODEL_FAST"):
        current_settings()


def test_openai_compatible_reasoning_defaults_to_fast(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL_FAST", "deepseek-v4-flash")

    settings = current_settings()

    assert settings.model_reasoning == "deepseek-v4-flash"


def test_missing_gemini_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")

    with pytest.raises(EnvironmentError, match="GEMINI_API_KEY"):
        current_settings()


def test_missing_key_makes_extract_return_none(monkeypatch, no_sleep):
    monkeypatch.delenv("GEMINI_API_KEY")

    assert _extract() is None


def test_collect_calls_isolated(monkeypatch, no_sleep):
    _fake_transport(monkeypatch, ['{"value": "a"}', '{"value": "b"}'])

    with collect_llm_calls() as outer:
        _extract()
        with collect_llm_calls() as inner:
            _extract()

    assert len(outer) == 1
    assert len(inner) == 1


def test_calls_outside_collector_are_not_recorded(monkeypatch, no_sleep):
    _fake_transport(monkeypatch, ['{"value": "a"}'])

    assert _extract() == Answer(value="a")


def test_settings_repr_hides_key():
    settings = LLMSettings(
        provider="gemini", api_key="super-secret", model_fast="a", model_reasoning="b"
    )

    assert "super-secret" not in repr(settings)


class ReadTimeout(Exception):
    """Same class name httpx uses for a timed-out request."""


def test_timeout_is_retryable(monkeypatch, no_sleep):
    _fake_transport(monkeypatch, [ReadTimeout("timed out"), '{"value": "ok"}'])

    assert _extract() == Answer(value="ok")


def test_gemini_transport_sets_timeout(monkeypatch):
    seen: dict = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            class R:
                text = '{"value": "ok"}'
            return R()

    class FakeClient:
        def __init__(self, api_key, http_options=None):
            seen["timeout"] = getattr(http_options, "timeout", None)
            self.models = FakeModels()

    import google.genai

    monkeypatch.setattr(google.genai, "Client", FakeClient)

    assert _extract() == Answer(value="ok")
    assert seen["timeout"] == llm_client._REQUEST_TIMEOUT_S * 1000
