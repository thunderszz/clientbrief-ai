"""Shared test fixtures. Tests are fully offline: no network, no real keys."""

import os

import pytest

# The API loads .env on import for local dev; tests must never see real keys.
os.environ["CLIENTBRIEF_SKIP_DOTENV"] = "1"

_CLEARED_ENV = [
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL_FAST",
    "LLM_MODEL_REASONING",
    "GEMINI_MODEL",
    "PUBLIC_DAILY_RUNS",
]


@pytest.fixture(autouse=True)
def _offline_env(monkeypatch):
    """Give every test dummy keys and a clean LLM configuration."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    for name in _CLEARED_ENV:
        monkeypatch.delenv(name, raising=False)
