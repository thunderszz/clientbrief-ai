"""Tests for the Tavily wrapper (the SDK client is faked; no network)."""

from __future__ import annotations

import threading
import time

import pytest

from services import tavily_client
from services.tavily_client import TavilySearchClient, use_tavily_key


class FakeSDK:
    """Stands in for ``tavily.TavilyClient``."""

    instances: list[FakeSDK] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.threads: set[str] = set()
        FakeSDK.instances.append(self)

    def search(self, query: str, **kwargs):
        self.threads.add(threading.current_thread().name)
        if query == "boom":
            raise RuntimeError("upstream failure")
        time.sleep(0.02)
        return {"results": [{"title": query, "url": f"https://{query}.test", "content": "c"}]}


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    FakeSDK.instances.clear()
    monkeypatch.setattr(tavily_client, "_TavilyClient", FakeSDK)


def test_env_key_used_by_default():
    TavilySearchClient()

    assert FakeSDK.instances[-1].api_key == "test-tavily-key"


def test_override_beats_env():
    with use_tavily_key("visitor-key"):
        TavilySearchClient()

    assert FakeSDK.instances[-1].api_key == "visitor-key"


def test_override_none_is_noop():
    with use_tavily_key(None):
        TavilySearchClient()

    assert FakeSDK.instances[-1].api_key == "test-tavily-key"


def test_explicit_key_beats_override():
    with use_tavily_key("visitor-key"):
        TavilySearchClient(api_key="explicit")

    assert FakeSDK.instances[-1].api_key == "explicit"


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY")

    with pytest.raises(EnvironmentError, match="TAVILY_API_KEY"):
        TavilySearchClient()


def test_batch_preserves_order():
    queries = ["a", "b", "c", "d", "e", "f"]

    responses = TavilySearchClient().search_batch(queries)

    assert [r.query for r in responses] == queries
    assert [r.results[0].title for r in responses] == queries


def test_batch_isolates_failures():
    responses = TavilySearchClient().search_batch(["a", "boom", "c"])

    assert [r.ok for r in responses] == [True, False, True]
    assert "upstream failure" in responses[1].error


def test_batch_runs_concurrently():
    client = TavilySearchClient()

    client.search_batch(["a", "b", "c", "d"])

    assert len(FakeSDK.instances[-1].threads) > 1


def test_empty_batch():
    assert TavilySearchClient().search_batch([]) == []
