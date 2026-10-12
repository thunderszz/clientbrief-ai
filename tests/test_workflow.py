"""Graph wiring, per-node instrumentation and an offline end-to-end run."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from core.pipeline import NODE_LABELS, PIPELINE_EDGES, PIPELINE_NODES
from core.workflow import build_graph, instrument
from services import llm_client, tavily_client
from services.llm_client import structured_extract


class _Answer(BaseModel):
    value: str


def test_pipeline_metadata_is_consistent():
    ids = [n["id"] for n in PIPELINE_NODES]

    assert len(ids) == len(set(ids)) == 11
    assert {n["phase"] for n in PIPELINE_NODES} == {"setup", "research", "reasoning", "output"}
    assert {n["kind"] for n in PIPELINE_NODES} <= {"deterministic", "llm", "search+llm"}
    for source, target in PIPELINE_EDGES:
        assert source in ids and target in ids
    assert NODE_LABELS["compose"] == "Compose brief"


def test_graph_matches_pipeline_metadata():
    graph = build_graph().compile().get_graph()

    nodes = {n for n in graph.nodes if not n.startswith("__")}
    edges = {
        (e.source, e.target)
        for e in graph.edges
        if not e.source.startswith("__") and not e.target.startswith("__")
    }

    assert nodes == {n["id"] for n in PIPELINE_NODES}
    assert edges == set(PIPELINE_EDGES)


def test_instrument_tags_llm_calls_with_node(monkeypatch):
    monkeypatch.setattr(llm_client, "_transport", lambda *a: '{"value": "ok"}')

    def node(state):
        structured_extract(system_prompt="s", user_prompt="u", response_model=_Answer)
        return {"company_profile": {"industry": "x"}}

    out = instrument("research_company", node)({})

    assert out["company_profile"] == {"industry": "x"}
    assert len(out["llm_calls"]) == 1
    assert out["llm_calls"][0]["node"] == "research_company"
    assert out["llm_calls"][0]["ok"] is True


def test_instrument_leaves_llm_free_nodes_untouched():
    out = instrument("plan", lambda state: {"research_plan": {}})({})

    assert out == {"research_plan": {}}


class _EmptyTavily:
    def __init__(self, api_key):
        pass

    def search(self, query, **kwargs):
        return {"results": []}


@pytest.fixture
def offline_services(monkeypatch):
    """Every LLM call fails (non-retryable) and every search comes back empty."""

    def failing_transport(*args):
        raise ValueError("offline")

    monkeypatch.setattr(llm_client, "_transport", failing_transport)
    monkeypatch.setattr(tavily_client, "_TavilyClient", _EmptyTavily)


def test_pipeline_completes_offline_when_everything_fails(offline_services):
    app = build_graph().compile()

    state = app.invoke(
        {
            "company_name": "Acme",
            "domain": "acme.com",
            "meeting_type": "sales_discovery",
            "stakeholder_role": "CFO",
            "meeting_goal": "Discuss finance automation",
            "contact_name": "Jane Doe",
        }
    )

    assert state["final_brief"].startswith("# Pre-Meeting Brief: Acme")
    assert state["reliability_report"]["score"] == 0
    assert state["errors"]
    tagged = {call["node"] for call in state["llm_calls"]}
    assert "infer_engagement" in tagged
    assert "hypothesize" in tagged
    assert all(call["ok"] is False for call in state["llm_calls"])
