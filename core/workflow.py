"""The LangGraph pipeline: eleven nodes, with the three research nodes running in parallel.

    orchestrate → infer_engagement → engagement_context → plan
        → research_company | research_signals | research_person
        → audience_context → hypothesize → strategize → compose

Node ids and edges come from ``core/pipeline.py``, which the API also serves to the UI.
Every node is wrapped by ``instrument()``, which attaches the LLM calls it made (model,
tier, latency, fallback) to its state update.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph

from agents import (
    audience_context_agent,
    brief_composer,
    company_research,
    engagement_context_agent,
    engagement_inference_agent,
    external_signals,
    meeting_strategy_agent,
    orchestrator,
    pain_point_agent,
    person_research,
    research_planner,
)
from core.pipeline import PIPELINE_EDGES, PIPELINE_NODES
from core.state import BriefingState
from services.llm_client import collect_llm_calls

NodeFn = Callable[[dict[str, Any]], dict[str, Any]]

NODE_FUNCTIONS: dict[str, NodeFn] = {
    "orchestrate": orchestrator.run,
    "infer_engagement": engagement_inference_agent.run,
    "engagement_context": engagement_context_agent.run,
    "plan": research_planner.run,
    "research_company": company_research.run,
    "research_signals": external_signals.run,
    "research_person": person_research.run,
    "audience_context": audience_context_agent.run,
    "hypothesize": pain_point_agent.run,
    "strategize": meeting_strategy_agent.run,
    "compose": brief_composer.run,
}


def instrument(node_id: str, fn: NodeFn) -> NodeFn:
    """Wrap a node so the LLM calls it makes are attached to its state update."""

    def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        with collect_llm_calls() as calls:
            update = fn(state)
        if calls:
            update = {**update, "llm_calls": [{**call, "node": node_id} for call in calls]}
        return update

    wrapped.__name__ = f"{node_id}_instrumented"
    return wrapped


def build_graph() -> StateGraph:
    """The uncompiled graph. ``audience_context`` waits for all three research nodes."""
    graph = StateGraph(BriefingState)

    for node in PIPELINE_NODES:
        graph.add_node(node["id"], instrument(node["id"], NODE_FUNCTIONS[node["id"]]))

    graph.set_entry_point(PIPELINE_NODES[0]["id"])
    for source, target in PIPELINE_EDGES:
        graph.add_edge(source, target)
    graph.add_edge(PIPELINE_NODES[-1]["id"], END)

    return graph


app = build_graph().compile()
