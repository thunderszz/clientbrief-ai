"""LangGraph workflow — wires all agents into a stateful pipeline.

Graph structure (v2 — engagement-first):

    orchestrate → infer_engagement → engagement_context → plan
        → [research_company, research_signals, research_person]
              ↓ (fan-in)
         audience_context → hypothesize → strategize → compose → END

Key changes from v1:
- Engagement inference and context resolution now happen BEFORE planning.
- The planner reads ``engagement_context`` (including ``research_mode``)
  to calibrate research depth and focus.
- The three research agents still run in parallel (fan-out from ``plan``,
  fan-in to ``audience_context``).
- ``audience_context`` can read ``engagement_context`` from shared state
  since it was written in an earlier step.

The node ids and edges come from ``core/pipeline.py`` (shared with the API
and UI). Every node is wrapped by ``instrument()``, which records the LLM
calls it made (model, tier, latency, fallback) into ``state["llm_calls"]``.

Usage::

    from core.workflow import app

    result = app.invoke({
        "company_name": "Acme Corp",
        "domain": "acme.com",
        "meeting_type": "sales_discovery",
        "stakeholder_role": "VP Finance",
        "meeting_goal": "Follow up on our pricing discussion from last week",
        # Engagement fields are optional — inference will fill gaps
    })

    print(result["final_brief"])
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
    """Construct the ClientBrief AI LangGraph pipeline.

    Returns the un-compiled ``StateGraph`` so callers can inspect or
    extend it before compiling.

    Fan-out from ``plan`` to the three research nodes runs them in parallel;
    ``audience_context`` waits for all three (fan-in).
    """
    graph = StateGraph(BriefingState)

    for node in PIPELINE_NODES:
        graph.add_node(node["id"], instrument(node["id"], NODE_FUNCTIONS[node["id"]]))

    graph.set_entry_point(PIPELINE_NODES[0]["id"])
    for source, target in PIPELINE_EDGES:
        graph.add_edge(source, target)
    graph.add_edge(PIPELINE_NODES[-1]["id"], END)

    return graph


# Pre-compiled application — importable by ``from core.workflow import app``.
app = build_graph().compile()
