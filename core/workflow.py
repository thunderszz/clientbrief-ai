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

Cache-aware routing
-------------------
Not implemented yet.  Every agent handles preloaded state gracefully.
Can be added via ``add_conditional_edges`` if profiling shows a need.

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

from langgraph.graph import StateGraph, END

from core.state import BriefingState
from agents import (
    orchestrator,
    research_planner,
    company_research,
    external_signals,
    person_research,
    audience_context_agent,
    engagement_context_agent,
    engagement_inference_agent,
    pain_point_agent,
    meeting_strategy_agent,
    brief_composer,
)


def build_graph() -> StateGraph:
    """Construct the ClientBrief AI LangGraph pipeline.

    Returns the un-compiled ``StateGraph`` so callers can inspect or
    extend it before compiling.
    """
    graph = StateGraph(BriefingState)

    # --- Register nodes (11 total) --------------------------------------------
    graph.add_node("orchestrate", orchestrator.run)
    graph.add_node("infer_engagement", engagement_inference_agent.run)
    graph.add_node("engagement_context", engagement_context_agent.run)
    graph.add_node("plan", research_planner.run)
    graph.add_node("research_company", company_research.run)
    graph.add_node("research_signals", external_signals.run)
    graph.add_node("research_person", person_research.run)
    graph.add_node("audience_context", audience_context_agent.run)
    graph.add_node("hypothesize", pain_point_agent.run)
    graph.add_node("strategize", meeting_strategy_agent.run)
    graph.add_node("compose", brief_composer.run)

    # --- Entry point ----------------------------------------------------------
    graph.set_entry_point("orchestrate")

    # --- Engagement-first chain (before planning) -----------------------------
    # Engagement inference + context resolve before planning so the planner
    # can use research_mode and engagement_context to calibrate depth.
    graph.add_edge("orchestrate", "infer_engagement")
    graph.add_edge("infer_engagement", "engagement_context")
    graph.add_edge("engagement_context", "plan")

    # --- Plan → three research agents (fan-out) -------------------------------
    # LangGraph executes nodes with all inbound edges satisfied.
    # Adding three edges from "plan" to distinct nodes triggers parallel
    # execution when the graph is compiled with the default executor.
    graph.add_edge("plan", "research_company")
    graph.add_edge("plan", "research_signals")
    graph.add_edge("plan", "research_person")

    # --- Three research agents → audience_context (fan-in) --------------------
    # audience_context starts only after all three research nodes complete.
    graph.add_edge("research_company", "audience_context")
    graph.add_edge("research_signals", "audience_context")
    graph.add_edge("research_person", "audience_context")

    # --- Sequential reasoning chain -------------------------------------------
    graph.add_edge("audience_context", "hypothesize")
    graph.add_edge("hypothesize", "strategize")
    graph.add_edge("strategize", "compose")

    # --- Compose → END --------------------------------------------------------
    graph.add_edge("compose", END)

    return graph


# Pre-compiled application — importable by ``from core.workflow import app``.
app = build_graph().compile()
