"""Research planner: what to look for, and how much, given the meeting and its stage.

Deterministic. The plan comes from a lookup table (``agents/research_plans.py``) keyed by
meeting type and the stakeholder's function, extended for later-stage meetings. Its search
queries add to the ones each research agent always runs; the research mode trims them for
follow-ups, where the basics are already known.
"""

from __future__ import annotations

from typing import Any

from agents.research_plans import DEFAULT_PLAN, PLANS, STAGE_OVERLAYS

# Extra words for the query that pairs the stakeholder's function with the meeting type.
_MEETING_FLAVOUR = {
    "consulting_intro": "challenges priorities strategy",
    "sales_discovery": "technology tools vendor evaluation",
    "account_review": "recent news updates changes",
}


def _trim_for_mode(queries: list[str], research_mode: str, company_name: str) -> list[str]:
    """``light`` keeps the two most targeted queries; ``update_only`` keeps one plus a news query."""
    if research_mode == "light":
        return queries[:2]
    if research_mode == "update_only":
        return [*queries[:1], f'"{company_name}" recent changes updates news']
    return queries


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Build ``research_plan`` from the meeting type, function and engagement context."""
    company_name = state.get("company_name", "")
    meeting_type = state.get("meeting_type", "")
    function = state.get("normalized_function", "general")
    engagement = state.get("engagement_context", {})
    research_mode = engagement.get("research_mode", "full")

    plan = PLANS.get((meeting_type, function)) or PLANS.get((meeting_type, "general")) or DEFAULT_PLAN
    overlay = STAGE_OVERLAYS.get(engagement.get("interaction_stage", "first_intro"), {})
    focus_areas = plan["focus_areas"] + overlay.get("extra_focus", [])

    queries = [f'"{company_name}" {area}' for area in focus_areas[:3]]
    if meeting_type in _MEETING_FLAVOUR:
        queries.append(f'"{company_name}" {function} {_MEETING_FLAVOUR[meeting_type]}')
    queries += [f'"{company_name}" {hint}' for hint in overlay.get("extra_query_hints", [])[:2]]

    return {
        "research_plan": {
            "research_objectives": plan["research_objectives"] + overlay.get("extra_objectives", []),
            "focus_areas": focus_areas,
            "evidence_needed": list(plan["evidence_needed"]),
            "tavily_queries": _trim_for_mode(queries, research_mode, company_name),
            "research_mode": research_mode,
        }
    }
