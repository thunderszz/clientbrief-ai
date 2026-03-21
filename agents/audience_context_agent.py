"""Audience Context Agent — LLM reasoning plus deterministic role context.

Builds an audience profile grounded in:
- normalized function and seniority (primary anchor)
- company profile (industry, complexity)
- meeting type and goal
- engagement context (interaction stage, conversation mode, tone)
- person profile — **only** when match_confidence is "medium" or "high"

If person-level evidence is weak or absent, the agent falls back entirely
to function/seniority/meeting-context reasoning.

Ownership:  ``audience_context`` only.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from services.llm_client import structured_extract

logger = logging.getLogger(__name__)

# Confidence values that permit using person-level details.
# See CLAUDE.md rule 4 / PROJECT_SPEC §5.
_RELIABLE_CONFIDENCE = frozenset({"medium", "high"})


# ---------------------------------------------------------------------------
# Pydantic schema for LLM output
# ---------------------------------------------------------------------------

class AudienceContext(BaseModel):
    """Structured audience profile for the meeting stakeholder."""

    department: str = Field(
        description="The business function or department (e.g. 'Finance', 'Engineering')."
    )
    role_seniority: Optional[str] = Field(
        None,
        description="Seniority level (e.g. 'C-level', 'VP', 'Director', 'Manager').  Null if unknown.",
    )
    likely_priorities: list[str] = Field(
        default_factory=list,
        description="3-6 priorities this audience likely cares about, grounded in their function and company context.",
    )
    relevant_topics: list[str] = Field(
        default_factory=list,
        description="Specific topics relevant to the upcoming meeting, derived from function + meeting goal.",
    )
    communication_style: str = Field(
        "",
        description="Recommended communication approach (e.g. 'data-driven and concise', 'strategic and big-picture').",
    )
    potential_sensitivities: list[str] = Field(
        default_factory=list,
        description="Topics or angles that might be sensitive for this audience.",
    )


# ---------------------------------------------------------------------------
# Function-based priority seeds (deterministic, per spec §6)
# ---------------------------------------------------------------------------

_FUNCTION_PRIORITIES: dict[str, list[str]] = {
    "finance": ["financial visibility", "forecasting accuracy", "margin control", "cost optimisation", "compliance"],
    "accounting": ["close-process efficiency", "reconciliation accuracy", "internal controls", "regulatory compliance"],
    "engineering": ["scalability", "system reliability", "technical debt reduction", "developer productivity"],
    "product": ["roadmap execution", "feature prioritisation", "product analytics", "user retention"],
    "sales": ["pipeline health", "conversion rates", "revenue predictability", "sales enablement"],
    "strategy": ["growth trajectory", "market expansion", "competitive positioning", "M&A readiness"],
    "it_data": ["system integration", "data governance", "data quality", "security posture"],
    "operations": ["process efficiency", "execution visibility", "cross-functional coordination", "supply chain resilience"],
    "transformation": ["change management effectiveness", "adoption metrics", "rollout governance", "stakeholder alignment"],
    "general": ["business performance", "strategic alignment", "operational efficiency"],
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an audience-analysis expert preparing a meeting brief.  You will
receive contextual information about a meeting stakeholder, the company,
and the engagement stage (where we are in the relationship lifecycle).

Your task: build a structured audience profile that will help the meeting
host tailor their conversation.

Rules:
- Anchor your reasoning in the stakeholder's FUNCTION and SENIORITY first,
  not just their title.
- Use company context (industry, complexity, recent signals) to sharpen
  priorities and topics.
- Factor in the ENGAGEMENT CONTEXT: the conversation mode, interaction
  stage, and recommended tone should influence which priorities and
  sensitivities you surface.  For example, a first_intro meeting should
  surface broad exploratory priorities, while a negotiation meeting should
  surface deal-specific concerns.
- If person-level details are provided, incorporate them to personalise
  the profile.  If not, reason purely from function/seniority/company.
- Do NOT hallucinate personal attributes or invent biographical details.
- Keep likely_priorities to 3-6 items, ordered by importance.
- Keep relevant_topics specific to the meeting goal — not generic.
- communication_style should be a practical recommendation (1-2 sentences).
- potential_sensitivities: real concerns this audience might have about
  the meeting topic (0-3 items).  Empty list if none are apparent.
"""


def _empty_context() -> dict[str, Any]:
    return AudienceContext(department="", communication_style="").model_dump()


# ---------------------------------------------------------------------------
# Build the user prompt with person-confidence gating
# ---------------------------------------------------------------------------

def _build_user_prompt(state: dict[str, Any]) -> str:
    """Assemble the user prompt, gating person details on confidence."""

    parts: list[str] = []

    parts.append(f"Stakeholder role: {state.get('stakeholder_role', 'unknown')}")
    parts.append(f"Contact title: {state.get('contact_title') or 'not provided'}")
    parts.append(f"Normalised function: {state.get('normalized_function', 'general')}")
    parts.append(f"Normalised seniority: {state.get('normalized_seniority') or 'unknown'}")
    parts.append(f"Meeting type: {state.get('meeting_type', '')}")
    parts.append(f"Meeting goal: {state.get('meeting_goal', '')}")

    # Function-based priority seeds (deterministic)
    fn = state.get("normalized_function", "general")
    seeds = _FUNCTION_PRIORITIES.get(fn, _FUNCTION_PRIORITIES["general"])
    parts.append(f"Function-based priority seeds: {', '.join(seeds)}")

    # Engagement context — stage, conversation mode, tone guidance
    ec: dict[str, Any] = state.get("engagement_context", {})
    if ec.get("interaction_stage"):
        parts.append(f"Interaction stage: {ec['interaction_stage']}")
        parts.append(f"Conversation mode: {ec.get('conversation_mode', 'exploratory')}")
        parts.append(f"Stage objective: {ec.get('stage_objective', '')}")
        parts.append(f"Recommended tone: {ec.get('recommended_tone', '')}")
        continuity = ec.get("continuity_context")
        if continuity:
            parts.append(f"Continuity context: {continuity}")

    # Company context
    cp: dict[str, Any] = state.get("company_profile", {})
    if any(cp.get(k) for k in ("industry", "business_model", "products")):
        parts.append(f"Company industry: {cp.get('industry') or 'unknown'}")
        parts.append(f"Business model: {cp.get('business_model') or 'unknown'}")
        products = cp.get("products", [])
        if products:
            parts.append(f"Products/services: {', '.join(products[:5])}")
        complexity = cp.get("complexity_signals", [])
        if complexity:
            parts.append(f"Complexity signals: {', '.join(complexity[:5])}")

    # Person profile — gated on confidence
    pp: dict[str, Any] = state.get("person_profile", {})
    confidence = pp.get("match_confidence", "none")

    if confidence in _RELIABLE_CONFIDENCE:
        parts.append("")
        parts.append("--- Reliable person profile (use to personalise) ---")
        if pp.get("title"):
            parts.append(f"Title: {pp['title']}")
        if pp.get("department"):
            parts.append(f"Department: {pp['department']}")
        if pp.get("background_summary"):
            parts.append(f"Background: {pp['background_summary']}")
        if pp.get("public_signals"):
            parts.append(f"Public signals: {', '.join(pp['public_signals'][:5])}")
        parts.append("--- End person profile ---")
    else:
        parts.append("")
        parts.append(
            "No reliable person profile available — reason from "
            "function, seniority, and company context only."
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run(state: dict[str, Any]) -> dict[str, Any]:
    """Produce audience context from role, company, and person evidence."""

    new_errors: list[str] = []

    user_prompt = _build_user_prompt(state)

    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=AudienceContext,
    )

    if result is None:
        new_errors.append("audience_context_agent: LLM reasoning failed, returning sparse context.")
        return {"audience_context": _empty_context(), "errors": new_errors}

    return {"audience_context": result.model_dump()}
