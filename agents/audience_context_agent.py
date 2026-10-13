"""Audience context: what this stakeholder is likely to care about, and how to talk to them.

The model reasons from the stakeholder's function and seniority first, then sharpens that
with the company profile and the engagement stage. Details about the contact are used only
when the contact lookup found a reliable match.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agents.person_research import has_reliable_match
from services.llm_client import Tier, structured_extract


class AudienceContext(BaseModel):
    """Structured audience profile for the meeting stakeholder."""

    department: str = Field(
        description="The business function or department (e.g. 'Finance', 'Engineering')."
    )
    role_seniority: str | None = Field(
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


# Typical priorities for each function, given to the model as a starting point.
_FUNCTION_PRIORITIES: dict[str, list[str]] = {
    "finance": [
        "financial visibility",
        "forecasting accuracy",
        "margin control",
        "cost optimisation",
        "compliance",
    ],
    "accounting": [
        "close-process efficiency",
        "reconciliation accuracy",
        "internal controls",
        "regulatory compliance",
    ],
    "engineering": [
        "scalability",
        "system reliability",
        "technical debt reduction",
        "developer productivity",
    ],
    "product": [
        "roadmap execution",
        "feature prioritisation",
        "product analytics",
        "user retention",
    ],
    "sales": ["pipeline health", "conversion rates", "revenue predictability", "sales enablement"],
    "strategy": [
        "growth trajectory",
        "market expansion",
        "competitive positioning",
        "M&A readiness",
    ],
    "it_data": ["system integration", "data governance", "data quality", "security posture"],
    "operations": [
        "process efficiency",
        "execution visibility",
        "cross-functional coordination",
        "supply chain resilience",
    ],
    "transformation": [
        "change management effectiveness",
        "adoption metrics",
        "rollout governance",
        "stakeholder alignment",
    ],
    "general": ["business performance", "strategic alignment", "operational efficiency"],
}


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


def empty_context() -> dict[str, Any]:
    return AudienceContext(department="", communication_style="").model_dump()


def _build_user_prompt(state: dict[str, Any]) -> str:
    function = state.get("normalized_function", "general")
    seeds = _FUNCTION_PRIORITIES.get(function, _FUNCTION_PRIORITIES["general"])
    parts = [
        f"Stakeholder role: {state.get('stakeholder_role', 'unknown')}",
        f"Contact title: {state.get('contact_title') or 'not provided'}",
        f"Normalised function: {function}",
        f"Normalised seniority: {state.get('normalized_seniority') or 'unknown'}",
        f"Meeting type: {state.get('meeting_type', '')}",
        f"Meeting goal: {state.get('meeting_goal', '')}",
        f"Function-based priority seeds: {', '.join(seeds)}",
    ]

    engagement = state.get("engagement_context", {})
    if engagement.get("interaction_stage"):
        parts += [
            f"Interaction stage: {engagement['interaction_stage']}",
            f"Conversation mode: {engagement.get('conversation_mode', 'exploratory')}",
            f"Stage objective: {engagement.get('stage_objective', '')}",
            f"Recommended tone: {engagement.get('recommended_tone', '')}",
        ]
        if engagement.get("continuity_context"):
            parts.append(f"Continuity context: {engagement['continuity_context']}")

    company = state.get("company_profile", {})
    if any(company.get(key) for key in ("industry", "business_model", "products")):
        parts += [
            f"Company industry: {company.get('industry') or 'unknown'}",
            f"Business model: {company.get('business_model') or 'unknown'}",
        ]
        if company.get("products"):
            parts.append(f"Products/services: {', '.join(company['products'][:5])}")
        if company.get("complexity_signals"):
            parts.append(f"Complexity signals: {', '.join(company['complexity_signals'][:5])}")

    person = state.get("person_profile", {})
    parts.append("")
    if has_reliable_match(person):
        parts.append("--- Reliable person profile (use to personalise) ---")
        for key, label in (
            ("title", "Title"),
            ("department", "Department"),
            ("background_summary", "Background"),
        ):
            if person.get(key):
                parts.append(f"{label}: {person[key]}")
        if person.get("public_signals"):
            parts.append(f"Public signals: {', '.join(person['public_signals'][:5])}")
        parts.append("--- End person profile ---")
    else:
        parts.append(
            "No reliable person profile available — reason from "
            "function, seniority, and company context only."
        )
    return "\n".join(parts)


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Profile the audience and return ``audience_context``."""
    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(state),
        response_model=AudienceContext,
        tier=Tier.REASONING,
    )
    if result is None:
        return {
            "audience_context": empty_context(),
            "errors": ["audience_context_agent: LLM reasoning failed, returning sparse context."],
        }
    return {"audience_context": result.model_dump()}
