"""Meeting strategy: the opening, talking points, questions, objections and next step.

Everything earlier in the pipeline feeds this node. The engagement stage drives the shape
of the strategy (a first intro and a negotiation need different questions and next steps),
and so does the user's own goal when they gave one.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agents.person_research import has_reliable_match
from agents.prompt_sections import (
    audience_lines,
    company_profile_lines,
    person_profile_lines,
    signals_section,
    user_side_lines,
)
from services.llm_client import Tier, structured_extract


class MeetingStrategy(BaseModel):
    """Structured meeting strategy for the upcoming stakeholder meeting."""

    executive_summary: str = Field(
        description=(
            "A concise (3-5 sentence) executive summary of the company situation, "
            "key findings, and what the meeting host should know going in.  "
            "This will appear at the top of the final brief."
        )
    )
    opening_angle: str = Field(
        description=(
            "A specific, credible way to open the meeting conversation.  "
            "Should reference a recent signal, company context, or audience "
            "priority — never a generic icebreaker."
        )
    )
    talking_points: list[str] = Field(
        default_factory=list,
        description=(
            "3-6 company-specific and audience-specific talking points "
            "for the meeting.  Each should be actionable and grounded "
            "in the research — not generic advice."
        ),
    )
    recommended_questions: list[str] = Field(
        default_factory=list,
        description=(
            "4-8 questions to ask during the meeting.  Each must be "
            "company-specific and audience-specific — grounded in the "
            "research findings, NOT generic discovery questions."
        ),
    )
    possible_objections: list[str] = Field(
        default_factory=list,
        description=(
            "1-4 objections or concerns the stakeholder might raise, "
            "along with brief guidance on how to address each.  "
            "Empty list if no obvious objections are apparent."
        ),
    )
    suggested_next_step: str = Field(
        description=(
            "A concrete next step to propose at the end of the meeting.  "
            "Should be realistic and appropriate for the meeting type."
        )
    )


_SYSTEM_PROMPT = """\
You are a meeting-preparation strategist building a pre-meeting brief.
You will receive structured research about a company, its recent signals,
audience context, engagement context, pain-point hypotheses, and
optionally a person profile.

Your task: turn this analysis into practical, actionable meeting strategy.

STRICT RULES:
1. EXECUTIVE SUMMARY: 3-5 sentences covering the company situation,
   key findings, and what the host needs to know.  This gets inserted
   at the top of the final brief — make it count.
2. OPENING ANGLE: reference something specific — a recent signal,
   a known priority, a company milestone.  Never a generic opener.
   The engagement stage matters: a first-intro opener differs from a
   follow-up or negotiation opener.
3. TALKING POINTS: 3-6 items.  Each must be grounded in the research.
   Tailor to the audience function, seniority, and ENGAGEMENT STAGE.
4. RECOMMENDED QUESTIONS: 4-8 items.  Each must be company-specific
   AND audience-specific.  Do NOT produce generic questions like
   "What are your biggest challenges?" — instead, reference specific
   signals, priorities, or pain points from the research.
   Calibrate question depth to the engagement stage: exploratory
   questions for early stages, solution-validation questions for later.
5. POSSIBLE OBJECTIONS: 1-4 items.  What might the stakeholder push
   back on?  Use the engagement context's decision_risks as seeds.
   Include brief guidance on handling each.  Empty list is fine.
6. SUGGESTED NEXT STEP: one concrete action appropriate for the
   meeting type AND engagement stage.  Match the next-step pressure:
   low-pressure stages → soft next steps (follow-up call, share info);
   high-pressure stages → commercial next steps (sign-off, contract).
7. ENGAGEMENT CONTEXT is a primary driver of your strategy.  The
   conversation_mode, recommended_tone, and stage_objective should
   shape the entire strategy — not just individual fields.
8. If person-profile details are included, use them to further tailor
   the strategy.  If absent, reason from company and audience context.
9. Prefer depth over breadth — fewer well-grounded items beat many
   generic ones.
10. USER-SIDE CONTEXT (if provided) is a PRIMARY DRIVER of strategy.
    It tells you WHO the meeting host is, WHAT they want, and HOW
    to frame the entire strategy:
    - user_role / user_company / user_function → shape credibility angle
    - engagement_type → set the strategic frame:
      * sales → qualify, handle objections, push for next commercial step
      * consulting → discover, hypothesise, propose diagnostic
      * interview → assess fit, understand team, show alignment
      * partnership → find strategic overlap, propose joint value
      * investing → assess risk, probe unit economics, evaluate team
    - desired_outcome / success_definition → calibrate the suggested
      next step and shape what "success" looks like in the strategy
    - meeting_goal → already used; desired_outcome is more specific
    If user-side context is absent, let meeting_type and the engagement
    context drive the strategy.
"""


def empty_strategy() -> dict[str, Any]:
    return MeetingStrategy(executive_summary="", opening_angle="", suggested_next_step="").model_dump()


def _build_user_prompt(state: dict[str, Any]) -> str:
    parts = [
        "=== Meeting Context ===",
        f"Company: {state.get('company_name', 'unknown')}",
        f"Meeting type: {state.get('meeting_type', '')}",
        f"Stakeholder role: {state.get('stakeholder_role', '')}",
        f"Meeting goal: {state.get('meeting_goal', '')}",
    ]

    user_side = user_side_lines(state)
    if user_side:
        parts += ["", "=== User-Side Context (strategy driver) ===", *user_side]

    engagement = state.get("engagement_context", {})
    if engagement.get("interaction_stage"):
        parts += [
            "",
            "=== Engagement Context ===",
            f"Interaction stage: {engagement['interaction_stage']}",
            f"Relationship status: {engagement.get('relationship_status') or 'unknown'}",
            f"Conversation mode: {engagement.get('conversation_mode', 'exploratory')}",
            f"Stage objective: {engagement.get('stage_objective', '')}",
            f"Recommended tone: {engagement.get('recommended_tone', '')}",
            f"Next-step pressure: {engagement.get('next_step_pressure', 'low')}",
        ]
        if engagement.get("decision_risks"):
            parts.append(f"Decision risks: {'; '.join(engagement['decision_risks'])}")
        if engagement.get("continuity_context"):
            parts.append(f"Continuity context: {engagement['continuity_context']}")

    company = state.get("company_profile", {})
    if any(company.get(key) for key in ("industry", "business_model", "company_positioning")):
        parts += ["", "=== Company Profile ===", *company_profile_lines(company)]

    parts += signals_section(state.get("external_signals", []))

    audience = state.get("audience_context", {})
    if audience.get("department"):
        parts += ["", "=== Audience Context ===", *audience_lines(audience, with_style=True)]

    pain_points = state.get("pain_point_hypotheses", {})
    if pain_points.get("hypotheses"):
        parts += ["", "=== Pain Point Hypotheses ==="]
        for number, item in enumerate(pain_points["hypotheses"][:8], 1):
            evidence = "; ".join(item.get("supporting_evidence", []))
            parts.append(
                f"{number}. [{item.get('confidence', '?')}] {item.get('hypothesis', '')}  "
                f"Evidence: {evidence or 'n/a'}"
            )
    if pain_points.get("items_to_validate"):
        parts.append(f"Items to validate: {'; '.join(pain_points['items_to_validate'][:6])}")

    person = state.get("person_profile", {})
    if has_reliable_match(person):
        parts += ["", "=== Person Profile (reliable — use to personalise) ===", *person_profile_lines(person)]
    else:
        parts += ["", "No reliable person profile — tailor strategy to company and audience context only."]
    return "\n".join(parts)


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Turn the analysis into ``meeting_strategy``."""
    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(state),
        response_model=MeetingStrategy,
        tier=Tier.REASONING,
    )
    if result is None:
        return {
            "meeting_strategy": empty_strategy(),
            "errors": ["meeting_strategy_agent: LLM reasoning failed, returning empty strategy."],
        }
    return {"meeting_strategy": result.model_dump()}
