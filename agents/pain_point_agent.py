"""Pain points: what the stakeholder is likely struggling with, kept apart by certainty.

Facts must come from the research, inferences must follow from facts, and hypotheses are
guesses to test in the meeting, each with its evidence and a confidence level. The user's
own context, when given, changes the angle of the analysis but never counts as evidence.
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


class Hypothesis(BaseModel):
    """A single pain-point hypothesis."""

    hypothesis: str = Field(
        description="A plausible business challenge the stakeholder may face."
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Facts or inferences that support this hypothesis.",
    )
    confidence: str = Field(
        description="'high', 'medium', or 'low' — how well-supported is this hypothesis?"
    )


class PainPointHypotheses(BaseModel):
    """Structured pain-point analysis with explicit fact/inference/hypothesis separation."""

    facts: list[str] = Field(
        default_factory=list,
        description=(
            "Confirmed, source-grounded facts extracted from the research.  "
            "Each fact must be verifiable from the evidence provided."
        ),
    )
    inferences: list[str] = Field(
        default_factory=list,
        description=(
            "Reasoned conclusions that follow logically from the facts.  "
            "Each inference must cite which fact(s) it derives from."
        ),
    )
    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description=(
            "Speculative but plausible pain points.  Each hypothesis must "
            "include supporting evidence and a confidence level.  "
            "Be cautious — do NOT produce overconfident hypotheses."
        ),
    )
    items_to_validate: list[str] = Field(
        default_factory=list,
        description=(
            "Questions or assumptions the user should verify during the "
            "meeting to confirm or refute the hypotheses."
        ),
    )


_SYSTEM_PROMPT = """\
You are a business-challenge analyst preparing a pre-meeting brief.
You will receive research about a company, recent strategic signals,
audience context, engagement context, and optionally a person profile.

Your task: synthesise this into a structured pain-point analysis.

STRICT RULES:
1. FACTS must be directly supported by the evidence — no invention.
2. INFERENCES must follow from facts.  State which fact each inference
   derives from (e.g. "Given [fact], it follows that…").
3. HYPOTHESES are speculative.  Each must include:
   - a clear hypothesis statement
   - the supporting evidence that motivates it
   - a confidence level (high / medium / low)
   Do NOT produce high-confidence hypotheses unless the evidence is strong.
4. ITEMS TO VALIDATE are questions the meeting host should ask to confirm
   or refute the hypotheses.
5. If evidence is sparse, produce fewer items.  Sparse is better than
   fabricated.
6. If person-profile details are included, you may use them.  If they
   are absent, reason only from company and function context.
7. Prefer function/department reasoning over title-only reasoning.
8. ENGAGEMENT CONTEXT matters: tailor pain-point framing to the
   interaction stage.  For exploratory meetings, surface broad business
   challenges.  For solution-oriented or commercial meetings, focus on
   pain points that a solution could address or that might block a deal.
   For client/expansion meetings, focus on unmet needs and growth friction.
9. USER-SIDE CONTEXT (if provided) shapes the LENS of your analysis.
   It does NOT provide evidence — facts must still come from research.
   But it tells you how to frame and prioritise hypotheses:
   - consulting engagement → emphasise process gaps, transformation needs,
     diagnostic-style framing
   - sales engagement → emphasise commercially relevant pain, urgency,
     qualification signals, budget/authority indicators
   - interview / candidate → emphasise role context, team challenges,
     what the hiring function cares about — NOT commercial pain
   - partnership → emphasise strategic fit, complementary gaps
   - investing → emphasise market risk, growth friction, competitive threats
   If user-side context is absent, ignore this rule and frame generically.
"""


def empty_hypotheses() -> dict[str, Any]:
    return PainPointHypotheses().model_dump()


def _build_user_prompt(state: dict[str, Any]) -> str:
    parts = ["=== Company Profile ===", *company_profile_lines(state.get("company_profile", {}))]
    parts += signals_section(state.get("external_signals", []))

    audience = state.get("audience_context", {})
    if audience.get("department"):
        parts += ["", "=== Audience Context ===", *audience_lines(audience)]

    engagement = state.get("engagement_context", {})
    if engagement.get("interaction_stage"):
        parts += [
            "",
            "=== Engagement Context ===",
            f"Interaction stage: {engagement['interaction_stage']}",
            f"Conversation mode: {engagement.get('conversation_mode', 'exploratory')}",
            f"Stage objective: {engagement.get('stage_objective', '')}",
        ]
        if engagement.get("decision_risks"):
            parts.append(f"Decision risks: {'; '.join(engagement['decision_risks'])}")
        if engagement.get("continuity_context"):
            parts.append(f"Continuity context: {engagement['continuity_context']}")

    person = state.get("person_profile", {})
    if has_reliable_match(person):
        parts += ["", "=== Person Profile (reliable — may use) ===", *person_profile_lines(person)]
    else:
        parts += ["", "No reliable person profile — reason from company and function context only."]

    user_side = user_side_lines(state, with_goal=True)
    if user_side:
        parts += ["", "=== User-Side Context (framing lens — NOT evidence) ===", *user_side]
    return "\n".join(parts)


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Synthesise the research into ``pain_point_hypotheses``."""
    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_build_user_prompt(state),
        response_model=PainPointHypotheses,
        tier=Tier.REASONING,
    )
    if result is None:
        return {
            "pain_point_hypotheses": empty_hypotheses(),
            "errors": ["pain_point_agent: LLM reasoning failed, returning empty hypotheses."],
        }
    return {"pain_point_hypotheses": result.model_dump()}
