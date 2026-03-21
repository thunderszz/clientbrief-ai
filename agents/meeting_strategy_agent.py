"""Meeting Strategy Agent — LLM reasoning for practical meeting preparation.

Reads company_profile, external_signals, person_profile, audience_context,
engagement_context, and pain_point_hypotheses to produce a structured
meeting strategy with:
- **executive_summary** — concise overview for the meeting host
- **opening_angle** — how to open the conversation credibly
- **talking_points** — company-specific and audience-specific points
- **recommended_questions** — grounded in research, not generic
- **possible_objections** — what the stakeholder might push back on
- **suggested_next_step** — a concrete action to propose at the end

The engagement context (interaction stage, conversation mode, tone,
next-step pressure) is a **primary driver** of the strategy.  A first
intro demands a different opening, questions, and next step than a
negotiation or account-expansion meeting.

Person-specific details may influence the strategy ONLY when
``person_profile.match_confidence`` is ``"medium"`` or ``"high"``.

Ownership:  ``meeting_strategy`` only.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from services.llm_client import structured_extract

logger = logging.getLogger(__name__)

# Confidence values that permit using person-level details.
# Both "low" and "none" (no search attempted) are non-reliable.
_RELIABLE_CONFIDENCE = frozenset({"medium", "high"})


# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

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
    If user-side context is absent, fall back to the existing logic
    (meeting_type + engagement_context as strategy drivers).
"""


def _empty_strategy() -> dict[str, Any]:
    return MeetingStrategy(
        executive_summary="",
        opening_angle="",
        suggested_next_step="",
    ).model_dump()


# ---------------------------------------------------------------------------
# Build the user prompt
# ---------------------------------------------------------------------------

def _build_user_prompt(state: dict[str, Any]) -> str:
    parts: list[str] = []

    # Meeting context
    parts.append("=== Meeting Context ===")
    parts.append(f"Company: {state.get('company_name', 'unknown')}")
    parts.append(f"Meeting type: {state.get('meeting_type', '')}")
    parts.append(f"Stakeholder role: {state.get('stakeholder_role', '')}")
    parts.append(f"Meeting goal: {state.get('meeting_goal', '')}")

    # User-side context — primary strategy driver when present
    _user_fields = {
        "user_role": "User role",
        "user_company": "User company",
        "user_function": "User function",
        "engagement_type": "Engagement type",
        "desired_outcome": "Desired outcome",
        "success_definition": "Success definition",
    }
    user_lines = []
    for key, label in _user_fields.items():
        val = state.get(key)
        if val:
            user_lines.append(f"{label}: {val}")
    if user_lines:
        parts.append("")
        parts.append("=== User-Side Context (strategy driver) ===")
        parts.extend(user_lines)

    # Engagement context — primary driver for strategy calibration
    ec: dict[str, Any] = state.get("engagement_context", {})
    if ec.get("interaction_stage"):
        parts.append("")
        parts.append("=== Engagement Context ===")
        parts.append(f"Interaction stage: {ec['interaction_stage']}")
        parts.append(f"Relationship status: {ec.get('relationship_status') or 'unknown'}")
        parts.append(f"Conversation mode: {ec.get('conversation_mode', 'exploratory')}")
        parts.append(f"Stage objective: {ec.get('stage_objective', '')}")
        parts.append(f"Recommended tone: {ec.get('recommended_tone', '')}")
        parts.append(f"Next-step pressure: {ec.get('next_step_pressure', 'low')}")
        risks = ec.get("decision_risks", [])
        if risks:
            parts.append(f"Decision risks: {'; '.join(risks)}")
        continuity = ec.get("continuity_context")
        if continuity:
            parts.append(f"Continuity context: {continuity}")

    # Company profile
    cp: dict[str, Any] = state.get("company_profile", {})
    if any(cp.get(k) for k in ("industry", "business_model", "company_positioning")):
        parts.append("")
        parts.append("=== Company Profile ===")
        for key in ("industry", "business_model", "company_positioning"):
            val = cp.get(key)
            if val:
                parts.append(f"{key}: {val}")
        for key in ("products", "customer_segments", "geographies", "complexity_signals"):
            items = cp.get(key, [])
            if items:
                parts.append(f"{key}: {', '.join(str(i) for i in items[:6])}")

    # External signals
    signals: list[dict] = state.get("external_signals", [])
    if signals:
        parts.append("")
        parts.append("=== Recent Strategic Signals ===")
        for i, sig in enumerate(signals[:10], 1):
            date_str = f" ({sig['date']})" if sig.get("date") else ""
            parts.append(
                f"{i}. [{sig.get('signal_type', '?')}]{date_str} "
                f"{sig.get('description', '')}  "
                f"Implication: {sig.get('implication', 'n/a')}  "
                f"(confidence: {sig.get('confidence', '?')})"
            )
    else:
        parts.append("\nNo recent strategic signals available.")

    # Audience context
    ac: dict[str, Any] = state.get("audience_context", {})
    if ac.get("department"):
        parts.append("")
        parts.append("=== Audience Context ===")
        parts.append(f"Department: {ac.get('department', '')}")
        parts.append(f"Seniority: {ac.get('role_seniority') or 'unknown'}")
        priorities = ac.get("likely_priorities", [])
        if priorities:
            parts.append(f"Likely priorities: {', '.join(priorities)}")
        topics = ac.get("relevant_topics", [])
        if topics:
            parts.append(f"Relevant topics: {', '.join(topics)}")
        style = ac.get("communication_style")
        if style:
            parts.append(f"Communication style: {style}")
        sensitivities = ac.get("potential_sensitivities", [])
        if sensitivities:
            parts.append(f"Potential sensitivities: {', '.join(sensitivities)}")

    # Pain point hypotheses
    pph: dict[str, Any] = state.get("pain_point_hypotheses", {})
    hypotheses = pph.get("hypotheses", [])
    if hypotheses:
        parts.append("")
        parts.append("=== Pain Point Hypotheses ===")
        for i, h in enumerate(hypotheses[:8], 1):
            evidence = "; ".join(h.get("supporting_evidence", []))
            parts.append(
                f"{i}. [{h.get('confidence', '?')}] {h.get('hypothesis', '')}  "
                f"Evidence: {evidence or 'n/a'}"
            )
    items_to_validate = pph.get("items_to_validate", [])
    if items_to_validate:
        parts.append(f"Items to validate: {'; '.join(items_to_validate[:6])}")

    # Person profile — gated on confidence
    pp: dict[str, Any] = state.get("person_profile", {})
    confidence = pp.get("match_confidence", "none")

    if confidence in _RELIABLE_CONFIDENCE:
        parts.append("")
        parts.append("=== Person Profile (reliable — use to personalise) ===")
        for key in ("name", "title", "department", "background_summary"):
            val = pp.get(key)
            if val:
                parts.append(f"{key}: {val}")
        pub = pp.get("public_signals", [])
        if pub:
            parts.append(f"Public signals: {', '.join(pub[:5])}")
    else:
        parts.append("")
        parts.append(
            "No reliable person profile — tailor strategy to company "
            "and audience context only."
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run(state: dict[str, Any]) -> dict[str, Any]:
    """Produce a meeting strategy from all accumulated research and analysis."""

    new_errors: list[str] = []

    user_prompt = _build_user_prompt(state)

    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=MeetingStrategy,
    )

    if result is None:
        new_errors.append("meeting_strategy_agent: LLM reasoning failed, returning empty strategy.")
        return {"meeting_strategy": _empty_strategy(), "errors": new_errors}

    return {"meeting_strategy": result.model_dump()}
