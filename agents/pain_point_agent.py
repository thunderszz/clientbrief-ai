"""Pain Point Hypothesis Agent — LLM reasoning for business-challenge synthesis.

Synthesises company profile, external signals, audience context,
engagement context, and (if reliable) person profile into a structured set of:
- **facts** — confirmed, source-grounded statements
- **inferences** — reasoned conclusions that follow from facts
- **hypotheses** — speculative but plausible pain points, each with
  supporting evidence and a confidence level
- **items_to_validate** — things the user should verify during the meeting

The engagement context (interaction stage, conversation mode) influences
*which* pain points are most relevant: early-stage meetings surface broad
challenges; later-stage meetings focus on deal-relevant or solution-specific
pain points.

Person-specific details may influence hypotheses ONLY when
``person_profile.match_confidence`` is ``"medium"`` or ``"high"``.

Ownership:  ``pain_point_hypotheses`` only.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from services.llm_client import structured_extract

logger = logging.getLogger(__name__)

# Confidence values that permit using person-level details.
_RELIABLE_CONFIDENCE = frozenset({"medium", "high"})


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

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
   If user-side context is absent, ignore this rule and frame generically
   as before.
"""


def _empty_hypotheses() -> dict[str, Any]:
    return PainPointHypotheses().model_dump()


# ---------------------------------------------------------------------------
# Build the user prompt
# ---------------------------------------------------------------------------

def _build_user_prompt(state: dict[str, Any]) -> str:
    parts: list[str] = []

    # Company profile
    cp: dict[str, Any] = state.get("company_profile", {})
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

    # Engagement context — stage, mode, risks
    ec: dict[str, Any] = state.get("engagement_context", {})
    if ec.get("interaction_stage"):
        parts.append("")
        parts.append("=== Engagement Context ===")
        parts.append(f"Interaction stage: {ec['interaction_stage']}")
        parts.append(f"Conversation mode: {ec.get('conversation_mode', 'exploratory')}")
        parts.append(f"Stage objective: {ec.get('stage_objective', '')}")
        risks = ec.get("decision_risks", [])
        if risks:
            parts.append(f"Decision risks: {'; '.join(risks)}")
        continuity = ec.get("continuity_context")
        if continuity:
            parts.append(f"Continuity context: {continuity}")

    # Person profile — gated on confidence
    pp: dict[str, Any] = state.get("person_profile", {})
    confidence = pp.get("match_confidence", "none")

    if confidence in _RELIABLE_CONFIDENCE:
        parts.append("")
        parts.append("=== Person Profile (reliable — may use) ===")
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
            "No reliable person profile — reason from company and "
            "function context only."
        )

    # User-side context — framing lens (not evidence)
    _user_fields = {
        "user_role": "User role",
        "user_company": "User company",
        "user_function": "User function",
        "engagement_type": "Engagement type",
        "meeting_goal": "Meeting goal",
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
        parts.append("=== User-Side Context (framing lens — NOT evidence) ===")
        parts.extend(user_lines)

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run(state: dict[str, Any]) -> dict[str, Any]:
    """Synthesise pain-point hypotheses from all available research."""

    new_errors: list[str] = []

    user_prompt = _build_user_prompt(state)

    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=PainPointHypotheses,
    )

    if result is None:
        new_errors.append("pain_point_agent: LLM reasoning failed, returning empty hypotheses.")
        return {"pain_point_hypotheses": _empty_hypotheses(), "errors": new_errors}

    return {"pain_point_hypotheses": result.model_dump()}
