"""Engagement Context Agent — fully deterministic, no LLM calls.

Resolves the final normalised engagement context by merging three sources
with strict precedence:

    1. Explicit user-provided values  (highest priority)
    2. Inferred values from ``engagement_inference``
    3. Safe defaults  (lowest priority)

This ensures that:
- user intent is always respected when stated
- LLM-inferred values fill gaps transparently
- the system never crashes on missing inputs

The agent also derives ``research_mode`` (full / light / update_only) from
the resolved engagement state, which the research planner uses to calibrate
research depth.

Inputs:
- ``meeting_stage``               — raw user input (may be None)
- ``previous_meeting_outcome``    — raw user input (may be None)
- ``relationship_status``         — raw user input (may be None)
- ``meeting_notes_summary``       — raw user input (may be None)
- ``meeting_type``                — always present
- ``normalized_seniority``        — from orchestrator
- ``engagement_inference``        — from engagement_inference_agent

Output: ``engagement_context`` dict with 9 fields (added ``research_mode``).

Ownership:  ``engagement_context`` only.
"""

from __future__ import annotations

from typing import Any


# =============================================================================
# Lookup tables
# =============================================================================

# --- Conversation mode per meeting stage -------------------------------------
_CONVERSATION_MODES: dict[str, str] = {
    "first_intro":          "exploratory",
    "discovery_followup":   "consultative",
    "solution_discussion":  "solution_oriented",
    "proposal_review":      "commercial",
    "negotiation":          "commercial",
    "client_kickoff":       "partnership",
    "account_expansion":    "consultative",
}

# --- Stage objective ---------------------------------------------------------
_STAGE_OBJECTIVES: dict[str, str] = {
    "first_intro": (
        "Establish credibility, understand the stakeholder's world, "
        "and secure a meaningful follow-up."
    ),
    "discovery_followup": (
        "Deepen understanding of specific challenges surfaced earlier "
        "and validate initial hypotheses."
    ),
    "solution_discussion": (
        "Align proposed solution to the stakeholder's requirements "
        "and surface evaluation criteria."
    ),
    "proposal_review": (
        "Address concerns about the proposal, reinforce value, "
        "and move toward a decision."
    ),
    "negotiation": (
        "Reach agreement on terms while preserving relationship quality "
        "and deal value."
    ),
    "client_kickoff": (
        "Align on outcomes, ways of working, and immediate priorities "
        "for the engagement."
    ),
    "account_expansion": (
        "Surface new needs and demonstrate value from the existing "
        "relationship to expand scope."
    ),
}

# --- Next-step pressure ------------------------------------------------------
_NEXT_STEP_PRESSURE: dict[str, str] = {
    "first_intro":          "low",
    "discovery_followup":   "low",
    "solution_discussion":  "moderate",
    "proposal_review":      "high",
    "negotiation":          "high",
    "client_kickoff":       "low",
    "account_expansion":    "moderate",
}

# --- Base decision risks per stage -------------------------------------------
_STAGE_RISKS: dict[str, list[str]] = {
    "first_intro": [
        "Stakeholder may not see enough relevance to grant a follow-up",
        "Conversation may stay too high-level to generate actionable next steps",
    ],
    "discovery_followup": [
        "Momentum from the first meeting may have faded",
        "Stakeholder priorities may have shifted since last contact",
    ],
    "solution_discussion": [
        "Technical or process misfit may surface during solution deep-dive",
        "Internal decision-makers not present may block progress later",
    ],
    "proposal_review": [
        "Budget constraints or competing priorities may delay decision",
        "Competitive alternatives may be under parallel evaluation",
    ],
    "negotiation": [
        "Price sensitivity or procurement process may stall the deal",
        "Internal champion may lack authority to close",
        "Competitor may be offering more aggressive terms",
    ],
    "client_kickoff": [
        "Misaligned expectations on scope or timelines may create friction",
        "Key stakeholders may not be fully bought in",
    ],
    "account_expansion": [
        "Satisfaction with current engagement may not extend to new areas",
        "Budget for expansion may not be secured yet",
        "Different stakeholders in the new area may have different priorities",
    ],
}

# --- Outcome-adjusted risk modifiers ----------------------------------------
_OUTCOME_RISK_MODIFIERS: dict[str, list[str]] = {
    "neutral": [
        "Previous interaction did not generate strong engagement — may need to re-establish relevance",
    ],
    "unclear": [
        "Unclear outcome from last meeting — stakeholder intent is ambiguous, proceed carefully",
    ],
}

# --- Recommended tone per stage x seniority ---------------------------------
_TONE_TEMPLATES: dict[str, dict[str, str]] = {
    "first_intro": {
        "_default":  "Curious and respectful — ask more than tell, demonstrate genuine interest in their challenges",
        "c_level":   "Strategic and concise — lead with business impact, respect their time, avoid operational detail",
        "vp":        "Strategic with depth — show understanding of their domain, balance big-picture and specifics",
    },
    "discovery_followup": {
        "_default":  "Consultative and attentive — reference prior conversation, show you listened, go deeper",
        "c_level":   "Executive and outcome-focused — connect prior discussion to business outcomes they care about",
    },
    "solution_discussion": {
        "_default":  "Confident and collaborative — present solutions clearly while inviting input and adaptation",
        "c_level":   "Outcome-driven and efficient — lead with value and results, keep technical detail for the team",
    },
    "proposal_review": {
        "_default":  "Professional and responsive — address concerns directly, reinforce value at every point",
        "c_level":   "Commercial and strategic — frame the proposal in terms of business impact and risk reduction",
    },
    "negotiation": {
        "_default":  "Firm but collaborative — seek mutual value, avoid positional bargaining",
        "c_level":   "Direct and partnership-oriented — frame terms as aligned investment, not transactional cost",
    },
    "client_kickoff": {
        "_default":  "Energetic and structured — set expectations, build confidence, show preparation",
        "c_level":   "Reassuring and outcome-focused — confirm strategic alignment, delegate operational detail",
    },
    "account_expansion": {
        "_default":  "Trusted-advisor mode — leverage relationship capital, lead with demonstrated results",
        "c_level":   "Strategic and forward-looking — connect expansion to their evolving strategic priorities",
    },
}

# --- Relationship-status inference from meeting_stage -----------------------
_DEFAULT_RELATIONSHIP: dict[str, str] = {
    "first_intro":          "prospect",
    "discovery_followup":   "prospect",
    "solution_discussion":  "active_opportunity",
    "proposal_review":      "active_opportunity",
    "negotiation":          "active_opportunity",
    "client_kickoff":       "client",
    "account_expansion":    "client",
}

# --- Research mode per meeting stage ----------------------------------------
# Determines how much baseline research the planner should run.
_RESEARCH_MODE_TABLE: dict[str, str] = {
    "first_intro":          "full",
    "discovery_followup":   "light",
    "solution_discussion":  "light",
    "proposal_review":      "update_only",
    "negotiation":          "update_only",
    "client_kickoff":       "update_only",
    "account_expansion":    "light",
}


# =============================================================================
# Helpers
# =============================================================================

def _resolve_with_precedence(
    explicit: Any | None,
    inferred: Any | None,
    default: Any,
) -> Any:
    """Return the first non-None value in precedence order:
    explicit > inferred > default.
    """
    if explicit is not None:
        return explicit
    if inferred is not None:
        return inferred
    return default


def _resolve_tone(meeting_stage: str, seniority: str | None) -> str:
    """Pick the best tone recommendation for stage x seniority."""
    stage_tones = _TONE_TEMPLATES.get(meeting_stage, _TONE_TEMPLATES["first_intro"])
    if seniority and seniority in stage_tones:
        return stage_tones[seniority]
    return stage_tones.get("_default", "Professional and attentive")


def _build_continuity_context(
    previous_meeting_outcome: str | None,
    meeting_notes_summary: str | None,
) -> str | None:
    """Format prior-interaction context into a concise summary."""
    parts: list[str] = []

    if previous_meeting_outcome and previous_meeting_outcome != "no_previous_meeting":
        outcome_labels = {
            "positive": "Previous interaction was positive.",
            "neutral": "Previous interaction was neutral — no strong signal either way.",
            "unclear": "Outcome of previous interaction is unclear — may need to reassess.",
        }
        parts.append(outcome_labels.get(
            previous_meeting_outcome,
            f"Previous outcome: {previous_meeting_outcome}.",
        ))

    if meeting_notes_summary and meeting_notes_summary.strip():
        parts.append(f"Notes: {meeting_notes_summary.strip()}")

    return " ".join(parts) if parts else None


def _resolve_research_mode(
    meeting_stage: str,
    relationship_status: str | None,
    inference_confidence: str | None,
) -> str:
    """Derive research_mode from resolved engagement state.

    Rules:
    - Base mode comes from meeting_stage → _RESEARCH_MODE_TABLE.
    - Former clients always get ``full`` research (need to re-learn context).
    - Low inference confidence upgrades to at least ``light`` (don't reduce
      research when we're uncertain about the engagement state).
    """
    base = _RESEARCH_MODE_TABLE.get(meeting_stage, "full")

    # Former clients need full re-research regardless of stage
    if relationship_status == "former_client":
        return "full"

    # Low inference confidence → don't reduce research aggressively
    if inference_confidence == "low" and base != "full":
        return "full"

    return base


# =============================================================================
# Public entry-point
# =============================================================================

def run(state: dict[str, Any]) -> dict[str, Any]:
    """Produce engagement context using hybrid resolution: explicit > inferred > default.

    Remains fully deterministic — no LLM calls.  Writes only ``engagement_context``.
    """

    # --- Read raw user inputs -------------------------------------------------
    explicit_stage: str | None = state.get("meeting_stage")
    explicit_relationship: str | None = state.get("relationship_status")
    explicit_outcome: str | None = state.get("previous_meeting_outcome")
    meeting_notes_summary: str | None = state.get("meeting_notes_summary")
    normalized_seniority: str | None = state.get("normalized_seniority")

    # --- Read inferred values -------------------------------------------------
    ei: dict[str, Any] = state.get("engagement_inference", {})
    inferred_stage: str | None = ei.get("inferred_meeting_stage")
    inferred_relationship: str | None = ei.get("inferred_relationship_status")
    inferred_outcome: str | None = ei.get("inferred_previous_meeting_outcome")
    inference_confidence: str | None = ei.get("confidence")

    # --- Resolve final values with precedence ---------------------------------
    meeting_stage: str = _resolve_with_precedence(
        explicit_stage, inferred_stage, "first_intro",
    )
    relationship_status: str = _resolve_with_precedence(
        explicit_relationship,
        inferred_relationship,
        _DEFAULT_RELATIONSHIP.get(meeting_stage, "prospect"),
    )
    previous_meeting_outcome: str | None = _resolve_with_precedence(
        explicit_outcome, inferred_outcome, None,
    )

    # --- Build decision risks -------------------------------------------------
    risks = list(_STAGE_RISKS.get(meeting_stage, []))
    outcome_for_risks = previous_meeting_outcome or "no_previous_meeting"
    if outcome_for_risks in _OUTCOME_RISK_MODIFIERS:
        risks.extend(_OUTCOME_RISK_MODIFIERS[outcome_for_risks])

    # --- Derive research_mode -------------------------------------------------
    research_mode = _resolve_research_mode(
        meeting_stage, relationship_status, inference_confidence,
    )

    # --- Build the context dict -----------------------------------------------
    engagement_context: dict[str, Any] = {
        "interaction_stage": meeting_stage,
        "relationship_status": relationship_status,
        "conversation_mode": _CONVERSATION_MODES.get(meeting_stage, "exploratory"),
        "stage_objective": _STAGE_OBJECTIVES.get(meeting_stage, ""),
        "recommended_tone": _resolve_tone(meeting_stage, normalized_seniority),
        "decision_risks": risks,
        "next_step_pressure": _NEXT_STEP_PRESSURE.get(meeting_stage, "low"),
        "continuity_context": _build_continuity_context(
            previous_meeting_outcome, meeting_notes_summary,
        ),
        "research_mode": research_mode,
    }

    return {"engagement_context": engagement_context}
