"""Engagement context: where this meeting sits in the relationship, and what follows from it.

Deterministic. Each field is resolved in order of trust: what the user stated, then what
the inference node guessed, then a default. The stage then sets the conversation mode, tone,
risks and next-step pressure, and how much baseline research the planner should run.
"""

from __future__ import annotations

from typing import Any

# Conversation mode per meeting stage.
_CONVERSATION_MODES: dict[str, str] = {
    "first_intro":          "exploratory",
    "discovery_followup":   "consultative",
    "solution_discussion":  "solution_oriented",
    "proposal_review":      "commercial",
    "negotiation":          "commercial",
    "client_kickoff":       "partnership",
    "account_expansion":    "consultative",
}

# What the meeting should achieve at each stage.
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

# Next-step pressure.
_NEXT_STEP_PRESSURE: dict[str, str] = {
    "first_intro":          "low",
    "discovery_followup":   "low",
    "solution_discussion":  "moderate",
    "proposal_review":      "high",
    "negotiation":          "high",
    "client_kickoff":       "low",
    "account_expansion":    "moderate",
}

# Decision risks per stage.
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

# Extra risks when the last meeting did not go clearly well.
_OUTCOME_RISK_MODIFIERS: dict[str, list[str]] = {
    "neutral": [
        "Previous interaction did not generate strong engagement — may need to re-establish relevance",
    ],
    "unclear": [
        "Unclear outcome from last meeting — stakeholder intent is ambiguous, proceed carefully",
    ],
}

# Recommended tone per stage, refined by seniority where it matters.
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

# Relationship to assume for each stage when it was neither stated nor inferred.
_DEFAULT_RELATIONSHIP: dict[str, str] = {
    "first_intro":          "prospect",
    "discovery_followup":   "prospect",
    "solution_discussion":  "active_opportunity",
    "proposal_review":      "active_opportunity",
    "negotiation":          "active_opportunity",
    "client_kickoff":       "client",
    "account_expansion":    "client",
}

# How much baseline research each stage needs: full, light or update_only.
_RESEARCH_MODE_TABLE: dict[str, str] = {
    "first_intro":          "full",
    "discovery_followup":   "light",
    "solution_discussion":  "light",
    "proposal_review":      "update_only",
    "negotiation":          "update_only",
    "client_kickoff":       "update_only",
    "account_expansion":    "light",
}


def _first_known(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _resolve_tone(meeting_stage: str, seniority: str | None) -> str:
    stage_tones = _TONE_TEMPLATES.get(meeting_stage, _TONE_TEMPLATES["first_intro"])
    if seniority and seniority in stage_tones:
        return stage_tones[seniority]
    return stage_tones.get("_default", "Professional and attentive")


_OUTCOME_SUMMARIES = {
    "positive": "Previous interaction was positive.",
    "neutral": "Previous interaction was neutral — no strong signal either way.",
    "unclear": "Outcome of previous interaction is unclear — may need to reassess.",
}


def _continuity(previous_outcome: str | None, notes: str | None) -> str | None:
    """One line about the last interaction, or None when there was none."""
    parts = []
    if previous_outcome and previous_outcome != "no_previous_meeting":
        parts.append(_OUTCOME_SUMMARIES.get(previous_outcome, f"Previous outcome: {previous_outcome}."))
    if notes and notes.strip():
        parts.append(f"Notes: {notes.strip()}")
    return " ".join(parts) or None


def _research_mode(meeting_stage: str, relationship_status: str | None, confidence: str | None) -> str:
    """How much baseline research to run. Former clients, and shaky guesses, get the full run."""
    if relationship_status == "former_client" or confidence == "low":
        return "full"
    return _RESEARCH_MODE_TABLE.get(meeting_stage, "full")


def empty_context() -> dict[str, Any]:
    """Placeholder until this node runs: a first introduction with nothing known."""
    return {
        "interaction_stage": "first_intro",
        "relationship_status": None,
        "conversation_mode": "exploratory",
        "stage_objective": "",
        "recommended_tone": "",
        "decision_risks": [],
        "next_step_pressure": "low",
        "continuity_context": None,
        "research_mode": "full",
    }


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Resolve the engagement fields and return ``engagement_context``."""
    inference = state.get("engagement_inference", {})
    stage = _first_known(state.get("meeting_stage"), inference.get("inferred_meeting_stage"), "first_intro")
    relationship = _first_known(
        state.get("relationship_status"),
        inference.get("inferred_relationship_status"),
        _DEFAULT_RELATIONSHIP.get(stage, "prospect"),
    )
    previous_outcome = _first_known(
        state.get("previous_meeting_outcome"), inference.get("inferred_previous_meeting_outcome")
    )

    risks = list(_STAGE_RISKS.get(stage, []))
    risks += _OUTCOME_RISK_MODIFIERS.get(previous_outcome or "no_previous_meeting", [])

    return {
        "engagement_context": {
            "interaction_stage": stage,
            "relationship_status": relationship,
            "conversation_mode": _CONVERSATION_MODES.get(stage, "exploratory"),
            "stage_objective": _STAGE_OBJECTIVES.get(stage, ""),
            "recommended_tone": _resolve_tone(stage, state.get("normalized_seniority")),
            "decision_risks": risks,
            "next_step_pressure": _NEXT_STEP_PRESSURE.get(stage, "low"),
            "continuity_context": _continuity(previous_outcome, state.get("meeting_notes_summary")),
            "research_mode": _research_mode(stage, relationship, inference.get("confidence")),
        }
    }
