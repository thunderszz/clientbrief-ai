"""Orchestrator node — fully deterministic, no LLM calls.

Responsibilities (per spec):
- validate all input fields (including optional meeting_stage)
- normalize company domain
- normalize role → function + seniority
- initialize every BriefingState field with safe defaults
- check the file cache and load on hit
- surface validation errors on the ``errors`` list

Ownership:  normalized inputs, cache flags, errors.
This node never writes to agent-output fields it does not own.
"""

from __future__ import annotations

from typing import Any

from core.cache import cache_read
from core.state import BriefingState
from core.utils import (
    normalize_domain,
    normalize_function,
    normalize_seniority,
    validate_inputs,
)


# ---------------------------------------------------------------------------
# Defaults for agent-output fields the orchestrator does not own.
# These provide safe starting values so downstream nodes always see
# the expected keys, even when they haven't run yet.
# ---------------------------------------------------------------------------

_EMPTY_COMPANY_PROFILE: dict[str, Any] = {
    "industry": None,
    "business_model": None,
    "products": [],
    "customer_segments": [],
    "geographies": [],
    "company_positioning": None,
    "complexity_signals": [],
    "sources": [],
}

_EMPTY_PERSON_PROFILE: dict[str, Any] = {
    "matched_person": False,
    "match_confidence": "none",
    "name": None,
    "title": None,
    "department": None,
    "tenure_hint": None,
    "background_summary": None,
    "public_signals": [],
    "sources": [],
}

_EMPTY_AUDIENCE_CONTEXT: dict[str, Any] = {
    "department": "",
    "role_seniority": None,
    "likely_priorities": [],
    "relevant_topics": [],
    "communication_style": "",
    "potential_sensitivities": [],
}

_EMPTY_ENGAGEMENT_INFERENCE: dict[str, Any] = {
    "inferred_meeting_stage": None,
    "inferred_relationship_status": None,
    "inferred_previous_meeting_outcome": None,
    "confidence": "not_needed",
    "reasoning_summary": "",
}

_EMPTY_ENGAGEMENT_CONTEXT: dict[str, Any] = {
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

_EMPTY_PAIN_POINTS: dict[str, Any] = {
    "facts": [],
    "inferences": [],
    "hypotheses": [],
    "items_to_validate": [],
}

_EMPTY_MEETING_STRATEGY: dict[str, Any] = {
    "executive_summary": "",
    "opening_angle": "",
    "talking_points": [],
    "recommended_questions": [],
    "possible_objections": [],
    "suggested_next_step": "",
}


def _init_state(
    company_name: str,
    domain: str,
    meeting_type: str,
    stakeholder_role: str,
    meeting_goal: str,
    contact_name: str | None,
    contact_title: str | None,
    meeting_stage: str | None,
    previous_meeting_outcome: str | None,
    relationship_status: str | None,
    meeting_notes_summary: str | None,
    normalized_function: str,
    normalized_seniority: str | None,
    *,
    user_role: str | None = None,
    user_company: str | None = None,
    user_function: str | None = None,
    engagement_type: str | None = None,
    desired_outcome: str | None = None,
    success_definition: str | None = None,
    raw_meeting_context: str | None = None,
    intake_missing_fields: list | None = None,
    intake_low_confidence_fields: list | None = None,
    intake_confirmation_summary: str | None = None,
) -> BriefingState:
    """Build a fully-initialised BriefingState with safe defaults.

    Note: ``meeting_stage`` may be ``None`` when the user does not provide it.
    The engagement-inference and engagement-context agents resolve the final
    value downstream.

    Intake metadata (``raw_meeting_context``, ``intake_*``) is passed through
    unchanged if the caller used the intake agent before invoking the workflow.
    """
    return BriefingState(
        # Pre-intake metadata (pass-through, may be absent)
        raw_meeting_context=raw_meeting_context,
        intake_missing_fields=intake_missing_fields or [],
        intake_low_confidence_fields=intake_low_confidence_fields or [],
        intake_confirmation_summary=intake_confirmation_summary,
        # User inputs (stored as-given; domain already normalized)
        company_name=company_name.strip(),
        domain=domain,
        meeting_type=meeting_type,
        stakeholder_role=stakeholder_role.strip(),
        meeting_goal=meeting_goal.strip(),
        contact_name=contact_name.strip() if contact_name else None,
        contact_title=contact_title.strip() if contact_title else None,
        # Engagement inputs — None means "not provided by user"
        meeting_stage=meeting_stage,
        previous_meeting_outcome=previous_meeting_outcome,
        relationship_status=relationship_status,
        meeting_notes_summary=meeting_notes_summary.strip() if meeting_notes_summary else None,
        # User-side context — pass through as-is
        user_role=user_role,
        user_company=user_company,
        user_function=user_function,
        engagement_type=engagement_type,
        desired_outcome=desired_outcome,
        success_definition=success_definition,
        # Derived
        normalized_function=normalized_function,
        normalized_seniority=normalized_seniority,
        # Engagement inference (owned by engagement_inference_agent)
        engagement_inference=_EMPTY_ENGAGEMENT_INFERENCE.copy(),
        # Agent-output defaults (owned by later nodes)
        research_plan={},
        company_profile=_EMPTY_COMPANY_PROFILE.copy(),
        external_signals=[],
        person_profile=_EMPTY_PERSON_PROFILE.copy(),
        audience_context=_EMPTY_AUDIENCE_CONTEXT.copy(),
        engagement_context=_EMPTY_ENGAGEMENT_CONTEXT.copy(),
        pain_point_hypotheses=_EMPTY_PAIN_POINTS.copy(),
        meeting_strategy=_EMPTY_MEETING_STRATEGY.copy(),
        final_brief="",
        # Metadata
        sources=[],
        brief_reliability=0,
        cache_hit=False,
        errors=[],
    )


# ---------------------------------------------------------------------------
# Public entry-point (LangGraph-compatible signature)
# ---------------------------------------------------------------------------

def run(state: dict[str, Any]) -> dict[str, Any]:
    """Orchestrate: validate → normalize → init state → cache check.

    Parameters
    ----------
    state:
        Raw input dict (at minimum the required user-input fields).
        When invoked by LangGraph the dict may already contain keys
        from a previous run; we only read input-level keys here.

    Returns
    -------
    dict:
        A partial state update.  LangGraph merges this into the
        shared state automatically.
    """

    # --- 1. Extract raw inputs ------------------------------------------------
    company_name: str = state.get("company_name", "")
    raw_domain: str = state.get("domain", "")
    meeting_type: str = state.get("meeting_type", "")
    stakeholder_role: str = state.get("stakeholder_role", "")
    meeting_goal: str = state.get("meeting_goal", "")
    contact_name: str | None = state.get("contact_name")
    contact_title: str | None = state.get("contact_title")

    # Engagement inputs — all optional.  None means "not provided by user".
    # The engagement_inference_agent will attempt to fill gaps downstream;
    # the engagement_context_agent resolves final values with defaults.
    meeting_stage: str | None = state.get("meeting_stage") or None
    previous_meeting_outcome: str | None = state.get("previous_meeting_outcome")
    relationship_status: str | None = state.get("relationship_status")
    meeting_notes_summary: str | None = state.get("meeting_notes_summary")

    # User-side context — optional, pass through as-is.
    user_role: str | None = state.get("user_role")
    user_company: str | None = state.get("user_company")
    user_function: str | None = state.get("user_function")
    engagement_type: str | None = state.get("engagement_type")
    desired_outcome: str | None = state.get("desired_outcome")
    success_definition: str | None = state.get("success_definition")

    # Intake metadata — pass through if present (populated by intake agent).
    raw_meeting_context: str | None = state.get("raw_meeting_context")
    intake_missing_fields: list = state.get("intake_missing_fields", [])
    intake_low_confidence_fields: list = state.get("intake_low_confidence_fields", [])
    intake_confirmation_summary: str | None = state.get("intake_confirmation_summary")

    # --- 2. Normalize domain --------------------------------------------------
    domain = normalize_domain(raw_domain) if raw_domain else ""

    # --- 3. Validate inputs ---------------------------------------------------
    errors = validate_inputs(
        company_name, domain, meeting_type, stakeholder_role,
        meeting_stage=meeting_stage,
    )
    if errors:
        # Return early with errors — downstream nodes should check this.
        return {
            "raw_meeting_context": raw_meeting_context,
            "intake_missing_fields": intake_missing_fields,
            "intake_low_confidence_fields": intake_low_confidence_fields,
            "intake_confirmation_summary": intake_confirmation_summary,
            "company_name": company_name.strip() if company_name else "",
            "domain": domain,
            "meeting_type": meeting_type,
            "stakeholder_role": stakeholder_role.strip() if stakeholder_role else "",
            "meeting_goal": meeting_goal.strip() if meeting_goal else "",
            "contact_name": contact_name,
            "contact_title": contact_title,
            "meeting_stage": meeting_stage,
            "previous_meeting_outcome": previous_meeting_outcome,
            "relationship_status": relationship_status,
            "meeting_notes_summary": meeting_notes_summary,
            "user_role": user_role,
            "user_company": user_company,
            "user_function": user_function,
            "engagement_type": engagement_type,
            "desired_outcome": desired_outcome,
            "success_definition": success_definition,
            "normalized_function": "general",
            "normalized_seniority": None,
            "engagement_inference": _EMPTY_ENGAGEMENT_INFERENCE.copy(),
            "research_plan": {},
            "company_profile": _EMPTY_COMPANY_PROFILE.copy(),
            "external_signals": [],
            "person_profile": _EMPTY_PERSON_PROFILE.copy(),
            "audience_context": _EMPTY_AUDIENCE_CONTEXT.copy(),
            "engagement_context": _EMPTY_ENGAGEMENT_CONTEXT.copy(),
            "pain_point_hypotheses": _EMPTY_PAIN_POINTS.copy(),
            "meeting_strategy": _EMPTY_MEETING_STRATEGY.copy(),
            "final_brief": "",
            "sources": [],
            "brief_reliability": 0,
            "cache_hit": False,
            "errors": errors,
        }

    # --- 4. Normalize role → function + seniority -----------------------------
    # We resolve function from contact_title first (richer); stakeholder_role
    # as fallback.
    role_source = contact_title or stakeholder_role
    normalized_function = normalize_function(role_source)
    normalized_seniority = normalize_seniority(role_source)

    # --- 5. Build initial state -----------------------------------------------
    full_state = _init_state(
        company_name=company_name,
        domain=domain,
        meeting_type=meeting_type,
        stakeholder_role=stakeholder_role,
        meeting_goal=meeting_goal,
        contact_name=contact_name,
        contact_title=contact_title,
        meeting_stage=meeting_stage,
        previous_meeting_outcome=previous_meeting_outcome,
        relationship_status=relationship_status,
        meeting_notes_summary=meeting_notes_summary,
        normalized_function=normalized_function,
        normalized_seniority=normalized_seniority,
        user_role=user_role,
        user_company=user_company,
        user_function=user_function,
        engagement_type=engagement_type,
        desired_outcome=desired_outcome,
        success_definition=success_definition,
        raw_meeting_context=raw_meeting_context,
        intake_missing_fields=intake_missing_fields,
        intake_low_confidence_fields=intake_low_confidence_fields,
        intake_confirmation_summary=intake_confirmation_summary,
    )

    # --- 6. Cache check -------------------------------------------------------
    cached = cache_read(
        domain=domain,
        meeting_type=meeting_type,
        normalized_function=normalized_function,
        stakeholder_role=stakeholder_role.strip(),
    )

    if cached is not None:
        # Merge cached research slices into state.
        # Only overwrite fields that downstream agents own and that the cache
        # actually contains.  Never overwrite orchestrator-owned fields with
        # stale cached values — the current run's normalizations take precedence.
        _CACHEABLE_KEYS = {
            "engagement_inference",
            "research_plan",
            "company_profile",
            "external_signals",
            "person_profile",
            "audience_context",
            "engagement_context",
            "pain_point_hypotheses",
            "meeting_strategy",
            "sources",
            "brief_reliability",
        }
        for key in _CACHEABLE_KEYS:
            if key in cached:
                full_state[key] = cached[key]  # type: ignore[literal-required]

        full_state["cache_hit"] = True

    return dict(full_state)
