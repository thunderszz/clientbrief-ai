"""Shared state definition for the ClientBrief AI pipeline.

This module defines the BriefingState TypedDict used across all LangGraph nodes,
plus the canonical enumerations for meeting types, business functions,
meeting stages, and relationship statuses.

List-typed fields that may be written by multiple parallel nodes in the same
step (``errors``, ``sources``, ``external_signals``) use ``Annotated`` with
``operator.add`` so LangGraph concatenates rather than overwrites them.
"""

import operator
from typing import Annotated, TypedDict, Optional


# --- Canonical enumerations ---------------------------------------------------

VALID_MEETING_TYPES: set[str] = {
    "consulting_intro",
    "sales_discovery",
    "account_review",
}

VALID_FUNCTIONS: set[str] = {
    "finance",
    "accounting",
    "engineering",
    "product",
    "sales",
    "strategy",
    "it_data",
    "operations",
    "transformation",
    "general",  # fallback when normalize_function() cannot match a role
}

SENIORITY_LEVELS: list[str] = [
    "c_level",
    "vp",
    "director",
    "head",
    "manager",
    "individual_contributor",
]

# --- Engagement-context enumerations ------------------------------------------

VALID_MEETING_STAGES: set[str] = {
    "first_intro",
    "discovery_followup",
    "solution_discussion",
    "proposal_review",
    "negotiation",
    "client_kickoff",
    "account_expansion",
}

VALID_PREVIOUS_OUTCOMES: set[str] = {
    "positive",
    "neutral",
    "unclear",
    "no_previous_meeting",
}

VALID_RELATIONSHIP_STATUSES: set[str] = {
    "prospect",
    "active_opportunity",
    "client",
    "former_client",
}

VALID_RESEARCH_MODES: set[str] = {
    "full",           # first-intro / low-context — rich baseline research
    "light",          # follow-up / known-account — reduce baseline, keep signals
    "update_only",    # advanced stages — minimal baseline, focus on recent updates
}

VALID_INFERENCE_CONFIDENCE: set[str] = {
    "high",
    "medium",
    "low",
    "not_needed",     # all engagement fields were explicitly provided
}


# --- State definition ---------------------------------------------------------

class BriefingState(TypedDict):
    """Full shared state passed through every LangGraph node.

    Each agent owns only its designated slice — see CLAUDE.md for ownership map.

    Fields annotated with ``Annotated[list, operator.add]`` use a LangGraph
    *add-reducer*: when multiple parallel nodes write to the same list field
    in a single step, the values are concatenated instead of overwritten.
    This is required for the fan-out/fan-in pattern where three research
    agents run concurrently and may each append errors.
    """

    # Pre-intake (optional — populated only when intake agent was used)
    raw_meeting_context: Optional[str]
    intake_missing_fields: list
    intake_low_confidence_fields: list
    intake_confirmation_summary: Optional[str]

    # User inputs
    company_name: str
    domain: str
    meeting_type: str
    stakeholder_role: str
    meeting_goal: str
    contact_name: Optional[str]
    contact_title: Optional[str]

    # Engagement inputs (optional — None means "not provided by user")
    meeting_stage: Optional[str]
    previous_meeting_outcome: Optional[str]
    relationship_status: Optional[str]
    meeting_notes_summary: Optional[str]

    # User-side context (optional — extracted by intake or provided directly)
    user_role: Optional[str]
    user_company: Optional[str]
    user_function: Optional[str]
    engagement_type: Optional[str]
    desired_outcome: Optional[str]
    success_definition: Optional[str]

    # Deterministic derived fields (orchestrator)
    normalized_function: str
    normalized_seniority: Optional[str]

    # Engagement inference (LLM-inferred gap-fill for missing engagement inputs)
    engagement_inference: dict

    # Agent outputs
    research_plan: dict
    company_profile: dict
    external_signals: Annotated[list, operator.add]
    person_profile: dict
    audience_context: dict
    engagement_context: dict
    pain_point_hypotheses: dict
    meeting_strategy: dict
    final_brief: str

    # Metadata
    sources: Annotated[list, operator.add]
    brief_reliability: int
    cache_hit: bool
    errors: Annotated[list, operator.add]
