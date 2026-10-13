"""The shared state passed between pipeline nodes, and the values its enum-like fields take.

Each node returns a partial update with the slice it owns. The list fields that the three
parallel research nodes may all write to (``errors``, ``sources``, ``external_signals``,
``llm_calls``) use an add-reducer, so LangGraph concatenates their updates.
"""

import operator
from typing import Annotated, TypedDict

VALID_MEETING_TYPES = frozenset({"consulting_intro", "sales_discovery", "account_review"})

VALID_MEETING_STAGES = frozenset(
    {
        "first_intro",
        "discovery_followup",
        "solution_discussion",
        "proposal_review",
        "negotiation",
        "client_kickoff",
        "account_expansion",
    }
)

VALID_RELATIONSHIP_STATUSES = frozenset({"prospect", "active_opportunity", "client", "former_client"})

VALID_PREVIOUS_OUTCOMES = frozenset({"positive", "neutral", "unclear", "no_previous_meeting"})

# What the user tells us about the meeting, in the order the UI and the API present it.
INPUT_FIELDS = (
    "company_name",
    "domain",
    "meeting_type",
    "stakeholder_role",
    "meeting_goal",
    "contact_name",
    "contact_title",
    "meeting_stage",
    "relationship_status",
    "previous_meeting_outcome",
    "meeting_notes_summary",
    "user_role",
    "user_company",
    "user_function",
    "engagement_type",
    "desired_outcome",
    "success_definition",
)


class BriefingState(TypedDict):
    # Set only when the meeting was described in free text and parsed by the intake agent.
    raw_meeting_context: str | None
    intake_missing_fields: list
    intake_low_confidence_fields: list
    intake_confirmation_summary: str | None

    # The meeting.
    company_name: str
    domain: str
    meeting_type: str
    stakeholder_role: str
    meeting_goal: str
    contact_name: str | None
    contact_title: str | None

    # Where the relationship stands. None means the user did not say, and the
    # engagement nodes infer or default it.
    meeting_stage: str | None
    previous_meeting_outcome: str | None
    relationship_status: str | None
    meeting_notes_summary: str | None

    # Who the user is and what they want. It frames the analysis; it is not evidence.
    user_role: str | None
    user_company: str | None
    user_function: str | None
    engagement_type: str | None
    desired_outcome: str | None
    success_definition: str | None

    # Derived from the contact's title, or the stakeholder role when there is none.
    normalized_function: str  # finance, accounting, engineering, ... or "general"
    normalized_seniority: str | None  # c_level, vp, director, head, manager, individual_contributor

    # Node outputs, in pipeline order.
    engagement_inference: dict
    engagement_context: dict
    research_plan: dict
    company_profile: dict
    external_signals: Annotated[list, operator.add]
    person_profile: dict
    audience_context: dict
    pain_point_hypotheses: dict
    meeting_strategy: dict
    final_brief: str

    sources: Annotated[list, operator.add]
    brief_reliability: int  # 0-5, derived from the 0-100 score in reliability_report
    reliability_report: dict
    errors: Annotated[list, operator.add]
    llm_calls: Annotated[list, operator.add]  # one record per LLM call, tagged with its node
