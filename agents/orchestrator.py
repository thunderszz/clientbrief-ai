"""First node: validate and normalise the input, and give every later field its empty value.

Deterministic. Later nodes can rely on every key being present. Validation problems go to
``errors``; the run carries on, and the brief's reliability score reflects them.
"""

from __future__ import annotations

from typing import Any

from agents.audience_context_agent import empty_context as empty_audience_context
from agents.company_research import empty_profile as empty_company_profile
from agents.engagement_context_agent import empty_context as empty_engagement_context
from agents.engagement_inference_agent import empty_inference
from agents.meeting_strategy_agent import empty_strategy
from agents.pain_point_agent import empty_hypotheses
from agents.person_research import empty_profile as empty_person_profile
from core.state import INPUT_FIELDS
from core.utils import normalize_domain, normalize_function, normalize_seniority, validate_inputs

_REQUIRED_TEXT = ("company_name", "stakeholder_role", "meeting_goal")
_OPTIONAL_TEXT = ("contact_name", "contact_title", "meeting_notes_summary")


def _empty_outputs() -> dict[str, Any]:
    return {
        "engagement_inference": empty_inference(),
        "research_plan": {},
        "company_profile": empty_company_profile(),
        "external_signals": [],
        "person_profile": empty_person_profile(),
        "audience_context": empty_audience_context(),
        "engagement_context": empty_engagement_context(),
        "pain_point_hypotheses": empty_hypotheses(),
        "meeting_strategy": empty_strategy(),
        "final_brief": "",
        "sources": [],
        "brief_reliability": 0,
        "reliability_report": {},
    }


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalise the meeting input, and initialise the rest of the state."""
    fields = {key: state.get(key) for key in INPUT_FIELDS}
    for key in _REQUIRED_TEXT:
        fields[key] = (fields[key] or "").strip()
    for key in _OPTIONAL_TEXT:
        fields[key] = fields[key].strip() if fields[key] else None
    fields["meeting_type"] = fields["meeting_type"] or ""
    fields["meeting_stage"] = fields["meeting_stage"] or None
    fields["domain"] = normalize_domain(fields["domain"]) if fields["domain"] else ""

    errors = validate_inputs(
        fields["company_name"],
        fields["domain"],
        fields["meeting_type"],
        fields["stakeholder_role"],
        meeting_stage=fields["meeting_stage"],
    )
    if errors:
        function, seniority = "general", None
    else:
        # The contact's own title says more about their function than the generic role.
        role = fields["contact_title"] or fields["stakeholder_role"]
        function, seniority = normalize_function(role), normalize_seniority(role)

    return {
        **fields,
        "raw_meeting_context": state.get("raw_meeting_context"),
        "intake_missing_fields": state.get("intake_missing_fields") or [],
        "intake_low_confidence_fields": state.get("intake_low_confidence_fields") or [],
        "intake_confirmation_summary": state.get("intake_confirmation_summary"),
        "normalized_function": function,
        "normalized_seniority": seniority,
        **_empty_outputs(),
        "errors": errors,
    }
