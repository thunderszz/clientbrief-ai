"""Engagement Inference Agent — LLM-assisted gap-fill for missing engagement inputs.

When the user provides all engagement fields (meeting_stage, relationship_status,
previous_meeting_outcome), this agent short-circuits and returns a sparse
``engagement_inference`` with ``confidence: "not_needed"``.

When one or more fields are missing, the agent uses LLM reasoning over the
available context (meeting_goal, stakeholder_role, meeting_type,
meeting_notes_summary, contact_title) to infer plausible values.

The inference object is stored separately from:
- the raw user inputs (``meeting_stage``, ``relationship_status``, etc.)
- the final normalised ``engagement_context`` (produced by the context agent)

This separation preserves full traceability so a future CLI/UI can show:
- what the user provided
- what was inferred (and with what confidence)
- what the system resolved as final values

Ownership:  ``engagement_inference`` only.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from services.llm_client import structured_extract

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema for LLM output
# ---------------------------------------------------------------------------

class EngagementInference(BaseModel):
    """Inferred engagement fields for gaps not provided by the user.

    Each inferred field should only be populated when the agent has
    meaningful evidence from the context.  When evidence is weak,
    leave the field as null rather than guessing.
    """

    inferred_meeting_stage: Optional[str] = Field(
        None,
        description=(
            "Inferred meeting stage.  Must be one of: first_intro, "
            "discovery_followup, solution_discussion, proposal_review, "
            "negotiation, client_kickoff, account_expansion.  "
            "Return null if no meaningful signal exists."
        ),
    )
    inferred_relationship_status: Optional[str] = Field(
        None,
        description=(
            "Inferred relationship status.  Must be one of: prospect, "
            "active_opportunity, client, former_client.  "
            "Return null if no meaningful signal exists."
        ),
    )
    inferred_previous_meeting_outcome: Optional[str] = Field(
        None,
        description=(
            "Inferred outcome of any previous interaction.  Must be one of: "
            "positive, neutral, unclear, no_previous_meeting.  "
            "Return null if no meaningful signal exists."
        ),
    )
    confidence: str = Field(
        description=(
            "Overall confidence in the inferences: "
            "'high' = strong signals clearly indicate the engagement state; "
            "'medium' = moderate signals suggest a plausible interpretation; "
            "'low' = weak signals, inferences are speculative."
        )
    )
    reasoning_summary: str = Field(
        description=(
            "1-3 sentence explanation of what signals led to the inferences.  "
            "If there is no meaningful signal, say so explicitly."
        )
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an engagement-context analyst.  You will receive information about
an upcoming meeting: the meeting goal, meeting type, stakeholder role, and
optionally prior meeting notes or a contact title.

Some engagement fields (meeting_stage, relationship_status,
previous_meeting_outcome) were NOT provided by the user.  Your task is to
infer the most plausible values for the MISSING fields based on the
available context.

STRICT RULES:
1. Only infer fields listed as "MISSING" — never re-derive provided fields.
2. Use the meeting_goal text as the primary signal.  Look for cues like:
   - "follow up" / "check in" → likely discovery_followup or later
   - "discuss proposal" / "review pricing" → likely proposal_review
   - "kick off" / "onboard" → likely client_kickoff
   - "expand" / "new area" / "cross-sell" → likely account_expansion
   - "negotiate" / "terms" / "contract" → likely negotiation
   - "introduce" / "first meeting" / "explore" → likely first_intro
3. Use meeting_notes_summary as a strong signal if present — it often
   reveals prior interaction history.
4. meeting_type provides context:
   - account_review often implies client or active_opportunity
   - consulting_intro often implies prospect
   - sales_discovery can be prospect or active_opportunity
5. Do NOT hallucinate certainty.  If evidence is ambiguous:
   - set confidence to "low"
   - leave inferred fields as null rather than guessing
   - say so in the reasoning_summary
6. For relationship_status, consider:
   - if meeting_stage is a late stage → likely active_opportunity or client
   - if meeting_type is account_review → likely client
   - if meeting_goal mentions existing relationship → likely client
7. For previous_meeting_outcome, only infer if notes or goal text
   clearly reference a prior interaction and its outcome.
   Default to null — do not guess.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY_INFERENCE: dict[str, Any] = {
    "inferred_meeting_stage": None,
    "inferred_relationship_status": None,
    "inferred_previous_meeting_outcome": None,
    "confidence": "not_needed",
    "reasoning_summary": "",
}

# Fields that, when all are present, mean no inference is needed.
_ENGAGEMENT_FIELDS = ("meeting_stage", "relationship_status", "previous_meeting_outcome")


def _all_provided(state: dict[str, Any]) -> bool:
    """Return True if every engagement field was explicitly provided by the user."""
    return all(state.get(f) is not None for f in _ENGAGEMENT_FIELDS)


def _build_user_prompt(state: dict[str, Any]) -> str:
    """Assemble the user prompt, indicating which fields are missing."""
    parts: list[str] = []

    # Meeting context
    parts.append(f"Meeting type: {state.get('meeting_type', 'unknown')}")
    parts.append(f"Meeting goal: {state.get('meeting_goal', '')}")
    parts.append(f"Stakeholder role: {state.get('stakeholder_role', 'unknown')}")

    contact_title = state.get("contact_title")
    if contact_title:
        parts.append(f"Contact title: {contact_title}")

    notes = state.get("meeting_notes_summary")
    if notes:
        parts.append(f"Meeting notes from prior interaction: {notes}")
    else:
        parts.append("Meeting notes: (none provided)")

    # Indicate which fields are provided vs missing
    parts.append("")
    parts.append("--- Engagement field status ---")

    meeting_stage = state.get("meeting_stage")
    if meeting_stage:
        parts.append(f"meeting_stage: PROVIDED = {meeting_stage}  (do NOT re-infer)")
    else:
        parts.append("meeting_stage: MISSING — please infer")

    relationship_status = state.get("relationship_status")
    if relationship_status:
        parts.append(f"relationship_status: PROVIDED = {relationship_status}  (do NOT re-infer)")
    else:
        parts.append("relationship_status: MISSING — please infer")

    previous_outcome = state.get("previous_meeting_outcome")
    if previous_outcome:
        parts.append(f"previous_meeting_outcome: PROVIDED = {previous_outcome}  (do NOT re-infer)")
    else:
        parts.append("previous_meeting_outcome: MISSING — please infer")

    parts.append("--- End engagement field status ---")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run(state: dict[str, Any]) -> dict[str, Any]:
    """Infer missing engagement fields from available meeting context.

    If all engagement fields are already provided, returns a sparse
    inference object with ``confidence: "not_needed"`` and skips the LLM.

    Writes only ``engagement_inference``.
    """
    new_errors: list[str] = []

    # --- Fast exit: all fields already provided ---
    if _all_provided(state):
        return {
            "engagement_inference": {
                **_EMPTY_INFERENCE,
                "confidence": "not_needed",
                "reasoning_summary": "All engagement fields were explicitly provided by the user.",
            },
        }

    # --- At least one field is missing → use LLM to infer ---
    user_prompt = _build_user_prompt(state)

    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=EngagementInference,
    )

    if result is None:
        new_errors.append(
            "engagement_inference_agent: LLM inference failed, "
            "returning empty inference.  Engagement context will use defaults."
        )
        return {
            "engagement_inference": {
                **_EMPTY_INFERENCE,
                "confidence": "low",
                "reasoning_summary": "LLM inference failed — no inferred values available.",
            },
            "errors": new_errors,
        }

    inference = result.model_dump()

    # Sanitise: only allow known enum values for inferred fields.
    # If the LLM returned something unexpected, null it out.
    from core.state import (
        VALID_MEETING_STAGES,
        VALID_RELATIONSHIP_STATUSES,
        VALID_PREVIOUS_OUTCOMES,
    )

    if inference.get("inferred_meeting_stage") and \
       inference["inferred_meeting_stage"] not in VALID_MEETING_STAGES:
        inference["inferred_meeting_stage"] = None

    if inference.get("inferred_relationship_status") and \
       inference["inferred_relationship_status"] not in VALID_RELATIONSHIP_STATUSES:
        inference["inferred_relationship_status"] = None

    if inference.get("inferred_previous_meeting_outcome") and \
       inference["inferred_previous_meeting_outcome"] not in VALID_PREVIOUS_OUTCOMES:
        inference["inferred_previous_meeting_outcome"] = None

    # Don't let inference override explicitly provided fields —
    # null out any inferred value for a field the user already set.
    if state.get("meeting_stage") is not None:
        inference["inferred_meeting_stage"] = None
    if state.get("relationship_status") is not None:
        inference["inferred_relationship_status"] = None
    if state.get("previous_meeting_outcome") is not None:
        inference["inferred_previous_meeting_outcome"] = None

    return {"engagement_inference": inference}
