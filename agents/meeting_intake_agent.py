"""Meeting intake: turn a free-text description of a meeting into the pipeline's input fields.

Runs before the graph, not inside it. The result says which important fields are still
missing and which values the model was unsure about, so the UI can ask the user to check
them. Guesses that would mislead the pipeline are dropped rather than passed on.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from core.state import (
    INPUT_FIELDS,
    VALID_MEETING_STAGES,
    VALID_MEETING_TYPES,
    VALID_PREVIOUS_OUTCOMES,
    VALID_RELATIONSHIP_STATUSES,
)
from services.llm_client import structured_extract

logger = logging.getLogger(__name__)

# Fields the pipeline needs, and fields that make the brief noticeably better.
_REQUIRED = {
    "company_name": "Company name",
    "domain": "Company domain",
    "meeting_type": "Meeting type",
    "stakeholder_role": "Stakeholder role",
    "meeting_goal": "Meeting goal",
}
_RECOMMENDED = {"meeting_stage": "Meeting stage", "relationship_status": "Relationship status"}

_ENUM_VALUES = {
    "meeting_type": VALID_MEETING_TYPES,
    "meeting_stage": VALID_MEETING_STAGES,
    "relationship_status": VALID_RELATIONSHIP_STATUSES,
    "previous_meeting_outcome": VALID_PREVIOUS_OUTCOMES,
}
_ENGAGEMENT_FIELDS = ("meeting_stage", "relationship_status", "previous_meeting_outcome")

# engagement_type is free text, but common phrasings map to a known lens.
_KNOWN_ENGAGEMENT_TYPES = frozenset(
    {"sales", "consulting", "interview", "partnership", "investing", "recruiting"}
)
_ENGAGEMENT_TYPE_ALIASES = {
    "sale": "sales",
    "selling": "sales",
    "pitch": "sales",
    "business development": "sales",
    "bd": "sales",
    "consult": "consulting",
    "advisory": "consulting",
    "strategy": "consulting",
    "hiring": "recruiting",
    "recruitment": "recruiting",
    "investment": "investing",
    "venture": "investing",
    "vc": "investing",
    "job interview": "interview",
    "candidate": "interview",
}


class LowConfidenceField(BaseModel):
    """A single field that was extracted but with low confidence."""

    field_name: str = Field(description="Name of the field (e.g. 'domain', 'meeting_stage').")
    extracted_value: str = Field(description="The value that was extracted or inferred.")
    reason: str = Field(
        description="Why confidence is low (e.g. 'guessed from company name', 'ambiguous context')."
    )


class MeetingIntakeResult(BaseModel):
    """Structured extraction from free-text meeting context.

    Every field is optional.  The LLM should populate only what it can
    confidently extract or conservatively infer.  When evidence is
    insufficient, leave the field as null.
    """

    # --- Core workflow fields ------------------------------------------------
    company_name: str | None = Field(
        None,
        description=(
            "The company being met with.  Extract the official name if "
            "mentioned.  Return null if no company is identifiable."
        ),
    )
    domain: str | None = Field(
        None,
        description=(
            "The company's web domain (e.g. 'stripe.com').  If not "
            "explicitly stated, you may infer it for well-known companies.  "
            "Return null if uncertain — do NOT guess for obscure companies."
        ),
    )
    contact_name: str | None = Field(
        None,
        description=(
            "Full name of the specific person to be met.  Return null "
            "if no individual name is mentioned."
        ),
    )
    contact_title: str | None = Field(
        None,
        description=(
            "Job title of the contact if mentioned (e.g. 'Head of Accounting', "
            "'CFO').  Return null if not stated."
        ),
    )
    stakeholder_role: str | None = Field(
        None,
        description=(
            "The role or function of the meeting audience.  If a specific "
            "title is given, use it.  If only a function is implied "
            "(e.g. 'finance team'), use that.  Return null if indeterminate."
        ),
    )
    meeting_goal: str | None = Field(
        None,
        description=(
            "What the user wants to achieve in this meeting.  Summarise "
            "concisely from the context.  Return null if no goal is stated "
            "or implied."
        ),
    )
    meeting_type: str | None = Field(
        None,
        description=(
            "One of: 'consulting_intro', 'sales_discovery', 'account_review'.  "
            "Infer from context if possible (e.g. mentions of selling/pitching "
            "→ sales_discovery; mentions of advisory/consulting → consulting_intro; "
            "mentions of existing client review → account_review).  "
            "Return null if ambiguous."
        ),
    )

    # --- Engagement fields ---------------------------------------------------
    meeting_stage: str | None = Field(
        None,
        description=(
            "One of: 'first_intro', 'discovery_followup', 'solution_discussion', "
            "'proposal_review', 'negotiation', 'client_kickoff', 'account_expansion'.  "
            "Infer from mentions of 'first meeting', 'follow up', 'pricing', "
            "'kick off', 'expand', etc.  Return null if ambiguous."
        ),
    )
    relationship_status: str | None = Field(
        None,
        description=(
            "One of: 'prospect', 'active_opportunity', 'client', 'former_client'.  "
            "Infer from context (e.g. 'existing client' → client; "
            "'never met before' → prospect).  Return null if ambiguous."
        ),
    )
    previous_meeting_outcome: str | None = Field(
        None,
        description=(
            "One of: 'positive', 'neutral', 'unclear', 'no_previous_meeting'.  "
            "Infer from mentions of prior interactions and their tone.  "
            "Return null if no prior interaction is mentioned or outcome is unclear."
        ),
    )
    meeting_notes_summary: str | None = Field(
        None,
        description=(
            "A concise 1-3 sentence summary of prior interaction context, "
            "cleaned and structured from the raw text.  Not a copy-paste of "
            "the input — distil the relevant meeting history.  "
            "Return null if no prior interaction is described."
        ),
    )

    # --- User-side context ---------------------------------------------------
    user_role: str | None = Field(
        None,
        description=(
            "The user's own role or title (e.g. 'Account Executive', "
            "'Managing Consultant', 'Founder', 'Student').  "
            "Extract if explicitly stated.  Return null if not mentioned."
        ),
    )
    user_company: str | None = Field(
        None,
        description=(
            "The company or organisation the user is from.  "
            "Extract if explicitly stated (e.g. 'I work at Deloitte').  "
            "Return null if not mentioned."
        ),
    )
    user_function: str | None = Field(
        None,
        description=(
            "The user's functional area (e.g. 'sales', 'consulting', "
            "'product', 'investing', 'recruiting').  Infer from user_role "
            "or context if reasonable.  Return null if ambiguous."
        ),
    )
    engagement_type: str | None = Field(
        None,
        description=(
            "The interaction lens from the user's side.  One of: 'sales', "
            "'consulting', 'interview', 'partnership', 'investing', "
            "'recruiting', or a free-text value.  Infer from context "
            "(e.g. 'pitch' → sales, 'advisory engagement' → consulting, "
            "'job interview' → interview).  Return null if ambiguous."
        ),
    )
    desired_outcome: str | None = Field(
        None,
        description=(
            "What the user wants the meeting to achieve — more specific "
            "than meeting_goal.  E.g. 'get verbal agreement to proceed "
            "with a pilot', 'qualify whether they have budget'.  "
            "Extract or infer conservatively.  Return null if not stated."
        ),
    )
    success_definition: str | None = Field(
        None,
        description=(
            "What a successful meeting would look like from the user's "
            "perspective.  E.g. 'they agree to a technical deep-dive next "
            "week', 'I leave with a clear understanding of their needs'.  "
            "Extract if stated.  Return null if not mentioned."
        ),
    )

    # --- Metadata for confirmation UI ----------------------------------------
    low_confidence_fields: list[LowConfidenceField] = Field(
        default_factory=list,
        description=(
            "Fields that were extracted or inferred but with LOW confidence.  "
            "Include any field where you are uncertain about the value.  "
            "Examples: domain guessed from company name, meeting_type inferred "
            "from vague context, meeting_stage inferred ambiguously."
        ),
    )
    confirmation_summary: str = Field(
        description=(
            "A concise, human-readable summary (3-6 sentences) of what was "
            "extracted and what is still unclear.  Written as if speaking to "
            "the user: 'I understood that you are meeting with [company], "
            "specifically [contact].  The meeting appears to be a [type].  "
            "I was unable to determine [missing fields].  "
            "I am not confident about [low-confidence fields].'  "
            "Be honest about what you don't know."
        ),
    )


_SYSTEM_PROMPT = """\
You are a meeting-context extraction assistant.  You will receive a
free-text description of an upcoming business meeting.  Your task is to
extract structured fields from this text for a pre-meeting intelligence
system.

STRICT RULES:

1. PREFER EXTRACTION OVER INFERENCE.
   If a fact is explicitly stated ("meeting with Stripe's CFO"), extract it.
   If a fact must be inferred ("they seemed interested" → positive outcome),
   do so conservatively and mark the field as low confidence.

2. RETURN NULL WHEN UNSUPPORTED.
   If a field cannot be extracted or reasonably inferred, return null.
   Do NOT hallucinate company names, people, or meeting stages.

3. DOMAIN INFERENCE.
   For well-known companies (Google, Stripe, Salesforce, etc.), you may
   infer the domain (google.com, stripe.com, salesforce.com) and mark it
   as low confidence.  For lesser-known companies, return null.

4. MEETING TYPE.
   Must be one of: consulting_intro, sales_discovery, account_review.
   If the text mentions selling, pitching, or demos → sales_discovery.
   If it mentions consulting, advisory, or strategy → consulting_intro.
   If it mentions reviewing existing work or account status → account_review.
   If ambiguous, return null.

5. MEETING STAGE.
   Must be one of: first_intro, discovery_followup, solution_discussion,
   proposal_review, negotiation, client_kickoff, account_expansion.
   Look for cues:
   - "first meeting" / "initial" / "intro" → first_intro
   - "second meeting" / "follow up" / "continue discussion" → discovery_followup
   - "discuss solution" / "demo" / "show approach" → solution_discussion
   - "review proposal" / "pricing" / "quote" → proposal_review
   - "negotiate" / "terms" / "contract" → negotiation
   - "kick off" / "onboard" / "start project" → client_kickoff
   - "expand" / "new area" / "additional services" → account_expansion

6. RELATIONSHIP STATUS.
   Must be one of: prospect, active_opportunity, client, former_client.
   Infer from context.  If ambiguous, return null.

7. PREVIOUS MEETING OUTCOME.
   Must be one of: positive, neutral, unclear, no_previous_meeting.
   Only infer if the text describes a prior interaction and its tone.
   "seemed interested" → positive.  "didn't commit" → neutral.
   "no prior contact" → no_previous_meeting.

8. MEETING NOTES SUMMARY.
   Produce a clean, concise 1-3 sentence summary of relevant prior
   interaction context.  Do NOT copy-paste the raw input.

9. LOW-CONFIDENCE FIELDS.
   Any field where your extraction required significant inference or
   where the evidence is ambiguous MUST be listed here.  Include the
   field name, the value you extracted, and the reason for low confidence.

10. CONFIRMATION SUMMARY.
    Write a concise human-readable summary addressed to the user.
    Clearly state what you understood, what you could not determine,
    and what you are uncertain about.  When user-side context is
    extracted, mention it briefly: "You are approaching this as a
    [engagement_type] meeting from [user_company]."

11. STAKEHOLDER ROLE.
    If a contact_title is extracted (e.g. "Head of Accounting"), also
    use it as stakeholder_role unless a different broader role description
    is given.  stakeholder_role should describe who the meeting audience is.

12. USER-SIDE CONTEXT.
    The text may describe who the USER is (not the contact being met).
    Look for cues like "I'm a consultant at…", "our sales team…",
    "I work in product at…", "we want to close…", "success would be…".
    - user_role: the user's own title/role
    - user_company: the user's own company
    - user_function: the user's functional area (sales, consulting, etc.)
    - engagement_type: the user's interaction lens (sales, consulting,
      interview, partnership, investing, recruiting).  This is NOT the
      same as meeting_type — meeting_type is a workflow enum
      (consulting_intro / sales_discovery / account_review), while
      engagement_type is the user's broader perspective.  They may
      correlate but serve different purposes.
    - desired_outcome: a SPECIFIC result the user wants from THIS
      meeting (e.g. "get verbal agreement for a pilot").  This is more
      concrete than meeting_goal.  If it would be identical to
      meeting_goal, leave desired_outcome null — avoid duplication.
    - success_definition: what a successful meeting looks like from
      the user's perspective.  Only extract if explicitly stated.
    These are OPTIONAL — only extract what is clearly stated or
    conservatively inferable.  Mark as low confidence when inferred.
    Do NOT confuse the user's company/role with the contact's company/role.
"""


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _missing(fields: dict[str, Any]) -> list[str]:
    missing = [label for key, label in _REQUIRED.items() if _is_blank(fields.get(key))]
    missing += [
        f"{label} (recommended)"
        for key, label in _RECOMMENDED.items()
        if _is_blank(fields.get(key))
    ]
    return missing


def _workflow_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """The pipeline's input fields; the required text ones are never None."""
    return {
        key: (fields.get(key) or "") if key in _REQUIRED else fields.get(key)
        for key in INPUT_FIELDS
    }


def _result(
    raw_text: str, fields: dict[str, Any], low_confidence: list[dict[str, str]], summary: str
) -> dict[str, Any]:
    return {
        **_workflow_fields(fields),
        "raw_meeting_context": raw_text,
        "intake_missing_fields": _missing(fields),
        "intake_low_confidence_fields": low_confidence,
        "intake_confirmation_summary": summary,
    }


def _normalise(extracted: dict[str, Any]) -> None:
    """Drop enum values the pipeline does not know, and tidy engagement_type and the role."""
    for key, allowed in _ENUM_VALUES.items():
        if extracted.get(key) and extracted[key] not in allowed:
            extracted[key] = None

    engagement_type = (extracted.get("engagement_type") or "").strip().lower()
    if engagement_type and engagement_type not in _KNOWN_ENGAGEMENT_TYPES:
        extracted["engagement_type"] = _ENGAGEMENT_TYPE_ALIASES.get(
            engagement_type, extracted["engagement_type"]
        )

    if not extracted.get("stakeholder_role") and extracted.get("contact_title"):
        extracted["stakeholder_role"] = extracted["contact_title"]


def _drop_shaky_guesses(extracted: dict[str, Any], raw_text: str) -> list[dict[str, str]]:
    """Null the low-confidence values that would mislead the pipeline; return the flagged fields.

    A wrong domain would steer every web search, so a guessed domain is dropped, unless the
    user wrote it themselves (small models sometimes flag stated values). Guessed engagement
    fields are left to the engagement-inference node, which sees more context.
    """
    flagged = [
        {"field": item["field_name"], "value": item["extracted_value"], "reason": item["reason"]}
        for item in extracted["low_confidence_fields"]
    ]
    domain = extracted.get("domain")
    if domain and domain.lower() in raw_text.lower():
        flagged = [item for item in flagged if item["field"] != "domain"]

    flagged_fields = {item["field"] for item in flagged}
    for key in ("domain", *_ENGAGEMENT_FIELDS):
        if key in flagged_fields and extracted.get(key):
            logger.info("Dropping low-confidence %s = %r", key, extracted[key])
            extracted[key] = None
    return flagged


def run(raw_meeting_context: str) -> dict[str, Any]:
    """Parse a meeting description into the pipeline's input fields plus review metadata."""
    if not raw_meeting_context or not raw_meeting_context.strip():
        return _result(raw_meeting_context or "", {}, [], "No meeting context was provided.")

    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=raw_meeting_context.strip(),
        response_model=MeetingIntakeResult,
    )
    if result is None:
        logger.warning("meeting_intake_agent: LLM extraction failed.")
        summary = "LLM extraction failed — please provide structured fields manually."
        return _result(raw_meeting_context, {}, [], summary)

    extracted = result.model_dump()
    _normalise(extracted)
    low_confidence = _drop_shaky_guesses(extracted, raw_meeting_context)
    return _result(
        raw_meeting_context, extracted, low_confidence, extracted["confirmation_summary"]
    )
