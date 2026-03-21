"""Meeting Intake Agent — LLM-powered free-text-to-structured-fields parser.

This agent is a **pre-workflow** layer.  It is called *before* the main
LangGraph pipeline.  Its job is to convert a free-text meeting description
into the structured fields that the existing workflow expects.

Typical flow::

    1. User provides raw free text  (e.g. "Second meeting with Stripe CFO…")
    2. Caller invokes  ``meeting_intake_agent.run(raw_text)``
    3. Agent returns:
       - extracted structured fields  (company_name, domain, …)
       - list of important fields still missing
       - list of fields inferred with low confidence
       - a human-readable confirmation summary
    4. Future CLI/UI shows confirmation, lets user edit/fill gaps
    5. Caller feeds the final structured dict to  ``app.invoke({…})``

The agent does NOT write to BriefingState directly.  It returns a plain
dict that the caller merges into the workflow input.

Ownership:  standalone pre-processing — not a LangGraph node.
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

class LowConfidenceField(BaseModel):
    """A single field that was extracted but with low confidence."""

    field_name: str = Field(
        description="Name of the field (e.g. 'domain', 'meeting_stage')."
    )
    extracted_value: str = Field(
        description="The value that was extracted or inferred."
    )
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
    company_name: Optional[str] = Field(
        None,
        description=(
            "The company being met with.  Extract the official name if "
            "mentioned.  Return null if no company is identifiable."
        ),
    )
    domain: Optional[str] = Field(
        None,
        description=(
            "The company's web domain (e.g. 'stripe.com').  If not "
            "explicitly stated, you may infer it for well-known companies.  "
            "Return null if uncertain — do NOT guess for obscure companies."
        ),
    )
    contact_name: Optional[str] = Field(
        None,
        description=(
            "Full name of the specific person to be met.  Return null "
            "if no individual name is mentioned."
        ),
    )
    contact_title: Optional[str] = Field(
        None,
        description=(
            "Job title of the contact if mentioned (e.g. 'Head of Accounting', "
            "'CFO').  Return null if not stated."
        ),
    )
    stakeholder_role: Optional[str] = Field(
        None,
        description=(
            "The role or function of the meeting audience.  If a specific "
            "title is given, use it.  If only a function is implied "
            "(e.g. 'finance team'), use that.  Return null if indeterminate."
        ),
    )
    meeting_goal: Optional[str] = Field(
        None,
        description=(
            "What the user wants to achieve in this meeting.  Summarise "
            "concisely from the context.  Return null if no goal is stated "
            "or implied."
        ),
    )
    meeting_type: Optional[str] = Field(
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
    meeting_stage: Optional[str] = Field(
        None,
        description=(
            "One of: 'first_intro', 'discovery_followup', 'solution_discussion', "
            "'proposal_review', 'negotiation', 'client_kickoff', 'account_expansion'.  "
            "Infer from mentions of 'first meeting', 'follow up', 'pricing', "
            "'kick off', 'expand', etc.  Return null if ambiguous."
        ),
    )
    relationship_status: Optional[str] = Field(
        None,
        description=(
            "One of: 'prospect', 'active_opportunity', 'client', 'former_client'.  "
            "Infer from context (e.g. 'existing client' → client; "
            "'never met before' → prospect).  Return null if ambiguous."
        ),
    )
    previous_meeting_outcome: Optional[str] = Field(
        None,
        description=(
            "One of: 'positive', 'neutral', 'unclear', 'no_previous_meeting'.  "
            "Infer from mentions of prior interactions and their tone.  "
            "Return null if no prior interaction is mentioned or outcome is unclear."
        ),
    )
    meeting_notes_summary: Optional[str] = Field(
        None,
        description=(
            "A concise 1-3 sentence summary of prior interaction context, "
            "cleaned and structured from the raw text.  Not a copy-paste of "
            "the input — distil the relevant meeting history.  "
            "Return null if no prior interaction is described."
        ),
    )

    # --- User-side context ---------------------------------------------------
    user_role: Optional[str] = Field(
        None,
        description=(
            "The user's own role or title (e.g. 'Account Executive', "
            "'Managing Consultant', 'Founder', 'Student').  "
            "Extract if explicitly stated.  Return null if not mentioned."
        ),
    )
    user_company: Optional[str] = Field(
        None,
        description=(
            "The company or organisation the user is from.  "
            "Extract if explicitly stated (e.g. 'I work at Deloitte').  "
            "Return null if not mentioned."
        ),
    )
    user_function: Optional[str] = Field(
        None,
        description=(
            "The user's functional area (e.g. 'sales', 'consulting', "
            "'product', 'investing', 'recruiting').  Infer from user_role "
            "or context if reasonable.  Return null if ambiguous."
        ),
    )
    engagement_type: Optional[str] = Field(
        None,
        description=(
            "The interaction lens from the user's side.  One of: 'sales', "
            "'consulting', 'interview', 'partnership', 'investing', "
            "'recruiting', or a free-text value.  Infer from context "
            "(e.g. 'pitch' → sales, 'advisory engagement' → consulting, "
            "'job interview' → interview).  Return null if ambiguous."
        ),
    )
    desired_outcome: Optional[str] = Field(
        None,
        description=(
            "What the user wants the meeting to achieve — more specific "
            "than meeting_goal.  E.g. 'get verbal agreement to proceed "
            "with a pilot', 'qualify whether they have budget'.  "
            "Extract or infer conservatively.  Return null if not stated."
        ),
    )
    success_definition: Optional[str] = Field(
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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

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
    and what you are uncertain about.

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

13. CONFIRMATION SUMMARY (updated).
    When user-side context is extracted, mention it briefly in the
    summary: "You are approaching this as a [engagement_type] meeting
    from [user_company]."
"""


# ---------------------------------------------------------------------------
# Important workflow fields for missing-fields detection
# ---------------------------------------------------------------------------

# These are the fields the main workflow requires or strongly benefits from.
# If any of these are null after intake, they are reported as missing.
_IMPORTANT_FIELDS: list[tuple[str, str]] = [
    ("company_name", "Company name"),
    ("domain", "Company domain"),
    ("meeting_type", "Meeting type"),
    ("stakeholder_role", "Stakeholder role"),
    ("meeting_goal", "Meeting goal"),
]

# Optional but highly useful fields.
_RECOMMENDED_FIELDS: list[tuple[str, str]] = [
    ("meeting_stage", "Meeting stage"),
    ("relationship_status", "Relationship status"),
]


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run(raw_meeting_context: str) -> dict[str, Any]:
    """Parse free-text meeting context into structured workflow fields.

    Parameters
    ----------
    raw_meeting_context:
        A free-text description of the upcoming meeting.  Can range from
        one sentence to several paragraphs.

    Returns
    -------
    dict with keys:
        - All extracted workflow fields (company_name, domain, … etc.)
          Each is a string or None.
        - ``raw_meeting_context``: the original text (preserved for audit)
        - ``intake_missing_fields``: list[str] of important fields still null
        - ``intake_low_confidence_fields``: list[dict] with field/value/reason
        - ``intake_confirmation_summary``: human-readable summary string
    """

    if not raw_meeting_context or not raw_meeting_context.strip():
        return _empty_result(raw_meeting_context or "")

    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=raw_meeting_context.strip(),
        response_model=MeetingIntakeResult,
    )

    if result is None:
        logger.warning("meeting_intake_agent: LLM extraction failed.")
        return _empty_result(
            raw_meeting_context,
            error="LLM extraction failed — please provide structured fields manually.",
        )

    # --- Build the output dict ------------------------------------------------
    extracted = result.model_dump()

    # Sanitise enum fields against known values.
    from core.state import (
        VALID_MEETING_TYPES,
        VALID_MEETING_STAGES,
        VALID_RELATIONSHIP_STATUSES,
        VALID_PREVIOUS_OUTCOMES,
    )

    if extracted.get("meeting_type") and extracted["meeting_type"] not in VALID_MEETING_TYPES:
        extracted["meeting_type"] = None
    if extracted.get("meeting_stage") and extracted["meeting_stage"] not in VALID_MEETING_STAGES:
        extracted["meeting_stage"] = None
    if extracted.get("relationship_status") and extracted["relationship_status"] not in VALID_RELATIONSHIP_STATUSES:
        extracted["relationship_status"] = None
    if extracted.get("previous_meeting_outcome") and extracted["previous_meeting_outcome"] not in VALID_PREVIOUS_OUTCOMES:
        extracted["previous_meeting_outcome"] = None

    # Soft-normalise engagement_type to known values when close enough.
    _KNOWN_ENGAGEMENT_TYPES = {
        "sales", "consulting", "interview", "partnership",
        "investing", "recruiting",
    }
    _ENGAGEMENT_TYPE_ALIASES: dict[str, str] = {
        "sale": "sales", "selling": "sales", "pitch": "sales",
        "business development": "sales", "bd": "sales",
        "consult": "consulting", "advisory": "consulting",
        "strategy": "consulting",
        "hiring": "recruiting", "recruitment": "recruiting",
        "investment": "investing", "venture": "investing", "vc": "investing",
        "job interview": "interview", "candidate": "interview",
    }
    raw_et = (extracted.get("engagement_type") or "").strip().lower()
    if raw_et and raw_et not in _KNOWN_ENGAGEMENT_TYPES:
        normalized_et = _ENGAGEMENT_TYPE_ALIASES.get(raw_et)
        if normalized_et:
            extracted["engagement_type"] = normalized_et
        # else: leave free-text value as-is — downstream agents handle it

    # Stakeholder role fallback: if stakeholder_role is empty but contact_title
    # is present, use contact_title as stakeholder_role.
    if not extracted.get("stakeholder_role") and extracted.get("contact_title"):
        extracted["stakeholder_role"] = extracted["contact_title"]

    # --- Format low-confidence fields -----------------------------------------
    low_confidence: list[dict[str, str]] = []
    low_confidence_names: set[str] = set()
    for lc in extracted.get("low_confidence_fields", []):
        if isinstance(lc, dict):
            fname = lc.get("field_name", "")
            low_confidence.append({
                "field": fname,
                "value": lc.get("extracted_value", ""),
                "reason": lc.get("reason", ""),
            })
            low_confidence_names.add(fname)

    # --- Null out low-confidence domain (avoid poisoning research queries) ----
    if "domain" in low_confidence_names and extracted.get("domain"):
        logger.info(
            "meeting_intake_agent: nulling low-confidence domain '%s'",
            extracted["domain"],
        )
        extracted["domain"] = None

    # --- Null out low-confidence engagement fields ----------------------------
    # These fields should be handled by engagement_inference_agent which has
    # richer context.  Letting a shaky intake guess masquerade as explicit user
    # input would bypass the inference layer entirely.
    _ENGAGEMENT_FIELDS = ("meeting_stage", "relationship_status", "previous_meeting_outcome")
    for ef in _ENGAGEMENT_FIELDS:
        if ef in low_confidence_names and extracted.get(ef):
            logger.info(
                "meeting_intake_agent: nulling low-confidence engagement field '%s' = '%s'",
                ef, extracted[ef],
            )
            extracted[ef] = None

    # --- Compute missing fields -----------------------------------------------
    missing: list[str] = []
    for field_key, field_label in _IMPORTANT_FIELDS:
        val = extracted.get(field_key)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing.append(field_label)

    for field_key, field_label in _RECOMMENDED_FIELDS:
        val = extracted.get(field_key)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing.append(f"{field_label} (recommended)")

    # --- Assemble final output ------------------------------------------------
    # Extract only the workflow-compatible fields (drop LLM metadata fields).
    workflow_fields = {
        "company_name": extracted.get("company_name") or "",
        "domain": extracted.get("domain") or "",
        "contact_name": extracted.get("contact_name"),
        "contact_title": extracted.get("contact_title"),
        "stakeholder_role": extracted.get("stakeholder_role") or "",
        "meeting_goal": extracted.get("meeting_goal") or "",
        "meeting_type": extracted.get("meeting_type") or "",
        "meeting_stage": extracted.get("meeting_stage"),
        "relationship_status": extracted.get("relationship_status"),
        "previous_meeting_outcome": extracted.get("previous_meeting_outcome"),
        "meeting_notes_summary": extracted.get("meeting_notes_summary"),
        # User-side context
        "user_role": extracted.get("user_role"),
        "user_company": extracted.get("user_company"),
        "user_function": extracted.get("user_function"),
        "engagement_type": extracted.get("engagement_type"),
        "desired_outcome": extracted.get("desired_outcome"),
        "success_definition": extracted.get("success_definition"),
    }

    return {
        **workflow_fields,
        "raw_meeting_context": raw_meeting_context,
        "intake_missing_fields": missing,
        "intake_low_confidence_fields": low_confidence,
        "intake_confirmation_summary": extracted.get("confirmation_summary", ""),
    }


def _empty_result(
    raw_meeting_context: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Return a valid but empty intake result."""
    summary = error or "No meeting context was provided."
    return {
        "company_name": "",
        "domain": "",
        "contact_name": None,
        "contact_title": None,
        "stakeholder_role": "",
        "meeting_goal": "",
        "meeting_type": "",
        "meeting_stage": None,
        "relationship_status": None,
        "previous_meeting_outcome": None,
        "meeting_notes_summary": None,
        "user_role": None,
        "user_company": None,
        "user_function": None,
        "engagement_type": None,
        "desired_outcome": None,
        "success_definition": None,
        "raw_meeting_context": raw_meeting_context,
        "intake_missing_fields": [
            "Company name",
            "Company domain",
            "Meeting type",
            "Stakeholder role",
            "Meeting goal",
            "Meeting stage (recommended)",
            "Relationship status (recommended)",
        ],
        "intake_low_confidence_fields": [],
        "intake_confirmation_summary": summary,
    }
