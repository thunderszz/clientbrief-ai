"""Contact lookup: does public evidence reliably identify the person we are meeting?

With no contact name there is nothing to search. Otherwise the model judges how well the
search results match the person, and anything below a medium-confidence match is stripped
back to the name, so later agents never build on the wrong person.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from agents.evidence import evidence_block, gather_evidence
from core.utils import build_person_queries
from services.llm_client import structured_extract

RELIABLE_MATCH = frozenset({"medium", "high"})

# Fields that describe the person; cleared when the match is not reliable.
_DETAIL_FIELDS = ("title", "department", "tenure_hint", "background_summary")


def has_reliable_match(person_profile: Mapping[str, Any]) -> bool:
    """Person-level detail may be used only for a medium or high confidence match."""
    return person_profile.get("match_confidence", "none") in RELIABLE_MATCH


class MatchConfidence(StrEnum):
    """How confidently we matched the contact to a real person."""

    high = "high"
    medium = "medium"
    low = "low"


class PersonProfile(BaseModel):
    """Structured person profile extracted from web evidence.

    The LLM must evaluate whether the search results actually describe
    the target person at the target company.  If evidence is weak or
    ambiguous, set ``matched_person=False`` and ``match_confidence="low"``.
    """

    matched_person: bool = Field(
        description=(
            "True if the evidence convincingly identifies this person at "
            "this company.  False if the match is uncertain or the evidence "
            "describes a different person."
        )
    )
    match_confidence: MatchConfidence = Field(
        description=(
            "'high' = strong name+company+title alignment across multiple sources; "
            "'medium' = plausible match from at least one credible source; "
            "'low' = ambiguous, conflicting, or no useful evidence."
        )
    )
    name: str | None = Field(
        None, description="Full name as it appears in the evidence."
    )
    title: str | None = Field(
        None, description="Job title as found in the evidence."
    )
    department: str | None = Field(
        None, description="Department or function (e.g. Engineering, Finance)."
    )
    tenure_hint: str | None = Field(
        None,
        description="Any hint about how long the person has been in the role or at the company.",
    )
    background_summary: str | None = Field(
        None,
        description=(
            "1-3 sentence summary of the person's professional background "
            "based ONLY on the evidence.  Return null if insufficient data."
        ),
    )
    public_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Notable public activities (talks, articles, awards, board seats) "
            "found in the evidence.  Empty list if none."
        ),
    )


_SYSTEM_PROMPT = """\
You are a professional-identity research analyst.  You will receive text
snippets retrieved from the web, collected while searching for a specific
person at a specific company.

Your task: determine whether the evidence identifies the target person and,
if so, extract a structured profile.

Rules:
- Evaluate match_confidence carefully:
    high  = name, company, AND title all align across reliable sources.
    medium = name + company match on at least one credible source,
             but title or details are incomplete.
    low   = evidence is ambiguous, describes a different person,
            or returns no useful results.
- If confidence is low: set matched_person=False, and return null for all
  detail fields (title, department, tenure_hint, background_summary).
  Only populate name if you are reasonably sure of the basic identity.
- Do NOT fabricate information that is not in the evidence.
- Preferred source types (in order of trust): company team pages,
  speaker bios, press releases, published interviews, public professional
  profiles.
- Ignore social-media posts, forums, or unverifiable sources.
"""


def empty_profile() -> dict[str, Any]:
    """The profile when no contact was named or nothing usable was found."""
    return {
        "matched_person": False,
        "match_confidence": "none",
        "name": None,
        **dict.fromkeys(_DETAIL_FIELDS),
        "public_signals": [],
        "sources": [],
    }


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Look the contact up and return ``person_profile``."""
    contact_name = state.get("contact_name")
    if not contact_name:
        return {"person_profile": empty_profile()}

    company_name = state.get("company_name", "")
    contact_title = state.get("contact_title")
    snippets, sources = gather_evidence(build_person_queries(contact_name, company_name, contact_title))
    name_only = {**empty_profile(), "name": contact_name, "sources": sources}
    if not snippets:
        error = f"person_research: no evidence found for '{contact_name}' at '{company_name}'."
        return {"person_profile": name_only, "errors": [error]}

    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=(
            f"Target person: {contact_name}\n"
            f"Company: {company_name}\n"
            f"Expected title (if known): {contact_title or 'not provided'}\n\n"
            f"{evidence_block(snippets, 20)}\n\n"
            "Evaluate the identity match and extract a person profile."
        ),
        response_model=PersonProfile,
    )
    if result is None:
        error = "person_research: LLM extraction failed, returning sparse profile."
        return {"person_profile": name_only, "errors": [error]}

    profile = result.model_dump(mode="json")
    if not has_reliable_match(profile):
        profile.update(dict.fromkeys(_DETAIL_FIELDS), matched_person=False, public_signals=[])
    return {"person_profile": {**profile, "sources": sources}}
