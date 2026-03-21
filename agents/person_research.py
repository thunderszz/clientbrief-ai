"""Person Research Agent — LLM-assisted identity matching and profile extraction.

Reads ``contact_name``, ``contact_title``, ``company_name``, and
``stakeholder_role`` from state.

If ``contact_name`` is not provided, no search is attempted and a sparse
low-confidence profile is returned immediately.

When a name is provided the agent:
1. Runs person-specific Tavily queries.
2. Feeds retrieved text to an LLM that evaluates identity-match confidence
   and extracts a structured person profile.
3. If confidence is ``"low"``, the returned profile is kept intentionally
   sparse — downstream agents must not over-interpret it.

Ownership:  ``person_profile`` only.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.utils import build_person_queries
from services.llm_client import structured_extract
from services.tavily_client import SearchResponse, TavilySearchClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema for the LLM's structured output
# ---------------------------------------------------------------------------

class MatchConfidence(str, Enum):
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
    name: Optional[str] = Field(
        None, description="Full name as it appears in the evidence."
    )
    title: Optional[str] = Field(
        None, description="Job title as found in the evidence."
    )
    department: Optional[str] = Field(
        None, description="Department or function (e.g. Engineering, Finance)."
    )
    tenure_hint: Optional[str] = Field(
        None,
        description="Any hint about how long the person has been in the role or at the company.",
    )
    background_summary: Optional[str] = Field(
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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_profile() -> dict[str, Any]:
    """Return a valid sparse person_profile dict (no match)."""
    return {
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


def _collect_evidence(
    contact_name: str,
    company_name: str,
    contact_title: str | None,
) -> tuple[list[str], list[str]]:
    """Run Tavily queries and return (evidence_snippets, source_urls)."""
    try:
        client = TavilySearchClient()
    except EnvironmentError as exc:
        logger.warning("Tavily client init failed: %s", exc)
        return [], []

    queries = build_person_queries(contact_name, company_name, contact_title)
    responses: list[SearchResponse] = client.search_batch(queries)

    snippets: list[str] = []
    urls: list[str] = []
    seen_urls: set[str] = set()

    for resp in responses:
        if not resp.ok:
            continue
        for hit in resp.results:
            snippets.append(f"[source: {hit.url}] [{hit.title}] {hit.content}")
            if hit.url and hit.url not in seen_urls:
                seen_urls.add(hit.url)
                urls.append(hit.url)

    return snippets, urls


def _enforce_low_confidence_sparsity(profile: dict[str, Any]) -> dict[str, Any]:
    """If confidence is not medium/high, strip detail fields.

    Per spec (CLAUDE.md rule 4): person-level information must only be used
    when match_confidence is "medium" or "high".  Both "low" and "none"
    (no search attempted) are treated as non-reliable.
    """
    # Only "medium" and "high" are considered reliable.
    if profile.get("match_confidence") not in ("medium", "high"):
        profile["matched_person"] = False
        profile["title"] = None
        profile["department"] = None
        profile["tenure_hint"] = None
        profile["background_summary"] = None
        profile["public_signals"] = []
    return profile


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run(state: dict[str, Any]) -> dict[str, Any]:
    """Run person research and return a ``person_profile`` update."""

    contact_name: str | None = state.get("contact_name")
    contact_title: str | None = state.get("contact_title")
    company_name: str = state.get("company_name", "")
    new_errors: list[str] = []

    # --- Fast exit: no contact_name → no search -------------------------------
    if not contact_name:
        return {"person_profile": _empty_profile()}

    # --- 1. Gather evidence via Tavily ----------------------------------------
    snippets, source_urls = _collect_evidence(
        contact_name, company_name, contact_title,
    )

    if not snippets:
        new_errors.append(
            f"person_research: no evidence found for '{contact_name}' at '{company_name}'."
        )
        profile = _empty_profile()
        profile["name"] = contact_name
        profile["sources"] = source_urls
        return {"person_profile": profile, "errors": new_errors}

    # --- 2. Ask the LLM to evaluate match + extract profile -------------------
    evidence_block = "\n\n".join(snippets[:20])   # person results tend to be shorter
    user_prompt = (
        f"Target person: {contact_name}\n"
        f"Company: {company_name}\n"
        f"Expected title (if known): {contact_title or 'not provided'}\n\n"
        f"--- Evidence ---\n{evidence_block}\n--- End evidence ---\n\n"
        "Evaluate the identity match and extract a person profile."
    )

    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=PersonProfile,
    )

    if result is None:
        new_errors.append("person_research: LLM extraction failed, returning sparse profile.")
        profile = _empty_profile()
        profile["name"] = contact_name
        profile["sources"] = source_urls
        return {"person_profile": profile, "errors": new_errors}

    # --- 3. Post-process ------------------------------------------------------
    profile = result.model_dump()

    # Normalise confidence enum to plain string
    if isinstance(profile.get("match_confidence"), MatchConfidence):
        profile["match_confidence"] = profile["match_confidence"].value

    # Enforce sparsity on low confidence
    profile = _enforce_low_confidence_sparsity(profile)

    # Attach source URLs (deterministic, not LLM-owned)
    profile["sources"] = source_urls

    return {"person_profile": profile}
