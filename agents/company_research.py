"""Company Research Agent — LLM-assisted fact extraction.

Reads ``company_name``, ``domain``, and ``research_plan`` from state.
Runs the standard company-level Tavily queries plus any supplementary
queries from the research plan.  Feeds retrieved text to an LLM that
extracts a structured company profile.

Ownership:  ``company_profile`` only.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.utils import build_company_queries
from services.llm_client import structured_extract
from services.tavily_client import SearchResponse, TavilySearchClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema for the LLM's structured output
# ---------------------------------------------------------------------------

class CompanyProfile(BaseModel):
    """Structured company profile extracted from retrieved web text.

    Every field must be grounded in the provided evidence.
    Use null / empty lists when evidence is insufficient.
    """

    industry: Optional[str] = Field(
        None, description="Primary industry the company operates in."
    )
    business_model: Optional[str] = Field(
        None, description="How the company makes money (e.g. SaaS, marketplace, manufacturing)."
    )
    products: list[str] = Field(
        default_factory=list,
        description="Key products or service lines mentioned in the evidence.",
    )
    customer_segments: list[str] = Field(
        default_factory=list,
        description="Types of customers the company serves (e.g. enterprise, SMB, government).",
    )
    geographies: list[str] = Field(
        default_factory=list,
        description="Regions or countries where the company operates.",
    )
    company_positioning: Optional[str] = Field(
        None, description="How the company positions itself competitively."
    )
    complexity_signals: list[str] = Field(
        default_factory=list,
        description="Signals of organisational or operational complexity (multi-entity, regulated, global, etc.).",
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a company-research analyst.  You will receive text snippets retrieved
from the web about a specific company.

Your task: extract a structured company profile using ONLY facts that are
directly supported by the provided text.

Rules:
- Do NOT invent or guess information that is not in the evidence.
- If a field cannot be determined from the evidence, return null (for strings)
  or an empty list (for arrays).
- Keep each value concise (one or two sentences max for string fields).
- For list fields, include only distinct, non-overlapping items.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_evidence(
    company_name: str,
    domain: str,
    research_plan: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Run Tavily queries and return (evidence_snippets, source_urls)."""
    try:
        client = TavilySearchClient()
    except EnvironmentError as exc:
        logger.warning("Tavily client init failed: %s", exc)
        return [], []

    # Standard company queries from spec
    queries = build_company_queries(company_name, domain)

    # Supplementary queries from the research planner
    plan_queries: list[str] = research_plan.get("tavily_queries", [])
    queries.extend(plan_queries)

    responses: list[SearchResponse] = client.search_batch(queries)

    snippets: list[str] = []
    urls: list[str] = []
    seen_urls: set[str] = set()

    for resp in responses:
        if not resp.ok:
            continue
        for hit in resp.results:
            snippets.append(f"[{hit.title}] {hit.content}")
            if hit.url and hit.url not in seen_urls:
                seen_urls.add(hit.url)
                urls.append(hit.url)

    return snippets, urls


def _empty_profile() -> dict[str, Any]:
    """Return a valid but empty company_profile dict."""
    return CompanyProfile().model_dump()


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run(state: dict[str, Any]) -> dict[str, Any]:
    """Run company research and return a ``company_profile`` update."""

    company_name: str = state.get("company_name", "")
    domain: str = state.get("domain", "")
    research_plan: dict[str, Any] = state.get("research_plan", {})
    new_errors: list[str] = []

    if not company_name:
        new_errors.append("company_research: company_name is empty, skipping.")
        return {
            "company_profile": _empty_profile() | {"sources": []},
            "errors": new_errors,
        }

    # --- 1. Gather evidence via Tavily ----------------------------------------
    snippets, source_urls = _collect_evidence(company_name, domain, research_plan)

    if not snippets:
        new_errors.append("company_research: no evidence retrieved from Tavily.")
        return {
            "company_profile": _empty_profile() | {"sources": source_urls},
            "errors": new_errors,
        }

    # --- 2. Ask the LLM to extract a structured profile -----------------------
    evidence_block = "\n\n".join(snippets[:30])   # cap context window usage
    user_prompt = (
        f"Company: {company_name}\n"
        f"Domain: {domain}\n\n"
        f"--- Evidence ---\n{evidence_block}\n--- End evidence ---\n\n"
        "Extract the company profile from the evidence above."
    )

    profile = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=CompanyProfile,
    )

    if profile is None:
        new_errors.append("company_research: LLM extraction failed, returning sparse profile.")
        return {
            "company_profile": _empty_profile() | {"sources": source_urls},
            "errors": new_errors,
        }

    # --- 3. Assemble output (add sources — not owned by the LLM) --------------
    result = profile.model_dump()
    result["sources"] = source_urls

    return {"company_profile": result}
