"""Company research: search the web, then extract a structured profile of the company."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agents.evidence import evidence_block, gather_evidence
from core.utils import build_company_queries
from services.llm_client import structured_extract


class CompanyProfile(BaseModel):
    """Structured company profile extracted from retrieved web text.

    Every field must be grounded in the provided evidence.
    Use null / empty lists when evidence is insufficient.
    """

    industry: str | None = Field(None, description="Primary industry the company operates in.")
    business_model: str | None = Field(
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
    company_positioning: str | None = Field(
        None, description="How the company positions itself competitively."
    )
    complexity_signals: list[str] = Field(
        default_factory=list,
        description="Signals of organisational or operational complexity (multi-entity, regulated, global, etc.).",
    )


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


def empty_profile() -> dict[str, Any]:
    return {**CompanyProfile().model_dump(), "sources": []}


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Research the company and return ``company_profile``, with the URLs it is based on."""
    company_name = state.get("company_name", "")
    domain = state.get("domain", "")
    if not company_name:
        return {
            "company_profile": empty_profile(),
            "errors": ["company_research: company_name is empty, skipping."],
        }

    plan_queries = state.get("research_plan", {}).get("tavily_queries", [])
    snippets, sources = gather_evidence(
        build_company_queries(company_name, domain) + plan_queries, cite_urls=False
    )
    if not snippets:
        return {
            "company_profile": {**empty_profile(), "sources": sources},
            "errors": ["company_research: no evidence retrieved from Tavily."],
        }

    profile = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=(
            f"Company: {company_name}\n"
            f"Domain: {domain}\n\n"
            f"{evidence_block(snippets, 30)}\n\n"
            "Extract the company profile from the evidence above."
        ),
        response_model=CompanyProfile,
    )
    if profile is None:
        return {
            "company_profile": {**empty_profile(), "sources": sources},
            "errors": ["company_research: LLM extraction failed, returning sparse profile."],
        }
    return {"company_profile": {**profile.model_dump(), "sources": sources}}
