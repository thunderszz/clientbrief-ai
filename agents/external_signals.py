"""External Signals Agent — LLM-assisted strategic-signal extraction.

Reads ``company_name`` and ``research_plan`` from state.
Runs the always-on signal queries plus supplementary research-plan queries
via Tavily.  Feeds retrieved text to an LLM that extracts structured
strategic signals with factual descriptions and inferential implications.

After LLM extraction, signals are deduplicated deterministically and dates
are normalised to YYYY-MM.

Ownership:  ``external_signals`` only.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.utils import build_signal_queries, deduplicate_signals, normalize_date
from services.llm_client import structured_extract
from services.tavily_client import SearchResponse, TavilySearchClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas for the LLM's structured output
# ---------------------------------------------------------------------------

class Signal(BaseModel):
    """A single strategic signal extracted from web evidence."""

    signal_type: str = Field(
        description=(
            "Category of signal, e.g. 'acquisition', 'partnership', "
            "'leadership_change', 'expansion', 'restructuring', "
            "'product_launch', 'financial', 'regulatory'."
        )
    )
    description: str = Field(
        description=(
            "Factual description of what happened, grounded in the evidence. "
            "Do NOT include speculation — only verifiable facts."
        )
    )
    implication: str = Field(
        description=(
            "A brief inference about what this signal might mean for the "
            "company's strategy or operations.  This IS allowed to be "
            "interpretive, but must be clearly derived from the fact."
        )
    )
    confidence: str = Field(
        description=(
            "How confident are you that this signal is accurate? "
            "'high' = multiple corroborating sources; "
            "'medium' = single credible source; "
            "'low' = mentioned but unverified."
        )
    )
    source: str = Field(
        description="URL of the primary source for this signal."
    )
    date: Optional[str] = Field(
        None,
        description=(
            "When the event occurred or was announced, if mentioned. "
            "Use the format shown in the source text (e.g. 'January 2025', "
            "'2025-03-01').  Return null if no date is available."
        ),
    )


class SignalList(BaseModel):
    """Wrapper for LLM structured output — a list of signals."""

    signals: list[Signal] = Field(
        default_factory=list,
        description=(
            "Strategic signals found in the evidence.  "
            "Return an empty list if no meaningful signals are found."
        ),
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a strategic-intelligence analyst.  You will receive text snippets
retrieved from the web about a specific company.

Your task: extract recent strategic signals — events, announcements, or
developments that reveal the company's direction.

Rules:
- description: factual only — what happened, who was involved, when.
- implication: inferential — what it might mean for the business.
- Do NOT invent signals that are not in the evidence.
- If two snippets describe the same event, report it only once (pick the
  most detailed version).
- Prefer signals from the last 18 months.
- Return an empty list if there are no meaningful signals.
- For 'source', use the URL from the snippet header if available.
- For 'date', use whatever date text appears in the source; the system
  will normalise it afterward.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_evidence(
    company_name: str,
    research_plan: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Run Tavily queries and return (evidence_snippets, source_urls)."""
    try:
        client = TavilySearchClient()
    except EnvironmentError as exc:
        logger.warning("Tavily client init failed: %s", exc)
        return [], []

    # Always-on signal queries from the spec
    plan_queries: list[str] = research_plan.get("tavily_queries", [])
    queries = build_signal_queries(company_name, extra_queries=plan_queries)

    responses: list[SearchResponse] = client.search_batch(queries)

    snippets: list[str] = []
    urls: list[str] = []
    seen_urls: set[str] = set()

    for resp in responses:
        if not resp.ok:
            continue
        for hit in resp.results:
            # Include source URL in snippet so the LLM can reference it
            snippets.append(f"[source: {hit.url}] [{hit.title}] {hit.content}")
            if hit.url and hit.url not in seen_urls:
                seen_urls.add(hit.url)
                urls.append(hit.url)

    return snippets, urls


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def run(state: dict[str, Any]) -> dict[str, Any]:
    """Run external-signals research and return an ``external_signals`` update."""

    company_name: str = state.get("company_name", "")
    research_plan: dict[str, Any] = state.get("research_plan", {})
    new_errors: list[str] = []

    if not company_name:
        new_errors.append("external_signals: company_name is empty, skipping.")
        return {"external_signals": [], "errors": new_errors}

    # --- 1. Gather evidence via Tavily ----------------------------------------
    snippets, source_urls = _collect_evidence(company_name, research_plan)

    if not snippets:
        new_errors.append("external_signals: no evidence retrieved from Tavily.")
        return {"external_signals": [], "errors": new_errors}

    # --- 2. Ask the LLM to extract structured signals -------------------------
    evidence_block = "\n\n".join(snippets[:40])   # generous cap for news
    user_prompt = (
        f"Company: {company_name}\n\n"
        f"--- Evidence ---\n{evidence_block}\n--- End evidence ---\n\n"
        "Extract strategic signals from the evidence above."
    )

    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        response_model=SignalList,
    )

    if result is None:
        new_errors.append("external_signals: LLM extraction failed, returning empty signals.")
        return {"external_signals": [], "errors": new_errors}

    # --- 3. Post-process: normalise dates + deduplicate -----------------------
    raw_signals: list[dict[str, Any]] = []
    for sig in result.signals:
        d = sig.model_dump()
        d["date"] = normalize_date(d.get("date"))
        raw_signals.append(d)

    deduped = deduplicate_signals(raw_signals)

    return {"external_signals": deduped}
