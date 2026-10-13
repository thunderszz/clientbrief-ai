"""Strategic signals: recent funding, deals, launches, hires and restructurings.

The model separates what happened (``description``, factual) from what it may mean
(``implication``, inferred). Dates are then normalised to ``YYYY-MM`` and duplicates dropped.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agents.evidence import evidence_block, gather_evidence
from core.utils import build_signal_queries, deduplicate_signals, normalize_date
from services.llm_client import structured_extract


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
    source: str = Field(description="URL of the primary source for this signal.")
    date: str | None = Field(
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


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Search for recent news about the company and return ``external_signals``."""
    company_name = state.get("company_name", "")
    if not company_name:
        return {
            "external_signals": [],
            "errors": ["external_signals: company_name is empty, skipping."],
        }

    plan_queries = state.get("research_plan", {}).get("tavily_queries", [])
    snippets, _ = gather_evidence(build_signal_queries(company_name, extra_queries=plan_queries))
    if not snippets:
        return {
            "external_signals": [],
            "errors": ["external_signals: no evidence retrieved from Tavily."],
        }

    result = structured_extract(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=(
            f"Company: {company_name}\n\n"
            f"{evidence_block(snippets, 40)}\n\n"
            "Extract strategic signals from the evidence above."
        ),
        response_model=SignalList,
    )
    if result is None:
        return {
            "external_signals": [],
            "errors": ["external_signals: LLM extraction failed, returning empty signals."],
        }

    signals = [
        {**signal.model_dump(), "date": normalize_date(signal.date)} for signal in result.signals
    ]
    return {"external_signals": deduplicate_signals(signals)}
