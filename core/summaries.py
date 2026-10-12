"""One-line, human-readable summaries of what each pipeline node produced.

Used for the live event log in the UI and the CLI progress view.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


def _count(n: int, word: str, plural: str | None = None) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {plural or word + 's'}"


def _orchestrate(u: Mapping[str, Any]) -> str:
    text = (
        f"function: {u.get('normalized_function') or 'general'}"
        f" · seniority: {u.get('normalized_seniority') or 'unknown'}"
    )
    errors = u.get("errors") or []
    return text + (f" · {_count(len(errors), 'warning')}" if errors else "")


def _infer_engagement(u: Mapping[str, Any]) -> str:
    ei = u.get("engagement_inference") or {}
    confidence = ei.get("confidence")
    if confidence == "not_needed":
        return "not needed: all engagement fields were given"
    stage = ei.get("inferred_meeting_stage") or "unknown"
    return f"inferred stage: {stage} ({confidence or 'low'} confidence)"


def _engagement_context(u: Mapping[str, Any]) -> str:
    ec = u.get("engagement_context") or {}
    return (
        f"stage: {ec.get('interaction_stage') or 'unknown'}"
        f" · research mode: {ec.get('research_mode') or 'full'}"
    )


def _plan(u: Mapping[str, Any]) -> str:
    rp = u.get("research_plan") or {}
    queries = rp.get("tavily_queries") or []
    return f"{_count(len(queries), 'planned query', 'planned queries')} · {rp.get('research_mode') or 'full'} mode"


def _research_company(u: Mapping[str, Any]) -> str:
    cp = u.get("company_profile") or {}
    text = _count(len(cp.get("sources") or []), "source")
    return text + (f" · {cp['industry']}" if cp.get("industry") else "")


def _research_signals(u: Mapping[str, Any]) -> str:
    signals = u.get("external_signals") or []
    types: list[str] = []
    for signal in signals:
        kind = signal.get("signal_type")
        if kind and kind not in types:
            types.append(kind)
    text = _count(len(signals), "signal")
    return text + (f" · {', '.join(types[:3])}" if types else "")


def _research_person(u: Mapping[str, Any]) -> str:
    pp = u.get("person_profile") or {}
    if not pp.get("name"):
        return "skipped: no contact named"
    return f"{pp['name']} · match confidence: {pp.get('match_confidence') or 'none'}"


def _audience_context(u: Mapping[str, Any]) -> str:
    ac = u.get("audience_context") or {}
    priorities = _count(len(ac.get("likely_priorities") or []), "priority", "priorities")
    return f"{ac['department']} · {priorities}" if ac.get("department") else priorities


def _hypothesize(u: Mapping[str, Any]) -> str:
    pph = u.get("pain_point_hypotheses") or {}
    return " · ".join(
        [
            _count(len(pph.get("facts") or []), "fact"),
            _count(len(pph.get("inferences") or []), "inference"),
            _count(len(pph.get("hypotheses") or []), "hypothesis", "hypotheses"),
        ]
    )


def _strategize(u: Mapping[str, Any]) -> str:
    ms = u.get("meeting_strategy") or {}
    return (
        f"{_count(len(ms.get('recommended_questions') or []), 'question')}"
        f" · {_count(len(ms.get('possible_objections') or []), 'objection')}"
    )


def _compose(u: Mapping[str, Any]) -> str:
    report = u.get("reliability_report") or {}
    return f"reliability {report.get('score', 0)}/100 · {_count(len(u.get('sources') or []), 'source')}"


_SUMMARISERS: dict[str, Callable[[Mapping[str, Any]], str]] = {
    "orchestrate": _orchestrate,
    "infer_engagement": _infer_engagement,
    "engagement_context": _engagement_context,
    "plan": _plan,
    "research_company": _research_company,
    "research_signals": _research_signals,
    "research_person": _research_person,
    "audience_context": _audience_context,
    "hypothesize": _hypothesize,
    "strategize": _strategize,
    "compose": _compose,
}


def summarize_update(node_id: str, update: Mapping[str, Any] | None) -> str:
    """Describe a node's state update in one short line ("" for unknown nodes)."""
    summariser = _SUMMARISERS.get(node_id)
    return summariser(update or {}) if summariser else ""
