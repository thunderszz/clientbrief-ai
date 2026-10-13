"""The public, JSON-serialisable result of a pipeline run.

The UI renders the brief from these structured slices; ``final_brief`` (markdown)
is kept for copy and download.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from core.reliability import compute_reliability_report
from core.state import INPUT_FIELDS

_DICT_KEYS = (
    "engagement_inference",
    "engagement_context",
    "research_plan",
    "company_profile",
    "person_profile",
    "audience_context",
    "pain_point_hypotheses",
    "meeting_strategy",
)

_LIST_KEYS = ("external_signals", "sources", "errors", "llm_calls")


def build_result(state: Mapping[str, Any]) -> dict[str, Any]:
    """Pick the public slices of the final state and make them JSON-safe."""
    result: dict[str, Any] = {"input": {key: state.get(key) for key in INPUT_FIELDS}}
    for key in _DICT_KEYS:
        result[key] = state.get(key) or {}
    for key in _LIST_KEYS:
        result[key] = list(state.get(key) or [])
    result["reliability"] = state.get("reliability_report") or compute_reliability_report(state)
    result["final_brief"] = state.get("final_brief") or ""
    result["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    # Round-trip through JSON so enums, dates etc. become plain values.
    return json.loads(json.dumps(result, default=str))
