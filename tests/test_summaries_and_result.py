"""Human-readable node summaries and the public result payload."""

from __future__ import annotations

import json

import pytest

from core.pipeline import PIPELINE_NODES
from core.result import build_result
from core.summaries import summarize_update


@pytest.mark.parametrize(
    ("node", "update", "expected"),
    [
        (
            "orchestrate",
            {"normalized_function": "finance", "normalized_seniority": "c_level", "errors": []},
            "function: finance · seniority: c_level",
        ),
        (
            "orchestrate",
            {"normalized_function": "general", "normalized_seniority": None, "errors": ["bad domain"]},
            "function: general · seniority: unknown · 1 warning",
        ),
        (
            "infer_engagement",
            {"engagement_inference": {"confidence": "not_needed"}},
            "not needed: all engagement fields were given",
        ),
        (
            "infer_engagement",
            {"engagement_inference": {"confidence": "high", "inferred_meeting_stage": "discovery_followup"}},
            "inferred stage: discovery_followup (high confidence)",
        ),
        (
            "engagement_context",
            {"engagement_context": {"interaction_stage": "first_intro", "research_mode": "full"}},
            "stage: first_intro · research mode: full",
        ),
        (
            "plan",
            {"research_plan": {"tavily_queries": ["a", "b"], "research_mode": "light"}},
            "2 planned queries · light mode",
        ),
        (
            "research_company",
            {"company_profile": {"industry": "Fintech", "sources": ["u1", "u2", "u3"]}},
            "3 sources · Fintech",
        ),
        ("research_company", {"company_profile": {"sources": ["u1"]}}, "1 source"),
        (
            "research_signals",
            {
                "external_signals": [
                    {"signal_type": "financial"},
                    {"signal_type": "expansion"},
                    {"signal_type": "financial"},
                ]
            },
            "3 signals · financial, expansion",
        ),
        ("research_signals", {}, "0 signals"),
        (
            "research_person",
            {"person_profile": {"name": None, "match_confidence": "none"}},
            "skipped: no contact named",
        ),
        (
            "research_person",
            {"person_profile": {"name": "Jane Doe", "match_confidence": "medium"}},
            "Jane Doe · match confidence: medium",
        ),
        (
            "audience_context",
            {"audience_context": {"department": "Finance", "likely_priorities": ["a", "b"]}},
            "Finance · 2 priorities",
        ),
        (
            "hypothesize",
            {"pain_point_hypotheses": {"facts": [1, 2], "inferences": [1], "hypotheses": [1, 2, 3]}},
            "2 facts · 1 inference · 3 hypotheses",
        ),
        (
            "strategize",
            {"meeting_strategy": {"recommended_questions": [1] * 5, "possible_objections": [1] * 3}},
            "5 questions · 3 objections",
        ),
        (
            "compose",
            {"reliability_report": {"score": 78}, "sources": ["a", "b"]},
            "reliability 78/100 · 2 sources",
        ),
        ("unknown_node", {"x": 1}, ""),
    ],
)
def test_summarize_update(node, update, expected):
    assert summarize_update(node, update) == expected


@pytest.mark.parametrize("node", [n["id"] for n in PIPELINE_NODES])
def test_every_node_handles_empty_update(node):
    assert isinstance(summarize_update(node, {}), str)


def test_build_result_shape_and_json():
    state = {
        "company_name": "Satispay",
        "domain": "satispay.com",
        "meeting_type": "sales_discovery",
        "external_signals": [{"description": "x"}],
        "final_brief": "# Brief",
        "reliability_report": {"score": 50, "rating": 3, "components": []},
        "llm_calls": [{"node": "plan"}],
        "errors": ["warn"],
        "normalized_function": "finance",
    }

    result = build_result(state)

    assert result["input"]["company_name"] == "Satispay"
    assert result["input"]["contact_name"] is None
    assert result["external_signals"] == [{"description": "x"}]
    assert result["reliability"]["score"] == 50
    assert result["final_brief"] == "# Brief"
    assert result["errors"] == ["warn"]
    assert result["llm_calls"] == [{"node": "plan"}]
    assert "generated_at" in result
    assert "normalized_function" not in result
    json.dumps(result)


def test_build_result_defaults_for_empty_state():
    result = build_result({})

    assert result["external_signals"] == []
    assert result["company_profile"] == {}
    assert result["reliability"]["score"] == 0
    json.dumps(result)
