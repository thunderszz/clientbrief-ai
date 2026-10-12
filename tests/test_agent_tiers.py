"""Each agent asks the LLM client for the right model tier."""

from __future__ import annotations

import pytest

from agents import (
    audience_context_agent,
    company_research,
    engagement_inference_agent,
    external_signals,
    meeting_strategy_agent,
    pain_point_agent,
)
from services.llm_client import Tier

_STATE = {
    "company_name": "Acme",
    "domain": "acme.com",
    "meeting_type": "sales_discovery",
    "stakeholder_role": "CFO",
    "meeting_goal": "Discuss finance automation",
    "normalized_function": "finance",
    "research_plan": {"tavily_queries": []},
}


def _capture_tier(monkeypatch, module):
    seen: list[Tier] = []

    def fake_extract(**kwargs):
        seen.append(kwargs.get("tier", Tier.FAST))
        return None

    monkeypatch.setattr(module, "structured_extract", fake_extract)
    return seen


@pytest.mark.parametrize(
    "module", [audience_context_agent, pain_point_agent, meeting_strategy_agent]
)
def test_reasoning_agents_use_reasoning_tier(monkeypatch, module):
    seen = _capture_tier(monkeypatch, module)

    module.run(dict(_STATE))

    assert seen == [Tier.REASONING]


def test_engagement_inference_uses_fast_tier(monkeypatch):
    seen = _capture_tier(monkeypatch, engagement_inference_agent)

    engagement_inference_agent.run(dict(_STATE))

    assert seen in ([], [Tier.FAST])


@pytest.mark.parametrize("module", [company_research, external_signals])
def test_research_agents_use_fast_tier(monkeypatch, module):
    seen = _capture_tier(monkeypatch, module)
    monkeypatch.setattr(module, "_collect_evidence", lambda *a, **k: (["snippet"], ["https://a.test"]))

    module.run(dict(_STATE))

    assert seen == [Tier.FAST]
