"""Brief composer writes the reliability report and a header that shows it."""

from __future__ import annotations

from agents import brief_composer


def _state():
    return {
        "company_name": "Satispay",
        "domain": "satispay.com",
        "meeting_type": "sales_discovery",
        "stakeholder_role": "CFO",
        "company_profile": {"industry": "Fintech", "sources": ["https://www.satispay.com/a"]},
        "external_signals": [
            {
                "signal_type": "financial",
                "description": "Raised a Series D",
                "implication": "Has money to spend",
                "confidence": "high",
                "source": "https://news.test/a",
            }
        ],
        "errors": [],
    }


def test_composer_writes_reliability_report():
    out = brief_composer.run(_state())

    report = out["reliability_report"]
    assert 0 < report["score"] <= 100
    assert out["brief_reliability"] == report["rating"]
    assert (
        f"**Brief reliability:** {report['rating']}/5 ({report['score']}/100)" in out["final_brief"]
    )


def test_composer_sources_are_aggregated():
    out = brief_composer.run(_state())

    assert out["sources"] == ["https://www.satispay.com/a", "https://news.test/a"]


def test_composer_handles_empty_state():
    out = brief_composer.run({})

    assert out["brief_reliability"] == 0
    assert out["reliability_report"]["score"] == 0
    assert "# Pre-Meeting Brief" in out["final_brief"]
