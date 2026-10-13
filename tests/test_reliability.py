"""Tests for the transparent reliability score."""

from __future__ import annotations

from datetime import date

import pytest

from core.reliability import aggregate_sources, compute_reliability_report, rating_for, site_of

TODAY = date(2026, 10, 12)

_SITES = [
    "https://www.satispay.com/en-it/",
    "https://www.ilsole24ore.com/a",
    "https://www.fintechfutures.com/b",
    "https://ffnews.com/c",
    "https://www.cbinsights.com/d",
    "https://www.linkedin.com/e",
    "https://startupluxembourg.com/f",
    "https://qonto.com/g",
    "https://www.reuters.com/h",
    "https://techcrunch.com/i",
    "https://sifted.eu/j",
    "https://www.ft.com/k",
]


def _signal(i, confidence="high", when="2026-09"):
    return {"source": f"https://news{i}.test/s", "confidence": confidence, "date": when}


def _full_state(**overrides):
    company_sources = _SITES + [f"https://www.satispay.com/page{i}" for i in range(8)]
    state = {
        "domain": "satispay.com",
        "contact_name": "Alberto Dalmasso",
        "company_profile": {"sources": company_sources},
        "external_signals": [_signal(i) for i in range(6)],
        "person_profile": {"match_confidence": "high", "sources": []},
        "errors": [],
        "llm_calls": [{"ok": True, "fallback": False}] * 5,
    }
    state.update(overrides)
    return state


def _report(state):
    return compute_reliability_report(state, today=TODAY)


def _component(report, key):
    return next(c for c in report["components"] if c["key"] == key)


def test_full_house_scores_100():
    report = _report(_full_state())

    assert report["score"] == 100
    assert report["rating"] == 5
    assert [c["key"] for c in report["components"]] == [
        "sources",
        "diversity",
        "signals",
        "freshness",
        "contact",
        "health",
    ]
    assert sum(c["max"] for c in report["components"]) == 100


def test_empty_state_scores_zero():
    report = _report({})

    assert report["score"] == 0
    assert report["rating"] == 0


def test_no_contact_not_penalised():
    report = _report(_full_state(contact_name=None, person_profile={}))

    contact = _component(report, "contact")
    assert contact["applicable"] is False
    assert contact["detail"] == "no contact named"
    assert report["score"] == 100


def test_low_contact_match_scores_partially():
    state = _full_state(person_profile={"match_confidence": "low", "sources": []})

    contact = _component(_report(state), "contact")

    assert contact["points"] == 3
    assert contact["applicable"] is True


def test_named_contact_without_profile_scores_zero():
    state = _full_state(person_profile={"match_confidence": "none", "sources": []})

    assert _component(_report(state), "contact")["points"] == 0


def test_sources_saturate_at_20():
    sources = [f"https://site{i}.test/x" for i in range(10)]

    component = _component(_report({"company_profile": {"sources": sources}}), "sources")

    assert component["points"] == pytest.approx(15 * 10 / 20)
    assert component["detail"] == "10 distinct sources"


def test_signal_confidence_weights():
    signals = [_signal(0, "high"), _signal(1, "medium"), _signal(2, "low")]

    component = _component(_report(_full_state(external_signals=signals)), "signals")

    assert component["points"] == pytest.approx(20 * 2.1 / 6)
    assert component["detail"] == "3 signals: 1 high, 1 medium, 1 low"


def test_freshness_rewards_recent_dated_signals():
    signals = [
        _signal(0, when="2026-09"),  # 1 month old
        _signal(1, when="2025-10"),  # exactly 12 months: still recent
        _signal(2, when="2025-09-30"),  # 13 months: old
        _signal(3, when=None),  # undated
    ]

    component = _component(_report(_full_state(external_signals=signals)), "freshness")

    assert component["points"] == pytest.approx(15 * (2 + 0.25) / 4, abs=0.01)
    assert component["detail"] == "2 of 4 signals from the last 12 months, 1 undated"


def test_freshness_without_signals_is_zero():
    component = _component(_report(_full_state(external_signals=[])), "freshness")

    assert component["points"] == 0
    assert component["detail"] == "no signals to date"


def test_errors_reduce_health():
    state = _full_state(errors=["company_research: x", "person_research: y"])

    health = _component(_report(state), "health")

    assert health["points"] == 5
    assert health["detail"] == "2 agent warnings"


def test_fallbacks_reduce_health():
    calls = [
        {"ok": True, "fallback": True},
        {"ok": True, "fallback": True},
        {"ok": False, "fallback": False},  # failures already surface as agent warnings
        {"ok": True, "fallback": False},
    ]

    health = _component(_report(_full_state(llm_calls=calls)), "health")

    assert health["points"] == 15 - 2 * 2
    assert health["detail"] == "2 model fallbacks"


def test_warnings_and_fallbacks_detail():
    state = _full_state(errors=["x"], llm_calls=[{"ok": True, "fallback": True}])

    assert _component(_report(state), "health")["detail"] == "1 agent warning, 1 model fallback"


def test_health_floors_at_zero():
    state = _full_state(errors=["e"] * 9)

    assert _component(_report(state), "health")["points"] == 0


def test_company_domain_bonus():
    with_own = _component(_report(_full_state()), "diversity")
    others = [s for s in _SITES if "satispay" not in s]
    without_own = _component(_report(_full_state(company_profile={"sources": others})), "diversity")

    assert with_own["detail"].endswith("including satispay.com")
    assert with_own["points"] - without_own["points"] == pytest.approx(5)


def test_subdomains_count_as_one_site():
    sources = [
        "https://www.satispay.com/a",
        "https://support.satispay.com/b",
        "https://developers.satispay.com/c",
    ]
    state = {"domain": "www.satispay.com", "company_profile": {"sources": sources}}

    diversity = _component(_report(state), "diversity")

    assert diversity["detail"] == "1 independent site, including satispay.com"


def test_country_code_second_level_domains():
    assert site_of("https://news.bbc.co.uk/b") == "bbc.co.uk"
    assert site_of("https://ft.co.uk/c") == "ft.co.uk"
    assert site_of("www.Satispay.com") == "satispay.com"


def test_default_today_is_used_when_not_given():
    report = compute_reliability_report(_full_state())

    assert 0 < report["score"] <= 100


def test_aggregate_sources_dedupes_in_order():
    state = {
        "company_profile": {"sources": ["https://a.test", "https://b.test"]},
        "external_signals": [{"source": "https://b.test"}, {"source": "https://c.test"}],
        "person_profile": {"sources": ["https://d.test", " https://a.test "]},
    }

    assert aggregate_sources(state) == [
        "https://a.test",
        "https://b.test",
        "https://c.test",
        "https://d.test",
    ]


@pytest.mark.parametrize(
    ("score", "rating"),
    [
        (100, 5),
        (85, 5),
        (84, 4),
        (70, 4),
        (69, 3),
        (50, 3),
        (49, 2),
        (30, 2),
        (29, 1),
        (1, 1),
        (0, 0),
    ],
)
def test_rating_thresholds(score, rating):
    assert rating_for(score) == rating
