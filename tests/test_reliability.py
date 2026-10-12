"""Tests for the transparent reliability score."""

from __future__ import annotations

import pytest

from core.reliability import aggregate_sources, compute_reliability_report, rating_for

_SITES = [
    "https://www.satispay.com/en-it/",
    "https://www.ilsole24ore.com/a",
    "https://www.fintechfutures.com/b",
    "https://ffnews.com/c",
    "https://www.cbinsights.com/d",
    "https://www.linkedin.com/e",
    "https://startupluxembourg.com/f",
    "https://qonto.com/g",
]


def _full_state(**overrides):
    company_sources = _SITES + [f"https://www.satispay.com/page{i}" for i in range(4)]
    state = {
        "domain": "satispay.com",
        "contact_name": "Alberto Dalmasso",
        "company_profile": {"sources": company_sources},
        "external_signals": [
            {"source": f"https://news{i}.test/s", "confidence": "high"} for i in range(5)
        ],
        "person_profile": {"match_confidence": "high", "sources": []},
        "errors": [],
    }
    state.update(overrides)
    return state


def _component(report, key):
    return next(c for c in report["components"] if c["key"] == key)


def test_full_house_scores_100():
    report = compute_reliability_report(_full_state())

    assert report["score"] == 100
    assert report["rating"] == 5
    assert [c["key"] for c in report["components"]] == [
        "sources",
        "diversity",
        "signals",
        "contact",
        "health",
    ]


def test_empty_state_scores_zero():
    report = compute_reliability_report({})

    assert report["score"] == 0
    assert report["rating"] == 0


def test_no_contact_not_penalised():
    report = compute_reliability_report(_full_state(contact_name=None, person_profile={}))

    contact = _component(report, "contact")
    assert contact["applicable"] is False
    assert contact["detail"] == "no contact named"
    assert report["score"] == 100


def test_low_contact_match_scores_partially():
    state = _full_state(person_profile={"match_confidence": "low", "sources": []})

    contact = _component(compute_reliability_report(state), "contact")

    assert contact["points"] == 3
    assert contact["applicable"] is True


def test_named_contact_without_profile_scores_zero():
    state = _full_state(person_profile={"match_confidence": "none", "sources": []})

    assert _component(compute_reliability_report(state), "contact")["points"] == 0


def test_errors_reduce_health():
    state = _full_state(errors=["company_research: x", "person_research: y"])

    health = _component(compute_reliability_report(state), "health")

    assert health["points"] == 5
    assert health["detail"] == "2 agent warnings"


def test_health_floors_at_zero():
    state = _full_state(errors=["e"] * 9)

    assert _component(compute_reliability_report(state), "health")["points"] == 0


def test_signal_confidence_weights():
    signals = [
        {"source": "https://a.test", "confidence": "high"},
        {"source": "https://b.test", "confidence": "medium"},
        {"source": "https://c.test", "confidence": "low"},
    ]

    component = _component(compute_reliability_report(_full_state(external_signals=signals)), "signals")

    assert component["points"] == pytest.approx(25 * 2.1 / 5)
    assert component["detail"] == "3 signals: 1 high, 1 medium, 1 low"


def test_company_domain_bonus():
    with_own = _component(compute_reliability_report(_full_state()), "diversity")
    others = [s for s in _SITES if "satispay" not in s]
    without_own = _component(
        compute_reliability_report(_full_state(company_profile={"sources": others})), "diversity"
    )

    assert with_own["detail"].endswith("including satispay.com")
    assert with_own["points"] - without_own["points"] == pytest.approx(5)


def test_subdomains_count_as_one_site():
    sources = [
        "https://www.satispay.com/a",
        "https://support.satispay.com/b",
        "https://developers.satispay.com/c",
    ]
    state = {"domain": "www.satispay.com", "company_profile": {"sources": sources}}

    diversity = _component(compute_reliability_report(state), "diversity")

    assert diversity["detail"] == "1 independent site, including satispay.com"


def test_country_code_second_level_domains():
    sources = ["https://www.bbc.co.uk/a", "https://news.bbc.co.uk/b", "https://ft.co.uk/c"]
    state = {"company_profile": {"sources": sources}}

    diversity = _component(compute_reliability_report(state), "diversity")

    assert diversity["detail"] == "2 independent sites"


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
    [(100, 5), (85, 5), (84, 4), (70, 4), (69, 3), (50, 3), (49, 2), (30, 2), (29, 1), (1, 1), (0, 0)],
)
def test_rating_thresholds(score, rating):
    assert rating_for(score) == rating
