"""Transparent, deterministic reliability score for a brief.

The score (0-100) is the sum of named components, each with a cap and a
human-readable detail line, so the UI can show *why* a brief scored what it
did. The 1-5 ``rating`` derived from it feeds the markdown header and CLI.

| component  | measures                                                  | max |
|------------|-----------------------------------------------------------|-----|
| sources    | distinct source URLs (saturates at 20)                     |  15 |
| diversity  | independent sites (saturates at 12) + company's own site   |  20 |
| signals    | strategic signals weighted by confidence (saturates at 6)  |  20 |
| freshness  | share of signals dated within the last 12 months           |  15 |
| contact    | person match confidence (only if a contact was named)      |  15 |
| health     | minus 5 per agent warning, minus 2 per model fallback      |  15 |

The caps are set so that a solid brief lands in the 70s-80s and 100 means
exceptional coverage, fresh news, a confirmed contact and a clean run.

A component that does not apply (no contact named) is left out of the
denominator, so a brief is never penalised for information the user did not
ask for.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urlparse

_SOURCES_MAX, _SOURCES_SATURATION = 15, 20
_DIVERSITY_MAX, _DIVERSITY_SITES_POINTS, _DIVERSITY_SATURATION, _OWN_SITE_BONUS = 20, 15, 12, 5
_SIGNALS_MAX, _SIGNALS_SATURATION = 20, 6.0
_FRESHNESS_MAX, _FRESH_MONTHS, _UNDATED_CREDIT = 15, 12, 0.25
_CONTACT_MAX = 15
_HEALTH_MAX, _WARNING_PENALTY, _FALLBACK_PENALTY = 15, 5, 2
_YEAR_MONTH = re.compile(r"^(\d{4})-(\d{2})")

_SIGNAL_WEIGHTS = {"high": 1.0, "medium": 0.7, "low": 0.4}
_CONTACT_POINTS = {"high": 15, "medium": 10, "low": 3}
_SECOND_LEVEL_LABELS = frozenset({"co", "com", "org", "net", "gov", "ac", "edu"})


def aggregate_sources(state: Mapping[str, Any]) -> list[str]:
    """Collect source URLs from company, signals and person research, deduplicated in order."""
    seen: set[str] = set()
    unique: list[str] = []

    def add(url: Any) -> None:
        url = str(url).strip()
        if url and url not in seen:
            seen.add(url)
            unique.append(url)

    for src in (state.get("company_profile") or {}).get("sources", []):
        add(src)
    for signal in state.get("external_signals") or []:
        if signal.get("source"):
            add(signal["source"])
    for src in (state.get("person_profile") or {}).get("sources", []):
        add(src)
    return unique


def site_of(url_or_host: str) -> str:
    """Reduce a URL or hostname to its registrable site, e.g. ``news.bbc.co.uk`` → ``bbc.co.uk``."""
    text = url_or_host.strip().lower()
    host = urlparse(text).hostname if "://" in text else text.split("/")[0].split(":")[0]
    if not host:
        return ""
    labels = host.split(".")
    if len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL_LABELS:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def rating_for(score: int) -> int:
    """Map a 0-100 score to the 0-5 rating used in the markdown header."""
    if score >= 85:
        return 5
    if score >= 70:
        return 4
    if score >= 50:
        return 3
    if score >= 30:
        return 2
    return 1 if score > 0 else 0


def _component(
    key: str, label: str, points: float, maximum: int, detail: str, applicable: bool = True
):
    return {
        "key": key,
        "label": label,
        "points": round(points, 2),
        "max": maximum,
        "detail": detail,
        "applicable": applicable,
    }


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _months_old(raw: Any, today: date) -> int | None:
    match = _YEAR_MONTH.match(str(raw or ""))
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    return (today.year - year) * 12 + (today.month - month)


def compute_reliability_report(
    state: Mapping[str, Any], today: date | None = None
) -> dict[str, Any]:
    """Score how well-grounded a brief is. See the module docstring for the components."""
    today = today or datetime.now(UTC).date()
    sources = aggregate_sources(state)
    signals = state.get("external_signals") or []
    errors = state.get("errors") or []

    # --- sources ------------------------------------------------------------
    n_sources = len(sources)
    sources_c = _component(
        "sources",
        "Source coverage",
        _SOURCES_MAX * min(n_sources, _SOURCES_SATURATION) / _SOURCES_SATURATION,
        _SOURCES_MAX,
        f"{n_sources} distinct sources" if n_sources != 1 else "1 distinct source",
    )

    # --- diversity ------------------------------------------------------------
    sites = {site for site in (site_of(url) for url in sources) if site}
    company_site = site_of(str(state.get("domain") or ""))
    has_own_site = bool(company_site) and company_site in sites
    diversity_detail = _plural(len(sites), "independent site")
    if has_own_site:
        diversity_detail += f", including {company_site}"
    diversity_c = _component(
        "diversity",
        "Source diversity",
        _DIVERSITY_SITES_POINTS * min(len(sites), _DIVERSITY_SATURATION) / _DIVERSITY_SATURATION
        + (_OWN_SITE_BONUS if has_own_site else 0),
        _DIVERSITY_MAX,
        diversity_detail,
    )

    # --- signals --------------------------------------------------------------
    counts = {level: 0 for level in _SIGNAL_WEIGHTS}
    weighted = 0.0
    for signal in signals:
        level = str(signal.get("confidence", "low")).lower()
        level = level if level in _SIGNAL_WEIGHTS else "low"
        counts[level] += 1
        weighted += _SIGNAL_WEIGHTS[level]
    breakdown = ", ".join(f"{n} {level}" for level, n in counts.items() if n)
    signals_detail = _plural(len(signals), "signal") + (f": {breakdown}" if breakdown else "")
    signals_c = _component(
        "signals",
        "Strategic signals",
        _SIGNALS_MAX * min(weighted, _SIGNALS_SATURATION) / _SIGNALS_SATURATION,
        _SIGNALS_MAX,
        signals_detail,
    )

    # --- freshness ----------------------------------------------------------------
    ages = [_months_old(signal.get("date"), today) for signal in signals]
    recent = sum(1 for age in ages if age is not None and age <= _FRESH_MONTHS)
    undated = sum(1 for age in ages if age is None)
    if signals:
        freshness_points = _FRESHNESS_MAX * (recent + _UNDATED_CREDIT * undated) / len(signals)
        freshness_detail = (
            f"{recent} of {len(signals)} signals from the last {_FRESH_MONTHS} months"
        )
        freshness_detail += f", {undated} undated" if undated else ""
    else:
        freshness_points, freshness_detail = 0, "no signals to date"
    freshness_c = _component(
        "freshness", "Freshness", freshness_points, _FRESHNESS_MAX, freshness_detail
    )

    # --- contact ----------------------------------------------------------------
    if state.get("contact_name"):
        confidence = str((state.get("person_profile") or {}).get("match_confidence") or "none")
        contact_c = _component(
            "contact",
            "Contact match",
            _CONTACT_POINTS.get(confidence, 0),
            _CONTACT_MAX,
            f"match confidence: {confidence}",
        )
    else:
        contact_c = _component(
            "contact", "Contact match", 0, _CONTACT_MAX, "no contact named", applicable=False
        )

    # --- health -------------------------------------------------------------------
    # Failed LLM calls are not counted here: agents already report them as warnings.
    fallbacks = sum(1 for call in state.get("llm_calls") or [] if call.get("fallback"))
    health_parts = []
    if errors:
        health_parts.append(_plural(len(errors), "agent warning"))
    if fallbacks:
        health_parts.append(_plural(fallbacks, "model fallback"))
    health_c = _component(
        "health",
        "Pipeline health",
        max(0, _HEALTH_MAX - _WARNING_PENALTY * len(errors) - _FALLBACK_PENALTY * fallbacks),
        _HEALTH_MAX,
        ", ".join(health_parts) or "no agent warnings",
    )

    components = [sources_c, diversity_c, signals_c, freshness_c, contact_c, health_c]

    if not sources and not signals:
        score = 0  # nothing was found, so there is nothing to be reliable about
    else:
        applicable = [c for c in components if c["applicable"]]
        earned = sum(c["points"] for c in applicable)
        possible = sum(c["max"] for c in applicable)
        score = round(100 * earned / possible)

    return {"score": score, "rating": rating_for(score), "components": components}
