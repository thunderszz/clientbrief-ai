"""Prompt sections shared by the reasoning agents, so the same research reads the same way."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def company_profile_lines(profile: Mapping[str, Any]) -> list[str]:
    """``key: value`` lines for the company profile fields that are filled in."""
    lines = [
        f"{key}: {profile[key]}"
        for key in ("industry", "business_model", "company_positioning")
        if profile.get(key)
    ]
    for key in ("products", "customer_segments", "geographies", "complexity_signals"):
        items = profile.get(key, [])
        if items:
            lines.append(f"{key}: {', '.join(str(item) for item in items[:6])}")
    return lines


def signals_section(signals: list[dict[str, Any]]) -> list[str]:
    """The ten most relevant signals, one numbered line each, or a note that there are none."""
    if not signals:
        return ["\nNo recent strategic signals available."]
    lines = ["", "=== Recent Strategic Signals ==="]
    for number, signal in enumerate(signals[:10], 1):
        date = f" ({signal['date']})" if signal.get("date") else ""
        lines.append(
            f"{number}. [{signal.get('signal_type', '?')}]{date} "
            f"{signal.get('description', '')}  "
            f"Implication: {signal.get('implication', 'n/a')}  "
            f"(confidence: {signal.get('confidence', '?')})"
        )
    return lines


def audience_lines(audience: Mapping[str, Any], *, with_style: bool = False) -> list[str]:
    """The audience context the previous node produced, optionally with style and sensitivities."""
    lines = [
        f"Department: {audience.get('department', '')}",
        f"Seniority: {audience.get('role_seniority') or 'unknown'}",
    ]
    if audience.get("likely_priorities"):
        lines.append(f"Likely priorities: {', '.join(audience['likely_priorities'])}")
    if audience.get("relevant_topics"):
        lines.append(f"Relevant topics: {', '.join(audience['relevant_topics'])}")
    if with_style:
        if audience.get("communication_style"):
            lines.append(f"Communication style: {audience['communication_style']}")
        if audience.get("potential_sensitivities"):
            lines.append(
                f"Potential sensitivities: {', '.join(audience['potential_sensitivities'])}"
            )
    return lines


def person_profile_lines(profile: Mapping[str, Any]) -> list[str]:
    """``key: value`` lines for a matched contact. Callers check the match is reliable first."""
    lines = [
        f"{key}: {profile[key]}"
        for key in ("name", "title", "department", "background_summary")
        if profile.get(key)
    ]
    if profile.get("public_signals"):
        lines.append(f"Public signals: {', '.join(profile['public_signals'][:5])}")
    return lines


def user_side_lines(state: Mapping[str, Any], *, with_goal: bool = False) -> list[str]:
    """What the user said about themselves and the outcome they want, one line per field."""
    fields = [
        ("user_role", "User role"),
        ("user_company", "User company"),
        ("user_function", "User function"),
        ("engagement_type", "Engagement type"),
    ]
    if with_goal:
        fields.append(("meeting_goal", "Meeting goal"))
    fields += [("desired_outcome", "Desired outcome"), ("success_definition", "Success definition")]
    return [f"{label}: {state[key]}" for key, label in fields if state.get(key)]
