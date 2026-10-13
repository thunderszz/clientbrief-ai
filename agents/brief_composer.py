"""Brief composer: lays the pipeline's output out as one markdown brief, and scores it.

Deterministic. The web UI renders the structured slices directly; this markdown version is
what the CLI prints and what users copy or download.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agents.person_research import has_reliable_match
from core.reliability import aggregate_sources, compute_reliability_report

_STAGE_LABELS = {
    "first_intro": "First Introduction",
    "discovery_followup": "Discovery Follow-up",
    "solution_discussion": "Solution Discussion",
    "proposal_review": "Proposal Review",
    "negotiation": "Negotiation",
    "client_kickoff": "Client Kick-off",
    "account_expansion": "Account Expansion",
}

_MEETING_TYPE_LABELS = {
    "consulting_intro": "Consulting Introduction",
    "sales_discovery": "Sales Discovery",
    "account_review": "Account Review",
}


def _bullets(items: list[Any], prefix: str = "") -> list[str]:
    return [f"- {prefix}{item}" for item in items]


def _section(title: str, lines: list[str], empty_note: str) -> str:
    """A ``## title`` section, or the title and *empty_note* when there is nothing to show."""
    return "\n".join([f"## {title}", *(lines or [f"\n*{empty_note}*"])])


def _header(state: dict[str, Any], report: dict[str, Any]) -> str:
    # The user may have left the stage out; the engagement context has the resolved one.
    stage = (
        state.get("meeting_stage")
        or state.get("engagement_context", {}).get("interaction_stage")
        or "first_intro"
    )
    meeting_type = state.get("meeting_type", "")
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return "\n".join(
        [
            f"# Pre-Meeting Brief: {state.get('company_name', 'Unknown Company')}",
            f"**Meeting type:** {_MEETING_TYPE_LABELS.get(meeting_type, meeting_type)}"
            f" | **Stage:** {_STAGE_LABELS.get(stage, stage)}"
            f" | **Audience:** {state.get('stakeholder_role', '')}",
            f"**Brief reliability:** {report['rating']}/5 ({report['score']}/100)",
            f"*Generated: {generated}*",
        ]
    )


def _your_position(state: dict[str, Any]) -> str:
    """The user's side of the meeting; omitted entirely when they said nothing about it."""
    who = " ".join(
        part
        for part in (state.get("user_role"), f"at {state['user_company']}" if state.get("user_company") else None)
        if part
    )
    lines = [f"\n**You:** {who}"] if who else []
    for key, label in (
        ("user_function", "Function"),
        ("engagement_type", "Engagement type"),
        ("desired_outcome", "Desired outcome"),
        ("success_definition", "Success looks like"),
    ):
        if state.get(key):
            lines.append(f"**{label}:** {state[key]}")
    return "\n".join(["## Your Position", *lines]) if lines else ""


def _executive_summary(state: dict[str, Any]) -> str:
    summary = state.get("meeting_strategy", {}).get("executive_summary", "").strip()
    if not summary:
        summary = "*No executive summary available — upstream strategy agent may have failed.*"
    return f"## Executive Summary\n\n{summary}"


def _company_snapshot(state: dict[str, Any]) -> str:
    profile = state.get("company_profile", {})
    lines = [
        f"- **{label}:** {profile[key]}"
        for key, label in (("industry", "Industry"), ("business_model", "Business model"), ("company_positioning", "Positioning"))
        if profile.get(key)
    ]
    for key, label in (
        ("products", "Products/Services"),
        ("customer_segments", "Customer segments"),
        ("geographies", "Geographies"),
        ("complexity_signals", "Complexity signals"),
    ):
        if profile.get(key):
            lines.append(f"- **{label}:** {', '.join(str(item) for item in profile[key])}")
    return _section("Company Snapshot", lines, "No company profile data available.")


def _signals(state: dict[str, Any]) -> str:
    lines = []
    for number, signal in enumerate(state.get("external_signals", []), 1):
        date = f" ({signal['date']})" if signal.get("date") else ""
        lines.append(f"{number}. **[{signal.get('signal_type', 'unknown')}]{date}** {signal.get('description', '')}")
        if signal.get("implication"):
            lines.append(f"   - *Implication:* {signal['implication']}")
        if signal.get("confidence"):
            lines.append(f"   - *Confidence:* {signal['confidence']}")
    return _section("Recent Strategic Signals", [""] + lines if lines else [], "No recent strategic signals found.")


def _contact(state: dict[str, Any]) -> str:
    profile = state.get("person_profile", {})
    name, title = state.get("contact_name"), state.get("contact_title")
    if not name:
        lines = ["\n*No contact name provided.*"]
    else:
        lines = [f"\n**Contact:** {name}"] + ([f"**Title:** {title}"] if title else [])

    if has_reliable_match(profile):
        lines.append(f"\n**Profile confidence:** {profile['match_confidence']}")
        for key, label in (("department", "Department"), ("tenure_hint", "Tenure"), ("background_summary", "Background")):
            if profile.get(key):
                lines.append(f"- **{label}:** {profile[key]}")
        if profile.get("public_signals"):
            lines.append(f"- **Public signals:** {', '.join(profile['public_signals'])}")
    elif name:
        confidence = profile.get("match_confidence", "none")
        lines.append(f"\n*No reliable public profile found (confidence: {confidence}).*")
    return "\n".join(["## Meeting Contact", *lines])


def _audience(state: dict[str, Any]) -> str:
    audience = state.get("audience_context", {})
    lines = []
    if audience.get("department"):
        lines.append(f"\n**Department:** {audience['department']}")
    if audience.get("role_seniority"):
        lines.append(f"**Seniority:** {audience['role_seniority']}")
    for key, label in (
        ("likely_priorities", "Likely priorities"),
        ("relevant_topics", "Relevant topics"),
        ("potential_sensitivities", "Potential sensitivities"),
    ):
        if audience.get(key):
            lines += [f"\n**{label}:**", *_bullets(audience[key])]
    if audience.get("communication_style"):
        lines.append(f"\n**Communication style:** {audience['communication_style']}")
    return _section("Audience Context", lines, "No audience context available.")


def _engagement(state: dict[str, Any]) -> str:
    context = state.get("engagement_context", {})
    stage = context.get("interaction_stage")
    if not stage:
        return _section("Engagement Context", [], "No engagement context available.")

    lines = [f"\n**Stage:** {_STAGE_LABELS.get(stage, stage)}"]
    if context.get("relationship_status"):
        lines.append(f"**Relationship:** {context['relationship_status'].replace('_', ' ')}")
    if context.get("conversation_mode"):
        lines.append(f"**Conversation mode:** {context['conversation_mode'].replace('_', ' ')}")
    for key, label in (
        ("stage_objective", "Stage objective"),
        ("recommended_tone", "Recommended tone"),
        ("next_step_pressure", "Next-step pressure"),
    ):
        if context.get(key):
            lines.append(f"**{label}:** {context[key]}")
    if context.get("decision_risks"):
        lines += ["\n**Decision risks:**", *_bullets(context["decision_risks"])]
    if context.get("continuity_context"):
        lines.append(f"\n**Prior interaction context:** {context['continuity_context']}")
    return _section("Engagement Context", lines, "")


def _pain_points(state: dict[str, Any]) -> str:
    analysis = state.get("pain_point_hypotheses", {})
    facts = analysis.get("facts", [])
    inferences = analysis.get("inferences", [])
    hypotheses = analysis.get("hypotheses", [])
    lines = []
    if facts:
        lines += ["\n**Facts:**", *_bullets(facts)]
    if inferences:
        lines += ["\n**Inferences:**", *_bullets(inferences, "[INFERENCE] ")]
    if hypotheses:
        lines.append("\n**Hypotheses:**")
        for item in hypotheses:
            lines.append(f"- [HYPOTHESIS — {item.get('confidence', '?')}] {item.get('hypothesis', '')}")
            lines += [f"  - {evidence}" for evidence in item.get("supporting_evidence", [])]
    return _section("Likely Pain Points", lines, "No pain-point analysis available.")


def _questions(state: dict[str, Any]) -> str:
    questions = state.get("meeting_strategy", {}).get("recommended_questions", [])
    lines = [f"{number}. {question}" for number, question in enumerate(questions, 1)]
    return _section("Suggested Meeting Questions", [""] + lines if lines else [], "No recommended questions available.")


def _items_to_validate(state: dict[str, Any]) -> str:
    items = state.get("pain_point_hypotheses", {}).get("items_to_validate", [])
    lines = [f"- [ ] {item}" for item in items]
    return _section(
        "Items to Validate During Meeting", [""] + lines if lines else [], "No specific validation items identified."
    )


def _prep_notes(state: dict[str, Any]) -> str:
    strategy = state.get("meeting_strategy", {})
    opening = strategy.get("opening_angle", "").strip()
    next_step = strategy.get("suggested_next_step", "").strip()
    lines = []
    if opening:
        lines.append(f"\n**Opening angle:** {opening}")
    if strategy.get("talking_points"):
        lines += ["\n**Talking points:**", *_bullets(strategy["talking_points"])]
    if strategy.get("possible_objections"):
        lines += ["\n**Possible objections:**", *_bullets(strategy["possible_objections"])]
    if next_step:
        lines.append(f"\n**Suggested next step:** {next_step}")
    return _section("Meeting Prep Notes", lines, "No meeting prep notes available.")


def _sources_footer(sources: list[str]) -> str:
    if not sources:
        return "*Sources: none*"
    return "\n".join(["---", "", "*Sources:*", "", *(f"{n}. {url}" for n, url in enumerate(sources, 1))])


def run(state: dict[str, Any]) -> dict[str, Any]:
    """Compose ``final_brief`` and score it (``reliability_report``, ``brief_reliability``)."""
    report = compute_reliability_report(state)
    sources = aggregate_sources(state)
    sections = [
        _header(state, report),
        _your_position(state),
        _executive_summary(state),
        _company_snapshot(state),
        _signals(state),
        _contact(state),
        _audience(state),
        _engagement(state),
        _pain_points(state),
        _questions(state),
        _items_to_validate(state),
        _prep_notes(state),
        _sources_footer(sources),
    ]
    return {
        "final_brief": "\n\n".join(section for section in sections if section),
        "brief_reliability": report["rating"],
        "reliability_report": report,
        "sources": sources,
    }
