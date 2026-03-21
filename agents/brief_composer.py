"""Brief Composer — fully deterministic, no LLM calls.

Assembles all structured agent outputs into a final markdown pre-meeting
brief.  Also computes the reliability score and aggregates a deduplicated
source list from all research agents.

Per spec:
- ``meeting_strategy["executive_summary"]`` is inserted at the top of
  the brief, under the header.
- ``brief_reliability = min(5, len(external_signals))``.
- Person-level information is shown only when ``match_confidence`` is
  ``"medium"`` or ``"high"``.

Ownership:  ``final_brief``, ``brief_reliability``, ``sources``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from core.utils import compute_reliability


# Confidence values that permit displaying person-level detail.
_RELIABLE_CONFIDENCE = frozenset({"medium", "high"})

# Meeting-stage display names for the header.
_STAGE_LABELS: dict[str, str] = {
    "first_intro": "First Introduction",
    "discovery_followup": "Discovery Follow-up",
    "solution_discussion": "Solution Discussion",
    "proposal_review": "Proposal Review",
    "negotiation": "Negotiation",
    "client_kickoff": "Client Kick-off",
    "account_expansion": "Account Expansion",
}

# Meeting-type display names for the header.
_MEETING_TYPE_LABELS: dict[str, str] = {
    "consulting_intro": "Consulting Introduction",
    "sales_discovery": "Sales Discovery",
    "account_review": "Account Review",
}


# =============================================================================
# Source aggregation
# =============================================================================

def _aggregate_sources(state: dict[str, Any]) -> list[str]:
    """Collect and deduplicate sources from all research agents.

    Sources come from three places:
    1. ``company_profile["sources"]``   — list[str]
    2. ``person_profile["sources"]``    — list[str]
    3. ``external_signals[*]["source"]`` — str per signal

    Returns a deduplicated list preserving insertion order.
    """
    seen: set[str] = set()
    unique: list[str] = []

    def _add(url: str) -> None:
        url = url.strip()
        if url and url not in seen:
            seen.add(url)
            unique.append(url)

    # Company profile sources
    cp: dict[str, Any] = state.get("company_profile", {})
    for src in cp.get("sources", []):
        _add(str(src))

    # External signal sources
    for sig in state.get("external_signals", []):
        src = sig.get("source")
        if src:
            _add(str(src))

    # Person profile sources
    pp: dict[str, Any] = state.get("person_profile", {})
    for src in pp.get("sources", []):
        _add(str(src))

    return unique


# =============================================================================
# Section builders — each returns a markdown string (may be empty)
# =============================================================================

def _build_header(state: dict[str, Any], reliability: int) -> str:
    company = state.get("company_name", "Unknown Company")
    meeting_type = state.get("meeting_type", "")
    # meeting_stage may be None if the user did not provide it; fall back to
    # the resolved stage from engagement_context, then to "first_intro".
    ec = state.get("engagement_context", {})
    meeting_stage = state.get("meeting_stage") or ec.get("interaction_stage") or "first_intro"
    stakeholder = state.get("stakeholder_role", "")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    type_label = _MEETING_TYPE_LABELS.get(meeting_type, meeting_type)
    stage_label = _STAGE_LABELS.get(meeting_stage, meeting_stage)

    lines = [
        f"# Pre-Meeting Brief: {company}",
        f"**Meeting type:** {type_label} | **Stage:** {stage_label} | **Audience:** {stakeholder}",
        f"**Brief reliability:** {reliability}/5",
        f"*Generated: {ts}*",
    ]

    # Surface cache-hit notice per spec
    if state.get("cache_hit"):
        lines.append("*[cache hit — research data loaded from local cache]*")

    return "\n".join(lines)


def _build_executive_summary(state: dict[str, Any]) -> str:
    ms: dict[str, Any] = state.get("meeting_strategy", {})
    summary = ms.get("executive_summary", "").strip()
    if not summary:
        return "## Executive Summary\n\n*No executive summary available — upstream strategy agent may have failed.*"
    return f"## Executive Summary\n\n{summary}"


def _build_company_snapshot(state: dict[str, Any]) -> str:
    cp: dict[str, Any] = state.get("company_profile", {})
    lines = ["## Company Snapshot"]

    industry = cp.get("industry")
    model = cp.get("business_model")
    positioning = cp.get("company_positioning")

    if industry:
        lines.append(f"- **Industry:** {industry}")
    if model:
        lines.append(f"- **Business model:** {model}")
    if positioning:
        lines.append(f"- **Positioning:** {positioning}")

    for key, label in [
        ("products", "Products/Services"),
        ("customer_segments", "Customer segments"),
        ("geographies", "Geographies"),
        ("complexity_signals", "Complexity signals"),
    ]:
        items = cp.get(key, [])
        if items:
            lines.append(f"- **{label}:** {', '.join(str(i) for i in items)}")

    if len(lines) == 1:
        lines.append("\n*No company profile data available.*")

    return "\n".join(lines)


def _build_signals(state: dict[str, Any]) -> str:
    signals: list[dict] = state.get("external_signals", [])
    lines = ["## Recent Strategic Signals"]

    if not signals:
        lines.append("\n*No recent strategic signals found.*")
        return "\n".join(lines)

    lines.append("")
    for i, sig in enumerate(signals, 1):
        sig_type = sig.get("signal_type", "unknown")
        desc = sig.get("description", "")
        impl = sig.get("implication", "")
        conf = sig.get("confidence", "")
        date = sig.get("date")
        date_str = f" ({date})" if date else ""

        lines.append(f"{i}. **[{sig_type}]{date_str}** {desc}")
        if impl:
            lines.append(f"   - *Implication:* {impl}")
        if conf:
            lines.append(f"   - *Confidence:* {conf}")

    return "\n".join(lines)


def _build_contact(state: dict[str, Any]) -> str:
    pp: dict[str, Any] = state.get("person_profile", {})
    confidence = pp.get("match_confidence", "none")
    lines = ["## Meeting Contact"]

    # Always show the user-provided contact info
    contact_name = state.get("contact_name")
    contact_title = state.get("contact_title")

    if contact_name:
        lines.append(f"\n**Contact:** {contact_name}")
        if contact_title:
            lines.append(f"**Title:** {contact_title}")
    else:
        lines.append("\n*No contact name provided.*")

    # Show enriched profile only when confidence is reliable
    if confidence in _RELIABLE_CONFIDENCE:
        lines.append(f"\n**Profile confidence:** {confidence}")
        for key, label in [
            ("department", "Department"),
            ("tenure_hint", "Tenure"),
            ("background_summary", "Background"),
        ]:
            val = pp.get(key)
            if val:
                lines.append(f"- **{label}:** {val}")
        pub = pp.get("public_signals", [])
        if pub:
            lines.append(f"- **Public signals:** {', '.join(pub)}")
    elif contact_name:
        lines.append(f"\n*No reliable public profile found (confidence: {confidence}).*")

    return "\n".join(lines)


def _build_audience_context(state: dict[str, Any]) -> str:
    ac: dict[str, Any] = state.get("audience_context", {})
    lines = ["## Audience Context"]

    dept = ac.get("department")
    seniority = ac.get("role_seniority")

    if dept:
        lines.append(f"\n**Department:** {dept}")
    if seniority:
        lines.append(f"**Seniority:** {seniority}")

    for key, label in [
        ("likely_priorities", "Likely priorities"),
        ("relevant_topics", "Relevant topics"),
        ("potential_sensitivities", "Potential sensitivities"),
    ]:
        items = ac.get(key, [])
        if items:
            lines.append(f"\n**{label}:**")
            for item in items:
                lines.append(f"- {item}")

    style = ac.get("communication_style")
    if style:
        lines.append(f"\n**Communication style:** {style}")

    if len(lines) == 1:
        lines.append("\n*No audience context available.*")

    return "\n".join(lines)


def _build_engagement_context(state: dict[str, Any]) -> str:
    ec: dict[str, Any] = state.get("engagement_context", {})
    lines = ["## Engagement Context"]

    stage = ec.get("interaction_stage")
    if not stage:
        lines.append("\n*No engagement context available.*")
        return "\n".join(lines)

    stage_label = _STAGE_LABELS.get(stage, stage)
    rel = ec.get("relationship_status")
    mode = ec.get("conversation_mode")
    objective = ec.get("stage_objective")
    tone = ec.get("recommended_tone")
    pressure = ec.get("next_step_pressure")

    lines.append(f"\n**Stage:** {stage_label}")
    if rel:
        lines.append(f"**Relationship:** {rel.replace('_', ' ')}")
    if mode:
        lines.append(f"**Conversation mode:** {mode.replace('_', ' ')}")
    if objective:
        lines.append(f"**Stage objective:** {objective}")
    if tone:
        lines.append(f"**Recommended tone:** {tone}")
    if pressure:
        lines.append(f"**Next-step pressure:** {pressure}")

    risks = ec.get("decision_risks", [])
    if risks:
        lines.append("\n**Decision risks:**")
        for r in risks:
            lines.append(f"- {r}")

    continuity = ec.get("continuity_context")
    if continuity:
        lines.append(f"\n**Prior interaction context:** {continuity}")

    return "\n".join(lines)


def _build_pain_points(state: dict[str, Any]) -> str:
    pph: dict[str, Any] = state.get("pain_point_hypotheses", {})
    lines = ["## Likely Pain Points"]

    facts = pph.get("facts", [])
    inferences = pph.get("inferences", [])
    hypotheses = pph.get("hypotheses", [])

    if not facts and not inferences and not hypotheses:
        lines.append("\n*No pain-point analysis available.*")
        return "\n".join(lines)

    if facts:
        lines.append("\n**Facts:**")
        for f in facts:
            lines.append(f"- {f}")

    if inferences:
        lines.append("\n**Inferences:**")
        for inf in inferences:
            lines.append(f"- [INFERENCE] {inf}")

    if hypotheses:
        lines.append("\n**Hypotheses:**")
        for h in hypotheses:
            conf = h.get("confidence", "?")
            hyp = h.get("hypothesis", "")
            evidence = h.get("supporting_evidence", [])
            lines.append(f"- [HYPOTHESIS — {conf}] {hyp}")
            if evidence:
                for e in evidence:
                    lines.append(f"  - {e}")

    return "\n".join(lines)


def _build_questions(state: dict[str, Any]) -> str:
    ms: dict[str, Any] = state.get("meeting_strategy", {})
    questions = ms.get("recommended_questions", [])
    lines = ["## Suggested Meeting Questions"]

    if not questions:
        lines.append("\n*No recommended questions available.*")
        return "\n".join(lines)

    lines.append("")
    for i, q in enumerate(questions, 1):
        lines.append(f"{i}. {q}")

    return "\n".join(lines)


def _build_items_to_validate(state: dict[str, Any]) -> str:
    pph: dict[str, Any] = state.get("pain_point_hypotheses", {})
    items = pph.get("items_to_validate", [])
    lines = ["## Items to Validate During Meeting"]

    if not items:
        lines.append("\n*No specific validation items identified.*")
        return "\n".join(lines)

    lines.append("")
    for item in items:
        lines.append(f"- [ ] {item}")

    return "\n".join(lines)


def _build_your_position(state: dict[str, Any]) -> str:
    """Render user-side context when any fields are present."""
    lines = ["## Your Position"]

    user_role = state.get("user_role")
    user_company = state.get("user_company")
    user_function = state.get("user_function")
    engagement_type = state.get("engagement_type")
    desired_outcome = state.get("desired_outcome")
    success_definition = state.get("success_definition")

    has_any = any([user_role, user_company, user_function,
                   engagement_type, desired_outcome, success_definition])

    if not has_any:
        return ""  # Omit section entirely when no user-side context

    identity_parts = []
    if user_role:
        identity_parts.append(user_role)
    if user_company:
        identity_parts.append(f"at {user_company}")
    if identity_parts:
        lines.append(f"\n**You:** {' '.join(identity_parts)}")

    if user_function:
        lines.append(f"**Function:** {user_function}")
    if engagement_type:
        lines.append(f"**Engagement type:** {engagement_type}")
    if desired_outcome:
        lines.append(f"**Desired outcome:** {desired_outcome}")
    if success_definition:
        lines.append(f"**Success looks like:** {success_definition}")

    return "\n".join(lines)


def _build_meeting_prep_notes(state: dict[str, Any]) -> str:
    ms: dict[str, Any] = state.get("meeting_strategy", {})
    lines = ["## Meeting Prep Notes"]

    opening = ms.get("opening_angle", "").strip()
    talking = ms.get("talking_points", [])
    objections = ms.get("possible_objections", [])
    next_step = ms.get("suggested_next_step", "").strip()

    if opening:
        lines.append(f"\n**Opening angle:** {opening}")

    if talking:
        lines.append("\n**Talking points:**")
        for tp in talking:
            lines.append(f"- {tp}")

    if objections:
        lines.append("\n**Possible objections:**")
        for obj in objections:
            lines.append(f"- {obj}")

    if next_step:
        lines.append(f"\n**Suggested next step:** {next_step}")

    if len(lines) == 1:
        lines.append("\n*No meeting prep notes available.*")

    return "\n".join(lines)


def _build_source_footer(sources: list[str]) -> str:
    if not sources:
        return "*Sources: none*"

    parts = ["---", "", "*Sources:*", ""]
    for i, src in enumerate(sources, 1):
        parts.append(f"{i}. {src}")

    return "\n".join(parts)


# =============================================================================
# Public entry-point
# =============================================================================

def run(state: dict[str, Any]) -> dict[str, Any]:
    """Compose the final markdown brief from all upstream state slices.

    Writes: ``final_brief``, ``brief_reliability``, ``sources``.
    """

    # --- 1. Compute reliability -----------------------------------------------
    external_signals: list = state.get("external_signals", [])
    reliability = compute_reliability(external_signals)

    # --- 2. Aggregate sources -------------------------------------------------
    sources = _aggregate_sources(state)

    # --- 3. Assemble sections -------------------------------------------------
    # "Your Position" is conditional — only appears when user-side context exists.
    your_position = _build_your_position(state)

    sections: list[str] = [_build_header(state, reliability)]
    if your_position:
        sections.append("")
        sections.append(your_position)
    sections.append("")
    sections.append(_build_executive_summary(state))
    sections.append("")
    sections.append(_build_company_snapshot(state))
    sections.append("")
    sections.append(_build_signals(state))
    sections.append("")
    sections.append(_build_contact(state))
    sections.append("")
    sections.append(_build_audience_context(state))
    sections.append("")
    sections.append(_build_engagement_context(state))
    sections.append("")
    sections.append(_build_pain_points(state))
    sections.append("")
    sections.append(_build_questions(state))
    sections.append("")
    sections.append(_build_items_to_validate(state))
    sections.append("")
    sections.append(_build_meeting_prep_notes(state))
    sections.append("")
    sections.append(_build_source_footer(sources))

    final_brief = "\n".join(sections)

    return {
        "final_brief": final_brief,
        "brief_reliability": reliability,
        "sources": sources,
    }
