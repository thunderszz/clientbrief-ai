"""ClientBrief AI — Streamlit single-page wizard.

Five-stage flow:
  1. Free-text meeting description input
  2. AI understanding: confirmation summary, missing fields, low-confidence fields
  3. Structured review/edit form (all fields editable)
  4. Workflow execution with progress (transient — spinner)
  5. Final brief rendered with reliability score and actions

Usage::

    cd clientbreef_ai
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ClientBrief AI",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAGE_LABELS: list[tuple[str, str]] = [
    ("orchestrate",        "Validating inputs"),
    ("infer_engagement",   "Inferring engagement context"),
    ("engagement_context", "Resolving engagement stage"),
    ("plan",               "Building research plan"),
    ("research_company",   "Researching company"),
    ("research_signals",   "Detecting strategic signals"),
    ("research_person",    "Looking up meeting contact"),
    ("audience_context",   "Building audience context"),
    ("hypothesize",        "Generating pain point hypotheses"),
    ("strategize",         "Crafting meeting strategy"),
    ("compose",            "Composing brief"),
]

# Fields for the structured edit form — grouped
_MEETING_CONTEXT_FIELDS: list[tuple[str, str, str, bool]] = [
    # (key, label, help_text, required)
    ("company_name",             "Company name",            "Name of the company you're meeting", True),
    ("domain",                   "Company domain",          "e.g. stripe.com", True),
    ("meeting_type",             "Meeting type",            "consulting_intro | sales_discovery | account_review", True),
    ("stakeholder_role",         "Stakeholder / audience",  "Role of the person you're meeting, e.g. VP Finance", True),
    ("meeting_goal",             "Meeting goal",            "What you want to achieve", True),
    ("contact_name",             "Contact name",            "Name of the person (optional)", False),
    ("contact_title",            "Contact title",           "Their job title (optional)", False),
    ("meeting_stage",            "Meeting stage",           "first_intro | discovery_followup | solution_discussion | proposal_review | negotiation | client_kickoff | account_expansion", False),
    ("relationship_status",      "Relationship status",     "prospect | active_opportunity | client | former_client", False),
    ("previous_meeting_outcome", "Previous outcome",        "positive | neutral | unclear | no_previous_meeting", False),
    ("meeting_notes_summary",    "Prior meeting notes",     "Summary of what was discussed before", False),
]

_USER_CONTEXT_FIELDS: list[tuple[str, str, str, bool]] = [
    ("user_role",           "Your role",             "e.g. Senior Consultant", False),
    ("user_company",        "Your company",          "e.g. McKinsey", False),
    ("user_function",       "Your function",         "e.g. Strategy consulting", False),
    ("engagement_type",     "Engagement type",       "sales | consulting | interview | partnership | investing", False),
    ("desired_outcome",     "Desired outcome",       "What success looks like for YOU in this meeting", False),
    ("success_definition",  "Success definition",    "How you'll measure if the meeting went well", False),
]

REQUIRED_KEYS = ["company_name", "domain", "meeting_type", "stakeholder_role", "meeting_goal"]


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    defaults = {
        "stage": 1,
        "raw_text": "",
        "intake_result": None,
        "workflow_input": {},
        "result": None,
        "run_errors": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _go_to(stage: int) -> None:
    st.session_state.stage = stage


def _reset() -> None:
    for k in ["raw_text", "intake_result", "workflow_input", "result", "run_errors"]:
        st.session_state[k] = "" if k == "raw_text" else None if k in ("intake_result", "result") else {} if k == "workflow_input" else []
    st.session_state.stage = 1


# ---------------------------------------------------------------------------
# Stage 1: Free-text intake
# ---------------------------------------------------------------------------

def _render_stage_1() -> None:
    st.markdown("### Describe your upcoming meeting")
    st.markdown(
        "Write freely — include the company name, who you're meeting, "
        "what you want to achieve, and any relevant history. "
        "The more context you provide, the better the brief."
    )

    raw = st.text_area(
        "Meeting description",
        value=st.session_state.raw_text,
        height=200,
        placeholder=(
            "Example: I have a second meeting with the CFO of Stripe next Tuesday. "
            "We discussed their finance automation needs last time and they seemed "
            "interested in our consulting approach. I want to move towards a proposal."
        ),
    )
    st.session_state.raw_text = raw

    col1, col2 = st.columns([1, 5])
    with col1:
        parse_clicked = st.button("Parse context", type="primary", disabled=not raw.strip())
    with col2:
        st.markdown("")  # spacer

    # About section — collapsed, below the action area
    with st.expander("About this project", expanded=False):
        st.markdown(
            "**ClientBrief AI** is a multi-agent pre-meeting intelligence system that prepares "
            "structured briefings for business meetings.\n\n"
            "**How it works**\n"
            "1. A free-text intake agent (LLM-powered) parses your meeting description into structured fields\n"
            "2. An 11-node LangGraph pipeline orchestrates the research and reasoning:\n"
            "   - Engagement inference and context resolution\n"
            "   - Parallel company, signals, and contact research (via Tavily)\n"
            "   - Audience context building\n"
            "   - Pain point hypothesis generation\n"
            "   - Meeting strategy crafting\n"
            "3. A deterministic composer assembles the final markdown brief\n\n"
            "**Stack:** Google Gemini · Tavily Search · LangGraph · Streamlit\n\n"
            "**Architecture principles**\n"
            "- Deterministic logic for validation, normalization, and composition — LLMs only where reasoning is needed\n"
            "- Facts, inferences, and hypotheses are always kept separate\n"
            "- Every field has clear agent ownership — no node overwrites what it doesn't own"
        )

    if parse_clicked and raw.strip():
        with st.spinner("Parsing your meeting description..."):
            from agents.meeting_intake_agent import run as intake_run
            result = intake_run(raw.strip())
        st.session_state.intake_result = result
        _go_to(2)
        st.rerun()


# ---------------------------------------------------------------------------
# Stage 2: AI understanding / confirmation
# ---------------------------------------------------------------------------

def _render_stage_2() -> None:
    fields = st.session_state.intake_result
    if not fields:
        _go_to(1)
        st.rerun()
        return

    st.markdown("### What I understood")

    # Confirmation summary
    summary = fields.get("intake_confirmation_summary", "")
    if summary:
        st.info(summary)

    # Show extracted fields as a table
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Meeting & External Context**")
        for key, label, _, required in _MEETING_CONTEXT_FIELDS:
            val = fields.get(key)
            if val:
                st.markdown(f"- **{label}:** {val}")
            elif required:
                st.markdown(f"- **{label}:** :red[MISSING]")
            else:
                st.markdown(f"- **{label}:** *—*")

    with col_right:
        st.markdown("**Your Context**")
        for key, label, _, _ in _USER_CONTEXT_FIELDS:
            val = fields.get(key)
            if val:
                st.markdown(f"- **{label}:** {val}")
            else:
                st.markdown(f"- **{label}:** *—*")

    # Missing fields warning
    missing = fields.get("intake_missing_fields", [])
    if missing:
        st.warning(f"**Missing fields:** {', '.join(missing)}")

    # Low-confidence fields
    low_conf = fields.get("intake_low_confidence_fields", [])
    if low_conf:
        st.markdown("**Low-confidence extractions:**")
        for lc in low_conf:
            fname = lc.get("field", lc.get("field_name", ""))
            val = lc.get("value", lc.get("extracted_value", ""))
            reason = lc.get("reason", "")
            st.markdown(f"- **{fname}** = `{val}` — *{reason}*")

    st.markdown("---")

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("Review & Edit", type="primary"):
            # Pre-populate workflow_input from intake
            st.session_state.workflow_input = {k: v for k, v in fields.items()}
            _go_to(3)
            st.rerun()
    with col2:
        if st.button("Back"):
            _go_to(1)
            st.rerun()


# ---------------------------------------------------------------------------
# Stage 3: Structured review / edit form
# ---------------------------------------------------------------------------

def _render_stage_3() -> None:
    fields = st.session_state.workflow_input
    if not fields:
        fields = st.session_state.intake_result or {}
        st.session_state.workflow_input = dict(fields)

    st.markdown("### Review & edit structured fields")
    st.markdown("All fields are editable. Required fields are marked with **\\***.")

    with st.form("edit_form"):
        col_left, col_right = st.columns(2)

        edited = {}

        with col_left:
            st.markdown("**Meeting & External Context**")
            for key, label, help_text, required in _MEETING_CONTEXT_FIELDS:
                display_label = f"{label} *" if required else label
                current = fields.get(key) or ""
                if key == "meeting_notes_summary":
                    val = st.text_area(display_label, value=current, help=help_text, height=80)
                else:
                    val = st.text_input(display_label, value=current, help=help_text)
                edited[key] = val.strip() if val.strip() else None

        with col_right:
            st.markdown("**Your Context**")
            for key, label, help_text, _ in _USER_CONTEXT_FIELDS:
                current = fields.get(key) or ""
                val = st.text_input(label, value=current, help=help_text)
                edited[key] = val.strip() if val.strip() else None

        st.markdown("---")
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            submitted = st.form_submit_button("Generate Brief", type="primary")
        with col2:
            back = st.form_submit_button("Back")

    if back:
        _go_to(2)
        st.rerun()

    if submitted:
        # Validate required fields
        missing = [k for k in REQUIRED_KEYS if not edited.get(k)]
        if missing:
            labels = {k: l for k, l, _, _ in _MEETING_CONTEXT_FIELDS}
            st.error(f"Missing required fields: {', '.join(labels.get(k, k) for k in missing)}")
            return

        # Merge intake metadata into workflow input
        intake = st.session_state.intake_result or {}
        workflow_input = dict(edited)
        for meta_key in ["raw_meeting_context", "intake_missing_fields",
                         "intake_low_confidence_fields", "intake_confirmation_summary"]:
            if meta_key in intake:
                workflow_input[meta_key] = intake[meta_key]

        st.session_state.workflow_input = workflow_input
        _go_to(4)
        st.rerun()


# ---------------------------------------------------------------------------
# Stage 4: Running workflow (transient — auto-advances to stage 5)
# ---------------------------------------------------------------------------

def _render_stage_4() -> None:
    workflow_input = st.session_state.workflow_input
    if not workflow_input:
        _go_to(1)
        st.rerun()
        return

    st.markdown("### Generating your brief...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Run the workflow
    from core.workflow import app as workflow_app

    status_text.markdown("**Initializing pipeline...**")
    progress_bar.progress(5)

    try:
        result = workflow_app.invoke(workflow_input)
    except Exception as e:
        st.error(f"Workflow failed: {e}")
        st.session_state.run_errors = [str(e)]
        if st.button("Back to edit"):
            _go_to(3)
            st.rerun()
        return

    progress_bar.progress(100)
    status_text.markdown("**Brief complete!**")

    st.session_state.result = result
    st.session_state.run_errors = result.get("errors", [])
    _go_to(5)
    st.rerun()


# ---------------------------------------------------------------------------
# Stage 5: Final brief
# ---------------------------------------------------------------------------

def _render_stage_5() -> None:
    result = st.session_state.result
    if not result:
        _go_to(1)
        st.rerun()
        return

    final_brief = result.get("final_brief", "")
    reliability = result.get("brief_reliability", 0)
    errors = st.session_state.run_errors or []

    # Header with reliability badge
    score_color = "green" if reliability >= 4 else "orange" if reliability >= 2 else "red"
    st.markdown(f"### Brief ready &nbsp; :{score_color}_circle: Reliability: **{reliability}/5**")

    # Errors / warnings
    if errors:
        with st.expander(f"Warnings ({len(errors)})", expanded=False):
            for err in errors:
                st.warning(err)

    # Engagement inference (transparency)
    engagement = result.get("engagement_inference", {})
    conf = engagement.get("confidence", "")
    if conf and conf != "not_needed":
        with st.expander("Engagement inference (AI-inferred)", expanded=False):
            stage = engagement.get("inferred_meeting_stage")
            rel = engagement.get("inferred_relationship_status")
            outcome = engagement.get("inferred_previous_meeting_outcome")
            reasoning = engagement.get("reasoning_summary", "")
            if stage:
                st.markdown(f"- **Inferred stage:** {stage}")
            if rel:
                st.markdown(f"- **Inferred relationship:** {rel}")
            if outcome:
                st.markdown(f"- **Inferred previous outcome:** {outcome}")
            st.markdown(f"- **Confidence:** {conf}")
            if reasoning:
                st.markdown(f"- **Reasoning:** {reasoning}")

    # The brief itself
    st.markdown("---")
    st.markdown(final_brief)
    st.markdown("---")

    # Action buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Start new"):
            _reset()
            st.rerun()
    with col2:
        if st.button("Edit context"):
            _go_to(3)
            st.rerun()
    with col3:
        if st.button("Regenerate"):
            _go_to(4)
            st.rerun()
    with col4:
        company = result.get("company_name") or st.session_state.workflow_input.get("company_name", "company")
        st.download_button(
            "Download .md",
            data=final_brief,
            file_name=f"{company}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

    # Save to output directory automatically
    _auto_save(result)


def _auto_save(result: dict) -> None:
    """Save the brief to ./output/ if not already saved this run."""
    if st.session_state.get("_saved_this_run"):
        return
    brief = result.get("final_brief", "")
    if not brief:
        return
    company = result.get("company_name") or st.session_state.workflow_input.get("company_name", "company")
    out_dir = _PROJECT_ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in company)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{safe_name}_{timestamp}.md"
    path.write_text(brief, encoding="utf-8")
    st.session_state._saved_this_run = True
    st.toast(f"Brief saved to {path.name}")


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _init_session_state()

    # Sidebar with minimal branding
    with st.sidebar:
        st.markdown("## ClientBrief AI")
        st.markdown("Pre-meeting intelligence")
        st.markdown("---")
        stage = st.session_state.stage
        steps = ["Describe meeting", "AI understanding", "Review & edit", "Generating...", "Brief ready"]
        for i, label in enumerate(steps, 1):
            if i < stage:
                st.markdown(f"~~{i}. {label}~~")
            elif i == stage:
                st.markdown(f"**{i}. {label}** ←")
            else:
                st.markdown(f"{i}. {label}")

        st.markdown("---")
        if st.button("Reset", key="sidebar_reset"):
            _reset()
            st.rerun()

    # Route to current stage
    stage = st.session_state.stage
    if stage == 1:
        _render_stage_1()
    elif stage == 2:
        _render_stage_2()
    elif stage == 3:
        _render_stage_3()
    elif stage == 4:
        _render_stage_4()
    elif stage == 5:
        _render_stage_5()
    else:
        _reset()
        st.rerun()


if __name__ == "__page__":
    main()

main()
