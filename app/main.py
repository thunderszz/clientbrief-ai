"""ClientBrief AI — Interactive CLI entrypoint.

Flow
----
1. Welcome screen
2. Guided free-text input  (user describes the meeting in plain language)
3. Intake agent parses text → shows what was understood
4. Confirmation / correction loop
   a. Show all extracted fields
   b. Prompt to fill any missing important fields
   c. Flag low-confidence fields — user can accept or correct them
5. Show engagement inference (if inferred) — user can accept or correct
6. Run LangGraph workflow with live stage progress
7. Display the full brief in the terminal
8. Save brief to ./output/{company}_{timestamp}.md

Usage::

    cd clientbreef_ai
    python -m app.main
    # or
    python app/main.py
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

# Load .env before anything else so API keys are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional — keys must be set manually in env

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.spinner import Spinner
from rich.live import Live
from rich.table import Table
from rich import box
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# Stage labels shown during workflow execution
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

# ---------------------------------------------------------------------------
# Field display metadata for the confirmation panel
# ---------------------------------------------------------------------------
_FIELD_META: list[tuple[str, str, bool]] = [
    # (key,                     label,                   required)
    ("company_name",            "Company name",           True),
    ("domain",                  "Company domain",         True),
    ("meeting_type",            "Meeting type",           True),
    ("stakeholder_role",        "Stakeholder / audience", True),
    ("meeting_goal",            "Meeting goal",           True),
    ("contact_name",            "Contact name",           False),
    ("contact_title",           "Contact title",          False),
    ("meeting_stage",           "Meeting stage",          False),
    ("relationship_status",     "Relationship status",    False),
    ("previous_meeting_outcome","Previous outcome",       False),
    ("meeting_notes_summary",   "Prior meeting notes",    False),
]

_VALID_MEETING_TYPES   = {"consulting_intro", "sales_discovery", "account_review"}
_VALID_STAGES          = {
    "first_intro", "discovery_followup", "solution_discussion",
    "proposal_review", "negotiation", "client_kickoff", "account_expansion",
}
_VALID_REL_STATUSES    = {"prospect", "active_opportunity", "client", "former_client"}
_VALID_PREV_OUTCOMES   = {"positive", "neutral", "unclear", "no_previous_meeting"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_line() -> None:
    console.print("")


def _ask(prompt: str, default: str = "", optional: bool = False) -> str | None:
    """Prompt for a string value. Returns None if optional and left blank."""
    suffix = " [dim](optional — press Enter to skip)[/dim]" if optional else ""
    value = Prompt.ask(f"[bold cyan]{prompt}[/bold cyan]{suffix}", default=default, console=console)
    if optional and not value.strip():
        return None
    return value.strip() or None


def _show_extracted_table(fields: dict) -> None:
    """Render extracted fields as a styled table."""
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta", expand=False)
    table.add_column("Field", style="dim", width=24)
    table.add_column("Extracted value")

    for key, label, required in _FIELD_META:
        val = fields.get(key)
        if val:
            table.add_row(label, f"[green]{val}[/green]")
        elif required:
            table.add_row(label, "[red]MISSING[/red]")
        else:
            table.add_row(label, "[dim]—[/dim]")

    console.print(table)


def _prompt_fill_missing(fields: dict, missing: list[str]) -> dict:
    """For each missing important field, ask the user to provide it."""
    if not missing:
        return fields

    # Filter to only the truly important ones (not recommended)
    important_missing = [m for m in missing if "(recommended)" not in m]
    if not important_missing:
        return fields

    console.print(Panel(
        "[yellow]Some required fields couldn't be extracted. Please fill them in:[/yellow]",
        border_style="yellow",
    ))

    key_for_label = {label: key for key, label, _ in _FIELD_META}

    for label in important_missing:
        key = key_for_label.get(label)
        if not key:
            continue
        if key == "meeting_type":
            console.print(
                "  [dim]Options: [/dim]"
                "[cyan]consulting_intro[/cyan] | [cyan]sales_discovery[/cyan] | [cyan]account_review[/cyan]"
            )
        val = _ask(f"  {label}")
        if val:
            fields[key] = val

    return fields


def _prompt_confirm_low_confidence(fields: dict, low_confidence: list[dict]) -> dict:
    """Show low-confidence fields and let the user accept or correct them."""
    if not low_confidence:
        return fields

    console.print(Panel(
        "[yellow]I extracted the following fields but I'm not fully confident. "
        "Please confirm or correct:[/yellow]",
        border_style="yellow",
    ))

    for lc in low_confidence:
        field = lc.get("field", "")
        value = lc.get("value", "")
        reason = lc.get("reason", "")

        console.print(
            f"\n  [bold]{field}[/bold] = [cyan]{value}[/cyan]\n"
            f"  [dim]Reason for uncertainty: {reason}[/dim]"
        )
        keep = Confirm.ask(f"  Accept [cyan]{value}[/cyan] for [bold]{field}[/bold]?", console=console, default=True)
        if not keep:
            new_val = _ask(f"  Enter correct value for {field}", optional=True)
            if new_val:
                # map field name back to key (low_confidence uses field_name from agent)
                # field already IS the key
                fields[field] = new_val
            else:
                fields[field] = None  # user cleared it — treat as missing

    return fields


def _show_engagement_inference(engagement: dict) -> None:
    """Display what the inference agent inferred about engagement context."""
    confidence = engagement.get("confidence", "")
    if confidence == "not_needed":
        return  # all fields were explicit — nothing to show

    console.print(Rule("[dim]Engagement inference[/dim]"))

    stage    = engagement.get("inferred_meeting_stage")
    rel      = engagement.get("inferred_relationship_status")
    outcome  = engagement.get("inferred_previous_meeting_outcome")
    summary  = engagement.get("reasoning_summary", "")

    table = Table(box=box.SIMPLE, show_header=False, expand=False)
    table.add_column("", style="dim", width=26)
    table.add_column("")

    if stage:
        table.add_row("Inferred meeting stage", f"[cyan]{stage}[/cyan]")
    if rel:
        table.add_row("Inferred relationship", f"[cyan]{rel}[/cyan]")
    if outcome:
        table.add_row("Inferred previous outcome", f"[cyan]{outcome}[/cyan]")
    table.add_row("Confidence", f"[yellow]{confidence}[/yellow]")
    if summary:
        table.add_row("Reasoning", f"[dim]{summary}[/dim]")

    console.print(table)


def _run_workflow(workflow_input: dict) -> dict:
    """Invoke the LangGraph pipeline with live progress display."""
    from core.workflow import app as workflow_app

    stages = [label for _, label in _STAGE_LABELS]
    current_stage = [0]
    result_holder: list[dict] = []

    def _invoke() -> None:
        result = workflow_app.invoke(workflow_input)
        result_holder.append(result)

    import threading

    thread = threading.Thread(target=_invoke, daemon=True)
    thread.start()

    with Live(console=console, refresh_per_second=4) as live:
        idx = 0
        while thread.is_alive():
            label = stages[min(idx, len(stages) - 1)]
            live.update(
                Panel(
                    f"[bold cyan]⏳  {label}…[/bold cyan]",
                    border_style="cyan",
                    title="[bold]Running ClientBrief AI[/bold]",
                )
            )
            thread.join(timeout=0.8)
            idx = min(idx + 1, len(stages) - 1)

        live.update(
            Panel("[bold green]✓  Brief complete[/bold green]", border_style="green")
        )

    return result_holder[0] if result_holder else {}


def _save_brief(company_name: str, brief: str) -> Path:
    """Write the brief to ./output/{company}_{timestamp}.md"""
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in company_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{safe_name}_{timestamp}.md"
    path.write_text(brief, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    console.print(Panel.fit(
        "[bold white]ClientBrief AI[/bold white]\n"
        "[dim]Pre-meeting intelligence — powered by Gemini + Tavily[/dim]",
        border_style="bright_blue",
        padding=(1, 4),
    ))

    console.print(
        "\n[bold]Describe your upcoming meeting in plain language.[/bold]\n"
        "[dim]Include as much as you can — the more context, the better the brief.\n\n"
        "Things that help:\n"
        "  • Company name and website\n"
        "  • Who you're meeting (name and title if known)\n"
        "  • What you want to achieve in the meeting\n"
        "  • Whether this is a first meeting or a follow-up\n"
        "  • Any relevant history or prior interactions\n[/dim]"
    )

    raw_text = Prompt.ask(
        "[bold cyan]Your meeting description[/bold cyan]",
        console=console,
    )

    if not raw_text.strip():
        console.print("[red]No input provided. Exiting.[/red]")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 1 — Run intake agent
    # -----------------------------------------------------------------------
    console.print("")
    with console.status("[cyan]Parsing your meeting description…[/cyan]", spinner="dots"):
        from agents.meeting_intake_agent import run as intake_run
        fields = intake_run(raw_text)

    console.print(Rule("[bold]What I understood[/bold]"))

    # Show confirmation summary
    summary = fields.get("intake_confirmation_summary", "")
    if summary:
        console.print(Panel(f"[dim]{summary}[/dim]", border_style="dim"))

    _show_extracted_table(fields)

    # -----------------------------------------------------------------------
    # Step 2 — Fill missing important fields
    # -----------------------------------------------------------------------
    fields = _prompt_fill_missing(fields, fields.get("intake_missing_fields", []))

    # -----------------------------------------------------------------------
    # Step 3 — Confirm / correct low-confidence fields
    # -----------------------------------------------------------------------
    fields = _prompt_confirm_low_confidence(fields, fields.get("intake_low_confidence_fields", []))

    # -----------------------------------------------------------------------
    # Step 4 — Optional: let user add anything that was missed
    # -----------------------------------------------------------------------
    _clear_line()
    console.print(Rule("[dim]Optional additions[/dim]"))
    console.print("[dim]If anything important is missing or wrong, you can correct it now.[/dim]\n")

    want_edit = Confirm.ask(
        "  Do you want to review or change any other field?",
        console=console,
        default=False,
    )
    if want_edit:
        for key, label, required in _FIELD_META:
            current = fields.get(key) or ""
            if key == "meeting_type":
                console.print(
                    "  [dim]Options: consulting_intro | sales_discovery | account_review[/dim]"
                )
            new_val = Prompt.ask(
                f"  [bold]{label}[/bold]",
                default=current,
                console=console,
            )
            if new_val.strip():
                fields[key] = new_val.strip()
            elif not required:
                fields[key] = None

    # -----------------------------------------------------------------------
    # Step 5 — Final validation before running
    # -----------------------------------------------------------------------
    required_keys = ["company_name", "domain", "meeting_type", "stakeholder_role", "meeting_goal"]
    still_missing = [k for k in required_keys if not fields.get(k)]
    if still_missing:
        console.print(
            f"\n[red]Cannot generate brief — still missing required fields: "
            f"{', '.join(still_missing)}[/red]"
        )
        sys.exit(1)

    # Build the workflow input (strip intake metadata keys — not part of BriefingState directly)
    workflow_input = {k: v for k, v in fields.items()}

    # -----------------------------------------------------------------------
    # Step 6 — Run the pipeline
    # -----------------------------------------------------------------------
    console.print("")
    console.print(Rule("[bold]Generating brief[/bold]"))
    result = _run_workflow(workflow_input)

    if not result:
        console.print("[red]Workflow returned no result. Check logs for errors.[/red]")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 7 — Show engagement inference (transparency)
    # -----------------------------------------------------------------------
    engagement = result.get("engagement_inference", {})
    _show_engagement_inference(engagement)

    # -----------------------------------------------------------------------
    # Step 8 — Show errors / warnings if any
    # -----------------------------------------------------------------------
    errors = result.get("errors", [])
    if errors:
        console.print(Rule("[yellow]Warnings[/yellow]"))
        for err in errors:
            console.print(f"  [yellow]⚠  {err}[/yellow]")

    # -----------------------------------------------------------------------
    # Step 9 — Display reliability score
    # -----------------------------------------------------------------------
    reliability = result.get("brief_reliability", 0)
    score_color = "green" if reliability >= 4 else "yellow" if reliability >= 2 else "red"
    console.print("")
    console.print(Panel(
        f"[bold {score_color}]Brief reliability: {reliability}/5[/bold {score_color}]",
        border_style=score_color,
        expand=False,
    ))

    # -----------------------------------------------------------------------
    # Step 10 — Render the brief
    # -----------------------------------------------------------------------
    final_brief = result.get("final_brief", "")
    if final_brief:
        console.print("")
        console.print(Rule("[bold]Your Pre-Meeting Brief[/bold]"))
        console.print(Markdown(final_brief))
    else:
        console.print("[red]No brief was produced. Check logs.[/red]")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 11 — Save to file
    # -----------------------------------------------------------------------
    company_name = result.get("company_name") or fields.get("company_name", "company")
    saved_path = _save_brief(company_name, final_brief)
    console.print("")
    console.print(Panel(
        f"[green]Brief saved to:[/green] [bold]{saved_path}[/bold]",
        border_style="green",
        expand=False,
    ))


if __name__ == "__main__":
    main()
