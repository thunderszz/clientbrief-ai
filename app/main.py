"""Terminal version of ClientBrief: describe the meeting, check the fields, read the brief.

    python -m app.main

The brief is printed and saved to ``output/<company>_<timestamp>.md``.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table

from core.pipeline import NODE_LABELS
from core.summaries import summarize_update

console = Console()

# (key, label, required) for the fields shown in the confirmation table.
_FIELDS: list[tuple[str, str, bool]] = [
    ("company_name", "Company name", True),
    ("domain", "Company domain", True),
    ("meeting_type", "Meeting type", True),
    ("stakeholder_role", "Stakeholder / audience", True),
    ("meeting_goal", "Meeting goal", True),
    ("contact_name", "Contact name", False),
    ("contact_title", "Contact title", False),
    ("meeting_stage", "Meeting stage", False),
    ("relationship_status", "Relationship status", False),
    ("previous_meeting_outcome", "Previous outcome", False),
    ("meeting_notes_summary", "Prior meeting notes", False),
]
_REQUIRED_KEYS = [key for key, _, required in _FIELDS if required]
_MEETING_TYPE_HINT = "consulting_intro | sales_discovery | account_review"


def _ask(prompt: str, default: str = "", optional: bool = False) -> str | None:
    """Ask for a value; blank answers come back as None."""
    suffix = " [dim](optional — press Enter to skip)[/dim]" if optional else ""
    value = Prompt.ask(f"[bold cyan]{prompt}[/bold cyan]{suffix}", default=default, console=console)
    return value.strip() or None


def _show_extracted_table(fields: dict) -> None:
    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta", expand=False)
    table.add_column("Field", style="dim", width=24)
    table.add_column("Extracted value")
    for key, label, required in _FIELDS:
        value = fields.get(key)
        if value:
            table.add_row(label, f"[green]{value}[/green]")
        elif required:
            table.add_row(label, "[red]MISSING[/red]")
        else:
            table.add_row(label, "[dim]—[/dim]")
    console.print(table)


def _prompt_fill_missing(fields: dict, missing: list[str]) -> dict:
    """Ask for each required field the intake could not find (recommended ones are skipped)."""
    required_missing = [label for label in missing if "(recommended)" not in label]
    if not required_missing:
        return fields

    console.print(Panel(
        "[yellow]Some required fields couldn't be extracted. Please fill them in:[/yellow]",
        border_style="yellow",
    ))
    key_for_label = {label: key for key, label, _ in _FIELDS}
    for label in required_missing:
        key = key_for_label.get(label)
        if not key:
            continue
        if key == "meeting_type":
            console.print(
                "  [dim]Options: [/dim]"
                "[cyan]consulting_intro[/cyan] | [cyan]sales_discovery[/cyan] | [cyan]account_review[/cyan]"
            )
        value = _ask(f"  {label}")
        if value:
            fields[key] = value
    return fields


def _prompt_confirm_low_confidence(fields: dict, low_confidence: list[dict]) -> dict:
    """Let the user accept or correct each value the intake was unsure about."""
    if not low_confidence:
        return fields

    console.print(Panel(
        "[yellow]I extracted the following fields but I'm not fully confident. "
        "Please confirm or correct:[/yellow]",
        border_style="yellow",
    ))
    for item in low_confidence:
        field, value = item.get("field", ""), item.get("value", "")
        console.print(
            f"\n  [bold]{field}[/bold] = [cyan]{value}[/cyan]\n"
            f"  [dim]Reason for uncertainty: {item.get('reason', '')}[/dim]"
        )
        if not Confirm.ask(f"  Accept [cyan]{value}[/cyan] for [bold]{field}[/bold]?", console=console, default=True):
            # An empty answer clears the field, which then counts as missing.
            fields[field] = _ask(f"  Enter correct value for {field}", optional=True)
    return fields


def _prompt_edit_any(fields: dict) -> dict:
    console.print("")
    console.print(Rule("[dim]Optional additions[/dim]"))
    console.print("[dim]If anything important is missing or wrong, you can correct it now.[/dim]\n")
    if not Confirm.ask("  Do you want to review or change any other field?", console=console, default=False):
        return fields

    for key, label, required in _FIELDS:
        if key == "meeting_type":
            console.print(f"  [dim]Options: {_MEETING_TYPE_HINT}[/dim]")
        value = Prompt.ask(f"  [bold]{label}[/bold]", default=fields.get(key) or "", console=console)
        if value.strip():
            fields[key] = value.strip()
        elif not required:
            fields[key] = None
    return fields


def _show_engagement_inference(inference: dict) -> None:
    """Show what the pipeline guessed about the relationship, unless it had nothing to guess."""
    confidence = inference.get("confidence", "")
    if confidence == "not_needed":
        return

    console.print(Rule("[dim]Engagement inference[/dim]"))
    table = Table(box=box.SIMPLE, show_header=False, expand=False)
    table.add_column("", style="dim", width=26)
    table.add_column("")
    for key, label in (
        ("inferred_meeting_stage", "Inferred meeting stage"),
        ("inferred_relationship_status", "Inferred relationship"),
        ("inferred_previous_meeting_outcome", "Inferred previous outcome"),
    ):
        if inference.get(key):
            table.add_row(label, f"[cyan]{inference[key]}[/cyan]")
    table.add_row("Confidence", f"[yellow]{confidence}[/yellow]")
    if inference.get("reasoning_summary"):
        table.add_row("Reasoning", f"[dim]{inference['reasoning_summary']}[/dim]")
    console.print(table)


def _run_workflow(workflow_input: dict, graph: Any = None) -> dict:
    """Run the pipeline, printing each node as it finishes, and return the final state."""
    if graph is None:
        from core.workflow import app as graph

    final_state: dict = {}
    with console.status("[bold cyan]Running ClientBrief AI…[/bold cyan]"):
        for mode, chunk in graph.stream(workflow_input, stream_mode=["updates", "values"]):
            if mode == "values":
                final_state = chunk
                continue
            for node_id, update in chunk.items():
                label = NODE_LABELS.get(node_id, node_id)
                summary = summarize_update(node_id, update)
                console.print(f"  [green]✓[/green] [bold]{label}[/bold]  [dim]{summary}[/dim]")

    console.print("[bold green]✓  Brief complete[/bold green]")
    return final_state


def _reliability_line(result: dict) -> str:
    rating = result.get("brief_reliability", 0)
    score = (result.get("reliability_report") or {}).get("score", 0)
    return f"Brief reliability: {rating}/5 ({score}/100)"


def _save_brief(company_name: str, brief: str) -> Path:
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in company_name)
    path = out_dir / f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path.write_text(brief, encoding="utf-8")
    return path


def _exit(message: str) -> NoReturn:
    console.print(f"[red]{message}[/red]")
    sys.exit(1)


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

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
    raw_text = Prompt.ask("[bold cyan]Your meeting description[/bold cyan]", console=console)
    if not raw_text.strip():
        _exit("No input provided. Exiting.")

    console.print("")
    with console.status("[cyan]Parsing your meeting description…[/cyan]", spinner="dots"):
        from agents.meeting_intake_agent import run as run_intake

        fields = run_intake(raw_text)

    console.print(Rule("[bold]What I understood[/bold]"))
    if fields.get("intake_confirmation_summary"):
        console.print(Panel(f"[dim]{fields['intake_confirmation_summary']}[/dim]", border_style="dim"))
    _show_extracted_table(fields)

    fields = _prompt_fill_missing(fields, fields.get("intake_missing_fields", []))
    fields = _prompt_confirm_low_confidence(fields, fields.get("intake_low_confidence_fields", []))
    fields = _prompt_edit_any(fields)

    still_missing = [key for key in _REQUIRED_KEYS if not fields.get(key)]
    if still_missing:
        _exit(f"\nCannot generate brief — still missing required fields: {', '.join(still_missing)}")

    console.print("")
    console.print(Rule("[bold]Generating brief[/bold]"))
    result = _run_workflow(fields)
    if not result:
        _exit("Workflow returned no result. Check logs for errors.")

    _show_engagement_inference(result.get("engagement_inference", {}))

    if result.get("errors"):
        console.print(Rule("[yellow]Warnings[/yellow]"))
        for error in result["errors"]:
            console.print(f"  [yellow]⚠  {error}[/yellow]")

    reliability = result.get("brief_reliability", 0)
    colour = "green" if reliability >= 4 else "yellow" if reliability >= 2 else "red"
    console.print("")
    console.print(Panel(f"[bold {colour}]{_reliability_line(result)}[/bold {colour}]", border_style=colour, expand=False))

    final_brief = result.get("final_brief", "")
    if not final_brief:
        _exit("No brief was produced. Check logs.")
    console.print("")
    console.print(Rule("[bold]Your Pre-Meeting Brief[/bold]"))
    console.print(Markdown(final_brief))

    saved_path = _save_brief(result.get("company_name") or fields.get("company_name", "company"), final_brief)
    console.print("")
    console.print(Panel(f"[green]Brief saved to:[/green] [bold]{saved_path}[/bold]", border_style="green", expand=False))


if __name__ == "__main__":
    main()
