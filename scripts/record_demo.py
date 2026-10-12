"""Record a real pipeline run as a replayable demo.

Runs the intake agent on a meeting description, applies optional field
overrides, runs the full pipeline and saves every UI event to
``demo/<slug>.json``. The public site replays these files at zero API cost.

Usage::

    python -m scripts.record_demo --slug bending-spoons --order 1 \\
        --title "Bending Spoons · first intro" --subtitle "Head of Finance" \\
        --text "First intro call with the Head of Finance at Bending Spoons..."

    # fix an intake mistake or add a field:
    python -m scripts.record_demo ... --set contact_name="Jane Doe"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demo"


def build_record(
    *,
    slug: str,
    title: str,
    subtitle: str,
    company: str,
    input_text: str,
    intake: dict[str, Any],
    events: list[dict[str, Any]],
    models: dict[str, str],
    order: int,
) -> dict[str, Any]:
    """Assemble a demo file. Refuses runs that did not produce a result."""
    if not events or events[-1].get("type") != "result":
        raise ValueError("The run did not end with a result event; not saving a broken demo.")
    return {
        "slug": slug,
        "order": order,
        "title": title,
        "subtitle": subtitle,
        "company": company,
        "input_text": input_text,
        "intake": intake,
        "models": models,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "events": events,
    }


def assert_no_secrets(record: dict[str, Any], secrets: list[str]) -> None:
    """Raise if any secret value appears anywhere in the record."""
    dumped = json.dumps(record)
    for secret in secrets:
        if secret and secret in dumped:
            raise ValueError("Refusing to save: a secret value appears in the demo record.")


def _parse_overrides(pairs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        if not key or not _:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        overrides[key.strip()] = value.strip()
    return overrides


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--text", required=True, help="Free-text meeting description")
    parser.add_argument("--order", type=int, default=100)
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    from agents.meeting_intake_agent import run as run_intake
    from api.runner import iter_pipeline_events
    from api.schemas import MeetingFields
    from services.llm_client import settings_from_env

    settings = settings_from_env()
    print(f"Models: fast={settings.model_fast} reasoning={settings.model_reasoning}")

    print("Running intake…")
    intake = run_intake(args.text)
    fields = {key: intake.get(key) for key in MeetingFields.model_fields}
    fields.update(_parse_overrides(args.set))
    workflow_input = MeetingFields(**fields).to_workflow_input()
    workflow_input["raw_meeting_context"] = args.text

    print("Running pipeline…")
    events: list[dict[str, Any]] = []
    for event in iter_pipeline_events(workflow_input):
        events.append(event)
        if event["type"] == "node_finished":
            print(f"  {event['t']:6.1f}s  {event['label']:<22} {event['summary']}")
        elif event["type"] == "error":
            print(f"  ERROR: {event['message']}")

    record = build_record(
        slug=args.slug,
        title=args.title,
        subtitle=args.subtitle,
        company=workflow_input["company_name"],
        input_text=args.text,
        intake=intake,
        events=events,
        models={"fast": settings.model_fast, "reasoning": settings.model_reasoning},
        order=args.order,
    )
    assert_no_secrets(
        record,
        [settings.api_key, os.environ.get("TAVILY_API_KEY", ""), os.environ.get("LLM_API_KEY", "")],
    )

    DEMO_DIR.mkdir(exist_ok=True)
    path = DEMO_DIR / f"{args.slug}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    score = events[-1]["result"]["reliability"]["score"]
    print(f"Saved {path.relative_to(PROJECT_ROOT)} · {events[-1]['t']:.0f}s · reliability {score}/100")
    return 0


if __name__ == "__main__":
    sys.exit(main())
