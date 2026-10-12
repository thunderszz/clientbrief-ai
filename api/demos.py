"""Recorded pipeline runs, replayed at zero API cost.

A demo file (``demo/<slug>.json``) holds the meeting text, the intake result and
the full event stream of a real run, recorded by ``scripts/record_demo.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,60}$")
_SUMMARY_KEYS = ("slug", "title", "subtitle", "company", "recorded_at", "models")


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Skipping unreadable demo file %s", path.name)
        return None
    if not isinstance(data, dict) or not data.get("events") or not data.get("slug"):
        logger.warning("Skipping malformed demo file %s", path.name)
        return None
    return data


def load_demo(demo_dir: Path, slug: str) -> dict[str, Any] | None:
    """Load one demo by slug; ``None`` for unknown or unsafe slugs."""
    if not _SLUG.match(slug):
        return None
    path = demo_dir / f"{slug}.json"
    return _read(path) if path.is_file() else None


def list_demos(demo_dir: Path) -> list[dict[str, Any]]:
    """Summaries of every valid demo, ordered by their ``order`` field, then slug."""
    demos = [d for d in (_read(p) for p in sorted(demo_dir.glob("*.json"))) if d]
    demos.sort(key=lambda d: (d.get("order", 100), d["slug"]))
    return [
        {**{k: d.get(k) for k in _SUMMARY_KEYS}, "duration_s": d["events"][-1].get("t", 0)}
        for d in demos
    ]


def demo_detail(demo: dict[str, Any]) -> dict[str, Any]:
    """Everything about a demo except the event stream."""
    return {k: v for k, v in demo.items() if k != "events"}


def demo_result(demo: dict[str, Any]) -> dict[str, Any] | None:
    for event in reversed(demo["events"]):
        if event.get("type") == "result":
            return event.get("result")
    return None


async def replay(
    events: list[dict[str, Any]],
    target_seconds: float = 24.0,
    speed: float = 1.0,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> AsyncIterator[dict[str, Any]]:
    """Yield recorded events with their original pacing, compressed to ~*target_seconds*."""
    total = events[-1].get("t", 0.0) if events else 0.0
    scale = max(1.0, total / target_seconds) * speed
    previous = 0.0
    for event in events:
        t = event.get("t", previous)
        delay = (t - previous) / scale
        if delay > 0:
            await sleep(delay)
        previous = t
        yield event
