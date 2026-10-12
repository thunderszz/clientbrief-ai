"""Turn a LangGraph run into a stream of UI events.

Event types (every event has ``type`` and ``t`` = seconds since start):

- ``run_started``
- ``node_started``  — ``node``, ``label``
- ``node_finished`` — ``node``, ``label``, ``duration``, ``summary``, ``llm_calls``, ``warnings``
- ``node_failed``   — ``node``, ``label``
- ``result``        — ``result`` (see ``core/result.py``)
- ``error``         — ``message`` (generic; details go to the server log only)

The same events are recorded to ``demo/*.json`` and replayed, so live runs and
replays render through the same UI code.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from core.pipeline import NODE_LABELS
from core.result import build_result
from core.summaries import summarize_update
from services.llm_client import LLMSettings, use_llm_settings
from services.tavily_client import use_tavily_key

logger = logging.getLogger(__name__)

HEARTBEAT = {"type": "heartbeat"}


def _default_graph():
    from core.workflow import app

    return app


def iter_pipeline_events(workflow_input: dict[str, Any], graph: Any = None) -> Iterator[dict[str, Any]]:
    """Run the pipeline synchronously, yielding UI events as nodes start and finish."""
    graph = graph or _default_graph()
    t0 = time.monotonic()

    def now() -> float:
        return round(time.monotonic() - t0, 2)

    started: dict[str, float] = {}
    finished: set[str] = set()
    final_state: dict[str, Any] = {}

    yield {"type": "run_started", "t": 0.0}
    try:
        for mode, chunk in graph.stream(workflow_input, stream_mode=["tasks", "updates", "values"]):
            if mode == "values":
                final_state = chunk
            elif mode == "tasks":
                node = chunk.get("name")
                if node not in NODE_LABELS:
                    continue
                if "result" not in chunk and "error" not in chunk:
                    started[node] = now()
                    yield {"type": "node_started", "t": started[node], "node": node, "label": NODE_LABELS[node]}
                elif chunk.get("error"):
                    finished.add(node)
                    yield {"type": "node_failed", "t": now(), "node": node, "label": NODE_LABELS[node]}
            elif mode == "updates":
                for node, update in chunk.items():
                    if node not in NODE_LABELS:
                        continue
                    update = update or {}
                    t = now()
                    finished.add(node)
                    yield {
                        "type": "node_finished",
                        "t": t,
                        "node": node,
                        "label": NODE_LABELS[node],
                        "duration": round(t - started.get(node, t), 2),
                        "summary": summarize_update(node, update),
                        "llm_calls": update.get("llm_calls", []),
                        "warnings": list(update.get("errors") or []),
                    }
    except Exception:
        logger.exception("Pipeline run failed")
        running = [node for node in started if node not in finished]
        for node in running:
            yield {"type": "node_failed", "t": now(), "node": node, "label": NODE_LABELS[node]}
        where = ", ".join(NODE_LABELS[node] for node in running)
        message = f"The pipeline stopped in {where}." if where else "The pipeline stopped unexpectedly."
        yield {"type": "error", "t": now(), "message": message}
        return

    yield {"type": "result", "t": now(), "result": build_result(final_state)}


async def stream_pipeline(
    workflow_input: dict[str, Any],
    llm: LLMSettings | None = None,
    tavily_key: str | None = None,
    graph: Any = None,
    heartbeat_s: float = 15.0,
) -> AsyncIterator[dict[str, Any]]:
    """Run the pipeline in a worker thread and yield its events as they happen.

    Emits ``HEARTBEAT`` when nothing happened for *heartbeat_s* so proxies keep
    the connection open during long LLM calls.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    def worker() -> None:
        try:
            with use_llm_settings(llm), use_tavily_key(tavily_key):
                for event in iter_pipeline_events(workflow_input, graph):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    # If the client disconnects, the worker still finishes the run in the background:
    # a sync LangGraph run cannot be cancelled midway.
    context = contextvars.copy_context()
    loop.run_in_executor(None, context.run, worker)
    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=heartbeat_s)
        except TimeoutError:
            yield HEARTBEAT
            continue
        if event is None:
            break
        yield event
