"""Pipeline event stream edge cases."""

from __future__ import annotations

import asyncio
import time

from api.runner import HEARTBEAT, iter_pipeline_events, stream_pipeline


class SlowGraph:
    def stream(self, workflow_input, stream_mode):
        yield "tasks", {"id": "1", "name": "plan", "input": {}, "triggers": []}
        time.sleep(0.3)
        yield "updates", {"plan": {"research_plan": {"tavily_queries": ["q"]}}}
        yield "values", {"final_brief": "# B"}


class BrokenBeforeAnyNode:
    def stream(self, workflow_input, stream_mode):
        raise RuntimeError("boom")
        yield  # pragma: no cover


class TaskErrorGraph:
    def stream(self, workflow_input, stream_mode):
        yield "tasks", {"id": "1", "name": "plan", "input": {}, "triggers": []}
        yield "tasks", {"id": "1", "name": "plan", "result": None, "error": "ValueError('x')"}
        yield "values", {}


class InternalNodesGraph:
    def stream(self, workflow_input, stream_mode):
        yield "tasks", {"id": "0", "name": "__start__", "input": {}, "triggers": []}
        yield "updates", {"__start__": {}}
        yield "values", {}


def test_heartbeat_while_a_node_is_slow():
    async def collect():
        return [e async for e in stream_pipeline({}, graph=SlowGraph(), heartbeat_s=0.05)]

    events = asyncio.run(collect())

    assert HEARTBEAT in events
    real = [e["type"] for e in events if e is not HEARTBEAT]
    assert real == ["run_started", "node_started", "node_finished", "result"]


def test_error_before_any_node():
    events = list(iter_pipeline_events({}, graph=BrokenBeforeAnyNode()))

    assert [e["type"] for e in events] == ["run_started", "error"]
    assert events[-1]["message"] == "The pipeline stopped unexpectedly."


def test_task_error_chunk_marks_node_failed():
    events = list(iter_pipeline_events({}, graph=TaskErrorGraph()))

    assert [e["type"] for e in events] == ["run_started", "node_started", "node_failed", "result"]


def test_internal_langgraph_nodes_are_ignored():
    events = list(iter_pipeline_events({}, graph=InternalNodesGraph()))

    assert [e["type"] for e in events] == ["run_started", "result"]
