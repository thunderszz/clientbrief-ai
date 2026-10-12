// Pure state machine for a pipeline run, driven by API events (live or replayed).

import type { BriefResult, LlmCall, PipelineEvent } from "./types";

export type NodeStatus = "idle" | "running" | "done" | "failed";

export interface NodeState {
  status: NodeStatus;
  duration?: number;
  summary?: string;
  calls: LlmCall[];
  warnings: string[];
}

export interface LogEntry {
  t: number;
  node: string;
  label: string;
  summary: string;
  duration?: number;
  calls: LlmCall[];
  failed: boolean;
}

export interface RunState {
  phase: "idle" | "running" | "done" | "error";
  nodes: Record<string, NodeState>;
  log: LogEntry[];
  elapsed: number;
  result?: BriefResult;
  error?: string;
}

export function initialRunState(nodeIds: string[]): RunState {
  return {
    phase: "idle",
    nodes: Object.fromEntries(nodeIds.map((id) => [id, { status: "idle", calls: [], warnings: [] }])),
    log: [],
    elapsed: 0,
  };
}

function patchNode(state: RunState, id: string, patch: Partial<NodeState>): RunState["nodes"] {
  return { ...state.nodes, [id]: { ...state.nodes[id], ...patch } };
}

export function runReducer(state: RunState, event: PipelineEvent): RunState {
  const elapsed = Math.max(state.elapsed, event.t ?? 0);
  switch (event.type) {
    case "run_started":
      return { ...state, phase: "running", elapsed };
    case "node_started":
      if (!(event.node in state.nodes)) return state;
      return { ...state, elapsed, nodes: patchNode(state, event.node, { status: "running" }) };
    case "node_finished":
      if (!(event.node in state.nodes)) return state;
      return {
        ...state,
        elapsed,
        nodes: patchNode(state, event.node, {
          status: "done",
          duration: event.duration,
          summary: event.summary,
          calls: event.llm_calls,
          warnings: event.warnings,
        }),
        log: [
          ...state.log,
          {
            t: event.t,
            node: event.node,
            label: event.label,
            summary: event.summary,
            duration: event.duration,
            calls: event.llm_calls,
            failed: false,
          },
        ],
      };
    case "node_failed":
      if (!(event.node in state.nodes)) return state;
      return {
        ...state,
        elapsed,
        nodes: patchNode(state, event.node, { status: "failed" }),
        log: [
          ...state.log,
          { t: event.t, node: event.node, label: event.label, summary: "failed", calls: [], failed: true },
        ],
      };
    case "result":
      return { ...state, elapsed, phase: "done", result: event.result };
    case "error":
      return { ...state, elapsed, phase: "error", error: event.message };
    default:
      return state;
  }
}
