import { describe, expect, it } from "vitest";
import { initialRunState, runReducer } from "./run";
import type { PipelineEvent } from "./types";

const NODES = ["orchestrate", "research_company", "compose"];

function play(events: PipelineEvent[]) {
  return events.reduce(runReducer, initialRunState(NODES));
}

describe("runReducer", () => {
  it("starts with every node idle", () => {
    const state = initialRunState(NODES);

    expect(state.phase).toBe("idle");
    expect(Object.values(state.nodes).map((n) => n.status)).toEqual(["idle", "idle", "idle"]);
  });

  it("tracks node lifecycle and builds the log", () => {
    const state = play([
      { type: "run_started", t: 0 },
      { type: "node_started", t: 0.1, node: "research_company", label: "Company research" },
      {
        type: "node_finished",
        t: 7.2,
        node: "research_company",
        label: "Company research",
        duration: 7.1,
        summary: "25 sources",
        llm_calls: [{ model: "gemini-3.5-flash-lite", tier: "fast", ok: true, latency_s: 1.5, fallback: false, attempts: 1, node: "research_company" }],
        warnings: [],
      },
    ]);

    expect(state.phase).toBe("running");
    expect(state.nodes.research_company.status).toBe("done");
    expect(state.nodes.research_company.duration).toBe(7.1);
    expect(state.nodes.research_company.calls[0].model).toBe("gemini-3.5-flash-lite");
    expect(state.log).toHaveLength(1);
    expect(state.log[0].summary).toBe("25 sources");
    expect(state.elapsed).toBe(7.2);
  });

  it("marks a node running between start and finish", () => {
    const state = play([
      { type: "run_started", t: 0 },
      { type: "node_started", t: 0.1, node: "orchestrate", label: "Validate inputs" },
    ]);

    expect(state.nodes.orchestrate.status).toBe("running");
  });

  it("records failures and terminal errors", () => {
    const state = play([
      { type: "run_started", t: 0 },
      { type: "node_started", t: 0.1, node: "research_company", label: "Company research" },
      { type: "node_failed", t: 3, node: "research_company", label: "Company research" },
      { type: "error", t: 3, message: "The pipeline stopped in Company research." },
    ]);

    expect(state.nodes.research_company.status).toBe("failed");
    expect(state.phase).toBe("error");
    expect(state.error).toBe("The pipeline stopped in Company research.");
    expect(state.log.at(-1)?.failed).toBe(true);
  });

  it("stores the result and finishes", () => {
    const state = play([
      { type: "run_started", t: 0 },
      { type: "result", t: 9, result: { final_brief: "# B" } as never },
    ]);

    expect(state.phase).toBe("done");
    expect(state.result?.final_brief).toBe("# B");
  });

  it("ignores events for unknown nodes", () => {
    const state = play([{ type: "node_started", t: 0, node: "mystery", label: "?" }]);

    expect(Object.keys(state.nodes)).toEqual(NODES);
  });
});
