import { Check, ChevronRight, X } from "lucide-react";
import { formatDuration } from "../lib/format";
import type { NodeState } from "../lib/run";
import type { PipelineNode } from "../lib/types";
import { ModelChip } from "./ui";

const PHASES: { id: PipelineNode["phase"]; title: string; parallel?: boolean }[] = [
  { id: "setup", title: "Set up" },
  { id: "research", title: "Research, in parallel", parallel: true },
  { id: "reasoning", title: "Reason" },
  { id: "output", title: "Write" },
];

const KIND_LABEL: Record<PipelineNode["kind"], string> = {
  deterministic: "Rules, no LLM",
  llm: "LLM",
  "search+llm": "Web search + LLM",
};

function StatusMarker({ status }: { status: NodeState["status"] }) {
  const base = "relative z-10 flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-full";
  switch (status) {
    case "running":
      return <span className={`${base} node-running border-2 border-ballpoint bg-surface`} />;
    case "done":
      return (
        <span className={`${base} bg-ink text-paper`}>
          <Check size={12} strokeWidth={3} aria-hidden="true" />
        </span>
      );
    case "failed":
      return (
        <span className={`${base} bg-alert text-paper`}>
          <X size={12} strokeWidth={3} aria-hidden="true" />
        </span>
      );
    default:
      return <span className={`${base} border-2 border-rule bg-paper`} />;
  }
}

const STATUS_WORD: Record<NodeState["status"], string> = {
  idle: "waiting",
  running: "running",
  done: "done",
  failed: "failed",
};

function NodeRow({
  node,
  state,
  explain,
  compact,
}: {
  node: PipelineNode;
  state?: NodeState;
  explain: boolean;
  compact: boolean;
}) {
  const status = state?.status ?? "idle";
  const models = [...new Map((state?.calls ?? []).map((c) => [c.model + c.fallback, c])).values()];
  return (
    <li className="relative flex gap-3 pb-5 last:pb-0">
      <StatusMarker status={status} />
      <div className="min-w-0 -mt-0.5">
        <p className="flex flex-wrap items-baseline gap-x-2 text-[15px] leading-snug">
          <span className={status === "idle" && !explain ? "text-ink-soft" : "font-medium text-ink"}>{node.label}</span>
          <span className="sr-only">{STATUS_WORD[status]}</span>
          {state?.duration !== undefined && status === "done" && (
            <span className="text-[13px] tabular-nums text-pencil">{formatDuration(state.duration)}</span>
          )}
        </p>
        {explain ? (
          <p className="mt-0.5 text-[13px] leading-snug text-ink-soft">
            {node.description} <span className="text-pencil">{KIND_LABEL[node.kind]}.</span>
          </p>
        ) : (
          !compact && state?.summary && <p className="mt-0.5 text-[13px] leading-snug text-ink-soft">{state.summary}</p>
        )}
        {models.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {models.map((c) => (
              <ModelChip key={c.model + c.fallback} model={c.model} fallback={c.fallback} />
            ))}
          </div>
        )}
      </div>
    </li>
  );
}

interface Props {
  nodes: PipelineNode[];
  states?: Record<string, NodeState>;
  /** Show what each node does instead of live results (the "how it works" view). */
  explain?: boolean;
  /** Hide per-node summaries (the run screen shows them in the log). */
  compact?: boolean;
}

export function PipelineGraph({ nodes, states, explain = false, compact = false }: Props) {
  return (
    <div className="@container">
    <div className="grid gap-x-4 gap-y-8 @lg:grid-cols-2 @3xl:grid-cols-[1fr_auto_1fr_auto_1fr_auto_0.8fr]">
      {PHASES.map((phase, index) => {
        const phaseNodes = nodes.filter((n) => n.phase === phase.id);
        return [
          <section key={phase.id} aria-label={phase.title}>
            <h3 className="mb-3 text-[13px] font-medium text-pencil">{phase.title}</h3>
            <ol
              className={`relative ${
                phase.parallel
                  ? "border-l-2 border-dashed border-rule pl-3"
                  : phaseNodes.length > 1
                    ? "before:absolute before:left-[8px] before:top-2 before:bottom-6 before:w-0.5 before:bg-rule"
                    : ""
              }`}
            >
              {phaseNodes.map((node) => (
                <NodeRow key={node.id} node={node} state={states?.[node.id]} explain={explain} compact={compact} />
              ))}
            </ol>
          </section>,
          index < PHASES.length - 1 && (
            <ChevronRight
              key={`${phase.id}-arrow`}
              size={18}
              aria-hidden="true"
              className="mt-8 hidden text-pencil @3xl:block"
            />
          ),
        ];
      })}
    </div>
    </div>
  );
}
