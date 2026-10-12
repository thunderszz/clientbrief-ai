import { ArrowLeft, FileText } from "lucide-react";
import { useEffect, useState } from "react";
import { EventLog } from "../components/EventLog";
import { PipelineGraph } from "../components/PipelineGraph";
import { Button, Notice } from "../components/ui";
import { formatDuration } from "../lib/format";
import type { RunState } from "../lib/run";
import type { PipelineNode } from "../lib/types";

interface Props {
  title: string;
  subtitle: string;
  isReplay: boolean;
  nodes: PipelineNode[];
  state: RunState;
  startedAt: number | null;
  onCancel: () => void;
  onOpenBrief: () => void;
}

function useWallClock(running: boolean, startedAt: number | null): number {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setNow(performance.now()), 100);
    return () => window.clearInterval(id);
  }, [running]);
  return startedAt === null ? 0 : (now - startedAt) / 1000;
}

export function RunScreen({ title, subtitle, isReplay, nodes, state, startedAt, onCancel, onOpenBrief }: Props) {
  const running = state.phase === "running" || state.phase === "idle";
  const wall = useWallClock(running, startedAt);
  const clock = isReplay ? state.elapsed : running ? wall : state.elapsed;
  const done = state.phase === "done" && state.result;

  // Give the finished graph a moment on screen, then open the brief.
  useEffect(() => {
    if (!done) return;
    const id = window.setTimeout(onOpenBrief, 1400);
    return () => window.clearTimeout(id);
  }, [done, onOpenBrief]);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 md:px-8 md:py-10">
      <button type="button" onClick={onCancel} className="flex items-center gap-1.5 text-[15px] text-ink-soft hover:text-ink">
        <ArrowLeft size={16} aria-hidden="true" /> {running ? "Stop and go back" : "Back"}
      </button>

      <div className="mt-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-[2rem] font-semibold tracking-[-0.02em] text-ink">{title}</h1>
          <p className="mt-1 text-[15px] text-ink-soft">{subtitle}</p>
        </div>
        <p className="text-right" aria-live="off">
          <span className="block text-[2rem] font-semibold tabular-nums leading-none text-ink">{formatDuration(clock)}</span>
          <span className="text-[13px] text-pencil">{isReplay ? "recorded time" : "elapsed"}</span>
        </p>
      </div>

      <div className="mt-8 grid gap-10 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <section aria-label="Pipeline progress" className="rounded-[3px] border border-rule bg-surface p-5 md:p-6">
          <PipelineGraph nodes={nodes} states={state.nodes} compact />
        </section>
        <section aria-labelledby="log-title">
          <h2 id="log-title" className="text-[15px] font-semibold text-ink">
            What the agents found
          </h2>
          <div className="mt-2">
            <EventLog entries={state.log} />
          </div>
        </section>
      </div>

      <div className="mt-8 min-h-12">
        {done && (
          <Button onClick={onOpenBrief}>
            <FileText size={16} aria-hidden="true" /> Open the brief
          </Button>
        )}
        {state.phase === "error" && (
          <Notice tone="error">
            <p>{state.error}</p>
            <button type="button" onClick={onCancel} className="mt-2 text-ballpoint hover:underline underline-offset-4">
              Go back
            </button>
          </Notice>
        )}
      </div>
    </div>
  );
}
