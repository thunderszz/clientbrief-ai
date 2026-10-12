import type { LogEntry } from "../lib/run";
import { ModelChip } from "./ui";

export function EventLog({ entries }: { entries: LogEntry[] }) {
  if (entries.length === 0) {
    return <p className="text-[14px] text-pencil">Agent results will appear here as each one finishes.</p>;
  }
  return (
    <ol aria-live="polite" className="divide-y divide-rule">
      {entries.map((entry) => {
        const calls = [...new Map(entry.calls.map((c) => [c.model + c.fallback, c])).values()];
        return (
          <li key={`${entry.node}-${entry.t}`} className="grid grid-cols-[4.5rem_1fr] gap-3 py-2.5 text-[14px]">
            <span className="tabular-nums text-pencil">{entry.t.toFixed(1)} s</span>
            <div className="min-w-0">
              <p>
                <span className={entry.failed ? "font-medium text-alert" : "font-medium text-ink"}>{entry.label}</span>{" "}
                <span className="text-ink-soft">{entry.summary}</span>
              </p>
              {calls.length > 0 && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {calls.map((c) => (
                    <ModelChip key={c.model + c.fallback} model={c.model} fallback={c.fallback} />
                  ))}
                </div>
              )}
            </div>
          </li>
        );
      })}
    </ol>
  );
}
