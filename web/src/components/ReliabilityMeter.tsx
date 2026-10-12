import type { ReliabilityReport } from "../lib/types";

export function ReliabilityMeter({ report }: { report: ReliabilityReport }) {
  const applicable = report.components.filter((c) => c.applicable);
  const total = applicable.reduce((sum, c) => sum + c.max, 0) || 1;
  return (
    <div>
      <p className="flex items-baseline gap-1.5">
        <span className="text-[2.75rem] font-semibold leading-none tracking-[-0.03em] tabular-nums">{report.score}</span>
        <span className="text-[15px] text-pencil">/100 reliability</span>
      </p>
      <div className="mt-3 flex h-2 gap-0.5" aria-hidden="true">
        {applicable.map((c) => (
          <span key={c.key} className="relative overflow-hidden bg-rule" style={{ width: `${(c.max / total) * 100}%` }}>
            <span className="absolute inset-y-0 left-0 bg-ink" style={{ width: `${(c.points / c.max) * 100}%` }} />
          </span>
        ))}
      </div>
      <details className="group mt-3">
        <summary className="cursor-pointer text-[14px] text-ballpoint hover:underline underline-offset-4">
          How this score is built
        </summary>
        <dl className="mt-3 space-y-2.5 text-[13px]">
          {report.components.map((c) => (
            <div key={c.key}>
              <dt className="flex justify-between gap-2 font-medium text-ink">
                <span>{c.label}</span>
                <span className="tabular-nums text-ink-soft">
                  {c.applicable ? `${Math.round(c.points * 10) / 10} / ${c.max}` : "not counted"}
                </span>
              </dt>
              <dd className="text-ink-soft">{c.detail}</dd>
            </div>
          ))}
        </dl>
      </details>
    </div>
  );
}
