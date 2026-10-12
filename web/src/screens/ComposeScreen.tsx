import { LoaderCircle, Play } from "lucide-react";
import { MarksLegend } from "../components/Marks";
import { PipelineGraph } from "../components/PipelineGraph";
import { Button, Notice } from "../components/ui";
import { formatDuration } from "../lib/format";
import type { AppConfig, DemoSummary } from "../lib/types";

const EXAMPLES = [
  {
    label: "Intro with a fintech CFO",
    text: "Intro call next week with the CFO of Revolut (revolut.com). I run a small consultancy that helps fintechs with regulatory reporting. I'd like to understand how they handle reporting across their banking licences and whether there's room for a scoping project.",
  },
  {
    label: "Pilot pitch to a data lead",
    text: "Follow-up demo with the Head of Data at Ferrari (ferrari.com). Our first call went well and they asked to see how our platform checks supplier data quality. I'm the account executive; the goal is to agree on a two-month pilot.",
  },
  {
    label: "Renewal with a client",
    text: "Quarterly review with our client Glovo (glovo.com), meeting their VP of Operations. We've run their courier-support chatbot for 18 months and the results are good. I want to renew the contract and propose extending it to Italy.",
  },
];

interface Props {
  config: AppConfig | null;
  demos: DemoSummary[];
  canRunLive: boolean;
  text: string;
  onTextChange: (text: string) => void;
  busy: boolean;
  error: string | null;
  onSubmit: () => void;
  onWatchDemo: (demo: DemoSummary) => void;
  onOpenKeys: () => void;
}

export function ComposeScreen({
  config,
  demos,
  canRunLive,
  text,
  onTextChange,
  busy,
  error,
  onSubmit,
  onWatchDemo,
  onOpenKeys,
}: Props) {
  return (
    <div className="mx-auto max-w-6xl px-4 md:px-8">
      <div className="grid gap-12 py-10 md:py-16 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)] lg:gap-16">
        <section aria-labelledby="hero-title">
          <h1
            id="hero-title"
            className="max-w-[16ch] text-[2.5rem] font-semibold leading-[1.04] tracking-[-0.025em] text-ink md:text-[3.5rem]"
          >
            Walk into your next client meeting already briefed.
          </h1>
          <p className="mt-5 max-w-[58ch] text-[17px] leading-relaxed text-ink-soft">
            Describe the meeting the way you would to a colleague. Eleven agents research the company, its recent
            news and the person you're seeing, then write a brief that keeps what's known apart from what's guessed.
          </p>
          <a href="#demos-title" className="mt-3 inline-block text-[15px] text-ballpoint underline-offset-4 hover:underline lg:hidden">
            Watch a recorded run first
          </a>

          <form
            className="mt-8"
            onSubmit={(e) => {
              e.preventDefault();
              if (text.trim() && !busy) onSubmit();
            }}
          >
            <label htmlFor="meeting-text" className="text-[15px] font-medium text-ink">
              Your meeting
            </label>
            <textarea
              id="meeting-text"
              rows={6}
              maxLength={4000}
              value={text}
              onChange={(e) => onTextChange(e.target.value)}
              placeholder="Who are you meeting, at which company, and what do you want out of it? Add anything you already know."
              className="mt-2 block w-full resize-y rounded-[3px] border border-rule bg-surface px-4 py-3 font-serif text-[17px] leading-relaxed text-ink placeholder:font-sans placeholder:text-[15px] placeholder:text-pencil focus:border-ballpoint focus:outline-none"
            />
            <div className="mt-3 flex flex-wrap items-center gap-2 text-[14px]">
                <span className="text-pencil">Try an example:</span>
                {EXAMPLES.map((example) => (
                  <button
                    key={example.label}
                    type="button"
                    onClick={() => onTextChange(example.text)}
                    className="rounded-full border border-rule px-3 py-1 text-ink-soft hover:border-ink hover:text-ink"
                  >
                    {example.label}
                  </button>
                ))}
            </div>

            <div className="mt-6 flex flex-wrap items-center gap-4">
              <Button type="submit" disabled={!text.trim() || busy || !canRunLive}>
                {busy && <LoaderCircle size={16} className="animate-spin" aria-hidden="true" />}
                {busy ? "Reading your notes" : "Check the details"}
              </Button>
              {!canRunLive && config && (
                <p className="text-[14px] text-ink-soft">
                  Live runs need API keys on this site.{" "}
                  <button type="button" onClick={onOpenKeys} className="text-ballpoint hover:underline underline-offset-4">
                    Add yours
                  </button>{" "}
                  or watch a recorded run.
                </p>
              )}
            </div>
            {error && (
              <div className="mt-4">
                <Notice tone="error">{error}</Notice>
              </div>
            )}
          </form>
        </section>

        <section aria-labelledby="demos-title" className="lg:pt-3">
          <h2 id="demos-title" className="text-[19px] font-semibold text-ink">
            Watch a recorded run
          </h2>
          <p className="mt-1 text-[15px] text-ink-soft">Real runs, replayed with their original timing. No keys needed.</p>
          <ul className="mt-5 divide-y divide-rule border-y border-rule">
            {demos.map((demo) => (
              <li key={demo.slug}>
                <button
                  type="button"
                  onClick={() => onWatchDemo(demo)}
                  className="group flex w-full items-center gap-4 py-4 text-left"
                >
                  <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-rule text-ink group-hover:border-ink group-hover:bg-ink group-hover:text-paper">
                    <Play size={14} aria-hidden="true" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block font-medium text-ink">{demo.title}</span>
                    <span className="block text-[14px] text-ink-soft">{demo.subtitle}</span>
                  </span>
                  <span className="text-[13px] tabular-nums text-pencil">{formatDuration(demo.duration_s)}</span>
                </button>
              </li>
            ))}
            {demos.length === 0 && <li className="py-4 text-[15px] text-pencil">No recorded runs on this server.</li>}
          </ul>
        </section>
      </div>

      {config && (
        <section aria-labelledby="how-title" className="border-t border-rule py-12">
          <h2 id="how-title" className="text-[24px] font-semibold tracking-[-0.01em] text-ink">
            How a brief gets built
          </h2>
          <p className="mt-2 max-w-[64ch] text-[15px] text-ink-soft">
            A LangGraph pipeline of eleven steps. Rules handle validation, planning and layout; language models are
            used only where judgement is needed, and every claim keeps its source.
          </p>
          <div className="mt-8">
            <PipelineGraph nodes={config.pipeline.nodes} explain />
          </div>
          <h3 className="mt-12 text-[17px] font-semibold text-ink">How to read a brief</h3>
          <div className="mt-3">
            <MarksLegend />
          </div>
        </section>
      )}
    </div>
  );
}
