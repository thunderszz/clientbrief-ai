import { ArrowLeft, Check, Copy, Download, Printer } from "lucide-react";
import { useState, type ReactNode } from "react";
import { MarksLegend } from "../components/Marks";
import { ReliabilityMeter } from "../components/ReliabilityMeter";
import { Button, ModelChip } from "../components/ui";
import { formatSignalDate, groupSourcesBySite, labelFor, safeUrl, siteOf, splitObjection } from "../lib/format";
import type { BriefResult } from "../lib/types";

const SECTIONS = [
  { id: "summary", title: "Summary" },
  { id: "signals", title: "Recent signals" },
  { id: "pain-points", title: "Known, inferred, suspected" },
  { id: "plan", title: "Meeting plan" },
  { id: "company", title: "The company" },
  { id: "person", title: "The person" },
  { id: "audience", title: "The audience" },
  { id: "stage", title: "Where things stand" },
  { id: "method", title: "How this brief was made" },
  { id: "sources", title: "Sources" },
];

function Section({ id, children }: { id: string; children: ReactNode }) {
  const title = SECTIONS.find((s) => s.id === id)?.title;
  return (
    <section id={id} aria-labelledby={`${id}-title`} className="scroll-mt-6 border-t border-rule pt-8">
      <h2 id={`${id}-title`} className="font-sans text-[21px] font-semibold tracking-[-0.01em] text-ink">
        {title}
      </h2>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function List({ items, className = "" }: { items?: string[]; className?: string }) {
  if (!items?.length) return <p className="text-pencil">Nothing found.</p>;
  return (
    <ul className={`list-disc space-y-1.5 pl-5 marker:text-pencil ${className}`}>
      {items.map((item, i) => (
        <li key={`${i}-${item}`}>{item}</li>
      ))}
    </ul>
  );
}

function Empty({ children }: { children: ReactNode }) {
  return <p className="text-pencil">{children}</p>;
}

function Fact({ term, children }: { term: string; children: ReactNode }) {
  if (!children || (Array.isArray(children) && children.length === 0)) return null;
  return (
    <div className="grid gap-1 py-2 sm:grid-cols-[10rem_1fr] sm:gap-4">
      <dt className="font-sans text-[14px] font-medium text-ink-soft">{term}</dt>
      <dd>{children}</dd>
    </div>
  );
}

interface Props {
  result: BriefResult;
  origin: string;
  nodeLabels: Record<string, string>;
  onNew: () => void;
}

export function BriefScreen({ result, origin, nodeLabels, onNew }: Props) {
  const [copied, setCopied] = useState(false);
  const input = result.input;
  const strategy = result.meeting_strategy;
  const pains = result.pain_point_hypotheses;
  const company = result.company_profile;
  const person = result.person_profile;
  const audience = result.audience_context;
  const engagement = result.engagement_context;
  const reliablePerson = person.match_confidence === "high" || person.match_confidence === "medium";
  const fileName = `${(input.company_name || "brief").replace(/[^\w-]+/g, "_")}_brief.md`;
  const modelsUsed = [...new Map(result.llm_calls.map((c) => [`${c.node}-${c.model}-${c.fallback}`, c])).values()];

  async function copyMarkdown() {
    try {
      await navigator.clipboard.writeText(result.final_brief);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }

  function download() {
    const url = URL.createObjectURL(new Blob([result.final_brief], { type: "text/markdown" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: fileName });
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 md:px-8 md:py-10">
      <div className="no-print flex flex-wrap items-center justify-between gap-3">
        <button type="button" onClick={onNew} className="flex items-center gap-1.5 text-[15px] text-ink-soft hover:text-ink">
          <ArrowLeft size={16} aria-hidden="true" /> New brief
        </button>
        <div className="flex flex-wrap gap-2">
          <Button variant="secondary" onClick={copyMarkdown}>
            {copied ? <Check size={16} aria-hidden="true" /> : <Copy size={16} aria-hidden="true" />}
            {copied ? "Copied" : "Copy as Markdown"}
          </Button>
          <Button variant="secondary" onClick={download}>
            <Download size={16} aria-hidden="true" /> Download .md
          </Button>
          <Button variant="secondary" onClick={() => window.print()}>
            <Printer size={16} aria-hidden="true" /> Print
          </Button>
        </div>
      </div>

      <div className="mt-8 grid gap-10 lg:grid-cols-[15rem_minmax(0,1fr)] lg:gap-14">
        <aside className="lg:sticky lg:top-6 lg:self-start">
          <ReliabilityMeter report={result.reliability} />
          <nav aria-label="Brief contents" className="no-print mt-8 hidden lg:block">
            <ol className="space-y-1.5 text-[14px]">
              {SECTIONS.map((s) => (
                <li key={s.id}>
                  <a href={`#${s.id}`} className="text-ink-soft hover:text-ink">
                    {s.title}
                  </a>
                </li>
              ))}
            </ol>
          </nav>
          <div className="mt-8 hidden border-t border-rule pt-5 lg:block">
            <MarksLegend compact />
          </div>
        </aside>

        <article className="min-w-0 max-w-[70ch] font-serif text-[17px] leading-[1.7] text-ink">
          <header className="font-sans">
            <p className="text-[14px] text-pencil">{origin}</p>
            <h1 className="mt-1 text-[2.5rem] font-semibold leading-[1.05] tracking-[-0.025em] md:text-[3rem]">
              {input.company_name}
            </h1>
            <p className="mt-3 text-[16px] text-ink-soft">
              {labelFor(input.meeting_type)} with {input.contact_name ? `${input.contact_name}, ` : ""}
              {input.contact_title || input.stakeholder_role}
              {engagement.interaction_stage ? `. Stage: ${labelFor(engagement.interaction_stage).toLowerCase()}.` : "."}
            </p>
            <p className="mt-1 text-[16px] text-ink-soft">Goal: {input.meeting_goal}</p>
            <div className="mt-6 lg:hidden">
              <MarksLegend compact />
            </div>
          </header>

          <div className="mt-10 space-y-12">
            <Section id="summary">
              <p className="text-[19px] leading-[1.65]">{strategy.executive_summary || "No summary was produced."}</p>
            </Section>

            <Section id="signals">
              {result.external_signals.length === 0 ? (
                <p className="text-pencil">No recent strategic signals were found. Ask about recent changes in the meeting.</p>
              ) : (
                <ol className="space-y-6">
                  {result.external_signals.map((signal, i) => (
                    <li key={`${i}-${signal.description}`} className="grid gap-1 sm:grid-cols-[6.5rem_1fr] sm:gap-5">
                      <p className="font-sans text-[14px] tabular-nums text-pencil">{formatSignalDate(signal.date)}</p>
                      <div>
                        <p className="font-sans text-[13px] font-medium text-ink-soft">{labelFor(signal.signal_type)}</p>
                        <p className="mt-0.5">
                          <span className="mark-fact">{signal.description}</span>
                        </p>
                        <p className="mt-1.5">
                          <span className="mark-inference">{signal.implication}</span>
                        </p>
                        {safeUrl(signal.source) && (
                          <a
                            href={safeUrl(signal.source)!}
                            target="_blank"
                            rel="noreferrer"
                            className="mt-1 inline-block font-sans text-[13px] text-ballpoint hover:underline"
                          >
                            {siteOf(signal.source)}
                          </a>
                        )}
                      </div>
                    </li>
                  ))}
                </ol>
              )}
            </Section>

            <Section id="pain-points">
              <h3 className="font-sans text-[15px] font-semibold">What the sources say</h3>
              {pains.facts?.length ? (
                <ul className="mt-2 space-y-2">
                  {pains.facts.map((fact, i) => (
                    <li key={`${i}-${fact}`}>
                      <span className="mark-fact">{fact}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <Empty>No facts could be confirmed from the sources.</Empty>
              )}
              <h3 className="mt-7 font-sans text-[15px] font-semibold">What follows from them</h3>
              {pains.inferences?.length ? (
                <ul className="mt-2 space-y-2">
                  {pains.inferences.map((inference, i) => (
                    <li key={`${i}-${inference}`}>
                      <span className="mark-inference">{inference.replace(/^\[INFERENCE\]\s*/i, "")}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <Empty>No inferences were drawn.</Empty>
              )}
              <h3 className="mt-7 font-sans text-[15px] font-semibold">Hypotheses to test in the meeting</h3>
              {!pains.hypotheses?.length && <Empty>No hypotheses were proposed.</Empty>}
              <ol className="mt-3 space-y-5">
                {(pains.hypotheses ?? []).map((h, i) => (
                  <li key={`${i}-${h.hypothesis}`} className="border-l-2 border-rule pl-4">
                    <p className="font-sans text-[13px] text-pencil">{labelFor(h.confidence)} confidence</p>
                    <p className="mark-hypothesis">{h.hypothesis}</p>
                    {h.supporting_evidence.length > 0 && (
                      <ul className="mt-2 list-disc space-y-1 pl-5 font-sans text-[14px] text-ink-soft marker:text-pencil">
                        {h.supporting_evidence.map((e, j) => (
                          <li key={`${j}-${e}`}>{e}</li>
                        ))}
                      </ul>
                    )}
                  </li>
                ))}
              </ol>
            </Section>

            <Section id="plan">
              <h3 className="font-sans text-[15px] font-semibold">How to open</h3>
              {strategy.opening_angle ? (
                <p className="mt-2 border-l-2 border-ballpoint pl-4">{strategy.opening_angle}</p>
              ) : (
                <Empty>No opening was suggested.</Empty>
              )}

              <h3 className="mt-7 font-sans text-[15px] font-semibold">Questions to ask</h3>
              {strategy.recommended_questions?.length ? (
                <ol className="mt-2 list-decimal space-y-2 pl-6 marker:font-sans marker:text-[14px] marker:text-pencil">
                  {strategy.recommended_questions.map((q, i) => (
                    <li key={`${i}-${q}`}>{q}</li>
                  ))}
                </ol>
              ) : (
                <Empty>No suggested questions.</Empty>
              )}

              <h3 className="mt-7 font-sans text-[15px] font-semibold">Points to make</h3>
              <List items={strategy.talking_points} className="mt-2" />

              <h3 className="mt-7 font-sans text-[15px] font-semibold">Likely objections</h3>
              {!strategy.possible_objections?.length && <Empty>No objections anticipated.</Empty>}
              <div className="mt-2 divide-y divide-rule border-y border-rule empty:hidden">
                {(strategy.possible_objections ?? []).map((raw, i) => {
                  const { objection, response } = splitObjection(raw);
                  return (
                    <details key={`${i}-${raw}`} className="group py-3">
                      <summary className="cursor-pointer list-none font-sans text-[15px] font-medium text-ink marker:hidden">
                        <span className="mr-2 inline-block text-pencil transition-transform group-open:rotate-90">›</span>
                        {objection}
                      </summary>
                      {response && <p className="mt-2 pl-5 text-ink-soft">{response}</p>}
                    </details>
                  );
                })}
              </div>

              <h3 className="mt-7 font-sans text-[15px] font-semibold">Next step to propose</h3>
              {strategy.suggested_next_step ? (
                <p className="mt-2">{strategy.suggested_next_step}</p>
              ) : (
                <Empty>No next step was suggested.</Empty>
              )}
            </Section>

            <Section id="company">
              <dl className="divide-y divide-rule">
                <Fact term="Industry">{company.industry}</Fact>
                <Fact term="Business model">{company.business_model}</Fact>
                <Fact term="Positioning">{company.company_positioning}</Fact>
                <Fact term="Products">{company.products?.join(", ")}</Fact>
                <Fact term="Customers">{company.customer_segments?.join(", ")}</Fact>
                <Fact term="Markets">{company.geographies?.join(", ")}</Fact>
                <Fact term="Complexity">{company.complexity_signals?.length ? <List items={company.complexity_signals} /> : null}</Fact>
              </dl>
            </Section>

            <Section id="person">
              {!input.contact_name ? (
                <p className="text-ink-soft">
                  No contact was named, so the brief is written for the role: {input.stakeholder_role}.
                </p>
              ) : reliablePerson ? (
                <>
                  <p className="font-sans text-[15px]">
                    <span className="font-semibold">{person.name}</span>
                    {person.title ? `, ${person.title}` : ""}
                    <span className="text-pencil"> ({person.match_confidence} confidence match)</span>
                  </p>
                  {person.background_summary && <p className="mt-2">{person.background_summary}</p>}
                  {person.public_signals && person.public_signals.length > 0 && (
                    <List items={person.public_signals} className="mt-3" />
                  )}
                </>
              ) : (
                <p className="text-ink-soft">
                  No reliable public profile was found for {input.contact_name}. Nothing about them is assumed in this
                  brief.
                </p>
              )}
            </Section>

            <Section id="audience">
              <dl className="divide-y divide-rule">
                <Fact term="Department">{audience.department}</Fact>
                <Fact term="Likely priorities">{audience.likely_priorities?.length ? <List items={audience.likely_priorities} /> : null}</Fact>
                <Fact term="Topics that land">{audience.relevant_topics?.length ? <List items={audience.relevant_topics} /> : null}</Fact>
                <Fact term="Handle with care">
                  {audience.potential_sensitivities?.length ? <List items={audience.potential_sensitivities} /> : null}
                </Fact>
                <Fact term="Tone">{audience.communication_style}</Fact>
              </dl>
            </Section>

            <Section id="stage">
              <dl className="divide-y divide-rule">
                <Fact term="Stage">{labelFor(engagement.interaction_stage)}</Fact>
                <Fact term="Relationship">{labelFor(engagement.relationship_status)}</Fact>
                <Fact term="Aim of this stage">{engagement.stage_objective}</Fact>
                <Fact term="Recommended tone">{engagement.recommended_tone}</Fact>
                <Fact term="Risks">{engagement.decision_risks?.length ? <List items={engagement.decision_risks} /> : null}</Fact>
                <Fact term="History">{engagement.continuity_context}</Fact>
              </dl>
              {result.engagement_inference.reasoning_summary &&
                result.engagement_inference.confidence !== "not_needed" && (
                  <p className="mt-4 font-sans text-[14px] text-ink-soft">
                    Some of this was inferred ({result.engagement_inference.confidence} confidence):{" "}
                    {result.engagement_inference.reasoning_summary}
                  </p>
                )}
            </Section>

            <Section id="method">
              <div className="font-sans text-[15px]">
                <p className="text-ink-soft">
                  Research mode: {engagement.research_mode ?? "full"}. Generated {new Date(result.generated_at).toLocaleString("en-GB", { dateStyle: "long", timeStyle: "short" })}.
                </p>
                {modelsUsed.length > 0 && (
                  <ul className="mt-3 space-y-1.5">
                    {modelsUsed.map((c) => (
                      <li key={`${c.node}-${c.model}-${c.fallback}`} className="flex flex-wrap items-center gap-2">
                        <span className="min-w-[11rem] text-ink">{nodeLabels[c.node] ?? labelFor(c.node)}</span>
                        <ModelChip model={c.model} fallback={c.fallback} />
                      </li>
                    ))}
                  </ul>
                )}
                {result.errors.length > 0 && (
                  <>
                    <h3 className="mt-5 font-semibold">Warnings</h3>
                    <List items={result.errors} className="mt-1 text-ink-soft" />
                  </>
                )}
              </div>
            </Section>

            <Section id="sources">
              <ul className="space-y-3 font-sans text-[14px]">
                {result.sources.length === 0 && <li className="text-pencil">No sources were collected.</li>}
                {groupSourcesBySite(result.sources.filter((url) => safeUrl(url))).map((group) => (
                  <li key={group.site}>
                    <p className="font-medium text-ink">
                      {group.site} <span className="font-normal text-pencil">({group.urls.length})</span>
                    </p>
                    <ul className="mt-1 space-y-0.5">
                      {group.urls.map((url) => (
                        <li key={url} className="truncate">
                          <a href={safeUrl(url)!} target="_blank" rel="noreferrer" className="text-ballpoint hover:underline">
                            {url.replace(/^https?:\/\/(www\.)?/, "")}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </li>
                ))}
              </ul>
            </Section>
          </div>
        </article>
      </div>
    </div>
  );
}
