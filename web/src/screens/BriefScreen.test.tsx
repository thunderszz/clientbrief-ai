// Server-side render of the brief: no DOM needed, catches crashes and empty states.
import { renderToString } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { BriefResult } from "../lib/types";
import { BriefScreen } from "./BriefScreen";

function emptyResult(overrides: Partial<BriefResult> = {}): BriefResult {
  return {
    input: {
      company_name: "Acme",
      domain: "acme.com",
      meeting_type: "sales_discovery",
      stakeholder_role: "CFO",
      meeting_goal: "Learn about their finance stack",
    },
    engagement_inference: {},
    engagement_context: {},
    research_plan: {},
    company_profile: {},
    external_signals: [],
    person_profile: {},
    audience_context: {},
    pain_point_hypotheses: {},
    meeting_strategy: {},
    sources: [],
    errors: [],
    llm_calls: [],
    reliability: { score: 0, rating: 0, components: [] },
    final_brief: "# Acme",
    generated_at: "2026-10-12T10:00:00+00:00",
    ...overrides,
  };
}

function render(result: BriefResult): string {
  return renderToString(<BriefScreen result={result} origin="Live run" nodeLabels={{}} onNew={() => {}} />);
}

describe("BriefScreen", () => {
  it("explains empty sections instead of leaving bare headings", () => {
    const html = render(emptyResult());

    expect(html).toContain("No recent strategic signals were found");
    expect(html).toContain("No facts could be confirmed from the sources");
    expect(html).toContain("No suggested questions");
    expect(html).toContain("No objections anticipated");
  });

  it("never renders a non-web link from model output", () => {
    const html = render(
      emptyResult({
        external_signals: [
          {
            signal_type: "partnership",
            description: "Signed a deal",
            implication: "More volume",
            confidence: "high",
            source: "javascript:alert(1)",
            date: "2026-09",
          },
        ],
        sources: ["javascript:alert(2)", "https://acme.com/news"],
      }),
    );

    expect(html).not.toContain("javascript:");
    expect(html).toContain('href="https://acme.com/news"');
  });

  it("renders duplicate model sentences without crashing", () => {
    const html = render(
      emptyResult({ pain_point_hypotheses: { facts: ["Same fact", "Same fact"], inferences: [], hypotheses: [] } }),
    );

    expect(html.match(/Same fact/g)).toHaveLength(2);
  });
});
