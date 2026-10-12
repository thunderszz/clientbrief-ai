import { describe, expect, it } from "vitest";
import {
  demoSlugFromHash,
  formatDuration,
  formatSignalDate,
  groupSourcesBySite,
  labelFor,
  splitObjection,
  siteOf,
} from "./format";

describe("formatDuration", () => {
  it.each([
    [0.04, "0.0 s"],
    [7.25, "7.3 s"],
    [59.9, "59.9 s"],
    [93, "1 min 33 s"],
    [138.3, "2 min 18 s"],
  ])("%s → %s", (seconds, text) => {
    expect(formatDuration(seconds)).toBe(text);
  });
});

describe("siteOf", () => {
  it.each([
    ["https://www.satispay.com/en-it/", "satispay.com"],
    ["https://support.satispay.com/x", "satispay.com"],
    ["https://news.bbc.co.uk/a", "bbc.co.uk"],
    ["not a url", "not a url"],
  ])("%s → %s", (url, site) => {
    expect(siteOf(url)).toBe(site);
  });
});

describe("groupSourcesBySite", () => {
  it("groups by site, biggest first, then alphabetical", () => {
    const groups = groupSourcesBySite([
      "https://b.com/1",
      "https://www.a.com/1",
      "https://a.com/2",
      "https://c.com/1",
    ]);

    expect(groups.map((g) => [g.site, g.urls.length])).toEqual([
      ["a.com", 2],
      ["b.com", 1],
      ["c.com", 1],
    ]);
  });
});

describe("labelFor", () => {
  it("names known enum values", () => {
    expect(labelFor("discovery_followup")).toBe("Discovery follow-up");
    expect(labelFor("sales_discovery")).toBe("Sales discovery");
    expect(labelFor("no_previous_meeting")).toBe("No previous meeting");
  });

  it("falls back to sentence case", () => {
    expect(labelFor("leadership_change")).toBe("Leadership change");
    expect(labelFor(null)).toBe("");
  });
});

describe("formatSignalDate", () => {
  it("formats year-month and handles missing dates", () => {
    expect(formatSignalDate("2026-09")).toBe("Sep 2026");
    expect(formatSignalDate("2025-12-01")).toBe("Dec 2025");
    expect(formatSignalDate(null)).toBe("Undated");
    expect(formatSignalDate("soon")).toBe("Undated");
  });
});

describe("splitObjection", () => {
  it("separates the objection from the guidance", () => {
    expect(
      splitObjection("Objection: 'We are busy right now.' Guidance: Offer a short pilot."),
    ).toEqual({ objection: "We are busy right now.", response: "Offer a short pilot." });
  });

  it("handles the Concern: variant and missing guidance", () => {
    expect(splitObjection("Concern about integration disruption")).toEqual({
      objection: "Concern about integration disruption",
      response: "",
    });
    expect(splitObjection("Data security: 'How do you audit?' Guidance: Show the audit log.")).toEqual({
      objection: "Data security: 'How do you audit?'",
      response: "Show the audit log.",
    });
  });
});

describe("demoSlugFromHash", () => {
  it("reads #demo=slug links", () => {
    expect(demoSlugFromHash("#demo=satispay")).toBe("satispay");
    expect(demoSlugFromHash("#demo=Bad Slug")).toBeNull();
    expect(demoSlugFromHash("")).toBeNull();
  });
});
