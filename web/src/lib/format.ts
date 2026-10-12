// Small, pure formatting helpers shared by the screens.

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const total = Math.round(seconds);
  return `${Math.floor(total / 60)} min ${total % 60} s`;
}

const SECOND_LEVEL = new Set(["co", "com", "org", "net", "gov", "ac", "edu"]);

/** Registrable site of a URL, mirroring core/reliability.py: news.bbc.co.uk → bbc.co.uk. */
export function siteOf(url: string): string {
  let host: string;
  try {
    host = new URL(url).hostname.toLowerCase();
  } catch {
    return url;
  }
  const labels = host.split(".");
  if (labels.length >= 3 && labels.at(-1)!.length === 2 && SECOND_LEVEL.has(labels.at(-2)!)) {
    return labels.slice(-3).join(".");
  }
  return labels.slice(-2).join(".");
}

export function groupSourcesBySite(urls: string[]): { site: string; urls: string[] }[] {
  const groups = new Map<string, string[]>();
  for (const url of urls) {
    const site = siteOf(url);
    groups.set(site, [...(groups.get(site) ?? []), url]);
  }
  return [...groups.entries()]
    .map(([site, list]) => ({ site, urls: list }))
    .sort((a, b) => b.urls.length - a.urls.length || a.site.localeCompare(b.site));
}

const LABELS: Record<string, string> = {
  consulting_intro: "Consulting intro",
  sales_discovery: "Sales discovery",
  account_review: "Account review",
  first_intro: "First intro",
  discovery_followup: "Discovery follow-up",
  solution_discussion: "Solution discussion",
  proposal_review: "Proposal review",
  negotiation: "Negotiation",
  client_kickoff: "Client kickoff",
  account_expansion: "Account expansion",
  prospect: "Prospect",
  active_opportunity: "Active opportunity",
  client: "Client",
  former_client: "Former client",
  positive: "Positive",
  neutral: "Neutral",
  unclear: "Unclear",
  no_previous_meeting: "No previous meeting",
  it_data: "IT and data",
  c_level: "C-level",
  product_launch: "Product launch",
};

export function labelFor(value: string | null | undefined): string {
  if (!value) return "";
  if (LABELS[value]) return LABELS[value];
  const words = value.replace(/_/g, " ").trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function formatSignalDate(raw: string | null | undefined): string {
  const match = /^(\d{4})-(\d{2})/.exec(raw ?? "");
  if (!match) return "Undated";
  const month = Number(match[2]);
  return month >= 1 && month <= 12 ? `${MONTHS[month - 1]} ${match[1]}` : "Undated";
}

/** Strategy objections arrive as "Objection: '…' Guidance: …"; split them for display. */
export function splitObjection(text: string): { objection: string; response: string } {
  const [head, ...rest] = text.split(/\s*Guidance:\s*/i);
  let objection = head.replace(/^\s*Objection:\s*/i, "").trim();
  const quoted = /^['"‘“](.*)['"’”]$/.exec(objection);
  if (quoted) objection = quoted[1].trim();
  return { objection, response: rest.join(" ").trim() };
}

export interface DemoLink {
  slug: string;
  view: "replay" | "brief";
}

/** Shareable links: #demo=<slug> replays a recorded run, #brief=<slug> opens its brief. */
export function demoLinkFromHash(hash: string): DemoLink | null {
  const match = /^#(demo|brief)=([a-z0-9][a-z0-9-]{0,60})$/.exec(hash);
  if (!match) return null;
  return { slug: match[2], view: match[1] === "demo" ? "replay" : "brief" };
}

/** Only http(s) links from model or search output become clickable. */
export function safeUrl(raw: string | null | undefined): string | null {
  if (!raw) return null;
  try {
    const url = new URL(raw.trim());
    return url.protocol === "http:" || url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}
