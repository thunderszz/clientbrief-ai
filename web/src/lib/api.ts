import { readEventStream } from "./sse";
import type { RequestCredentials } from "./settings";
import type {
  AppConfig,
  DemoDetail,
  DemoSummary,
  IntakeResponse,
  MeetingFields,
  PipelineEvent,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(detail);
  }
}

async function check(response: Response): Promise<Response> {
  if (response.ok) return response;
  let detail = `http_${response.status}`;
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") detail = body.detail;
    else if (response.status === 422) detail = "invalid_input";
  } catch {
    // Non-JSON error body: keep the status-based detail.
  }
  throw new ApiError(response.status, detail);
}

async function getJson<T>(path: string): Promise<T> {
  return (await check(await fetch(path))).json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return (await check(response)).json() as Promise<T>;
}

type OnEvent = (event: PipelineEvent) => void;

export const api = {
  config: () => getJson<AppConfig>("/api/config"),
  demos: () => getJson<DemoSummary[]>("/api/demos"),
  demo: (slug: string) => getJson<DemoDetail>(`/api/demos/${encodeURIComponent(slug)}/detail`),
  intake: (text: string, credentials?: RequestCredentials) =>
    postJson<IntakeResponse>("/api/intake", { text, credentials }),

  async run(
    fields: MeetingFields,
    rawText: string | undefined,
    credentials: RequestCredentials | undefined,
    onEvent: OnEvent,
    signal?: AbortSignal,
  ): Promise<void> {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fields, raw_text: rawText, credentials }),
      signal,
    });
    await readEventStream(await check(response), (e) => onEvent(e as PipelineEvent));
  },

  async replay(slug: string, onEvent: OnEvent, signal?: AbortSignal, speed = 1): Promise<void> {
    const response = await fetch(`/api/demos/${encodeURIComponent(slug)}?speed=${speed}`, { signal });
    await readEventStream(await check(response), (e) => onEvent(e as PipelineEvent));
  },
};

const MESSAGES: Record<string, string> = {
  no_keys: "This server has no API keys set up. Add your own keys in Settings, or watch a recorded run.",
  quota_exhausted:
    "Today's free runs on this server are used up. Add your own keys in Settings, or watch a recorded run.",
  byok_needs_tavily: "Add a Tavily key in Settings too: the research agents need web search.",
  invalid_input: "Some fields are missing or too long. Check the fields marked below.",
  demo_not_found: "That recorded run doesn't exist any more.",
};

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return MESSAGES[error.detail] ?? `The server answered with an error (${error.status}). Try again.`;
  }
  if (error instanceof DOMException && error.name === "AbortError") return "Stopped.";
  return "Can't reach the server. Check your connection and try again.";
}
