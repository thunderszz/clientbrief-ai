// Visitor API keys live in the browser only: sessionStorage by default,
// localStorage when "remember on this device" is ticked. Storage can be
// blocked (private mode, previews), so every access is guarded.

import type { ProviderId } from "./types";

export interface StoredCredentials {
  llm_provider: ProviderId;
  llm_api_key: string;
  llm_model_fast: string;
  llm_model_reasoning: string;
  tavily_api_key: string;
  remember: boolean;
}

export interface RequestCredentials {
  llm_provider: ProviderId;
  llm_api_key: string;
  llm_model_fast?: string;
  llm_model_reasoning?: string;
  tavily_api_key?: string;
}

interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export interface Stores {
  session: KeyValueStore;
  local: KeyValueStore;
}

const KEY = "clientbrief.credentials";

function browserStores(): Stores | null {
  try {
    return { session: window.sessionStorage, local: window.localStorage };
  } catch {
    return null;
  }
}

function safe<T>(fn: () => T, fallback: T): T {
  try {
    return fn();
  } catch {
    return fallback;
  }
}

export function loadCredentials(stores: Stores | null = browserStores()): StoredCredentials | null {
  if (!stores) return null;
  const raw = safe(() => stores.local.getItem(KEY), null) ?? safe(() => stores.session.getItem(KEY), null);
  if (!raw) return null;
  return safe(() => JSON.parse(raw) as StoredCredentials, null);
}

export function saveCredentials(creds: StoredCredentials, stores: Stores | null = browserStores()): void {
  if (!stores) return;
  const [target, other] = creds.remember ? [stores.local, stores.session] : [stores.session, stores.local];
  safe(() => other.removeItem(KEY), undefined);
  safe(() => target.setItem(KEY, JSON.stringify(creds)), undefined);
}

export function clearCredentials(stores: Stores | null = browserStores()): void {
  if (!stores) return;
  safe(() => stores.session.removeItem(KEY), undefined);
  safe(() => stores.local.removeItem(KEY), undefined);
}

export function toRequestCredentials(creds: StoredCredentials | null): RequestCredentials | undefined {
  if (!creds || !creds.llm_api_key.trim()) return undefined;
  const out: RequestCredentials = { llm_provider: creds.llm_provider, llm_api_key: creds.llm_api_key.trim() };
  if (creds.llm_model_fast.trim()) out.llm_model_fast = creds.llm_model_fast.trim();
  if (creds.llm_model_reasoning.trim()) out.llm_model_reasoning = creds.llm_model_reasoning.trim();
  if (creds.tavily_api_key.trim()) out.tavily_api_key = creds.tavily_api_key.trim();
  return out;
}
