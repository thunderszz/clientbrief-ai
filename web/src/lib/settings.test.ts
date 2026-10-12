import { describe, expect, it } from "vitest";
import { clearCredentials, loadCredentials, saveCredentials, toRequestCredentials } from "./settings";
import type { StoredCredentials } from "./settings";

class MemoryStorage {
  data = new Map<string, string>();
  getItem(k: string) {
    return this.data.get(k) ?? null;
  }
  setItem(k: string, v: string) {
    this.data.set(k, v);
  }
  removeItem(k: string) {
    this.data.delete(k);
  }
}

class BrokenStorage {
  getItem(): string | null {
    throw new Error("blocked");
  }
  setItem() {
    throw new Error("blocked");
  }
  removeItem() {
    throw new Error("blocked");
  }
}

const creds: StoredCredentials = {
  llm_provider: "groq",
  llm_api_key: "gsk_123",
  llm_model_fast: "",
  llm_model_reasoning: "",
  tavily_api_key: "tvly_456",
  remember: false,
};

describe("credential storage", () => {
  it("keeps keys for the session only unless remembered", () => {
    const stores = { session: new MemoryStorage(), local: new MemoryStorage() };

    saveCredentials(creds, stores);

    expect(stores.local.data.size).toBe(0);
    expect(loadCredentials(stores)).toEqual(creds);
  });

  it("uses localStorage when remember is ticked and clears the other store", () => {
    const stores = { session: new MemoryStorage(), local: new MemoryStorage() };

    saveCredentials(creds, stores);
    saveCredentials({ ...creds, remember: true }, stores);

    expect(stores.session.data.size).toBe(0);
    expect(loadCredentials(stores)?.remember).toBe(true);
  });

  it("survives storage that throws", () => {
    const stores = { session: new BrokenStorage(), local: new BrokenStorage() };

    expect(() => saveCredentials(creds, stores)).not.toThrow();
    expect(loadCredentials(stores)).toBeNull();
    expect(() => clearCredentials(stores)).not.toThrow();
  });

  it("builds request credentials only when an LLM key is present", () => {
    expect(toRequestCredentials(null)).toBeUndefined();
    expect(toRequestCredentials({ ...creds, llm_api_key: "  " })).toBeUndefined();
    expect(toRequestCredentials(creds)).toEqual({
      llm_provider: "groq",
      llm_api_key: "gsk_123",
      tavily_api_key: "tvly_456",
    });
  });
});
