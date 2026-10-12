import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { StoredCredentials } from "../lib/settings";
import type { ProviderId, ProviderPreset } from "../lib/types";
import { Button } from "./ui";

interface Props {
  open: boolean;
  providers: ProviderPreset[];
  initial: StoredCredentials | null;
  onSave: (creds: StoredCredentials) => void;
  onClear: () => void;
  onClose: () => void;
}

const EMPTY: StoredCredentials = {
  llm_provider: "gemini",
  llm_api_key: "",
  llm_model_fast: "",
  llm_model_reasoning: "",
  tavily_api_key: "",
  remember: false,
};

const inputClass =
  "mt-1 w-full rounded-[3px] border border-rule bg-surface px-3 py-2 text-[15px] text-ink placeholder:text-pencil focus:border-ballpoint focus:outline-none";

export function SettingsDrawer({ open, providers, initial, onSave, onClear, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [form, setForm] = useState<StoredCredentials>(initial ?? EMPTY);

  useEffect(() => {
    const el = dialog.current;
    if (!el) return;
    if (open && !el.open) {
      setForm(initial ?? EMPTY);
      el.showModal();
    }
    if (!open && el.open) el.close();
  }, [open, initial]);

  const preset = providers.find((p) => p.id === form.llm_provider);
  const set = (patch: Partial<StoredCredentials>) => setForm((f) => ({ ...f, ...patch }));
  const missingModel = Boolean(preset?.needs_model && form.llm_api_key && !form.llm_model_fast.trim());

  return (
    <dialog
      ref={dialog}
      onClose={onClose}
      className="m-0 ml-auto h-dvh max-h-dvh w-full max-w-md bg-paper p-0 text-ink backdrop:bg-ink/40"
    >
      <form
        method="dialog"
        className="flex h-full flex-col"
        onSubmit={(e) => {
          e.preventDefault();
          if (missingModel) return;
          onSave(form);
        }}
      >
        <div className="flex items-center justify-between border-b border-rule px-5 py-4">
          <h2 className="text-[19px] font-semibold">Your API keys</h2>
          <button type="button" onClick={onClose} aria-label="Close" className="p-1 text-ink-soft hover:text-ink">
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 space-y-5 overflow-y-auto px-5 py-5 text-[15px]">
          <p className="text-ink-soft">
            Use your own keys to run real briefs. They stay in this browser and are sent only with your own
            requests; the server never stores them.
          </p>

          <label className="block">
            <span className="font-medium">Language model provider</span>
            <select
              className={inputClass}
              value={form.llm_provider}
              onChange={(e) => set({ llm_provider: e.target.value as ProviderId })}
            >
              {providers.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.label}
                  {p.free_tier ? " (free tier)" : ""}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="font-medium">{preset?.label ?? "Provider"} API key</span>
            <input
              type="password"
              autoComplete="off"
              spellCheck={false}
              className={inputClass}
              value={form.llm_api_key}
              onChange={(e) => set({ llm_api_key: e.target.value })}
            />
            {preset && (
              <a href={preset.key_url} target="_blank" rel="noreferrer" className="mt-1 inline-block text-[13px] text-ballpoint hover:underline">
                Get a {preset.label} key
              </a>
            )}
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="font-medium">Fast model</span>
              <input
                className={inputClass}
                placeholder={preset?.model_fast ?? "required"}
                value={form.llm_model_fast}
                onChange={(e) => set({ llm_model_fast: e.target.value })}
                aria-invalid={missingModel}
              />
            </label>
            <label className="block">
              <span className="font-medium">Reasoning model</span>
              <input
                className={inputClass}
                placeholder={preset?.model_reasoning ?? "same as fast"}
                value={form.llm_model_reasoning}
                onChange={(e) => set({ llm_model_reasoning: e.target.value })}
              />
            </label>
          </div>
          {missingModel && (
            <p className="-mt-3 text-[13px] text-alert">{preset?.label} needs a model name, for example a model id from its catalogue.</p>
          )}

          <label className="block">
            <span className="font-medium">Tavily API key</span>
            <input
              type="password"
              autoComplete="off"
              spellCheck={false}
              className={inputClass}
              value={form.tavily_api_key}
              onChange={(e) => set({ tavily_api_key: e.target.value })}
            />
            <span className="mt-1 block text-[13px] text-ink-soft">
              The research agents use Tavily for web search.{" "}
              <a href="https://app.tavily.com" target="_blank" rel="noreferrer" className="text-ballpoint hover:underline">
                Free key
              </a>
            </span>
          </label>

          <label className="flex items-start gap-2">
            <input
              type="checkbox"
              className="mt-1 accent-[var(--ink)]"
              checked={form.remember}
              onChange={(e) => set({ remember: e.target.checked })}
            />
            <span>
              Remember on this device
              <span className="block text-[13px] text-ink-soft">Otherwise the keys are forgotten when you close the tab.</span>
            </span>
          </label>
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-rule px-5 py-4">
          <Button variant="quiet" onClick={onClear}>
            Remove keys
          </Button>
          <Button type="submit" disabled={missingModel}>
            Save keys
          </Button>
        </div>
      </form>
    </dialog>
  );
}
