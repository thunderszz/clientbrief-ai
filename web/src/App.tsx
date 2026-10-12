import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Header } from "./components/Header";
import { SettingsDrawer } from "./components/SettingsDrawer";
import { api, errorMessage } from "./lib/api";
import { demoLinkFromHash, labelFor } from "./lib/format";
import {
  clearCredentials,
  loadCredentials,
  saveCredentials,
  toRequestCredentials,
  type StoredCredentials,
} from "./lib/settings";
import { applyTheme, storedTheme, systemTheme, type Theme } from "./lib/theme";
import type { AppConfig, BriefResult, DemoSummary, IntakeResponse, MeetingFields } from "./lib/types";
import { usePipelineRun } from "./lib/usePipelineRun";
import { BriefScreen } from "./screens/BriefScreen";
import { ComposeScreen } from "./screens/ComposeScreen";
import { ReviewScreen } from "./screens/ReviewScreen";
import { RunScreen } from "./screens/RunScreen";

type View =
  | { name: "compose" }
  | { name: "review"; intake: IntakeResponse }
  | { name: "run"; title: string; subtitle: string; demo?: DemoSummary; fields?: MeetingFields }
  | { name: "brief"; result: BriefResult; origin: string };

function setHash(hash: string) {
  try {
    window.history.replaceState(null, "", hash ? `#${hash}` : window.location.pathname);
  } catch {
    // History API unavailable (sandboxed preview): links just won't update.
  }
}

function recordedOrigin(demo: DemoSummary): string {
  const date = new Date(demo.recorded_at).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
  return `Recorded run from ${date}`;
}

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [demos, setDemos] = useState<DemoSummary[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [view, setView] = useState<View>({ name: "compose" });
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [creds, setCreds] = useState<StoredCredentials | null>(() => loadCredentials());
  const [keysOpen, setKeysOpen] = useState(false);
  const [theme, setTheme] = useState<Theme>(() => storedTheme() ?? systemTheme());
  const deepLinkHandled = useRef(false);

  const nodeIds = useMemo(() => config?.pipeline.nodes.map((n) => n.id) ?? [], [config]);
  const nodeLabels = useMemo(
    () => Object.fromEntries((config?.pipeline.nodes ?? []).map((n) => [n.id, n.label])),
    [config],
  );
  const run = usePipelineRun(nodeIds);
  const requestCreds = toRequestCredentials(creds);
  const canRunLive = Boolean(requestCreds) || Boolean(config?.server_runs.available);

  useEffect(() => {
    applyTheme(theme, storedTheme() !== null);
  }, [theme]);

  useEffect(() => {
    Promise.all([api.config(), api.demos()])
      .then(([cfg, list]) => {
        setConfig(cfg);
        setDemos(list);
      })
      .catch((e) => setLoadError(errorMessage(e)));
  }, []);

  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [view.name]);

  const watchDemo = useCallback(
    (demo: DemoSummary) => {
      setHash(`demo=${demo.slug}`);
      setView({ name: "run", title: demo.title, subtitle: `${demo.subtitle}. A recorded run, replayed.`, demo });
      run.start({ mode: "demo", slug: demo.slug });
    },
    [run],
  );

  // Shareable links: #demo=<slug> replays a recorded run, #brief=<slug> opens its brief.
  useEffect(() => {
    if (deepLinkHandled.current || !config || demos.length === 0) return;
    deepLinkHandled.current = true;
    const link = demoLinkFromHash(window.location.hash);
    const demo = demos.find((d) => d.slug === link?.slug);
    if (!link || !demo) return;
    if (link.view === "replay") {
      watchDemo(demo);
      return;
    }
    api
      .demoResult(demo.slug)
      .then((result) => setView({ name: "brief", result, origin: recordedOrigin(demo) }))
      .catch((e) => setLoadError(errorMessage(e)));
  }, [config, demos, watchDemo]);

  function goHome() {
    run.cancel();
    setHash("");
    setError(null);
    setView({ name: "compose" });
  }

  async function submitText() {
    setBusy(true);
    setError(null);
    try {
      const intake = await api.intake(text.trim(), requestCreds);
      setView({ name: "review", intake });
    } catch (e) {
      setError(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  function startLive(fields: MeetingFields) {
    setError(null);
    const models = requestCreds ? creds?.llm_provider : config?.models?.fast;
    setView({
      name: "run",
      title: fields.company_name,
      subtitle: `${labelFor(fields.meeting_type)} with the ${fields.contact_title || fields.stakeholder_role}. Live run${
        models ? ` on ${models}${config?.models && !requestCreds ? ` and ${config.models.reasoning}` : ""}` : ""
      }.`,
      fields,
    });
    run.start({ mode: "live", fields, rawText: text.trim() || undefined, credentials: requestCreds });
  }

  const openBrief = useCallback(() => {
    if (view.name !== "run" || !run.state.result) return;
    const origin = view.demo ? recordedOrigin(view.demo) : "Live run";
    setView({ name: "brief", result: run.state.result, origin });
  }, [view, run.state.result]);

  return (
    <div className="min-h-dvh bg-paper text-ink">
      <Header
        theme={theme}
        hasKeys={Boolean(requestCreds)}
        onHome={goHome}
        onToggleTheme={() => {
          const next: Theme = theme === "dark" ? "light" : "dark";
          applyTheme(next, true);
          setTheme(next);
        }}
        onOpenKeys={() => setKeysOpen(true)}
      />

      <main>
        {loadError && (
          <p role="alert" className="mx-auto max-w-6xl px-4 pt-6 text-[15px] text-alert md:px-8">
            {loadError}
          </p>
        )}
        {view.name === "compose" && (
          <ComposeScreen
            config={config}
            demos={demos}
            canRunLive={canRunLive}
            text={text}
            onTextChange={setText}
            busy={busy}
            error={error}
            onSubmit={submitText}
            onWatchDemo={watchDemo}
            onOpenKeys={() => setKeysOpen(true)}
          />
        )}
        {view.name === "review" && (
          <ReviewScreen
            intake={view.intake}
            busy={false}
            error={error}
            onBack={() => setView({ name: "compose" })}
            onSubmit={startLive}
          />
        )}
        {view.name === "run" && config && (
          <RunScreen
            title={view.title}
            subtitle={view.subtitle}
            isReplay={Boolean(view.demo)}
            nodes={config.pipeline.nodes}
            state={run.state}
            startedAt={run.startedAt}
            onCancel={goHome}
            onOpenBrief={openBrief}
          />
        )}
        {view.name === "brief" && (
          <BriefScreen result={view.result} origin={view.origin} nodeLabels={nodeLabels} onNew={goHome} />
        )}
      </main>

      <footer className="no-print border-t border-rule">
        <div className="mx-auto flex max-w-6xl flex-wrap justify-between gap-2 px-4 py-6 text-[14px] text-pencil md:px-8">
          <p>Built by Francesco Cipriano with LangGraph, FastAPI and React.</p>
          <a href="https://github.com/thunderszz/clientbrief-ai" className="hover:text-ink">
            Source code on GitHub
          </a>
        </div>
      </footer>

      {config && (
        <SettingsDrawer
          open={keysOpen}
          providers={config.providers}
          initial={creds}
          onSave={(next) => {
            saveCredentials(next);
            setCreds(next);
            setKeysOpen(false);
          }}
          onClear={() => {
            clearCredentials();
            setCreds(null);
            setKeysOpen(false);
          }}
          onClose={() => setKeysOpen(false)}
        />
      )}
    </div>
  );
}
