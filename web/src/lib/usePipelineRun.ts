// Owns one pipeline run (live or replayed). Runs start from user actions,
// never from effects, so StrictMode double-mounting can't trigger a paid run twice.

import { useCallback, useRef, useState } from "react";
import { api, errorMessage } from "./api";
import { initialRunState, runReducer, type RunState } from "./run";
import type { RequestCredentials } from "./settings";
import type { MeetingFields, PipelineEvent } from "./types";

export type RunSource =
  | { mode: "live"; fields: MeetingFields; rawText?: string; credentials?: RequestCredentials }
  | { mode: "demo"; slug: string };

export function usePipelineRun(nodeIds: string[]) {
  const [state, setState] = useState<RunState>(() => initialRunState(nodeIds));
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const controller = useRef<AbortController | null>(null);

  const start = useCallback(
    (source: RunSource) => {
      controller.current?.abort();
      const abort = new AbortController();
      controller.current = abort;
      setState(initialRunState(nodeIds));
      setStartedAt(performance.now());

      const onEvent = (event: PipelineEvent) => {
        if (!abort.signal.aborted) setState((s) => runReducer(s, event));
      };
      const request =
        source.mode === "live"
          ? api.run(source.fields, source.rawText, source.credentials, onEvent, abort.signal)
          : api.replay(source.slug, onEvent, abort.signal);

      request
        .then(() => {
          // A stream that closes without a result or error was cut off.
          setState((s) =>
            s.phase === "running" || s.phase === "idle"
              ? { ...s, phase: "error", error: "The connection closed before the brief was finished. Try again." }
              : s,
          );
        })
        .catch((error) => {
          if (abort.signal.aborted) return;
          setState((s) => ({ ...s, phase: "error", error: errorMessage(error) }));
        });
    },
    [nodeIds],
  );

  const cancel = useCallback(() => {
    controller.current?.abort();
    controller.current = null;
  }, []);

  return { state, startedAt, start, cancel };
}
