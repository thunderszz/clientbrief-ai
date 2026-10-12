# ClientBrief AI v2: showcase redesign

Status: approved 2026-10-12 (Gemini default, model routing added after review)

## Why

The pipeline itself is solid (11-node LangGraph graph, parallel research, facts vs.
inferences vs. hypotheses kept apart), but nothing a visitor sees shows that:

- the UI is stock Streamlit, and the brief comes out as one long markdown wall;
- the progress bar is fake: `app.invoke()` blocks, so it jumps from 5% to 100% and the
  multi-agent part is invisible;
- the reliability score is `min(5, number of signals)`, which a technical reader will
  question in ten seconds;
- there are no tests, no CI, no screenshots and no live demo, and trying it requires
  your own Gemini and Tavily keys;
- the README on GitHub lost its formatting (the diagram and tables were flattened).

## Who it is for and what success looks like

Two readers, both of whom need to be convinced:

1. **A recruiter with a link and 30 seconds.** They open the demo, click one example,
   watch the agents work live, and land on a brief that looks like a real product.
   No signup, no key, no cost to Francesco.
2. **A technical interviewer reading the repo.** They find a clear README with an
   architecture diagram and design decisions, tests that run offline, CI that is green,
   and engineering choices they can ask about (streaming, provider abstraction,
   transparent scoring).

## Decisions already made

- Frontend: **FastAPI + React, one deploy.** FastAPI serves the API and the built UI.
- Audience: both of the above.
- LLM provider: **Gemini stays the default, and it must keep working on a free tier.**
  The client is made provider-agnostic so any OpenAI-compatible endpoint works by
  changing env vars: Groq (free tier), DeepSeek (cheap, not free), OpenRouter, or local
  Ollama.
- **Hard date: 2026-11-15.** Francesco expects to lose free Gemini API access that day.
  The replay demos must be recorded with the real pipeline before then, and the public
  demo must not depend on a live key.

### Model routing

Benchmarked on 2026-10-12 with the project key, using a structured-output call:

| model | result |
|---|---|
| `gemini-3.8-flash` | works, 7.5 s, thinking model |
| `gemini-3.5-flash-lite` | works, 1.1 s |
| `gemini-3.1-flash-lite` (current default) | works, 20.6 s |
| `gemini-3.5-flash` | 503, overloaded at test time |
| `gemini-3.1-pro-preview`, `gemini-pro-latest` | 429, no quota on this key |

The agents are split into two tiers, each set by its own env var:

- `LLM_MODEL_REASONING` (default `gemini-3.8-flash`): `audience_context_agent`,
  `pain_point_agent`, `meeting_strategy_agent`.
- `LLM_MODEL_FAST` (default `gemini-3.5-flash-lite`): intake, engagement inference,
  company/signals/person extraction.

On 429/503 a call retries with backoff, then falls back to the other tier's model, so a
busy model degrades the brief instead of failing it. The model actually used per node is
reported in the `node_finished` event and shown in the UI.

## Architecture

```
web/  (React + Vite + TypeScript + Tailwind)
  │  fetch + SSE
  ▼
api/  (FastAPI)
  ├── POST /api/intake            free text → structured fields (intake agent)
  ├── POST /api/runs              structured fields → SSE stream of pipeline events
  ├── GET  /api/demos             list of recorded example runs
  ├── GET  /api/demos/{slug}      SSE replay of a recorded run (no API calls)
  ├── GET  /api/health
  └── /                           built web app (static files)
  │
  ▼
core/workflow.py  (existing LangGraph graph, unchanged topology)
agents/           (existing agents, small changes listed below)
services/llm_client.py  → provider-agnostic
```

One Docker image: stage 1 builds `web/`, stage 2 is the Python app serving `web/dist`.

### Pipeline streaming

`api/runs.py` runs the sync graph in a worker thread with
`graph.stream(input, stream_mode=["tasks", "updates"])` and pushes events onto an
`asyncio.Queue` that the SSE response drains. Event types:

| event | payload |
|---|---|
| `node_started` | node id, timestamp |
| `node_finished` | node id, duration, short human summary (e.g. "7 sources, 4 signals") |
| `node_failed` | node id, error message |
| `result` | the full structured result (see below) |
| `error` | fatal error, run aborted |

The same event format is what gets recorded for demos, so live runs and replays render
through exactly the same UI code.

### Structured result

Today the UI only gets `final_brief` (markdown). v2 returns the structured state the
agents already produce: `company_profile`, `external_signals`, `person_profile`,
`audience_context`, `engagement_context`, `pain_point_hypotheses`, `meeting_strategy`,
`sources`, `reliability`, plus `final_brief` for copy/download. No agent output changes
shape; this is just exposing what's already in the state.

### Provider-agnostic LLM client

`structured_extract()` keeps its signature. Behind it:

- `LLM_PROVIDER=gemini` (default): the existing google-genai path with native JSON schema,
  with per-agent tier selection as described in "Model routing".
- `LLM_PROVIDER=openai_compatible` + `LLM_BASE_URL` + `LLM_API_KEY` + `LLM_MODEL`: uses the
  `openai` SDK with JSON mode, puts the Pydantic JSON schema in the system prompt,
  validates with Pydantic, and on a validation error does one repair retry that sends
  back the error.
- Both paths retry on 429/5xx with exponential backoff (free tiers rate-limit hard).
- Credentials come from a `contextvars.ContextVar` when set (per-request "bring your own
  key"), otherwise from env. Must be checked that LangGraph's thread pool copies context
  into parallel branches. If it does not, credentials go through
  `config["configurable"]` instead.

The Tavily client gets the same per-request key override.

### Reliability score v2

The score is deterministic and transparent: 0–100, built from named components that the UI shows
in a breakdown popover:

| component | what it measures | weight |
|---|---|---|
| Source coverage | number of distinct source URLs | 25 |
| Source diversity | distinct domains, bonus for the company's own domain | 20 |
| Strategic signals | count of signals, weighted by their confidence | 25 |
| Contact match | person match confidence (none/low/medium/high) | 15 |
| Pipeline health | penalty per agent error / empty LLM response | 15 |

`brief_reliability` (1–5) is still written for the markdown header, derived from the
0–100 score, so the CLI and markdown output keep working.

### Demo mode and keys

- **Replay**: `scripts/record_demo.py` runs the real pipeline once and saves the event
  stream with timings to `demo/<slug>.json`. Replays play back with the original
  pacing, compressed to roughly 20–30 s. Three demos: Satispay, Bending Spoons, and one more
  picked when recording.
- **Bring your own key**: a settings drawer in the UI takes an LLM key (and optionally provider,
  base URL and model) plus a Tavily key. They are sent with the request, held in memory for that run
  only, and never logged or stored server-side. In the browser they live in
  `sessionStorage` unless the user ticks "remember on this device".
- **Server keys**: real runs on the server's own keys are capped by
  `PUBLIC_DAILY_RUNS` (default 0 on the public deploy, unlimited locally). This is an
  in-memory counter per day, which is enough for a portfolio demo.

## Frontend

Built with the frontend-design skill at implementation time for the visual direction.
Four screens in one single-page flow:

1. **Compose.** One-line pitch, a large textarea, clickable example prompts, and a row of
   "Watch a demo run" cards. The key settings sit in a drawer.
2. **Review.** The intake agent's summary, the extracted fields as an editable form,
   missing required fields highlighted, and low-confidence fields marked with the agent's
   reason.
3. **Live run.** The pipeline drawn as a graph (the real fan-out/fan-in shape: four
   sequential nodes, three parallel research nodes, four reasoning nodes) with each
   node going idle → running → done/failed and showing its duration, next to a running
   log of node summaries.
4. **Brief.** A dashboard instead of a wall of text:
   - header with company, meeting type/stage, and a reliability gauge with breakdown;
   - executive summary;
   - signals as cards with category, implication, confidence and source link;
   - pain points that make the core design principle visible: **facts**, **inferences**
     and **hypotheses** styled distinctly, with hypotheses showing confidence and
     supporting evidence;
   - strategy with the opening angle, questions (copyable), objections as expandable
     items, and the next step;
   - contact, audience and engagement context, plus the full source list;
   - actions: copy markdown, download `.md`, print-friendly view, run again.

Light and dark themes. It should work on a phone, since recruiters often open links there.

## Engineering showcase

- `tests/` (pytest, fully offline): reliability scoring, brief composer, research planner
  mode trimming, intake enum sanitisation, LLM client parsing + repair retry with a fake
  transport, API demo replay via FastAPI `TestClient`.
- `.github/workflows/ci.yml`: ruff, pytest, web typecheck + build.
- `README.md` rewritten: demo GIF/screenshots, live link, a Mermaid architecture diagram
  (GitHub renders it), design decisions, provider setup, and "what I'd do next".
- `requirements.txt` with pinned versions, `.python-version`, `Dockerfile`, `.env.example`
  covering both provider modes.

## Removed

- `app/streamlit_app.py` and the Streamlit dependency (it's still in git history). The
  rich CLI in `app/main.py` stays, since it's a nice second interface and costs nothing.

## Out of scope

User accounts, saved brief history, a database, auth, payments, fine-tuning,
evaluation datasets. Wiring the existing file cache into the graph is also left out:
it isn't needed for the showcase, and it is listed under "what I'd do next".

## Deploy

The Dockerfile makes the target portable. Free options with Docker support include Hugging Face
Spaces, Render (sleeps when idle) and Koyeb. The target is picked at the end. It needs
Francesco's account login, so that step is his.

## Risks

- Free tiers change often (Gemini 2.5 Flash is already closed to new users). The
  provider abstraction and the replay demos make the public demo independent of any
  one provider's free tier.
- Docker isn't installed on this machine, so the image can't be built locally. CI
  builds it instead.
- LangGraph context propagation for BYOK (see above) is verified first, before the
  rest is built on it.
