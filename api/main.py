"""FastAPI app: serves the API and the built web UI from one process.

Run locally::

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.demos import demo_detail, demo_result, list_demos, load_demo, replay
from api.quota import DailyQuota
from api.runner import HEARTBEAT, stream_pipeline
from api.schemas import Credentials, IntakeRequest, RunRequest
from core.pipeline import PIPELINE_EDGES, PIPELINE_NODES
from core.state import INPUT_FIELDS
from services.llm_client import LLMSettings, settings_from_env, use_llm_settings
from services.providers import public_presets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PROJECT_ROOT / "demo"
WEB_DIST = PROJECT_ROOT / "web" / "dist"


_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def _encode(event: dict[str, Any]) -> str:
    if event is HEARTBEAT:
        return ": ping\n\n"
    return f"data: {json.dumps(event, separators=(',', ':'))}\n\n"


async def _sse(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[str]:
    async for event in events:
        yield _encode(event)


def _server_llm() -> LLMSettings | None:
    try:
        return settings_from_env()
    except OSError:
        return None


def create_app(
    demo_dir: Path = DEMO_DIR,
    web_dist: Path = WEB_DIST,
    graph: Any = None,
    intake_fn: Callable[[str], dict[str, Any]] | None = None,
    quota: DailyQuota | None = None,
) -> FastAPI:
    app = FastAPI(title="ClientBrief AI", version="2.0.0")
    quota = quota or DailyQuota.from_env()

    def run_intake(text: str) -> dict[str, Any]:
        if intake_fn is not None:
            return intake_fn(text)
        from agents.meeting_intake_agent import run

        return run(text)

    def server_runs_available() -> bool:
        return (
            _server_llm() is not None
            and bool(os.environ.get("TAVILY_API_KEY"))
            and quota.has_remaining()
        )

    def resolve_run_keys(credentials: Credentials | None) -> tuple[LLMSettings | None, str | None]:
        if credentials and credentials.has_llm_key():
            if not credentials.tavily_key():
                raise HTTPException(400, detail="byok_needs_tavily")
            return credentials.llm_settings(), credentials.tavily_key()
        if _server_llm() is None or not os.environ.get("TAVILY_API_KEY"):
            raise HTTPException(400, detail="no_keys")
        if not quota.try_consume():
            raise HTTPException(429, detail="quota_exhausted")
        return None, None

    # --- meta ------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "demos": len(list_demos(demo_dir)),
            "web_app": (web_dist / "index.html").is_file(),
        }

    @app.get("/api/config")
    def config() -> dict[str, Any]:
        server = _server_llm()
        return {
            "pipeline": {"nodes": PIPELINE_NODES, "edges": PIPELINE_EDGES},
            "provider": server.provider if server else None,
            "models": {"fast": server.model_fast, "reasoning": server.model_reasoning}
            if server
            else None,
            "server_runs": {"available": server_runs_available(), "remaining": quota.remaining},
            "providers": public_presets(),
        }

    # --- demos -----------------------------------------------------------------

    def get_demo(slug: str) -> dict[str, Any]:
        demo = load_demo(demo_dir, slug)
        if demo is None:
            raise HTTPException(404, detail="demo_not_found")
        return demo

    @app.get("/api/demos")
    def demos() -> list[dict[str, Any]]:
        return list_demos(demo_dir)

    @app.get("/api/demos/{slug}")
    def demo_stream(slug: str, speed: float = Query(1.0, ge=0.1, le=1000)) -> StreamingResponse:
        demo = get_demo(slug)
        return StreamingResponse(
            _sse(replay(demo["events"], speed=speed)),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    @app.get("/api/demos/{slug}/detail")
    def demo_meta(slug: str) -> dict[str, Any]:
        return demo_detail(get_demo(slug))

    @app.get("/api/demos/{slug}/result")
    def demo_final(slug: str) -> dict[str, Any]:
        result = demo_result(get_demo(slug))
        if result is None:
            raise HTTPException(404, detail="demo_has_no_result")
        return result

    # --- live runs ----------------------------------------------------------------

    @app.post("/api/intake")
    async def intake(request: IntakeRequest) -> dict[str, Any]:
        credentials = request.credentials
        if credentials and credentials.has_llm_key():
            llm = credentials.llm_settings()
        else:
            if _server_llm() is None:
                raise HTTPException(400, detail="no_keys")
            if not quota.has_remaining():
                raise HTTPException(429, detail="quota_exhausted")
            llm = None

        def work() -> dict[str, Any]:
            with use_llm_settings(llm):
                return run_intake(request.text)

        raw = await run_in_threadpool(work)
        fields = {key: raw.get(key) for key in INPUT_FIELDS}
        return {
            "ok": bool(
                fields["company_name"] or fields["meeting_goal"] or fields["stakeholder_role"]
            ),
            "fields": fields,
            "missing": raw.get("intake_missing_fields", []),
            "low_confidence": raw.get("intake_low_confidence_fields", []),
            "summary": raw.get("intake_confirmation_summary", ""),
        }

    @app.post("/api/runs")
    def runs(request: RunRequest) -> StreamingResponse:
        llm, tavily_key = resolve_run_keys(request.credentials)
        workflow_input = request.fields.to_workflow_input()
        if request.raw_text:
            workflow_input["raw_meeting_context"] = request.raw_text
        return StreamingResponse(
            _sse(stream_pipeline(workflow_input, llm, tavily_key, graph)),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    # --- web UI ----------------------------------------------------------------------

    index = web_dist / "index.html"
    if index.is_file():
        if (web_dist / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="assets")
        root = web_dist.resolve()

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str) -> FileResponse:
            if full_path == "api" or full_path.startswith("api/"):
                raise HTTPException(404)
            candidate = (web_dist / full_path).resolve()
            if full_path and candidate.is_file() and root in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(index)

    return app


if not os.environ.get("CLIENTBRIEF_SKIP_DOTENV"):
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

app = create_app()
