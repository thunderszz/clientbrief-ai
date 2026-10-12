"""API tests: fake graph, fake intake, temp demo dir. No network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from api.quota import DailyQuota
from api.schemas import Credentials
from services.llm_client import current_settings
from services.tavily_client import _key_override

FIELDS = {
    "company_name": "Acme",
    "domain": "acme.com",
    "meeting_type": "sales_discovery",
    "stakeholder_role": "CFO",
    "meeting_goal": "Discuss finance automation",
}


class FakeGraph:
    """Mimics ``CompiledStateGraph.stream`` with tasks/updates/values chunks."""

    def __init__(self, fail_in: str | None = None) -> None:
        self.fail_in = fail_in
        self.seen_settings = []
        self.seen_tavily = []

    def stream(self, workflow_input, stream_mode):
        assert stream_mode == ["tasks", "updates", "values"]
        self.seen_settings.append(current_settings())
        self.seen_tavily.append(_key_override.get())
        state = dict(workflow_input)
        for node, update in [
            ("orchestrate", {"normalized_function": "finance", "errors": []}),
            ("research_signals", {"external_signals": [{"signal_type": "financial"}]}),
            ("compose", {"final_brief": "# Brief", "reliability_report": {"score": 70}, "sources": []}),
        ]:
            yield "tasks", {"id": node, "name": node, "input": {}, "triggers": []}
            if node == self.fail_in:
                raise RuntimeError("secret-ish internal detail")
            update = {**update, "llm_calls": [{"model": "m", "node": node}]}
            yield "updates", {node: update}
            yield "tasks", {"id": node, "name": node, "result": update, "error": None}
            state.update(update)
            yield "values", dict(state)


def fake_intake(text):
    return {
        **FIELDS,
        "contact_name": None,
        "raw_meeting_context": text,
        "intake_missing_fields": [],
        "intake_low_confidence_fields": [{"field": "domain", "value": "acme.com", "reason": "guessed"}],
        "intake_confirmation_summary": "Sales discovery with Acme's CFO.",
    }


def _demo(slug="acme-demo"):
    return {
        "slug": slug,
        "title": "Acme discovery call",
        "subtitle": "CFO, finance automation",
        "company": "Acme",
        "input_text": "Meeting with Acme's CFO",
        "intake": fake_intake("Meeting with Acme's CFO"),
        "recorded_at": "2026-10-12T10:00:00+00:00",
        "models": {"fast": "gemini-3.5-flash-lite", "reasoning": "gemini-3.6-flash"},
        "events": [
            {"type": "run_started", "t": 0.0},
            {"type": "node_started", "t": 0.1, "node": "orchestrate"},
            {"type": "node_finished", "t": 0.5, "node": "orchestrate", "duration": 0.4},
            {"type": "result", "t": 1.0, "result": {"final_brief": "# Demo brief"}},
        ],
    }


@pytest.fixture
def demo_dir(tmp_path: Path) -> Path:
    (tmp_path / "acme-demo.json").write_text(json.dumps(_demo()))
    (tmp_path / "broken.json").write_text("{not json")
    return tmp_path


def make_client(demo_dir, graph=None, quota_limit=None, web_dist=None):
    app = create_app(
        demo_dir=demo_dir,
        web_dist=web_dist or demo_dir / "no-web",
        graph=graph or FakeGraph(),
        intake_fn=fake_intake,
        quota=DailyQuota(quota_limit),
    )
    return TestClient(app)


def sse_events(response) -> list[dict]:
    events = []
    for block in response.text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


# --- health & config ----------------------------------------------------------


def test_health(demo_dir):
    response = make_client(demo_dir).get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_config_describes_pipeline_and_server_runs(demo_dir):
    body = make_client(demo_dir, quota_limit=3).get("/api/config").json()

    assert len(body["pipeline"]["nodes"]) == 11
    assert ["plan", "research_company"] in body["pipeline"]["edges"]
    assert body["server_runs"] == {"available": True, "remaining": 3}
    assert body["models"] == {"fast": "gemini-3.5-flash-lite", "reasoning": "gemini-3.6-flash"}
    assert {p["id"] for p in body["providers"]} >= {"gemini", "groq", "deepseek", "openrouter"}


def test_config_without_server_keys(demo_dir, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")

    body = make_client(demo_dir).get("/api/config").json()

    assert body["server_runs"]["available"] is False


def test_config_with_zero_quota(demo_dir):
    body = make_client(demo_dir, quota_limit=0).get("/api/config").json()

    assert body["server_runs"] == {"available": False, "remaining": 0}


# --- demos ------------------------------------------------------------------------


def test_demos_list_skips_broken_files(demo_dir):
    demos = make_client(demo_dir).get("/api/demos").json()

    assert [d["slug"] for d in demos] == ["acme-demo"]
    assert demos[0]["title"] == "Acme discovery call"
    assert demos[0]["duration_s"] == 1.0
    assert "events" not in demos[0]


def test_demo_replay_streams_recorded_events(demo_dir):
    response = make_client(demo_dir).get("/api/demos/acme-demo?speed=1000")

    assert response.headers["content-type"].startswith("text/event-stream")
    events = sse_events(response)
    assert [e["type"] for e in events] == ["run_started", "node_started", "node_finished", "result"]
    assert events[-1]["result"]["final_brief"] == "# Demo brief"


def test_demo_detail_and_result(demo_dir):
    client = make_client(demo_dir)

    detail = client.get("/api/demos/acme-demo/detail").json()
    result = client.get("/api/demos/acme-demo/result").json()

    assert detail["intake"]["company_name"] == "Acme"
    assert "events" not in detail
    assert result["final_brief"] == "# Demo brief"


@pytest.mark.parametrize("slug", ["missing", "..%2Fsecrets", "UPPER"])
def test_unknown_or_unsafe_demo_slug_404(demo_dir, slug):
    assert make_client(demo_dir).get(f"/api/demos/{slug}").status_code == 404


# --- runs ----------------------------------------------------------------------------


def test_run_with_server_keys_streams_events(demo_dir):
    response = make_client(demo_dir).post("/api/runs", json={"fields": FIELDS})

    events = sse_events(response)
    types = [e["type"] for e in events]
    assert types[0] == "run_started"
    assert types[-1] == "result"
    assert types.count("node_started") == 3
    finished = [e for e in events if e["type"] == "node_finished"]
    assert [e["node"] for e in finished] == ["orchestrate", "research_signals", "compose"]
    assert finished[1]["summary"] == "1 signal · financial"
    assert finished[1]["llm_calls"] == [{"model": "m", "node": "research_signals"}]
    assert all(e["duration"] >= 0 for e in finished)
    assert events[-1]["result"]["final_brief"] == "# Brief"


def test_runs_without_keys_400(demo_dir, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")

    response = make_client(demo_dir).post("/api/runs", json={"fields": FIELDS})

    assert response.status_code == 400
    assert response.json()["detail"] == "no_keys"


def test_runs_without_tavily_key_400(demo_dir, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY")

    response = make_client(demo_dir).post("/api/runs", json={"fields": FIELDS})

    assert response.status_code == 400
    assert response.json()["detail"] == "no_keys"


def test_quota_exhausted_429(demo_dir):
    client = make_client(demo_dir, quota_limit=1)

    first = client.post("/api/runs", json={"fields": FIELDS})
    second = client.post("/api/runs", json={"fields": FIELDS})

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "quota_exhausted"


def test_byok_run_uses_visitor_keys_and_skips_quota(demo_dir):
    graph = FakeGraph()
    client = make_client(demo_dir, graph=graph, quota_limit=0)
    credentials = {
        "llm_provider": "groq",
        "llm_api_key": "visitor-llm",
        "tavily_api_key": "visitor-tavily",
    }

    response = client.post("/api/runs", json={"fields": FIELDS, "credentials": credentials})

    assert response.status_code == 200
    settings = graph.seen_settings[0]
    assert settings.api_key == "visitor-llm"
    assert settings.provider == "openai_compatible"
    assert settings.base_url == "https://api.groq.com/openai/v1"
    assert graph.seen_tavily == ["visitor-tavily"]


def test_byok_run_needs_tavily_key(demo_dir):
    credentials = {"llm_provider": "gemini", "llm_api_key": "visitor-llm"}

    response = make_client(demo_dir).post(
        "/api/runs", json={"fields": FIELDS, "credentials": credentials}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "byok_needs_tavily"


def test_byok_rejects_custom_base_url(demo_dir):
    credentials = {
        "llm_provider": "groq",
        "llm_api_key": "k",
        "tavily_api_key": "t",
        "llm_base_url": "http://169.254.169.254/latest",
    }

    response = make_client(demo_dir).post(
        "/api/runs", json={"fields": FIELDS, "credentials": credentials}
    )

    assert response.status_code == 422


def test_openrouter_byok_requires_model(demo_dir):
    credentials = {"llm_provider": "openrouter", "llm_api_key": "k", "tavily_api_key": "t"}

    response = make_client(demo_dir).post(
        "/api/runs", json={"fields": FIELDS, "credentials": credentials}
    )

    assert response.status_code == 422


def test_missing_required_field_422(demo_dir):
    fields = {**FIELDS, "company_name": "  "}

    response = make_client(demo_dir).post("/api/runs", json={"fields": fields})

    assert response.status_code == 422


def test_stream_node_exception(demo_dir):
    response = make_client(demo_dir, graph=FakeGraph(fail_in="research_signals")).post(
        "/api/runs", json={"fields": FIELDS}
    )

    events = sse_events(response)
    assert response.status_code == 200
    assert events[-2] == {**events[-2], "type": "node_failed", "node": "research_signals"}
    assert events[-1]["type"] == "error"
    assert "Strategic signals" in events[-1]["message"]
    assert "secret-ish" not in response.text


# --- intake ---------------------------------------------------------------------------


def test_intake_returns_fields(demo_dir):
    response = make_client(demo_dir).post("/api/intake", json={"text": "Meeting with Acme's CFO"})

    body = response.json()
    assert response.status_code == 200
    assert body["ok"] is True
    assert body["fields"]["company_name"] == "Acme"
    assert body["summary"] == "Sales discovery with Acme's CFO."
    assert body["low_confidence"][0]["field"] == "domain"


def test_intake_reports_failure(demo_dir):
    app = create_app(
        demo_dir=demo_dir,
        web_dist=demo_dir / "no-web",
        graph=FakeGraph(),
        intake_fn=lambda text: {"company_name": "", "intake_confirmation_summary": "LLM extraction failed"},
        quota=DailyQuota(None),
    )

    body = TestClient(app).post("/api/intake", json={"text": "hello"}).json()

    assert body["ok"] is False


def test_intake_rejects_oversized_text(demo_dir):
    response = make_client(demo_dir).post("/api/intake", json={"text": "x" * 5000})

    assert response.status_code == 422


def test_intake_blocked_when_quota_is_zero(demo_dir):
    response = make_client(demo_dir, quota_limit=0).post("/api/intake", json={"text": "hi"})

    assert response.status_code == 429


# --- secrets & static ------------------------------------------------------------------


def test_credentials_repr_hides_secrets():
    credentials = Credentials(llm_provider="gemini", llm_api_key="sk-visible?", tavily_api_key="tv-visible?")

    assert "visible" not in repr(credentials)
    assert "visible" not in str(credentials)


def test_spa_fallback_serves_index(demo_dir, tmp_path):
    web = tmp_path / "dist"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_text("<html>app</html>")
    (web / "assets" / "app.js").write_text("console.log(1)")
    client = make_client(demo_dir, web_dist=web)

    assert client.get("/").text == "<html>app</html>"
    assert client.get("/brief/anything").text == "<html>app</html>"
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/api/nope").status_code == 404
