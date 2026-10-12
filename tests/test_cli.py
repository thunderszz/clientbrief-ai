"""The CLI shows real per-node progress from the graph stream."""

from __future__ import annotations

from app import main as cli


class FakeGraph:
    def stream(self, workflow_input, stream_mode):
        assert stream_mode == ["updates", "values"]
        yield "updates", {"research_signals": {"external_signals": [{"signal_type": "financial"}]}}
        yield "values", {"final_brief": "partial"}
        yield "updates", {"compose": {"reliability_report": {"score": 64}, "sources": ["a"]}}
        yield "values", {"final_brief": "# Done", "brief_reliability": 3}


def test_run_workflow_reports_each_node(capsys):
    result = cli._run_workflow({"company_name": "Acme"}, graph=FakeGraph())

    out = capsys.readouterr().out
    assert result == {"final_brief": "# Done", "brief_reliability": 3}
    assert "Strategic signals" in out and "1 signal · financial" in out
    assert "Compose brief" in out and "reliability 64/100" in out


def test_reliability_line_shows_score():
    line = cli._reliability_line({"brief_reliability": 4, "reliability_report": {"score": 78}})

    assert line == "Brief reliability: 4/5 (78/100)"
