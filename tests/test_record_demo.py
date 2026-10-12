"""Demo records are well-formed and never contain secrets."""

from __future__ import annotations

import pytest

from scripts.record_demo import assert_no_secrets, build_record


def _record(**overrides):
    kwargs = {
        "slug": "acme",
        "title": "Acme intro",
        "subtitle": "CFO",
        "company": "Acme",
        "input_text": "Meeting with Acme",
        "intake": {"company_name": "Acme"},
        "events": [{"type": "run_started", "t": 0.0}, {"type": "result", "t": 3.0, "result": {}}],
        "models": {"fast": "f", "reasoning": "r"},
        "order": 2,
    }
    kwargs.update(overrides)
    return build_record(**kwargs)


def test_build_record_shape():
    record = _record()

    assert record["slug"] == "acme"
    assert record["order"] == 2
    assert record["events"][-1]["type"] == "result"
    assert record["recorded_at"].endswith("+00:00")


def test_build_record_rejects_run_without_result():
    with pytest.raises(ValueError, match="result"):
        _record(events=[{"type": "run_started", "t": 0.0}, {"type": "error", "t": 1.0}])


def test_assert_no_secrets_passes_clean_record():
    assert_no_secrets(_record(), ["sk-secret-123"])


def test_assert_no_secrets_catches_leak():
    record = _record(intake={"company_name": "Acme", "note": "key sk-secret-123"})

    with pytest.raises(ValueError, match="secret"):
        assert_no_secrets(record, ["sk-secret-123", ""])
