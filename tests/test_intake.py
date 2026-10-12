"""Intake post-processing guards against the model's own mistakes."""

from __future__ import annotations

from agents import meeting_intake_agent
from agents.meeting_intake_agent import LowConfidenceField, MeetingIntakeResult


def _patch_llm(monkeypatch, **fields):
    result = MeetingIntakeResult(confirmation_summary="ok", **fields)
    monkeypatch.setattr(meeting_intake_agent, "structured_extract", lambda **kwargs: result)


def test_domain_stated_in_text_is_kept_even_if_flagged(monkeypatch):
    _patch_llm(
        monkeypatch,
        company_name="Bending Spoons",
        domain="bendingspoons.com",
        low_confidence_fields=[
            LowConfidenceField(
                field_name="domain",
                extracted_value="bendingspoons.com",
                reason="Explicitly provided in the text.",
            )
        ],
    )

    out = meeting_intake_agent.run("Call with Bending Spoons (BendingSpoons.com) finance lead")

    assert out["domain"] == "bendingspoons.com"
    assert all(lc["field"] != "domain" for lc in out["intake_low_confidence_fields"])


def test_guessed_domain_is_still_nulled(monkeypatch):
    _patch_llm(
        monkeypatch,
        company_name="Acme",
        domain="acme.io",
        low_confidence_fields=[
            LowConfidenceField(field_name="domain", extracted_value="acme.io", reason="guessed")
        ],
    )

    out = meeting_intake_agent.run("Call with Acme's CFO next week")

    assert out["domain"] == ""
    assert any(lc["field"] == "domain" for lc in out["intake_low_confidence_fields"])
