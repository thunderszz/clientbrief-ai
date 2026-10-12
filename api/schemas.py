"""Request models. Secrets are ``SecretStr`` so they never show up in logs or reprs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from services.llm_client import LLMSettings
from services.providers import PROVIDER_PRESETS, settings_for_preset

_MODEL_NAME = r"^[A-Za-z0-9._:/-]{1,100}$"
_Short = Field(default=None, max_length=200)


class Credentials(BaseModel):
    """Visitor-supplied keys ("bring your own key"). Held for one request only."""

    model_config = ConfigDict(extra="forbid")

    llm_provider: Literal["gemini", "groq", "deepseek", "openrouter"] = "gemini"
    llm_api_key: SecretStr | None = Field(default=None, max_length=400)
    llm_model_fast: str | None = Field(default=None, pattern=_MODEL_NAME)
    llm_model_reasoning: str | None = Field(default=None, pattern=_MODEL_NAME)
    tavily_api_key: SecretStr | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def _model_required_for_some_providers(self) -> Credentials:
        if self.has_llm_key() and not (
            self.llm_model_fast or PROVIDER_PRESETS[self.llm_provider]["model_fast"]
        ):
            raise ValueError(f"{self.llm_provider} needs llm_model_fast")
        return self

    def has_llm_key(self) -> bool:
        return bool(self.llm_api_key and self.llm_api_key.get_secret_value().strip())

    def llm_settings(self) -> LLMSettings | None:
        if not self.has_llm_key():
            return None
        return settings_for_preset(
            self.llm_provider,
            self.llm_api_key.get_secret_value().strip(),
            self.llm_model_fast,
            self.llm_model_reasoning,
        )

    def tavily_key(self) -> str | None:
        if not self.tavily_api_key:
            return None
        return self.tavily_api_key.get_secret_value().strip() or None


class MeetingFields(BaseModel):
    """Structured meeting input, as reviewed by the user. Lengths are capped."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    company_name: str = Field(min_length=1, max_length=200)
    domain: str = Field(min_length=1, max_length=253)
    meeting_type: str = Field(min_length=1, max_length=50)
    stakeholder_role: str = Field(min_length=1, max_length=200)
    meeting_goal: str = Field(min_length=1, max_length=1000)
    contact_name: str | None = _Short
    contact_title: str | None = _Short
    meeting_stage: str | None = Field(default=None, max_length=50)
    relationship_status: str | None = Field(default=None, max_length=50)
    previous_meeting_outcome: str | None = Field(default=None, max_length=50)
    meeting_notes_summary: str | None = Field(default=None, max_length=3000)
    user_role: str | None = _Short
    user_company: str | None = _Short
    user_function: str | None = _Short
    engagement_type: str | None = Field(default=None, max_length=50)
    desired_outcome: str | None = Field(default=None, max_length=1000)
    success_definition: str | None = Field(default=None, max_length=1000)

    def to_workflow_input(self) -> dict[str, Any]:
        return {key: (value or None) for key, value in self.model_dump().items()}


class IntakeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    credentials: Credentials | None = None


class RunRequest(BaseModel):
    fields: MeetingFields
    raw_text: str | None = Field(default=None, max_length=4000)
    credentials: Credentials | None = None
