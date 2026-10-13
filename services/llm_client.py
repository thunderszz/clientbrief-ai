"""Provider-agnostic helper for structured LLM calls.

Every agent goes through a single ``structured_extract()`` function that:
- sends a system + user prompt to the configured model
- parses the response into a caller-supplied Pydantic model
- returns ``None`` on any failure so callers can fall back to safe empty
  defaults without crashing

Providers
---------
``LLM_PROVIDER=gemini`` (default)
    Google Gemini via ``google-genai``, with native JSON-schema output.
    Key: ``GEMINI_API_KEY``.
``LLM_PROVIDER=openai_compatible``
    Any OpenAI-compatible endpoint (Groq, DeepSeek, OpenRouter, Ollama...).
    Keys: ``LLM_API_KEY``, ``LLM_BASE_URL``, ``LLM_MODEL_FAST`` (required).
    The JSON schema goes into the system prompt and the reply is validated
    with Pydantic, with one repair retry on validation errors.

Model tiers
-----------
Agents ask for a tier, not a model. ``Tier.FAST`` covers extraction and
parsing; ``Tier.REASONING`` covers the agents that synthesise (audience,
pain points, strategy). ``LLM_MODEL_FAST`` / ``LLM_MODEL_REASONING`` pick
the models.

Resilience
----------
Free tiers rate-limit hard, so 429/5xx errors are retried with backoff on
the requested model, then the call falls back to the other tier's model.
A busy model degrades the brief instead of failing it.

Per-request credentials
-----------------------
``use_llm_settings()`` overrides the env configuration for the current
context (used by the API for "bring your own key"). LangGraph copies the
context into parallel nodes, so the override reaches every agent of a run.

Call records
------------
Inside ``collect_llm_calls()`` every call appends a record (model, tier,
latency, attempts, fallback) that the workflow attaches to node events.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_DEFAULT_TEMPERATURE = 0.1  # low temp for extraction tasks
GEMINI_DEFAULT_FAST = "gemini-3.5-flash-lite"
GEMINI_DEFAULT_REASONING = "gemini-3.6-flash"

_PRIMARY_ATTEMPTS = 3
_FALLBACK_ATTEMPTS = 2
_REQUEST_TIMEOUT_S = 45  # a stuck call becomes a retryable timeout
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_RETRYABLE_MARKERS = ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE_EXCEEDED")

_sleep = time.sleep  # patched in tests


class Tier(StrEnum):
    """Which class of model an agent needs."""

    FAST = "fast"
    REASONING = "reasoning"


@dataclass(frozen=True)
class LLMSettings:
    """Resolved provider configuration for one run."""

    provider: str  # "gemini" | "openai_compatible"
    api_key: str = field(repr=False)
    model_fast: str
    model_reasoning: str
    base_url: str | None = None

    def model_for(self, tier: Tier) -> str:
        return self.model_reasoning if tier is Tier.REASONING else self.model_fast

    def other_model(self, tier: Tier) -> str:
        return self.model_fast if tier is Tier.REASONING else self.model_reasoning


_settings_override: ContextVar[LLMSettings | None] = ContextVar(
    "llm_settings_override", default=None
)
_call_log: ContextVar[list[dict[str, Any]] | None] = ContextVar("llm_call_log", default=None)


def settings_from_env() -> LLMSettings:
    """Build settings from environment variables.

    Raises ``EnvironmentError`` when the configuration is incomplete.
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip() or "gemini"

    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise OSError(
                "GEMINI_API_KEY is not set. Export it as an environment variable or add it to .env."
            )
        return LLMSettings(
            provider="gemini",
            api_key=api_key,
            model_fast=os.environ.get("LLM_MODEL_FAST") or GEMINI_DEFAULT_FAST,
            model_reasoning=os.environ.get("LLM_MODEL_REASONING") or GEMINI_DEFAULT_REASONING,
        )

    if provider == "openai_compatible":
        api_key = os.environ.get("LLM_API_KEY", "")
        model_fast = os.environ.get("LLM_MODEL_FAST", "")
        if not api_key:
            raise OSError("LLM_API_KEY is not set (required for LLM_PROVIDER=openai_compatible).")
        if not model_fast:
            raise OSError("LLM_MODEL_FAST is not set (required for LLM_PROVIDER=openai_compatible).")
        return LLMSettings(
            provider="openai_compatible",
            api_key=api_key,
            model_fast=model_fast,
            model_reasoning=os.environ.get("LLM_MODEL_REASONING") or model_fast,
            base_url=os.environ.get("LLM_BASE_URL") or None,
        )

    raise OSError(f"Unknown LLM_PROVIDER {provider!r}: use 'gemini' or 'openai_compatible'.")


def current_settings() -> LLMSettings:
    """Return the per-request override if one is active, else the env configuration."""
    override = _settings_override.get()
    return override if override is not None else settings_from_env()


@contextmanager
def use_llm_settings(settings: LLMSettings | None) -> Iterator[None]:
    """Use *settings* for every LLM call in this context. ``None`` is a no-op."""
    if settings is None:
        yield
        return
    token = _settings_override.set(settings)
    try:
        yield
    finally:
        _settings_override.reset(token)


@contextmanager
def collect_llm_calls() -> Iterator[list[dict[str, Any]]]:
    """Collect a record of every LLM call made inside this context."""
    records: list[dict[str, Any]] = []
    token = _call_log.set(records)
    try:
        yield records
    finally:
        _call_log.reset(token)


def _gemini_generate(
    settings: LLMSettings,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    temperature: float,
) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=settings.api_key,
        http_options=types.HttpOptions(timeout=_REQUEST_TIMEOUT_S * 1000),
    )
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": system_prompt,
            "temperature": temperature,
            "response_mime_type": "application/json",
            "response_json_schema": schema,
        },
    )
    if not response.text:
        raise ValueError(f"Gemini returned an empty response (model={model})")
    return response.text


def _openai_compatible_generate(
    settings: LLMSettings,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    temperature: float,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, max_retries=0, timeout=_REQUEST_TIMEOUT_S)
    schema_instruction = (
        "\n\nRespond with a single JSON object that conforms to this JSON schema. "
        "Output JSON only, no prose and no code fences.\n" + json.dumps(schema)
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt + schema_instruction},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"Provider returned an empty response (model={model})")
    return content


def _transport(
    settings: LLMSettings,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    temperature: float,
) -> str:
    """Send one request to the configured provider and return the raw JSON text."""
    if settings.provider == "gemini":
        return _gemini_generate(settings, model, system_prompt, user_prompt, schema, temperature)
    return _openai_compatible_generate(
        settings, model, system_prompt, user_prompt, schema, temperature
    )


def _is_retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int) and code in _RETRYABLE_STATUS:
        return True
    name = type(exc).__name__
    if "Timeout" in name or "Connection" in name:
        return True
    message = str(exc)
    return any(marker in message for marker in _RETRYABLE_MARKERS)


def structured_extract(
    *,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    tier: Tier = Tier.FAST,
    model: str | None = None,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> T | None:
    """Call the configured model and parse the response into *response_model*.

    Parameters
    ----------
    tier:
        Which model class to use (ignored when *model* is given).
    model:
        Explicit model override. Disables the cross-tier fallback.

    Returns
    -------
    T | None
        A populated Pydantic model on success, or ``None`` on any failure.
    """
    try:
        settings = current_settings()
    except OSError as exc:
        logger.warning("structured_extract: %s", exc)
        return None

    primary = model or settings.model_for(tier)
    chain: list[tuple[str, bool]] = [(primary, False)] * _PRIMARY_ATTEMPTS
    fallback = None if model else settings.other_model(tier)
    if fallback and fallback != primary:
        chain += [(fallback, True)] * _FALLBACK_ATTEMPTS

    schema = response_model.model_json_schema()
    started = time.monotonic()
    record: dict[str, Any] = {
        "model": primary,
        "tier": tier.value,
        "ok": False,
        "latency_s": 0.0,
        "fallback": False,
        "attempts": 0,
    }

    def send(model_name: str, prompt: str) -> str:
        record["attempts"] += 1
        return _transport(settings, model_name, system_prompt, prompt, schema, temperature)

    try:
        text = _send_with_retries(chain, send, user_prompt, record)
        if text is not None:
            parsed = _parse_or_repair(text, response_model, send, user_prompt, record["model"])
            record["ok"] = parsed is not None
            return parsed
        return None
    finally:
        record["latency_s"] = round(time.monotonic() - started, 2)
        log = _call_log.get()
        if log is not None:
            log.append(record)


def _send_with_retries(chain, send, user_prompt: str, record: dict[str, Any]) -> str | None:
    """Walk the (model, is_fallback) chain until one call succeeds."""
    same_model_failures = 0
    for index, (model_name, is_fallback) in enumerate(chain):
        record["model"] = model_name
        record["fallback"] = is_fallback
        try:
            return send(model_name, user_prompt)
        except Exception as exc:
            if not _is_retryable(exc):
                logger.warning("structured_extract failed (%s): %s", model_name, exc)
                return None
            logger.info("structured_extract retryable error (%s): %s", model_name, exc)
            next_model = chain[index + 1][0] if index + 1 < len(chain) else None
            if next_model is None:
                logger.warning("structured_extract gave up after %d attempts", len(chain))
                return None
            if next_model == model_name:
                same_model_failures += 1
                _sleep(float(2 ** (same_model_failures - 1)))
            else:
                same_model_failures = 0
    return None


def _parse_or_repair(text: str, response_model: type[T], send, user_prompt: str, model_name: str):
    """Validate *text*; on failure ask the same model once to fix its output."""
    try:
        return response_model.model_validate_json(text)
    except ValidationError as exc:
        repair_prompt = (
            f"{user_prompt}\n\nYour previous response failed validation:\n{exc}\n"
            "Return only the corrected JSON object."
        )
    try:
        repaired = send(model_name, repair_prompt)
        return response_model.model_validate_json(repaired)
    except Exception as exc:
        logger.warning("structured_extract repair failed (%s): %s", model_name, exc)
        return None
