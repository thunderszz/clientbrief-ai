"""Shared helper for structured LLM calls via Google Gemini.

Provides a single ``structured_extract()`` function that:
- sends a system + user prompt to a Google Gemini model
- parses the response into a caller-supplied Pydantic model
- returns ``None`` on any failure (network, parse, refusal) so callers
  can fall back to safe empty defaults without crashing

The helper is intentionally thin.  It does NOT manage retries, streaming,
or multi-turn conversation — those can be added when needed.

Model selection:
    The default model is read from the ``GEMINI_MODEL`` env var, falling
    back to ``"gemini-3.1-flash-lite-preview"``.
    To switch models per-agent later, callers can pass ``model=`` explicitly.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TypeVar

from google import genai
from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
_DEFAULT_TEMPERATURE = 0.1           # low temp for extraction tasks


def _get_client() -> genai.Client:
    """Lazily build a Gemini client (reads GEMINI_API_KEY from env)."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Export it as an environment variable or add it to .env."
        )
    return genai.Client(api_key=api_key)


def structured_extract(
    *,
    system_prompt: str,
    user_prompt: str,
    response_model: type[T],
    model: str = _DEFAULT_MODEL,
    temperature: float = _DEFAULT_TEMPERATURE,
) -> T | None:
    """Call a Gemini model and parse the response into *response_model*.

    Parameters
    ----------
    system_prompt:
        Instructions that frame the extraction task.
    user_prompt:
        The concrete evidence / question to process.
    response_model:
        A Pydantic ``BaseModel`` subclass.  The Gemini structured-output
        feature constrains the response to match this schema.
    model:
        Gemini model name (default from ``GEMINI_MODEL`` env var).
    temperature:
        Sampling temperature (default 0.1 — near-deterministic).

    Returns
    -------
    T | None
        A populated Pydantic model on success, or ``None`` on any failure.
    """
    try:
        client = _get_client()

        # Build the combined prompt with system instructions + user content.
        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config={
                "system_instruction": system_prompt,
                "temperature": temperature,
                "response_mime_type": "application/json",
                "response_json_schema": response_model.model_json_schema(),
            },
        )

        # Check for empty / blocked response
        if not response.text:
            logger.warning("Gemini returned empty response for model=%s", model)
            return None

        # Parse the JSON text into the Pydantic model
        parsed = response_model.model_validate_json(response.text)
        return parsed

    except json.JSONDecodeError as exc:
        logger.warning("structured_extract JSON parse failed (%s): %s", model, exc)
        return None
    except Exception as exc:
        logger.warning("structured_extract failed (%s): %s", model, exc)
        return None
