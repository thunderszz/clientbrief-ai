"""Known LLM providers that visitors can use with their own key.

Visitors pick a preset; they never supply a base URL. That keeps the public
server from being pointed at arbitrary hosts (SSRF). Operators can still use
any OpenAI-compatible endpoint through env vars (see ``services/llm_client.py``).
"""

from __future__ import annotations

from typing import Any

from services.llm_client import GEMINI_DEFAULT_FAST, GEMINI_DEFAULT_REASONING, LLMSettings

PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "gemini": {
        "label": "Google Gemini",
        "provider": "gemini",
        "base_url": None,
        "model_fast": GEMINI_DEFAULT_FAST,
        "model_reasoning": GEMINI_DEFAULT_REASONING,
        "key_url": "https://aistudio.google.com/app/apikey",
        "free_tier": True,
    },
    "groq": {
        "label": "Groq",
        "provider": "openai_compatible",
        "base_url": "https://api.groq.com/openai/v1",
        "model_fast": "openai/gpt-oss-20b",
        "model_reasoning": "openai/gpt-oss-120b",
        "key_url": "https://console.groq.com/keys",
        "free_tier": True,
    },
    "deepseek": {
        "label": "DeepSeek",
        "provider": "openai_compatible",
        "base_url": "https://api.deepseek.com",
        "model_fast": "deepseek-v4-flash",
        "model_reasoning": "deepseek-v4-flash",
        "key_url": "https://platform.deepseek.com/api_keys",
        "free_tier": False,
    },
    "openrouter": {
        "label": "OpenRouter",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "model_fast": None,
        "model_reasoning": None,
        "key_url": "https://openrouter.ai/keys",
        "free_tier": False,
    },
}


def settings_for_preset(
    preset_id: str,
    api_key: str,
    model_fast: str | None = None,
    model_reasoning: str | None = None,
) -> LLMSettings:
    """Build settings from a preset; raises ``ValueError`` when a model is required but missing."""
    preset = PROVIDER_PRESETS.get(preset_id)
    if preset is None:
        raise ValueError(f"Unknown provider {preset_id!r}")
    fast = model_fast or preset["model_fast"]
    if not fast:
        raise ValueError(f"{preset['label']} needs a model name")
    return LLMSettings(
        provider=preset["provider"],
        api_key=api_key,
        model_fast=fast,
        model_reasoning=model_reasoning or preset["model_reasoning"] or fast,
        base_url=preset["base_url"],
    )


def public_presets() -> list[dict[str, Any]]:
    """Presets as served to the UI."""
    return [
        {
            "id": preset_id,
            "label": p["label"],
            "model_fast": p["model_fast"],
            "model_reasoning": p["model_reasoning"],
            "key_url": p["key_url"],
            "free_tier": p["free_tier"],
            "needs_model": p["model_fast"] is None,
        }
        for preset_id, p in PROVIDER_PRESETS.items()
    ]
