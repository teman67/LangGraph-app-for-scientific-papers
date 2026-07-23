"""
Thin wrapper around the Anthropic and OpenAI SDKs.

Provides:
- `run_llm`            : free-text completion (kept for simple/manual use).
- `run_llm_structured` : schema-enforced completion — Anthropic via forced
  tool-use, OpenAI via `response_format={"type": "json_schema", ...}`. This
  is far more reliable than asking for "JSON only" in the prompt and
  regexing it out of prose.

Unlike the main app, calls here are never cached — every extraction hits
the provider API fresh.
"""
from __future__ import annotations

import json
from typing import Any, Dict

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",    # balanced; use claude-fable-5 for max accuracy, claude-haiku-4-5 for speed
    "openai": "gpt-5.6-terra",         # balanced; use gpt-5.6-sol for max accuracy, gpt-5.6-luna for speed
}


class InvalidAPIKeyError(RuntimeError):
    """Raised when the provider rejects the API key (HTTP 401)."""


def _check_auth_error(provider: str, exc: Exception) -> None:
    """Re-raise `exc` as an InvalidAPIKeyError if it's an authentication failure.

    Detected via `status_code == 401` (both the anthropic and openai SDKs set
    this on their APIStatusError subclasses) rather than importing each SDK's
    AuthenticationError class, since these providers are imported lazily and
    a user may only have one of the two packages installed.
    """
    status = getattr(exc, "status_code", None)
    if status == 401 or type(exc).__name__ == "AuthenticationError":
        label = "Claude (Anthropic)" if provider == "anthropic" else "OpenAI"
        raise InvalidAPIKeyError(
            f"{label} rejected the API key (401 Unauthorized). Double-check the key "
            "in the sidebar — it may be missing, revoked, or copied for the wrong provider."
        ) from exc


# ----------------------------------------------------------------------------
# Free-text completion
# ----------------------------------------------------------------------------
def run_llm(
    provider: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    reasoning_effort: str = "high",
) -> str:
    if not api_key:
        raise ValueError(f"No API key provided for {provider}.")

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        try:
            resp = client.messages.create(
                model=model or DEFAULT_MODELS["anthropic"],
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception as e:  # noqa: BLE001
            _check_auth_error("anthropic", e)
            raise
        text = "".join(block.text for block in resp.content if block.type == "text")

    elif provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        _model = model or DEFAULT_MODELS["openai"]
        if _model.startswith(("gpt-5.", "o1", "o3", "o4")):
            # These models don't accept temperature; use seed + reasoning_effort.
            _extra: dict = {"seed": 42, "reasoning_effort": reasoning_effort}
        else:
            _extra = {"temperature": 0, "seed": 42}
        try:
            resp = client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **_extra,
            )
        except Exception as e:  # noqa: BLE001
            _check_auth_error("openai", e)
            raise
        text = resp.choices[0].message.content or ""

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return text


# ----------------------------------------------------------------------------
# Structured (schema-enforced) completion
# ----------------------------------------------------------------------------
def run_llm_structured(
    provider: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
    tool_name: str = "emit_records",
    reasoning_effort: str = "high",
) -> Dict[str, Any]:
    """Returns the parsed JSON object matching `json_schema` (a JSON-schema dict
    for a top-level object, e.g. {"type": "object", "properties": {"records": ...}}).
    """
    if not api_key:
        raise ValueError(f"No API key provided for {provider}.")

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        try:
            resp = client.messages.create(
                model=model or DEFAULT_MODELS["anthropic"],
                max_tokens=8192,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                tools=[
                    {
                        "name": tool_name,
                        "description": "Emit the extracted records matching the required schema.",
                        "input_schema": json_schema,
                    }
                ],
                tool_choice={"type": "tool", "name": tool_name},
            )
        except Exception as e:  # noqa: BLE001
            _check_auth_error("anthropic", e)
            raise
        parsed = None
        for block in resp.content:
            if block.type == "tool_use" and block.name == tool_name:
                parsed = block.input
                break
        if parsed is None:
            raise RuntimeError("Model did not return a tool_use block with the expected schema.")

    elif provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        _model = model or DEFAULT_MODELS["openai"]
        if _model.startswith(("gpt-5.", "o1", "o3", "o4")):
            _extra: dict = {"seed": 42, "reasoning_effort": reasoning_effort}
        else:
            _extra = {"temperature": 0, "seed": 42}
        try:
            resp = client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": tool_name, "strict": True, "schema": json_schema},
                },
                **_extra,
            )
        except Exception as e:  # noqa: BLE001
            _check_auth_error("openai", e)
            raise
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    return parsed
