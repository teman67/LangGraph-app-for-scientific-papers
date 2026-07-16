"""
Thin wrapper around the Anthropic and OpenAI SDKs.

Provides:
- `run_llm`            : free-text completion (kept for simple/manual use).
- `run_llm_structured` : schema-enforced completion — Anthropic via forced
  tool-use, OpenAI via `response_format={"type": "json_schema", ...}`. This
  is far more reliable than asking for "JSON only" in the prompt and
  regexing it out of prose.

Both are wrapped by a small on-disk cache so re-running the pipeline after
fixing one paper doesn't re-bill/re-call the API for documents that didn't
change.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4.1",
}

DEFAULT_CACHE_DIR = ".llm_cache"


# ----------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------
def _cache_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _cache_get(cache_dir: str, key: str) -> Optional[Any]:
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


def _cache_set(cache_dir: str, key: str, value: Any) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    path = Path(cache_dir) / f"{key}.json"
    path.write_text(json.dumps(value))


# ----------------------------------------------------------------------------
# Free-text completion
# ----------------------------------------------------------------------------
def run_llm(
    provider: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> str:
    if not api_key:
        raise ValueError(f"No API key provided for {provider}.")

    key = _cache_key("text", provider, model, system_prompt, user_prompt)
    if use_cache:
        cached = _cache_get(cache_dir, key)
        if cached is not None:
            return cached["text"]

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model or DEFAULT_MODELS["anthropic"],
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")

    elif provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model or DEFAULT_MODELS["openai"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        text = resp.choices[0].message.content or ""

    else:
        raise ValueError(f"Unknown provider: {provider}")

    if use_cache:
        _cache_set(cache_dir, key, {"text": text})
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
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Dict[str, Any]:
    """Returns the parsed JSON object matching `json_schema` (a JSON-schema dict
    for a top-level object, e.g. {"type": "object", "properties": {"records": ...}}).
    """
    if not api_key:
        raise ValueError(f"No API key provided for {provider}.")

    cache_key_material = json.dumps(json_schema, sort_keys=True)
    key = _cache_key("structured", provider, model, system_prompt, user_prompt, cache_key_material)
    if use_cache:
        cached = _cache_get(cache_dir, key)
        if cached is not None:
            return cached

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
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
        resp = client.chat.completions.create(
            model=model or DEFAULT_MODELS["openai"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": tool_name, "strict": True, "schema": json_schema},
            },
        )
        content = resp.choices[0].message.content or "{}"
        parsed = json.loads(content)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    if use_cache:
        _cache_set(cache_dir, key, parsed)
    return parsed
