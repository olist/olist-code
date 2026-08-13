"""Configuration file I/O utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import AdapterConfig

CONFIG_DIR = Path.home() / ".olist-code-adapter"
CONFIG_FILE = CONFIG_DIR / "config.json"
CLAUDE_SETTINGS_FILE = Path.home() / ".claude" / "settings.json"
OPENCODE_SETTINGS_FILE = Path.home() / ".config" / "opencode" / "opencode.json"
OPENCODE_SETTINGS_FILE_JSONC = OPENCODE_SETTINGS_FILE.with_suffix(".jsonc")
OPENCODE_PROVIDER_ID = "olist-ai-gateway"
OPENCODE_SCHEMA_URL = "https://opencode.ai/config.json"

# Matches // and /* */ comments that are not inside a "..." string.
_JSONC_COMMENT_RE = re.compile(r'("(?:\\.|[^"\\])*")|(//[^\n]*|/\*.*?\*/)', re.DOTALL)


def _strip_jsonc_comments(text: str) -> str:
    return _JSONC_COMMENT_RE.sub(lambda m: m.group(1) or "", text)


def _resolve_opencode_settings_file() -> Path:
    """Prefer an existing opencode.jsonc over creating a separate opencode.json."""
    if OPENCODE_SETTINGS_FILE_JSONC.exists():
        return OPENCODE_SETTINGS_FILE_JSONC
    return OPENCODE_SETTINGS_FILE


def _load_jsonc(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.loads(_strip_jsonc_comments(f.read()))


_CLAUDE_ENV_KEYS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
)


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> AdapterConfig | None:
    if not CONFIG_FILE.exists():
        return None
    with open(CONFIG_FILE) as f:
        data: dict[str, Any] = json.load(f)
    return AdapterConfig(**data)


def save_config(config: AdapterConfig) -> None:
    ensure_config_dir()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config.model_dump(), f, indent=2)


def claude_env(config: AdapterConfig) -> dict[str, str]:
    """Env vars that point Claude Code at the local proxy.

    Injected into the `olist-code claude` child process instead of being written to
    the user's global settings.json, so a plain `claude` keeps using their own account.
    """
    return {
        "ANTHROPIC_BASE_URL": f"http://localhost:{config.port}",
        "ANTHROPIC_AUTH_TOKEN": "default",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": config.models.opus,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": config.models.sonnet or config.models.opus,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": config.models.haiku or config.models.sonnet or config.models.opus,
    }


def update_claude_settings(config: AdapterConfig) -> None:
    """Point every `claude` at the proxy by writing the env into the global settings.

    Reaches surfaces that never go through `olist-code claude` — the IDE extensions,
    the desktop app, `claude -p` in scripts — at the cost of taking over the user's
    plain `claude`. See claude_env() for the opt-out.
    """
    CLAUDE_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if CLAUDE_SETTINGS_FILE.exists():
        with open(CLAUDE_SETTINGS_FILE) as f:
            existing = json.load(f)

    env = existing.get("env")
    if not isinstance(env, dict):
        env = {}
    existing["env"] = {**env, **claude_env(config)}

    with open(CLAUDE_SETTINGS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def restore_claude_settings() -> None:
    """Undo update_claude_settings(), leaving any other settings untouched."""
    if not CLAUDE_SETTINGS_FILE.exists():
        return

    with open(CLAUDE_SETTINGS_FILE) as f:
        existing: dict[str, Any] = json.load(f)

    env = existing.get("env")
    if isinstance(env, dict):
        for key in _CLAUDE_ENV_KEYS:
            env.pop(key, None)
        if not env:
            existing.pop("env", None)

    with open(CLAUDE_SETTINGS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def update_opencode_settings(config: AdapterConfig) -> None:
    target = _resolve_opencode_settings_file()
    target.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if target.exists():
        existing = _load_jsonc(target)

    existing.setdefault("$schema", OPENCODE_SCHEMA_URL)

    if "provider" not in existing or not isinstance(existing["provider"], dict):
        existing["provider"] = {}

    models: dict[str, Any] = {config.models.opus: {}}
    if config.models.sonnet:
        models[config.models.sonnet] = {}
    if config.models.haiku:
        models[config.models.haiku] = {}

    existing["provider"][OPENCODE_PROVIDER_ID] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Olist AI Gateway",
        "options": {
            "baseURL": f"http://localhost:{config.port}/v1",
            "apiKey": "default",
        },
        "models": models,
    }

    with open(target, "w") as f:
        json.dump(existing, f, indent=2)


def restore_opencode_settings() -> None:
    """Undo update_opencode_settings(), leaving any other providers/keys untouched."""
    target = _resolve_opencode_settings_file()
    if not target.exists():
        return

    existing = _load_jsonc(target)

    provider = existing.get("provider")
    if isinstance(provider, dict):
        provider.pop(OPENCODE_PROVIDER_ID, None)
        if not provider:
            existing.pop("provider", None)

    if not existing or existing == {"$schema": OPENCODE_SCHEMA_URL}:
        target.unlink()
        return

    with open(target, "w") as f:
        json.dump(existing, f, indent=2)


def update_claude_json() -> None:
    claude_json = Path.home() / ".claude.json"
    if not claude_json.exists():
        return
    try:
        with open(claude_json) as f:
            data: dict[str, Any] = json.load(f)
        data["hasCompletedOnboarding"] = True
        with open(claude_json, "w") as f:
            json.dump(data, f, indent=2)
    except (OSError, json.JSONDecodeError):
        pass
