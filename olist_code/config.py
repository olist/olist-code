"""Configuration file I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import AdapterConfig

CONFIG_DIR = Path.home() / ".olist-code-adapter"
CONFIG_FILE = CONFIG_DIR / "config.json"
CLAUDE_SETTINGS_FILE = Path.home() / ".claude" / "settings.json"


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


def update_claude_settings(config: AdapterConfig) -> None:
    CLAUDE_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if CLAUDE_SETTINGS_FILE.exists():
        with open(CLAUDE_SETTINGS_FILE) as f:
            existing = json.load(f)

    if "env" not in existing:
        existing["env"] = {}

    env = existing["env"]
    if not isinstance(env, dict):
        env = {}
        existing["env"] = env

    base_url = f"http://localhost:{config.port}"
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_AUTH_TOKEN"] = "default"
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = config.models.opus
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = config.models.sonnet or config.models.opus
    if config.models.haiku:
        env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = config.models.haiku

    with open(CLAUDE_SETTINGS_FILE, "w") as f:
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
