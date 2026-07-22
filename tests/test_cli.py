"""Tests for the harness-selection dispatch in the CLI."""

from __future__ import annotations

import json

import pytest

from olist_code import cli, config
from olist_code.models import AdapterConfig, ModelConfig


def make_config(harness: str = "both", **overrides) -> AdapterConfig:
    defaults = dict(
        base_url="http://localhost:8000",
        api_key="sk-olist-x",
        sso=None,
        models=ModelConfig(opus="glm-4.6", sonnet="glm-4.5", haiku=None),
        port=3080,
        harness=harness,
    )
    defaults.update(overrides)
    return AdapterConfig(**defaults)


@pytest.fixture
def claude_settings_file(tmp_path, monkeypatch):
    path = tmp_path / ".claude" / "settings.json"
    monkeypatch.setattr(config, "CLAUDE_SETTINGS_FILE", path)
    return path


@pytest.fixture
def opencode_settings_file(tmp_path, monkeypatch):
    path = tmp_path / ".config" / "opencode" / "opencode.json"
    monkeypatch.setattr(config, "OPENCODE_SETTINGS_FILE", path)
    monkeypatch.setattr(config, "OPENCODE_SETTINGS_FILE_JSONC", path.with_suffix(".jsonc"))
    return path


class TestApplySettings:
    def test_claude_only_does_not_touch_opencode(self, claude_settings_file, opencode_settings_file):
        cli._apply_settings(make_config(harness="claude"))

        assert claude_settings_file.exists()
        assert not opencode_settings_file.exists()

    def test_opencode_only_does_not_touch_claude(self, claude_settings_file, opencode_settings_file):
        cli._apply_settings(make_config(harness="opencode"))

        assert not claude_settings_file.exists()
        assert opencode_settings_file.exists()

    def test_both_touches_both(self, claude_settings_file, opencode_settings_file):
        cli._apply_settings(make_config(harness="both"))

        assert claude_settings_file.exists()
        assert opencode_settings_file.exists()


class TestResolveHarness:
    def test_defaults_to_both_when_nothing_set(self):
        assert cli._resolve_harness(None, None) == "both"

    def test_explicit_flag_wins(self):
        existing = make_config(harness="both")
        assert cli._resolve_harness("claude", existing) == "claude"

    def test_falls_back_to_saved_config(self):
        existing = make_config(harness="opencode")
        assert cli._resolve_harness(None, existing) == "opencode"

    def test_invalid_value_exits(self):
        with pytest.raises(Exception):
            cli._resolve_harness("not-a-harness", None)


def test_json_dump_smoke(claude_settings_file):
    # Sanity check that the fixture path plumbing above actually produces valid JSON.
    cli._apply_settings(make_config(harness="claude"))
    json.loads(claude_settings_file.read_text())
