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

        assert not opencode_settings_file.exists()

    def test_opencode_only_writes_the_provider(self, claude_settings_file, opencode_settings_file):
        cli._apply_settings(make_config(harness="opencode"))

        assert opencode_settings_file.exists()

    @pytest.mark.parametrize("harness", ["claude", "opencode", "both"])
    def test_never_creates_the_global_claude_settings(self, harness, claude_settings_file, opencode_settings_file):
        cli._apply_settings(make_config(harness=harness))

        assert not claude_settings_file.exists()

    def test_strips_env_vars_left_by_older_versions(self, claude_settings_file, opencode_settings_file):
        claude_settings_file.parent.mkdir(parents=True)
        claude_settings_file.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://localhost:3080", "OTHER": "1"}})
        )

        cli._apply_settings(make_config(harness="claude"))

        data = json.loads(claude_settings_file.read_text())
        assert data["env"] == {"OTHER": "1"}


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


def test_json_dump_smoke(opencode_settings_file):
    # Sanity check that the fixture path plumbing above actually produces valid JSON.
    cli._apply_settings(make_config(harness="opencode"))
    json.loads(opencode_settings_file.read_text())


class TestChildEnv:
    def test_passes_env_through_when_not_frozen(self, monkeypatch):
        monkeypatch.delattr(cli.sys, "frozen", raising=False)
        monkeypatch.setenv("LD_LIBRARY_PATH", "/opt/lib")

        env = cli._child_env({"ANTHROPIC_BASE_URL": "http://localhost:3080"})

        assert env["LD_LIBRARY_PATH"] == "/opt/lib"
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:3080"

    def test_drops_pyinstaller_loader_paths_when_frozen(self, monkeypatch):
        monkeypatch.setattr(cli.sys, "frozen", True, raising=False)
        monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEIabc123")
        monkeypatch.setenv("_MEIPASS2", "/tmp/_MEIabc123")

        env = cli._child_env()

        assert "LD_LIBRARY_PATH" not in env
        assert "_MEIPASS2" not in env

    def test_restores_the_users_own_loader_path_when_frozen(self, monkeypatch):
        monkeypatch.setattr(cli.sys, "frozen", True, raising=False)
        monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEIabc123")
        monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/opt/lib")

        env = cli._child_env()

        assert env["LD_LIBRARY_PATH"] == "/opt/lib"
        assert "LD_LIBRARY_PATH_ORIG" not in env


class TestRequireConfig:
    def test_exits_without_a_saved_config(self, monkeypatch):
        monkeypatch.setattr(cli, "load_config", lambda: None)
        with pytest.raises(Exception):
            cli._require_config()

    def test_exits_when_no_model_was_ever_picked(self, monkeypatch):
        monkeypatch.setattr(cli, "load_config", lambda: make_config(models=ModelConfig(opus="")))
        with pytest.raises(Exception):
            cli._require_config()

    def test_returns_the_saved_config(self, monkeypatch):
        saved = make_config()
        monkeypatch.setattr(cli, "load_config", lambda: saved)
        assert cli._require_config() is saved
