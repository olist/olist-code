"""Tests for settings-file writers (Claude Code and opencode)."""

from __future__ import annotations

import json

import pytest

from olist_code import config
from olist_code.models import AdapterConfig, ModelConfig


def make_config(**overrides) -> AdapterConfig:
    defaults = dict(
        base_url="http://localhost:8000",
        api_key="sk-olist-x",
        sso=None,
        models=ModelConfig(opus="glm-4.6", sonnet="glm-4.5", haiku=None),
        port=3080,
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
    return path


class TestUpdateClaudeSettings:
    def test_writes_env_vars(self, claude_settings_file):
        config.update_claude_settings(make_config())

        data = json.loads(claude_settings_file.read_text())
        assert data["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:3080"
        assert data["env"]["ANTHROPIC_AUTH_TOKEN"] == "default"
        assert data["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "glm-4.6"
        assert data["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "glm-4.5"
        assert "ANTHROPIC_DEFAULT_HAIKU_MODEL" not in data["env"]

    def test_preserves_unrelated_keys(self, claude_settings_file):
        claude_settings_file.parent.mkdir(parents=True)
        claude_settings_file.write_text(json.dumps({"foo": "bar", "env": {"OTHER": "1"}}))

        config.update_claude_settings(make_config())

        data = json.loads(claude_settings_file.read_text())
        assert data["foo"] == "bar"
        assert data["env"]["OTHER"] == "1"
        assert data["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:3080"


class TestUpdateOpencodeSettings:
    def test_writes_provider(self, opencode_settings_file):
        config.update_opencode_settings(make_config())

        data = json.loads(opencode_settings_file.read_text())
        provider = data["provider"]["olist-ai-gateway"]
        assert provider["npm"] == "@ai-sdk/openai-compatible"
        assert provider["options"]["baseURL"] == "http://localhost:3080/v1"
        assert provider["options"]["apiKey"] == "default"
        assert "glm-4.6" in provider["models"]
        assert "glm-4.5" in provider["models"]

    def test_preserves_other_providers(self, opencode_settings_file):
        opencode_settings_file.parent.mkdir(parents=True)
        opencode_settings_file.write_text(
            json.dumps({"provider": {"anthropic": {"npm": "@ai-sdk/anthropic"}}})
        )

        config.update_opencode_settings(make_config())

        data = json.loads(opencode_settings_file.read_text())
        assert "anthropic" in data["provider"]
        assert "olist-ai-gateway" in data["provider"]

    def test_omits_unset_models(self, opencode_settings_file):
        config.update_opencode_settings(make_config(models=ModelConfig(opus="glm-4.6")))

        data = json.loads(opencode_settings_file.read_text())
        models = data["provider"]["olist-ai-gateway"]["models"]
        assert list(models.keys()) == ["glm-4.6"]


class TestRestoreClaudeSettings:
    def test_noop_when_file_missing(self, claude_settings_file):
        config.restore_claude_settings()
        assert not claude_settings_file.exists()

    def test_removes_only_our_keys(self, claude_settings_file):
        claude_settings_file.parent.mkdir(parents=True)
        claude_settings_file.write_text(json.dumps({"foo": "bar", "env": {"OTHER": "1"}}))

        config.update_claude_settings(make_config())
        config.restore_claude_settings()

        data = json.loads(claude_settings_file.read_text())
        assert data["foo"] == "bar"
        assert data["env"] == {"OTHER": "1"}

    def test_drops_env_key_if_it_becomes_empty(self, claude_settings_file):
        claude_settings_file.parent.mkdir(parents=True)
        claude_settings_file.write_text(json.dumps({"foo": "bar"}))

        config.update_claude_settings(make_config())
        config.restore_claude_settings()

        data = json.loads(claude_settings_file.read_text())
        assert data == {"foo": "bar"}


class TestRestoreOpencodeSettings:
    def test_noop_when_file_missing(self, opencode_settings_file):
        config.restore_opencode_settings()
        assert not opencode_settings_file.exists()

    def test_removes_only_our_provider(self, opencode_settings_file):
        opencode_settings_file.parent.mkdir(parents=True)
        opencode_settings_file.write_text(json.dumps({"provider": {"anthropic": {"npm": "@ai-sdk/anthropic"}}}))

        config.update_opencode_settings(make_config())
        config.restore_opencode_settings()

        data = json.loads(opencode_settings_file.read_text())
        assert data["provider"] == {"anthropic": {"npm": "@ai-sdk/anthropic"}}

    def test_deletes_file_if_we_created_it(self, opencode_settings_file):
        config.update_opencode_settings(make_config())
        config.restore_opencode_settings()

        assert not opencode_settings_file.exists()
