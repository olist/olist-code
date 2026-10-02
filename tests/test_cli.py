"""Tests for the harness-selection dispatch in the CLI."""

from __future__ import annotations

import json
import logging
import socket

import pytest
import typer
from typer.testing import CliRunner

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

    def test_opencode_only_does_not_touch_claude(self, claude_settings_file, opencode_settings_file):
        cli._apply_settings(make_config(harness="opencode"))

        assert not claude_settings_file.exists()
        assert opencode_settings_file.exists()

    def test_both_touches_both(self, claude_settings_file, opencode_settings_file):
        cli._apply_settings(make_config(harness="both"))

        assert claude_settings_file.exists()
        assert opencode_settings_file.exists()

    def test_default_mode_still_points_plain_claude_at_the_proxy(self, claude_settings_file):
        cli._apply_settings(make_config(harness="claude"))

        data = json.loads(claude_settings_file.read_text())
        assert data["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:3080"

    def test_lists_gateway_models_in_the_picker(self, claude_settings_file):
        cli._apply_settings(make_config(harness="claude"), model_ids=["glm-4.6", "grok-4"])

        data = json.loads(claude_settings_file.read_text())
        assert data["modelPicker"] == config.claude_model_picker(["glm-4.6", "grok-4"])


class TestApplySettingsStandalone:
    @pytest.mark.parametrize("harness", ["claude", "opencode", "both"])
    def test_never_creates_the_global_claude_settings(self, harness, claude_settings_file, opencode_settings_file):
        cli._apply_settings(make_config(harness=harness), isolated=True)

        assert not claude_settings_file.exists()

    def test_opencode_is_configured_either_way(self, claude_settings_file, opencode_settings_file):
        cli._apply_settings(make_config(harness="both"), isolated=True)

        assert opencode_settings_file.exists()

    def test_takes_back_what_a_plain_run_wrote(self, claude_settings_file, opencode_settings_file):
        claude_settings_file.parent.mkdir(parents=True)
        claude_settings_file.write_text(
            json.dumps({"env": {"ANTHROPIC_BASE_URL": "http://localhost:3080", "OTHER": "1"}})
        )

        cli._apply_settings(make_config(harness="claude"), isolated=True)

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


class TestGatewayModelIds:
    def test_returns_the_gateway_ids(self, monkeypatch):
        async def fake_fetch(_config):
            return [{"id": "glm-4.6", "owned_by": "zai"}, {"owned_by": "broken"}, {"id": "grok-4"}]

        monkeypatch.setattr(cli, "fetch_gateway_models", fake_fetch)

        assert cli._gateway_model_ids(make_config()) == ["glm-4.6", "grok-4"]

    def test_empty_when_the_gateway_is_unreachable(self, monkeypatch):
        # A picker without gateway models is still usable; failing startup over it is not.
        async def fake_fetch(_config):
            raise RuntimeError("boom")

        monkeypatch.setattr(cli, "fetch_gateway_models", fake_fetch)

        assert cli._gateway_model_ids(make_config()) == []


class TestClaudeSettingsArgs:
    def test_passes_the_picker_to_the_session(self):
        args = cli._claude_settings_args(["glm-4.6", "grok-4"])

        assert args[0] == "--settings"
        assert json.loads(args[1]) == {"modelPicker": config.claude_model_picker(["glm-4.6", "grok-4"])}

    def test_no_flag_without_gateway_models(self):
        assert cli._claude_settings_args([]) == []


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


class TestEnsurePortFree:
    def test_exits_when_another_server_listens_with_reuseport(self):
        # granian binds with SO_REUSEPORT, so a second granian would silently share the port.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
            other.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            other.bind(("0.0.0.0", 0))
            other.listen()
            port = other.getsockname()[1]

            with pytest.raises(typer.Exit) as exc:
                cli._ensure_port_free(port)

        assert exc.value.exit_code == 1

    def test_passes_when_port_is_free(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("0.0.0.0", 0))
            port = probe.getsockname()[1]

        cli._ensure_port_free(port)


class TestLogLevelOptions:
    @pytest.fixture
    def run_server(self, monkeypatch):
        calls = []
        monkeypatch.delenv("OLIST_CODE_LOG_LEVEL", raising=False)
        monkeypatch.setattr(cli, "_run_server", lambda port, harness, isolated, log_level: calls.append(log_level))
        return calls

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            ([], logging.INFO),
            (["--debug"], logging.DEBUG),
            (["--log-level", "warning"], logging.WARNING),
            (["standalone", "--debug"], logging.DEBUG),
            (["standalone", "--log-level", "error"], logging.ERROR),
        ],
    )
    def test_server_commands_resolve_the_level(self, run_server, args, expected):
        result = CliRunner().invoke(cli.cli, args)

        assert result.exit_code == 0, result.output
        assert run_server == [expected]

    def test_env_is_used_without_flags(self, run_server, monkeypatch):
        monkeypatch.setenv("OLIST_CODE_LOG_LEVEL", "warning")

        CliRunner().invoke(cli.cli, ["standalone"])

        assert run_server == [logging.WARNING]

    def test_unknown_level_exits(self, run_server):
        result = CliRunner().invoke(cli.cli, ["--log-level", "loud"])

        assert result.exit_code == 1
        assert run_server == []


class TestStartGranian:
    def test_disables_granian_access_log_and_uses_our_logging(self, monkeypatch, tmp_path):
        import granian

        captured = {}

        class FakeGranian:
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)

            def serve(self):
                pass

        monkeypatch.setattr(granian, "Granian", FakeGranian)
        monkeypatch.setattr(cli, "LOG_FILE", tmp_path / "olist-code.log")

        cli._start_granian(make_config(), logging.DEBUG)

        assert captured["log_access"] is False
        assert captured["log_level"] == "info"
        assert captured["log_dictconfig"]["root"]["level"] == logging.DEBUG


class _FakeStatsResponse:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}

    def json(self) -> dict:
        return self._body


class TestUsage:
    @pytest.fixture
    def stats_response(self, monkeypatch):
        state = {"response": _FakeStatsResponse(200, {"started_at": "2026-10-02T12:00:00+00:00", "models": {}})}
        calls = []

        def fake_get(url, timeout):
            calls.append(url)
            if isinstance(state["response"], Exception):
                raise state["response"]
            return state["response"]

        monkeypatch.setattr(cli, "load_config", lambda: make_config())
        monkeypatch.setattr(cli.httpx, "get", fake_get)
        state["calls"] = calls
        return state

    def test_prints_rows_and_totals(self, stats_response):
        model = dict(requests=2, errors=1, input_tokens=1500, output_tokens=300, cached_tokens=1000, duration_ms=4200)
        stats_response["response"] = _FakeStatsResponse(
            200,
            {
                "started_at": "2026-10-02T12:00:00+00:00",
                "models": {"glm-4.6": {**model, "cost": 0.25}, "grok-4": {**model, "cost": 0.5}},
            },
        )

        result = CliRunner().invoke(cli.cli, ["usage"], terminal_width=200)

        assert result.exit_code == 0, result.output
        assert stats_response["calls"] == ["http://localhost:3080/olist/stats"]
        assert "glm-4.6" in result.output
        assert "grok-4" in result.output
        assert "Total" in result.output
        assert "0.2500" in result.output
        assert "0.7500" in result.output
        assert "3,000" in result.output
        assert "2026-10-02" in result.output

    def test_empty_stats(self, stats_response):
        result = CliRunner().invoke(cli.cli, ["usage"])

        assert result.exit_code == 0, result.output
        assert "Nenhuma requisição" in result.output

    def test_proxy_down_exits(self, stats_response):
        stats_response["response"] = cli.httpx.ConnectError("refused")

        result = CliRunner().invoke(cli.cli, ["usage"])

        assert result.exit_code == 1
        assert "não está respondendo" in result.output

    def test_old_proxy_without_stats_exits(self, stats_response):
        stats_response["response"] = _FakeStatsResponse(404)

        result = CliRunner().invoke(cli.cli, ["usage"])

        assert result.exit_code == 1
        assert "reinicie" in result.output.lower()

    def test_proxy_error_status_exits(self, stats_response):
        stats_response["response"] = _FakeStatsResponse(500)

        result = CliRunner().invoke(cli.cli, ["usage"])

        assert result.exit_code == 1
        assert "HTTP 500" in result.output
