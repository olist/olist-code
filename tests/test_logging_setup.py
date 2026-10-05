"""Tests for log level resolution and the logging config handed to Granian."""

from __future__ import annotations

import logging
import logging.config
import os

import pytest

from olist_code import logging_setup


@pytest.fixture(autouse=True)
def no_env_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLIST_CODE_LOG_LEVEL", raising=False)


@pytest.fixture
def restore_logging():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    for handler in root.handlers:
        if handler not in handlers:
            handler.close()
    root.handlers[:] = handlers
    root.setLevel(level)


class TestResolveLogLevel:
    def test_defaults_to_info(self) -> None:
        assert logging_setup.resolve_log_level(None, debug=False) == logging.INFO

    def test_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLIST_CODE_LOG_LEVEL", "warning")
        assert logging_setup.resolve_log_level(None, debug=False) == logging.WARNING

    def test_flag_wins_over_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLIST_CODE_LOG_LEVEL", "debug")
        assert logging_setup.resolve_log_level("ERROR", debug=False) == logging.ERROR

    def test_debug_shortcut_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLIST_CODE_LOG_LEVEL", "error")
        assert logging_setup.resolve_log_level(None, debug=True) == logging.DEBUG

    def test_rejects_unknown_level(self) -> None:
        with pytest.raises(ValueError, match="verbose"):
            logging_setup.resolve_log_level("verbose", debug=False)


class TestBuildLogConfig:
    def test_writes_formatted_lines_with_request_id_to_the_file(self, tmp_path, restore_logging) -> None:
        log_file = tmp_path / "logs" / "olist-code.log"
        logging.config.dictConfig(logging_setup.build_log_config(logging.INFO, log_file))

        token = logging_setup.request_id.set("ab12cd")
        try:
            logging.getLogger("olist_code.server").error("upstream failed")
        finally:
            logging_setup.request_id.reset(token)
        logging.getLogger("olist_code.server").info("no request")

        lines = log_file.read_text().splitlines()
        assert lines[0].endswith(" ERROR olist_code.server [req=ab12cd] upstream failed")
        assert lines[1].endswith(" INFO olist_code.server no request")

    def test_level_filters_lower_records(self, tmp_path, restore_logging) -> None:
        log_file = tmp_path / "olist-code.log"
        logging.config.dictConfig(logging_setup.build_log_config(logging.WARNING, log_file))

        logging.getLogger("olist_code.server").info("hidden")

        assert "hidden" not in log_file.read_text()

    def test_skips_file_when_dir_cannot_be_created(self, tmp_path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("")

        config = logging_setup.build_log_config(logging.INFO, blocker / "logs" / "olist-code.log")

        assert "file" not in config["handlers"]
        assert config["root"]["handlers"] == ["console"]

    def test_no_file_handler_when_log_file_is_none(self) -> None:
        config = logging_setup.build_log_config(logging.INFO, log_file=None)

        assert "file" not in config["handlers"]
        assert config["root"]["handlers"] == ["console"]

    def test_existing_log_file_is_made_private(self, tmp_path) -> None:
        log_file = tmp_path / "olist-code.log"
        log_file.write_text("")
        log_file.chmod(0o644)

        logging_setup.build_log_config(logging.INFO, log_file)

        assert log_file.stat().st_mode & 0o777 == 0o600


class TestSharedRotatingFileHandler:
    def _handler(self, log_file, max_bytes: int = 0):
        handler = logging_setup.SharedRotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        return handler

    def _record(self, msg: str) -> logging.LogRecord:
        return logging.LogRecord("t", logging.INFO, __file__, 1, msg, None, None)

    def test_rotated_files_are_private(self, tmp_path) -> None:
        log_file = tmp_path / "olist-code.log"
        handler = self._handler(log_file, max_bytes=20)
        try:
            for i in range(5):
                handler.emit(self._record(f"line number {i}"))
        finally:
            handler.close()

        files = [log_file, *tmp_path.glob("olist-code.log.*")]
        assert len(files) > 1
        assert all(f.stat().st_mode & 0o777 == 0o600 for f in files)

    def test_reopens_after_another_process_rotated_the_file(self, tmp_path) -> None:
        log_file = tmp_path / "olist-code.log"
        handler = self._handler(log_file)
        try:
            handler.emit(self._record("before"))
            os.rename(log_file, tmp_path / "olist-code.log.1")
            log_file.write_text("")
            handler.emit(self._record("after"))
        finally:
            handler.close()

        assert log_file.read_text() == "after\n"
        assert (tmp_path / "olist-code.log.1").read_text() == "before\n"

    def test_build_log_config_uses_shared_handler(self, tmp_path) -> None:
        config = logging_setup.build_log_config(logging.INFO, tmp_path / "olist-code.log")

        assert config["handlers"]["file"]["class"] is logging_setup.SharedRotatingFileHandler


class TestGranianLevel:
    @pytest.mark.parametrize(
        ("level", "expected"),
        [(logging.DEBUG, "info"), (logging.INFO, "info"), (logging.WARNING, "warning"), (logging.ERROR, "error")],
    )
    def test_never_goes_below_info(self, level: int, expected: str) -> None:
        assert logging_setup.granian_log_level(level) == expected
