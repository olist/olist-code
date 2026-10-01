"""Logging config for the proxy: level resolution, request-id tagging, stderr + rotating file."""

from __future__ import annotations

import logging
import logging.handlers
import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .config import CONFIG_DIR

LOG_FILE = CONFIG_DIR / "logs" / "olist-code.log"
LOG_LEVEL_ENV = "OLIST_CODE_LOG_LEVEL"
LOG_LEVELS = ("debug", "info", "warning", "error")

request_id: ContextVar[str | None] = ContextVar("olist_code_request_id", default=None)


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rid = request_id.get()
        record.req = f"[req={rid}] " if rid else ""
        return True


def resolve_log_level(flag: str | None, debug: bool) -> int:
    if debug:
        return logging.DEBUG
    value = (flag or os.environ.get(LOG_LEVEL_ENV) or "info").strip().lower()
    if value not in LOG_LEVELS:
        raise ValueError(f"Invalid log level {value!r}; use one of: {', '.join(LOG_LEVELS)}")
    return logging.getLevelNamesMapping()[value.upper()]


def granian_log_level(level: int) -> str:
    """Granian's own debug output is internal noise, so it never goes below info."""
    return logging.getLevelName(max(level, logging.INFO)).lower()


class SharedRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Granian's parent and worker (and other proxies) append to the same file: reopen it when
    another process rotated it, so nobody keeps writing to the renamed backup. Logs may carry
    sensitive data, so the file is kept private."""

    def _open(self):
        stream = super()._open()
        os.chmod(self.baseFilename, 0o600)
        return stream

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        if self.stream is not None and self._rotated_elsewhere():
            self.stream.close()
            self.stream = self._open()
        return super().shouldRollover(record)

    def _rotated_elsewhere(self) -> bool:
        try:
            current = os.stat(self.baseFilename)
        except FileNotFoundError:
            return True
        opened = os.fstat(self.stream.fileno())
        return (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)


def _file_is_writable(log_file: Path) -> bool:
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a"):
            pass
        log_file.chmod(0o600)
    except OSError:
        return False
    return True


def build_log_config(level: int, log_file: Path | None = LOG_FILE) -> dict[str, Any]:
    """A dictConfig, handed to Granian so each worker process applies it on start."""
    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "default",
            "filters": ["request_id"],
        },
    }
    if log_file is not None and _file_is_writable(log_file):
        handlers["file"] = {
            "class": SharedRotatingFileHandler,
            "filename": str(log_file),
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 3,
            "encoding": "utf-8",
            "formatter": "default",
            "filters": ["request_id"],
        }
    names = list(handlers)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"request_id": {"()": RequestIdFilter}},
        "formatters": {
            "default": {
                "format": "%(asctime)s %(levelname)s %(name)s %(req)s%(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": handlers,
        "root": {"level": level, "handlers": names},
        "loggers": {
            "_granian": {"handlers": names, "propagate": False},
            "granian.access": {"handlers": names, "propagate": False},
            "httpx": {"level": logging.INFO if level <= logging.DEBUG else logging.WARNING},
            "httpcore": {"level": logging.WARNING},
        },
    }
