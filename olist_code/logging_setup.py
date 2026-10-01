"""Logging config for the proxy: level resolution, request-id tagging, stderr + rotating file."""

from __future__ import annotations

import logging
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


def _file_is_writable(log_file: Path) -> bool:
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a"):
            pass
    except OSError:
        return False
    return True


def build_log_config(level: int, log_file: Path = LOG_FILE) -> dict[str, Any]:
    """A dictConfig, handed to Granian so each worker process applies it on start."""
    handlers: dict[str, Any] = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
            "formatter": "default",
            "filters": ["request_id"],
        },
    }
    if _file_is_writable(log_file):
        handlers["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
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
