"""Structured logging.

Every log line is a JSON object (when ``json`` is enabled) with a UTC
timestamp, level, logger name, event message and any extra fields passed
via ``extra=``. Example::

    log = get_logger(__name__)
    log.info("bars_downloaded", extra={"timeframe": "M5", "rows": 8640})
"""
from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path

_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime", "taskName"}
_SECRET_KEYS = {"password", "token", "secret", "api_key"}


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            payload[key] = "***" if key.lower() in _SECRET_KEYS else value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class KeyValueFormatter(logging.Formatter):
    """Human-friendly fallback: ``time LEVEL logger event key=value ...``."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        extras = " ".join(f"{k}={'***' if k.lower() in _SECRET_KEYS else v}"
                          for k, v in record.__dict__.items()
                          if k not in _STANDARD_ATTRS and not k.startswith("_"))
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts}Z {record.levelname:<7} {record.name} {record.getMessage()} {extras}".rstrip()
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def setup_logging(level: str = "INFO", json_format: bool = True, file: str | None = None) -> None:
    """Configure the root ``quant_engine`` logger (idempotent)."""
    root = logging.getLogger("quant_engine")
    root.setLevel(level.upper())
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt: logging.Formatter = JsonFormatter() if json_format else KeyValueFormatter()
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    if file:
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(file, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(JsonFormatter())
        root.addHandler(fh)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``quant_engine`` namespace."""
    return logging.getLogger(name if name.startswith("quant_engine") else f"quant_engine.{name}")
