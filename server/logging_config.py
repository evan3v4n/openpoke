from __future__ import annotations

import logging
import os

logger = logging.getLogger("openpoke.server")

# Attributes present on every LogRecord; anything else came from `extra=`.
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime", "taskName"}


class _ExtraFormatter(logging.Formatter):
    """Standard formatter that also prints `extra={...}` fields."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = {k: v for k, v in record.__dict__.items() if k not in _STANDARD_ATTRS}
        if extras:
            base += " | " + " ".join(f"{k}={v!r}" for k, v in extras.items())
        return base


def configure_logging() -> None:
    """Configure logging. Set OPENPOKE_LOG_LEVEL=DEBUG for verbose output."""
    if logger.handlers:
        return

    level = getattr(logging, os.getenv("OPENPOKE_LOG_LEVEL", "INFO").upper(), logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(
        _ExtraFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)
    logger.setLevel(level)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
