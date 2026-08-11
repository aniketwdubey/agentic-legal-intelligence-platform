"""Structured logging setup.

Uses ``structlog`` so every log line carries the run's context (trace_id,
agent, step) as key/value pairs. In production (``LOG_FORMAT=json``) lines are
machine-parseable JSON; in dev they are colourised key=value pairs.

Bind per-request/per-run context with ``bind_run(trace_id=...)`` and every
subsequent log from that agent chain inherits it.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog
from structlog.typing import Processor

from legalintel.config import Settings

_CONFIGURED = False


def configure_logging(settings: Settings) -> None:
    """Idempotently configure stdlib + structlog. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
    )

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_format == "json"
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound logger for ``name``."""
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def bind_run(**kwargs: object) -> None:
    """Bind context vars (e.g. trace_id) shared by every log in this run."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_run() -> None:
    """Clear run-scoped context vars (call at the end of a request)."""
    structlog.contextvars.clear_contextvars()
