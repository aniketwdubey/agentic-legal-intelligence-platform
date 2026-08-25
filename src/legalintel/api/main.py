"""FastAPI application factory.

The app owns wiring only — configuration, logging, and building the supervisor
once at startup. No business logic lives in this module or in the route
handlers (see ``routes.py``); handlers delegate straight to the supervisor.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from legalintel.agents import build_supervisor
from legalintel.config import get_settings

log = structlog.get_logger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    log.info("api.startup", provider=settings.llm_provider, embedder=settings.embedder)
    # Build the pipeline once (corpus load + index) and reuse across requests.
    app.state.supervisor = build_supervisor(settings)
    yield
    log.info("api.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agentic Legal Intelligence Platform",
        version="0.1.0",
        summary="Grounded, citation-verified legal answers. Abstains rather than hallucinates.",
        lifespan=lifespan,
    )
    from legalintel.api.routes import router

    app.include_router(router)
    return app


app = create_app()
