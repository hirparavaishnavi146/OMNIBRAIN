"""FastAPI application entry point for OmniBrain.

Sets up the app with lifespan management, exception handlers,
structured logging, and the API router.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from omnibrain.api.router import api_router
from omnibrain.config import get_settings
from omnibrain.dependencies import AppState
from omnibrain.exceptions import register_exception_handlers


def _configure_logging(level: str) -> None:
    """Set up structured logging with a consistent format."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Quieten noisy third-party loggers.
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan — startup and shutdown hooks."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    logger = logging.getLogger(__name__)
    logger.info("OmniBrain starting up…")

    app_state = AppState(settings)
    await app_state.startup()
    app.state.app_state = app_state

    yield

    logger.info("OmniBrain shutting down…")
    await app_state.shutdown()


def create_app() -> FastAPI:
    """Factory function that builds and returns the FastAPI application."""
    app = FastAPI(
        title="OmniBrain",
        description=(
            "Agentic Multimodal Retrieval-Augmented Generation (RAG) system. "
            "Upload PDFs and ask natural-language questions with grounded, "
            "cited answers."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS (permissive for local dev; restrict in production) ────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ─────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ────────────────────────────────────────────────────
    app.include_router(api_router)

    # ── Health check ───────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health_check() -> dict[str, str]:
        """Simple liveness probe."""
        return {"status": "healthy"}

    return app


app = create_app()
