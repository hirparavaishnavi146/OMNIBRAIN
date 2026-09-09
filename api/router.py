"""Top-level API router — aggregates all sub-routers."""

from __future__ import annotations

from fastapi import APIRouter

from omnibrain.api.documents import router as documents_router
from omnibrain.api.questions import router as questions_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(documents_router)
api_router.include_router(questions_router)
