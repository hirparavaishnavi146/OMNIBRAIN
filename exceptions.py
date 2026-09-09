"""Custom exception hierarchy and FastAPI exception handlers.

Every domain error has a dedicated exception class so that the global
handler can map it to a clean, specific HTTP response.  Raw stack traces
are never returned to the client.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ── Base exception ─────────────────────────────────────────────────
class OmniBrainError(Exception):
    """Base for all OmniBrain domain errors."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


# ── Document errors ────────────────────────────────────────────────
class DocumentNotFoundError(OmniBrainError):
    """Raised when a document ID does not exist in the database."""


class DocumentNotReadyError(OmniBrainError):
    """Raised when a question is asked against a document still ingesting."""


class DuplicateDocumentError(OmniBrainError):
    """Raised when uploading a PDF whose hash already exists."""


# ── Ingestion / parsing errors ─────────────────────────────────────
class PDFParsingError(OmniBrainError):
    """Raised when PyMuPDF cannot parse the PDF (corrupt, encrypted, etc.)."""


class IngestionError(OmniBrainError):
    """Raised when any step in the ingestion pipeline fails."""


class EmptyDocumentError(OmniBrainError):
    """Raised when a PDF has no extractable text, tables, or images."""


# ── File validation errors ─────────────────────────────────────────
class FileTooLargeError(OmniBrainError):
    """Raised when the uploaded file exceeds the configured size limit."""


class InvalidFileTypeError(OmniBrainError):
    """Raised when the uploaded file is not a PDF."""


# ── Retrieval / embedding errors ───────────────────────────────────
class EmbeddingError(OmniBrainError):
    """Raised when the embedding model fails."""


class RetrievalError(OmniBrainError):
    """Raised when vector search fails."""


# ── LLM / agent errors ────────────────────────────────────────────
class LLMError(OmniBrainError):
    """Raised when a Gemini API call fails (network, auth, rate-limit)."""


class AgentError(OmniBrainError):
    """Raised when an agent (supervisor, retrieval, vision, SQL) fails."""


class SQLExecutionError(OmniBrainError):
    """Raised when the SQL agent generates or executes an unsafe/invalid query."""


# ── Citation errors ────────────────────────────────────────────────
class CitationNotFoundError(OmniBrainError):
    """Raised when a citation ID does not exist."""


# ── Exception → HTTP status mapping ───────────────────────────────
_STATUS_MAP: dict[type[OmniBrainError], int] = {
    DocumentNotFoundError: status.HTTP_404_NOT_FOUND,
    CitationNotFoundError: status.HTTP_404_NOT_FOUND,
    DocumentNotReadyError: status.HTTP_409_CONFLICT,
    DuplicateDocumentError: status.HTTP_409_CONFLICT,
    PDFParsingError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    EmptyDocumentError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    FileTooLargeError: status.HTTP_413_CONTENT_TOO_LARGE,
    InvalidFileTypeError: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    EmbeddingError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    RetrievalError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    LLMError: status.HTTP_502_BAD_GATEWAY,
    AgentError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    SQLExecutionError: status.HTTP_422_UNPROCESSABLE_CONTENT,
    IngestionError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all exception handlers to the FastAPI application.

    Must be called once during app startup (in ``main.py``).
    """

    @app.exception_handler(OmniBrainError)
    async def _omnibrain_error_handler(
        request: Request, exc: OmniBrainError
    ) -> JSONResponse:
        status_code = _STATUS_MAP.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
        logger.warning(
            "Domain error: %s (status=%d, path=%s)",
            exc.message,
            status_code,
            request.url.path,
        )
        return JSONResponse(
            status_code=status_code,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTPException",
                "message": str(exc.detail),
                "details": {},
            },
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "ValidationError",
                "message": "Request validation failed.",
                "details": {"errors": exc.errors()},
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred.",
                "details": {},
            },
        )
