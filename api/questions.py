"""Question-answering API endpoints (backward-compatible interface).

Provides ``/questions/ask`` and ``/questions/citations/{citation_id}`` routing
directly to the LangGraph Multi-Agent Orchestration state machine.

DEPRECATION NOTICE: This router is kept for backward compatibility for any external
API consumers. Internal clients (like the frontend) should use the document-scoped
endpoints (e.g. /api/v1/documents/{id}/query).
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from omnibrain.api.documents import get_citation_detail as _get_citation_detail
from omnibrain.api.documents import query_document
from omnibrain.dependencies import AppState
from omnibrain.schemas.questions import (
    AnswerResponse,
    CitationDetailResponse,
    QueryRequest,
    QuestionRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/questions", tags=["Questions"])


def _get_state(request: Request) -> AppState:
    """Retrieve the shared ``AppState`` from the ASGI app."""
    return request.app.state.app_state


@router.post(
    "/ask",
    response_model=AnswerResponse,
    summary="Ask a question about a document (LangGraph Orchestrated)",
    description=(
        "Submit a question with document_id in the body. Runs the full LangGraph "
        "multi-agent orchestration pipeline across Retrieval, Vision, and SQL sub-agents. "
        "DEPRECATED: Use /api/v1/documents/{document_id}/query instead."
    ),
    deprecated=True,
)
async def ask_question(
    body: QuestionRequest, request: Request
) -> AnswerResponse:
    """Delegate to query_document with body.document_id."""
    return await query_document(
        doc_id=body.document_id,
        body=QueryRequest(question=body.question),
        request=request,
    )


@router.get(
    "/citations/{citation_id}",
    response_model=CitationDetailResponse,
    summary="Get citation detail (deprecated)",
    description=(
        "Retrieve the citation detail by citation ID across all documents. "
        "DEPRECATED: Use /api/v1/documents/{document_id}/citations/{citation_id} instead."
    ),
    deprecated=True,
)
async def get_citation_detail(
    citation_id: str, request: Request
) -> CitationDetailResponse:
    """Search for the citation across all documents."""
    state = _get_state(request)
    docs = await state.database.list_documents()

    for d in docs:
        try:
            return await _get_citation_detail(d["id"], citation_id, request)
        except Exception:
            continue

    from omnibrain.exceptions import CitationNotFoundError

    raise CitationNotFoundError(
        f"Citation '{citation_id}' not found.",
        details={"citation_id": citation_id},
    )
