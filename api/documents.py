"""Document management and query API endpoints.

Handles upload, status polling, document listing, multi-modal LangGraph querying,
citation detail retrieval, and deletion.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile

from omnibrain.dependencies import AppState
from omnibrain.exceptions import (
    CitationNotFoundError,
    DocumentNotFoundError,
    DocumentNotReadyError,
    FileTooLargeError,
    InvalidFileTypeError,
)
from omnibrain.schemas.documents import (
    DocumentDeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentStatus,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from omnibrain.schemas.questions import (
    AnswerResponse,
    Citation,
    CitationDetailResponse,
    QueryRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


def _get_state(request: Request) -> AppState:
    """Retrieve the shared ``AppState`` from the ASGI app."""
    return request.app.state.app_state


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload a PDF document",
    description="Upload a corporate PDF for asynchronous ingestion (text, tables, and CLIP image embeddings).",
)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to upload"),
) -> DocumentUploadResponse:
    """Accept a PDF upload, validate file size/magic bytes, and launch background ingestion."""
    state = _get_state(request)

    # ── Validate file type ─────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise InvalidFileTypeError(
            "Only PDF files are accepted.",
            details={"filename": file.filename or ""},
        )

    if file.content_type and file.content_type != "application/pdf":
        raise InvalidFileTypeError(
            "Content type must be application/pdf.",
            details={"content_type": file.content_type},
        )

    # ── Read and validate size ─────────────────────────────────────
    pdf_bytes = await file.read()
    max_size = state.settings.max_file_size_bytes
    if len(pdf_bytes) > max_size:
        raise FileTooLargeError(
            f"File exceeds the {state.settings.max_file_size_mb} MB limit.",
            details={"size_bytes": len(pdf_bytes), "max_bytes": max_size},
        )

    if len(pdf_bytes) == 0:
        raise InvalidFileTypeError("Uploaded file is empty.")

    # ── Quick PDF magic-byte check ─────────────────────────────────
    if not pdf_bytes[:5] == b"%PDF-":
        raise InvalidFileTypeError(
            "File does not appear to be a valid PDF format.",
            details={"filename": file.filename},
        )

    # ── Enqueue processing ─────────────────────────────────────────
    doc_id = str(uuid.uuid4())
    background_tasks.add_task(
        state.ingestion_processor.ingest,
        doc_id,
        file.filename or "untitled.pdf",
        pdf_bytes,
    )

    logger.info("Upload accepted: %s (id=%s, %d bytes).", file.filename, doc_id, len(pdf_bytes))

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename or "untitled.pdf",
        status=DocumentStatus.PENDING,
        message="Document uploaded successfully. Multi-modal ingestion has started.",
    )


@router.get(
    "/{doc_id}/status",
    response_model=DocumentStatusResponse,
    summary="Check ingestion status",
    description="Poll the current processing status of a document (PENDING → PARSING → CHUNKING → EMBEDDING → READY/FAILED).",
)
async def get_document_status(doc_id: str, request: Request) -> DocumentStatusResponse:
    """Return the current processing status and content counts for a document."""
    state = _get_state(request)
    doc = await state.database.get_document(doc_id)
    if not doc:
        raise DocumentNotFoundError(
            f"Document '{doc_id}' not found.",
            details={"document_id": doc_id},
        )

    chunk_count = await state.database.get_chunk_count(doc_id)
    table_count = await state.database.get_table_count(doc_id)
    image_count = await state.database.get_image_count(doc_id)

    # Check if CLIP image embeddings exist in the vector store
    image_embeddings_generated: bool | None = None
    if image_count > 0 and state.vector_store:
        try:
            import numpy as np
            # Probe with a zero vector — if we get results, embeddings exist
            zero_vec = np.zeros(state.settings.clip_embedding_dimension, dtype=np.float32)
            probe = await state.vector_store.search(
                doc_id, zero_vec, top_k=1, min_score=0.0, modality_filter="image"
            )
            image_embeddings_generated = len(probe) > 0
        except Exception:
            image_embeddings_generated = False
    elif image_count == 0:
        image_embeddings_generated = None  # No images to embed

    return DocumentStatusResponse(
        document_id=doc_id,
        filename=doc["filename"],
        status=DocumentStatus(doc["status"]),
        error_message=doc.get("error_message"),
        page_count=doc.get("page_count"),
        chunk_count=chunk_count,
        table_count=table_count,
        image_count=image_count,
        image_embeddings_generated=image_embeddings_generated,
        created_at=doc.get("created_at"),
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all documents",
)
async def list_documents(request: Request) -> DocumentListResponse:
    """Return a list of all uploaded documents."""
    state = _get_state(request)
    docs = await state.database.list_documents()
    items = [
        DocumentInfo(
            document_id=d["id"],
            filename=d["filename"],
            status=DocumentStatus(d["status"]),
            page_count=d.get("page_count"),
            created_at=d.get("created_at"),
        )
        for d in docs
    ]
    return DocumentListResponse(documents=items, total=len(items))


@router.get(
    "/{doc_id}",
    response_model=DocumentStatusResponse,
    summary="Get document details",
)
async def get_document_detail(doc_id: str, request: Request) -> DocumentStatusResponse:
    """Return full details for a single document (alias for status)."""
    return await get_document_status(doc_id, request)


@router.post(
    "/{doc_id}/query",
    response_model=AnswerResponse,
    summary="Ask a question about a document via LangGraph orchestrator",
    description=(
        "Submit a natural-language question about an uploaded document. "
        "The LangGraph orchestrator dispatches across Retrieval, Vision, and SQL agents, "
        "executes Self-RAG loops, validates grounding via NeMo Guardrails, and traces in Langfuse."
    ),
)
async def query_document(
    doc_id: str, body: QueryRequest, request: Request
) -> AnswerResponse:
    """Execute LangGraph multi-agent RAG workflow for a document."""
    state = _get_state(request)

    # Validate document existence and readiness
    doc = await state.database.get_document(doc_id)
    if not doc:
        raise DocumentNotFoundError(
            f"Document '{doc_id}' not found.",
            details={"document_id": doc_id},
        )

    if doc["status"] != DocumentStatus.READY.value:
        raise DocumentNotReadyError(
            f"Document is not ready for querying (status: {doc['status']}).",
            details={"document_id": doc_id, "status": doc["status"]},
        )

    # Start top-level Langfuse trace
    trace = None
    if state.tracer and state.tracer.enabled:
        trace = state.tracer.start_trace(
            name="OmniBrain.DocumentQuery",
            metadata={"document_id": doc_id, "filename": doc["filename"], "question": body.question},
        )

    try:
        # Run LangGraph State Machine
        answer_response = await state.graph_orchestrator.ainvoke(
            document_id=doc_id,
            question=body.question,
            trace=trace,
        )
        return answer_response
    finally:
        if state.tracer and state.tracer.enabled:
            state.tracer.flush()


@router.get(
    "/{doc_id}/citations/{citation_id}",
    response_model=CitationDetailResponse,
    summary="Get citation detail",
    description="Retrieve the exact source snippet, page number, and source type for a given citation.",
)
async def get_citation_detail(
    doc_id: str, citation_id: str, request: Request
) -> CitationDetailResponse:
    """Retrieve full citation provenance."""
    state = _get_state(request)
    from omnibrain.citations.engine import CitationEngine

    engine = CitationEngine()
    chunks = await state.database.get_chunks_for_document(doc_id)

    # 1. Check chunks
    for chunk in chunks:
        test_id = engine._generate_id(doc_id, chunk["page_number"], chunk["text"])
        if test_id == citation_id:
            return CitationDetailResponse(
                citation=Citation(
                    citation_id=citation_id,
                    document_id=doc_id,
                    page_number=chunk["page_number"],
                    section_title=chunk.get("section_title"),
                    text_snippet=chunk["text"][:500],
                    relevance_score=1.0,
                    source_type="text",
                )
            )

    # 2. Check images
    images = await state.database.get_images_for_document(doc_id)
    for img in images:
        desc = (
            f"Visual element / Chart on Page {img.get('page_number', 1)} "
            f"(Image {img.get('image_index', 0)})"
        )
        test_id = engine._generate_id(doc_id, img["page_number"], desc)
        if test_id == citation_id or str(img["id"]) == citation_id:
            return CitationDetailResponse(
                citation=Citation(
                    citation_id=citation_id,
                    document_id=doc_id,
                    page_number=img["page_number"],
                    section_title=f"Chart / Image (Page {img['page_number']})",
                    text_snippet=desc,
                    relevance_score=0.9,
                    source_type="image",
                )
            )

    # 3. Check tables
    tables = await state.database.get_tables_for_document(doc_id)
    for tbl in tables:
        desc = (
            f"Table '{tbl['table_name']}' on Page {tbl.get('page_number', 1)} "
            f"(Columns: {', '.join(tbl.get('headers', []))})"
        )
        test_id = engine._generate_id(doc_id, tbl["page_number"], desc)
        if test_id == citation_id or str(tbl["id"]) == citation_id:
            return CitationDetailResponse(
                citation=Citation(
                    citation_id=citation_id,
                    document_id=doc_id,
                    page_number=tbl["page_number"],
                    section_title=f"Table (Page {tbl['page_number']})",
                    text_snippet=desc,
                    relevance_score=0.95,
                    source_type="table",
                )
            )

    raise CitationNotFoundError(
        f"Citation '{citation_id}' not found for document '{doc_id}'.",
        details={"document_id": doc_id, "citation_id": citation_id},
    )


@router.delete(
    "/{doc_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document",
    description="Delete a document, its database records, on-disk files, and vector embeddings.",
)
async def delete_document(doc_id: str, request: Request) -> DocumentDeleteResponse:
    """Delete all records and artifacts associated with a document."""
    state = _get_state(request)
    doc = await state.database.get_document(doc_id)
    if not doc:
        raise DocumentNotFoundError(
            f"Document '{doc_id}' not found.",
            details={"document_id": doc_id},
        )

    # Delete vector embeddings from Qdrant/FAISS, database tables, and on-disk files
    await state.vector_store.delete_collection(doc_id)
    await state.database.delete_document(doc_id)
    state.file_store.delete_document(doc_id)

    logger.info("Document %s and all associated vectors deleted.", doc_id)
    return DocumentDeleteResponse(
        document_id=doc_id,
        message="Document and all associated data deleted successfully.",
    )
