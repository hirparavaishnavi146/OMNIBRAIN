"""Pydantic models for document-related API requests and responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Lifecycle states for a document during ingestion."""

    PENDING = "PENDING"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    READY = "READY"
    FAILED = "FAILED"


class DocumentUploadResponse(BaseModel):
    """Returned immediately after a successful upload."""

    document_id: str = Field(..., description="Unique identifier for the uploaded document.")
    filename: str = Field(..., description="Original filename.")
    status: DocumentStatus = Field(..., description="Current processing status.")
    message: str = Field(..., description="Human-readable status message.")


class DocumentStatusResponse(BaseModel):
    """Detailed ingestion status for a single document."""

    document_id: str
    filename: str
    status: DocumentStatus
    error_message: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    table_count: int | None = None
    image_count: int | None = None
    image_embeddings_generated: bool | None = None
    created_at: datetime | None = None


class DocumentInfo(BaseModel):
    """Summary information returned in document listings."""

    document_id: str
    filename: str
    status: DocumentStatus
    page_count: int | None = None
    created_at: datetime | None = None


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    documents: list[DocumentInfo]
    total: int


class DocumentDeleteResponse(BaseModel):
    """Confirmation of document deletion."""

    document_id: str
    message: str
