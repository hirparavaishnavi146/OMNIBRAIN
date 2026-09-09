"""Pydantic models for question and query API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Incoming query for a specific document (used with /documents/{id}/query)."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question about the document.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        """Reject whitespace-only questions."""
        if not v.strip():
            raise ValueError("Question must contain non-whitespace characters.")
        return v.strip()


class QuestionRequest(BaseModel):
    """Incoming question with document_id in body (for /questions/ask)."""

    document_id: str = Field(..., description="ID of the document to query.")
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language question about the document.",
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, v: str) -> str:
        """Reject whitespace-only questions."""
        if not v.strip():
            raise ValueError("Question must contain non-whitespace characters.")
        return v.strip()


class Citation(BaseModel):
    """A single citation pointing back to a source location in the document."""

    citation_id: str = Field(..., description="Unique ID for this citation.")
    document_id: str = Field(..., description="Parent document identifier.")
    page_number: int = Field(..., ge=1, description="1-indexed page number in the original PDF.")
    section_title: str | None = Field(None, description="Section heading or title, if detected.")
    text_snippet: str = Field(..., description="The relevant source text, table, or chart description.")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score between 0 and 1.")
    source_type: str = Field(
        ..., description="Type of source modality: 'text', 'image', or 'table'."
    )


class AnswerResponse(BaseModel):
    """Final grounded answer synthesized across multimodal agents."""

    answer: str = Field(..., description="Generated answer with inline citation markers like [1], [2].")
    citations: list[Citation] = Field(default_factory=list, description="Structured source citations.")
    agents_used: list[str] = Field(
        default_factory=list,
        description="List of sub-agents that contributed to the answer (RETRIEVAL, VISION, SQL).",
    )
    document_id: str = Field(..., description="Document identifier.")
    question: str = Field(..., description="The original user question.")


class CitationDetailResponse(BaseModel):
    """Full detail for a single citation (used by the citation-detail endpoint)."""

    citation: Citation
