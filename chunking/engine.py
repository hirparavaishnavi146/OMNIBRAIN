"""Recursive semantic chunking engine.

Splits document text into overlapping, semantically coherent chunks
while preserving page-number and section metadata on every chunk.

Strategy (applied recursively):
1. Split by page boundaries.
2. Within each page, split by section headers (detected via font-size).
3. Within each section, split by paragraph boundaries (``\\n\\n``).
4. If a paragraph exceeds ``chunk_size``, split by sentences.
5. Apply ``chunk_overlap`` token overlap between consecutive chunks
   within the same section.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from omnibrain.config import Settings

logger = logging.getLogger(__name__)

# Rough approximation: 1 token ≈ 4 characters for English text.
_CHARS_PER_TOKEN = 4

# Sentence boundary regex — handles abbreviations reasonably well.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class TextChunk:
    """A single text chunk with provenance metadata."""

    text: str
    page_number: int
    section_title: str | None
    chunk_index: int
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the chunk for database insertion."""
        return {
            "text": self.text,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "chunk_index": self.chunk_index,
            "metadata": {
                "document_id": self.document_id,
                **self.metadata,
            },
        }


@dataclass
class PageContent:
    """Structured text content for a single page.

    ``blocks`` contains dicts from PyMuPDF's ``get_text("dict")`` output,
    each with ``lines`` → ``spans`` carrying font-size information.
    """

    page_number: int
    raw_text: str
    blocks: list[dict[str, Any]] = field(default_factory=list)


class ChunkingEngine:
    """Splits parsed document text into semantically coherent chunks."""

    def __init__(self, settings: Settings) -> None:
        self._chunk_size = settings.chunk_size * _CHARS_PER_TOKEN
        self._chunk_overlap = settings.chunk_overlap * _CHARS_PER_TOKEN
        self._min_chunk_size = settings.min_chunk_size * _CHARS_PER_TOKEN

    def chunk_document(
        self, document_id: str, pages: list[PageContent]
    ) -> list[TextChunk]:
        """Chunk an entire document, returning a flat list of ``TextChunk``s.

        Args:
            document_id: Unique identifier of the document.
            pages: One ``PageContent`` per page, in order.

        Returns:
            Ordered list of ``TextChunk`` objects with global chunk indexes.
        """
        all_chunks: list[TextChunk] = []
        global_index = 0

        for page in pages:
            sections = self._split_into_sections(page)
            for section_title, section_text in sections:
                paragraphs = self._split_into_paragraphs(section_text)
                raw_pieces: list[str] = []
                for para in paragraphs:
                    if self._char_len(para) > self._chunk_size:
                        raw_pieces.extend(self._split_into_sentences(para))
                    else:
                        raw_pieces.append(para)

                merged = self._merge_with_overlap(raw_pieces)

                for text in merged:
                    text = text.strip()
                    if len(text) < self._min_chunk_size:
                        # Too small to be useful — try appending to previous.
                        if all_chunks and all_chunks[-1].page_number == page.page_number:
                            all_chunks[-1].text += " " + text
                            continue
                        elif not text:
                            continue
                    all_chunks.append(
                        TextChunk(
                            text=text,
                            page_number=page.page_number,
                            section_title=section_title,
                            chunk_index=global_index,
                            document_id=document_id,
                        )
                    )
                    global_index += 1

        logger.info(
            "Chunked document %s into %d chunks from %d pages.",
            document_id,
            len(all_chunks),
            len(pages),
        )
        return all_chunks

    # ── Internal splitting helpers ─────────────────────────────────

    def _split_into_sections(
        self, page: PageContent
    ) -> list[tuple[str | None, str]]:
        """Detect section boundaries using font-size heuristics.

        Returns a list of ``(section_title, section_text)`` tuples.
        If no structural blocks are available, treats the entire page as
        one section with ``section_title = None``.
        """
        if not page.blocks:
            return [(None, page.raw_text)]

        # Determine the dominant (most common) font size on the page.
        font_sizes: list[float] = []
        for block in page.blocks:
            if block.get("type") != 0:  # type 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    font_sizes.append(span.get("size", 12.0))

        if not font_sizes:
            return [(None, page.raw_text)]

        dominant_size = max(set(font_sizes), key=font_sizes.count)
        heading_threshold = dominant_size * 1.15  # 15 % larger → heading

        sections: list[tuple[str | None, str]] = []
        current_title: str | None = None
        current_text_parts: list[str] = []

        for block in page.blocks:
            if block.get("type") != 0:
                continue
            block_text_parts: list[str] = []
            is_heading = False
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    line_text += span.get("text", "")
                    if span.get("size", 12.0) >= heading_threshold:
                        is_heading = True
                block_text_parts.append(line_text)

            block_text = " ".join(block_text_parts).strip()
            if not block_text:
                continue

            if is_heading and len(block_text) < 200:
                # Start a new section.
                if current_text_parts:
                    sections.append((current_title, "\n".join(current_text_parts)))
                current_title = block_text
                current_text_parts = []
            else:
                current_text_parts.append(block_text)

        if current_text_parts:
            sections.append((current_title, "\n".join(current_text_parts)))

        return sections if sections else [(None, page.raw_text)]

    @staticmethod
    def _split_into_paragraphs(text: str) -> list[str]:
        """Split text on double-newline boundaries."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs if paragraphs else [text]

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text into sentences using regex heuristics."""
        sentences = _SENTENCE_RE.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _merge_with_overlap(self, pieces: list[str]) -> list[str]:
        """Merge small pieces into chunks of ~``chunk_size`` with overlap."""
        if not pieces:
            return []

        merged: list[str] = []
        current = pieces[0]

        for piece in pieces[1:]:
            combined = current + " " + piece
            if self._char_len(combined) <= self._chunk_size:
                current = combined
            else:
                merged.append(current)
                # Overlap: keep the tail of the previous chunk.
                if self._chunk_overlap > 0:
                    tail = current[-self._chunk_overlap :]
                    # Snap to word boundary — avoid mid-word splits
                    space_idx = tail.find(" ")
                    if space_idx > 0:
                        tail = tail[space_idx + 1 :]
                    current = tail + " " + piece
                else:
                    current = piece

        if current.strip():
            merged.append(current)

        return merged

    @staticmethod
    def _char_len(text: str) -> int:
        """Return the character length of text."""
        return len(text)
