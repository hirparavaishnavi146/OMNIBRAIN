"""PDF parsing — extracts text, images, and tables from PDF files.

Uses PyMuPDF (fitz) as the primary extraction engine.  Provides
structured output suitable for the chunking and ingestion pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf as fitz  # PyMuPDF

from omnibrain.config import Settings
from omnibrain.exceptions import EmptyDocumentError, PDFParsingError

logger = logging.getLogger(__name__)


@dataclass
class PageText:
    """Parsed text content for a single page."""

    page_number: int  # 1-indexed
    raw_text: str
    blocks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExtractedImage:
    """An image extracted from the PDF."""

    page_number: int
    image_index: int
    image_bytes: bytes
    extension: str  # e.g. "png", "jpeg"
    width: int
    height: int
    xref: int  # PDF internal object reference (used for dedup)


@dataclass
class ExtractedTable:
    """A table detected and extracted from the PDF."""

    page_number: int
    table_index: int
    headers: list[str]
    data: list[list[str]]


@dataclass
class ParseResult:
    """Complete result of parsing a PDF document."""

    page_count: int
    pages: list[PageText]
    images: list[ExtractedImage]
    tables: list[ExtractedTable]


class PDFParser:
    """Extracts structured content from PDF files using PyMuPDF."""

    def __init__(self, settings: Settings) -> None:
        self._min_image_dim = settings.min_image_dimension

    def parse(self, pdf_path: Path) -> ParseResult:
        """Parse a PDF file and extract all content types.

        Args:
            pdf_path: Path to the PDF file on disk.

        Returns:
            A ``ParseResult`` containing pages, images, and tables.

        Raises:
            PDFParsingError: If the PDF is corrupted or cannot be opened.
            EmptyDocumentError: If no content could be extracted.
        """
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            raise PDFParsingError(
                f"Cannot open PDF: {exc}",
                details={"path": str(pdf_path)},
            ) from exc

        try:
            pages = self._extract_text(doc)
            images = self._extract_images(doc)
            tables = self._extract_tables(doc)
        except PDFParsingError:
            raise
        except Exception as exc:
            raise PDFParsingError(
                f"Error during PDF parsing: {exc}",
                details={"path": str(pdf_path)},
            ) from exc
        finally:
            doc.close()

        # Check for completely empty documents.
        has_text = any(p.raw_text.strip() for p in pages)
        has_images = len(images) > 0
        has_tables = len(tables) > 0
        if not (has_text or has_images or has_tables):
            raise EmptyDocumentError(
                "PDF contains no extractable text, images, or tables.",
                details={"path": str(pdf_path), "page_count": len(pages)},
            )

        logger.info(
            "Parsed PDF: %d pages, %d images, %d tables.",
            len(pages),
            len(images),
            len(tables),
        )
        return ParseResult(
            page_count=len(pages),
            pages=pages,
            images=images,
            tables=tables,
        )

    def _extract_text(self, doc: fitz.Document) -> list[PageText]:
        """Extract page-by-page text with structural block information."""
        pages: list[PageText] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            raw_text = page.get_text("text")
            # Get dict output for structural info (font sizes, spans).
            try:
                page_dict = page.get_text("dict")
                blocks = page_dict.get("blocks", [])
            except Exception:
                blocks = []

            pages.append(
                PageText(
                    page_number=page_num + 1,  # 1-indexed
                    raw_text=raw_text,
                    blocks=blocks,
                )
            )
        return pages

    def _extract_images(self, doc: fitz.Document) -> list[ExtractedImage]:
        """Extract embedded images, deduplicated by xref, filtering tiny ones."""
        images: list[ExtractedImage] = []
        seen_xrefs: set[int] = set()
        global_index = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                image_list = page.get_images(full=True)
            except Exception as exc:
                logger.warning(
                    "Failed to extract images from page %d: %s", page_num + 1, exc
                )
                continue

            for img_info in image_list:
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                except Exception as exc:
                    logger.warning(
                        "Failed to extract image xref=%d: %s", xref, exc
                    )
                    continue

                width = base_image.get("width", 0)
                height = base_image.get("height", 0)

                # Skip tiny images (icons, bullets, decorations).
                if width < self._min_image_dim or height < self._min_image_dim:
                    continue

                images.append(
                    ExtractedImage(
                        page_number=page_num + 1,
                        image_index=global_index,
                        image_bytes=base_image["image"],
                        extension=base_image.get("ext", "png"),
                        width=width,
                        height=height,
                        xref=xref,
                    )
                )
                global_index += 1

        return images

    def _extract_tables(self, doc: fitz.Document) -> list[ExtractedTable]:
        """Extract tables using PyMuPDF's built-in table detection."""
        tables: list[ExtractedTable] = []
        global_index = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                found_tables = page.find_tables()
            except Exception as exc:
                logger.warning(
                    "Table detection failed on page %d: %s", page_num + 1, exc
                )
                continue

            for tab in found_tables:
                try:
                    raw_data = tab.extract()
                except Exception as exc:
                    logger.warning(
                        "Table extraction failed on page %d: %s", page_num + 1, exc
                    )
                    continue

                if not raw_data or len(raw_data) < 2:
                    continue  # Need at least a header row + one data row.

                # First row as headers; clean None values.
                headers = [
                    str(cell).strip() if cell else f"col_{i}"
                    for i, cell in enumerate(raw_data[0])
                ]
                data = [
                    [str(cell).strip() if cell else "" for cell in row]
                    for row in raw_data[1:]
                ]

                tables.append(
                    ExtractedTable(
                        page_number=page_num + 1,
                        table_index=global_index,
                        headers=headers,
                        data=data,
                    )
                )
                global_index += 1

        return tables
