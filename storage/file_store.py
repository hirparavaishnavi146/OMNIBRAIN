"""On-disk file management for PDFs, extracted images, and FAISS indexes.

Directory layout per document::

    data/documents/{doc_id}/
        original.pdf
        images/
            page3_img0.png
            page5_img1.jpeg
        index.faiss
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from omnibrain.config import Settings

logger = logging.getLogger(__name__)


class FileStore:
    """Manages on-disk files for uploaded documents and their artifacts."""

    def __init__(self, settings: Settings) -> None:
        self._documents_dir: Path = settings.documents_dir

    def _doc_dir(self, doc_id: str) -> Path:
        """Return the per-document directory, creating it if needed."""
        d = self._documents_dir / doc_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── PDF ────────────────────────────────────────────────────────

    def save_pdf(self, doc_id: str, pdf_bytes: bytes) -> Path:
        """Persist the uploaded PDF and return its path."""
        path = self._doc_dir(doc_id) / "original.pdf"
        path.write_bytes(pdf_bytes)
        logger.info("Saved PDF for %s (%d bytes).", doc_id, len(pdf_bytes))
        return path

    def get_pdf_path(self, doc_id: str) -> Path:
        """Return the path to a document's PDF.

        Raises:
            FileNotFoundError: If the PDF file does not exist on disk.
        """
        path = self._documents_dir / doc_id / "original.pdf"
        if not path.exists():
            raise FileNotFoundError(f"PDF not found for document {doc_id}")
        return path

    # ── Images ─────────────────────────────────────────────────────

    def save_image(
        self,
        doc_id: str,
        page_number: int,
        image_index: int,
        image_bytes: bytes,
        extension: str,
    ) -> Path:
        """Save an extracted image and return its path."""
        images_dir = self._doc_dir(doc_id) / "images"
        images_dir.mkdir(exist_ok=True)
        filename = f"page{page_number}_img{image_index}.{extension}"
        path = images_dir / filename
        path.write_bytes(image_bytes)
        logger.debug("Saved image %s for document %s.", filename, doc_id)
        return path

    def get_image_path(self, doc_id: str, page_number: int, image_index: int) -> Path | None:
        """Find an image file on disk by its coordinates.

        Returns ``None`` if the image does not exist (any extension).
        """
        images_dir = self._documents_dir / doc_id / "images"
        if not images_dir.exists():
            return None
        prefix = f"page{page_number}_img{image_index}"
        for f in images_dir.iterdir():
            if f.stem == prefix:
                return f
        return None

    # ── FAISS index ────────────────────────────────────────────────

    def get_index_path(self, doc_id: str) -> Path:
        """Return the FAISS index path for a document (may not exist yet)."""
        return self._doc_dir(doc_id) / "index.faiss"

    def index_exists(self, doc_id: str) -> bool:
        """Check whether a saved FAISS index exists for the document."""
        return (self._documents_dir / doc_id / "index.faiss").exists()

    # ── Cleanup ────────────────────────────────────────────────────

    def delete_document(self, doc_id: str) -> None:
        """Remove all on-disk files for a document."""
        doc_dir = self._documents_dir / doc_id
        if doc_dir.exists():
            shutil.rmtree(doc_dir)
            logger.info("Deleted files for document %s.", doc_id)

    def ensure_directories(self) -> None:
        """Create the top-level data directories if they don't exist."""
        self._documents_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured data directories exist at %s.", self._documents_dir)
