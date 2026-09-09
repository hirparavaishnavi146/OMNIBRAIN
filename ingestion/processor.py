"""Ingestion orchestrator — runs the multimodal document processing pipeline.

Pipeline stages: PARSE → CHUNK → EMBED → READY.
Extracts text, images, and tables. Generates CLIP embeddings for images and
dense text embeddings for chunks, upserting all into Qdrant/FAISS with modality metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from pathlib import Path

from omnibrain.chunking.engine import ChunkingEngine, PageContent
from omnibrain.config import Settings
from omnibrain.embeddings.engine import CLIPEmbeddingEngine, EmbeddingEngine
from omnibrain.exceptions import DuplicateDocumentError, IngestionError
from omnibrain.ingestion.parser import PDFParser
from omnibrain.retrieval.vector_store import VectorStoreBase
from omnibrain.schemas.documents import DocumentStatus
from omnibrain.storage.database import Database
from omnibrain.storage.file_store import FileStore

logger = logging.getLogger(__name__)


def compute_file_hash(data: bytes) -> str:
    """Return the SHA-256 hex digest of raw file bytes."""
    return hashlib.sha256(data).hexdigest()


class IngestionProcessor:
    """Orchestrates the end-to-end multimodal document ingestion pipeline."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        file_store: FileStore,
        parser: PDFParser,
        chunking_engine: ChunkingEngine,
        embedding_engine: EmbeddingEngine,
        clip_engine: CLIPEmbeddingEngine,
        vector_store: VectorStoreBase,
    ) -> None:
        self._settings = settings
        self._database = database
        self._file_store = file_store
        self._parser = parser
        self._chunking_engine = chunking_engine
        self._embedding_engine = embedding_engine
        self._clip_engine = clip_engine
        self._vector_store = vector_store
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_ingestions)

    async def ingest(self, doc_id: str, filename: str, pdf_bytes: bytes) -> None:
        """Run the full ingestion pipeline for a document.

        Designed to be called as a background task. Limits concurrency via a semaphore.
        """
        async with self._semaphore:
            try:
                await self._run_pipeline(doc_id, filename, pdf_bytes)
            except Exception as exc:
                logger.error(
                    "Ingestion failed for document %s: %s", doc_id, exc, exc_info=True
                )
                await self._database.update_document_status(
                    doc_id,
                    DocumentStatus.FAILED.value,
                    error_message=str(exc),
                )

    async def _run_pipeline(
        self, doc_id: str, filename: str, pdf_bytes: bytes
    ) -> None:
        """Execute each pipeline stage sequentially with atomic status updates."""

        # ── Dedup check ────────────────────────────────────────────
        file_hash = compute_file_hash(pdf_bytes)
        existing = await self._database.get_document_by_hash(file_hash)
        if existing and existing["id"] != doc_id:
            raise DuplicateDocumentError(
                f"This document has already been uploaded as '{existing['filename']}'.",
                details={"existing_document_id": existing["id"]},
            )

        # ── Save PDF to disk ──────────────────────────────────────
        pdf_path = await asyncio.to_thread(
            self._file_store.save_pdf, doc_id, pdf_bytes
        )

        # ── Create DB record ──────────────────────────────────────
        await self._database.create_document(doc_id, filename, file_hash)

        # ── STAGE 1: PARSING ──────────────────────────────────────
        await self._database.update_document_status(
            doc_id, DocumentStatus.PARSING.value
        )
        logger.info("Stage 1/3: Parsing document %s (pages, images, tables).", doc_id)

        parse_result = await asyncio.to_thread(self._parser.parse, pdf_path)

        await self._database.update_document_status(
            doc_id,
            DocumentStatus.PARSING.value,
            page_count=parse_result.page_count,
        )

        # Save extracted images to disk + DB
        extracted_images_data = []
        for img in parse_result.images:
            img_path = await asyncio.to_thread(
                self._file_store.save_image,
                doc_id,
                img.page_number,
                img.image_index,
                img.image_bytes,
                img.extension,
            )
            await self._database.insert_image(
                document_id=doc_id,
                page_number=img.page_number,
                image_index=img.image_index,
                file_path=str(img_path),
            )
            extracted_images_data.append(
                (img.image_index, img.image_bytes, img.page_number)
            )

        # Save extracted tables to DB (creates dynamic SQLite query tables)
        for table in parse_result.tables:
            await self._database.insert_table(
                document_id=doc_id,
                page_number=table.page_number,
                table_index=table.table_index,
                headers=table.headers,
                data=table.data,
            )

        # ── STAGE 2: CHUNKING ────────────────────────────────────
        await self._database.update_document_status(
            doc_id, DocumentStatus.CHUNKING.value
        )
        logger.info("Stage 2/3: Chunking document %s.", doc_id)

        page_contents = [
            PageContent(
                page_number=p.page_number,
                raw_text=p.raw_text,
                blocks=p.blocks,
            )
            for p in parse_result.pages
        ]

        chunks = await asyncio.to_thread(
            self._chunking_engine.chunk_document, doc_id, page_contents
        )

        if not chunks:
            # Handle image/table only documents gracefully
            logger.warning("No text chunks generated for document %s.", doc_id)

        if chunks:
            # Persist chunks to DB
            await self._database.insert_chunks(
                doc_id, [c.to_dict() for c in chunks]
            )

        # ── STAGE 3: MULTI-MODAL EMBEDDING ────────────────────────
        await self._database.update_document_status(
            doc_id, DocumentStatus.EMBEDDING.value
        )
        logger.info("Stage 3/3: Generating text + CLIP image embeddings for %s.", doc_id)

        # 3a. Embed text chunks
        if chunks:
            texts = [c.text for c in chunks]
            text_vectors = await self._embedding_engine.embed_chunks(texts)

            db_chunks = await self._database.get_chunks_for_document(doc_id)
            chunk_ids = [c["id"] for c in db_chunks]
            metadatas = [
                {
                    "page_number": c["page_number"],
                    "section_title": c.get("section_title") or "General",
                    "text": c["text"][:300],
                }
                for c in db_chunks
            ]

            await self._vector_store.add_vectors(
                doc_id=doc_id,
                vectors=text_vectors,
                ids=chunk_ids,
                modality="text",
                metadatas=metadatas,
            )

        # 3b. Embed images with CLIP
        image_embeddings_generated = False
        if extracted_images_data:
            img_bytes_list = [item[1] for item in extracted_images_data]
            img_indices = [item[0] for item in extracted_images_data]
            img_pages = [item[2] for item in extracted_images_data]

            try:
                clip_vectors = await self._clip_engine.embed_images_batch(img_bytes_list)
                img_metadatas = [
                    {"page_number": p, "image_index": idx}
                    for p, idx in zip(img_pages, img_indices)
                ]

                await self._vector_store.add_vectors(
                    doc_id=doc_id,
                    vectors=clip_vectors,
                    ids=img_indices,
                    modality="image",
                    metadatas=img_metadatas,
                )
                image_embeddings_generated = True
                logger.info(
                    "Upserted %d CLIP image embeddings for doc %s.",
                    len(extracted_images_data),
                    doc_id,
                )
            except Exception as exc:
                logger.error(
                    "CLIP image embedding FAILED for doc %s: %s",
                    doc_id,
                    exc,
                    exc_info=True,
                )

        # ── DONE ─────────────────────────────────────────────────
        await self._database.update_document_status(
            doc_id, DocumentStatus.READY.value
        )
        logger.info(
            "Document %s ingestion complete — status READY (image_embeddings=%s).",
            doc_id,
            image_embeddings_generated,
        )
