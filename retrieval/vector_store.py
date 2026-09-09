"""Vector store abstraction — Qdrant (primary) with FAISS fallback.

Provides a unified interface (``VectorStoreBase``) with two concrete
implementations.  The factory ``create_vector_store`` selects the backend
based on configuration and connectivity.
"""

from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from omnibrain.config import Settings
from omnibrain.exceptions import RetrievalError

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single retrieval hit from the vector store."""

    chunk_id: int
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStoreBase(abc.ABC):
    """Abstract interface for vector storage backends."""

    @abc.abstractmethod
    async def add_vectors(
        self,
        doc_id: str,
        vectors: np.ndarray,
        ids: list[int],
        *,
        modality: str = "text",
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add embedding vectors with their IDs and optional metadata."""

    @abc.abstractmethod
    async def search(
        self,
        doc_id: str,
        query_vector: np.ndarray,
        top_k: int,
        min_score: float = 0.0,
        modality_filter: str | None = None,
    ) -> list[SearchResult]:
        """Retrieve the top-k most similar vectors, optionally filtered by modality."""

    @abc.abstractmethod
    async def delete_collection(self, doc_id: str) -> None:
        """Remove all data for a document."""

    @abc.abstractmethod
    async def health_check(self) -> bool:
        """Check connectivity to the backend."""

    @abc.abstractmethod
    def index_exists(self, doc_id: str) -> bool:
        """Check whether data exists for the document."""


# ═════════════════════════════════════════════════════════════════════
#  Qdrant Implementation
# ═════════════════════════════════════════════════════════════════════


class QdrantVectorStore(VectorStoreBase):
    """Qdrant-backed vector store.

    Uses a single Qdrant collection with payload-based filtering
    for ``document_id`` and ``modality`` (text / image).
    """

    def __init__(self, settings: Settings) -> None:
        self._url = settings.qdrant_url
        self._api_key = settings.qdrant_api_key or None
        self._collection_name = settings.qdrant_collection_name
        self._text_dim = settings.embedding_dimension
        self._clip_dim = settings.clip_embedding_dimension
        self._client: Any = None
        self._initialised = False

    async def _ensure_client(self) -> Any:
        """Lazy-initialise the async Qdrant client and collection."""
        if self._client is not None:
            return self._client

        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import (
                Distance,
                FieldCondition,
                MatchValue,
                NamedVector,
                PointStruct,
                VectorParams,
            )

            self._client = AsyncQdrantClient(
                url=self._url,
                api_key=self._api_key,
                timeout=30,
            )

            # Create collection with named vectors for text and image
            collections = await self._client.get_collections()
            existing_names = [c.name for c in collections.collections]

            if self._collection_name not in existing_names:
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config={
                        "text": VectorParams(
                            size=self._text_dim, distance=Distance.COSINE
                        ),
                        "image": VectorParams(
                            size=self._clip_dim, distance=Distance.COSINE
                        ),
                    },
                )
                logger.info(
                    "Created Qdrant collection '%s' with text(%d) + image(%d) vectors.",
                    self._collection_name,
                    self._text_dim,
                    self._clip_dim,
                )

            self._initialised = True
            logger.info("Qdrant client connected to %s.", self._url)
            return self._client
        except Exception as exc:
            logger.error("Failed to connect to Qdrant at %s: %s", self._url, exc)
            raise RetrievalError(
                f"Qdrant connection failed: {exc}",
                details={"url": self._url},
            ) from exc

    async def add_vectors(
        self,
        doc_id: str,
        vectors: np.ndarray,
        ids: list[int],
        *,
        modality: str = "text",
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert vectors into the Qdrant collection."""
        client = await self._ensure_client()

        from qdrant_client.models import PointStruct

        vector_name = modality  # "text" or "image"
        points = []
        for i, (vec, point_id) in enumerate(zip(vectors, ids)):
            payload = {
                "document_id": doc_id,
                "modality": modality,
                "chunk_id": point_id,
            }
            if metadatas and i < len(metadatas):
                payload.update(metadatas[i])

            # Use a stable string-based point ID to avoid collisions
            qdrant_id = f"{doc_id}_{modality}_{point_id}"

            points.append(
                PointStruct(
                    id=abs(hash(qdrant_id)) % (2**63),
                    vector={vector_name: vec.tolist()},
                    payload=payload,
                )
            )

        # Batch upsert (max 100 points per batch)
        batch_size = 100
        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start : batch_start + batch_size]
            try:
                await client.upsert(
                    collection_name=self._collection_name,
                    points=batch,
                )
            except Exception as exc:
                raise RetrievalError(
                    f"Qdrant upsert failed: {exc}",
                    details={"doc_id": doc_id, "batch_size": len(batch)},
                ) from exc

        logger.info(
            "Upserted %d %s vectors for document %s into Qdrant.",
            len(points),
            modality,
            doc_id,
        )

    async def search(
        self,
        doc_id: str,
        query_vector: np.ndarray,
        top_k: int,
        min_score: float = 0.0,
        modality_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search Qdrant with document and optional modality filtering."""
        client = await self._ensure_client()

        from qdrant_client.models import FieldCondition, Filter, MatchValue, NamedVector

        # Build filter conditions
        must_conditions = [
            FieldCondition(key="document_id", match=MatchValue(value=doc_id))
        ]
        if modality_filter:
            must_conditions.append(
                FieldCondition(key="modality", match=MatchValue(value=modality_filter))
            )

        # Determine which named vector to search
        vector_name = modality_filter or "text"
        query_vec = query_vector.flatten().tolist()

        try:
            hits = await client.search(
                collection_name=self._collection_name,
                query_vector=NamedVector(name=vector_name, vector=query_vec),
                query_filter=Filter(must=must_conditions),
                limit=top_k,
                score_threshold=min_score if min_score > 0 else None,
            )
        except Exception as exc:
            raise RetrievalError(f"Qdrant search failed: {exc}") from exc

        results = []
        for hit in hits:
            results.append(
                SearchResult(
                    chunk_id=hit.payload.get("chunk_id", 0),
                    score=float(hit.score),
                    text="",  # Populated by caller from DB
                    metadata={
                        k: v
                        for k, v in hit.payload.items()
                        if k not in ("document_id", "modality", "chunk_id")
                    },
                )
            )

        logger.debug(
            "Qdrant search for doc=%s returned %d results (top_k=%d).",
            doc_id,
            len(results),
            top_k,
        )
        return results

    async def delete_collection(self, doc_id: str) -> None:
        """Delete all points for a specific document from Qdrant."""
        client = await self._ensure_client()

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        try:
            await client.delete(
                collection_name=self._collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id", match=MatchValue(value=doc_id)
                        )
                    ]
                ),
            )
            logger.info("Deleted Qdrant vectors for document %s.", doc_id)
        except Exception as exc:
            logger.warning("Qdrant delete failed for %s: %s", doc_id, exc)

    async def health_check(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            client = await self._ensure_client()
            collections = await client.get_collections()
            return True
        except Exception:
            return False

    def index_exists(self, doc_id: str) -> bool:
        """Qdrant always has the collection; return True if client is ready."""
        return self._initialised


# ═════════════════════════════════════════════════════════════════════
#  FAISS Fallback Implementation
# ═════════════════════════════════════════════════════════════════════


class FAISSVectorStore(VectorStoreBase):
    """FAISS-backed vector store with per-document indexes.

    Used as a local-dev fallback when Qdrant is not available.
    Each document gets its own ``IndexIDMap2(IndexFlatIP)`` index
    persisted to disk.
    """

    def __init__(self, settings: Settings) -> None:
        self._text_dim: int = settings.embedding_dimension
        self._clip_dim: int = settings.clip_embedding_dimension
        self._documents_dir: Path = settings.documents_dir
        self._lock = asyncio.Lock()
        self._indexes: dict[str, Any] = {}  # key: "{doc_id}_{modality}"

    def _index_key(self, doc_id: str, modality: str = "text") -> str:
        """Composite key for per-document, per-modality indexes."""
        return f"{doc_id}_{modality}"

    def _index_path(self, doc_id: str, modality: str = "text") -> Path:
        """On-disk path for a document's FAISS index."""
        suffix = f"index_{modality}.faiss"
        return self._documents_dir / doc_id / suffix

    def _get_dim(self, modality: str) -> int:
        """Return the dimension for the given modality."""
        return self._clip_dim if modality == "image" else self._text_dim

    def _create_index(self, modality: str = "text") -> Any:
        """Create a fresh FAISS index for cosine similarity."""
        import faiss

        dim = self._get_dim(modality)
        base = faiss.IndexFlatIP(dim)
        return faiss.IndexIDMap2(base)

    async def _load_index(self, doc_id: str, modality: str = "text") -> None:
        """Load or create an index for a document + modality."""
        import faiss

        key = self._index_key(doc_id, modality)
        path = self._index_path(doc_id, modality)

        # Also check for legacy "index.faiss" filename (backward compatibility)
        legacy_path = self._documents_dir / doc_id / "index.faiss"

        if path.exists():
            index = await asyncio.to_thread(faiss.read_index, str(path))
            self._indexes[key] = index
            logger.info(
                "Loaded FAISS %s index for %s (%d vectors).",
                modality,
                doc_id,
                index.ntotal,
            )
        elif modality == "text" and not path.exists() and legacy_path.exists():
            # Migrate legacy index file to new naming scheme
            logger.info(
                "Found legacy index.faiss for %s — migrating to %s.",
                doc_id,
                path.name,
            )
            legacy_path.rename(path)
            index = await asyncio.to_thread(faiss.read_index, str(path))
            self._indexes[key] = index
            logger.info(
                "Loaded FAISS %s index for %s (%d vectors) from legacy file.",
                modality,
                doc_id,
                index.ntotal,
            )
        else:
            self._indexes[key] = self._create_index(modality)
            logger.debug("Created empty FAISS %s index for %s.", modality, doc_id)

    async def _save_index(self, doc_id: str, modality: str = "text") -> None:
        """Persist a FAISS index to disk."""
        import faiss

        key = self._index_key(doc_id, modality)
        index = self._indexes.get(key)
        if index is None:
            return
        path = self._index_path(doc_id, modality)
        path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(faiss.write_index, index, str(path))

    async def add_vectors(
        self,
        doc_id: str,
        vectors: np.ndarray,
        ids: list[int],
        *,
        modality: str = "text",
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add vectors to the per-document FAISS index."""
        expected_dim = self._get_dim(modality)
        if vectors.shape[1] != expected_dim:
            raise RetrievalError(
                f"Vector dimension mismatch: expected {expected_dim}, got {vectors.shape[1]}."
            )

        async with self._lock:
            key = self._index_key(doc_id, modality)
            if key not in self._indexes:
                await self._load_index(doc_id, modality)

            index = self._indexes[key]
            id_array = np.array(ids, dtype=np.int64)
            vecs = np.ascontiguousarray(vectors, dtype=np.float32)

            try:
                index.add_with_ids(vecs, id_array)
            except Exception as exc:
                raise RetrievalError(f"FAISS add_vectors failed: {exc}") from exc

            logger.info(
                "Added %d %s vectors to FAISS for %s (total: %d).",
                len(ids),
                modality,
                doc_id,
                index.ntotal,
            )

        await self._save_index(doc_id, modality)

    async def search(
        self,
        doc_id: str,
        query_vector: np.ndarray,
        top_k: int,
        min_score: float = 0.0,
        modality_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search the FAISS index for a document."""
        modality = modality_filter or "text"
        key = self._index_key(doc_id, modality)

        if key not in self._indexes:
            await self._load_index(doc_id, modality)

        index = self._indexes.get(key)
        if index is None or index.ntotal == 0:
            logger.warning("No %s vectors in FAISS for document %s.", modality, doc_id)
            return []

        qvec = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)

        try:
            scores, ids_arr = await asyncio.to_thread(
                index.search, qvec, min(top_k, index.ntotal)
            )
        except Exception as exc:
            raise RetrievalError(f"FAISS search failed: {exc}") from exc

        results: list[SearchResult] = []
        for score, chunk_id in zip(scores[0], ids_arr[0]):
            if chunk_id == -1:
                continue
            if score < min_score:
                continue
            results.append(
                SearchResult(
                    chunk_id=int(chunk_id),
                    score=float(score),
                    text="",
                    metadata={},
                )
            )

        logger.debug(
            "FAISS %s search for %s returned %d results.",
            modality,
            doc_id,
            len(results),
        )
        return results

    async def delete_collection(self, doc_id: str) -> None:
        """Remove all FAISS indexes for a document."""
        for modality in ("text", "image"):
            key = self._index_key(doc_id, modality)
            self._indexes.pop(key, None)
            path = self._index_path(doc_id, modality)
            if path.exists():
                path.unlink()
        logger.info("Deleted FAISS indexes for document %s.", doc_id)

    async def health_check(self) -> bool:
        """FAISS is always available (in-process)."""
        return True

    def index_exists(self, doc_id: str) -> bool:
        """Check if any FAISS index file exists for the document."""
        return self._index_path(doc_id, "text").exists()


# ═════════════════════════════════════════════════════════════════════
#  Factory
# ═════════════════════════════════════════════════════════════════════


async def create_vector_store(settings: Settings) -> VectorStoreBase:
    """Factory that selects the vector store backend.

    Tries Qdrant first if ``use_qdrant`` is True.  Falls back to FAISS
    if Qdrant is unreachable.
    """
    if settings.use_qdrant:
        qdrant_store = QdrantVectorStore(settings)
        try:
            healthy = await qdrant_store.health_check()
            if healthy:
                logger.info("Using Qdrant vector store at %s.", settings.qdrant_url)
                return qdrant_store
            else:
                logger.warning(
                    "Qdrant at %s is not healthy — falling back to FAISS.",
                    settings.qdrant_url,
                )
        except Exception as exc:
            logger.warning(
                "Qdrant connection failed (%s) — falling back to FAISS.", exc
            )

    logger.info("Using FAISS vector store (local).")
    return FAISSVectorStore(settings)
