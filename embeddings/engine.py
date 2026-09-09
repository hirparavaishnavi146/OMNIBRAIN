"""Embedding generation — text (sentence-transformers) and image (CLIP).

The ``EmbeddingEngine`` handles text embeddings using sentence-transformers.
The ``CLIPEmbeddingEngine`` handles image and cross-modal text embeddings
using OpenAI's CLIP model via HuggingFace Transformers.

Both produce L2-normalised float32 vectors suitable for cosine similarity
via inner product.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

import numpy as np

from omnibrain.config import Settings
from omnibrain.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Generates dense vector embeddings for text chunks and queries."""

    def __init__(self, settings: Settings) -> None:
        self._model_name: str = settings.embedding_model
        self._dimension: int = settings.embedding_dimension
        self._model: Any | None = None  # Lazy-loaded SentenceTransformer

    def _load_model(self) -> None:
        """Lazy-load the SentenceTransformer model on first use."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading text embedding model: %s …", self._model_name)
            self._model = SentenceTransformer(self._model_name)
            logger.info("Text embedding model loaded (dim=%d).", self._dimension)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load text embedding model '{self._model_name}': {exc}"
            ) from exc

    @property
    def dimension(self) -> int:
        """The dimensionality of the text embedding vectors."""
        return self._dimension

    def embed_chunks_sync(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of text chunks synchronously.

        Args:
            texts: List of text strings to embed.

        Returns:
            A numpy array of shape ``(len(texts), dimension)`` with float32 dtype,
            L2-normalized for cosine similarity via inner product.

        Raises:
            EmbeddingError: If encoding fails.
        """
        self._load_model()
        if not texts:
            return np.empty((0, self._dimension), dtype=np.float32)
        try:
            embeddings = self._model.encode(  # type: ignore[union-attr]
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return np.ascontiguousarray(embeddings, dtype=np.float32)
        except Exception as exc:
            raise EmbeddingError(f"Text embedding failed: {exc}") from exc

    async def embed_chunks(self, texts: list[str]) -> np.ndarray:
        """Async wrapper around ``embed_chunks_sync``.

        Runs the CPU-bound encoding in a thread to avoid blocking the
        event loop.
        """
        return await asyncio.to_thread(self.embed_chunks_sync, texts)

    def embed_query_sync(self, query: str) -> np.ndarray:
        """Embed a single query string synchronously.

        Returns:
            A 1-D numpy array of shape ``(dimension,)``.
        """
        self._load_model()
        try:
            embedding = self._model.encode(  # type: ignore[union-attr]
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return np.ascontiguousarray(embedding[0], dtype=np.float32)
        except Exception as exc:
            raise EmbeddingError(f"Query embedding failed: {exc}") from exc

    async def embed_query(self, query: str) -> np.ndarray:
        """Async wrapper around ``embed_query_sync``."""
        return await asyncio.to_thread(self.embed_query_sync, query)


class CLIPEmbeddingEngine:
    """Generates CLIP embeddings for images and cross-modal text queries.

    Uses HuggingFace's ``transformers`` with the CLIP model to produce
    embeddings in a shared vision-language space, enabling cross-modal
    similarity search (e.g., text query → relevant chart/image).
    """

    def __init__(self, settings: Settings) -> None:
        self._model_name: str = settings.clip_model_name
        self._dimension: int = settings.clip_embedding_dimension
        self._model: Any | None = None
        self._processor: Any | None = None

    def _load_model(self) -> None:
        """Lazy-load the CLIP model and processor on first use."""
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info("Loading CLIP model: %s …", self._model_name)
            self._model = CLIPModel.from_pretrained(self._model_name)
            self._processor = CLIPProcessor.from_pretrained(self._model_name)
            self._model.eval()
            logger.info("CLIP model loaded (dim=%d).", self._dimension)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load CLIP model '{self._model_name}': {exc}"
            ) from exc

    @property
    def dimension(self) -> int:
        """The dimensionality of CLIP embedding vectors."""
        return self._dimension

    def embed_image_sync(self, image_bytes: bytes) -> np.ndarray:
        """Embed a single image synchronously.

        Args:
            image_bytes: Raw image bytes (PNG, JPEG, etc.).

        Returns:
            A 1-D numpy array of shape ``(dimension,)``, L2-normalized.

        Raises:
            EmbeddingError: If image processing or encoding fails.
        """
        self._load_model()
        try:
            import torch
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self._processor(images=image, return_tensors="pt")  # type: ignore[misc]

            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)  # type: ignore[union-attr]

            # L2 normalize
            embedding = outputs[0].numpy().astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"CLIP image embedding failed: {exc}") from exc

    async def embed_image(self, image_bytes: bytes) -> np.ndarray:
        """Async wrapper around ``embed_image_sync``."""
        return await asyncio.to_thread(self.embed_image_sync, image_bytes)

    def embed_images_batch_sync(self, images_bytes: list[bytes]) -> np.ndarray:
        """Embed a batch of images synchronously.

        Args:
            images_bytes: List of raw image bytes.

        Returns:
            Array of shape ``(len(images_bytes), dimension)``, L2-normalized.
        """
        self._load_model()
        if not images_bytes:
            return np.empty((0, self._dimension), dtype=np.float32)
        try:
            import torch
            from PIL import Image

            images = [
                Image.open(io.BytesIO(b)).convert("RGB") for b in images_bytes
            ]
            inputs = self._processor(images=images, return_tensors="pt")  # type: ignore[misc]

            with torch.no_grad():
                outputs = self._model.get_image_features(**inputs)  # type: ignore[union-attr]

            embeddings = outputs.numpy().astype(np.float32)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            return embeddings / norms
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"CLIP batch image embedding failed: {exc}") from exc

    async def embed_images_batch(self, images_bytes: list[bytes]) -> np.ndarray:
        """Async wrapper around ``embed_images_batch_sync``."""
        return await asyncio.to_thread(self.embed_images_batch_sync, images_bytes)

    def embed_text_sync(self, text: str) -> np.ndarray:
        """Embed text using CLIP's text encoder (for cross-modal search).

        Args:
            text: Query text to embed in the CLIP space.

        Returns:
            A 1-D numpy array of shape ``(dimension,)``, L2-normalized.
        """
        self._load_model()
        try:
            import torch

            inputs = self._processor(  # type: ignore[misc]
                text=[text], return_tensors="pt", padding=True, truncation=True
            )

            with torch.no_grad():
                outputs = self._model.get_text_features(**inputs)  # type: ignore[union-attr]

            embedding = outputs[0].numpy().astype(np.float32)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"CLIP text embedding failed: {exc}") from exc

    async def embed_text(self, text: str) -> np.ndarray:
        """Async wrapper around ``embed_text_sync``."""
        return await asyncio.to_thread(self.embed_text_sync, text)
