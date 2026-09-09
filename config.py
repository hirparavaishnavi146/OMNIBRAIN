"""Application configuration loaded from environment variables and .env files.

Uses pydantic-settings for type-safe configuration with validation.
All secrets and tunable parameters are configured here — never hardcoded.
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for OmniBrain.

    Values are loaded from environment variables and a ``.env`` file
    (if present) in the project root.  Environment variables take
    precedence over the file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Gemini / LLM (via OpenAI-compatible endpoint) ─────────────
    gemini_api_key: str = ""
    openai_api_key: str = ""  # Kept for backward compat; gemini_api_key takes priority
    openai_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    openai_model: str = "gemini-3.6-flash"
    openai_max_tokens: int = 4096

    @property
    def effective_api_key(self) -> str:
        """Returns gemini_api_key if set, else falls back to openai_api_key."""
        return self.gemini_api_key or self.openai_api_key

    # ── Qdrant vector database ────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection_name: str = "omnibrain"
    use_qdrant: bool = True  # False → fall back to FAISS

    # ── Langfuse observability ────────────────────────────────────
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    langfuse_enabled: bool = True

    # ── CLIP (multi-modal embeddings) ─────────────────────────────
    clip_model_name: str = "openai/clip-vit-base-patch32"
    clip_embedding_dimension: int = 512

    # ── Text embedding model ──────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ── LLaVA fallback VLM ────────────────────────────────────────
    llava_endpoint: str = "http://localhost:11434"
    llava_model: str = "llava:13b"

    # ── NeMo Guardrails ───────────────────────────────────────────
    guardrails_config_dir: str = ""  # Defaults to bundled config

    # ── Self-RAG parameters ───────────────────────────────────────
    self_rag_max_retries: int = 3
    self_rag_min_confidence: float = 0.15

    # ── Chunking ──────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50

    # ── Retrieval ─────────────────────────────────────────────────
    top_k_results: int = 5
    min_relevance_score: float = 0.1

    # ── File handling ─────────────────────────────────────────────
    max_file_size_mb: int = 50
    min_image_dimension: int = 50  # ignore tiny images (px)

    # ── Storage paths ─────────────────────────────────────────────
    data_dir: Path = Path("./data")

    # ── Server ────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    max_concurrent_ingestions: int = 2

    # ── Derived paths (not from env) ──────────────────────────────
    @property
    def documents_dir(self) -> Path:
        """Root directory for per-document storage."""
        return self.data_dir / "documents"

    @property
    def database_path(self) -> Path:
        """Full path to the SQLite database file."""
        return self.data_dir / "omnibrain.db"

    @property
    def max_file_size_bytes(self) -> int:
        """Maximum upload size in bytes."""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def resolved_guardrails_config_dir(self) -> str:
        """Resolved guardrails config directory path."""
        if self.guardrails_config_dir:
            return self.guardrails_config_dir
        # Default to the bundled config alongside the guardrails module.
        return str(Path(__file__).parent / "guardrails" / "config")


def get_settings() -> Settings:
    """Factory that returns a validated ``Settings`` instance.

    Called once at startup via dependency injection; cached thereafter.
    """
    return Settings()
