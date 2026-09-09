"""FastAPI dependency injection — shared singletons for the application.

All heavy objects (database, embedding models, vector store, agents, LangGraph orchestrator,
guardrails, and Langfuse tracer) are initialized once at startup and made available to request handlers.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from omnibrain.agents.graph import OmniBrainGraphOrchestrator
from omnibrain.agents.retrieval_agent import RetrievalAgent
from omnibrain.agents.sql_agent import SQLAgent
from omnibrain.agents.supervisor import SupervisorAgent
from omnibrain.agents.vision_agent import VisionAgent
from omnibrain.chunking.engine import ChunkingEngine
from omnibrain.citations.engine import CitationEngine
from omnibrain.config import Settings, get_settings
from omnibrain.embeddings.engine import CLIPEmbeddingEngine, EmbeddingEngine
from omnibrain.generation.engine import GenerationEngine
from omnibrain.guardrails.engine import GuardrailsEngine
from omnibrain.ingestion.parser import PDFParser
from omnibrain.ingestion.processor import IngestionProcessor
from omnibrain.observability.tracer import LangfuseTracer
from omnibrain.retrieval.vector_store import VectorStoreBase, create_vector_store
from omnibrain.storage.database import Database
from omnibrain.storage.file_store import FileStore

logger = logging.getLogger(__name__)


@lru_cache
def get_cached_settings() -> Settings:
    """Return a cached ``Settings`` instance (singleton)."""
    return get_settings()


class AppState:
    """Holds all shared application state — initialized once at startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings)
        self.file_store = FileStore(settings)
        self.embedding_engine = EmbeddingEngine(settings)
        self.clip_engine = CLIPEmbeddingEngine(settings)
        self.chunking_engine = ChunkingEngine(settings)
        self.citation_engine = CitationEngine()
        self.parser = PDFParser(settings)
        self.tracer = LangfuseTracer(settings)
        self.guardrails_engine = GuardrailsEngine(settings)
        self.generation_engine = GenerationEngine(settings=settings, tracer=self.tracer)

        # Vector store will be initialized asynchronously during startup()
        self.vector_store: VectorStoreBase | None = None
        self.retrieval_agent: RetrievalAgent | None = None
        self.vision_agent: VisionAgent | None = None
        self.sql_agent: SQLAgent | None = None
        self.supervisor_agent: SupervisorAgent | None = None
        self.graph_orchestrator: OmniBrainGraphOrchestrator | None = None
        self.ingestion_processor: IngestionProcessor | None = None

    async def startup(self) -> None:
        """Perform async initialization (database, vector store, agents, LangGraph)."""
        self.file_store.ensure_directories()
        await self.database.initialize()
        reset_count = await self.database.reset_stuck_documents()
        if reset_count:
            logger.warning("Reset %d stuck documents on startup.", reset_count)

        # Initialize vector store (Qdrant primary with FAISS fallback)
        self.vector_store = await create_vector_store(self.settings)

        # Initialize Ingestion Processor
        self.ingestion_processor = IngestionProcessor(
            settings=self.settings,
            database=self.database,
            file_store=self.file_store,
            parser=self.parser,
            chunking_engine=self.chunking_engine,
            embedding_engine=self.embedding_engine,
            clip_engine=self.clip_engine,
            vector_store=self.vector_store,
        )

        # Initialize Specialized Agents with Self-RAG & Langfuse tracing
        self.retrieval_agent = RetrievalAgent(
            settings=self.settings,
            embedding_engine=self.embedding_engine,
            vector_store=self.vector_store,
            citation_engine=self.citation_engine,
            database=self.database,
            tracer=self.tracer,
        )
        self.vision_agent = VisionAgent(
            settings=self.settings,
            database=self.database,
            citation_engine=self.citation_engine,
            clip_engine=self.clip_engine,
            vector_store=self.vector_store,
            tracer=self.tracer,
        )
        self.sql_agent = SQLAgent(
            settings=self.settings,
            database=self.database,
            citation_engine=self.citation_engine,
            tracer=self.tracer,
        )
        self.supervisor_agent = SupervisorAgent(
            settings=self.settings,
            database=self.database,
            tracer=self.tracer,
        )

        # Initialize LangGraph Orchestration State Machine
        self.graph_orchestrator = OmniBrainGraphOrchestrator(
            settings=self.settings,
            supervisor_agent=self.supervisor_agent,
            retrieval_agent=self.retrieval_agent,
            vision_agent=self.vision_agent,
            sql_agent=self.sql_agent,
            generation_engine=self.generation_engine,
            guardrails_engine=self.guardrails_engine,
            tracer=self.tracer,
        )

        logger.info("OmniBrain AppState initialized successfully with LangGraph.")

    async def shutdown(self) -> None:
        """Teardown connections and flush Langfuse traces."""
        if self.tracer:
            self.tracer.shutdown()
        await self.database.close()
        logger.info("OmniBrain AppState shut down cleanly.")
