"""Retrieval agent — performs semantic search over document chunks with Self-RAG.

Embeds the user's question, searches the vector store (Qdrant with FAISS fallback)
for relevant text chunks, evaluates retrieval quality, rewrites the query if
confidence is below threshold, and retries up to configured max retries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from omnibrain.citations.engine import CitationEngine
from omnibrain.config import Settings
from omnibrain.embeddings.engine import EmbeddingEngine
from omnibrain.exceptions import AgentError
from omnibrain.observability.tracer import LangfuseTracer
from omnibrain.retrieval.vector_store import SearchResult, VectorStoreBase
from omnibrain.schemas.questions import Citation
from omnibrain.storage.database import Database

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Standardised output from any sub-agent."""

    context_text: str
    citations: list[Citation] = field(default_factory=list)
    agent_name: str = "RETRIEVAL"
    raw_data: Any = None
    retries: int = 0
    agent_errored: bool = False
    error_message: str | None = None


class RetrievalAgent:
    """Semantic Q&A agent that searches document chunks with Self-RAG self-correction."""

    def __init__(
        self,
        settings: Settings,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStoreBase,
        citation_engine: CitationEngine,
        database: Database,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self._settings = settings
        self._embedding_engine = embedding_engine
        self._vector_store = vector_store
        self._citation_engine = citation_engine
        self._database = database
        self._tracer = tracer
        self._openai_client = AsyncOpenAI(api_key=settings.effective_api_key, base_url=settings.openai_base_url)
        self._max_retries = settings.self_rag_max_retries
        self._min_confidence = settings.self_rag_min_confidence

    async def run(
        self,
        document_id: str,
        question: str,
        *,
        trace: Any = None,
        parent_span: Any = None,
    ) -> AgentResult:
        """Embed the question, retrieve chunks, and execute Self-RAG correction loop if needed.

        Args:
            document_id: The document to search within.
            question: The user's natural-language question.
            trace: Optional Langfuse trace.
            parent_span: Optional parent Langfuse span.

        Returns:
            An ``AgentResult`` with the retrieved context and citations.

        Raises:
            AgentError: If embedding or retrieval fails unrecoverably.
        """
        span = None
        if self._tracer and self._tracer.enabled:
            span = self._tracer.start_span(
                "RetrievalAgent.run",
                trace=trace,
                parent=parent_span,
                input_data={"document_id": document_id, "question": question},
            )

        try:
            logger.info("RetrievalAgent running for document %s.", document_id)

            current_query = question
            retries = 0
            best_results: list[SearchResult] = []
            best_score = 0.0

            while retries <= self._max_retries:
                # 1. Embed query
                query_vector = await self._embedding_engine.embed_query(current_query)

                # 2. Search vector store
                search_results = await self._vector_store.search(
                    doc_id=document_id,
                    query_vector=query_vector,
                    top_k=self._settings.top_k_results,
                    min_score=self._settings.min_relevance_score,
                    modality_filter="text",
                )

                top_score = search_results[0].score if search_results else 0.0

                if top_score > best_score:
                    best_score = top_score
                    best_results = search_results

                # Hydrate text/metadata from database (per-call, no shared mutation)
                for r in search_results:
                    if not r.text:
                        chunk = await self._database.get_chunk_by_id(r.chunk_id)
                        if chunk:
                            r.text = chunk["text"]
                            r.metadata = {
                                "page_number": chunk["page_number"],
                                "section_title": chunk.get("section_title"),
                                "document_id": document_id,
                                "chunk_index": chunk["chunk_index"],
                            }

                # Evaluate confidence: Is top score >= threshold and do we have results?
                is_confident = (
                    len(search_results) > 0 and top_score >= self._min_confidence
                )

                if is_confident or retries >= self._max_retries:
                    if retries > 0:
                        logger.info(
                            "Self-RAG completed after %d retries (top_score=%.3f, is_confident=%s).",
                            retries,
                            best_score,
                            is_confident,
                        )
                    break

                # Self-RAG correction: Rewrite query for better semantic matching
                logger.info(
                    "Self-RAG triggering query rewrite (attempt %d/%d, score=%.3f < threshold=%.3f).",
                    retries + 1,
                    self._max_retries,
                    top_score,
                    self._min_confidence,
                )
                current_query = await self._rewrite_query(
                    original_question=question,
                    attempted_query=current_query,
                    top_score=top_score,
                    trace=trace,
                    parent_span=span,
                )
                retries += 1

            if not best_results:
                logger.info("No relevant chunks found for document %s.", document_id)
                res = AgentResult(
                    context_text="",
                    citations=[],
                    agent_name="RETRIEVAL",
                    retries=retries,
                )
                if self._tracer and span:
                    self._tracer.end_span(span, output="No chunks found")
                return res

            citations = self._citation_engine.build_citations(document_id, best_results)

            context_parts: list[str] = []
            for i, result in enumerate(best_results, start=1):
                section = result.metadata.get("section_title", "General")
                page = result.metadata.get("page_number", "?")
                text_snippet = result.text or result.metadata.get("text", "")
                context_parts.append(
                    f"[{i}] (Page {page}, Section: {section})\n{text_snippet}"
                )

            context_text = "\n\n".join(context_parts)

            logger.info(
                "RetrievalAgent found %d results (retries=%d) for document %s.",
                len(best_results),
                retries,
                document_id,
            )
            result = AgentResult(
                context_text=context_text,
                citations=citations,
                agent_name="RETRIEVAL",
                raw_data=best_results,
                retries=retries,
            )

            if self._tracer and span:
                self._tracer.end_span(
                    span,
                    output={
                        "num_results": len(best_results),
                        "retries": retries,
                        "top_score": best_score,
                    },
                )
            return result

        except Exception as exc:
            logger.error("RetrievalAgent failed: %s", exc, exc_info=True)
            if self._tracer and span:
                self._tracer.end_span(
                    span, output={"error": str(exc)}, level="ERROR", status_message=str(exc)
                )
            raise AgentError(f"Retrieval agent failed: {exc}") from exc

    async def _rewrite_query(
        self,
        original_question: str,
        attempted_query: str,
        top_score: float,
        *,
        trace: Any = None,
        parent_span: Any = None,
    ) -> str:
        """Use Gemini to rewrite a search query for better semantic density and keyword coverage."""
        prompt = (
            "You are a Self-RAG query reformulator for document retrieval.\n"
            f"Original Question: {original_question}\n"
            f"Previous Query: {attempted_query}\n"
            f"Previous Search Match Score: {top_score:.3f} (Too low, need better semantic match)\n\n"
            "Reformulate this into a more direct, keyword-dense search query that targets "
            "the essential facts, corporate finance terminology, and potential passage text "
            "in a financial/technical PDF document.\n"
            "Return ONLY the reformulated query string without quotes or explanations."
        )

        gen = None
        if self._tracer and self._tracer.enabled:
            gen = self._tracer.start_generation(
                "RetrievalAgent.rewrite_query",
                trace=trace,
                parent=parent_span,
                model=self._settings.openai_model,
                input_data=prompt,
            )

        try:
            response = await self._openai_client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
            )
            rewritten = response.choices[0].message.content or original_question
            rewritten = rewritten.strip().strip('"').strip("'")

            if self._tracer and gen:
                usage = None
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                self._tracer.end_generation(gen, output=rewritten, usage=usage)

            return rewritten
        except Exception as exc:
            logger.warning("Query rewrite failed, falling back to original: %s", exc)
            if self._tracer and gen:
                self._tracer.end_generation(
                    gen, output=original_question, level="ERROR", status_message=str(exc)
                )
            return original_question
