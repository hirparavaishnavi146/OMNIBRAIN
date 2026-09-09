"""Citation engine — maps search results back to source locations.

Produces deduplicated ``Citation`` objects that the generation engine
embeds as inline references (``[1]``, ``[2]``, …) in the final answer.
"""

from __future__ import annotations

import hashlib
import logging

from omnibrain.retrieval.vector_store import SearchResult
from omnibrain.schemas.questions import Citation

logger = logging.getLogger(__name__)


class CitationEngine:
    """Builds citations from retrieval results."""

    def build_citations(
        self,
        document_id: str,
        results: list[SearchResult],
    ) -> list[Citation]:
        """Convert a list of ``SearchResult`` into deduplicated ``Citation`` objects.

        Citations that overlap (same page + same section) are merged,
        keeping the higher relevance score.

        Args:
            document_id: The source document ID.
            results: Retrieval results with populated text and metadata.

        Returns:
            Ordered list of ``Citation`` objects.
        """
        seen: dict[str, Citation] = {}

        for result in results:
            page = result.metadata.get("page_number", 1)
            section = result.metadata.get("section_title")
            dedup_key = f"{page}:{section or ''}"

            if dedup_key in seen:
                existing = seen[dedup_key]
                if result.score > existing.relevance_score:
                    # Replace with higher-scoring result.
                    seen[dedup_key] = self._make_citation(
                        document_id, result, page, section
                    )
                continue

            seen[dedup_key] = self._make_citation(
                document_id, result, page, section
            )

        citations = sorted(
            seen.values(),
            key=lambda c: c.relevance_score,
            reverse=True,
        )
        logger.info(
            "Built %d citations for document %s from %d results.",
            len(citations),
            document_id,
            len(results),
        )
        return citations

    def build_citation_from_agent(
        self,
        document_id: str,
        page_number: int,
        text_snippet: str,
        source_type: str,
        relevance_score: float = 1.0,
        section_title: str | None = None,
    ) -> Citation:
        """Create a single citation for agent-generated content (vision, SQL).

        Useful when an agent produces output that doesn't come from
        the vector store.
        """
        citation_id = self._generate_id(document_id, page_number, text_snippet)
        return Citation(
            citation_id=citation_id,
            document_id=document_id,
            page_number=page_number,
            section_title=section_title,
            text_snippet=text_snippet[:500],
            relevance_score=relevance_score,
            source_type=source_type,
        )

    # ── Private helpers ────────────────────────────────────────────

    @staticmethod
    def _make_citation(
        document_id: str,
        result: SearchResult,
        page: int,
        section: str | None,
    ) -> Citation:
        cid = CitationEngine._generate_id(document_id, page, result.text)
        return Citation(
            citation_id=cid,
            document_id=document_id,
            page_number=page,
            section_title=section,
            text_snippet=result.text[:500],
            relevance_score=round(result.score, 4),
            source_type="text",
        )

    @staticmethod
    def _generate_id(document_id: str, page: int, text: str) -> str:
        """Deterministic short citation ID."""
        raw = f"{document_id}:{page}:{text[:100]}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
