"""Vision agent — analyses document charts, graphs, and images using Gemini with LLaVA fallback.

Employs CLIP cross-modal similarity to identify the most relevant extracted images,
transmits them to Gemini (or open-weight LLaVA fallback), and returns structured
visual reasoning with page-accurate citations.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import httpx
from openai import AsyncOpenAI

from omnibrain.agents.retrieval_agent import AgentResult
from omnibrain.citations.engine import CitationEngine
from omnibrain.config import Settings
from omnibrain.embeddings.engine import CLIPEmbeddingEngine
from omnibrain.exceptions import AgentError
from omnibrain.observability.tracer import LangfuseTracer
from omnibrain.retrieval.vector_store import VectorStoreBase
from omnibrain.storage.database import Database

logger = logging.getLogger(__name__)

_SUPPORTED_MEDIA_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

_MAX_IMAGES = 5


class VisionAgent:
    """Answers questions about charts, graphs, tables, and images extracted from documents."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        citation_engine: CitationEngine,
        clip_engine: CLIPEmbeddingEngine | None = None,
        vector_store: VectorStoreBase | None = None,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self._settings = settings
        self._database = database
        self._citation_engine = citation_engine
        self._clip_engine = clip_engine
        self._vector_store = vector_store
        self._tracer = tracer
        self._openai_client = AsyncOpenAI(api_key=settings.effective_api_key, base_url=settings.openai_base_url)

    async def run(
        self,
        document_id: str,
        question: str,
        *,
        trace: Any = None,
        parent_span: Any = None,
    ) -> AgentResult:
        """Select relevant images and analyze them with Gemini (or LLaVA fallback).

        Args:
            document_id: Document whose images to analyze.
            question: The user's visual/chart-related question.
            trace: Optional Langfuse trace.
            parent_span: Optional parent Langfuse span.

        Returns:
            ``AgentResult`` with visual analysis text and image-level citations.

        Raises:
            AgentError: On unrecoverable model or file-read failures.
        """
        span = None
        if self._tracer and self._tracer.enabled:
            span = self._tracer.start_span(
                "VisionAgent.run",
                trace=trace,
                parent=parent_span,
                input_data={"document_id": document_id, "question": question},
            )

        try:
            logger.info("VisionAgent running for document %s.", document_id)
            all_images = await self._database.get_images_for_document(document_id)

            if not all_images:
                logger.info("No images found in database for document %s.", document_id)
                res = AgentResult(context_text="", citations=[], agent_name="VISION")
                if self._tracer and span:
                    self._tracer.end_span(span, output="No images available")
                return res

            # Select most relevant images via CLIP cross-modal search if available
            selected_images = await self._select_relevant_images(
                document_id, question, all_images
            )

            # Read and encode images
            image_payloads: list[dict[str, Any]] = []
            valid_images: list[dict[str, Any]] = []

            for img in selected_images:
                img_path = Path(img["file_path"])
                if not img_path.exists():
                    logger.warning("Image file does not exist on disk: %s", img_path)
                    continue

                ext = img_path.suffix.lower()
                media_type = _SUPPORTED_MEDIA_TYPES.get(ext, "image/png")
                try:
                    raw_bytes = img_path.read_bytes()
                    b64_data = base64.b64encode(raw_bytes).decode("utf-8")
                    image_payloads.append(
                        {
                            "media_type": media_type,
                            "b64": b64_data,
                            "page_number": img.get("page_number", 1),
                            "image_index": img.get("image_index", 0),
                            "path": str(img_path),
                        }
                    )
                    valid_images.append(img)
                except Exception as exc:
                    logger.warning("Failed to read image %s: %s", img_path, exc)

            if not image_payloads:
                res = AgentResult(context_text="", citations=[], agent_name="VISION")
                if self._tracer and span:
                    self._tracer.end_span(span, output="No valid image files")
                return res

            # Self-RAG loop for Vision reasoning
            analysis_text = ""
            retries = 0
            while retries <= self._settings.self_rag_max_retries:
                try:
                    # Attempt Primary VLM: Gemini
                    analysis_text = await self._analyze_with_gemini_vision(
                        question, image_payloads, retries=retries, trace=trace, parent_span=span
                    )
                except Exception as gpt_err:
                    logger.warning(
                        "Gemini Vision call failed (%s), checking LLaVA availability...", gpt_err
                    )
                    # Quick health check before burning full-timeout retry cycles
                    llava_reachable = False
                    try:
                        async with httpx.AsyncClient(timeout=3.0) as hc:
                            probe = await hc.get(self._settings.llava_endpoint.rstrip("/"))
                            llava_reachable = probe.status_code < 500
                    except Exception:
                        llava_reachable = False

                    if not llava_reachable:
                        logger.warning(
                            "LLaVA endpoint %s is not reachable — skipping fallback.",
                            self._settings.llava_endpoint,
                        )
                        if retries >= self._settings.self_rag_max_retries:
                            raise AgentError(
                                f"Vision analysis failed (Gemini: {gpt_err}; LLaVA: unreachable)"
                            ) from gpt_err
                    else:
                        try:
                            # Attempt Fallback VLM: LLaVA
                            analysis_text = await self._analyze_with_llava(
                                question, image_payloads, trace=trace, parent_span=span
                            )
                        except Exception as llava_err:
                            logger.error("LLaVA fallback also failed: %s", llava_err)
                            if retries >= self._settings.self_rag_max_retries:
                                raise AgentError(
                                    f"Vision analysis failed on both Gemini ({gpt_err}) and LLaVA ({llava_err})"
                                ) from gpt_err

                # Self-RAG evaluation: check if analysis produced meaningful content
                if analysis_text and len(analysis_text.strip()) > 20:
                    break

                retries += 1
                logger.info("VisionAgent retrying reasoning (attempt %d)...", retries)

            # Build image citations
            citations = [
                self._citation_engine.build_citation_from_agent(
                    document_id=document_id,
                    page_number=max(img.get("page_number", 1), 1),
                    text_snippet=(
                        f"Visual element / Chart on Page {img.get('page_number', 1)} "
                        f"(Image {img.get('image_index', 0)})"
                    ),
                    source_type="image",
                    relevance_score=0.9,
                    section_title=f"Chart / Image (Page {img.get('page_number', 1)})",
                )
                for img in valid_images
            ]

            context_text = f"Visual Chart & Image Analysis:\n{analysis_text}"

            logger.info(
                "VisionAgent completed analysis of %d images for document %s.",
                len(valid_images),
                document_id,
            )

            result = AgentResult(
                context_text=context_text,
                citations=citations,
                agent_name="VISION",
                raw_data={"images_analyzed": len(valid_images)},
                retries=retries,
            )

            if self._tracer and span:
                self._tracer.end_span(
                    span,
                    output={
                        "images_count": len(valid_images),
                        "retries": retries,
                        "analysis_len": len(analysis_text),
                    },
                )
            return result

        except Exception as exc:
            logger.error("VisionAgent error: %s", exc, exc_info=True)
            if self._tracer and span:
                self._tracer.end_span(
                    span, output={"error": str(exc)}, level="ERROR", status_message=str(exc)
                )
            raise AgentError(f"Vision agent failed: {exc}") from exc

    async def _select_relevant_images(
        self,
        document_id: str,
        question: str,
        all_images: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Use CLIP cross-modal embedding search to select the top relevant images."""
        if self._clip_engine and self._vector_store:
            try:
                # Embed query in CLIP text space
                clip_text_vec = await self._clip_engine.embed_text(question)
                # Search image modality in vector store
                hits = await self._vector_store.search(
                    doc_id=document_id,
                    query_vector=clip_text_vec,
                    top_k=_MAX_IMAGES,
                    min_score=0.0,
                    modality_filter="image",
                )

                if hits:
                    hit_ids = {h.chunk_id for h in hits}
                    selected = [img for img in all_images if img.get("image_index") in hit_ids]
                    if selected:
                        logger.info(
                            "CLIP cross-modal search selected %d images for query.",
                            len(selected),
                        )
                        return selected
            except Exception as exc:
                logger.warning("CLIP image selection failed, falling back to top images: %s", exc)

        return all_images[:_MAX_IMAGES]

    async def _analyze_with_gemini_vision(
        self,
        question: str,
        image_payloads: list[dict[str, Any]],
        *,
        retries: int = 0,
        trace: Any = None,
        parent_span: Any = None,
    ) -> str:
        """Call Gemini Vision API (via OpenAI-compatible endpoint) with base64 images."""
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are a financial quantitative and visual analysis expert. "
                    "Analyze the provided image(s), chart(s), diagram(s), or visual tables "
                    "extracted from the corporate document to answer the question with exact numbers, "
                    "trends, axis labels, legends, and data points.\n\n"
                    f"Question: {question}\n\n"
                    f"{'Note: Be extra thorough in reading exact values from the axes and legends.' if retries > 0 else ''}"
                ),
            }
        ]

        for payload in image_payloads:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{payload['media_type']};base64,{payload['b64']}",
                        "detail": "high",
                    },
                }
            )

        gen = None
        if self._tracer and self._tracer.enabled:
            gen = self._tracer.start_generation(
                "VisionAgent.gemini_vision_call",
                trace=trace,
                parent=parent_span,
                model=self._settings.openai_model,
                input_data={"question": question, "image_count": len(image_payloads)},
            )

        try:
            response = await self._openai_client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[{"role": "user", "content": content}],
                max_tokens=self._settings.openai_max_tokens,
                temperature=0.1,
            )

            text = response.choices[0].message.content or ""

            if self._tracer and gen:
                usage = None
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                self._tracer.end_generation(gen, output=text[:500], usage=usage)

            return text
        except Exception as exc:
            if self._tracer and gen:
                self._tracer.end_generation(
                    gen, output={"error": str(exc)}, level="ERROR", status_message=str(exc)
                )
            raise

    async def _analyze_with_llava(
        self,
        question: str,
        image_payloads: list[dict[str, Any]],
        *,
        trace: Any = None,
        parent_span: Any = None,
    ) -> str:
        """Call LLaVA endpoint (Ollama-compatible API) as open-weight fallback."""
        endpoint = f"{self._settings.llava_endpoint.rstrip('/')}/api/generate"
        prompt = (
            f"Analyze this document chart/image to answer the question:\n{question}\n"
            "Provide specific values, labels, and exact readings."
        )

        b64_images = [p["b64"] for p in image_payloads[:2]]  # LLaVA typically takes 1-2 images

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(
                endpoint,
                json={
                    "model": self._settings.llava_model,
                    "prompt": prompt,
                    "images": b64_images,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")
