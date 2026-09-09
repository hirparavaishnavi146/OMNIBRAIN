"""RAG answer generation engine using Gemini with grounded citations.

Assembles multimodal context, constructs a strictly grounded prompt with citation
markers ([1], [2], ...), calls Gemini, and traces token usage + latency via Langfuse.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from omnibrain.config import Settings
from omnibrain.exceptions import LLMError
from omnibrain.observability.tracer import LangfuseTracer

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are OmniBrain, an enterprise-grade multimodal financial and data science analysis assistant.
Your job is to answer questions strictly and exclusively based on the provided context extracted from the document.

Rules:
1. Answer using ONLY the information in the context below. Do NOT use outside general knowledge.
2. Use inline citations like [1], [2], etc. to reference specific pieces of context. Number them in the order they appear in the context.
3. If the context contains multiple sources (text passages, SQL tabular results, chart visual analysis), synthesize them into a coherent, comprehensive explanation.
4. When comparing tables and charts, explicitly highlight any data agreements or discrepancies.
5. If the context does not contain enough information to answer the question, state: "Based on the available document content, I don't have enough information to fully answer this question."
6. Maintain quantitative precision: report exact figures, percentages, dates, and units as given in the context.
7. Be structured, concise, and rigorous.
"""


class GenerationEngine:
    """Produces grounded answers by calling Gemini with multimodal agent context."""

    def __init__(
        self,
        settings: Settings,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self._settings = settings
        self._tracer = tracer
        self._openai_client = AsyncOpenAI(api_key=settings.effective_api_key, base_url=settings.openai_base_url)

    async def generate_raw(
        self,
        question: str,
        context: str,
        *,
        retry_count: int = 0,
        trace: Any = None,
        parent_span: Any = None,
    ) -> str:
        """Call Gemini with question and context string to produce raw answer text."""
        gen = None
        if self._tracer and self._tracer.enabled:
            gen = self._tracer.start_generation(
                "GenerationEngine.generate_raw",
                trace=trace,
                parent=parent_span,
                model=self._settings.openai_model,
                input_data={"question": question, "context_len": len(context), "retry_count": retry_count},
            )

        system_instruction = _SYSTEM_PROMPT
        if retry_count > 0:
            system_instruction += (
                "\nCRITICAL: Previous response was flagged by Guardrails. "
                "Ensure EVERY claim is strictly supported by the context below and cited."
            )

        user_content = f"CONTEXT:\n{context}\n\nQUESTION: {question}"

        try:
            response = await self._openai_client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=self._settings.openai_max_tokens,
                temperature=0.1,
            )

            answer_text = response.choices[0].message.content or ""

            if self._tracer and gen:
                usage = None
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                self._tracer.end_generation(gen, output=answer_text, usage=usage)

            return answer_text

        except Exception as exc:
            logger.error("Gemini generation failed: %s", exc, exc_info=True)
            if self._tracer and gen:
                self._tracer.end_generation(
                    gen, output={"error": str(exc)}, level="ERROR", status_message=str(exc)
                )
            raise LLMError(f"Answer generation failed: {exc}") from exc
