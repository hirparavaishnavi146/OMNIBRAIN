"""Supervisor agent — routes questions to specialised sub-agents using Gemini.

Examines the user's query and the document's capabilities (text chunks, embedded images,
extracted tables) to classify which sub-agent(s) to dispatch: RETRIEVAL, VISION, SQL.
Supports multi-agent routing for compound queries (e.g., comparing tables and charts).
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from omnibrain.config import Settings
from omnibrain.observability.tracer import LangfuseTracer
from omnibrain.storage.database import Database

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Routes questions to the appropriate sub-agents via classify_route()."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self._settings = settings
        self._database = database
        self._tracer = tracer
        self._openai_client = AsyncOpenAI(api_key=settings.effective_api_key, base_url=settings.openai_base_url)

    async def classify_route(
        self,
        document_id: str,
        question: str,
        *,
        trace: Any = None,
        parent_span: Any = None,
    ) -> list[str]:
        """Classify which agents are needed for the question based on available document data."""
        # Query document capabilities from DB
        images = await self._database.get_images_for_document(document_id)
        tables = await self._database.get_tables_for_document(document_id)
        has_images = len(images) > 0
        has_tables = len(tables) > 0

        agents_to_run = await self._classify(
            question, has_images, has_tables, trace=trace, parent_span=parent_span
        )

        # Filter out agents for data types absent in the document
        if not has_images and "VISION" in agents_to_run:
            agents_to_run.remove("VISION")
        if not has_tables and "SQL" in agents_to_run:
            agents_to_run.remove("SQL")

        # Guarantee at least RETRIEVAL runs if nothing else was chosen
        if not agents_to_run:
            agents_to_run = ["RETRIEVAL"]

        return agents_to_run

    async def _classify(
        self,
        question: str,
        has_images: bool,
        has_tables: bool,
        *,
        trace: Any = None,
        parent_span: Any = None,
    ) -> list[str]:
        """Use Gemini to decide which agents should handle the question."""
        available = ["RETRIEVAL"]
        if has_images:
            available.append("VISION")
        if has_tables:
            available.append("SQL")

        if len(available) == 1:
            return ["RETRIEVAL"]

        prompt = (
            "You are the central routing supervisor for OmniBrain, an enterprise multi-modal financial RAG system.\n\n"
            f"Available Specialized Agents: {available}\n"
            "- RETRIEVAL: Searches dense text chunks for general qualitative text, policy, narratives, and footnotes.\n"
            "- VISION: Performs multi-modal visual reasoning over charts, graphs, diagrams, and visual plots.\n"
            "- SQL: Queries structured tabular data via Text-to-SQL for quantitative figures, row lookups, and numeric comparisons.\n\n"
            f"User Question: \"{question}\"\n\n"
            "Task: Identify all agents needed. If a question asks to compare a chart with a table, select BOTH VISION and SQL.\n"
            "Respond ONLY with a JSON object in this exact schema:\n"
            '{"agents": ["RETRIEVAL", "SQL"]}'
        )

        gen = None
        if self._tracer and self._tracer.enabled:
            gen = self._tracer.start_generation(
                "SupervisorAgent.classify",
                trace=trace,
                parent=parent_span,
                model=self._settings.openai_model,
                input_data={"question": question, "available": available},
            )

        try:
            response = await self._openai_client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100,
            )

            text = response.choices[0].message.content or "{}"
            text = text.strip()

            # Try direct JSON parse first
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Fallback: extract JSON object from response text using regex
                import re
                json_match = re.search(r'\{[^{}]*\}', text)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    # Last resort: look for agent names directly in text
                    found = [a for a in ["RETRIEVAL", "VISION", "SQL"] if a in text.upper()]
                    parsed = {"agents": found if found else ["RETRIEVAL"]}

            agents = parsed.get("agents", ["RETRIEVAL"])

            if isinstance(agents, list):
                valid = [a for a in agents if a in {"RETRIEVAL", "VISION", "SQL"}]
                if valid:
                    if self._tracer and gen:
                        self._tracer.end_generation(gen, output=valid)
                    return valid

            return ["RETRIEVAL"]

        except Exception as exc:
            logger.warning("Supervisor classification error, falling back to RETRIEVAL: %s", exc)
            if self._tracer and gen:
                self._tracer.end_generation(
                    gen, output=["RETRIEVAL"], level="WARNING", status_message=str(exc)
                )
            return ["RETRIEVAL"]
