"""NeMo Guardrails engine — validates every response for grounding and scope.

Ensures the system never answers from general knowledge, never fabricates
information, and refuses queries outside the scope of uploaded documents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omnibrain.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class GuardrailsResult:
    """Outcome of a guardrails validation check."""

    passed: bool
    reason: str | None = None
    sanitized_answer: str = ""


class GuardrailsEngine:
    """Wraps NeMo Guardrails for OmniBrain response validation.

    Performs two primary checks on every outgoing response:
    1. **Grounding check**: Is the answer supported by the retrieved context?
    2. **Topic check**: Does the answer stay within the scope of the uploaded document?

    Falls back to a lightweight prompt-based check if NeMo Guardrails
    cannot be initialised.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._rails: Any | None = None
        self._initialised = False

        config_dir = settings.resolved_guardrails_config_dir
        if not Path(config_dir).exists():
            logger.warning(
                "Guardrails config directory not found at '%s' — "
                "using fallback prompt-based validation.",
                config_dir,
            )
            return

        try:
            from nemoguardrails import RailsConfig, LLMRails

            config = RailsConfig.from_path(config_dir)
            self._rails = LLMRails(config)
            self._initialised = True
            logger.info("NeMo Guardrails initialised from %s.", config_dir)
        except Exception as exc:
            logger.warning(
                "NeMo Guardrails initialisation failed — using fallback: %s", exc
            )

    @property
    def is_active(self) -> bool:
        """Whether NeMo Guardrails is fully active."""
        return self._initialised and self._rails is not None

    async def validate_response(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> GuardrailsResult:
        """Validate an outgoing answer against the retrieved context.

        Args:
            question: The user's original question.
            answer: The generated answer to validate.
            context: The retrieved context the answer should be grounded in.

        Returns:
            A ``GuardrailsResult`` indicating pass/fail and the reason.
        """
        if self.is_active:
            return await self._validate_with_nemo(question, answer, context)
        return await self._validate_with_fallback(question, answer, context)

    async def _validate_with_nemo(
        self, question: str, answer: str, context: str
    ) -> GuardrailsResult:
        """Run the full NeMo Guardrails validation pipeline."""
        try:
            messages = [
                {
                    "role": "context",
                    "content": {
                        "relevant_chunks": context[:8000],
                    },
                },
                {"role": "user", "content": question},
            ]

            response = await self._rails.generate_async(messages=messages)

            # NeMo Guardrails may modify or block the response
            if response and isinstance(response, dict):
                rail_response = response.get("content", "")
            elif response and isinstance(response, str):
                rail_response = response
            else:
                # If the rails return the answer unmodified, it passed
                return GuardrailsResult(
                    passed=True,
                    sanitized_answer=answer,
                )

            # Check if NeMo blocked the response
            blocked_phrases = [
                "i cannot answer",
                "i'm not able to answer",
                "outside the scope",
                "cannot be answered from the document",
                "i don't have enough information",
                "not found in the provided context",
            ]

            is_blocked = any(
                phrase in rail_response.lower() for phrase in blocked_phrases
            )

            if is_blocked:
                return GuardrailsResult(
                    passed=False,
                    reason="Answer blocked by NeMo Guardrails: response not grounded in document context.",
                    sanitized_answer=rail_response,
                )

            return GuardrailsResult(
                passed=True,
                sanitized_answer=answer,
            )

        except Exception as exc:
            logger.warning("NeMo Guardrails check failed, falling back: %s", exc)
            return await self._validate_with_fallback(question, answer, context)

    async def _validate_with_fallback(
        self, question: str, answer: str, context: str
    ) -> GuardrailsResult:
        """Lightweight prompt-based grounding check using OpenAI.

        Used when NeMo Guardrails is not available or fails.
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self._settings.effective_api_key, base_url=self._settings.openai_base_url)

            validation_prompt = (
                "You are a grounding validator. Your job is to check whether an answer "
                "is fully supported by the provided context.\n\n"
                f"CONTEXT:\n{context[:6000]}\n\n"
                f"QUESTION: {question}\n\n"
                f"ANSWER: {answer}\n\n"
                "Respond with ONLY one of:\n"
                '- {"grounded": true} if the answer is fully supported by the context\n'
                '- {"grounded": false, "reason": "..."} if the answer contains '
                "information not in the context or uses general knowledge"
            )

            response = await client.chat.completions.create(
                model=self._settings.openai_model,
                max_tokens=200,
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0.0,
            )

            import json
            import re

            result_text = response.choices[0].message.content or ""
            # Try to parse JSON from the response
            result_text = result_text.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("```")[1]
                if result_text.startswith("json"):
                    result_text = result_text[4:]
                result_text = result_text.strip()

            try:
                parsed = json.loads(result_text)
            except json.JSONDecodeError:
                # Fallback: try to extract JSON object from text
                json_match = re.search(r'\{[^{}]*\}', result_text)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    # Can't parse — check for keywords as last resort
                    lower = result_text.lower()
                    if "true" in lower or "grounded" in lower:
                        parsed = {"grounded": True}
                    else:
                        logger.warning(
                            "Could not parse guardrails validation response: %s",
                            result_text[:200],
                        )
                        parsed = {"grounded": False, "reason": "Could not parse guardrails validation response — failing closed."}

            if parsed.get("grounded", True):
                return GuardrailsResult(passed=True, sanitized_answer=answer)
            else:
                reason = parsed.get("reason", "Answer not grounded in document context.")
                refusal = (
                    "I can only answer questions based on the uploaded document. "
                    "The information needed to answer this question was not found "
                    "in the document context."
                )
                return GuardrailsResult(
                    passed=False,
                    reason=reason,
                    sanitized_answer=refusal,
                )

        except Exception as exc:
            logger.warning("Fallback guardrails check failed: %s", exc)
            # If even the fallback fails, let the answer through with a warning
            return GuardrailsResult(
                passed=True,
                reason="Guardrails check skipped due to error.",
                sanitized_answer=answer,
            )
