"""Langfuse tracing wrapper for OmniBrain.

Provides a singleton ``LangfuseTracer`` that instruments every LLM call,
agent decision, and tool invocation with structured traces.  Falls back
gracefully to no-op logging when Langfuse is unreachable.
"""

from __future__ import annotations

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

from omnibrain.config import Settings

logger = logging.getLogger(__name__)


class _NoOpSpan:
    """Stub span returned when Langfuse is disabled or unavailable."""

    def end(self, **kwargs: Any) -> None:
        pass

    def update(self, **kwargs: Any) -> None:
        pass


class _NoOpGeneration:
    """Stub generation returned when Langfuse is disabled or unavailable."""

    def end(self, **kwargs: Any) -> None:
        pass

    def update(self, **kwargs: Any) -> None:
        pass


class LangfuseTracer:
    """Wraps the Langfuse Python SDK for OmniBrain observability.

    Every LLM call, agent decision, and tool invocation is traced with
    token usage, latency, and full execution context.

    If Langfuse credentials are missing or the service is unreachable,
    all methods degrade gracefully to no-ops with a logged warning.
    """

    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.langfuse_enabled
        self._client: Any = None

        if not self._enabled:
            logger.info("Langfuse tracing disabled by configuration.")
            return

        if not settings.langfuse_public_key or not settings.langfuse_secret_key:
            logger.warning(
                "Langfuse keys not configured — tracing disabled. "
                "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable."
            )
            self._enabled = False
            return

        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info("Langfuse tracer initialized (host=%s).", settings.langfuse_host)
        except Exception as exc:
            logger.warning("Failed to initialize Langfuse — tracing disabled: %s", exc)
            self._enabled = False

    @property
    def enabled(self) -> bool:
        """Whether tracing is active."""
        return self._enabled and self._client is not None

    # ── Trace lifecycle ───────────────────────────────────────────

    def start_trace(
        self,
        name: str,
        *,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Any:
        """Create a new top-level trace.

        Returns a Langfuse trace object, or ``None`` if tracing is off.
        """
        if not self.enabled:
            return None
        try:
            return self._client.trace(
                name=name,
                metadata=metadata or {},
                user_id=user_id,
                session_id=session_id,
            )
        except Exception as exc:
            logger.warning("Langfuse trace creation failed: %s", exc)
            return None

    # ── Span lifecycle ────────────────────────────────────────────

    def start_span(
        self,
        name: str,
        *,
        trace: Any = None,
        parent: Any = None,
        metadata: dict[str, Any] | None = None,
        input_data: Any = None,
    ) -> Any:
        """Create a span within a trace.

        Args:
            name: Descriptive name for the span.
            trace: The parent trace object.
            parent: A parent span (for nesting).
            metadata: Arbitrary metadata dict.
            input_data: Input data for the span.

        Returns:
            A Langfuse span object, or a no-op stub.
        """
        if not self.enabled:
            return _NoOpSpan()
        try:
            target = parent or trace
            if target is None:
                return _NoOpSpan()
            return target.span(
                name=name,
                metadata=metadata or {},
                input=input_data,
            )
        except Exception as exc:
            logger.warning("Langfuse span creation failed: %s", exc)
            return _NoOpSpan()

    def end_span(
        self,
        span: Any,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        status_message: str | None = None,
        level: str = "DEFAULT",
    ) -> None:
        """End a span with output data."""
        if isinstance(span, _NoOpSpan):
            return
        try:
            span.end(
                output=output,
                metadata=metadata,
                status_message=status_message,
                level=level,
            )
        except Exception as exc:
            logger.warning("Langfuse span end failed: %s", exc)

    # ── Generation lifecycle (LLM calls) ──────────────────────────

    def start_generation(
        self,
        name: str,
        *,
        trace: Any = None,
        parent: Any = None,
        model: str | None = None,
        input_data: Any = None,
        model_parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Record the start of an LLM generation call.

        Args:
            name: Name for the generation (e.g., "supervisor_classify").
            trace: The parent trace.
            parent: A parent span.
            model: Model identifier (e.g., "gemini-3.6-flash").
            input_data: The prompt / messages sent to the model.
            model_parameters: Temperature, max_tokens, etc.
            metadata: Arbitrary metadata.

        Returns:
            A Langfuse generation object, or a no-op stub.
        """
        if not self.enabled:
            return _NoOpGeneration()
        try:
            target = parent or trace
            if target is None:
                return _NoOpGeneration()
            return target.generation(
                name=name,
                model=model,
                input=input_data,
                model_parameters=model_parameters or {},
                metadata=metadata or {},
            )
        except Exception as exc:
            logger.warning("Langfuse generation start failed: %s", exc)
            return _NoOpGeneration()

    def end_generation(
        self,
        generation: Any,
        *,
        output: Any = None,
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
        level: str = "DEFAULT",
        status_message: str | None = None,
    ) -> None:
        """Record the completion of an LLM generation call.

        Args:
            generation: The generation object from ``start_generation``.
            output: The model's response text.
            usage: Token usage dict ``{prompt_tokens, completion_tokens, total_tokens}``.
            metadata: Additional metadata.
            level: Severity level.
            status_message: Status message.
        """
        if isinstance(generation, _NoOpGeneration):
            return
        try:
            generation.end(
                output=output,
                usage=usage,
                metadata=metadata,
                level=level,
                status_message=status_message,
            )
        except Exception as exc:
            logger.warning("Langfuse generation end failed: %s", exc)

    # ── Convenience: timed span context manager ───────────────────

    @contextmanager
    def span_context(
        self,
        name: str,
        *,
        trace: Any = None,
        parent: Any = None,
        metadata: dict[str, Any] | None = None,
        input_data: Any = None,
    ) -> Generator[Any, None, None]:
        """Context manager that auto-starts and ends a span.

        Usage::

            with tracer.span_context("my_operation", trace=trace) as span:
                result = do_work()
        """
        span = self.start_span(
            name, trace=trace, parent=parent, metadata=metadata, input_data=input_data
        )
        start_time = time.monotonic()
        try:
            yield span
        except Exception as exc:
            self.end_span(
                span,
                output={"error": str(exc)},
                level="ERROR",
                status_message=str(exc),
                metadata={"duration_ms": int((time.monotonic() - start_time) * 1000)},
            )
            raise
        else:
            self.end_span(
                span,
                metadata={"duration_ms": int((time.monotonic() - start_time) * 1000)},
            )

    # ── Flush ─────────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush any buffered traces to Langfuse."""
        if self.enabled and self._client is not None:
            try:
                self._client.flush()
            except Exception as exc:
                logger.warning("Langfuse flush failed: %s", exc)

    def shutdown(self) -> None:
        """Gracefully shut down the Langfuse client."""
        if self.enabled and self._client is not None:
            try:
                self._client.flush()
                self._client.shutdown()
            except Exception as exc:
                logger.warning("Langfuse shutdown failed: %s", exc)


def traced(name: str | None = None) -> Callable:
    """Decorator for auto-instrumenting async functions with Langfuse spans.

    Requires the function's first positional argument (typically ``self``)
    to have a ``_tracer`` attribute of type ``LangfuseTracer``.

    Args:
        name: Optional span name; defaults to the function's ``__name__``.
    """

    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Try to find a tracer on self.
            tracer: LangfuseTracer | None = None
            if args and hasattr(args[0], "_tracer"):
                tracer = args[0]._tracer

            if tracer is None or not tracer.enabled:
                return await func(*args, **kwargs)

            trace = kwargs.get("_trace") or kwargs.get("trace")
            parent = kwargs.get("_parent_span") or kwargs.get("parent_span")
            span = tracer.start_span(span_name, trace=trace, parent=parent)
            start_time = time.monotonic()
            try:
                result = await func(*args, **kwargs)
                tracer.end_span(
                    span,
                    output=str(result)[:500] if result else None,
                    metadata={
                        "duration_ms": int((time.monotonic() - start_time) * 1000)
                    },
                )
                return result
            except Exception as exc:
                tracer.end_span(
                    span,
                    output={"error": str(exc)},
                    level="ERROR",
                    status_message=str(exc),
                    metadata={
                        "duration_ms": int((time.monotonic() - start_time) * 1000)
                    },
                )
                raise

        return wrapper

    return decorator
