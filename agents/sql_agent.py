"""SQL agent — answers quantitative and tabular questions via generated SQL over SQLite/DuckDB.

Converts structured/tabular data extracted from PDFs into dynamically queryable tables.
Generates SQL using Gemini, validates for read-only safety, executes queries, and
employs a Self-RAG loop to correct invalid SQL or empty result sets against schema.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from omnibrain.agents.retrieval_agent import AgentResult
from omnibrain.citations.engine import CitationEngine
from omnibrain.config import Settings
from omnibrain.exceptions import AgentError, SQLExecutionError
from omnibrain.observability.tracer import LangfuseTracer
from omnibrain.storage.database import Database

logger = logging.getLogger(__name__)


class SQLAgent:
    """Generates, validates, self-corrects, and executes SQL queries over extracted tabular data."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        citation_engine: CitationEngine,
        tracer: LangfuseTracer | None = None,
    ) -> None:
        self._settings = settings
        self._database = database
        self._citation_engine = citation_engine
        self._tracer = tracer
        self._openai_client = AsyncOpenAI(api_key=settings.effective_api_key, base_url=settings.openai_base_url)
        self._max_retries = settings.self_rag_max_retries

    async def run(
        self,
        document_id: str,
        question: str,
        *,
        trace: Any = None,
        parent_span: Any = None,
    ) -> AgentResult:
        """Generate and execute SQL to answer the question with a Self-RAG retry loop.

        Args:
            document_id: Document whose tables to query.
            question: The user's question about tabular/numeric data.
            trace: Optional Langfuse trace.
            parent_span: Optional parent Langfuse span.

        Returns:
            ``AgentResult`` with query results, generated SQL, and table citations.

        Raises:
            SQLExecutionError: If valid SQL cannot be produced after max retries.
            AgentError: On other agent failures.
        """
        span = None
        if self._tracer and self._tracer.enabled:
            span = self._tracer.start_span(
                "SQLAgent.run",
                trace=trace,
                parent=parent_span,
                input_data={"document_id": document_id, "question": question},
            )

        try:
            logger.info("SQLAgent running for document %s.", document_id)
            tables = await self._database.get_tables_for_document(document_id)

            if not tables:
                logger.info("No tables found in document %s for SQLAgent.", document_id)
                res = AgentResult(context_text="", citations=[], agent_name="SQL")
                if self._tracer and span:
                    self._tracer.end_span(span, output="No tables available")
                return res

            # Build detailed schema string
            schema_lines: list[str] = []
            for t in tables:
                headers = t.get("headers", [])
                sample_row = t["data"][0] if t.get("data") else []
                schema_lines.append(
                    f"Table: \"{t['table_name']}\" (from Page {t.get('page_number', 1)})\n"
                    f"  Columns: {headers}\n"
                    f"  Sample row: {sample_row}"
                )
            schema_str = "\n\n".join(schema_lines)

            # Self-RAG loop for SQL generation and execution
            sql_query = ""
            results: list[dict[str, Any]] = []
            last_error: str | None = None
            retries = 0

            while retries <= self._max_retries:
                # 1. Generate SQL with Gemini
                sql_query = await self._generate_sql(
                    question=question,
                    schema_str=schema_str,
                    previous_query=sql_query if retries > 0 else None,
                    error_feedback=last_error if retries > 0 else None,
                    trace=trace,
                    parent_span=span,
                )

                # 2. Validate safety
                cleaned_query = sql_query.strip().removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
                if not cleaned_query.upper().lstrip().startswith("SELECT"):
                    last_error = f"Query must start with SELECT. Got: {cleaned_query}"
                    retries += 1
                    logger.warning("SQL validation failed: %s (attempt %d)", last_error, retries)
                    continue

                # 3. Execute SQL against database
                try:
                    results = await self._database.execute_table_query(cleaned_query, document_id=document_id)
                    last_error = None
                    # Success!
                    logger.info(
                        "SQLAgent query succeeded on attempt %d: %d rows returned.",
                        retries + 1,
                        len(results),
                    )
                    sql_query = cleaned_query
                    break
                except SQLExecutionError as exc:
                    last_error = f"SQLite Execution Error: {exc.message}"
                    retries += 1
                    logger.warning(
                        "SQL execution failed on attempt %d: %s. Retrying Self-RAG...",
                        retries,
                        exc.message,
                    )
                except Exception as exc:
                    last_error = f"Unexpected execution error: {exc}"
                    retries += 1
                    logger.warning("SQL execution failed: %s. Retrying...", exc)

            if last_error and not results and retries > self._max_retries:
                raise SQLExecutionError(
                    f"SQLAgent failed after {self._max_retries} Self-RAG retries. Last error: {last_error}",
                    details={"last_query": sql_query, "last_error": last_error},
                )

            # Format results into structured markdown/JSON context
            if results:
                formatted_rows = json.dumps(results[:50], indent=2, default=str)
                context_text = (
                    f"SQL Agent Extracted Structured Data:\n"
                    f"Executed Query: {sql_query}\n\n"
                    f"Result ({len(results)} rows):\n"
                    f"{formatted_rows}"
                )
            else:
                context_text = f"Executed Query: {sql_query}\n\nNo matching rows found in tabular data."

            # Extract table names referenced in the executed SQL
            import re as _re
            referenced_tables = set(_re.findall(r'"(td_[^"]+)"', sql_query))
            # Also match unquoted table names
            referenced_tables |= set(_re.findall(r'\b(td_\w+)\b', sql_query))

            citations = [
                self._citation_engine.build_citation_from_agent(
                    document_id=document_id,
                    page_number=max(t.get("page_number", 1), 1),
                    text_snippet=(
                        f"Table '{t['table_name']}' on Page {t.get('page_number', 1)} "
                        f"(Columns: {', '.join(t.get('headers', []))})"
                    ),
                    source_type="table",
                    relevance_score=0.95,
                    section_title=f"Table (Page {t.get('page_number', 1)})",
                )
                for t in tables
                if t["table_name"] in referenced_tables
            ]

            result = AgentResult(
                context_text=context_text,
                citations=citations,
                agent_name="SQL",
                raw_data={"query": sql_query, "row_count": len(results)},
                retries=retries,
            )

            if self._tracer and span:
                self._tracer.end_span(
                    span,
                    output={
                        "query": sql_query,
                        "row_count": len(results),
                        "retries": retries,
                    },
                )
            return result

        except SQLExecutionError:
            raise
        except Exception as exc:
            logger.error("SQLAgent failed: %s", exc, exc_info=True)
            if self._tracer and span:
                self._tracer.end_span(
                    span, output={"error": str(exc)}, level="ERROR", status_message=str(exc)
                )
            raise AgentError(f"SQL agent failed: {exc}") from exc

    async def _generate_sql(
        self,
        question: str,
        schema_str: str,
        previous_query: str | None = None,
        error_feedback: str | None = None,
        *,
        trace: Any = None,
        parent_span: Any = None,
    ) -> str:
        """Generate or correct SQLite SELECT statement using Gemini."""
        if previous_query and error_feedback:
            prompt = (
                "You are an expert SQL engineer. Your previous SQL query failed.\n\n"
                f"DATABASE SCHEMAS:\n{schema_str}\n\n"
                f"USER QUESTION: {question}\n\n"
                f"PREVIOUS FAILED QUERY:\n{previous_query}\n\n"
                f"ERROR FEEDBACK:\n{error_feedback}\n\n"
                "Fix the query. Rules:\n"
                "- Write a valid SQLite SELECT query to answer the question.\n"
                "- Return ONLY the raw SQL query without markdown fences or commentary.\n"
                "- Double-quote table and column names (e.g. \"Revenue\", \"td_xxx_0\").\n"
                "- Only use SELECT statements (no INSERT/UPDATE/DROP/ALTER)."
            )
        else:
            prompt = (
                "You are an expert SQL engineer. Convert the user's natural-language "
                "question into an accurate SQLite SELECT query over the extracted document tables.\n\n"
                f"DATABASE SCHEMAS:\n{schema_str}\n\n"
                f"USER QUESTION: {question}\n\n"
                "Rules:\n"
                "- Return ONLY the raw SQL query without markdown fences, explanation, or code blocks.\n"
                "- Use ONLY SELECT statements.\n"
                "- Double-quote all table and column names exactly as shown in the schemas.\n"
                "- Do NOT use DDL or DML statements."
            )

        gen = None
        if self._tracer and self._tracer.enabled:
            gen = self._tracer.start_generation(
                "SQLAgent.generate_sql",
                trace=trace,
                parent=parent_span,
                model=self._settings.openai_model,
                input_data={"question": question, "has_feedback": bool(error_feedback)},
            )

        try:
            response = await self._openai_client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.0,
            )
            sql = response.choices[0].message.content or ""
            sql = sql.strip()

            if self._tracer and gen:
                usage = None
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                self._tracer.end_generation(gen, output=sql, usage=usage)

            return sql
        except Exception as exc:
            if self._tracer and gen:
                self._tracer.end_generation(
                    gen, output={"error": str(exc)}, level="ERROR", status_message=str(exc)
                )
            raise
