"""Async SQLite database layer for document metadata, chunks, tables, and images.

All writes use explicit transactions to ensure atomicity.  The schema is
created on first connection via ``initialize()``.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

from omnibrain.config import Settings
from omnibrain.exceptions import SQLExecutionError

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'PENDING',
    error_message TEXT,
    page_count    INTEGER,
    file_hash     TEXT    UNIQUE,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text          TEXT    NOT NULL,
    page_number   INTEGER NOT NULL,
    section_title TEXT,
    chunk_index   INTEGER NOT NULL,
    metadata      TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);

CREATE TABLE IF NOT EXISTS extracted_tables (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number   INTEGER NOT NULL,
    table_index   INTEGER NOT NULL,
    table_name    TEXT    NOT NULL,
    headers       TEXT    NOT NULL,
    data          TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tables_doc ON extracted_tables(document_id);

CREATE TABLE IF NOT EXISTS images (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id   TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_number   INTEGER NOT NULL,
    image_index   INTEGER NOT NULL,
    file_path     TEXT    NOT NULL,
    description   TEXT
);
CREATE INDEX IF NOT EXISTS idx_images_doc ON images(document_id);
"""

# Only allow read-only SELECT for the SQL agent.
_UNSAFE_SQL_PATTERN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


class Database:
    """Async wrapper around an SQLite database using ``aiosqlite``."""

    def __init__(self, settings: Settings) -> None:
        self._db_path: Path = settings.database_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Open the connection and create tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(_SCHEMA_SQL)
        await self._conn.commit()
        logger.info("Database initialized at %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Database connection closed.")

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self._conn

    # ── Document CRUD ──────────────────────────────────────────────

    async def create_document(
        self,
        doc_id: str,
        filename: str,
        file_hash: str,
    ) -> None:
        """Insert a new document record."""
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "INSERT INTO documents (id, filename, file_hash, status, created_at) VALUES (?, ?, ?, 'PENDING', ?)",
            (doc_id, filename, file_hash, now),
        )
        await self._db.commit()
        logger.info("Document created: %s (%s)", doc_id, filename)

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch a single document by ID, or ``None`` if missing."""
        cursor = await self._db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_document_by_hash(self, file_hash: str) -> dict[str, Any] | None:
        """Look up a document by its file hash (for dedup)."""
        cursor = await self._db.execute(
            "SELECT * FROM documents WHERE file_hash = ?", (file_hash,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_documents(self) -> list[dict[str, Any]]:
        """Return all documents ordered by creation time (newest first)."""
        cursor = await self._db.execute(
            "SELECT * FROM documents ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def update_document_status(
        self,
        doc_id: str,
        status: str,
        *,
        error_message: str | None = None,
        page_count: int | None = None,
    ) -> None:
        """Atomically update a document's processing status."""
        parts: list[str] = ["status = ?"]
        params: list[Any] = [status]
        if error_message is not None:
            parts.append("error_message = ?")
            params.append(error_message)
        if page_count is not None:
            parts.append("page_count = ?")
            params.append(page_count)
        params.append(doc_id)
        await self._db.execute(
            f"UPDATE documents SET {', '.join(parts)} WHERE id = ?",  # noqa: S608
            params,
        )
        await self._db.commit()
        logger.info("Document %s status → %s", doc_id, status)

    async def delete_document(self, doc_id: str) -> None:
        """Delete a document and all cascading child records."""
        # Also drop any dynamic table_data tables.
        cursor = await self._db.execute(
            "SELECT table_name FROM extracted_tables WHERE document_id = ?", (doc_id,)
        )
        table_rows = await cursor.fetchall()
        for row in table_rows:
            safe_name = row["table_name"]
            await self._db.execute(f'DROP TABLE IF EXISTS "{safe_name}"')  # noqa: S608
        await self._db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await self._db.commit()
        logger.info("Document %s deleted from database.", doc_id)

    async def reset_stuck_documents(self) -> int:
        """On startup, reset any document in a non-terminal state to PENDING.

        Returns the number of documents reset.
        """
        cursor = await self._db.execute(
            "UPDATE documents SET status = 'PENDING', error_message = NULL "
            "WHERE status NOT IN ('READY', 'FAILED', 'PENDING')"
        )
        await self._db.commit()
        count = cursor.rowcount
        if count:
            logger.warning("Reset %d stuck documents to PENDING.", count)
        return count

    # ── Chunks ─────────────────────────────────────────────────────

    async def insert_chunks(
        self, document_id: str, chunks: list[dict[str, Any]]
    ) -> None:
        """Bulk-insert text chunks for a document."""
        await self._db.executemany(
            "INSERT INTO chunks (document_id, text, page_number, section_title, chunk_index, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    document_id,
                    c["text"],
                    c["page_number"],
                    c.get("section_title"),
                    c["chunk_index"],
                    json.dumps(c.get("metadata", {})),
                )
                for c in chunks
            ],
        )
        await self._db.commit()
        logger.info("Inserted %d chunks for document %s.", len(chunks), document_id)

    async def get_chunks_for_document(self, document_id: str) -> list[dict[str, Any]]:
        """Return all chunks for a document, ordered by chunk_index."""
        cursor = await self._db.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_chunk_count(self, document_id: str) -> int:
        """Count chunks for a document."""
        cursor = await self._db.execute(
            "SELECT COUNT(*) as cnt FROM chunks WHERE document_id = ?", (document_id,)
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    async def get_chunk_by_id(self, chunk_id: int) -> dict[str, Any] | None:
        """Fetch a single chunk by its primary key."""
        cursor = await self._db.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    # ── Extracted tables ───────────────────────────────────────────

    async def insert_table(
        self,
        document_id: str,
        page_number: int,
        table_index: int,
        headers: list[str],
        data: list[list[str]],
    ) -> str:
        """Insert an extracted table and create a queryable dynamic table.

        Returns the generated table_name used for SQL queries.
        """
        table_name = f"td_{document_id.replace('-', '_')}_{table_index}"
        # Store metadata
        await self._db.execute(
            "INSERT INTO extracted_tables (document_id, page_number, table_index, table_name, headers, data) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                document_id,
                page_number,
                table_index,
                table_name,
                json.dumps(headers),
                json.dumps(data),
            ),
        )
        # Create the dynamic table for SQL agent queries.
        safe_cols = ", ".join(f'"{h}" TEXT' for h in headers)
        await self._db.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" ({safe_cols})')  # noqa: S608
        # Insert rows
        placeholders = ", ".join("?" for _ in headers)
        await self._db.executemany(
            f'INSERT INTO "{table_name}" VALUES ({placeholders})',  # noqa: S608
            data,
        )
        await self._db.commit()
        logger.info(
            "Created table %s (%d rows) for document %s page %d.",
            table_name,
            len(data),
            document_id,
            page_number,
        )
        return table_name

    async def get_tables_for_document(self, document_id: str) -> list[dict[str, Any]]:
        """Return metadata for all tables extracted from a document."""
        cursor = await self._db.execute(
            "SELECT * FROM extracted_tables WHERE document_id = ? ORDER BY page_number, table_index",
            (document_id,),
        )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["headers"] = json.loads(d["headers"])
            d["data"] = json.loads(d["data"])
            result.append(d)
        return result

    async def get_table_count(self, document_id: str) -> int:
        """Count tables for a document."""
        cursor = await self._db.execute(
            "SELECT COUNT(*) as cnt FROM extracted_tables WHERE document_id = ?",
            (document_id,),
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0

    async def execute_table_query(self, query: str, *, document_id: str | None = None) -> list[dict[str, Any]]:
        """Execute a read-only SQL query generated by the SQL agent.

        Args:
            query: The SQL SELECT query to execute.
            document_id: If provided, validates all referenced table names belong to this document.

        Raises:
            SQLExecutionError: If the query is not a SELECT, references other documents' tables, or execution fails.
        """
        if _UNSAFE_SQL_PATTERN.search(query):
            raise SQLExecutionError(
                "Only SELECT queries are allowed.",
                details={"query": query},
            )

        # Per-document table isolation: validate referenced table names
        if document_id:
            # Extract all table names from the query (quoted and unquoted td_ patterns)
            referenced_tables = set(re.findall(r'"(td_[^"]+)"', query))
            referenced_tables |= set(re.findall(r'\b(td_\w+)\b', query))

            if referenced_tables:
                # Get this document's own table names
                cursor = await self._db.execute(
                    "SELECT table_name FROM extracted_tables WHERE document_id = ?",
                    (document_id,),
                )
                rows = await cursor.fetchall()
                allowed_tables = {row["table_name"] for row in rows}

                unauthorized = referenced_tables - allowed_tables
                if unauthorized:
                    raise SQLExecutionError(
                        f"Query references tables not belonging to document '{document_id}': {unauthorized}",
                        details={"query": query, "unauthorized_tables": list(unauthorized)},
                    )

        try:
            cursor = await self._db.execute(query)
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            raise SQLExecutionError(
                f"Query execution failed: {exc}",
                details={"query": query},
            ) from exc

    # ── Images ─────────────────────────────────────────────────────

    async def insert_image(
        self,
        document_id: str,
        page_number: int,
        image_index: int,
        file_path: str,
        description: str | None = None,
    ) -> None:
        """Record an extracted image."""
        await self._db.execute(
            "INSERT INTO images (document_id, page_number, image_index, file_path, description) "
            "VALUES (?, ?, ?, ?, ?)",
            (document_id, page_number, image_index, file_path, description),
        )
        await self._db.commit()

    async def get_images_for_document(self, document_id: str) -> list[dict[str, Any]]:
        """Return all images for a document."""
        cursor = await self._db.execute(
            "SELECT * FROM images WHERE document_id = ? ORDER BY page_number, image_index",
            (document_id,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_image_count(self, document_id: str) -> int:
        """Count images for a document."""
        cursor = await self._db.execute(
            "SELECT COUNT(*) as cnt FROM images WHERE document_id = ?", (document_id,)
        )
        row = await cursor.fetchone()
        return row["cnt"] if row else 0
