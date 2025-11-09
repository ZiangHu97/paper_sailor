from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .storage import ensure_dirs, vector_store_path


class VectorStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        ensure_dirs()
        self.path = Path(path or vector_store_path())
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    session_id TEXT NOT NULL,
                    chunk_id TEXT PRIMARY KEY,
                    paper_id TEXT,
                    text TEXT,
                    embedding TEXT,
                    metadata TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_session ON embeddings(session_id)"
            )
            # Best-effort schema extension for multimodal support
            try:
                conn.execute("ALTER TABLE embeddings ADD COLUMN content_type TEXT DEFAULT 'text'")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE embeddings ADD COLUMN visual_description TEXT")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE embeddings ADD COLUMN image_path TEXT")
            except Exception:
                pass

    def upsert(self, session_id: str, records: Iterable[Dict]) -> None:
        rows = []
        for rec in records:
            emb = rec.get("embedding")
            chunk_id = rec.get("chunk_id")
            if not emb or not chunk_id:
                continue
            rows.append(
                (
                    session_id,
                    chunk_id,
                    rec.get("paper_id"),
                    rec.get("text"),
                    json.dumps(list(map(float, emb))),
                    json.dumps(rec.get("metadata", {})),
                )
            )
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                "REPLACE INTO embeddings (session_id, chunk_id, paper_id, text, embedding, metadata) VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )

    def upsert_multimodal(self, session_id: str, records: Iterable[Dict]) -> None:
        """Insert/update records with optional multimodal fields."""
        rows = []
        for rec in records:
            emb = rec.get("embedding")
            chunk_id = rec.get("chunk_id")
            if not emb or not chunk_id:
                continue
            rows.append(
                (
                    session_id,
                    chunk_id,
                    rec.get("paper_id"),
                    rec.get("text"),
                    json.dumps(list(map(float, emb))),
                    json.dumps(rec.get("metadata", {})),
                    rec.get("content_type") or "text",
                    rec.get("visual_description"),
                    rec.get("image_path"),
                )
            )
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                "REPLACE INTO embeddings (session_id, chunk_id, paper_id, text, embedding, metadata, content_type, visual_description, image_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def delete_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM embeddings WHERE session_id = ?", (session_id,))

    def query(self, session_id: str, embedding: List[float], top_k: int = 5) -> List[Dict]:
        if not embedding:
            return []
        with self._connect() as conn:
            try:
                rows = conn.execute(
                    "SELECT chunk_id, paper_id, text, embedding, metadata, content_type, visual_description, image_path FROM embeddings WHERE session_id = ?",
                    (session_id,),
                ).fetchall()
                has_extra = True
            except Exception:
                rows = conn.execute(
                    "SELECT chunk_id, paper_id, text, embedding, metadata FROM embeddings WHERE session_id = ?",
                    (session_id,),
                ).fetchall()
                has_extra = False

        query_norm = _vector_norm(embedding)
        if query_norm == 0:
            return []

        scored: List[Dict] = []
        for row in rows:
            if len(row) >= 8:
                chunk_id, paper_id, text, emb_json, meta_json, content_type, visual_desc, image_path = row[:8]
            else:
                chunk_id, paper_id, text, emb_json, meta_json = row[:5]
                content_type, visual_desc, image_path = "text", None, None
            try:
                emb = json.loads(emb_json)
            except Exception:
                continue
            score = _cosine_similarity(embedding, emb, query_norm)
            metadata = {}
            if meta_json:
                try:
                    metadata = json.loads(meta_json)
                except Exception:
                    metadata = {}
            scored.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "text": text,
                    "score": score,
                    "metadata": metadata,
                    "content_type": content_type,
                    "visual_description": visual_desc,
                    "image_path": image_path,
                }
            )

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


def _cosine_similarity(query: List[float], item: List[float], query_norm: Optional[float] = None) -> float:
    if not item or len(query) != len(item):
        return -1.0
    if query_norm is None:
        query_norm = _vector_norm(query)
    item_norm = _vector_norm(item)
    if query_norm == 0 or item_norm == 0:
        return -1.0
    dot = sum(q * v for q, v in zip(query, item))
    return dot / (query_norm * item_norm)


def _vector_norm(vec: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))
