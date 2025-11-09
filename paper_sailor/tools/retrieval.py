from __future__ import annotations

import re
from typing import Dict, List, Sequence

from ..vectorstore import VectorStore
from .embeddings import embed_texts
from ..memory import MemoryManager


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def keyword_retrieve(chunks: Sequence[Dict], question: str, top_n: int = 5) -> List[Dict]:
    """Very simple keyword overlap scorer for MVP without extra deps."""
    q_tok = set(_tokenize(question))
    if not q_tok:
        return []
    scored = []
    for ch in chunks:
        text = ch.get("text", "")
        t_tok = set(_tokenize(text))
        overlap = len(q_tok & t_tok)
        if overlap > 0:
            scored.append((overlap, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ch for _, ch in scored[:top_n]]


def vector_retrieve(session_id: str, question: str, store: VectorStore, top_n: int = 5) -> List[Dict]:
    embeddings = embed_texts([question])
    if not embeddings:
        return []
    result = store.query(session_id, embeddings[0], top_k=top_n)
    return result


def multimodal_retrieve(
    session_id: str,
    question: str,
    store: VectorStore,
    memory_manager: MemoryManager,
    top_n: int = 5,
    content_types: List[str] | None = None,
) -> Dict[str, List[Dict]]:
    """Retrieve relevant text, figures, and tables plus memory context."""
    if content_types is None:
        content_types = ["text", "figure", "table"]
    embeddings = embed_texts([question])
    if not embeddings:
        return {"text_chunks": [], "figures": [], "tables": [], "memory_context": []}
    hits = store.query(session_id, embeddings[0], top_k=max(top_n * 4, top_n))
    # Partition by type
    texts: List[Dict] = []
    figures: List[Dict] = []
    tables: List[Dict] = []
    for h in hits:
        ctype = (h.get("content_type") or "text").lower()
        if ctype == "figure" and "figure" in content_types:
            figures.append(h)
        elif ctype == "table" and "table" in content_types:
            tables.append(h)
        elif "text" in content_types:
            texts.append(h)
        # Stop early if enough collected
        if len(texts) >= top_n and len(figures) >= top_n and len(tables) >= top_n:
            break
    mem = memory_manager.get_relevant_context(session_id, question)
    mem_items = [{"level": "session", "text": mem}] if mem else []
    return {
        "text_chunks": texts[:top_n],
        "figures": figures[:top_n],
        "tables": tables[:top_n],
        "memory_context": mem_items,
    }

