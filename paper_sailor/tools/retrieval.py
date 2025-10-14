from __future__ import annotations

import re
from typing import Dict, List, Sequence

from ..vectorstore import VectorStore
from .embeddings import embed_texts


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

