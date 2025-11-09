from __future__ import annotations

import math
import random
from typing import Dict, List

from paper_sailor.vectorstore import VectorStore
from paper_sailor.tools.retrieval import multimodal_retrieve
from paper_sailor.memory import MemoryManager
import paper_sailor.tools.embeddings as embmod
import paper_sailor.tools.retrieval as retmod


def _hash_token(tok: str, dim: int) -> int:
    # Simple, deterministic hash function
    h = 0
    for ch in tok:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h % dim


def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]


def _hash_embedding(text: str, dim: int = 64) -> List[float]:
    vec = [0.0] * dim
    for tok in _tokenize(text):
        idx = _hash_token(tok, dim)
        vec[idx] += 1.0
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _fake_embed_texts(texts: List[str], model: str | None = None) -> List[List[float]]:
    return [_hash_embedding(t) for t in texts]


def _mk_record(chunk_id: str, paper_id: str, text: str, ctype: str = "text") -> Dict:
    return {
        "chunk_id": chunk_id,
        "paper_id": paper_id,
        "text": text,
        "embedding": _hash_embedding(text),
        "metadata": {"section": ctype.title(), "page_from": 1, "page_to": 1},
        "content_type": ctype,
        "visual_description": text if ctype in {"figure", "table"} else None,
        "image_path": None,
    }


def run() -> None:
    # Monkeypatch embeddings to avoid network
    embmod.embed_texts = _fake_embed_texts  # type: ignore[assignment]
    retmod.embed_texts = _fake_embed_texts  # type: ignore[assignment]

    session_id = "smoke_mm"
    store = VectorStore()
    store.delete_session(session_id)

    # Seed data
    records = [
        _mk_record("p1:0001", "p1", "Transformer architecture uses attention and multi-head layers.", "text"),
        _mk_record("p1:0002", "p1", "Common datasets and metrics include CIFAR-10 accuracy and ImageNet top-1.", "text"),
        _mk_record("p1:fig:001:0001", "p1", "Bar chart comparing CIFAR-10 accuracy across models.", "figure"),
        _mk_record("p1:tbl:001:0002", "p1", "Table showing dataset sizes and metrics: CIFAR-10 (10k test).", "table"),
    ]
    store.upsert_multimodal(session_id, records)

    # Memory context
    mm = MemoryManager()
    mm.add_session_context(
        session_id,
        {
            "topic": "vision models",
            "papers_read": ["p1"],
            "key_findings": ["Datasets and metrics: CIFAR-10 accuracy widely cited."],
        },
    )

    # Query
    q = "What datasets and metrics are common?"
    result = multimodal_retrieve(session_id, q, store, mm, top_n=2)
    assert isinstance(result, dict), "Result must be a dict"
    assert result.get("text_chunks"), "Expected text results"
    assert result.get("figures"), "Expected figure results"
    assert result.get("tables"), "Expected table results"
    assert result.get("memory_context"), "Expected non-empty memory context"

    # Basic signal checks
    text_join = " ".join(r.get("text", "") for r in result["text_chunks"])
    assert "CIFAR-10" in text_join or "accuracy" in text_join, "Text results should mention datasets/metrics"

    print("OK: smoke multimodal retrieval passed.")


if __name__ == "__main__":
    run()


