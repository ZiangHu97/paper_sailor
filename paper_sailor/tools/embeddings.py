from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterable, List

from ..config import get_openai_settings


USER_AGENT = "paper-sailor/0.2"


def embed_texts(texts: Iterable[str], model: str | None = None) -> List[List[float]]:
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []

    settings = get_openai_settings()
    if not settings.api_key:
        raise RuntimeError("OpenAI API key missing; set OPENAI_API_KEY or config.openai.api_key")

    target = settings.base_url.rstrip("/") + "/embeddings"
    payload = json.dumps({
        "model": model or settings.embedding_model,
        "input": texts,
    }).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.api_key}",
        "User-Agent": USER_AGENT,
    }
    headers.update(settings.extra_headers)

    req = urllib.request.Request(target, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=settings.timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Embedding request failed: {exc.code} {exc.reason}: {detail}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - depends on network
        raise RuntimeError(f"Embedding request failed: {exc}") from exc

    data = json.loads(body)
    if not isinstance(data, dict) or "data" not in data:
        raise RuntimeError(f"Unexpected embedding response: {data}")

    embeddings: List[List[float]] = []
    for item in data.get("data", []):
        emb = item.get("embedding")
        if isinstance(emb, list):
            embeddings.append([float(x) for x in emb])
    return embeddings


def embed_multimodal(items: Iterable[dict], model: str | None = "text-embedding-3-large") -> List[List[float]]:
    """Generate embeddings for multimodal items by embedding their textual descriptions.

    Each item can be one of:
      - {'type': 'text', 'content': '...'} or with 'text' key
      - {'type': 'figure'|'table', 'content': '...'} or with 'text'/'visual_description'
    """
    texts: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        text = (
            it.get("content")
            or it.get("text")
            or it.get("visual_description")
            or ""
        )
        text = str(text).strip()
        if text:
            texts.append(text)
    return embed_texts(texts, model=model)

