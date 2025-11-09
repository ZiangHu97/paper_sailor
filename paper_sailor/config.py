from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional
    tomllib = None  # type: ignore


ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class OpenAISettings:
    api_key: Optional[str]
    base_url: str
    embedding_model: str
    chat_model: str
    timeout: float
    extra_headers: Dict[str, str]


_OPENAI_CACHE: Optional[OpenAISettings] = None
_MEM0_CACHE = None  # type: ignore[assignment]
_VISION_CACHE = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MEM0Settings:
    api_key: Optional[str]
    base_url: str
    organization_id: Optional[str]


@dataclass(frozen=True)
class VisionSettings:
    model: str
    max_tokens: int


def _load_local_config() -> Dict:
    paths = [ROOT / "config.toml", ROOT / "config.json"]
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".toml":
            if not tomllib:
                raise RuntimeError("tomllib not available; install tomli or use config.json")
            with path.open("rb") as f:
                return tomllib.load(f)
        if path.suffix == ".json":
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def get_openai_settings() -> OpenAISettings:
    global _OPENAI_CACHE
    if _OPENAI_CACHE is not None:
        return _OPENAI_CACHE

    data = _load_local_config()
    section = data.get("openai", {}) if isinstance(data, dict) else {}

    api_key = os.getenv("OPENAI_API_KEY") or section.get("api_key")
    base_url = os.getenv("OPENAI_BASE_URL") or section.get("base_url") or "https://api.openai.com/v1"
    base_url = base_url.rstrip("/")
    embed_model = os.getenv("OPENAI_EMBED_MODEL") or section.get("embedding_model") or "text-embedding-3-small"
    chat_model = os.getenv("OPENAI_MODEL") or section.get("model") or "gpt-4o-mini"
    timeout_raw = os.getenv("OPENAI_TIMEOUT") or section.get("timeout") or 30
    try:
        timeout = float(timeout_raw)
    except (TypeError, ValueError):
        timeout = 30.0

    extra_headers = {}
    if isinstance(section.get("extra_headers"), dict):
        extra_headers.update({str(k): str(v) for k, v in section["extra_headers"].items()})

    org = os.getenv("OPENAI_ORG") or section.get("organization")
    if org:
        extra_headers.setdefault("OpenAI-Organization", str(org))

    _OPENAI_CACHE = OpenAISettings(
        api_key=api_key,
        base_url=base_url,
        embedding_model=embed_model,
        chat_model=chat_model,
        timeout=timeout,
        extra_headers=extra_headers,
    )
    return _OPENAI_CACHE


def get_mem0_settings() -> MEM0Settings:
    global _MEM0_CACHE
    if _MEM0_CACHE is not None:
        return _MEM0_CACHE  # type: ignore[return-value]

    data = _load_local_config()
    section = data.get("mem0", {}) if isinstance(data, dict) else {}

    api_key = os.getenv("MEM0_API_KEY") or section.get("api_key")
    base_url = os.getenv("MEM0_BASE_URL") or section.get("base_url") or "https://api.mem0.ai/v1"
    base_url = str(base_url).rstrip("/")
    org_id = os.getenv("MEM0_ORG_ID") or section.get("organization_id")

    _MEM0_CACHE = MEM0Settings(
        api_key=str(api_key) if api_key else None,
        base_url=base_url,
        organization_id=str(org_id) if org_id else None,
    )
    return _MEM0_CACHE  # type: ignore[return-value]


def get_vision_settings() -> VisionSettings:
    global _VISION_CACHE
    if _VISION_CACHE is not None:
        return _VISION_CACHE  # type: ignore[return-value]

    data = _load_local_config()
    section = data.get("openai", {}) if isinstance(data, dict) else {}

    model = os.getenv("OPENAI_VISION_MODEL") or section.get("vision_model") or "gpt-4o"
    max_tokens_raw = os.getenv("OPENAI_VISION_MAX_TOKENS") or section.get("max_vision_tokens") or 4096
    try:
        max_tokens = int(max_tokens_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        max_tokens = 4096

    _VISION_CACHE = VisionSettings(
        model=str(model),
        max_tokens=max_tokens,
    )
    return _VISION_CACHE  # type: ignore[return-value]
