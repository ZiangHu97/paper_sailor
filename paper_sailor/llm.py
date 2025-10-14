from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional

from .config import get_openai_settings


USER_AGENT = "paper-sailor/0.3"


def _normalize_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
    norm: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role") or msg.get("type") or "user"
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            pieces: List[str] = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    pieces.append(str(part["text"]))
                elif isinstance(part, dict) and "content" in part:
                    pieces.append(str(part["content"]))
                else:
                    pieces.append(str(part))
            text = "\n".join(pieces)
        else:
            text = json.dumps(content, ensure_ascii=False)
        norm.append({"role": role, "content": text})
    return norm


def call_llm(
    messages: Iterable[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    temperature: float = 0.2,
    max_output_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    settings = get_openai_settings()
    if not settings.api_key:
        raise RuntimeError("OpenAI API key missing; set OPENAI_API_KEY or configure config.toml")

    target = settings.base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": settings.chat_model,
        "messages": _normalize_messages(messages),
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
    if response_format:
        payload["response_format"] = response_format
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.api_key}",
        "User-Agent": USER_AGENT,
    }
    headers.update(settings.extra_headers)

    req = urllib.request.Request(target, data=data, headers=headers)
    timeout = max(float(settings.timeout or 0), 60.0)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM call failed: {exc.code} {exc.reason}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM call failed: {exc}") from exc

    result = json.loads(body)
    if not isinstance(result, dict) or "choices" not in result:
        raise RuntimeError(f"Unexpected LLM response: {result}")

    choices = result.get("choices", [])
    text_output = ""
    role = "assistant"
    if choices:
        choice = choices[0]
        message = choice.get("message") or {}
        role = message.get("role", role)
        text_output = message.get("content", "") or ""

    result.setdefault("output", [
        {
            "type": "message",
            "role": role,
            "content": [
                {
                    "type": "output_text",
                    "text": text_output,
                }
            ],
        }
    ])
    return result
