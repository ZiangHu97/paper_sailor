from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterable, Optional


OPENALEX_API_BASE = "https://api.openalex.org"
_USER_AGENT = "paper-sailor/0.1"


class OpenAlexError(RuntimeError):
    """Raised when the OpenAlex API returns an unexpected response."""


def _normalize_identifier(identifier: str) -> str:
    """Convert common id forms (arxiv:1234, 1234, OA:W123) to API-ready path."""
    if not identifier:
        return ""
    identifier = identifier.strip()
    if identifier.startswith("https://") or identifier.startswith("http://"):
        return identifier
    if identifier.lower().startswith("arxiv:"):
        return f"arXiv:{identifier.split(':', 1)[1]}"
    if identifier.startswith("W"):
        # Already an OpenAlex work id
        return identifier
    if identifier.startswith("OA:"):
        return identifier.split(":", 1)[1]
    if identifier.startswith("10."):
        return f"https://doi.org/{identifier}"
    # Fallback: assume raw arXiv id
    return f"arXiv:{identifier}"


def _build_url(identifier: str, params: Optional[Dict[str, Any]] = None) -> str:
    path = _normalize_identifier(identifier)
    if not path:
        raise OpenAlexError("Cannot build OpenAlex request without an identifier")
    base = f"{OPENALEX_API_BASE}/works/{path}"
    if not params:
        return base
    query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    return f"{base}?{query}" if query else base


def fetch_work(
    identifier: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
) -> Dict[str, Any]:
    """Fetch a single OpenAlex work object."""
    url = _build_url(identifier, params=params)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise OpenAlexError(f"OpenAlex returned status {resp.status}")
            payload = resp.read()
    except OpenAlexError:
        raise
    except Exception as exc:
        raise OpenAlexError(f"Request to OpenAlex failed: {exc}") from exc
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise OpenAlexError(f"Failed to decode OpenAlex response: {exc}") from exc
    if not isinstance(data, dict):
        raise OpenAlexError("OpenAlex response was not a JSON object")
    return data


def _select_fields(work: Dict[str, Any], concepts_limit: int = 8) -> Dict[str, Any]:
    concept_items = work.get("concepts") or []
    primary_concepts = []
    for item in concept_items[:concepts_limit]:
        if not isinstance(item, dict):
            continue
        primary_concepts.append(
            {
                "id": item.get("id"),
                "display_name": item.get("display_name"),
                "level": item.get("level"),
                "score": item.get("score"),
            }
        )
    return {
        "id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("display_name"),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "cited_by_count": work.get("cited_by_count"),
        "referenced_works": work.get("referenced_works") or [],
        "concepts": primary_concepts,
        "related_works": work.get("related_works") or [],
        "primary_topic": (work.get("primary_topic") or {}).get("display_name"),
    }


def enrich_papers_with_openalex(
    papers: Iterable[Dict[str, Any]],
    *,
    timeout: int = 20,
    concepts_limit: int = 8,
) -> None:
    """Augment paper dicts (mutates) with an 'openalex' metadata field."""
    for paper in papers:
        paper_id = str(paper.get("id") or "")
        identifier = paper_id.split(":", 1)[1] if ":" in paper_id else paper_id
        try:
            work = fetch_work(
                identifier,
                params={"select": "id,display_name,doi,publication_year,publication_date,cited_by_count,referenced_works,related_works,concepts,primary_topic"},
                timeout=timeout,
            )
        except OpenAlexError:
            paper.setdefault("warnings", []).append("openalex_lookup_failed")
            continue
        paper["openalex"] = _select_fields(work, concepts_limit=concepts_limit)
        if work.get("doi") and not paper.get("doi"):
            paper["doi"] = work.get("doi")
