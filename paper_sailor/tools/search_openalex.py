from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple


OPENALEX_WORKS_API = "https://api.openalex.org/works"
_USER_AGENT = "paper-sailor/0.1"


def _reconstruct_abstract(data: Optional[Dict[str, List[int]]]) -> str:
    if not data or not isinstance(data, dict):
        return ""
    max_index = -1
    positions: List[Tuple[int, str]] = []
    for word, idx_list in data.items():
        if not isinstance(idx_list, list):
            continue
        for idx in idx_list:
            try:
                pos = int(idx)
            except (TypeError, ValueError):
                continue
            max_index = max(max_index, pos)
            positions.append((pos, word))
    if max_index < 0:
        return ""
    slots = [""] * (max_index + 1)
    for pos, word in positions:
        if 0 <= pos < len(slots):
            slots[pos] = word
    words = [w for w in slots if w]
    return " ".join(words)


def _normalize_id(oa_id: str) -> str:
    if not oa_id:
        return ""
    if "/" in oa_id:
        return oa_id.rstrip("/").split("/")[-1]
    return oa_id


def search_openalex(
    query: str,
    *,
    max_results: int = 20,
    page: int = 1,
    mailto: Optional[str] = None,
) -> List[Dict]:
    params = {
        "search": query,
        "per-page": max_results,
        "page": page,
    }
    if mailto:
        params["mailto"] = mailto
    url = f"{OPENALEX_WORKS_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = resp.read()
    except Exception:
        return []
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception:
        return []
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list):
        return []

    out: List[Dict] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        oa_id_raw = item.get("id", "")
        oa_id = _normalize_id(str(oa_id_raw))
        authorships = item.get("authorships") or []
        authors = []
        for auth in authorships:
            if not isinstance(auth, dict):
                continue
            author = auth.get("author") or {}
            name = author.get("display_name")
            if name:
                authors.append(name)
        best_oa = item.get("best_oa_location") or {}
        primary_location = item.get("primary_location") or {}
        pdf_url = best_oa.get("url_for_pdf") or primary_location.get("pdf_url")
        landing_url = (
            best_oa.get("url")
            or primary_location.get("landing_page_url")
            or item.get("host_venue", {}).get("url")
            or (f"https://doi.org/{item.get('doi')}" if item.get("doi") else None)
        )
        abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))

        record = {
            "id": f"openalex:{oa_id}" if oa_id else "",
            "source": "openalex",
            "title": item.get("display_name") or "",
            "authors": authors,
            "year": item.get("publication_year"),
            "url": landing_url,
            "pdf_url": pdf_url,
            "summary": abstract,
            "doi": item.get("doi"),
            "openalex": {
                "id": oa_id_raw,
                "display_name": item.get("display_name"),
                "publication_year": item.get("publication_year"),
                "concepts": item.get("concepts"),
                "cited_by_count": item.get("cited_by_count"),
                "referenced_works": item.get("referenced_works"),
                "related_works": item.get("related_works"),
                "best_oa_location": item.get("best_oa_location"),
            },
        }
        if record["id"]:
            out.append(record)
    return out

