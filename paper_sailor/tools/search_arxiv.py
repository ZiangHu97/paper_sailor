from __future__ import annotations

import datetime as dt
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict, List


ARXIV_API = "http://export.arxiv.org/api/query"


def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()


def search_arxiv(query: str, max_results: int = 20, start: int = 0, sort: str = "submittedDate", order: str = "descending") -> List[Dict]:
    """Search arXiv via Atom API using stdlib only.

    Returns a list of papers: {id, source, title, authors, year, url, pdf_url?, summary}
    """
    q = urllib.parse.urlencode({
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": sort,
        "sortOrder": order,
    })
    url = f"{ARXIV_API}?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "paper-sailor/0.1"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read()

    root = ET.fromstring(data)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out: List[Dict] = []
    for entry in root.findall("a:entry", ns):
        id_text = entry.findtext("a:id", default="", namespaces=ns)
        title = (entry.findtext("a:title", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("a:summary", default="", namespaces=ns) or "").strip()
        published = entry.findtext("a:published", default="", namespaces=ns) or ""
        year = None
        if published:
            try:
                year = dt.datetime.fromisoformat(published.replace("Z", "+00:00")).year
            except Exception:
                year = None
        authors: List[str] = [a.findtext("a:name", default="", namespaces=ns) or "" for a in entry.findall("a:author", ns)]

        url_html = None
        pdf_url = None
        for link in entry.findall("a:link", ns):
            rel = link.attrib.get("rel")
            href = link.attrib.get("href")
            link_type = link.attrib.get("type")
            if rel == "alternate" and href:
                url_html = href
            if link_type == "application/pdf" and href:
                pdf_url = href

        arxiv_id = id_text.split("/abs/")[-1] if "/abs/" in id_text else id_text.rsplit("/", 1)[-1]
        out.append({
            "id": f"arxiv:{arxiv_id}",
            "source": "arxiv",
            "title": re.sub(r"\s+", " ", title),
            "authors": authors,
            "year": year,
            "url": url_html or f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "summary": summary,
        })
    return out

