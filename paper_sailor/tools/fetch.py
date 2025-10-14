from __future__ import annotations

import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Optional

from ..storage import pdf_path


USER_AGENT = "paper-sailor/0.2"


def fetch_html(url: str, timeout: float = 20) -> Optional[str]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "").lower()
            if "html" not in ctype:
                return None
            encoding = resp.headers.get_content_charset() if hasattr(resp.headers, "get_content_charset") else None
            raw = resp.read()
            return raw.decode(encoding or "utf-8", errors="ignore")
    except Exception:
        return None


class _PdfLinkParser(HTMLParser):
    def __init__(self, base: str) -> None:
        super().__init__()
        self.base = base
        self.result: Optional[str] = None

    def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
        if self.result is not None:
            return
        attr = {k.lower(): v for k, v in attrs}
        href = attr.get("href") or attr.get("data-href") or attr.get("data-pdf")
        if tag.lower() in {"meta", "link"}:
            name = attr.get("name", "").lower()
            if name == "citation_pdf_url" and attr.get("content"):
                self.result = urllib.parse.urljoin(self.base, attr["content"])
                return
            if attr.get("type", "").lower() == "application/pdf" and href:
                self.result = urllib.parse.urljoin(self.base, href)
                return

        if tag.lower() != "a":
            return
        type_attr = attr.get("type", "").lower()
        aria = attr.get("aria-label", "").lower()
        title = attr.get("title", "").lower()
        text_hint = "pdf" in aria or "pdf" in title
        if not href:
            return
        if href.lower().endswith(".pdf") or "pdf" in type_attr or text_hint:
            self.result = urllib.parse.urljoin(self.base, href)


def discover_pdf_url(html: str, base_url: str) -> Optional[str]:
    meta_match = re.search(r'<meta[^>]+name=["\']citation_pdf_url["\'][^>]+content=["\']([^"\']+)["\']', html, flags=re.I)
    if meta_match:
        return urllib.parse.urljoin(base_url, meta_match.group(1))

    parser = _PdfLinkParser(base_url)
    try:
        parser.feed(html)
    except Exception:
        parser.result = None
    if parser.result:
        return parser.result

    # Fallback: regex search for .pdf links
    pdf_matches = re.findall(r'href=["\']([^"\']+\.pdf)["\']', html, flags=re.I)
    for href in pdf_matches:
        return urllib.parse.urljoin(base_url, href)
    return None


def download_file(url: str, paper_id: str, kind: str = "pdf", max_bytes: int = 100 * 1024 * 1024) -> Optional[str]:
    """Download a file (PDF) to data dir, returns local path or None.

    Uses stdlib urllib only; sets UA and a basic size guard.
    """
    if kind == "pdf":
        out_path = pdf_path(paper_id)
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            ctype = resp.headers.get("Content-Type", "").lower()
            if kind == "pdf" and "pdf" not in ctype and not url.lower().endswith(".pdf"):
                # Not a PDF, skip
                return None
            total = 0
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        return None
                    f.write(chunk)
        return str(out_path)
    except Exception:
        return None
