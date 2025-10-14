from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


try:  # Optional dependency
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional import
    fitz = None  # type: ignore

try:  # Optional dependency
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextBox, LTTextContainer
except Exception:  # pragma: no cover - optional import
    extract_pages = None  # type: ignore
    LTTextBox = LTTextContainer = None  # type: ignore


CHUNK_CHAR_LIMIT = 1000
CHUNK_CHAR_MIN = 400


@dataclass
class _Paragraph:
    page: int
    text: str


def parse_pdf_text(local_pdf_path: str, *, paper_id: str) -> List[Dict]:
    if not os.path.exists(local_pdf_path):
        return []

    pages = _extract_pages(local_pdf_path)
    if not pages:
        return []

    paragraphs = list(_iter_paragraphs(pages))
    if not paragraphs:
        return []

    chunks: List[Dict] = []
    buffer: List[str] = []
    section: Optional[str] = None
    page_start: Optional[int] = None
    chunk_index = 0

    def flush() -> None:
        nonlocal buffer, section, page_start, chunk_index, current_len
        if not buffer:
            return
        text = " ".join(buffer).strip()
        if not text:
            buffer = []
            return
        chunk_index += 1
        chunk = {
            "id": f"{paper_id}:{chunk_index:04d}",
            "paper_id": paper_id,
            "section": section,
            "page_from": page_start or 0,
            "page_to": current_page,
            "text": text,
        }
        chunks.append(chunk)
        buffer = []
        page_start = None
        current_len = 0

    current_len = 0
    current_page = 0

    for para in paragraphs:
        current_page = para.page
        cleaned = para.text.strip()
        if not cleaned:
            continue
        heading = _maybe_heading(cleaned)
        if heading:
            flush()
            section = heading
            continue

        if page_start is None:
            page_start = para.page

        if current_len + len(cleaned) > CHUNK_CHAR_LIMIT and buffer:
            flush()
            current_len = 0
            page_start = para.page

        buffer.append(cleaned)
        current_len += len(cleaned)

        if current_len >= CHUNK_CHAR_MIN:
            flush()
            current_len = 0
            page_start = None

    flush()
    return chunks


def _extract_pages(path: str) -> List[Tuple[int, str]]:
    if fitz is not None:
        try:
            doc = fitz.open(path)
        except Exception:
            doc = None
        if doc is not None:
            pages = []
            for idx, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                pages.append((idx, text))
            doc.close()
            if pages:
                return pages

    if extract_pages is not None:
        pages = []
        try:
            for idx, layout in enumerate(extract_pages(path), start=1):  # type: ignore[arg-type]
                lines: List[str] = []
                for element in layout:
                    if isinstance(element, (LTTextBox, LTTextContainer)):
                        lines.append(element.get_text())
                pages.append((idx, "".join(lines)))
        except Exception:
            pages = []
        if pages:
            return pages

    return []


def _iter_paragraphs(pages: List[Tuple[int, str]]) -> Iterator[_Paragraph]:
    for page_num, text in pages:
        cleaned = text.replace("\r", "\n")
        blocks = re.split(r"\n\s*\n", cleaned)
        for block in blocks:
            lines = [re.sub(r"^[\-•\*]\s+", "", line).strip() for line in block.splitlines()]
            lines = [line for line in lines if line]
            if not lines:
                continue
            paragraph = " ".join(lines)
            yield _Paragraph(page=page_num, text=re.sub(r"\s+", " ", paragraph))


def _maybe_heading(text: str) -> Optional[str]:
    candidate = text.strip()
    if not candidate:
        return None
    candidate = candidate.replace("\n", " ")
    if len(candidate) < 5 or len(candidate) > 120:
        return None
    if candidate.lower() in {"abstract", "introduction", "conclusion", "related work", "method", "results"}:
        return candidate.title()
    if candidate.endswith(":"):
        return candidate.rstrip(":").strip().title()
    if re.match(r"^\d+(\.\d+)*\s+.+", candidate):
        return candidate
    letters = [ch for ch in candidate if ch.isalpha()]
    if letters and sum(ch.isupper() for ch in letters) / len(letters) > 0.6:
        return candidate.title()
    return None
