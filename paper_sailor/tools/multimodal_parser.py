from __future__ import annotations

import base64
from typing import Dict, List, Optional, Tuple
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

try:  # Optional dependency
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional import
    fitz = None  # type: ignore

import json
import urllib.error
import urllib.request

from ..config import get_openai_settings, get_vision_settings

USER_AGENT = "paper-sailor/0.3"


def _b64_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def describe_visual_with_gpt4v(image_bytes: bytes, *, context: str = "") -> str:
    """Call a vision-capable chat model to describe an image (figure/table).

    The prompt is crafted for scientific papers to capture axes, trends, units and key values.
    """
    settings = get_openai_settings()
    vision = get_vision_settings()
    if not settings.api_key:
        raise RuntimeError("OpenAI API key missing; set OPENAI_API_KEY or config.openai.api_key")

    target = settings.base_url.rstrip("/") + "/chat/completions"
    image_url = f"data:image/png;base64,{_b64_bytes(image_bytes)}"
    prompt = (
        "You are describing a figure or table from a scientific paper. "
        "Provide a concise description with: what it shows, axes/units (if any), key values/trends, "
        "and any notable observations. Keep it under 120 words.\n"
    )
    if context:
        prompt += f"Context: {context.strip()}\n"
    payload = {
        "model": vision.model,
        "temperature": 0.0,
        "max_tokens": vision.max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
                ],
            }
        ],
    }
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
    except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Vision describe failed: {exc.code} {exc.reason}: {detail}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover
        raise RuntimeError(f"Vision describe failed: {exc}") from exc

    result = json.loads(body)
    try:
        return (result.get("choices", [{}])[0].get("message", {}) or {}).get("content", "").strip()
    except Exception:
        return ""


def _extract_page_images(doc, page_index: int) -> List[Tuple[int, bytes, str]]:
    """Return list of (page_number, image_bytes, ext)."""
    page = doc[page_index]
    images = []
    for info in page.get_images(full=True):
        xref = info[0]
        try:
            img = doc.extract_image(xref)
            image_bytes = img.get("image", b"")
            ext = (img.get("ext") or "png").lower()
            if image_bytes:
                images.append((page_index + 1, image_bytes, ext))
        except Exception:
            continue
    return images


def _extract_page_tables(page) -> List[Tuple[int, Optional[str]]]:
    """Attempt to extract tables; returns list of (page_number, markdown_or_none)."""
    out: List[Tuple[int, Optional[str]]] = []
    try:
        if hasattr(page, "find_tables"):
            tables = page.find_tables()
        else:
            tables = None
    except Exception:
        tables = None
    if not tables:
        return out
    try:
        for t in getattr(tables, "tables", []):
            md: Optional[str] = None
            # Try common exports
            if hasattr(t, "to_markdown"):
                try:
                    md = t.to_markdown()  # type: ignore[assignment]
                except Exception:
                    md = None
            if md is None and hasattr(t, "extract"):
                try:
                    grid = t.extract()
                    # grid is a 2D list of cell strings
                    lines = [" | ".join(str(c or "").strip() for c in row) for row in grid]
                    md = "\n".join(lines)
                except Exception:
                    md = None
            out.append((page.number + 1, md))
    except Exception:
        return out
    return out


def _describe_image_task(args: Tuple) -> Tuple[int, str, Optional[str]]:
    """Helper function for parallel image description.
    
    Returns: (image_index, context, description_or_none)
    """
    idx, img_bytes, context = args
    try:
        desc = describe_visual_with_gpt4v(img_bytes, context=context)
        return (idx, context, desc)
    except Exception as exc:
        return (idx, context, None)


def extract_figures_and_tables(
    pdf_path: str, 
    paper_id: str, 
    *, 
    verbose: bool = False, 
    max_pages: int = None,
    extract_tables: bool = False,
    max_workers: int = 4
) -> List[Dict]:
    """Extract figures (and optionally tables) with parallel vision API calls.
    
    Args:
        pdf_path: Path to PDF file
        paper_id: Paper identifier
        verbose: Print detailed progress
        max_pages: Maximum number of pages to process (None = all pages)
        extract_tables: Whether to extract tables (default: False, only figures)
        max_workers: Number of parallel workers for vision API calls (default: 4)
    """
    if fitz is None:
        if verbose:
            print("   ⚠️ PyMuPDF (fitz) not available")
        return []
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        if verbose:
            print(f"   ⚠️ Failed to open PDF: {exc}")
        return []

    total_pages = len(doc)
    pages_to_process = min(max_pages, total_pages) if max_pages else total_pages
    
    if verbose:
        print(f"   📖 PDF opened: {total_pages} pages (processing {pages_to_process})")

    # Step 1: Collect all images from all pages
    all_images: List[Tuple[int, int, bytes, str]] = []  # (page_index, page_num, img_bytes, ext)
    
    try:
        for page_index in range(pages_to_process):
            page_images = _extract_page_images(doc, page_index)
            
            for page_num, img_bytes, ext in page_images:
                if len(img_bytes) < 1000:  # Skip very small images
                    if verbose:
                        print(f"   ⏭️  Page {page_index+1}: Skipping small image ({len(img_bytes)} bytes)")
                    continue
                all_images.append((page_index, page_num, img_bytes, ext))
            
            if verbose and page_images:
                filtered_count = sum(1 for _, img, _ in page_images if len(img) >= 1000)
                print(f"   📸 Page {page_index+1}: Found {filtered_count} images (skipped {len(page_images) - filtered_count} small)")
    finally:
        try:
            doc.close()
        except Exception:
            pass
    
    if verbose:
        print(f"\n   📊 Total images to process: {len(all_images)}")
        print(f"   🔄 Using {max_workers} parallel workers for vision API calls...")
    
    # Step 2: Parallel vision API calls
    items: List[Dict] = []
    
    if all_images:
        # Prepare tasks for parallel processing
        tasks = []
        for idx, (page_index, page_num, img_bytes, ext) in enumerate(all_images):
            context = f"Paper {paper_id}, page {page_num}"
            tasks.append((idx, img_bytes, context))
        
        # Execute in parallel
        descriptions = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_describe_image_task, task): task[0] for task in tasks}
            
            completed = 0
            for future in as_completed(futures):
                idx, context, desc = future.result()
                descriptions[idx] = desc
                completed += 1
                
                if verbose:
                    status = "✅" if desc else "⚠️"
                    print(f"   {status} Progress: {completed}/{len(all_images)} - Image {idx+1}: {desc[:60] if desc else 'Failed'}...")
        
        # Step 3: Build result items
        for idx, (page_index, page_num, img_bytes, ext) in enumerate(all_images):
            desc = descriptions.get(idx, "")
            chunk_id = f"{paper_id}:fig:{page_index+1:03d}:{len(items)+1:04d}"
            items.append({
                "id": chunk_id,
                "paper_id": paper_id,
                "section": "Figure",
                "page_from": page_num,
                "page_to": page_num,
                "text": desc or "(figure)",
                "content_type": "figure",
                "visual_description": desc,
            })
    
    # Step 4: Optionally extract tables (sequential, usually fast)
    if extract_tables:
        if verbose:
            print(f"\n   📊 Extracting tables...")
        
        try:
            doc = fitz.open(pdf_path)
            for page_index in range(pages_to_process):
                page = doc[page_index]
                page_tables = _extract_page_tables(page)
                
                if verbose and page_tables:
                    print(f"   📊 Page {page_index+1}: Found {len(page_tables)} tables")
                
                for page_num, md in page_tables:
                    text = md or ""
                    chunk_id = f"{paper_id}:tbl:{page_index+1:03d}:{len(items)+1:04d}"
                    items.append({
                        "id": chunk_id,
                        "paper_id": paper_id,
                        "section": "Table",
                        "page_from": page_num,
                        "page_to": page_num,
                        "text": text or "(table)",
                        "content_type": "table",
                        "visual_description": text,
                    })
        finally:
            try:
                doc.close()
            except Exception:
                pass
    
    if verbose:
        figures = sum(1 for item in items if item.get("content_type") == "figure")
        tables = sum(1 for item in items if item.get("content_type") == "table")
        print(f"\n   ✅ Total extracted: {len(items)} items ({figures} figures, {tables} tables)")
    
    return items


