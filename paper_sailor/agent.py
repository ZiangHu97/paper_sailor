from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List

from .storage import (
    ensure_dirs,
    json_write,
    jsonl_append,
    papers_jsonl,
    pdf_path,
    session_path,
    write_chunks,
)
from .tools import (
    discover_pdf_url,
    download_file,
    fetch_html,
    keyword_retrieve,
    parse_pdf_text,
    search_arxiv,
    vector_retrieve,
)
from .vectorstore import VectorStore


DEFAULT_QUESTIONS = [
    "What are recent trends and core methods?",
    "What are key limitations and open problems?",
    "What datasets and metrics are common?",
]


def _batched(seq: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_session(topic: str, session_id: str, max_papers: int = 10) -> Dict[str, Any]:
    """Search, fetch, chunk, index, and draft structured notes for a session."""
    ensure_dirs()
    vector_store = VectorStore()
    vector_store.delete_session(session_id)

    try:
        papers = search_arxiv(topic, max_results=max_papers)
    except Exception:
        papers = []

    paper_ids: List[str] = []
    for p in papers:
        paper_ids.append(p["id"])
        jsonl_append(papers_jsonl(), p)

    all_chunks: List[Dict[str, Any]] = []
    for paper in papers:
        paper_id = paper["id"]
        pdf_local = None

        pdf_url = paper.get("pdf_url")
        if pdf_url:
            pdf_local = download_file(pdf_url, paper_id=paper_id, kind="pdf")

        if not pdf_local:
            html = fetch_html(paper.get("url") or "") if paper.get("url") else None
            if html:
                alt_pdf = discover_pdf_url(html, paper.get("url") or "")
                if alt_pdf:
                    pdf_local = download_file(alt_pdf, paper_id=paper_id, kind="pdf")

        pdf_path_str = str(pdf_path(paper_id))
        local_chunks: List[Dict[str, Any]] = []
        try:
            local_chunks = parse_pdf_text(pdf_path_str, paper_id=paper_id)
        except Exception:
            local_chunks = []

        if not local_chunks and paper.get("summary"):
            local_chunks = [
                {
                    "id": f"{paper_id}:summary",
                    "paper_id": paper_id,
                    "section": "Summary",
                    "page_from": 0,
                    "page_to": 0,
                    "text": paper["summary"],
                }
            ]

        if local_chunks:
            write_chunks(paper_id, local_chunks)
            for chunk in local_chunks:
                all_chunks.append(chunk)

    # Build vector index from chunks
    embedding_errors: List[str] = []
    from .tools import embed_texts  # Imported lazily to avoid circular deps

    embedding_failed = False

    for batch in _batched(all_chunks, 32):
        if embedding_failed:
            break
        texts = [c.get("text", "") for c in batch]
        try:
            embeddings = embed_texts(texts)
        except Exception as exc:
            embedding_errors.append(str(exc))
            embeddings = []
            embedding_failed = True
        if not embeddings:
            continue
        records = []
        for chunk, emb in zip(batch, embeddings):
            records.append(
                {
                    "chunk_id": chunk["id"],
                    "paper_id": chunk.get("paper_id"),
                    "text": chunk.get("text"),
                    "embedding": emb,
                    "metadata": {
                        "section": chunk.get("section"),
                        "page_from": chunk.get("page_from"),
                        "page_to": chunk.get("page_to"),
                    },
                }
            )
        vector_store.upsert(session_id, records)

    # Draft findings using vector store with keyword fallback
    questions = DEFAULT_QUESTIONS
    findings = []

    for question in questions:
        vector_hits: List[Dict[str, Any]] = []
        try:
            vector_hits = vector_retrieve(session_id, question, vector_store, top_n=3)
        except Exception:
            vector_hits = []

        if vector_hits:
            citations = [
                {
                    "paper_id": hit.get("paper_id"),
                    "chunk_id": hit.get("chunk_id"),
                    "page_from": hit.get("metadata", {}).get("page_from"),
                    "page_to": hit.get("metadata", {}).get("page_to"),
                    "score": hit.get("score"),
                }
                for hit in vector_hits
            ]
        else:
            fallback_chunks = keyword_retrieve(all_chunks, question, top_n=3)
            citations = [
                {
                    "paper_id": ch.get("paper_id"),
                    "chunk_id": ch.get("id"),
                    "page_from": ch.get("page_from"),
                    "page_to": ch.get("page_to"),
                    "score": None,
                }
                for ch in fallback_chunks
            ]

        findings.append(
            {
                "question": question,
                "answer": "Auto-generated skeleton; refine with agent reasoning pipeline.",
                "citations": citations,
            }
        )

    ideas = [
        {
            "title": "Idea 1: Improve retrieval with domain lexicon",
            "motivation": "Summaries indicate terminology drift; a controlled vocabulary can help.",
            "method": "Curate domain terms → expand queries → re-rank.",
            "eval": "Measure coverage and citation accuracy on held-out papers.",
            "risks": "Limited generalization; maintenance cost.",
            "refs": paper_ids[:3],
        },
        {
            "title": "Idea 2: Evidence-linked summarization",
            "motivation": "Ensure every claim links to a page/section.",
            "method": "Chunk retrieval → claim extraction → cite with coordinates.",
            "eval": "Human judge citation correctness; automate overlap metrics.",
            "risks": "PDF parsing noise; OCR issues.",
            "refs": paper_ids[:3],
        },
    ]

    note = {
        "topic": topic,
        "session_id": session_id,
        "created_at": int(time.time()),
        "papers": paper_ids,
        "questions": questions,
        "findings": findings,
        "ideas": ideas,
        "reading_list": [{"paper_id": pid, "reason": "From arXiv search"} for pid in paper_ids[:10]],
    }
    if embedding_errors:
        note["warnings"] = embedding_errors
    json_write(session_path(session_id), note)
    return note
