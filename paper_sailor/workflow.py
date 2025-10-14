from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Tuple

from .planner import Planner
from .storage import (
    ensure_dirs,
    json_write,
    jsonl_append,
    load_json_default,
    papers_jsonl,
    pdf_path,
    save_session_state,
    session_path,
    session_state_path,
    write_chunks,
)
from .tools import (
    download_file,
    embed_texts,
    fetch_html,
    keyword_retrieve,
    parse_pdf_text,
    search_arxiv,
    vector_retrieve,
)
from .vectorstore import VectorStore


DEFAULT_STATE: Dict[str, Any] = {
    "step": 0,
    "tasks": [],
    "queries": [],
    "papers": {},
    "chunks": {},
    "history": [],
    "findings": [],
    "warnings": [],
}


def _format_query(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    if ":" in raw:
        return raw
    terms = [t for t in re.split(r"\s+", raw) if t]
    if not terms:
        return raw
    return " AND ".join(f"all:{term}" for term in terms)


def _summarize_results(results: List[Dict[str, Any]]) -> str:
    lines = []
    for item in results[:5]:
        lines.append(f"- {item.get('id')}: {item.get('title')}")
    remainder = max(0, len(results) - 5)
    if remainder:
        lines.append(f"... and {remainder} more")
    return "\n".join(lines) if lines else "(no hits)"


def _download_and_chunk(paper: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    paper_id = paper["id"]
    pdf_local = None
    warnings: List[str] = []

    pdf_url = paper.get("pdf_url")
    if pdf_url:
        pdf_local = download_file(pdf_url, paper_id=paper_id, kind="pdf")

    if not pdf_local and paper.get("url"):
        html = fetch_html(paper["url"])
        if html:
            from .tools import discover_pdf_url

            alt_pdf = discover_pdf_url(html, paper["url"])
            if alt_pdf:
                pdf_local = download_file(alt_pdf, paper_id=paper_id, kind="pdf")

    pdf_path_str = str(pdf_path(paper_id))
    chunks: List[Dict[str, Any]] = []
    if pdf_local:
        try:
            chunks = parse_pdf_text(pdf_path_str, paper_id=paper_id)
        except Exception as exc:  # pragma: no cover - parser failures
            warnings.append(f"parse_failed:{paper_id}:{exc}")
            chunks = []

    if not chunks:
        summary = (paper.get("summary") or "").strip()
        if summary:
            chunks = [
                {
                    "id": f"{paper_id}:summary",
                    "paper_id": paper_id,
                    "section": "Summary",
                    "page_from": 0,
                    "page_to": 0,
                    "text": summary,
                }
            ]
        else:
            warnings.append(f"no_content:{paper_id}")

    if chunks:
        write_chunks(paper_id, chunks)

    return chunks, warnings


def _index_chunks(session_id: str, chunks: List[Dict[str, Any]], store: VectorStore) -> List[str]:
    texts = [c.get("text", "") for c in chunks]
    if not any(texts):
        return []
    try:
        embeddings = embed_texts(texts)
    except Exception as exc:
        return [f"embedding_failed:{exc}"]
    records = []
    for chunk, emb in zip(chunks, embeddings):
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
    store.upsert(session_id, records)
    return []


def run_planner_session(topic: str, session_id: str, *, max_rounds: int = 6, search_limit: int = 8) -> Dict[str, Any]:
    ensure_dirs()
    state_path = session_state_path(session_id)
    is_new_session = not state_path.exists()
    state = load_json_default(state_path, DEFAULT_STATE)
    state.setdefault("topic", topic)
    state.setdefault("papers", {})
    state.setdefault("queries", [])
    state.setdefault("history", [])
    state.setdefault("findings", [])
    state.setdefault("tasks", [])
    state.setdefault("warnings", [])
    state.setdefault("chunks", {})

    vector_store = VectorStore()
    if is_new_session:
        vector_store.delete_session(session_id)
    planner = Planner(topic)
    observation = "Session started. Awaiting planner direction."

    for _ in range(max_rounds):
        action, raw_payload = planner.next_action(state, observation)
        state["step"] = state.get("step", 0) + 1

        exec_result, warnings = _execute_action(
            action,
            state,
            vector_store,
            search_limit=search_limit,
            session_id=session_id,
        )

        state.setdefault("warnings", []).extend(warnings)
        state.setdefault("history", []).append(
            {
                "step": state["step"],
                "action": action,
                "result": exec_result,
                "planner_payload": raw_payload,
                "timestamp": int(time.time()),
            }
        )
        save_session_state(session_id, state)
        observation = exec_result

        if action["action"] == "finish":
            break

    note = _build_note(state, topic, session_id)
    json_write(session_path(session_id), note)
    return note


def _execute_action(
    action: Dict[str, Any],
    state: Dict[str, Any],
    store: VectorStore,
    *,
    search_limit: int,
    session_id: str,
) -> Tuple[str, List[str]]:
    kind = action["action"]
    if kind == "search":
        return _do_search(action, state, search_limit), []
    if kind == "read":
        return _do_read(action, state, store, session_id)
    if kind == "summarize":
        return _do_summarize(action, state, store, session_id)
    if kind == "finish":
        return action.get("notes", "Planner decided to finish."), []
    return f"Unsupported action {kind}", [f"unsupported_action:{kind}"]


def _do_search(action: Dict[str, Any], state: Dict[str, Any], search_limit: int) -> str:
    queries = action.get("queries", [])
    if not isinstance(queries, list) or not queries:
        return "Planner requested search but provided no queries."

    all_results: List[Dict[str, Any]] = []
    for q in queries:
        if isinstance(q, dict):
            query_text = q.get("q") or q.get("query") or ""
        else:
            query_text = str(q)
        formatted = _format_query(query_text)
        if not formatted:
            continue
        state.setdefault("queries", []).append({"raw": query_text, "formatted": formatted})
        try:
            results = search_arxiv(formatted, max_results=search_limit)
        except Exception as exc:
            all_results.append({"id": "error", "title": f"search failed: {exc}"})
            continue
        for paper in results:
            jsonl_append(papers_jsonl(), paper)
            state.setdefault("papers", {})[paper["id"]] = {
                **paper,
                "status": state["papers"].get(paper["id"], {}).get("status", "discovered"),
            }
            all_results.append(paper)

    summary = _summarize_results(all_results)
    return f"Search completed. Notes: {action.get('notes', '')}\n{summary}"


def _do_read(
    action: Dict[str, Any],
    state: Dict[str, Any],
    store: VectorStore,
    session_id: str,
) -> Tuple[str, List[str]]:
    paper_ids = action.get("papers", [])
    if not isinstance(paper_ids, list) or not paper_ids:
        return "Planner requested read but provided no paper ids.", []

    warnings: List[str] = []
    lines: List[str] = []

    for pid in paper_ids:
        pid = str(pid)
        paper = state.get("papers", {}).get(pid)
        if not paper:
            warnings.append(f"unknown_paper:{pid}")
            continue
        chunks, chunk_warnings = _download_and_chunk(paper)
        warnings.extend(chunk_warnings)
        if not chunks:
            lines.append(f"{pid}: no chunks available")
            continue
        chunk_warn = _index_chunks(session_id, chunks, store)
        warnings.extend(chunk_warn)
        chunk_store = state.setdefault("chunks", {})
        for chunk in chunks:
            chunk_store[chunk["id"]] = chunk
        paper_meta = state["papers"][pid]
        paper_meta["status"] = "read"
        paper_meta["notes"] = action.get("notes", "")
        lines.append(f"{pid}: processed {len(chunks)} chunks")

    summary = "\n".join(lines) if lines else "No papers processed."
    return summary, warnings


def _do_summarize(
    action: Dict[str, Any],
    state: Dict[str, Any],
    store: VectorStore,
    session_id: str,
) -> Tuple[str, List[str]]:
    focus_items = action.get("focus", [])
    if not isinstance(focus_items, list) or not focus_items:
        return "Planner requested summarize without focus questions.", []

    findings = state.setdefault("findings", [])
    lines: List[str] = []

    for question in focus_items:
        question_text = str(question).strip()
        if not question_text:
            continue
        vector_hits = []
        try:
            vector_hits = vector_retrieve(session_id, question_text, store, top_n=4)
        except Exception:
            vector_hits = []
        if not vector_hits:
            cache = list(state.get("chunks", {}).values())
            fallback_hits = keyword_retrieve(cache, question_text, top_n=4)
            vector_hits = [
                {
                    "chunk_id": ch.get("id"),
                    "paper_id": ch.get("paper_id"),
                    "text": ch.get("text"),
                    "score": 0.0,
                    "metadata": {
                        "section": ch.get("section"),
                        "page_from": ch.get("page_from"),
                        "page_to": ch.get("page_to"),
                    },
                }
                for ch in fallback_hits
            ]
        answer = _llm_answer(question_text, vector_hits)
        citations_clean = [
            {
                "paper_id": hit.get("paper_id"),
                "chunk_id": hit.get("chunk_id"),
                "section": (hit.get("metadata") or {}).get("section"),
                "score": hit.get("score"),
            }
            for hit in vector_hits
        ]
        findings.append(
            {
                "question": question_text,
                "answer": answer,
                "citations": citations_clean,
                "step": state.get("step"),
            }
        )
        lines.append(f"Summarized '{question_text}'")

    summary = "\n".join(lines) if lines else "No summaries produced."
    return summary, []


def _llm_answer(question: str, hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "Insufficient evidence collected yet."
    context = []
    for hit in hits:
        score = hit.get("score")
        label = ""
        try:
            if score is not None:
                label = f"score={float(score):.3f}"
        except (ValueError, TypeError):
            label = ""
        prefix = hit.get("paper_id") or "unknown"
        if label:
            prefix = f"{prefix} {label}"
        context.append(f"[{prefix}] {hit.get('text', '')[:1200]}")
    from .llm import call_llm

    messages = [
        {
            "role": "system",
            "content": "You are a research assistant. Write concise answers grounded in the provided excerpts. Cite using paper ids.",
        },
        {
            "role": "user",
            "content": f"Question: {question}\nExcerpts:\n" + "\n\n".join(context),
        },
    ]
    response = call_llm(messages, temperature=0.3, max_output_tokens=600)
    text = ""
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text += content.get("text", "")
    return text.strip() or "No answer returned."


def _build_note(state: Dict[str, Any], topic: str, session_id: str) -> Dict[str, Any]:
    papers = state.get("papers", {})
    reading_list = [
        {"paper_id": pid, "reason": meta.get("status", "discovered")}
        for pid, meta in papers.items()
    ]
    note = {
        "topic": topic,
        "session_id": session_id,
        "created_at": int(time.time()),
        "tasks": state.get("tasks", []),
        "queries": state.get("queries", []),
        "papers": list(papers.keys()),
        "history_steps": state.get("history", []),
        "findings": state.get("findings", []),
        "reading_list": reading_list,
        "warnings": state.get("warnings", []),
    }
    return note
