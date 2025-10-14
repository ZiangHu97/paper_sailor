from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Tuple

from .llm import call_llm


SYSTEM_PROMPT = """
You are the Planner for the Paper Sailor research agent. Your job is to guide a
multi-step exploration of scientific papers for the given topic. At each turn
you will see the current memory and a summary of the previous executor result.

Always respond with a single JSON object. The JSON must contain:

{
  "action": string,            # one of: search, read, summarize, finish
  "queries": [ ... ],          # required when action == "search"
  "papers": [ ... ],           # required when action == "read"
  "focus": [ ... ],            # required when action == "summarize"
  "notes": string,             # brief intent rationale
  "todo": [
      {"title": string, "status": "todo" | "doing" | "done"}
  ]
}

Rules:
- Generate 1-3 search queries when searching. Prefer arXiv field syntax
  (e.g., "all:graph AND all:molecules"), otherwise plain keywords.
- When reading, choose from known paper ids.
- When summarizing, list focus questions or themes you want the executor to
  synthesize using available notes/chunks.
- Use the todo list to track medium-term subgoals. Update statuses explicitly.
- Finish only when you believe major questions are answered or the budget is
  exhausted. Provide a short summary in notes when finishing.
"""


def _extract_text(response: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                pieces.append(content.get("text", ""))
    text = "".join(pieces).strip()
    if text:
        return text
    choices = response.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        return (msg.get("content") or "").strip()
    return ""


def _ensure_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    clean = []
    for task in tasks:
        title = str(task.get("title", "")).strip()
        if not title:
            continue
        status = task.get("status", "todo").lower()
        if status not in {"todo", "doing", "done"}:
            status = "todo"
        task_id = task.get("id") or uuid.uuid4().hex[:8]
        clean.append({"id": task_id, "title": title, "status": status})
    return clean


def _merge_tasks(existing: List[Dict[str, Any]], updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_title = {t["title"].lower(): t for t in existing}
    for upd in updates:
        key = upd["title"].lower()
        if key in by_title:
            by_title[key]["status"] = upd["status"]
        else:
            by_title[key] = upd
    return list(by_title.values())


def _render_state(state: Dict[str, Any], observation: str) -> str:
    tasks = state.get("tasks", [])
    queries = state.get("queries", [])
    papers = state.get("papers", {})
    raw_findings = state.get("findings", [])

    recent_history = state.get("history", [])[-2:]

    snapshot = {
        "step": state.get("step", 0),
        "open_tasks": [
            {"title": t.get("title"), "status": t.get("status")}
            for t in tasks[:6]
        ],
        "known_papers": [
            {
                "id": pid,
                "status": meta.get("status", "discovered"),
            }
            for pid, meta in list(papers.items())[:10]
        ],
        "queries_tried": queries[-5:],
        "findings": [
            {
                "question": f.get("question"),
                "answered": bool(f.get("answer")),
                "citations": [c.get("paper_id") for c in f.get("citations", [])],
                "step": f.get("step"),
            }
            for f in raw_findings[-5:]
        ],
        "recent_steps": [
            {
                "step": item.get("step"),
                "action": (item.get("action") or {}).get("action"),
                "notes": (item.get("action") or {}).get("notes"),
                "result": (item.get("result") or "")[:300],
            }
            for item in recent_history
        ],
        "last_observation": observation[:400],
    }
    return json.dumps(snapshot, ensure_ascii=False, indent=2)


class Planner:
    def __init__(self, topic: str) -> None:
        self.topic = topic

    def next_action(self, state: Dict[str, Any], observation: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        user_prompt = f"Topic: {self.topic}\nState:\n{_render_state(state, observation)}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        response = call_llm(
            messages,
            temperature=0.2,
            max_output_tokens=1600,
            response_format={"type": "json_object"},
        )
        text = _extract_text(response)
        if not text:
            raise RuntimeError(f"Planner returned empty response: {json.dumps(response, ensure_ascii=False)[:2000]}")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Planner output is not valid JSON: {text}") from exc

        action = payload.get("action", "").strip().lower()
        if action not in {"search", "read", "summarize", "finish"}:
            raise RuntimeError(f"Planner returned unsupported action: {action}")

        todo_list = _ensure_tasks(payload.get("todo", []))
        tasks = _merge_tasks(state.get("tasks", []), todo_list)
        state["tasks"] = tasks

        # Attach normalized fields used by executor
        result = {
            "action": action,
            "notes": payload.get("notes", ""),
            "queries": payload.get("queries", []) if action == "search" else [],
            "papers": payload.get("papers", []) if action == "read" else [],
            "focus": payload.get("focus", []) if action == "summarize" else [],
        }
        return result, payload
