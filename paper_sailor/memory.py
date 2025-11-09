from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .storage import DATA_DIR, ensure_dirs
from .config import get_mem0_settings


MEMORY_DIR = DATA_DIR / "memory"
USER_AGENT = "paper-sailor/0.3"


def _ensure_memory_dir() -> None:
    ensure_dirs()
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class MemoryEndpoints:
    create: str
    search: str


class MemoryManager:
    """Multi-level memory manager with MEM0 integration.

    Uses MEM0 Python SDK if available and configured, otherwise falls back
    to local JSON storage.
    """

    def __init__(self) -> None:
        self.settings = get_mem0_settings()
        _ensure_memory_dir()
        
        # Try to use MEM0 SDK
        self.use_mem0 = False
        self.mem0_client = None
        
        if self.settings.api_key:
            try:
                # Try MemoryClient first (for MEM0 Cloud)
                try:
                    from mem0 import MemoryClient
                    self.mem0_client = MemoryClient(api_key=self.settings.api_key)
                    self.use_mem0 = True
                    print(f"✅ MEM0 Cloud SDK initialized")
                except ImportError:
                    # Fallback to Memory class (for MEM0 OSS)
                    from mem0 import Memory
                    from .config import get_openai_settings
                    openai_settings = get_openai_settings()
                    
                    config = {
                        "version": "v1.1",
                        "vector_store": {
                            "provider": "qdrant",
                            "config": {
                                "collection_name": "paper_sailor_memory",
                                "host": "localhost",
                                "port": 6333,
                            }
                        },
                        "llm": {
                            "provider": "openai",
                            "config": {
                                "model": "gpt-4o-mini",
                                "temperature": 0.0,
                                "api_key": openai_settings.api_key,
                                "base_url": openai_settings.base_url
                            }
                        },
                        "embedder": {
                            "provider": "openai",
                            "config": {
                                "model": "text-embedding-3-small",
                                "api_key": openai_settings.api_key,
                                "base_url": openai_settings.base_url
                            }
                        }
                    }
                    self.mem0_client = Memory.from_config(config)
                    self.use_mem0 = True
                    print(f"✅ MEM0 OSS SDK initialized")
            except ImportError:
                print("⚠️  mem0ai not installed, using local storage")
            except Exception as exc:
                print(f"⚠️  MEM0 initialization failed: {exc}, using local storage")

    # ---------- Local fallback helpers ----------
    def _user_path(self, user_id: str) -> Path:
        return MEMORY_DIR / f"user_{user_id}.json"

    def _session_path(self, session_id: str) -> Path:
        return MEMORY_DIR / f"session_{session_id}.json"

    def _agent_path(self) -> Path:
        return MEMORY_DIR / "agent.json"

    # ---------- Public API ----------
    def add_user_preference(self, user_id: str, preference: str) -> None:
        """Store user-level memory (preferences & interests)."""
        if not user_id or not preference:
            return
        
        # Try MEM0 first
        if self.use_mem0 and self.mem0_client:
            try:
                self.mem0_client.add(
                    messages=[{"role": "user", "content": preference}],
                    user_id=user_id
                )
                print(f"✅ MEM0: Added user preference for {user_id}")
                return
            except Exception as exc:
                print(f"⚠️  MEM0 add failed: {exc}, using local fallback")
        
        # Local fallback
        data = _read_json(self._user_path(user_id))
        prefs: List[str] = list(map(str, data.get("preferences", [])))
        if preference and preference not in prefs:
            prefs.append(preference)
        data["preferences"] = prefs
        _write_json(self._user_path(user_id), data)

    def add_session_context(self, session_id: str, context: Dict[str, Any]) -> None:
        """Store session-level memory (topic, selected papers, notes)."""
        if not session_id or not context:
            return
        
        # Try MEM0 first
        if self.use_mem0 and self.mem0_client:
            try:
                # Convert context dict to string for MEM0
                context_str = json.dumps(context, ensure_ascii=False)
                self.mem0_client.add(
                    messages=[{"role": "assistant", "content": f"Session context: {context_str}"}],
                    user_id=f"session_{session_id}"
                )
                print(f"✅ MEM0: Added session context for {session_id}")
                return
            except Exception as exc:
                print(f"⚠️  MEM0 add failed: {exc}, using local fallback")
        
        # Local fallback
        data = _read_json(self._session_path(session_id))
        ctx = data.get("context", {})
        if not isinstance(ctx, dict):
            ctx = {}
        ctx.update({k: v for k, v in (context or {}).items()})
        data["context"] = ctx
        _write_json(self._session_path(session_id), data)

    def add_agent_knowledge(self, knowledge: str) -> None:
        """Store agent-level knowledge (methods, heuristics)."""
        if not knowledge:
            return
        
        # Try MEM0 first
        if self.use_mem0 and self.mem0_client:
            try:
                self.mem0_client.add(
                    messages=[{"role": "system", "content": knowledge}],
                    user_id="agent_global"
                )
                print(f"✅ MEM0: Added agent knowledge")
                return
            except Exception as exc:
                print(f"⚠️  MEM0 add failed: {exc}, using local fallback")
        
        # Local fallback
        data = _read_json(self._agent_path())
        items: List[str] = list(map(str, data.get("knowledge", [])))
        items.append(knowledge)
        data["knowledge"] = items[-200:]  # cap growth
        _write_json(self._agent_path(), data)

    def search_memory(self, query: str, level: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memory across one level."""
        q = (query or "").strip()
        if not q:
            return []
        
        # Try MEM0 first
        if self.use_mem0 and self.mem0_client:
            try:
                # MEM0 v2 API requires filters
                filters = {}
                if level == "user":
                    filters["user_id"] = "test_user"
                elif level == "session":
                    # For session search, skip MEM0 and use local
                    pass
                elif level == "agent":
                    filters["user_id"] = "agent_global"
                
                if filters:
                    results = self.mem0_client.search(
                        query=q,
                        filters=filters,
                        limit=limit
                    )
                    
                    if results and isinstance(results, list):
                        print(f"✅ MEM0: Found {len(results)} memories")
                        return [
                            {"level": level, "text": r.get("memory", ""), "score": r.get("score", 0)}
                            for r in results
                        ]
            except Exception as exc:
                # Expected for some searches, fallback gracefully
                pass
        
        # Local fallback
        results: List[Dict[str, Any]] = []
        q_lower = q.lower()
        
        if level == "user":
            for path in MEMORY_DIR.glob("user_*.json"):
                data = _read_json(path)
                for pref in data.get("preferences", []):
                    text = str(pref)
                    if q_lower in text.lower():
                        results.append({"level": "user", "text": text})
        elif level == "session":
            for path in MEMORY_DIR.glob("session_*.json"):
                data = _read_json(path)
                ctx = data.get("context", {})
                for k, v in ctx.items():
                    text = f"{k}: {v}"
                    if q_lower in text.lower():
                        results.append({"level": "session", "text": text})
        else:  # agent
            data = _read_json(self._agent_path())
            for item in data.get("knowledge", []):
                text = str(item)
                if q_lower in text.lower():
                    results.append({"level": "agent", "text": text})
        
        return results[:limit]

    def get_relevant_context(self, session_id: str, question: str) -> str:
        """Return a short context string from session memory for prompting."""
        # Try MEM0 first
        if self.use_mem0 and self.mem0_client:
            try:
                results = self.mem0_client.search(
                    query=question,
                    filters={"user_id": f"session_{session_id}"},
                    limit=3
                )
                if results and isinstance(results, list):
                    print(f"✅ MEM0: Retrieved context for session {session_id}")
                    return " | ".join([r.get("memory", "") for r in results])
            except Exception:
                # Fallback gracefully
                pass
        
        # Local fallback
        data = _read_json(self._session_path(session_id))
        ctx = data.get("context", {})
        if not isinstance(ctx, dict) or not ctx:
            return ""
        parts = []
        topic = ctx.get("topic")
        if topic:
            parts.append(f"Topic: {topic}")
        papers = ctx.get("papers_read") or ctx.get("papers")
        if papers:
            parts.append(f"Papers: {', '.join(map(str, papers))[:300]}")
        key_findings = ctx.get("key_findings")
        if isinstance(key_findings, list) and key_findings:
            parts.append("Key findings: " + "; ".join(map(str, key_findings))[:400])
        return " | ".join(parts)


