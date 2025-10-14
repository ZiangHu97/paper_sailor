import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHUNK_DIR = DATA_DIR / "chunks"
NOTES_DIR = DATA_DIR / "notes"
STATE_DIR = DATA_DIR / "sessions"
VECTOR_DB = DATA_DIR / "vectors.sqlite3"


def ensure_dirs() -> None:
    for p in (DATA_DIR, PDF_DIR, CHUNK_DIR, NOTES_DIR, STATE_DIR):
        p.mkdir(parents=True, exist_ok=True)


def jsonl_append(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def jsonl_write(path: Path, objs: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def json_write(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def json_read(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def jsonl_read(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def list_sessions() -> List[str]:
    ensure_dirs()
    return [p.stem for p in NOTES_DIR.glob("*.json")]


def session_path(session_id: str) -> Path:
    return NOTES_DIR / f"{session_id}.json"


def session_state_path(session_id: str) -> Path:
    return STATE_DIR / f"{session_id}.json"


def load_json_default(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return default.copy() if default else {}
    return json_read(path)


def save_session_state(session_id: str, state: Dict[str, Any]) -> None:
    json_write(session_state_path(session_id), state)


def papers_jsonl() -> Path:
    return DATA_DIR / "papers.jsonl"


def pdf_path(paper_id: str) -> Path:
    return PDF_DIR / f"{paper_id}.pdf"


def chunks_path(paper_id: str) -> Path:
    return CHUNK_DIR / f"{paper_id}.jsonl"


def vector_store_path() -> Path:
    ensure_dirs()
    return VECTOR_DB


def write_chunks(paper_id: str, chunks: Iterable[Dict[str, Any]]) -> None:
    jsonl_write(chunks_path(paper_id), chunks)
