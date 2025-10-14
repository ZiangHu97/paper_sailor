from __future__ import annotations

import http.server
import json
import os
import posixpath
import re
import socketserver
from pathlib import Path
from typing import Tuple

from .storage import DATA_DIR, NOTES_DIR, list_sessions, session_path, json_read, papers_jsonl, jsonl_read


ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        # Serve UI static files by default
        path = posixpath.normpath(path)
        words = filter(None, path.split("/"))
        p = UI_DIR
        for w in words:
            if os.path.dirname(w) or w in ("..", "."):
                continue
            p = p / w
        return str(p)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        # API routes
        if self.path.startswith("/api/"):
            self._handle_api()
            return
        # Static UI
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        return super().do_GET()

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

    def _json(self, obj, code: int = 200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle_api(self):
        if self.path == "/api/sessions":
            return self._json({"sessions": list_sessions()})

        m = re.match(r"^/api/sessions/([\w\-\.]+)$", self.path)
        if m:
            sid = m.group(1)
            path = session_path(sid)
            if not path.exists():
                return self._json({"error": "not found"}, 404)
            return self._json(json_read(path))

        if self.path == "/api/papers":
            papers = list(jsonl_read(papers_jsonl())) if papers_jsonl().exists() else []
            return self._json({"papers": papers})

        return self._json({"error": "unknown endpoint"}, 404)


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    handler = Handler
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"Paper Sailor UI at http://{args.host}:{args.port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

