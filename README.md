Paper Sailor (MVP)

Deep-research agent scaffold that searches arXiv, fetches PDFs/HTML, chunks content, indexes it in a lightweight vector store, and serves results to a simple web UI. Built around OAI SDK-style tooling with pluggable steps.

Status
- arXiv search, HTML discovery, and PDF downloading implemented via standard library.
- PDF parser supports PyMuPDF (`fitz`) or pdfminer.six when available; falls back to structured summaries if parsing fails.
- Chunk embeddings stored in a local SQLite vector store (per session) using OpenAI-compatible embeddings.
- CLI runs either the legacy single-pass pipeline or the new planner-driven loop; web UI (`python -m paper_sailor.server`) lists sessions, findings, ideas, and reading lists.

Directory
- `paper_sailor/` core package (agent, tools, vector store, server)
- `ui/` static front-end bundled with local API
- `data/` cache (`pdfs/`, `chunks/`, `notes/`, `vectors.sqlite3`)

Configuration
- Copy `config.example.toml` to `config.toml` (or set env vars) and provide credentials for your OpenAI-compatible relay:
  ```toml
  [openai]
  api_key = "sk-..."
  base_url = "https://your-relay.example.com/v1"
  embedding_model = "text-embedding-3-small"
  model = "gpt-4o-mini"
  ```
- Env overrides: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_EMBED_MODEL`, `OPENAI_TIMEOUT`, `OPENAI_ORG`.
- Extra headers (e.g., custom auth for proxy) can be added via `[openai.extra_headers]` in the TOML file.

Quick Start
1. Create and activate a Python 3.10+ environment.
2. Install optional parsers/embeddings deps as needed:
   - `pip install pymupdf pdfminer.six`
3. Configure API access (see above).
4. Planner-driven exploration:
   - `python -m paper_sailor.cli plan --topic "your topic" --session sail_session --max-rounds 6`
5. (Legacy) single-pass sweep: `python -m paper_sailor.cli run --topic "your topic" --session baseline`
6. Launch the viewer: `python -m paper_sailor.server --port 8000` and open http://localhost:8000

Notes
- Network access is required for live search, PDF downloads, and embeddings. In offline setups, seed `data/` with cached results.
- If embeddings fail (missing key, network issues), the agent falls back to keyword retrieval and records warnings in the session note.
