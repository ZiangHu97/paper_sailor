Paper Sailor

Deep-research agent scaffold with **multimodal understanding** and **multi-level memory**. Searches arXiv, extracts text/figures/tables from PDFs, indexes in a hybrid vector+memory store, and serves results via a web UI.

✨ **New in v0.3**: MEM0 Memory System + Multimodal Support

Status
- ✅ arXiv search, HTML discovery, and PDF downloading
- ✅ PDF parser with PyMuPDF/pdfminer.six support
- ✅ **Multimodal extraction**: figures + tables with Vision API
- ✅ **Parallel processing**: 3-4x faster figure processing
- ✅ **Multi-level memory**: user/session/agent memory with MEM0
- ✅ **Enhanced retrieval**: text + figures + tables + memory context
- ✅ CLI planner-driven workflow + web UI

Directory
- `paper_sailor/` core package (agent, tools, vector store, server)
- `ui/` static front-end bundled with local API
- `data/` cache (`pdfs/`, `chunks/`, `notes/`, `vectors.sqlite3`)

Configuration
- Copy `config.example.toml` to `config.toml` and configure:
  ```toml
  [openai]
  api_key = "sk-..."
  base_url = "https://api.openai.com/v1"
  embedding_model = "text-embedding-3-large"  # Multimodal embeddings
  model = "gpt-4o-mini"
  vision_model = "qwen-vl-max"  # For figure/table description
  max_vision_tokens = 4096

  [mem0]
  api_key = "m0-..."  # Optional: use MEM0 Cloud
  organization_id = "your-org-id"
  # Omit for local JSON storage (default)
  ```
- Env overrides: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_EMBED_MODEL`, `OPENAI_MODEL`, `OPENAI_TIMEOUT`
- Extra headers can be added via `[openai.extra_headers]` in TOML

Quick Start
1. Create and activate a Python 3.10+ environment
2. Install dependencies:
   ```bash
   pip install pymupdf pdfminer.six mem0ai
   ```
3. Configure API access (copy `config.example.toml` to `config.toml`)
4. Run planner workflow with multimodal + memory:
   ```bash
   python -m paper_sailor.cli plan --topic "attention mechanisms" --session my_research --max-rounds 6
   ```
5. (Optional) Test multimodal integration:
   ```bash
   python test_mem0_integration.py  # Verify memory system
   python test_e2e_multimodal.py     # End-to-end with real APIs
   ```
6. Launch web UI: `python -m paper_sailor.server --port 8000`

New Features (v0.3)
- 🖼️ **Multimodal**: Extracts figures & tables from PDFs using Vision API
- ⚡ **Parallel**: 3-4x faster with 6 parallel workers
- 🧠 **Memory**: Multi-level (user/session/agent) with MEM0
- 🔍 **Retrieval**: Unified search across text/figures/tables + memory context
- 📊 **Reports**: See `UPGRADE_SUMMARY.md` for complete documentation

Testing
```bash
python test_mem0_integration.py   # MEM0 integration (5/5 tests)
python test_parallel_vision.py    # Parallel processing
python test_e2e_multimodal.py     # Full pipeline
```

Notes
- Multimodal extraction requires PyMuPDF and Vision API configured
- Memory system works with local JSON (default) or MEM0 Cloud (optional)
- Fallback to keyword retrieval if embeddings fail
