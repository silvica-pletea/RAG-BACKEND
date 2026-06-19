# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Lightweight RAG backend: FastAPI + LangChain orchestration, ChromaDB for vector
storage, VoyageAI for embeddings, and the Anthropic API for generation. Local
development / testing only — not production-hardened.

## Commands

```bash
# Run the API (PYTHONPATH must include repo root so `app.*` imports resolve)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Install deps
pip install -r requirements.txt
```

The API entry point is `app.main:app`. (The README's `uvicorn api:app` is stale —
use `app.main:app`, matching the Dockerfile and `.vscode/launch.json`.)

Swagger UI for manual endpoint testing: `http://127.0.0.1:8000/docs`.

There is no test suite, linter, or build step configured.

## Architecture

Request flow is layered: **routers → services → utils**, with config read once
from `.env` at import time.

- `app/main.py` — FastAPI app, CORS, router registration.
- `app/routers/files.py` — `/files` upload (PDF only, 10 MB cap), list, delete.
  Upload validates, saves to disk via `FileService`, then synchronously embeds
  via `EmbeddingService` (noted in code as work that ideally belongs in a worker).
- `app/routers/chat.py` — `/chat` POST; delegates to a module-level `RAGService`.
- `app/services/file_service.py` — local file persistence under `SAVE_FILES_PATH`.
- `app/services/embedding_service.py` — PDF → page `Document`s (pypdf) → chunks →
  one Chroma collection per file.
- `app/services/rag_service.py` — the RAG chain (see below).
- `app/utils/langchain.py` — `LangchainUtils`: shared VoyageAI embeddings, text
  splitter, and the single `chromadb.PersistentClient`. Owns collection
  create/delete/list and retriever construction.
- `app/config/settings.py` — loads all env vars; defines `ALLOWED_TYPE` (.pdf)
  and `MAX_FILE_SIZE_BYTES`.

### Key design points

- **One Chroma collection per uploaded file.** `EmbeddingService.process_file`
  creates a collection named after the (sanitized) filename. `create_collection`
  deletes any existing same-name collection first, so re-uploading replaces it.
  Collection names are sanitized in `LangchainUtils.sanitize_collection_name`
  (strips extension, replaces invalid chars, enforces 3–512 char Chroma rules).

- **Retrieval fans out across ALL collections.** `RAGService.ask` builds one
  retriever per collection (MMR, `k=5`, `fetch_k=20`) via `get_retrievers()`,
  runs them in a `RunnableParallel`, then `merge_docs` flattens results and tags
  each chunk with its `source_collection` for `[Source N]` citation.

- **Two LLMs.** `llm` (`ANTHROPIC_MODEL`) generates answers; `fast_llm`
  (`ANTHROPIC_FAST_MODEL`) only reformulates the question. A `RunnableBranch`
  contextualizes (rewrites the question to standalone using history) **only when
  chat history is non-empty**; otherwise the raw question passes through.

- **Chat history is in-memory and global.** `RAGService` is instantiated once at
  import in `chat.py`, and `self.chat_history` is a single shared list (capped at
  20 messages). All callers share one conversation; history does not survive a
  restart. Note `_reformulate_question` references a non-existent
  `REFORMULATE_PROMPT` — it is dead code; the live path uses `CONTEXTUALIZE_PROMPT`.

### Embedding consistency caveat

Embeddings are configured in two places in `LangchainUtils`: `embeddings`
(`VoyageAIEmbeddings`, used by LangChain `Chroma` for retrieval) and
`embeddings_fn` (`VoyageAIEmbeddingFunction`, passed to the raw chromadb
collection on write). Both must use the same `VOYAGEAI_MODEL` or write-time and
read-time vectors will mismatch.

## Environment

Copy `.env.example` to `.env` and fill in keys. Required vars (all read in
`settings.py`, no defaults — missing/empty values raise at startup):
`SAVE_FILES_PATH`, `CHROMA_PERSIST_PATH`, `CHUNK_SIZE`, `CHUNK_OVERLAP`,
`VOYAGEAI_API_KEY`, `VOYAGEAI_MODEL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`,
`ANTHROPIC_FAST_MODEL`.

Persistent state lives under `storage/` (`uploads/` for source files,
`chroma_db/` for vectors).
