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
- `app/services/embedding_service.py` — PDF → page `Document`s (pypdf) → chunks,
  added to the single shared Chroma collection tagged by `source` metadata.
- `app/services/rag_service.py` — the RAG chain (see below).
- `app/utils/langchain.py` — `LangchainUtils`: shared VoyageAI embeddings, text
  splitter, the single `chromadb.PersistentClient`, and the one `documents`
  collection. Owns add/delete-by-source and retriever construction, including a
  cached BM25 retriever.
- `app/config/settings.py` — loads all env vars; defines `ALLOWED_TYPE` (.pdf),
  `MAX_FILE_SIZE_BYTES`, `NO_DOCS` (per-retriever top-k), and `COLLECTION_NAME`.

### Key design points

- **Single Chroma collection for all files.** Every uploaded file's chunks go
  into one collection (`COLLECTION_NAME = "documents"`), tagged with a `source`
  metadata field (the original filename). `EmbeddingService.process_file` calls
  `LangchainUtils.add_documents`, prefixing chunk IDs with a sanitized filename
  (`sanitize_source_id`) so IDs don't collide across files. Deleting a file
  calls `delete_by_source`, which removes chunks via `where={"source": ...}`
  and raises `FileNotFoundError` if none match.

- **Retrieval hits the single collection directly** — no more per-collection
  fan-out or `RunnableParallel`. `RAGService.ask` picks one retriever
  (`get_bm25_retriever`, `get_vector_retriever`, or `get_hybrid_retriever`,
  MMR `k=NO_DOCS`, `fetch_k=20`) based on `SearchType`, invokes it once, and
  `_collect_docs` formats each chunk with `[Source: <filename>]` from its
  `source` metadata for citations.

- **BM25 retriever is cached** on `LangchainUtils._bm25_retriever_cache` (a
  class attribute, shared across instances) and invalidated on every
  `add_documents`/`delete_by_source` call, avoiding a full corpus reload +
  rebuild on every request. Building `BM25Retriever` on an empty corpus divides
  by zero internally (`rank_bm25.BM25Okapi`), so `get_bm25_retriever` returns a
  `RunnableLambda(lambda _: [])` stand-in when the collection has no documents.

- **Two LLMs.** `llm` (`ANTHROPIC_MODEL`) generates answers; `fast_llm`
  (`ANTHROPIC_FAST_MODEL`) only reformulates the question. A `RunnableBranch`
  contextualizes (rewrites the question to standalone using history) **only when
  chat history is non-empty**; otherwise the raw question passes through.

- **Chat history is in-memory, global, and per search mode.** `RAGService` is
  instantiated once at import in `chat.py`, and `self.chat_histories` holds one
  list per `SearchType` (capped at 20 messages each). All callers share the same
  conversation per mode; history does not survive a restart, and a follow-up
  asked under one mode has no knowledge of turns asked under another.
  `ChatRequest.type` defaults to `SearchType.HYBRID`.

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
