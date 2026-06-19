# Lightweight RAG Backend

A lightweight RAG (Retrieval-Augmented Generation) app combining FastAPI,LangChain, ChromaDB, the Anthropic API, and VoyageAI embeddings.

## Structure

```text
app/
├── config/
│   └── settings.py          # Reads and loads all keys from .env file
├── routes/
│   ├── chat.py              # API endpoints for chat/conversation interactions
│   └── files.py             # API endpoints for uploading and managing source files
├── services/
│   ├── embedding_service.py # Handles text chunking and embedding generation via VoyageAI
│   ├── file_service.py      # Handles local file storage
│   └── rag_service.py       # Core RAG pipeline (retrieval + generation)
├── utils/
│   └── langchain.py         # LangChain helper functions and wrappers
└── main.py                  # The entry point of the FastAPI application
```

### IMPORTANT

This setup is intended for local development and testing only. Do not use this code as-is in production environments.

Chat history is kept in a single in-memory `RAGService` instance shared by all requests, so the API is effectively single-user and history is lost on restart.

# Quick Start – From Zero to Working RAG

## 1. Clone the repository

```bash
git clone https://github.com/silvica-pletea/RAG-BACKEND.git
cd RAG-BACKEND
```

## 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

```bash
cp .env.example .env
```
## Open .env and fill in your API keys

```text
# Local directory where uploaded source files are saved before processing
SAVE_FILES_PATH=./storage/files 

# Local directory where ChromaDB persists its vector data (collections, embeddings, metadata)
CHROMA_PERSIST_PATH=./storage/chroma_db

# Maximum number of characters (or tokens, depending on splitter) per text chunk when splitting documents before embedding
CHUNK_SIZE=500

# Number of characters (or tokens) that overlap between consecutive chunks, used to preserve context across chunk boundaries
CHUNK_OVERLAP=50

# API key for authenticating with VoyageAI's embedding service. Navigate to https://www.voyageai.com/ to get an API key (free to get started)
VOYAGEAI_API_KEY= 

# VoyageAI embedding model used to convert text into vectors.
VOYAGEAI_MODEL=voyage-3

# API key for authenticating with the Anthropic API
ANTHROPIC_API_KEY=

# Main Claude model used for generating answers / complex reasoning
ANTHROPIC_MODEL=claude-sonnet-4-6

# Lighter/faster Claude model used for quick tasks like question reformulation (contextualize step) where lower latency matters more than maximum reasoning quality (e.g. claude-haiku-4-5-20251001)
ANTHROPIC_FAST_MODEL=claude-haiku-4-5-20251001 
```

## 5. Start the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 6. Explore the API

Open `http://127.0.0.1:8000/docs` in your browser to view and test all available endpoints using the Swagger UI.

## Docker builder

### 1. Build the image

```bash
docker build -t rag-backend .
```

### 2. Run the container

```bash
docker run -d \
  --name rag-backend \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/storage:/app/storage \
  rag-backend
```

### 3. Check the logs

```bash
docker logs -f rag-backend
```

### 4. Explore the API
Open `http://127.0.0.1:8000/docs` in your browser to view and test all available endpoints using the Swagger UI.

### 5. Stop the container

```bash
docker stop rag-backend
docker rm rag-backend
```

