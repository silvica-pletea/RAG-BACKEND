import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
SAVE_FILES_PATH = Path(os.getenv("SAVE_FILES_PATH"))
CHROMA_PERSIST_PATH = Path(os.getenv("CHROMA_PERSIST_PATH"))
VOYAGEAI_API_KEY = os.getenv("VOYAGEAI_API_KEY")
VOYAGEAI_MODEL = os.getenv("VOYAGEAI_MODEL")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL")
ANTHROPIC_FAST_MODEL = os.getenv("ANTHROPIC_FAST_MODEL")

ALLOWED_TYPE: str = '.pdf'
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024