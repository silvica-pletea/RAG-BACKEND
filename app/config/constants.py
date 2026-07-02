"""Application-wide constant values."""

from enum import StrEnum

class SearchType(StrEnum):
    """Supported chat retrieval modes."""

    HYBRID = "hybrid-search"
    SEMANTIC = "semantic-search"
    BM25 = "bm25-search"
