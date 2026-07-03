import re
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import VoyageAIEmbeddingFunction
from langchain_voyageai import VoyageAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnableLambda
from app.config.settings import CHUNK_SIZE, CHUNK_OVERLAP, NO_DOCS, COLLECTION_NAME, VOYAGEAI_MODEL, VOYAGEAI_API_KEY, CHROMA_PERSIST_PATH

class LangchainUtils:

    embeddings = VoyageAIEmbeddings(
        voyage_api_key=VOYAGEAI_API_KEY,
        model=VOYAGEAI_MODEL
    )

    embeddings_fn = VoyageAIEmbeddingFunction(
        api_key=VOYAGEAI_API_KEY,
        model_name=VOYAGEAI_MODEL
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,  # chunk size (characters)
        chunk_overlap=CHUNK_OVERLAP,  # chunk overlap (characters)
        length_function=len, # function to compute the length of the text
        add_start_index=True,  # track index in original document
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)

    _bm25_retriever_cache: BM25Retriever | RunnableLambda | None = None

    def _get_or_create_collection(self) -> Collection:
        return LangchainUtils.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=LangchainUtils.embeddings_fn,
        )

    def _load_documents(self) -> list[Document]:
        col = self._get_or_create_collection()
        data = col.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(data["documents"], data["metadatas"])
        ]

    def _get_vector_store(self) -> Chroma:
        return Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=LangchainUtils.embeddings,
            persist_directory=CHROMA_PERSIST_PATH,
        )

    # BM25 is keyword-based (good for exact matches)
    def get_bm25_retriever(self) -> BM25Retriever | RunnableLambda:
        if LangchainUtils._bm25_retriever_cache is not None:
            return LangchainUtils._bm25_retriever_cache

        docs = self._load_documents()
        # BM25Okapi divides by the corpus size, so it cannot be built on an empty corpus
        if not docs:
            retriever = RunnableLambda(lambda _: [])
        else:
            retriever = BM25Retriever.from_documents(docs, k=NO_DOCS)

        LangchainUtils._bm25_retriever_cache = retriever
        return retriever

    def _invalidate_bm25_cache(self) -> None:
        LangchainUtils._bm25_retriever_cache = None

    # Vector retriever is semantic (good for meaning)
    def get_vector_retriever(self):
        return self._get_vector_store().as_retriever(
            search_type="mmr", search_kwargs={"k": NO_DOCS, "fetch_k": 20}
        )

    # Combine both - gets the best of keyword and semantic search
    def get_hybrid_retriever(self) -> EnsembleRetriever:
        return EnsembleRetriever(
            retrievers=[self.get_bm25_retriever(), self.get_vector_retriever()],
            weights=[0.4, 0.6],  # Weight semantic search slightly higher
        )

    def add_documents(self, source: str, chunks: list[Document]) -> None:
        prefix = self.sanitize_source_id(source)
        documents = [d.page_content for d in chunks]
        metadatas = [d.metadata for d in chunks]
        ids = [f"{prefix}_{i}" for i in range(len(chunks))]
        collection = self._get_or_create_collection()
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        self._invalidate_bm25_cache()

    def delete_by_source(self, source: str) -> None:
        collection = self._get_or_create_collection()
        existing = collection.get(where={"source": source}, include=[])
        if not existing["ids"]:
            raise FileNotFoundError(f"No chunks found for source '{source}'")
        collection.delete(where={"source": source})
        self._invalidate_bm25_cache()

    def sanitize_source_id(self, source: str) -> str:
        # remove extension
        name = source.rsplit(".", 1)[0]
        # replace spaces and invalid chars with underscore
        name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
        # must start and end with alphanumeric
        name = name.strip("._-")
        # ensure minimum 3 characters
        if len(name) < 3:
            name = name + "_doc"
        return name
