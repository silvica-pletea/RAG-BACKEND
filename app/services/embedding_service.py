from pypdf import PdfReader
from app.config.settings import CHROMA_PERSIST_PATH
from app.utils.langchain import LangchainUtils
from langchain_core.documents import Document

langchain_utils = LangchainUtils()

class EmbeddingService:

    # Constructor
    def __init__(self):
        self.text_splitter = langchain_utils.text_splitter
        self.embeddings = langchain_utils.embeddings

    def process_file(self, file_path, file_name) -> None:
        documents = self._read_pdf(file_path)
        chunks = self._get_chunk_text(documents)
        self._save_to_chroma(file_name, chunks)

    def delete_file(self, file_name) -> None:
        langchain_utils.delete_collection(file_name)

    # Private functions
    def _get_chunk_text(self, documents: list[Document]) -> list[Document]:
        return self.text_splitter.split_documents(documents)
    
    def _save_to_chroma(self, file_name: str, chunks: list[Document]) -> None:
        file_path = CHROMA_PERSIST_PATH
        file_path.parent.mkdir(parents=True, exist_ok=True)
        documents = [d.page_content for d in chunks]
        metadatas = [d.metadata for d in chunks]
        ids = [f"doc_{i}" for i in range(len(chunks))]
        collection = langchain_utils.create_collection(file_name)
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    def _read_pdf(self, filename) -> list[Document]:
        reader = PdfReader(filename)
        return [
            Document(page_content = p.extract_text().strip(), metadata= {"source": filename, "page": p.page_number })
            for p in reader.pages if p.extract_text().strip() #filter empty text
        ]
