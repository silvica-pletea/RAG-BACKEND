from pypdf import PdfReader
from app.utils.langchain import LangchainUtils
from langchain_core.documents import Document

langchain_utils = LangchainUtils()

class EmbeddingService:

    # Constructor
    def __init__(self):
        self.text_splitter = langchain_utils.text_splitter

    def process_file(self, file_path, file_name) -> None:
        documents = self._read_pdf(file_path, file_name)
        chunks = self._get_chunk_text(documents)
        langchain_utils.add_documents(file_name, chunks)

    def delete_file(self, file_name) -> None:
        langchain_utils.delete_by_source(file_name)

    # Private functions
    def _get_chunk_text(self, documents: list[Document]) -> list[Document]:
        return self.text_splitter.split_documents(documents)

    def _read_pdf(self, file_path, file_name) -> list[Document]:
        reader = PdfReader(file_path)
        documents = []
        for page in reader.pages:
            text = page.extract_text().strip()
            if text:  # filter empty pages
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_name, "page": page.page_number},
                    )
                )
        return documents
