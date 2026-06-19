import re
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import VoyageAIEmbeddingFunction
from langchain_voyageai import VoyageAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from app.config.settings import CHUNK_SIZE, CHUNK_OVERLAP, VOYAGEAI_MODEL, VOYAGEAI_API_KEY, CHROMA_PERSIST_PATH

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

    def _collection_exists(self, file_name: str) -> bool:
        collections = self._get_collections()
        collection_name = self.sanitize_collection_name(file_name)
        return False if collection_name not in collections else True
        
    def _get_collections(self) -> list[str]:
        return [c.name for c in LangchainUtils.client.list_collections()]
    
    def _get_vector_store(self, collection_name: str) -> Chroma:
        return Chroma(
            collection_name = collection_name,
            embedding_function = LangchainUtils.embeddings,
            persist_directory = CHROMA_PERSIST_PATH
        )
    
    def get_retrievers(self):
        collection_names = self._get_collections()
        return {
            name: Chroma(
                collection_name=name,
                embedding_function= LangchainUtils.embeddings,
                persist_directory=CHROMA_PERSIST_PATH,
            ).as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20}) for name in collection_names
        }

    def create_collection(self, file_name: str) -> Collection:
        exception = None
        try:
            self.delete_collection(file_name)
        except Exception as e:
            if type(e) is not FileNotFoundError:
                exception = e
        finally:
            if exception is None:
                collection = LangchainUtils.client.create_collection(name=self.sanitize_collection_name(file_name), embedding_function=LangchainUtils.embeddings_fn)
            else:
                raise exception
        return collection

    def delete_collection(self, file_name: str) -> None:
        if self._collection_exists(file_name):
            LangchainUtils.client.delete_collection(self.sanitize_collection_name(file_name))
        else:
            raise FileNotFoundError(f"Collection '{file_name}' not found")

    def sanitize_collection_name(self, file_name: str) -> str:
        # remove extension
        name = file_name.rsplit(".", 1)[0]
        # replace spaces and invalid chars with underscore
        name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
        # must start and end with alphanumeric
        name = name.strip("._-")
        # ensure minimum 3 characters
        if len(name) < 3:
            name = name + "_doc"
        # ensure maximum 512 characters
        return name[:512]