import sys
from app.config.constants import SearchType
from app.services.rag_service import RAGService
from app.utils.langchain import LangchainUtils
from langchain_classic.retrievers import EnsembleRetriever

l = LangchainUtils()
rag_service = RAGService();

def checkBM2():
    print('BM25 Search')
    retriever = l._get_bm25_retriever(collection_name)
    print_results(retriever)

def checkSemantic():
    print('Semantic Search')
    retriever = l._get_vector_retriever(collection_name)
    print_results(retriever)

def checkHybrid():
    print('Hybrid Search')
    vector_retriever = l._get_vector_retriever(collection_name)
    bm25_retriever = l._get_bm25_retriever(collection_name)
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6]
    )
    print_results(retriever)

def ask():
    response = rag_service.ask(question, SearchType.HYBRID)
    print(response)


def print_results(retriever):
    results = retriever.invoke(question)
    for i, doc in enumerate(results, 1):
        print(f"Match {i}:")
        print(doc.page_content)
        print("-" * 80)

if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "XDR-471"
    collection_name = sys.argv[2] if len(sys.argv) > 2 else "report"
    #checkBM2()
    #checkSemantic()
    #checkHybrid()
    ask()

