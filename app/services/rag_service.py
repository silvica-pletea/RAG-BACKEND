from app.config.settings import ANTHROPIC_MODEL, ANTHROPIC_FAST_MODEL, ANTHROPIC_API_KEY
from app.config.constants import SearchType
from langchain_anthropic import ChatAnthropic
from app.services.file_service import FileService
from app.utils.langchain import LangchainUtils
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnableBranch


file_service = FileService()
langchain_utils = LangchainUtils()

class RAGService:

    # Reformulates the question into a standalone one using chat history
    CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """Given the chat history and the latest user question, reformulate \
    the question to be standalone – meaning it can be understood without the chat history. \
    Do NOT answer the question, just reformulate it if needed. \
    If it's already standalone, return it as is."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    QA_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.

        Rules:
        - Answer ONLY using information found in the context below. Do not add outside knowledge.
        - Return the exact paragraph(s) from the context that answer the question. Do not rephrase, summarize, or modify the text in any way.
        - If the answer is not in the context, respond exactly with: "I could not find relevant information in the uploaded documents."
        - After the paragraph(s), cite the source using the exact [Source: name] label shown in the context.


        Context:
        {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

    #constructor
    def __init__(self):
        
        self.llm = ChatAnthropic(
            api_key=ANTHROPIC_API_KEY,
            model_name=ANTHROPIC_MODEL,
            max_tokens=1000,
            temperature=0
        )

        self.fast_llm = ChatAnthropic(
            api_key=ANTHROPIC_API_KEY,
            model_name=ANTHROPIC_FAST_MODEL,
            max_tokens=1000,
            temperature=0
        )
        
        # One chat history per search mode
        self.chat_histories = {SearchType.HYBRID: [], SearchType.SEMANTIC: [], SearchType.BM25: []}

    def _merge_docs(self, inputs: dict) -> str:
        all_docs = []
        for collection_name, docs in inputs.items():
            for doc in docs:
                doc.metadata["source_collection"] = collection_name
                all_docs.append(doc)

        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source_collection', 'unknown')}]\n{d.page_content}"
            for d in all_docs
        )

    def ask(self, question: str, type: SearchType) -> str:
        if type == SearchType.HYBRID:
            retrievers = langchain_utils.get_hybrid_retrievers()
        elif type == SearchType.SEMANTIC:
            retrievers = langchain_utils.get_vector_retrievers()
        else:
            retrievers = langchain_utils.get_bm25_retrievers()
            
        retrieve_all = RunnableParallel(**retrievers)

        chat_history = self.chat_histories[type]

        # Only reformulate if there is chat history
        contextualize_question = RunnableBranch(
            (
                # If chat_history is not empty → reformulate
                lambda x: len(x["chat_history"]) > 0,
                RAGService.CONTEXTUALIZE_PROMPT | self.fast_llm | StrOutputParser(),
            ),
            # Otherwise → return the question as is
            RunnableLambda(lambda x: x["question"]),
        )

        chain = (
            # Step 1 — keep original input, add the reformulated question
            RunnableParallel(
                reformulated_question=contextualize_question,
                chat_history=RunnableLambda(lambda x: x.get("chat_history", [])),
                original_question=RunnableLambda(lambda x: x["question"]),
            )
            # Step 2 — retrieve using the reformulated question
            | RunnableLambda(lambda x: {
                "retrieved": retrieve_all.invoke(x["reformulated_question"]),
                "question": x["reformulated_question"],
                "chat_history": x["chat_history"],
            })
            # Step 3 — merge docs + build QA input
            | RunnableLambda(lambda x: {
                "context": self._merge_docs(x["retrieved"]),
                "question": x["question"],
                "chat_history": x["chat_history"],
            })
            # Step 4 — answer
            | RAGService.QA_PROMPT
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke({
            "chat_history": chat_history,
            "question": question
        })

        print(answer)

        # Update history (keep last 10 turns)
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=answer))
        if len(chat_history) > 20:
            del chat_history[:-20]

        return answer