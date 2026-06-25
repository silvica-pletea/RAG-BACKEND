from app.config.settings import ANTHROPIC_MODEL, ANTHROPIC_FAST_MODEL, ANTHROPIC_API_KEY
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
            Only answer based on the context below.
            If the answer is not in the context, say "I could not find relevant information in the uploaded documents."

            Cite sources using the exact [Source: name] labels shown in the context.

            Format the answer as clean, readable Markdown:
            - Use ## and ### headings to separate sections.
            - Use **bold** for key terms, identifiers, and figures.
            - Use - bullet lists or numbered lists for steps or grouped facts.
            - Write short paragraphs separated by a blank line.
            - Always put a blank line before a heading or a list.

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
        
        self.chat_history = []

    def merge_docs(self, inputs: dict) -> str:
        all_docs = []
        for collection_name, docs in inputs.items():
            for doc in docs:
                doc.metadata["source_collection"] = collection_name
                all_docs.append(doc)

        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source_collection', 'unknown')}]\n{d.page_content}"
            for d in all_docs
        )

    def ask(self, question: str) -> str:
        retrievers = langchain_utils.get_retrievers()
        retrieve_all = RunnableParallel(**retrievers)

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
                "context": self.merge_docs(x["retrieved"]),
                "question": x["question"],
                "chat_history": x["chat_history"],
            })
            # Step 4 — answer
            | RAGService.QA_PROMPT
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke({
            "chat_history": self.chat_history,
            "question": question
        })

        # Update history (keep last 10 turns)
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]

        return answer