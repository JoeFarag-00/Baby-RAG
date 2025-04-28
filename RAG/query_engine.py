from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_core.vectorstores import VectorStoreRetriever

RAG_PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Context:
{context}

Question: {question}

Answer:
"""

def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(llm: ChatGroq, retriever: VectorStoreRetriever) -> Runnable:
    print("Building RAG chain...")
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: _format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    print("RAG chain built successfully.")
    return rag_chain_with_source

def query_rag(chain: Runnable, question: str) -> Dict[str, Any]:

    if not question:
        print("Query Error: Question cannot be empty.")
        return {"question": question, "context": [], "answer": "Error: Question cannot be empty."}

    print(f"Querying RAG chain with question: '{question}'")
    try:
        result = chain.invoke(question)
        print("Query completed.")
        return result
    except Exception as e:
        print(f"Error during query execution: {e}")
        return {"question": question, "context": [], "answer": f"Error processing query: {e}"} 