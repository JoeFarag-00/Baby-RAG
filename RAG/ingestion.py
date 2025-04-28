from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import config 

def split_documents(docs: List[Document]) -> List[Document]:
    if not docs:
        return []
    print(f"Splitting {len(docs)} document(s) into chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} chunks.")
    return splits

def ingest_documents(docs: List[Document], vector_store: PineconeVectorStore):

    if not isinstance(docs, list) or not all(isinstance(d, Document) for d in docs):
         raise TypeError("Input must be a list of Document objects.")

    if not docs:
        print("No documents provided for ingestion. Skipping.")
        return

    splits = split_documents(docs)
    if not splits:
         print("No chunks created after splitting. Ingestion skipped.")
         return

    print(f"Indexing {len(splits)} chunks into Pinecone...")
    try:
        vector_store.add_documents(splits, batch_size=100)
        print("Indexing complete.")
    except Exception as e:
        print(f"Error during indexing to Pinecone: {e}")
        raise