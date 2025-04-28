import warnings
import config

from langchain_groq import ChatGroq
from langchain_voyageai import VoyageAIEmbeddings
from langchain_pinecone import PineconeVectorStore

try:
    from pinecone import Pinecone
except ImportError:
    Pinecone = None

def init_llm() -> ChatGroq:
    try:
        llm = ChatGroq(
            temperature=0,
            groq_api_key=config.GROQ_API_KEY,
            model_name=config.LLM_MODEL_NAME,
        )
        print(f"Groq LLM initialized ({config.LLM_MODEL_NAME}).")
        return llm
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChatGroq LLM: {e}")

def init_embeddings() -> VoyageAIEmbeddings:
    try:
        embeddings = VoyageAIEmbeddings(
            model=config.EMBEDDING_MODEL_NAME,
            voyage_api_key=config.VOYAGE_API_KEY,
        )
        print(f"VoyageAI Embeddings initialized ({config.EMBEDDING_MODEL_NAME}).")
        return embeddings
    except Exception as e:
        raise RuntimeError(f"Failed to initialize VoyageAIEmbeddings: {e}")

def init_vector_store(embeddings: VoyageAIEmbeddings) -> PineconeVectorStore:
    print(f"Initializing PineconeVectorStore with index: {config.PINECONE_INDEX_NAME}")

    # Optional: Check if index exists using Pinecone client
    if Pinecone and config.PINECONE_API_KEY:
        try:
            pc = Pinecone(api_key=config.PINECONE_API_KEY)
            index_list = pc.list_indexes()
            if config.PINECONE_INDEX_NAME not in (idx.name for idx in index_list):
                 warnings.warn(
                     f"Pinecone index '{config.PINECONE_INDEX_NAME}' not found. "
                     "Please ensure the index exists and is configured correctly.",
                     UserWarning
                 )
            else:
                print(f"Confirmed Pinecone index '{config.PINECONE_INDEX_NAME}' exists.")
        except Exception as e:
            warnings.warn(f"Could not verify Pinecone index existence or configuration: {e}", UserWarning)
    elif not Pinecone:
        print("pinecone error")

    try:
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=config.PINECONE_INDEX_NAME,
            embedding=embeddings,
        )
        print("PineconeVectorStore initialized successfully.")
        return vector_store
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize PineconeVectorStore: {e}. "
            f"Ensure index '{config.PINECONE_INDEX_NAME}' exists, is configured for "
            f"dimension {config.PINECONE_EMBEDDING_DIMENSION} (for {config.EMBEDDING_MODEL_NAME}), "
            f"and API keys are correct."
        )