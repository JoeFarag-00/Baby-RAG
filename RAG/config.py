import os
import warnings
from dotenv import load_dotenv

def load_environment():
    load_dotenv()

def get_env_var(var_name: str, required: bool = True) -> str:
    """Gets an environment variable, raising an error if required and not found."""
    value = os.getenv(var_name)
    if required and not value:
        raise EnvironmentError(f"Missing required environment variable: {var_name}")
    return value if value else ""

def check_required_env_vars(required_vars: list[str]):
    """Checks if all required environment variables are set."""
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() != "true":
        warnings.warn(
            "LANGCHAIN_TRACING_V2 is not set to 'true'. LangSmith tracing may be disabled.", UserWarning
        )

load_environment()

GROQ_API_KEY = get_env_var("GROQ_API_KEY")
PINECONE_API_KEY = get_env_var("PINECONE_API_KEY")
VOYAGE_API_KEY = get_env_var("VOYAGE_API_KEY")
LANGCHAIN_API_KEY = get_env_var("LANGCHAIN_API_KEY")
PINECONE_INDEX_NAME = get_env_var("PINECONE_INDEX_NAME")
LANGCHAIN_TRACING_V2 = get_env_var("LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = get_env_var("LANGCHAIN_PROJECT", required=False)


EMBEDDING_MODEL_NAME = "voyage-3-large"
LLM_MODEL_NAME = "llama3-70b-8192"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PINECONE_EMBEDDING_DIMENSION = 1024 

REQUIRED_VARS = [
    "GROQ_API_KEY",
    "PINECONE_API_KEY",
    "VOYAGE_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "PINECONE_INDEX_NAME",
]