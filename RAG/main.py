import os
import sys
import traceback
from typing import List, Dict, Any

import config
from components import init_llm, init_embeddings, init_vector_store
from data_loader import load_from_url, load_from_directory
from ingestion import ingest_documents
from query_engine import build_rag_chain, query_rag
from evaluation import run_langsmith_evaluation

try:
    from pinecone import Pinecone, ApiException
except ImportError:
    Pinecone = None

from langchain_core.runnables import RunnablePassthrough


def get_vector_count(index_name: str) -> int | None:
    if not Pinecone or not config.PINECONE_API_KEY:
        return None
    try:
        print(f"Checking vector count for Pinecone index '{index_name}'...")
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        index_list = pc.list_indexes()
        if index_name not in (idx.name for idx in index_list):
            print(f"Warning: Pinecone index '{index_name}' not found.")
            return None 

        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        count = stats.total_vector_count
        print(f"Index '{index_name}' contains {count} vectors.")
        return count
    except ApiException as api_e:
        print(f"Pinecone API error checking index stats: {api_e}")
        if api_e.status == 404:
            print(f"Index '{index_name}' not found (API 404).")
        return None 
    except Exception as e:
        print(f"Could not check Pinecone index stats: {e}")
        return None

def get_user_ingestion_choice(vector_count: int | None) -> bool:
    if vector_count is None: 
        return input("Attempt data ingestion anyway? (yes/no): ").lower().strip() == 'yes'
    elif vector_count == 0:
        return input("Proceed with ingestion? (yes/no): ").lower().strip() == 'yes'
    else:
        return input("Do you want to ADD MORE data? (yes/no): ").lower().strip() == 'yes'

def perform_interactive_ingestion(vector_store) -> None:
    docs_to_ingest = []
    while not docs_to_ingest:
        load_choice = input("Load data from a 'link' (1) or local 'document' folder (2)? Enter 1 or 2: ").strip()

        if load_choice == '1':
            source_input = input("Please enter the URL: ").strip()
            if source_input:
                docs_to_ingest = load_from_url(source_input)
            else:
                print("No URL provided.")
        elif load_choice == '2':
            script_dir = os.path.dirname(__file__)
            default_docs_folder = os.path.abspath(os.path.join(script_dir, "Documents"))
            source_input = input(f"Enter path to documents folder [default: {default_docs_folder}]: ").strip()
            if not source_input:
                source_input = default_docs_folder 

            if os.path.isdir(source_input):
                docs_to_ingest = load_from_directory(source_input)
            else:
                print(f"Invalid directory path: {source_input}")
        else:
            print("Invalid choice. Please enter 1 or 2.")

        if not docs_to_ingest:
            try_again = input("Try loading again? (yes/no): ").lower().strip()
            if try_again != 'yes':
                print("Skipping ingestion.")
                return # Exit ingestion process
        else:
            # Perform ingestion
            try:
                print("Starting ingestion process...")
                ingest_documents(docs_to_ingest, vector_store)
                print("Ingestion successful.")
                break # Exit loop after successful ingestion
            except Exception as e:
                print(f"Error during ingestion: {e}")
                # Decide if you want to break or allow retry
                print("Ingestion failed.")
                return # Exit ingestion process

def run_interactive_query_loop(rag_chain) -> None:
    print("\n--- Ready to Query --- (Type 'quit' to exit)")
    while True:
        question = input("\nEnter your question: ").strip()
        if question.lower() == 'quit':
            break
        if not question:
            continue

        result = query_rag(rag_chain, question)

        print("\n--- Query Result ---")
        print(f"Question: {result.get('question', 'N/A')}")
        print(f"Retrieved Context Docs: {len(result.get('context', []))}")
      
        print(f"\nAnswer:\n{result.get('answer', 'No answer generated.')}")
        print("--------------------")

def run_interactive_evaluation(rag_chain) -> None:
    run_eval = input("\nRun LangSmith evaluation? (yes/no): ").lower().strip()
    if run_eval == 'yes':
        dataset_name = input("Enter the LangSmith dataset name (e.g., rag-eval-dataset): ").strip()
        if dataset_name:
            print("Preparing evaluation...")

            def llm_or_chain_factory():
              
                chain_for_eval = (
                    RunnablePassthrough.assign(question=lambda x: x["input"])
                    | rag_chain 
                    | (lambda x: {"output": x['answer']})
                )
                return chain_for_eval

            metadata = {
                "llm_model": config.LLM_MODEL_NAME,
                "embedding_model": config.EMBEDDING_MODEL_NAME,
                "vector_store": "PineconeVectorStore",
                "index_name": config.PINECONE_INDEX_NAME,
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
            }

            eval_results = run_langsmith_evaluation(
                llm_or_chain_factory=llm_or_chain_factory,
                dataset_name=dataset_name,
                experiment_prefix="rag-refactored-run",
                metadata=metadata
            )
            if eval_results:
                print("Evaluation results object received (details in LangSmith).")
                # print(eval_results) # Optional: Print results details
            # Error messages handled within run_langsmith_evaluation

        else:
            print("No dataset name provided. Skipping evaluation.")
    else:
        print("Skipping LangSmith evaluation.")

def main():
    """Main function to run the RAG application."""
    print("--- Starting RAG Application ---")
    try:
        # 1. Check Environment Variables
        config.check_required_env_vars(config.REQUIRED_VARS)
        print("Environment variables checked.")

        # 2. Initialize Core Components
        print("Initializing components...")
        llm = init_llm()
        embeddings = init_embeddings()
        vector_store = init_vector_store(embeddings)
        retriever = vector_store.as_retriever()
        print("Core components initialized.")

        # 3. Build RAG Chain
        rag_chain = build_rag_chain(llm, retriever)

        # 4. Handle Ingestion (Optional, Interactive)
        current_vector_count = get_vector_count(config.PINECONE_INDEX_NAME)
        if get_user_ingestion_choice(current_vector_count):
            perform_interactive_ingestion(vector_store)
        else:
            print("Skipping data ingestion step.")

        # 5. Start Query Loop
        run_interactive_query_loop(rag_chain)

        run_interactive_evaluation(rag_chain)

    except EnvironmentError as e:
        print(f"\n--- CONFIGURATION ERROR ---", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        print("Please check your .env file and environment variables.", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n--- INITIALIZATION ERROR ---", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        print("Check API keys, service availability (Groq, Voyage, Pinecone), and index configuration.", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"\n--- IMPORT ERROR ---", file=sys.stderr)
        print(f"A required package might be missing: {e}", file=sys.stderr)
        print("Please ensure all dependencies are installed (e.g., pip install -r requirements.txt).", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n--- User interruption detected. Exiting. ---")
        sys.exit(0)
    except Exception as e:
        print(f"\n--- UNEXPECTED ERROR ---", file=sys.stderr)
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n--- RAG Application Finished ---")

if __name__ == "__main__":
    main()
