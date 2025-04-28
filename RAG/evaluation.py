import warnings
from typing import Optional, List, Dict, Any, Callable

from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.errors import NotFoundError 
from langchain_core.runnables import RunnablePassthrough, Runnable 
import config

def run_langsmith_evaluation(
    llm_or_chain_factory: Callable[[], Runnable],
    dataset_name: str,
    experiment_prefix: str,
    metadata: Dict[str, Any],
    summary_evaluators: Optional[List[str]] = None,
):
   
    print(f"\n--- Starting LangSmith Evaluation ---")
    print(f"Dataset: {dataset_name}")
    print(f"Experiment Prefix: {experiment_prefix}")

    if summary_evaluators is None:
        summary_evaluators = ["cot_qa"]

    try:
        langsmith_client = Client(api_key=config.LANGCHAIN_API_KEY)
        try:
            langsmith_client.read_dataset(dataset_name=dataset_name)
            print(f"Found LangSmith dataset: {dataset_name}")
        except Exception as e:
            print(f"Error accessing LangSmith dataset '{dataset_name}': {e}")
            return None

        evaluation_results = evaluate(
            llm_or_chain_factory,
            data=dataset_name,
            experiment_prefix=experiment_prefix,
            summary_evaluators=summary_evaluators,
            metadata=metadata,
        )
        print(f"LangSmith evaluation completed successfully.")
        return evaluation_results

    except Exception as e:
        print(f"LangSmith evaluation failed: {e}")
        return None
    finally:
         print(f"--- LangSmith Evaluation Finished ---")
