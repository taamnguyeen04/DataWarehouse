import torch
import json
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

from retrieve import PaperRetriever
from config import Config


def load_test_queries(queries_path: str) -> List[Dict]:
    """
    Load test queries from JSONL file.

    Args:
        queries_path: Path to test_queries.jsonl file

    Returns:
        List of dicts with '_id' and 'text' fields
    """
    queries = []
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line.strip())
            queries.append(query)

    print(f"Loaded {len(queries)} test queries from {queries_path}")
    return queries


def generate_test_results(
    retriever: PaperRetriever,
    queries: List[Dict],
    top_k: int = 1000,
    metric_type: str = "L2"
) -> Dict[str, Dict[str, float]]:
    """
    Generate retrieval results for all test queries.

    Args:
        retriever: PaperRetriever instance
        queries: List of query dicts with '_id' and 'text'
        top_k: Number of documents to retrieve per query
        metric_type: Similarity metric type

    Returns:
        Results dict in format: {query_id: {doc_id: score, ...}, ...}
    """
    results = {}

    print(f"\nGenerating retrieval results for {len(queries)} queries...")
    print(f"Top-K: {top_k}, Metric: {metric_type}")

    for query in tqdm(queries, desc="Processing queries"):
        query_id = query['_id']
        query_text = query['text']

        # Retrieve top-k documents
        retrieved_docs = retriever.search(
            patient_text=query_text,
            top_k=top_k,
            metric_type=metric_type
        )

        # Format results for this query
        query_results = {}
        for doc in retrieved_docs:
            doc_id = str(doc['pmid'])  # Convert to string to match evaluation format
            score = doc['score']
            query_results[doc_id] = score

        results[query_id] = query_results

    return results


def save_results(results: Dict, output_path: str):
    """
    Save results to JSON file.

    Args:
        results: Results dict
        output_path: Path to output JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"Total queries: {len(results)}")

    # Print sample result
    if results:
        sample_query_id = list(results.keys())[0]
        sample_results = results[sample_query_id]
        print(f"\nSample result for query '{sample_query_id}':")
        print(f"  Number of retrieved documents: {len(sample_results)}")
        top_3_docs = list(sample_results.items())[:3]
        for doc_id, score in top_3_docs:
            print(f"    {doc_id}: {score:.4f}")


def main():
    # Configuration
    QUERIES_PATH = "C:/Users/tam/Desktop/Data/Data Warehouse/ReCDS_benchmark/queries/test_queries.jsonl"
    MODEL_PATH = "best_model.pt"
    OUTPUT_PATH = "./PAR/results/test_results_3.json"
    TOP_K = 1000
    METRIC_TYPE = "L2"
    USE_PRETRAINED = False # Set to False to use fine-tuned model
    DEVICE = None # None for auto-detect

    print("="*80)
    print("Generate Test Results for PAR Task")
    print("="*80)
    print(f"Queries path: {QUERIES_PATH}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Top-K: {TOP_K}")
    print(f"Metric type: {METRIC_TYPE}")
    print(f"Use pretrained: {USE_PRETRAINED}")
    print("="*80)

    # Load test queries
    queries = load_test_queries(QUERIES_PATH)

    # Initialize retriever
    print("\nInitializing retriever...")
    retriever = PaperRetriever(
        model_path=MODEL_PATH if not USE_PRETRAINED else None,
        use_pretrained=USE_PRETRAINED,
        device=DEVICE
    )

    # Generate results
    results = generate_test_results(
        retriever=retriever,
        queries=queries,
        top_k=TOP_K,
        metric_type=METRIC_TYPE
    )

    # Save results
    save_results(results, OUTPUT_PATH)

    print("\n" + "="*80)
    print("Done! You can now evaluate using:")
    print(f"python evaluation.py --task PAR --split test --result_path {OUTPUT_PATH}")
    print("="*80)


if __name__ == "__main__":
    main()