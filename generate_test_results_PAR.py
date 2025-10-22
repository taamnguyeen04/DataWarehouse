"""
Generate test results for PAR (Patient-to-Article Retrieval) task evaluation.

This script:
1. Loads test queries from test_queries.jsonl
2. Uses the trained model to retrieve relevant articles for each query
3. Generates a JSON file in the format required by evaluation.py

Output format:
{
    "query_id_1": {
        "doc_id_1": score1,
        "doc_id_2": score2,
        ...
    },
    "query_id_2": {
        "doc_id_3": score3,
        "doc_id_4": score4,
        ...
    },
    ...
}
"""

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
    metric_type: str = "COSINE"
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
    parser = argparse.ArgumentParser(
        description="Generate test results for PAR task evaluation"
    )
    parser.add_argument(
        "--queries_path",
        type=str,
        default=r"C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\queries\test_queries.jsonl",
        help="Path to test_queries.jsonl file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./PAR/checkpoints/best_model.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./PAR/results/test_results_1.json",
        help="Path to save output JSON file"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1000,
        help="Number of documents to retrieve per query (default: 1000 for R@1k metric)"
    )
    parser.add_argument(
        "--metric_type",
        type=str,
        default="COSINE",
        choices=["COSINE", "IP", "L2"],
        help="Similarity metric type (default: COSINE)"
    )
    parser.add_argument(
        "--use_pretrained",
        type=bool,
        default=True,
        help="Use pretrained model without fine-tuning (True/False)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu, default: auto)"
    )

    args = parser.parse_args()

    print("="*80)
    print("Generate Test Results for PAR Task")
    print("="*80)
    print(f"Queries path: {args.queries_path}")
    print(f"Model path: {args.model_path}")
    print(f"Output path: {args.output_path}")
    print(f"Top-K: {args.top_k}")
    print(f"Metric type: {args.metric_type}")
    print(f"Use pretrained: {args.use_pretrained}")
    print("="*80)

    # Load test queries
    queries = load_test_queries(args.queries_path)

    # Initialize retriever
    print("\nInitializing retriever...")
    retriever = PaperRetriever(
        model_path=args.model_path if not args.use_pretrained else None,
        use_pretrained=args.use_pretrained,
        device=args.device
    )

    # Generate results
    results = generate_test_results(
        retriever=retriever,
        queries=queries,
        top_k=args.top_k,
        metric_type=args.metric_type
    )

    # Save results
    save_results(results, args.output_path)

    print("\n" + "="*80)
    print("Done! You can now evaluate using:")
    print(f"python evaluation.py --task PAR --split test --result_path {args.output_path}")
    print("="*80)


if __name__ == "__main__":
    main()