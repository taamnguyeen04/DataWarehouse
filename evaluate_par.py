import os
import torch
import numpy as np
from tqdm import tqdm
import json
from transformers import AutoTokenizer

from model import BiEncoder
from data_loader import PARDataset


class PAREvaluator:
    """
    Evaluator for PAR bi-encoder model
    Computes retrieval metrics: MRR, Recall@K, NDCG@K
    """
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

        # Load model
        self.model = BiEncoder(
            model_name=config['model_name'],
            embedding_dim=config['embedding_dim'],
            pooling=config['pooling']
        ).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")

    @torch.no_grad()
    def encode_queries(self, queries_file):
        """
        Encode all queries
        Returns:
            query_embeddings: dict {query_id: embedding}
        """
        print(f"Encoding queries from {queries_file}...")

        queries = {}
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line.strip())
                queries[query['_id']] = query['text']

        query_embeddings = {}

        for query_id, query_text in tqdm(queries.items(), desc="Encoding queries"):
            encoding = self.tokenizer(
                query_text,
                max_length=self.config['max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)

            embedding = self.model.encode_query(input_ids, attention_mask)
            query_embeddings[query_id] = embedding.cpu().numpy()

        return query_embeddings

    @torch.no_grad()
    def encode_corpus(self, corpus_file, max_docs=None):
        """
        Encode all documents in corpus
        Returns:
            doc_embeddings: dict {doc_id: embedding}
        """
        print(f"Encoding corpus from {corpus_file}...")

        doc_embeddings = {}
        count = 0

        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Encoding documents"):
                if max_docs and count >= max_docs:
                    break

                doc = json.loads(line.strip())
                doc_id = str(doc.get('_id', ''))
                title = doc.get('title', '').strip()
                abstract = doc.get('text', '').strip()

                if not title and not abstract:
                    continue

                doc_text = f"{title} {abstract}".strip()

                encoding = self.tokenizer(
                    doc_text,
                    max_length=self.config['max_length'],
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                embedding = self.model.encode_doc(input_ids, attention_mask)
                doc_embeddings[doc_id] = embedding.cpu().numpy()

                count += 1

        print(f"Encoded {len(doc_embeddings)} documents")
        return doc_embeddings

    def compute_retrieval_metrics(self, query_embeddings, doc_embeddings, qrels, k_values=[1, 5, 10, 20, 100]):
        """
        Compute retrieval metrics
        Args:
            query_embeddings: dict {query_id: embedding}
            doc_embeddings: dict {doc_id: embedding}
            qrels: dict {query_id: [relevant_doc_ids]}
            k_values: list of k values for Recall@K and NDCG@K
        Returns:
            metrics: dict with MRR, Recall@K, NDCG@K
        """
        print("Computing retrieval metrics...")

        # Convert embeddings to matrices
        doc_ids = list(doc_embeddings.keys())
        doc_matrix = np.vstack([doc_embeddings[doc_id] for doc_id in doc_ids])

        metrics = {
            'mrr': [],
            **{f'recall@{k}': [] for k in k_values},
            **{f'ndcg@{k}': [] for k in k_values}
        }

        for query_id, query_emb in tqdm(query_embeddings.items(), desc="Evaluating queries"):
            if query_id not in qrels or len(qrels[query_id]) == 0:
                continue

            # Compute similarities
            query_emb = query_emb.reshape(1, -1)
            similarities = np.dot(query_emb, doc_matrix.T).flatten()

            # Get top-k documents
            top_indices = np.argsort(similarities)[::-1]
            top_doc_ids = [doc_ids[idx] for idx in top_indices]

            relevant_docs = set(qrels[query_id])

            # MRR (Mean Reciprocal Rank)
            for rank, doc_id in enumerate(top_doc_ids, 1):
                if doc_id in relevant_docs:
                    metrics['mrr'].append(1.0 / rank)
                    break
            else:
                metrics['mrr'].append(0.0)

            # Recall@K
            for k in k_values:
                retrieved_k = set(top_doc_ids[:k])
                recall = len(retrieved_k & relevant_docs) / len(relevant_docs)
                metrics[f'recall@{k}'].append(recall)

            # NDCG@K
            for k in k_values:
                dcg = 0.0
                idcg = sum([1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_docs)))])

                for rank, doc_id in enumerate(top_doc_ids[:k], 1):
                    if doc_id in relevant_docs:
                        dcg += 1.0 / np.log2(rank + 1)

                ndcg = dcg / idcg if idcg > 0 else 0.0
                metrics[f'ndcg@{k}'].append(ndcg)

        # Average metrics
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}

        return avg_metrics

    def evaluate(self, queries_file, qrels_file, corpus_file, max_docs=None):
        """
        Full evaluation pipeline
        """
        # Load qrels
        qrels = {}
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    query_id, _, doc_id, relevance = parts[0], parts[1], parts[2], parts[3]
                    if int(relevance) > 0:
                        if query_id not in qrels:
                            qrels[query_id] = []
                        qrels[query_id].append(doc_id)

        # Encode queries and documents
        query_embeddings = self.encode_queries(queries_file)
        doc_embeddings = self.encode_corpus(corpus_file, max_docs)

        # Compute metrics
        metrics = self.compute_retrieval_metrics(query_embeddings, doc_embeddings, qrels)

        # Print results
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80)
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        print("=" * 80)

        return metrics


def main():
    """Main evaluation script"""

    # Configuration
    config = {
        'model_name': 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        'embedding_dim': 768,
        'pooling': 'cls',
        'max_length': 512,
    }

    # Paths
    model_path = './checkpoints/best_model.pt'  # Update with your checkpoint path
    queries_file = r'C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\queries\test_queries.jsonl'
    qrels_file = r'C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\PAR\qrels_test.tsv'
    corpus_file = r'C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\corpus.jsonl'

    # Create evaluator
    evaluator = PAREvaluator(model_path, config)

    # Evaluate (set max_docs for faster testing, remove for full evaluation)
    metrics = evaluator.evaluate(queries_file, qrels_file, corpus_file, max_docs=None)

    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/evaluation_results.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nResults saved to results/evaluation_results.json")


if __name__ == "__main__":
    main()