"""
Extract domain-specific embeddings from trained contrastive model
For use in downstream tasks: retrieval, clustering, label propagation
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from contrastive_pretrain import ContrastiveModel, StreamingContrastiveDataset
from config import Config
import pickle
import os
from tqdm import tqdm
import logging
import argparse


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extract embeddings from trained contrastive model"""

    def __init__(self, model_path, projection_dim=768, temperature=0.07):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model
        self.model = ContrastiveModel(projection_dim=projection_dim, temperature=temperature)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.max_length = 512

        logger.info(f"Loaded model from: {model_path}")
        logger.info(f"Model device: {self.device}")

    def encode_text(self, texts, batch_size=64, show_progress=True):
        """
        Encode texts to embeddings

        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            numpy array of embeddings [num_texts, embedding_dim]
        """
        embeddings = []

        # Process in batches
        with torch.no_grad():
            iterator = range(0, len(texts), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc="Encoding texts")

            for i in iterator:
                batch_texts = texts[i:i + batch_size]

                # Tokenize
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                # Move to device
                input_ids = encoded['input_ids'].to(self.device)
                attention_mask = encoded['attention_mask'].to(self.device)

                # Get embeddings
                batch_embeddings = self.model(input_ids, attention_mask)
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def extract_paper_embeddings(self, corpus_file, output_file, max_papers=None):
        """
        Extract embeddings for all papers in corpus

        Args:
            corpus_file: Path to corpus JSONL file
            output_file: Path to save embeddings
            max_papers: Maximum number of papers to process
        """
        logger.info(f"Extracting embeddings from: {corpus_file}")

        papers = []
        titles = []
        abstracts = []
        pmids = []

        # Load papers
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_papers and i >= max_papers:
                    break

                try:
                    paper = json.loads(line.strip())
                    title = paper.get('title', '').strip()
                    abstract = paper.get('text', '').strip()
                    pmid = paper.get('_id', str(i))

                    if title and abstract and len(title) > 10 and len(abstract) > 50:
                        papers.append(paper)
                        titles.append(title)
                        abstracts.append(abstract)
                        pmids.append(pmid)

                    if len(papers) % 100000 == 0:
                        logger.info(f"Loaded {len(papers)} papers...")

                except (json.JSONDecodeError, KeyError):
                    continue

        logger.info(f"Processing {len(papers)} papers")

        # Extract embeddings
        title_embeddings = self.encode_text(titles, show_progress=True)
        abstract_embeddings = self.encode_text(abstracts, show_progress=True)

        # Average title and abstract embeddings for final representation
        paper_embeddings = (title_embeddings + abstract_embeddings) / 2
        paper_embeddings = F.normalize(torch.from_numpy(paper_embeddings), p=2, dim=1).numpy()

        # Save embeddings and metadata
        embedding_data = {
            'embeddings': paper_embeddings,
            'pmids': pmids,
            'titles': titles,
            'abstracts': abstracts,
            'title_embeddings': title_embeddings,
            'abstract_embeddings': abstract_embeddings,
            'model_name': Config.MODEL_NAME,
            'embedding_dim': paper_embeddings.shape[1]
        }

        with open(output_file, 'wb') as f:
            pickle.dump(embedding_data, f)

        logger.info(f"Saved {len(papers)} embeddings to: {output_file}")
        logger.info(f"Embedding shape: {paper_embeddings.shape}")

        return embedding_data

    def compute_similarity_matrix(self, embeddings, top_k=100):
        """
        Compute similarity matrix for embeddings

        Args:
            embeddings: numpy array of embeddings
            top_k: Number of top similar papers to keep for each paper

        Returns:
            Dictionary with similarity indices and scores
        """
        logger.info("Computing similarity matrix...")

        embeddings_tensor = torch.from_numpy(embeddings).to(self.device)

        # Compute cosine similarities in batches to avoid memory issues
        batch_size = 1000
        similarities = {}

        with torch.no_grad():
            for i in tqdm(range(0, len(embeddings), batch_size), desc="Computing similarities"):
                batch_embeddings = embeddings_tensor[i:i + batch_size]

                # Compute similarity with all embeddings
                sims = torch.matmul(batch_embeddings, embeddings_tensor.T)

                # Get top-k for each paper in batch
                top_scores, top_indices = torch.topk(sims, k=top_k + 1, dim=1)  # +1 to exclude self

                for j, (scores, indices) in enumerate(zip(top_scores, top_indices)):
                    paper_idx = i + j
                    # Remove self-similarity (should be at position 0)
                    mask = indices != paper_idx
                    similarities[paper_idx] = {
                        'indices': indices[mask][:top_k].cpu().numpy(),
                        'scores': scores[mask][:top_k].cpu().numpy()
                    }

        return similarities

    def evaluate_retrieval(self, title_embeddings, abstract_embeddings, num_queries=1000):
        """
        Evaluate retrieval performance: title -> abstract matching

        Args:
            title_embeddings: Title embeddings
            abstract_embeddings: Abstract embeddings
            num_queries: Number of queries to evaluate

        Returns:
            Dictionary with retrieval metrics
        """
        logger.info("Evaluating retrieval performance...")

        # Random sample for evaluation
        indices = np.random.choice(len(title_embeddings), size=num_queries, replace=False)

        query_embeddings = title_embeddings[indices]

        # Compute similarities
        similarities = np.dot(query_embeddings, abstract_embeddings.T)

        # Compute metrics
        top1_correct = 0
        top5_correct = 0
        top10_correct = 0
        mrr_scores = []

        for i, query_idx in enumerate(indices):
            # Get ranked results
            scores = similarities[i]
            ranked_indices = np.argsort(scores)[::-1]

            # Find position of correct abstract
            correct_pos = np.where(ranked_indices == query_idx)[0][0] + 1

            # Update metrics
            if correct_pos <= 1:
                top1_correct += 1
            if correct_pos <= 5:
                top5_correct += 1
            if correct_pos <= 10:
                top10_correct += 1

            mrr_scores.append(1.0 / correct_pos)

        metrics = {
            'top1_accuracy': top1_correct / num_queries,
            'top5_accuracy': top5_correct / num_queries,
            'top10_accuracy': top10_correct / num_queries,
            'mrr': np.mean(mrr_scores),
            'num_queries': num_queries
        }

        logger.info("Retrieval Results:")
        logger.info(f"Top-1 Accuracy: {metrics['top1_accuracy']:.4f}")
        logger.info(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4f}")
        logger.info(f"Top-10 Accuracy: {metrics['top10_accuracy']:.4f}")
        logger.info(f"MRR: {metrics['mrr']:.4f}")

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from trained contrastive model")
    parser.add_argument("--model_path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--corpus_file", help="Path to corpus file (default: from config)")
    parser.add_argument("--output_file", default="paper_embeddings.pkl", help="Output embeddings file")
    parser.add_argument("--max_papers", type=int, help="Maximum number of papers to process")
    parser.add_argument("--compute_similarities", action="store_true", help="Compute similarity matrix")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate retrieval performance")

    args = parser.parse_args()

    # Initialize extractor
    corpus_file = args.corpus_file or Config.CORPUS_FILE

    extractor = EmbeddingExtractor(args.model_path)

    # Extract embeddings
    embedding_data = extractor.extract_paper_embeddings(
        corpus_file=corpus_file,
        output_file=args.output_file,
        max_papers=args.max_papers
    )

    # Compute similarities if requested
    if args.compute_similarities:
        similarities = extractor.compute_similarity_matrix(embedding_data['embeddings'])

        similarity_file = args.output_file.replace('.pkl', '_similarities.pkl')
        with open(similarity_file, 'wb') as f:
            pickle.dump(similarities, f)
        logger.info(f"Saved similarities to: {similarity_file}")

    # Evaluate retrieval if requested
    if args.evaluate:
        metrics = extractor.evaluate_retrieval(
            embedding_data['title_embeddings'],
            embedding_data['abstract_embeddings']
        )

        metrics_file = args.output_file.replace('.pkl', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to: {metrics_file}")


if __name__ == "__main__":
    main()