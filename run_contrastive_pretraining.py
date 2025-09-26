"""
Complete pipeline for contrastive pretraining on 11M papers
Usage examples and main execution script
"""

import os
import sys
import torch
import argparse
from train_contrastive import main as train_main
from extract_embeddings import EmbeddingExtractor
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if all requirements are met"""
    issues = []

    # Check CUDA
    if not torch.cuda.is_available():
        issues.append("CUDA not available - training will be very slow on CPU")
    else:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory < 8:
            issues.append(f"Low GPU memory ({gpu_memory:.1f}GB) - consider reducing batch size")

    # Check corpus file
    if not os.path.exists(Config.CORPUS_FILE):
        issues.append(f"Corpus file not found: {Config.CORPUS_FILE}")

    # Check disk space (rough estimate: embeddings ~11M * 768 * 4 bytes = ~33GB)
    if issues:
        logger.warning("Issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    return len(issues) == 0


def run_training():
    """Run contrastive pretraining"""
    logger.info("Starting contrastive pretraining...")

    # Check system
    if not check_requirements():
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Run training
    train_main()
    logger.info("Training completed!")


def run_extraction(model_path, output_file="paper_embeddings.pkl", max_papers=None):
    """Extract embeddings from trained model"""
    logger.info("Extracting embeddings...")

    extractor = EmbeddingExtractor(model_path)

    # Extract embeddings
    embedding_data = extractor.extract_paper_embeddings(
        corpus_file=Config.CORPUS_FILE,
        output_file=output_file,
        max_papers=max_papers
    )

    # Evaluate performance
    metrics = extractor.evaluate_retrieval(
        embedding_data['title_embeddings'],
        embedding_data['abstract_embeddings'],
        num_queries=1000
    )

    logger.info("Extraction completed!")
    return embedding_data, metrics


def print_usage_examples():
    """Print usage examples"""
    print("""
=== Contrastive Pretraining Usage Examples ===

1. Full 11M paper pretraining:
   python run_contrastive_pretraining.py --mode train

2. Quick test with 100k papers:
   python run_contrastive_pretraining.py --mode train --max_samples 100000

3. Extract embeddings from trained model:
   python run_contrastive_pretraining.py --mode extract --model_path ./contrastive_checkpoints/final_model.pt

4. Custom configuration:
   python train_contrastive.py

5. Extract embeddings with evaluation:
   python extract_embeddings.py --model_path ./contrastive_checkpoints/final_model.pt --evaluate

=== Expected Results ===

After successful pretraining, you should see:
- Title->Abstract retrieval accuracy > 0.7
- Mean cosine similarity between paired (title, abstract) > 0.6
- Embedding files ready for downstream tasks:
  - Retrieval: Use embeddings for similarity search
  - Clustering: Apply k-means/hierarchical clustering
  - Label propagation: Use similarities for semi-supervised learning

=== Hardware Recommendations ===

Minimum:
- GPU: RTX 3060 (12GB VRAM)
- RAM: 32GB
- Storage: 100GB free space
- Batch size: 64-128

Recommended:
- GPU: RTX 4090 (24GB VRAM) or A6000
- RAM: 64GB
- Storage: 200GB free space
- Batch size: 512-1024

Optimal:
- GPU: A100 (40GB VRAM)
- RAM: 128GB
- Storage: 500GB free space
- Batch size: 1024+

=== Training Progress ===

Expected training time (11M papers, 5 epochs):
- RTX 4090: ~2-3 days
- A100: ~1-2 days
- RTX 3080: ~4-5 days

Monitor these metrics during training:
- Loss should decrease from ~7-8 to ~2-3
- Title->Abstract accuracy should increase from ~0.05 to ~0.7+
- Learning rate should decay smoothly

=== Files Created ===

Training outputs:
- ./contrastive_checkpoints/final_model.pt (trained model)
- ./contrastive_checkpoints/epoch_X.pt (epoch checkpoints)
- ./contrastive_checkpoints/config.txt (configuration)

Embedding outputs:
- paper_embeddings.pkl (all paper embeddings + metadata)
- paper_embeddings_similarities.pkl (precomputed similarities)
- paper_embeddings_metrics.json (evaluation metrics)

=== Next Steps ===

After pretraining, use embeddings for:
1. Retrieval: Find similar papers using cosine similarity
2. Clustering: Group papers by topic/domain
3. Label propagation: Transfer labels from labeled to unlabeled papers
4. Semi-supervised learning: Train classifiers with pseudo-labels

===============================================
    """)


def main():
    parser = argparse.ArgumentParser(description="Contrastive pretraining pipeline")
    parser.add_argument("--mode", choices=['train', 'extract', 'examples'],
                       default='examples', help="Mode to run")
    parser.add_argument("--model_path", help="Path to trained model (for extraction)")
    parser.add_argument("--output_file", default="paper_embeddings.pkl",
                       help="Output embeddings file")
    parser.add_argument("--max_samples", type=int, help="Maximum samples for training/extraction")

    args = parser.parse_args()

    if args.mode == 'examples':
        print_usage_examples()

    elif args.mode == 'train':
        if args.max_samples:
            logger.info(f"Training with {args.max_samples} samples")
            # Note: max_samples can be passed to training function if needed

        run_training()

    elif args.mode == 'extract':
        if not args.model_path:
            logger.error("--model_path required for extraction mode")
            sys.exit(1)

        run_extraction(args.model_path, args.output_file, args.max_samples)


if __name__ == "__main__":
    main()