import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# Configuration
RESULTS_PATH = "./PAR/results/test_results_2.json"
QRELS_PATH = "C:/Users/tam/Desktop/Data/Data Warehouse/ReCDS_benchmark/PAR/qrels_test.tsv"
OUTPUT_IMAGE = "rank_distribution.png"

def load_qrels(qrels_path):
    """Load ground truth qrels from TSV file."""
    qrels = {}
    print(f"Loading qrels from {qrels_path}...")
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    qid, docid, rel = parts[0], parts[1], int(parts[2])
                    if rel > 0:
                        if qid not in qrels:
                            qrels[qid] = set()
                        qrels[qid].add(docid)
                except ValueError:
                    continue
    return qrels

def load_results(results_path):
    """Load retrieval results from JSON file."""
    print(f"Loading results from {results_path}...")
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_first_relevant_ranks(results, qrels):
    """Calculate the rank of the first relevant document for each query."""
    ranks = []
    not_found_count = 0
    
    print("Calculating ranks...")
    for qid, doc_scores in tqdm(results.items()):
        if qid not in qrels:
            continue
            
        relevant_docs = qrels[qid]
        
        # Sort documents by score descending (just to be safe)
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        found_rank = -1
        for rank, (docid, score) in enumerate(sorted_docs, 1):
            if docid in relevant_docs:
                found_rank = rank
                break
        
        if found_rank != -1:
            ranks.append(found_rank)
        else:
            not_found_count += 1
            
    print(f"Found relevant documents for {len(ranks)} queries.")
    print(f"No relevant documents found in top-k for {not_found_count} queries.")
    return ranks

def plot_distribution(ranks, output_file):
    """Plot the distribution of ranks."""
    if not ranks:
        print("No ranks to plot.")
        return

    plt.figure(figsize=(12, 6))
    
    # Histogram for ranks 1-100
    plt.hist([r for r in ranks if r <= 100], bins=100, range=(1, 101), color='skyblue', edgecolor='black', alpha=0.7)
    
    plt.title('Distribution of First Relevant Document Rank (Top 100)')
    plt.xlabel('Rank')
    plt.ylabel('Number of Queries')
    plt.grid(axis='y', alpha=0.5)
    
    # Add some statistics
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)
    mrr = np.mean([1/r for r in ranks])
    
    stats_text = (
        f"Total Queries: {len(ranks)}\n"
        f"Mean Rank: {mean_rank:.1f}\n"
        f"Median Rank: {median_rank:.1f}\n"
        f"MRR (Calculated): {mrr:.4f}"
    )
    
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")
    plt.close()

    # Additional plot: Cumulative Distribution Function (CDF)
    plt.figure(figsize=(12, 6))
    sorted_ranks = np.sort(ranks)
    yvals = np.arange(len(sorted_ranks)) / float(len(sorted_ranks) - 1)
    
    plt.plot(sorted_ranks, yvals, marker='.', linestyle='none', markersize=2)
    plt.title('CDF of First Relevant Document Rank')
    plt.xlabel('Rank (Log Scale)')
    plt.ylabel('Cumulative Proportion of Queries')
    plt.xscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.savefig(output_file.replace('.png', '_cdf.png'))
    print(f"CDF Chart saved to {output_file.replace('.png', '_cdf.png')}")
    plt.close()

def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"Error: Results file not found at {RESULTS_PATH}")
        return
        
    qrels = load_qrels(QRELS_PATH)
    results = load_results(RESULTS_PATH)
    
    ranks = calculate_first_relevant_ranks(results, qrels)
    
    plot_distribution(ranks, OUTPUT_IMAGE)

if __name__ == "__main__":
    main()
