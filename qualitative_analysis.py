import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from sklearn.manifold import TSNE
import random

# Try importing UMAP, handle if missing
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️ UMAP not found. Skipping UMAP visualization.")

from config import Config
from model import BiEncoder
from transformers import AutoTokenizer

# --- Configuration ---
RESULTS_STAGE_1 = r"C:\Users\tam\Documents\GitHub\DataWarehouse\PAR\results\test_results_3.json"
RESULTS_STAGE_2 = r"C:\Users\tam\Documents\GitHub\DataWarehouse\partest_results_reranked1.json"
OUTPUT_DIR = "./analysis_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_qrels(path):
    qrels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                # Skip header if present
                if parts[2] == 'score': continue
                
                qid, docid, rel = parts[0], parts[1], int(parts[2])
                if rel > 0:
                    if qid not in qrels: qrels[qid] = set()
                    qrels[qid].add(docid)
    return qrels

def get_rank(doc_id, ranked_list):
    """Return 1-based rank of doc_id in ranked_list (list of (doc_id, score)). Returns 1001 if not found."""
    for rank, (d, _) in enumerate(ranked_list, 1):
        if d == doc_id:
            return rank
    return 1001 # Not in top K

def analyze_rank_shift(res1, res2, qrels):
    print("Analyzing Rank Shift...")
    data = []
    
    # Common queries
    common_qids = set(res1.keys()) & set(res2.keys()) & set(qrels.keys())
    
    for qid in common_qids:
        # Sort results just in case
        list1 = sorted(res1[qid].items(), key=lambda x: x[1], reverse=True)
        list2 = sorted(res2[qid].items(), key=lambda x: x[1], reverse=True)
        
        relevant_docs = qrels[qid]
        
        for doc_id in relevant_docs:
            rank1 = get_rank(doc_id, list1)
            rank2 = get_rank(doc_id, list2)
            
            # Only consider if at least one model found it in top 1000
            if rank1 <= 1000 or rank2 <= 1000:
                data.append({
                    'QueryID': qid,
                    'DocID': doc_id,
                    'Rank_Stage1': rank1 if rank1 <= 1000 else 1000, # Cap for visualization
                    'Rank_Stage2': rank2 if rank2 <= 1000 else 1000
                })
                
    df = pd.DataFrame(data)
    
    # --- Plot 1: Scatter Plot (Rank 1 vs Rank 2) ---
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df, x='Rank_Stage1', y='Rank_Stage2', alpha=0.6, edgecolor=None)
    
    # Diagonal line y=x
    max_val = max(df['Rank_Stage1'].max(), df['Rank_Stage2'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', label='No Change (y=x)')
    
    plt.title('Rank Shift Analysis: Bi-Encoder vs Cross-Encoder')
    plt.xlabel('Bi-Encoder Rank (Stage 1)')
    plt.ylabel('Cross-Encoder Rank (Stage 2)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate regions
    plt.text(max_val*0.7, max_val*0.2, 'Improved by Reranker\n(Rank 2 < Rank 1)', 
             fontsize=1000, color='green', ha='center', bbox=dict(facecolor='white', alpha=0.8))
    plt.text(max_val*0.2, max_val*0.7, 'Worsened by Reranker\n(Rank 2 > Rank 1)', 
             fontsize=1000, color='red', ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'rank_shift_scatter.png'))
    print(f"Saved rank_shift_scatter.png to {OUTPUT_DIR}")
    
    # --- Plot 2: Histogram of Ranks ---
    plt.figure(figsize=(12, 6))
    plt.hist(df['Rank_Stage1'], bins=50, alpha=0.5, label='Bi-Encoder', range=(1, 100))
    plt.hist(df['Rank_Stage2'], bins=50, alpha=0.5, label='Cross-Encoder', range=(1, 100))
    plt.title('Distribution of Ranks for Relevant Documents (Top 100)')
    plt.xlabel('Rank')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'rank_histogram.png'))
    print(f"Saved rank_histogram.png to {OUTPUT_DIR}")

def load_corpus_subset(needed_ids):
    print(f"Loading corpus subset ({len(needed_ids)} docs)...")
    corpus = {}
    with open(Config.CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            did = str(doc['_id'])
            if did in needed_ids:
                corpus[did] = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
                if len(corpus) == len(needed_ids):
                    break
    return corpus

def load_queries_subset(needed_ids):
    print(f"Loading queries subset ({len(needed_ids)} queries)...")
    queries = {}
    with open(Config.TEST_QUERIES, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            qid = str(q['_id'])
            if qid in needed_ids:
                queries[qid] = q['text']
                if len(queries) == len(needed_ids):
                    break
    return queries

def visualize_embeddings(res1, qrels, num_samples=5):
    print("\nGenerating Embedding Visualizations...")
    
    # Select random queries that have relevant docs in top results
    candidate_qids = []
    for qid, docs in res1.items():
        if qid in qrels:
            # Check if at least one relevant doc is in top 50
            top_docs = [d for d, _ in sorted(docs.items(), key=lambda x: x[1], reverse=True)[:50]]
            if any(d in qrels[qid] for d in top_docs):
                candidate_qids.append(qid)
    
    if not candidate_qids:
        print("No suitable queries found for visualization.")
        return

    selected_qids = random.sample(candidate_qids, min(num_samples, len(candidate_qids)))
    
    # Collect all needed IDs
    needed_doc_ids = set()
    for qid in selected_qids:
        # Get relevant docs
        needed_doc_ids.update(qrels[qid])
        # Get some hard negatives (top retrieved but not relevant)
        top_docs = [d for d, _ in sorted(res1[qid].items(), key=lambda x: x[1], reverse=True)[:20]]
        for d in top_docs:
            if d not in qrels[qid]:
                needed_doc_ids.add(d)
                
    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model to {device}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = BiEncoder(Config.MODEL_NAME, Config.EMBEDDING_DIM, Config.POOLING).to(device)
    model.eval()
    
    # Load Data
    corpus = load_corpus_subset(needed_doc_ids)
    queries = load_queries_subset(set(selected_qids))
    
    # Process each query
    for qid in selected_qids:
        print(f"Visualizing Query {qid}...")
        
        # Prepare data points
        texts = []
        labels = [] # 0: Query, 1: Positive, 2: Hard Negative
        
        # 1. Query
        texts.append(queries[qid])
        labels.append('Query')
        
        # 2. Positives
        pos_docs = [d for d in qrels[qid] if d in corpus]
        for d in pos_docs:
            texts.append(corpus[d])
            labels.append('Positive')
            
        # 3. Hard Negatives (Top retrieved non-relevant)
        top_docs = [d for d, _ in sorted(res1[qid].items(), key=lambda x: x[1], reverse=True)[:50]]
        neg_docs = [d for d in top_docs if d not in qrels[qid] and d in corpus][:10] # Take top 10 hard negs
        for d in neg_docs:
            texts.append(corpus[d])
            labels.append('Hard Negative')
            
        # Encode
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt').to(device)
                # Use encode_query for query and encode_doc for docs? 
                # Bi-Encoder uses shared weights usually, but let's stick to model methods
                if labels[len(embeddings)] == 'Query':
                    emb = model.encode_query(inputs['input_ids'], inputs['attention_mask'])
                else:
                    emb = model.encode_doc(inputs['input_ids'], inputs['attention_mask'])
                embeddings.append(emb.cpu().numpy()[0])
        
        embeddings = np.array(embeddings)
        
        # --- t-SNE ---
        tsne = TSNE(n_components=2, perplexity=min(5, len(embeddings)-1), random_state=42)
        emb_tsne = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=emb_tsne[:,0], y=emb_tsne[:,1], hue=labels, style=labels, s=100, palette={'Query': 'red', 'Positive': 'green', 'Hard Negative': 'gray'})
        plt.title(f't-SNE Visualization for Query {qid}')
        plt.savefig(os.path.join(OUTPUT_DIR, f'tsne_query_{qid}.png'))
        plt.close()
        
        # --- UMAP ---
        if HAS_UMAP:
            reducer = umap.UMAP(n_neighbors=min(5, len(embeddings)-1), min_dist=0.3, metric='cosine', random_state=42)
            emb_umap = reducer.fit_transform(embeddings)
            
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=emb_umap[:,0], y=emb_umap[:,1], hue=labels, style=labels, s=100, palette={'Query': 'red', 'Positive': 'green', 'Hard Negative': 'gray'})
            plt.title(f'UMAP Visualization for Query {qid}')
            plt.savefig(os.path.join(OUTPUT_DIR, f'umap_query_{qid}.png'))
            plt.close()

def main():
    print("Loading Stage 1 Results...")
    res1 = load_json(RESULTS_STAGE_1)
    
    print("Loading Stage 2 Results...")
    res2 = load_json(RESULTS_STAGE_2)
    
    print("Loading Qrels...")
    qrels = load_qrels(Config.TEST_QRELS)
    
    # 1. Rank Shift Analysis
    analyze_rank_shift(res1, res2, qrels)
    
    # 2. Embedding Visualization
    visualize_embeddings(res1, qrels, num_samples=3) # Do 3 samples to save time
    
    print(f"\n✅ Analysis Complete! Check results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
