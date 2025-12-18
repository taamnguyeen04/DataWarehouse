import json
import os
import numpy as np
import torch
import rerun as rr
from sklearn.manifold import TSNE
import random
from transformers import AutoTokenizer

from config import Config
from model import BiEncoder

# --- Configuration ---
RESULTS_STAGE_1 = r"C:\Users\tam\Documents\GitHub\DataWarehouse\PAR\results\test_results_3.json"
# We only need Stage 1 results to find candidates (positives + hard negatives)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_qrels(path):
    qrels = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                if parts[2] == 'score': continue
                qid, docid, rel = parts[0], parts[1], int(parts[2])
                if rel > 0:
                    if qid not in qrels: qrels[qid] = set()
                    qrels[qid].add(docid)
    return qrels

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

def visualize_3d_embeddings(res1, qrels, num_samples=1, target_qid=None, min_visual_dist=5.0):
    print("\nGenerating 3D Embedding Visualizations...")
    
    selected_qids = []
    if target_qid:
        if target_qid in res1 and target_qid in qrels:
            print(f"Targeting specific query: {target_qid}")
            selected_qids = [target_qid]
        else:
            print(f"Warning: Target query {target_qid} not found in results or qrels.")
            return
    else:
        # Select random queries
        candidate_qids = []
        for qid, docs in res1.items():
            if qid in qrels:
                top_docs = [d for d, _ in sorted(docs.items(), key=lambda x: x[1], reverse=True)[:50]]
                if any(d in qrels[qid] for d in top_docs):
                    candidate_qids.append(qid)
        
        if not candidate_qids:
            print("No suitable queries found.")
            return

        selected_qids = random.sample(candidate_qids, min(num_samples, len(candidate_qids)))
    
    # Collect IDs
    needed_doc_ids = set()
    for qid in selected_qids:
        needed_doc_ids.update(qrels[qid])
        top_docs = [d for d, _ in sorted(res1[qid].items(), key=lambda x: x[1], reverse=True)[:50]]
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
    
    # Initialize Rerun
    rr.init("Embedding_Space_3D", spawn=True)
    
    for qid in selected_qids:
        print(f"Visualizing Query {qid}...")
        
        texts = []
        labels = [] # Type labels
        doc_ids = [] # Actual IDs for hover
        
        # 1. Query
        texts.append(queries[qid])
        labels.append('Query')
        doc_ids.append(f"Query: {qid}")
        
        # 2. Positives
        pos_docs = [d for d in qrels[qid] if d in corpus]
        for d in pos_docs:
            texts.append(corpus[d])
            labels.append('Positive')
            doc_ids.append(f"Pos: {d}")
            
        # 3. Hard Negatives
        top_docs = [d for d, _ in sorted(res1[qid].items(), key=lambda x: x[1], reverse=True)[:200]]
        neg_docs = [d for d in top_docs if d not in qrels[qid] and d in corpus][:150] 
        for d in neg_docs:
            texts.append(corpus[d])
            labels.append('Hard Negative')
            doc_ids.append(f"Neg: {d}")
            
        # Encode
        embeddings = []
        with torch.no_grad():
            for i, text in enumerate(texts):
                inputs = tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt').to(device)
                if labels[i] == 'Query':
                    emb = model.encode_query(inputs['input_ids'], inputs['attention_mask'])
                else:
                    emb = model.encode_doc(inputs['input_ids'], inputs['attention_mask'])
                embeddings.append(emb.cpu().numpy()[0])
        
        embeddings = np.array(embeddings)
        
        # t-SNE to 3D
        print("Running t-SNE (3D)...")
        tsne = TSNE(n_components=3, perplexity=min(5, len(embeddings)-1), random_state=42)
        emb_3d = tsne.fit_transform(embeddings)
        
        # --- Filter Close Negatives ---
        query_pos = emb_3d[0]
        dists = np.linalg.norm(emb_3d - query_pos, axis=1)
        
        keep_mask = []
        for i, label in enumerate(labels):
            if label == 'Hard Negative':
                if dists[i] > min_visual_dist:
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)
            else:
                keep_mask.append(True) # Keep Query and Positives
        
        keep_mask = np.array(keep_mask)
        emb_3d = emb_3d[keep_mask]
        labels = [l for i, l in enumerate(labels) if keep_mask[i]]
        doc_ids = [d for i, d in enumerate(doc_ids) if keep_mask[i]]
        
        print(f"Filtered {len(keep_mask) - sum(keep_mask)} close negatives.")

        # Colors
        colors = []
        for l in labels:
            if l == 'Query': colors.append([255, 0, 0])
            elif l == 'Positive': colors.append([0, 255, 0])
            else: colors.append([128, 128, 128])
        colors = np.array(colors)
        
        # Log to Rerun
        rr.log(
            f"query_{qid}",
            rr.Points3D(emb_3d, colors=colors, radii=1.5, labels=doc_ids)
        )
        print(f"Logged query {qid} to Rerun.")

def main():
    print("Loading Stage 1 Results...")
    res1 = load_json(RESULTS_STAGE_1)
    
    print("Loading Qrels...")
    qrels = load_qrels(Config.TEST_QRELS)
    
    visualize_3d_embeddings(res1, qrels, num_samples=1, target_qid="2811306-1")

if __name__ == "__main__":
    main()
