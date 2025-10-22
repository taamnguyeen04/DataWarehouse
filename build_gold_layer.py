"""
Gold Layer Pipeline for PAR Dataset
Builds BM25 hard negatives for training bi-encoder model

Steps:
1. Build BM25 index using Pyserini
2. Retrieve top-k BM25 results for each query
3. Filter positives to create hard negatives
4. Create training pairs (query, pos_doc, neg_doc)
"""

# ⚙️ PHẢI CẤU HÌNH JVM TRƯỚC KHI IMPORT PYSERINI
import jnius_config
jnius_config.add_options(
    '-Xms1g',
    '-Xmx4g',
    '-Dorg.apache.lucene.search.BooleanQuery.maxClauseCount=65536'
)

import json
import os
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
import subprocess
import sys

# Configuration
class GoldConfig:
    # Input paths (Silver layer)
    CORPUS_FILE = r"C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\PAR\corpus.jsonl"
    TRAIN_QUERIES = r"C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\queries\train_queries.jsonl"
    QRELS_TRAIN = r"C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\PAR\qrels_train.tsv"

    # Output paths (Gold layer)
    GOLD_DIR = r"C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\PAR\gold"
    BM25_INDEX_DIR = r"C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\PAR\gold\bm25_index"
    BM25_CANDIDATES = r"C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\PAR\gold\bm25_candidates_topk.json"
    BM25_HARD_NEGS = r"C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\PAR\gold\bm25_hard_negs.json"
    PAIRS_TRAIN = r"C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\PAR\gold\pairs_train.jsonl"

    # Parameters
    TOP_K = 100  # Number of BM25 candidates to retrieve
    NUM_HARD_NEGS = 5  # Number of hard negatives per query


def load_corpus(corpus_file: str) -> Dict[str, Dict]:
    """Load corpus from JSONL file"""
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = doc
    print(f"Loaded {len(corpus)} documents from corpus")
    return corpus


def load_queries(queries_file: str) -> Dict[str, str]:
    """Load queries from JSONL file"""
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line)
            queries[query['_id']] = query['text']
    print(f"Loaded {len(queries)} queries")
    return queries


def load_qrels(qrels_file: str) -> Dict[str, Set[str]]:
    """Load qrels (query -> positive docs mapping)"""
    qrels = defaultdict(set)
    with open(qrels_file, 'r', encoding='utf-8') as f:
        next(f)  # Skip header line
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id, doc_id, rel = parts[0], parts[1], int(parts[2])
                if rel > 0:  # relevance >= 1
                    qrels[query_id].add(doc_id)
    print(f"Loaded qrels for {len(qrels)} queries")
    return qrels


def build_bm25_index(corpus_file: str, index_dir: str):
    """Build BM25 index using Pyserini"""
    print("\n=== Step 1: Building BM25 Index ===")

    # Create index directory
    os.makedirs(index_dir, exist_ok=True)

    # Prepare corpus directory for Pyserini (needs directory structure)
    corpus_dir = os.path.join(os.path.dirname(index_dir), "corpus_for_index")
    os.makedirs(corpus_dir, exist_ok=True)

    # ✅ Convert corpus to Pyserini format: {"id": ..., "contents": ...}
    print("Converting corpus to Pyserini format...")
    corpus_dest = os.path.join(corpus_dir, "corpus.jsonl")

    converted_count = 0
    with open(corpus_file, 'r', encoding='utf-8') as fin, \
         open(corpus_dest, 'w', encoding='utf-8') as fout:
        for line in fin:
            doc = json.loads(line)

            # Pyserini format: combine title + text into contents
            pyserini_doc = {
                "id": doc["_id"],
                "contents": f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            }

            fout.write(json.dumps(pyserini_doc, ensure_ascii=False) + '\n')
            converted_count += 1

            if converted_count % 100000 == 0:
                print(f"  Converted {converted_count} documents...")

    print(f"✅ Converted {converted_count} documents to Pyserini format")

    # Build index using Pyserini
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", corpus_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "8",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]

    print(f"\nBuilding index...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("STDERR:", result.stderr)
        raise RuntimeError(f"Failed to build BM25 index: {result.stderr}")

    print("✅ BM25 index built successfully!")


def retrieve_bm25_candidates(queries: Dict[str, str], index_dir: str, top_k: int) -> Dict[str, List[str]]:
    """Retrieve top-k BM25 candidates for each query"""
    print(f"\n=== Step 2: Retrieving top-{top_k} BM25 candidates ===")

    # ✅ JVM đã được cấu hình ở đầu file với maxClauseCount=65536
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(index_dir)
    searcher.set_bm25(k1=0.9, b=0.4)  # Standard BM25 parameters

    candidates = {}
    queries_with_fallback = 0

    for query_id, query_text in queries.items():
        original_query = query_text
        success = False

        # Chiến lược: thử giảm dần độ dài query cho đến khi thành công
        max_words_attempts = [200, 150, 100, 50, 20]  # Thử từ dài đến ngắn

        for max_words in max_words_attempts:
            try:
                words = original_query.split()
                if len(words) > max_words:
                    query_text = ' '.join(words[:max_words])
                else:
                    query_text = original_query

                hits = searcher.search(query_text, k=top_k)
                candidates[query_id] = [hit.docid for hit in hits]
                success = True

                if len(words) > max_words:
                    queries_with_fallback += 1

                break  # Thành công, thoát vòng lặp

            except Exception as e:
                error_msg = str(e)
                if "TooManyClauses" in error_msg or "maxClauseCount" in error_msg:
                    # Thử với query ngắn hơn ở lần lặp tiếp theo
                    continue
                else:
                    # Lỗi khác, dừng ngay
                    print(f"  ❌ Error processing query {query_id}: {error_msg}")
                    break

        # Nếu tất cả các lần thử đều thất bại, dùng random sampling
        if not success:
            print(f"  ⚠️ Query {query_id} không search được, sẽ dùng random negatives")
            candidates[query_id] = []  # Sẽ xử lý ở bước create_hard_negatives

        if len(candidates) % 100 == 0:
            print(f"Processed {len(candidates)} queries...")

    print(f"Retrieved candidates for {len(candidates)} queries")
    if queries_with_fallback > 0:
        print(f"⚠️ {queries_with_fallback} queries used truncated search")

    return candidates


def create_hard_negatives(candidates: Dict[str, List[str]],
                          qrels: Dict[str, Set[str]],
                          corpus: Dict[str, Dict]) -> Dict[str, List[str]]:
    """Filter out positives from candidates to create hard negatives

    For queries without BM25 candidates (too complex), use random sampling
    """
    print("\n=== Step 3: Creating Hard Negatives ===")

    import random

    # Lấy danh sách tất cả doc_ids để random sampling
    all_doc_ids = list(corpus.keys())

    hard_negs = {}
    queries_with_random_negs = 0

    for query_id, cand_list in candidates.items():
        positives = qrels.get(query_id, set())

        if cand_list:
            # Có BM25 candidates -> filter out positives
            hard_negs[query_id] = [doc_id for doc_id in cand_list if doc_id not in positives]
        else:
            # Không có BM25 candidates (query quá phức tạp)
            # -> random sample negatives từ corpus
            queries_with_random_negs += 1

            # Sample 100 random docs và loại bỏ positives
            random_samples = random.sample(all_doc_ids, min(100, len(all_doc_ids)))
            hard_negs[query_id] = [doc_id for doc_id in random_samples if doc_id not in positives]

    avg_hard_negs = sum(len(negs) for negs in hard_negs.values()) / len(hard_negs) if hard_negs else 0
    print(f"Created hard negatives for {len(hard_negs)} queries")
    print(f"Average hard negatives per query: {avg_hard_negs:.2f}")
    if queries_with_random_negs > 0:
        print(f"⚠️ {queries_with_random_negs} queries used random negatives (instead of BM25 hard negatives)")

    return hard_negs


def create_training_pairs(queries: Dict[str, str],
                         qrels: Dict[str, Set[str]],
                         hard_negs: Dict[str, List[str]],
                         num_hard_negs: int) -> List[Dict]:
    """Create training pairs (query, pos_doc, neg_docs)"""
    print(f"\n=== Step 4: Creating Training Pairs ===")

    pairs = []
    queries_without_negs = 0

    for query_id, query_text in queries.items():
        positives = list(qrels.get(query_id, []))
        negatives = hard_negs.get(query_id, [])

        if not positives:
            continue  # Skip queries without positive documents

        if not negatives:
            queries_without_negs += 1

        # For each positive, create a pair with hard negatives
        for pos_id in positives:
            pair = {
                "query_id": query_id,
                "pos_id": pos_id,
                "neg_ids": negatives[:num_hard_negs]  # Take top N hard negatives
            }
            pairs.append(pair)

    print(f"Created {len(pairs)} training pairs")
    print(f"Queries without hard negatives: {queries_without_negs}")

    return pairs


def main():
    """Main pipeline"""
    print("=" * 60)
    print("GOLD LAYER PIPELINE FOR PAR DATASET")
    print("=" * 60)

    # Create output directory
    os.makedirs(GoldConfig.GOLD_DIR, exist_ok=True)

    # Load input data
    print("\nLoading input data...")
    corpus = load_corpus(GoldConfig.CORPUS_FILE)
    queries = load_queries(GoldConfig.TRAIN_QUERIES)
    qrels = load_qrels(GoldConfig.QRELS_TRAIN)

    # Step 1: Build BM25 index
    # ⚠️ Xóa index cũ nếu muốn rebuild với format mới
    if not os.path.exists(GoldConfig.BM25_INDEX_DIR):
        build_bm25_index(GoldConfig.CORPUS_FILE, GoldConfig.BM25_INDEX_DIR)
    else:
        print(f"\n⚠️ BM25 index already exists at {GoldConfig.BM25_INDEX_DIR}")
        print("If you want to rebuild the index, delete it first and run again.")

    # Step 2: Retrieve BM25 candidates
    if not os.path.exists(GoldConfig.BM25_CANDIDATES):
        candidates = retrieve_bm25_candidates(queries, GoldConfig.BM25_INDEX_DIR, GoldConfig.TOP_K)
        with open(GoldConfig.BM25_CANDIDATES, 'w', encoding='utf-8') as f:
            json.dump(candidates, f, indent=2, ensure_ascii=False)
        print(f"Saved candidates to {GoldConfig.BM25_CANDIDATES}")
    else:
        print(f"\nLoading existing candidates from {GoldConfig.BM25_CANDIDATES}")
        with open(GoldConfig.BM25_CANDIDATES, 'r', encoding='utf-8') as f:
            candidates = json.load(f)

    # Step 3: Create hard negatives
    hard_negs = create_hard_negatives(candidates, qrels, corpus)
    with open(GoldConfig.BM25_HARD_NEGS, 'w', encoding='utf-8') as f:
        json.dump(hard_negs, f, indent=2, ensure_ascii=False)
    print(f"Saved hard negatives to {GoldConfig.BM25_HARD_NEGS}")

    # Step 4: Create training pairs
    pairs = create_training_pairs(queries, qrels, hard_negs, GoldConfig.NUM_HARD_NEGS)
    with open(GoldConfig.PAIRS_TRAIN, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f"Saved training pairs to {GoldConfig.PAIRS_TRAIN}")

    print("\n" + "=" * 60)
    print("GOLD LAYER PIPELINE COMPLETED!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - BM25 Index: {GoldConfig.BM25_INDEX_DIR}")
    print(f"  - BM25 Candidates: {GoldConfig.BM25_CANDIDATES}")
    print(f"  - Hard Negatives: {GoldConfig.BM25_HARD_NEGS}")
    print(f"  - Training Pairs: {GoldConfig.PAIRS_TRAIN}")


if __name__ == "__main__":
    main()


