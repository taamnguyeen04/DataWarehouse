import torch
import json
import random
import numpy as np
from transformers import AutoTokenizer
from model import BiEncoder
from config import Config
from tqdm import tqdm

def evaluate_local_sanity_check(model_path="best_model.pt", num_samples=1000):
    """
    Kiểm tra nhanh model mà KHÔNG cần Milvus.
    Mục tiêu: Xem model có xếp hạng document đúng (positive) cao hơn document ngẫu nhiên không.
    """
    print(f"🔍 Đang kiểm tra sanity check với {num_samples} mẫu...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = BiEncoder(Config.MODEL_NAME, Config.EMBEDDING_DIM, Config.POOLING).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        # Fix DataParallel prefix if needed
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print(f"✅ Loaded model from {model_path} (Epoch {checkpoint.get('epoch')})")
    except Exception as e:
        print(f"❌ Không load được model: {e}")
        return

    model.eval()

    # 2. Load Data (Test set)
    queries_path = Config.DATA_DIR + "/ReCDS_benchmark/queries/test_queries.jsonl"
    qrels_path = Config.DATA_DIR + "/ReCDS_benchmark/PAR/qrels_test.tsv"
    corpus_path = Config.CORPUS_FILE

    print("Loading data...")
    # Load Queries
    queries = {}
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q['_id']] = q['text']
            
    # Load Qrels
    qrels = {}
    with open(qrels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                try:
                    qid, docid, rel = parts[0], parts[1], int(parts[2])
                    if rel > 0:
                        if qid not in qrels: qrels[qid] = []
                        qrels[qid].append(docid)
                except ValueError:
                    continue # Skip header or malformed lines

    # Load Corpus (Lazy or just load needed ones)
    # Để nhanh, ta chỉ load docs cần thiết cho sample
    sample_qids = list(qrels.keys())
    if len(sample_qids) > num_samples:
        sample_qids = random.sample(sample_qids, num_samples)
    
    needed_docids = set()
    for qid in sample_qids:
        needed_docids.update(qrels[qid])
    
    # Thêm vài random negatives
    all_docids_in_corpus = [] # Sẽ fill khi đọc corpus
    
    print(f"Scanning corpus for {len(needed_docids)} documents...")
    docs = {}
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            did = str(d['_id'])
            all_docids_in_corpus.append(did)
            if did in needed_docids:
                docs[did] = f"{d.get('title','')} {d.get('text','')}"
                
    # 3. Evaluation Loop
    mrr_scores = []
    top1_acc = []
    
    print("\n🚀 Bắt đầu đánh giá...")
    with torch.no_grad():
        for qid in tqdm(sample_qids):
            if qid not in queries: continue
            
            query_text = queries[qid]
            pos_doc_ids = qrels[qid]
            
            # Lấy 1 positive doc
            pos_did = pos_doc_ids[0]
            if pos_did not in docs: continue
            pos_text = docs[pos_did]
            
            # Lấy 999 negative docs ngẫu nhiên (giả lập top 1000 re-ranking)
            neg_dids = random.sample(all_docids_in_corpus, 999)
            # Đảm bảo không trùng positive
            neg_dids = [d for d in neg_dids if d not in pos_doc_ids]
            
            # Load text cho negatives (cần đọc lại file hoặc chấp nhận random text nếu lười - ở đây ta cần text thật)
            # Cách đơn giản: ta chỉ so sánh với positive vs positive của query khác (in-batch negatives style)
            # Nhưng để chính xác, ta nên so sánh Pos vs Random Corpus Docs.
            # Để code đơn giản, tôi sẽ so sánh: Pos Doc vs 999 Docs của các Sample khác
            
            candidate_docs = [{'id': pos_did, 'text': pos_text, 'label': 1}]
            
            # Mượn docs của các query khác làm negatives (nhanh hơn đọc file)
            other_doc_ids = list(docs.keys())
            random.shuffle(other_doc_ids)
            for odid in other_doc_ids:
                if odid != pos_did and len(candidate_docs) < 1000:
                    candidate_docs.append({'id': odid, 'text': docs[odid], 'label': 0})
            
            if len(candidate_docs) < 2: continue

            # Encode Query
            q_enc = tokenizer(query_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt').to(device)
            q_emb = model.encode_query(q_enc['input_ids'], q_enc['attention_mask'])
            
            # Encode Candidates
            texts = [d['text'] for d in candidate_docs]
            d_enc = tokenizer(texts, max_length=512, padding='max_length', truncation=True, return_tensors='pt').to(device)
            d_embs = model.encode_doc(d_enc['input_ids'], d_enc['attention_mask'])
            
            # Compute Scores (Dot product)
            scores = torch.matmul(q_emb, d_embs.T).squeeze(0).cpu().numpy()
            
            # Rank
            ranked_indices = np.argsort(-scores) # Descending
            
            # Find rank of positive doc (index 0 in candidate_docs)
            pos_rank = -1
            for rank, idx in enumerate(ranked_indices):
                if candidate_docs[idx]['label'] == 1:
                    pos_rank = rank + 1
                    break
            
            if pos_rank != -1:
                mrr_scores.append(1.0 / pos_rank)
                top1_acc.append(1 if pos_rank == 1 else 0)

    print("\n" + "="*50)
    print(f"KẾT QUẢ SANITY CHECK ({len(mrr_scores)} queries)")
    print("="*50)
    print(f"MRR (trên tập nhỏ 1000 docs): {np.mean(mrr_scores):.4f}")
    print(f"Top-1 Accuracy: {np.mean(top1_acc):.4f}")
    print("-" * 50)
    if np.mean(mrr_scores) > 0.1:
        print("✅ Model CÓ học được patterns (MRR > 0.1).")
        print("👉 Vấn đề chắc chắn nằm ở Milvus Index (Embedding mismatch).")
        print("👉 Giải pháp: Chạy lại generate_embeddings.py và insert lại vào Milvus.")
    else:
        print("⚠️ Model có vẻ chưa học được gì (MRR rất thấp).")
        print("👉 Kiểm tra lại quá trình training, learning rate, hoặc data.")
    print("="*50)

if __name__ == "__main__":
    evaluate_local_sanity_check()
