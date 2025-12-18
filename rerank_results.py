import json
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import os
from config import Config

# Configuration
# Configuration
INPUT_RESULTS = "./PAR/results/test_results_3.json"
OUTPUT_RESULTS = "./PAR/results/test_results_3_reranked.json"
# Model PubMedBERT gốc (Cần train thêm mới hiệu quả, nếu dùng ngay sẽ như random)
CROSS_ENCODER_MODEL = "./output/cross-encoder-pubmedbert" 
TOP_K_RERANK = 50 # Giảm xuống 50 để tối ưu tốc độ (Latency)
SCORE_THRESHOLD = 0.5 # Ngưỡng điểm để lọc kết quả không liên quan (Accuracy)

def load_data():
    """Load queries, corpus, and existing results."""
    print("Loading queries...")
    queries = {}
    with open(Config.TEST_QUERIES, 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q['_id']] = q['text']

    print("Loading corpus (this might take a while)...")
    corpus = {}
    # Lưu ý: Load hết corpus vào RAM có thể nặng. 
    # Nếu RAM yếu, nên dùng cơ chế lazy load hoặc chỉ load docs có trong results.
    # Ở đây mình demo cách tối ưu: Chỉ load docs cần thiết.
    
    # 1. Đọc results trước để biết cần doc nào
    print(f"Loading results from {INPUT_RESULTS}...")
    with open(INPUT_RESULTS, 'r', encoding='utf-8') as f:
        results = json.load(f)
        
    needed_doc_ids = set()
    for qid, doc_scores in results.items():
        # Lấy top K doc ids của mỗi query
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_RERANK]
        for doc_id, _ in sorted_docs:
            needed_doc_ids.add(doc_id)
            
    print(f"Need to load {len(needed_doc_ids)} unique documents for reranking.")
    
    # 2. Scan corpus và chỉ lấy docs cần
    with open(Config.CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_id = str(doc['_id'])
            if doc_id in needed_doc_ids:
                corpus[doc_id] = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
                
    return queries, corpus, results

def rerank():
    queries, corpus, results = load_data()
    
    print(f"Loading Cross-Encoder: {CROSS_ENCODER_MODEL}...")
    # num_labels=1 để output ra 1 điểm số (regression/ranking) thay vì classification
    model = CrossEncoder(CROSS_ENCODER_MODEL, num_labels=1, max_length=512, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs for Reranking!")
        model.model = torch.nn.DataParallel(model.model)
    
    reranked_results = {}
    
    print(f"Reranking top {TOP_K_RERANK} for {len(results)} queries...")
    for qid, doc_scores in tqdm(results.items()):
        if qid not in queries: continue
        
        query_text = queries[qid]
        
        # Lấy top K candidates từ kết quả retrieval cũ
        sorted_candidates = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_RERANK]
        
        # Chuẩn bị pairs để đưa vào Cross-Encoder: (Query, Doc)
        pairs = []
        valid_doc_ids = []
        
        for doc_id, initial_score in sorted_candidates:
            if doc_id in corpus:
                pairs.append([query_text, corpus[doc_id]])
                valid_doc_ids.append(doc_id)
        
        if not pairs:
            reranked_results[qid] = doc_scores # Giữ nguyên nếu không có gì để rerank
            continue
            
        # Predict scores
        cross_scores = model.predict(pairs)
        
        # Cập nhật lại điểm số
        new_scores = {}
        # Giữ lại các docs nằm ngoài top K (không được rerank) với điểm cũ (hoặc bỏ qua tùy chiến lược)
        # Chiến lược an toàn: Copy toàn bộ điểm cũ, sau đó ghi đè top K bằng điểm mới
        # Tuy nhiên điểm Cross-Encoder (logits) khác thang đo với Dot Product/L2.
        # Nên tốt nhất là tách biệt: Top K reranked đứng đầu, còn lại xếp sau.
        
        # Ở đây mình sẽ tạo dict kết quả mới chỉ chứa Top K đã rerank (để evaluate MRR/NDCG@10)
        # Nếu muốn giữ Recall@1000, cần merge khéo léo hơn.
        
        filtered_scores = []
        for doc_id, score in zip(valid_doc_ids, cross_scores):
            # Chỉ giữ lại các bài có điểm cao hơn ngưỡng (nếu cần thiết)
            # Hoặc giữ tất cả nhưng sắp xếp lại
            filtered_scores.append((doc_id, float(score)))
            
        # Sắp xếp lại theo điểm Cross-Encoder giảm dần
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Lọc theo threshold (Optional: Nếu bạn muốn loại bỏ hẳn các bài kém)
        final_scores = {}
        for doc_id, score in filtered_scores:
            if score >= SCORE_THRESHOLD:
                final_scores[doc_id] = score
        
        # Nếu không còn bài nào đạt ngưỡng, có thể fallback lấy bài cao điểm nhất hoặc trả về rỗng
        if not final_scores and filtered_scores:
             # Fallback: Lấy bài tốt nhất dù điểm thấp, hoặc để trống để báo "Không tìm thấy"
             # Ở đây mình demo lấy top 1 nếu rỗng để tránh lỗi pipeline, nhưng log warning
             # final_scores[filtered_scores[0][0]] = filtered_scores[0][1]
             pass

        reranked_results[qid] = final_scores

    # Save
    print(f"Saving reranked results to {OUTPUT_RESULTS}...")
    with open(OUTPUT_RESULTS, 'w', encoding='utf-8') as f:
        json.dump(reranked_results, f, indent=2)
        
    print("Done! Run evaluation on the new file.")

if __name__ == "__main__":
    rerank()
