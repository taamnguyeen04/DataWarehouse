# DataWarehouse
- test lần 1 train 1epoch: {'MRR': 0.01965, 'P@10': 0.00354, 'NDCG@10': 0.00703, 'R@1k': 0.0356}
{'MRR': 0.00706, 'P@10': 0.00081, 'NDCG@10': 0.00088, 'R@1k': 0.41731}
{'MRR': 0.00836, 'P@10': 0.00118, 'NDCG@10': 0.00116, 'R@1k': 0.453}
{'MRR': 0.06923, 'P@10': 0.01878, 'NDCG@10': 0.02329, 'R@1k': 0.453}
{'MRR': 0.19807, 'P@10': 0.06101, 'NDCG@10': 0.08311, 'R@1k': 0.453}
---

# 🟡 TẦNG GOLD – Chuẩn bị dữ liệu cho BM25 và huấn luyện Bi-Encoder

---

## 🎯 Mục tiêu của tầng Gold

Tầng **Gold** = “Feature Engineering Layer”
→ nơi bạn **tạo ra tất cả các dữ liệu trung gian đặc thù cho mô hình**, từ **Silver** (đã clean) để **mô hình chỉ cần đọc và train, không xử lý thêm**.

Trong bài toán PAR:

> Gold layer phải sinh ra các **negatives chất lượng cao** (BM25 hard negatives)
> và các **dataset pairs** `(query, pos_doc, neg_doc)` sẵn sàng cho DataLoader.

---

## ⚙️ 1️⃣ Input cho tầng Gold

| Tên file                    | Nguồn tầng Silver | Vai trò                                                    |
| --------------------------- | ----------------- | ---------------------------------------------------------- |
| `corpus_clean.jsonl`        | Silver            | Toàn bộ corpus bài báo đã làm sạch (title + abstract).     |
| `train_queries_clean.jsonl` | Silver            | Tập query bệnh nhân đã chuẩn hóa text.                     |
| `qrels_train.tsv`           | Silver            | Mapping query_id → positive_doc_id (relevance = 1 hoặc 2). |

---

## 🧩 2️⃣ Các bước xử lý trong tầng Gold

Tầng Gold gồm 4 bước chính (theo pipeline):

### **Bước 1. Build BM25 Index**

**Mục đích:**
Tạo index lexical (BM25) để có thể truy vấn các bài viết bằng text.

**Thực hiện bằng:** [Pyserini](https://github.com/castorini/pyserini)

**Input:** `corpus_clean.jsonl`
**Output:**

* Thư mục `bm25_index/` (Lucene index)

**Cấu trúc output:**

```
/PAR/gold/bm25_index/
    ├── segments_1/
    ├── write.lock
    ├── _SUCCESS
    └── ...
```

**Code minh họa:**
[corpus.jsonl](../../data/Data%20Warehouse/ReCDS_benchmark/PAR/corpus.jsonl)
"C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\PAR\qrels_train.tsv"
"C:\Users\tam\Documents\data\Data Warehouse\ReCDS_benchmark\queries\train_queries.jsonl"
```bash
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input C:\Users\tam\Documents\data\Data Warehouse/PAR/silver \
  --index C:\Users\tam\Documents\data\Data Warehouse/PAR/gold/bm25_index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw
```

> 🧠 Tip: Pyserini mặc định sẽ lowercase + remove stopwords (đúng với BM25 logic).
> Nếu bạn muốn giữ từ chuyên ngành, có thể dùng custom stopword list.

---

### **Bước 2. Retrieve top-k BM25 results cho mỗi query**

**Mục đích:**
Lấy **các bài báo “gần giống”** với query theo BM25 — dùng làm **candidates for hard negatives**.

**Input:**

* `train_queries_clean.jsonl`
* `qrels_train.tsv`
* BM25 index (`bm25_index/`)

**Output:**

* `bm25_candidates_topk.json`

**Cấu trúc file output:**

```json
{
  "P001": ["PM123", "PM456", "PM789", "PM111", ...],
  "P002": ["PM222", "PM333", "PM444", "PM555", ...],
  ...
}
```

---

### **Bước 3. Loại bỏ các positive khỏi top-k để tạo hard negatives**

**Mục đích:**
Từ top-k BM25, loại bỏ các tài liệu *đúng* (positive trong qrels), giữ lại những tài liệu *sai nhưng gần đúng* làm **hard negatives**.

**Input:**

* `bm25_candidates_topk.json`
* `qrels_train.tsv`

**Output:**

* `bm25_hard_negs.json`

**Cấu trúc output:**

```json
{
  "P001": ["PM456", "PM789", "PM111", "PM222"],
  "P002": ["PM333", "PM444", "PM555"],
  ...
}
```

---

### **Bước 4. Tạo cặp train-ready (query, pos_doc, neg_doc)**

**Mục đích:**
Ghép mỗi query với một positive (từ qrels) và vài negative (từ BM25 hoặc random)
→ mô hình bi-encoder có thể train trực tiếp.

**Input:**

* `train_queries_clean.jsonl`
* `qrels_train.tsv`
* `bm25_hard_negs.json`

**Output:**

* `pairs_train.jsonl`

**Cấu trúc file output:**

```json
{"query_id": "P001", "pos_id": "PM123", "neg_ids": ["PM456", "PM789"]}
{"query_id": "P002", "pos_id": "PM222", "neg_ids": ["PM333"]}
```

---

## 📦 3️⃣ Output đầy đủ của tầng Gold

| File                              | Vai trò                                          | Được dùng bởi           |
| --------------------------------- | ------------------------------------------------ | ----------------------- |
| `/gold/bm25_index/`               | Lucene index để truy vấn BM25                    | Bước 2 (retrieve top-k) |
| `/gold/bm25_candidates_topk.json` | Top-k kết quả BM25 cho mỗi query                 | Bước 3                  |
| `/gold/bm25_hard_negs.json`       | Danh sách hard negatives (BM25)                  | DataLoader khi train    |
| `/gold/pairs_train.jsonl`         | Dataset (query, pos_doc, neg_doc) cho huấn luyện | Bi-Encoder model        |
| `/gold/pairs_dev.jsonl`           | (tuỳ chọn) tạo từ dev split                      | Validation              |

---

## 🧠 4️⃣ Cách các file này được dùng trong huấn luyện

Trong code `train1.py`, bạn chỉ cần thay dòng khởi tạo dataset:

```python
train_dataset = PARDatasetOptimized(
    queries_file=Config.TRAIN_QUERIES,
    qrels_file=Config.TRAIN_QRELS,
    corpus_file=Config.CORPUS_FILE,
    tokenizer=tokenizer,
    max_length=Config.MAX_LENGTH,
    hard_negatives_file="/mnt/par/.../gold/bm25_hard_negs.json"
)
```

Dataset sẽ:

* Load positive từ qrels,
* Lấy 1–n negative từ file `bm25_hard_negs.json`,
* Tokenize và trả về cho mô hình train InfoNCE loss.

---

## 🪄 5️⃣ Tổng kết pipeline tầng Gold (một dòng mỗi bước)

```
1️⃣ Build BM25 index (Pyserini)
2️⃣ Retrieve top-k documents / query
3️⃣ Filter out positives → create hard negatives
4️⃣ Merge qrels + hard_negatives → pairs_train.jsonl
```

---

## 🧩 6️⃣ Mối quan hệ với tầng Silver và Model

```
SILVER
 ├── corpus_clean.jsonl
 ├── train_queries_clean.jsonl
 └── qrels_train.tsv
   ↓
GOLD
 ├── bm25_index/
 ├── bm25_candidates_topk.json
 ├── bm25_hard_negs.json
 └── pairs_train.jsonl
   ↓
MODEL
 └── train1.py (BiEncoder)
```
