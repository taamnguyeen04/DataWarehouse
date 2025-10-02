# PAR (Patient Article Retrieval) Bi-Encoder Training

Implementation of bi-encoder contrastive learning for Patient Article Retrieval using PubMedBERT.

## Architecture

### Bi-Encoder Model
- **Encoder**: PubMedBERT (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`)
- **Query Encoder**: Encodes patient summaries
- **Document Encoder**: Encodes article (title + abstract)
- **Loss**: InfoNCE / Contrastive Loss with in-batch negatives
- **Pooling**: CLS token, Mean pooling, or Max pooling
- **Output**: L2-normalized embeddings for cosine similarity

## Files

### 1. `data_loader.py`
Contains `PARDataset` class for loading PAR training data:
- Loads queries from JSONL files
- Loads qrels (query-document relevance) from TSV files
- Loads corpus (articles) from JSONL
- Creates positive query-document pairs for training
- Uses in-batch negatives (other documents in the same batch)

### 2. `par_biencoder.py`
- `BiEncoder`: Dual-encoder architecture with shared/separate encoders
- `InfoNCELoss`: Contrastive loss with temperature scaling
- Support for different pooling strategies (CLS, mean, max)

### 3. `train_par.py`
Main training script with:
- Large batch sizes (128-512) for more in-batch negatives
- AdamW optimizer with learning rate warmup
- Gradient clipping
- Periodic checkpointing
- Logging and evaluation on dev set

### 4. `evaluate_par.py`
Evaluation script that computes:
- MRR (Mean Reciprocal Rank)
- Recall@K (K=1, 5, 10, 20, 100)
- NDCG@K (Normalized Discounted Cumulative Gain)

## Data Format

### Queries (JSONL)
```json
{"_id": "query_id", "text": "patient summary..."}
```

### Qrels (TSV)
```
query_id    0    doc_id    relevance
```

### Corpus (JSONL)
```json
{"_id": "doc_id", "title": "...", "text": "abstract..."}
```

## Training Strategy

### 1. Contrastive Learning with In-Batch Negatives
- Each batch contains N query-document pairs
- For each query, the correct document is the positive
- Other (N-1) documents in the batch are negatives
- Large batch size → more negatives → better training

### 2. InfoNCE Loss
```python
similarity_matrix = query_emb @ doc_emb.T / temperature
loss = CrossEntropy(similarity_matrix, diagonal_labels)
```

### 3. Bi-directional Loss
- Query-to-Document loss
- Document-to-Query loss
- Final loss = average of both

## Usage

### Training
```bash
python train.py
```

**Configuration (in `train_par.py`):**
- `batch_size`: 128 (increase for more negatives, adjust based on GPU)
- `num_epochs`: 10
- `learning_rate`: 2e-5
- `temperature`: 0.05
- `max_length`: 512
- `pooling`: 'cls' (or 'mean', 'max')

### Evaluation
```bash
python evaluate_par.py
```

Update `model_path` in the script to your best checkpoint.

### Using Trained Model for Retrieval
```python
from par_biencoder import BiEncoder
from transformers import AutoTokenizer
import torch

# Load model
model = BiEncoder()
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')

# Encode query
query_text = "patient summary..."
query_encoding = tokenizer(query_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
query_emb = model.encode_query(query_encoding['input_ids'], query_encoding['attention_mask'])

# Encode documents (batch processing recommended)
doc_text = "article title and abstract..."
doc_encoding = tokenizer(doc_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
doc_emb = model.encode_doc(doc_encoding['input_ids'], doc_encoding['attention_mask'])

# Compute similarity
similarity = torch.cosine_similarity(query_emb, doc_emb)
```

## Optimization Tips

### For Limited GPU Memory
1. **Reduce batch size** (but try to keep ≥32 for decent negatives)
2. **Use gradient accumulation**:
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(train_loader):
       loss = loss / accumulation_steps
       loss.backward()

       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **Use MoCo (Momentum Contrast)** for memory bank of negatives
4. **Use mixed precision training** (FP16):
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()
   with autocast():
       query_emb, doc_emb = model(...)
       loss = loss_fn(query_emb, doc_emb)

   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

### For Faster Training
1. **Increase num_workers** in DataLoader (e.g., 4-8)
2. **Use pin_memory=True** for faster GPU transfer
3. **Pre-encode documents** and store in FAISS index
4. **Use DistributedDataParallel** for multi-GPU training

## Integration with Milvus

After training, encode all documents and insert into Milvus for fast ANN search:

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import torch

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Create collection
fields = [
    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
]
schema = CollectionSchema(fields)
collection = Collection("par_articles", schema)

# Encode and insert documents
batch_size = 1000
doc_ids = []
embeddings = []

for doc in corpus:
    doc_text = f"{doc['title']} {doc['text']}"
    encoding = tokenizer(doc_text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
    emb = model.encode_doc(encoding['input_ids'], encoding['attention_mask'])

    doc_ids.append(doc['_id'])
    embeddings.append(emb.cpu().numpy().flatten().tolist())

    if len(doc_ids) >= batch_size:
        collection.insert([doc_ids, embeddings])
        doc_ids = []
        embeddings = []

# Create index
index_params = {
    "metric_type": "IP",  # Inner Product (for normalized vectors = cosine similarity)
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
collection.create_index("embedding", index_params)

# Search
collection.load()
query_emb = model.encode_query(...)
results = collection.search(
    data=[query_emb.cpu().numpy().flatten().tolist()],
    anns_field="embedding",
    param={"metric_type": "IP", "params": {"nprobe": 10}},
    limit=10
)
```

## Expected Performance

Typical metrics on PAR benchmark:
- **MRR**: 0.30 - 0.45
- **Recall@10**: 0.40 - 0.60
- **NDCG@10**: 0.35 - 0.50

(Actual performance depends on dataset, hyperparameters, and training setup)

## References

- PubMedBERT: https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
- InfoNCE Loss: https://arxiv.org/abs/1807.03748
- Bi-Encoder for Retrieval: https://arxiv.org/abs/1908.10084