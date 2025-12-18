import os
import json
import shutil
import warnings
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from tqdm.autonotebook import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

warnings.filterwarnings("ignore")

class Config:
    # Paths
    DATA_DIR = "C:/Users/tam/Desktop/Data/Data Warehouse"
    CORPUS_FILE = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/corpus.jsonl")
    TRAIN_QUERIES = os.path.join(DATA_DIR, "ReCDS_benchmark/queries/train_queries.jsonl")
    TRAIN_QRELS = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/qrels_train.tsv")
    DEV_QUERIES = os.path.join(DATA_DIR, "ReCDS_benchmark/queries/dev_queries.jsonl")
    DEV_QRELS = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/qrels_dev.tsv")
    BM25_HARD_NEGS_FILE = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/gold/bm25_hard_negs.json")
    PAIRS_TRAIN_FILE = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/gold/pairs_train.jsonl")
    
    # Preprocessed data paths
    PREPROCESSED_DIR = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/preprocessed")
    CORPUS_CHUNKS_DIR = os.path.join(PREPROCESSED_DIR, "corpus_chunks")
    CORPUS_INDEX_FILE = os.path.join(PREPROCESSED_DIR, "corpus_index.json")
    TOKENIZED_TRAIN_QUERIES = os.path.join(PREPROCESSED_DIR, "tokenized_train_queries.pt")
    TOKENIZED_DEV_QUERIES = os.path.join(PREPROCESSED_DIR, "tokenized_dev_queries.pt")

    # Model
    MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    MAX_LENGTH = 384
    EMBEDDING_DIM = 768
    POOLING = "mean"

    # Training
    BATCH_SIZE = 256
    NUM_EPOCHS = 1
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    TEMPERATURE = 0.05
    NUM_HARD_NEGATIVES = 2
    USE_MIXED_PRECISION = True
    # System
    NUM_WORKERS = 4  # Tối ưu cho CPU multiprocessing (tăng lên 8-12 nếu CPU có nhiều cores)
    PREFETCH_FACTOR = 2  # Pre-fetch batches để giảm idle time
    PERSISTENT_WORKERS = True  # Giữ workers alive giữa các epochs
    CHECKPOINT_DIR = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/checkpoints")
    LOG_DIR = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/logs")

class PARDatasetOptimized(Dataset):
    def __init__(self, queries_file, qrels_file, corpus_file, tokenizer, max_length=512,
                 bm25_hard_negs_file=None, pairs_train_file=None, num_hard_negatives=2):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_hard_negatives = num_hard_negatives

        print("Loading queries...")
        self.queries = {}
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line.strip())
                self.queries[query['_id']] = query['text']
        print(f"Loaded {len(self.queries)} queries")

        print(f"Loading training pairs...")
        self.pairs = []
        with open(pairs_train_file, 'r', encoding='utf-8') as f:
            for line in f:
                pair = json.loads(line.strip())
                self.pairs.append({
                    'query_id': pair['query_id'],
                    'pos_id': pair['pos_id'],
                    'neg_ids': pair.get('neg_ids', [])
                })
        print(f"Loaded {len(self.pairs)} training pairs")

        # Load required documents
        needed_doc_ids = set()
        for pair in self.pairs:
            needed_doc_ids.add(pair['pos_id'])
            needed_doc_ids.update(pair['neg_ids'])
        print(f"Need to load {len(needed_doc_ids)} documents")

        print(f"Loading documents...")
        self.corpus = {}
        loaded_count = 0
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = str(doc.get('_id', ''))
                if doc_id in needed_doc_ids:
                    title = doc.get('title', '').strip()
                    abstract = doc.get('text', '').strip()
                    if title or abstract:
                        self.corpus[doc_id] = {'title': title, 'abstract': abstract}
                        loaded_count += 1
                # if loaded_count >= len(needed_doc_ids):
                #     break

        print(f"Loaded {len(self.corpus)} documents")

        # Filter pairs
        self.pairs = [
            {
                'query_id': p['query_id'],
                'pos_id': p['pos_id'],
                'neg_ids': [n for n in p['neg_ids'] if n in self.corpus]
            }
            for p in self.pairs
            if p['query_id'] in self.queries and p['pos_id'] in self.corpus
        ]
        print(f"Filtered to {len(self.pairs)} valid pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        query_id = pair['query_id']
        pos_id = pair['pos_id']
        neg_ids = pair['neg_ids']
    
        # ✅ Kiểm tra dữ liệu bị thiếu
        if query_id not in self.queries or pos_id not in self.corpus:
            return None
    
        query_text = self.queries[query_id]
        pos_doc = self.corpus[pos_id]
        pos_text = f"{pos_doc['title']} {pos_doc['abstract']}".strip()
    
        query_enc = self.tokenizer(
            query_text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        pos_enc = self.tokenizer(
            pos_text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
    
        result = {
            'query_input_ids': query_enc['input_ids'].squeeze(0),
            'query_attention_mask': query_enc['attention_mask'].squeeze(0),
            'pos_doc_input_ids': pos_enc['input_ids'].squeeze(0),
            'pos_doc_attention_mask': pos_enc['attention_mask'].squeeze(0),
        }
    
        # Encode hard negatives nếu có
        if neg_ids:
            neg_ids_list = []
            neg_masks_list = []
            for neg_id in neg_ids[:self.num_hard_negatives]:
                if neg_id in self.corpus:
                    neg_doc = self.corpus[neg_id]
                    neg_text = f"{neg_doc['title']} {neg_doc['abstract']}".strip()
                    neg_enc = self.tokenizer(
                        neg_text, max_length=self.max_length,
                        padding='max_length', truncation=True, return_tensors='pt'
                    )
                    neg_ids_list.append(neg_enc['input_ids'].squeeze(0))
                    neg_masks_list.append(neg_enc['attention_mask'].squeeze(0))
    
            if neg_ids_list:
                result['neg_doc_input_ids'] = torch.stack(neg_ids_list)
                result['neg_doc_attention_mask'] = torch.stack(neg_masks_list)
            else:
                result['neg_doc_input_ids'] = torch.empty(0, self.max_length, dtype=torch.long)
                result['neg_doc_attention_mask'] = torch.empty(0, self.max_length, dtype=torch.long)
        else:
            result['neg_doc_input_ids'] = torch.empty(0, self.max_length, dtype=torch.long)
            result['neg_doc_attention_mask'] = torch.empty(0, self.max_length, dtype=torch.long)
    
        # ✅ Luôn return result ở cuối, không nằm trong else
        return result

class PARDatasetPreTokenized(Dataset):
    """
    Dataset sử dụng dữ liệu đã tokenize sẵn từ preprocessed folder
    Lazy loading cho corpus chunks để tiết kiệm RAM
    """
    def __init__(self, queries_file, qrels_file, corpus_file, tokenizer, max_length=512,
                 bm25_hard_negs_file=None, pairs_train_file=None, num_hard_negatives=2):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_hard_negatives = num_hard_negatives
        
        # Load corpus index mapping (doc_id -> chunk_file)
        print("Loading corpus index...")
        with open(Config.CORPUS_INDEX_FILE, 'r') as f:
            self.corpus_index = json.load(f)
        print(f"Loaded corpus index with {len(self.corpus_index)} documents")
        
        # Cache cho các chunks đã load (LRU cache)
        self.loaded_chunks = {}
        self.max_cached_chunks = 3  # Chỉ cache tối đa 3 chunks trong RAM
        self.chunk_access_order = []  # Theo dõi thứ tự access để LRU
        
        # Load tokenized queries
        print("Loading pre-tokenized queries...")
        if 'train' in queries_file:
            tokenized_queries = torch.load(Config.TOKENIZED_TRAIN_QUERIES)
        else:
            tokenized_queries = torch.load(Config.TOKENIZED_DEV_QUERIES)
        
        # Convert to dict with query_id -> query_text mapping (for qrels)
        self.queries = {}
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line.strip())
                self.queries[query['_id']] = query['text']
        
        self.tokenized_queries = tokenized_queries
        print(f"Loaded {len(self.tokenized_queries)} pre-tokenized queries")
        
        # Load training pairs
        print(f"Loading training pairs...")
        self.pairs = []
        with open(pairs_train_file, 'r', encoding='utf-8') as f:
            for line in f:
                pair = json.loads(line.strip())
                self.pairs.append({
                    'query_id': pair['query_id'],
                    'pos_id': pair['pos_id'],
                    'neg_ids': pair.get('neg_ids', [])
                })
        print(f"Loaded {len(self.pairs)} training pairs")
        
        # Filter pairs (chỉ giữ lại pairs có query và document hợp lệ)
        self.pairs = [
            {
                'query_id': p['query_id'],
                'pos_id': p['pos_id'],
                'neg_ids': [n for n in p['neg_ids'] if n in self.corpus_index]
            }
            for p in self.pairs
            if p['query_id'] in self.tokenized_queries and p['pos_id'] in self.corpus_index
        ]
        print(f"Filtered to {len(self.pairs)} valid pairs")
    
    def _load_chunk(self, chunk_filename):
        """Load chunk file với LRU caching"""
        if chunk_filename in self.loaded_chunks:
            # Move to end (most recently used)
            self.chunk_access_order.remove(chunk_filename)
            self.chunk_access_order.append(chunk_filename)
            return self.loaded_chunks[chunk_filename]
        
        # Load new chunk
        chunk_path = os.path.join(Config.CORPUS_CHUNKS_DIR, chunk_filename)
        chunk_data = torch.load(chunk_path)
        
        # Add to cache
        self.loaded_chunks[chunk_filename] = chunk_data
        self.chunk_access_order.append(chunk_filename)
        
        # Remove oldest chunk if cache is full
        if len(self.loaded_chunks) > self.max_cached_chunks:
            oldest_chunk = self.chunk_access_order.pop(0)
            del self.loaded_chunks[oldest_chunk]
        
        return chunk_data
    
    def _get_doc_tokens(self, doc_id):
        """Lấy tokens của document từ chunk file"""
        if doc_id not in self.corpus_index:
            return None
        
        chunk_filename = self.corpus_index[doc_id]
        chunk_data = self._load_chunk(chunk_filename)
        
        if doc_id not in chunk_data:
            return None
        
        return chunk_data[doc_id]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        query_id = pair['query_id']
        pos_id = pair['pos_id']
        neg_ids = pair['neg_ids']
        
        # Get query tokens
        if query_id not in self.tokenized_queries:
            return None
        
        query_tokens = self.tokenized_queries[query_id]
        
        # Get positive document tokens
        pos_tokens = self._get_doc_tokens(pos_id)
        if pos_tokens is None:
            return None
        
        result = {
            'query_input_ids': query_tokens['input_ids'],
            'query_attention_mask': query_tokens['attention_mask'],
            'pos_doc_input_ids': pos_tokens['input_ids'],
            'pos_doc_attention_mask': pos_tokens['attention_mask'],
        }
        
        # Get hard negative tokens
        if neg_ids:
            neg_ids_list = []
            neg_masks_list = []
            for neg_id in neg_ids[:self.num_hard_negatives]:
                neg_tokens = self._get_doc_tokens(neg_id)
                if neg_tokens is not None:
                    neg_ids_list.append(neg_tokens['input_ids'])
                    neg_masks_list.append(neg_tokens['attention_mask'])
            
            if neg_ids_list:
                result['neg_doc_input_ids'] = torch.stack(neg_ids_list)
                result['neg_doc_attention_mask'] = torch.stack(neg_masks_list)
            else:
                result['neg_doc_input_ids'] = torch.empty(0, self.max_length, dtype=torch.long)
                result['neg_doc_attention_mask'] = torch.empty(0, self.max_length, dtype=torch.long)
        else:
            result['neg_doc_input_ids'] = torch.empty(0, self.max_length, dtype=torch.long)
            result['neg_doc_attention_mask'] = torch.empty(0, self.max_length, dtype=torch.long)
        
        return result

class BiEncoder(nn.Module):
    def __init__(self, model_name, embedding_dim, pooling='mean'):
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.doc_encoder = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def pool_embeddings(self, last_hidden_state, attention_mask):
        if self.pooling == 'cls':
            return last_hidden_state[:, 0, :]
        elif self.pooling == 'mean':
            token_embeddings = last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        elif self.pooling == 'max':
            token_embeddings = last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9
            return torch.max(token_embeddings, 1)[0]

    def encode_query(self, input_ids, attention_mask):
        outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self.pool_embeddings(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=1)

    def encode_doc(self, input_ids, attention_mask):
        outputs = self.doc_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = self.pool_embeddings(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=1)

    def forward(self, query_input_ids=None, query_attention_mask=None,
                doc_input_ids=None, doc_attention_mask=None, mode='dual'):
        if mode == 'dual':
            return self.encode_query(query_input_ids, query_attention_mask), \
                   self.encode_doc(doc_input_ids, doc_attention_mask)
        elif mode == 'doc':
            return self.encode_doc(doc_input_ids, doc_attention_mask)
        elif mode == 'query':
            return self.encode_query(query_input_ids, query_attention_mask)
        
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings, hard_neg_embeddings=None):
        batch_size = query_embeddings.size(0)
        similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T) / self.temperature

        if hard_neg_embeddings is not None and hard_neg_embeddings.size(1) > 0:
            hard_neg_sim = torch.bmm(
                query_embeddings.unsqueeze(1),
                hard_neg_embeddings.transpose(1, 2)
            ).squeeze(1) / self.temperature
            similarity_matrix = torch.cat([similarity_matrix, hard_neg_sim], dim=1)

        labels = torch.arange(batch_size, device=query_embeddings.device)
        loss_q2d = self.criterion(similarity_matrix, labels)
        
        doc2query_sim = torch.matmul(doc_embeddings, query_embeddings.T) / self.temperature
        loss_d2q = self.criterion(doc2query_sim, labels)
        
        return (loss_q2d + loss_d2q) / 2.0
    
def compute_metrics(query_embeddings, doc_embeddings, query_ids, doc_ids, qrels_dict, k_values=[10, 100, 1000]):
    similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T).cpu().numpy()
    metrics = defaultdict(list)

    for i, query_id in enumerate(query_ids):
        relevant_docs = qrels_dict.get(query_id, set())
        if not relevant_docs:
            continue

        scores = similarity_matrix[i]
        sorted_indices = np.argsort(-scores)
        sorted_doc_ids = [doc_ids[idx] for idx in sorted_indices]

        for k in k_values:
            top_k_docs = sorted_doc_ids[:k]
            num_relevant = len(set(top_k_docs) & relevant_docs)
            
            metrics[f'Recall@{k}'].append(num_relevant / len(relevant_docs))
            metrics[f'P@{k}'].append(num_relevant / k)
            
            dcg = sum(1.0 / np.log2(rank + 2) for rank, doc_id in enumerate(top_k_docs) if doc_id in relevant_docs)
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant_docs), k)))
            metrics[f'nDCG@{k}'].append(dcg / idcg if idcg > 0 else 0.0)

        for rank, doc_id in enumerate(sorted_doc_ids, start=1):
            if doc_id in relevant_docs:
                metrics['MRR'].append(1.0 / rank)
                break
        else:
            metrics['MRR'].append(0.0)

    return {name: np.mean(values) for name, values in metrics.items()}


def evaluate_full_corpus(model, dataset, device, batch_size=32, use_amp=False):
    """
    Evaluation cho pre-tokenized dataset
    Hỗ trợ mixed precision trong quá trình evaluation
    """
    from torch.cuda.amp import autocast
    
    model.eval()
    max_length = dataset.max_length

    # Build qrels từ pairs
    qrels_dict = defaultdict(set)
    for pair in dataset.pairs:
        qrels_dict[pair['query_id']].add(pair['pos_id'])
    
    # Collect all unique doc_ids từ pairs (chỉ evaluate trên docs cần thiết)
    needed_doc_ids = set()
    for pair in dataset.pairs:
        needed_doc_ids.add(pair['pos_id'])
    
    # Encode queries từ pre-tokenized data
    query_ids = list(dataset.queries.keys())
    query_embeddings_list = []
    print("Encoding queries from pre-tokenized data...")
    
    for i in tqdm(range(0, len(query_ids), batch_size)):
        batch_ids = query_ids[i:i + batch_size]
        
        # Stack pre-tokenized queries
        batch_input_ids = []
        batch_attention_masks = []
        for qid in batch_ids:
            if qid in dataset.tokenized_queries:
                batch_input_ids.append(dataset.tokenized_queries[qid]['input_ids'])
                batch_attention_masks.append(dataset.tokenized_queries[qid]['attention_mask'])
        
        if not batch_input_ids:
            continue
            
        batch_input_ids = torch.stack(batch_input_ids)
        batch_attention_masks = torch.stack(batch_attention_masks)
        
        with torch.no_grad():
            if use_amp:
                with autocast():
                    embs = model(query_input_ids=batch_input_ids.to(device),
                                query_attention_mask=batch_attention_masks.to(device), mode='query')
            else:
                embs = model(query_input_ids=batch_input_ids.to(device),
                            query_attention_mask=batch_attention_masks.to(device), mode='query')
            query_embeddings_list.append(embs.cpu())
    
    query_embeddings = torch.cat(query_embeddings_list, dim=0)
    
    # Encode documents từ pre-tokenized chunks
    doc_ids = list(needed_doc_ids)
    doc_embeddings_list = []
    print("Encoding documents from pre-tokenized chunks...")
    
    for i in tqdm(range(0, len(doc_ids), batch_size)):
        batch_ids = doc_ids[i:i + batch_size]
        
        # Load pre-tokenized documents từ chunks
        batch_input_ids = []
        batch_attention_masks = []
        valid_batch_ids = []
        
        for did in batch_ids:
            doc_tokens = dataset._get_doc_tokens(did)
            if doc_tokens is not None:
                batch_input_ids.append(doc_tokens['input_ids'])
                batch_attention_masks.append(doc_tokens['attention_mask'])
                valid_batch_ids.append(did)
        
        if not batch_input_ids:
            continue
        
        batch_input_ids = torch.stack(batch_input_ids)
        batch_attention_masks = torch.stack(batch_attention_masks)
        
        with torch.no_grad():
            if use_amp:
                with autocast():
                    embs = model(doc_input_ids=batch_input_ids.to(device),
                                doc_attention_mask=batch_attention_masks.to(device), mode='doc')
            else:
                embs = model(doc_input_ids=batch_input_ids.to(device),
                            doc_attention_mask=batch_attention_masks.to(device), mode='doc')
            doc_embeddings_list.append(embs.cpu())
    
    doc_embeddings = torch.cat(doc_embeddings_list, dim=0)

    return compute_metrics(query_embeddings, doc_embeddings, query_ids, doc_ids, qrels_dict)


def save_checkpoint(filepath, epoch, step, model, optimizer, loss, mrr=None, scaler=None):
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'mrr': mrr if mrr is not None else 0.0,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # Save scaler state if using mixed precision
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    last_path = os.path.join(filepath, "last_model.pt")
    torch.save(checkpoint, last_path)

    best_path = os.path.join(filepath, "best_model.pt")
    if not os.path.exists(best_path):
        torch.save(checkpoint, best_path)
    else:
        best_checkpoint = torch.load(best_path, map_location='cpu')
        best_mrr = best_checkpoint.get('mrr', 0.0)
        if mrr is not None and mrr > best_mrr:
            torch.save(checkpoint, best_path)
            print(f"✅ Updated best model: MRR {best_mrr:.4f} → {mrr:.4f}")


def load_checkpoint(filepath, model, optimizer, device, scaler=None):
    last_path = os.path.join(filepath, "last_model.pt")
    best_path = os.path.join(filepath, "best_model.pt")

    model_to_load = model.module if isinstance(model, nn.DataParallel) else model

    if os.path.isfile(last_path):
        checkpoint = torch.load(last_path, map_location=device)
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Loaded from last_model.pt epoch {checkpoint['epoch']}")
        return checkpoint['epoch'] + 1, checkpoint.get('step', 0), \
               checkpoint.get('loss', float('inf')), checkpoint.get('mrr', 0.0)

    if os.path.isfile(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        print(f"Loaded from best_model.pt epoch {checkpoint['epoch']}")
        return checkpoint['epoch'] + 1, checkpoint.get('step', 0), \
               checkpoint.get('loss', float('inf')), checkpoint.get('mrr', 0.0)

    print("No checkpoint found, starting from scratch")
    return 0, 0, float('inf'), 0.0

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)
    
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    
    use_amp = Config.USE_MIXED_PRECISION and device.type == 'cuda'
    if use_amp:
        print("✓ Mixed Precision Training ENABLED (FP16) - Expect ~2x speedup")
    else:
        print("✗ Mixed Precision Training DISABLED")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_fast=True)
    model = BiEncoder(Config.MODEL_NAME, Config.EMBEDDING_DIM, Config.POOLING).to(device)
    
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel on {n_gpus} GPUs")

    train_dataset = PARDatasetPreTokenized(
        queries_file=Config.TRAIN_QUERIES,
        qrels_file=Config.TRAIN_QRELS,
        corpus_file=Config.CORPUS_FILE,
        tokenizer=tokenizer,
        max_length=Config.MAX_LENGTH,
        bm25_hard_negs_file=Config.BM25_HARD_NEGS_FILE,
        pairs_train_file=Config.PAIRS_TRAIN_FILE,
        num_hard_negatives=Config.NUM_HARD_NEGATIVES
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_skip_none,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        persistent_workers=Config.PERSISTENT_WORKERS if Config.NUM_WORKERS > 0 else False
    )

    dev_dataset = PARDatasetPreTokenized(
        queries_file=Config.DEV_QUERIES,
        qrels_file=Config.DEV_QRELS,
        corpus_file=Config.CORPUS_FILE,
        tokenizer=tokenizer,
        max_length=Config.MAX_LENGTH,
        bm25_hard_negs_file=None,  # Dev không cần hard negatives
        pairs_train_file=Config.PAIRS_TRAIN_FILE,
        num_hard_negatives=0
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        shuffle=False,
        drop_last=False,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=Config.PREFETCH_FACTOR if Config.NUM_WORKERS > 0 else None,
        persistent_workers=Config.PERSISTENT_WORKERS if Config.NUM_WORKERS > 0 else False
    )

    criterion = InfoNCELoss(Config.TEMPERATURE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, 
                                 weight_decay=Config.WEIGHT_DECAY)

    # Initialize GradScaler for mixed precision
    scaler = GradScaler() if use_amp else None

    total_steps = len(train_dataloader) * Config.NUM_EPOCHS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if os.path.isdir(Config.LOG_DIR):
        shutil.rmtree(Config.LOG_DIR)
    os.makedirs(Config.LOG_DIR)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    writer = SummaryWriter(Config.LOG_DIR)
    start_epoch, global_step, best_loss, best_mrr = load_checkpoint(
        Config.CHECKPOINT_DIR, model, optimizer, device, scaler
    )

    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        model.train()
        all_losses = []

        progress_bar = tqdm(train_dataloader, colour="BLUE")
        for i, batch in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # Mixed precision training
            if use_amp:
                with autocast():
                    query_embeddings, doc_embeddings = model(
                        batch['query_input_ids'].to(device),
                        batch['query_attention_mask'].to(device),
                        batch['pos_doc_input_ids'].to(device),
                        batch['pos_doc_attention_mask'].to(device)
                    )

                    hard_neg_embeddings = None
                    if 'neg_doc_input_ids' in batch and batch['neg_doc_input_ids'].size(1) > 0:
                        neg_ids = batch['neg_doc_input_ids'].to(device)
                        neg_mask = batch['neg_doc_attention_mask'].to(device)
                        batch_size, num_negs, max_len = neg_ids.size()
                        
                        neg_embs = model(doc_input_ids=neg_ids.view(-1, max_len),
                                        doc_attention_mask=neg_mask.view(-1, max_len), mode='doc')
                        hard_neg_embeddings = neg_embs.view(batch_size, num_negs, -1)

                    loss = criterion(query_embeddings, doc_embeddings, hard_neg_embeddings)
                
                # Backward with gradient scaling
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 training
                query_embeddings, doc_embeddings = model(
                    batch['query_input_ids'].to(device),
                    batch['query_attention_mask'].to(device),
                    batch['pos_doc_input_ids'].to(device),
                    batch['pos_doc_attention_mask'].to(device)
                )

                hard_neg_embeddings = None
                if 'neg_doc_input_ids' in batch and batch['neg_doc_input_ids'].size(1) > 0:
                    neg_ids = batch['neg_doc_input_ids'].to(device)
                    neg_mask = batch['neg_doc_attention_mask'].to(device)
                    batch_size, num_negs, max_len = neg_ids.size()
                    
                    neg_embs = model(doc_input_ids=neg_ids.view(-1, max_len),
                                    doc_attention_mask=neg_mask.view(-1, max_len), mode='doc')
                    hard_neg_embeddings = neg_embs.view(batch_size, num_negs, -1)

                loss = criterion(query_embeddings, doc_embeddings, hard_neg_embeddings)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                optimizer.step()
            
            scheduler.step()
            
            progress_bar.set_description(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} Loss {loss.item():.4f} LR {scheduler.get_last_lr()[0]:.2e}"
            )

            all_losses.append(loss.item())
            writer.add_scalar("Train/loss", loss.item(), global_step)
            writer.add_scalar("Train/learning_rate", scheduler.get_last_lr()[0], global_step)

            # Calculate MRR from current batch
            with torch.no_grad():
                batch_size = query_embeddings.size(0)
                similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T).cpu().numpy()

                batch_mrr = []
                for i in range(batch_size):
                    scores = similarity_matrix[i]
                    # Positive document is at index i (diagonal)
                    rank = 1 + np.sum(scores > scores[i])
                    batch_mrr.append(1.0 / rank)

                current_mrr = np.mean(batch_mrr)

            writer.add_scalar("Train/MRR", current_mrr, global_step)
            global_step += 1

            # Evaluate and save every 1000 steps
            if global_step % 1000 == 0:
                print(f"\n💾 Saved checkpoint at step {global_step} (MRR: {current_mrr:.4f})")
                save_checkpoint(Config.CHECKPOINT_DIR, epoch, global_step, model, optimizer,
                                loss.item(), current_mrr, scaler)

            # Save regular checkpoint every 500 steps (but not at 1000, 2000, ...)
            elif global_step % 500 == 0:
                ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"step_{global_step}.pt")
                save_checkpoint(Config.CHECKPOINT_DIR, epoch, global_step, model, optimizer,
                                loss.item(), None, scaler)
                print(f"💾 Saved checkpoint at step {global_step} → {ckpt_path}")
        train_loss = np.mean(all_losses)
        print(f"\nEpoch {epoch+1} - Train Loss: {train_loss:.4f}")

        # Validation
        model.eval()
        dev_losses = []
        with torch.no_grad():
            for batch in tqdm(dev_dataloader, desc="Dev loss"):
                if use_amp:
                    with autocast():
                        query_embs, doc_embs = model(
                            batch['query_input_ids'].to(device),
                            batch['query_attention_mask'].to(device),
                            batch['pos_doc_input_ids'].to(device),
                            batch['pos_doc_attention_mask'].to(device)
                        )
                        loss = criterion(query_embs, doc_embs, None)
                else:
                    query_embs, doc_embs = model(
                        batch['query_input_ids'].to(device),
                        batch['query_attention_mask'].to(device),
                        batch['pos_doc_input_ids'].to(device),
                        batch['pos_doc_attention_mask'].to(device)
                    )
                    loss = criterion(query_embs, doc_embs, None)
                dev_losses.append(loss.item())

        dev_loss = np.mean(dev_losses)
        print(f"Epoch {epoch+1} - Dev Loss: {dev_loss:.4f}")

        dev_metrics = evaluate_full_corpus(model, dev_dataset, device, use_amp=use_amp)
        print(f"\nDev Metrics:")
        print(f"  MRR:       {dev_metrics.get('MRR', 0):.4f}")
        print(f"  nDCG@10:   {dev_metrics.get('nDCG@10', 0):.4f}")
        print(f"  Recall@1k: {dev_metrics.get('Recall@1000', 0):.4f}")

        writer.add_scalar("Dev/loss", dev_loss, epoch)
        writer.add_scalar("Dev/MRR", dev_metrics.get('MRR', 0), epoch)
        writer.add_scalar("Dev/nDCG@10", dev_metrics.get('nDCG@10', 0), epoch)
        writer.add_scalar("Dev/Recall@1k", dev_metrics.get('Recall@1000', 0), epoch)

        save_checkpoint(Config.CHECKPOINT_DIR, epoch, global_step, model, optimizer,
                       dev_loss, dev_metrics.get('MRR', 0), scaler)

    writer.close()


if __name__ == '__main__':
    train()
