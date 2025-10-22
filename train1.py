import os
import json
import shutil
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from tqdm.autonotebook import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

warnings.filterwarnings("ignore")
class Config:
    # Paths
    DATA_DIR = "/mnt/par/data_warehouse/Data Warehouse"
    CORPUS_FILE = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/corpus.jsonl")
    PMID_MESH_FILE = os.path.join(DATA_DIR, "meta_data/PMID2MeSH.json")
    PATIENTS_FILE = os.path.join(DATA_DIR, "PMC-Patients.json")
    RELEVANCE_FILE = os.path.join(DATA_DIR, "patient2article_relevance.json")

    # PAR Dataset Paths
    TRAIN_QUERIES = os.path.join(DATA_DIR, "ReCDS_benchmark/queries/train_queries.jsonl")
    TRAIN_QRELS = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/qrels_train.tsv")
    DEV_QUERIES = os.path.join(DATA_DIR, "ReCDS_benchmark/queries/dev_queries.jsonl")
    DEV_QRELS = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/qrels_dev.tsv")
    TEST_QUERIES = os.path.join(DATA_DIR, "ReCDS_benchmark/queries/test_queries.jsonl")
    TEST_QRELS = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/qrels_test.tsv")
    RECDS_CORPUS = os.path.join(DATA_DIR, "ReCDS_benchmark/corpus.jsonl")

    # Model
    MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    MAX_LENGTH = 256
    EMBEDDING_DIM = 768
    POOLING = "mean"  # 'cls', 'mean', or 'max'

    # Training Hyperparameters
    BATCH_SIZE = 448
    NUM_EPOCHS = 10
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    TEMPERATURE = 0.05  # For InfoNCE loss

    # Legacy (for backward compatibility)
    EPOCHS = NUM_EPOCHS

    # System
    NUM_WORKERS = 0  # Set to 0 on Windows to avoid multiprocessing issues
    CHECKPOINT_DIR = "/mnt/par/data_warehouse/Data Warehouse/PAR/checkpoints"
    LOG_DIR = "/mnt/par/data_warehouse/Data Warehouse/PAR/logs"
    SAVE_EVERY = 2
    LOG_INTERVAL = 50

    # Milvus
    MILVUS_HOST = "10.243.88.63"
    MILVUS_PORT = "19530"
    MILVUS_USER = "root"
    MILVUS_PASSWORD = "aiostorm"
    COLLECTION_NAME = "pmc_papers"

    # Tasks weights
    CLASSIFICATION_WEIGHT = 1.0
    LINK_PREDICTION_WEIGHT = 1.0
    RETRIEVAL_WEIGHT = 1.0

class PARDatasetOptimized(Dataset):
    """
    Optimized Dataset for PAR (Patient Article Retrieval) bi-encoder training
    Chỉ load các documents cần thiết thay vì load toàn bộ corpus vào RAM
    """
    def __init__(self, queries_file, qrels_file, corpus_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.corpus_file = corpus_file

        print("Loading queries...")
        # Load queries
        self.queries = {}
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line.strip())
                self.queries[query['_id']] = query['text']
        print(f"Loaded {len(self.queries)} queries")

        print("Loading qrels...")
        # Load qrels (query-document relevance)
        self.qrels = {}  # {query_id: [list of relevant doc_ids]}
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    query_id, doc_id, relevance = parts
                    try:
                        if int(relevance) > 0:
                            if query_id not in self.qrels:
                                self.qrels[query_id] = []
                            self.qrels[query_id].append(doc_id)
                    except ValueError:
                        continue
        print(f"Loaded {len(self.qrels)} query-document relevance pairs")

        # Create training pairs: (query_id, positive_doc_id)
        print("Creating training pairs...")
        self.pairs = []
        for query_id, doc_ids in self.qrels.items():
            if query_id in self.queries:
                for doc_id in doc_ids:
                    self.pairs.append((query_id, doc_id))
        print(f"Created {len(self.pairs)} query-document pairs")

        # Tìm tất cả doc_ids cần thiết
        needed_doc_ids = set()
        for query_id, doc_ids in self.qrels.items():
            needed_doc_ids.update(doc_ids)
        print(f"Need to load {len(needed_doc_ids)} documents from corpus")

        # Build corpus index: CHỈ load các documents cần thiết
        print(f"Loading only required documents from {corpus_file}...")
        self.corpus = {}
        loaded_count = 0
        total_lines = 0

        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
                doc = json.loads(line.strip())
                doc_id = str(doc.get('_id', ''))

                # CHỈ load nếu doc_id có trong needed_doc_ids
                if doc_id in needed_doc_ids:
                    title = doc.get('title', '').strip()
                    abstract = doc.get('text', '').strip()
                    if title or abstract:
                        self.corpus[doc_id] = {
                            'title': title,
                            'abstract': abstract
                        }
                        loaded_count += 1

                # Progress indicator
                if total_lines % 100000 == 0:
                    print(f"Scanned {total_lines} lines, loaded {loaded_count}/{len(needed_doc_ids)} needed docs")

                # Early exit nếu đã load đủ tất cả docs cần thiết
                if loaded_count >= len(needed_doc_ids):
                    print(f"All required documents loaded. Stopping scan.")
                    break

        print(f"Loaded {len(self.corpus)} documents (out of {total_lines} scanned)")

        # Lọc lại pairs để chỉ giữ những pairs có cả query và doc
        original_pairs_count = len(self.pairs)
        self.pairs = [(q_id, d_id) for q_id, d_id in self.pairs if d_id in self.corpus]
        print(f"Filtered pairs: {original_pairs_count} -> {len(self.pairs)} (removed {original_pairs_count - len(self.pairs)} pairs with missing docs)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query_id, pos_doc_id = self.pairs[idx]

        # Encode query (patient summary)
        query_text = self.queries[query_id]
        query_encoding = self.tokenizer(
            query_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Encode positive document (title + abstract)
        pos_doc = self.corpus[pos_doc_id]
        pos_doc_text = f"{pos_doc['title']} {pos_doc['abstract']}".strip()
        pos_doc_encoding = self.tokenizer(
            pos_doc_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'pos_doc_input_ids': pos_doc_encoding['input_ids'].squeeze(0),
            'pos_doc_attention_mask': pos_doc_encoding['attention_mask'].squeeze(0),
            'query_id': query_id,
            'pos_doc_id': pos_doc_id
        }


class PARDatasetLazyLoad(Dataset):
    """
    Lazy loading version - không load corpus vào RAM
    Chỉ index vị trí các documents trong file, đọc on-the-fly khi cần
    Phù hợp khi corpus RẤT lớn (>10GB)
    """
    def __init__(self, queries_file, qrels_file, corpus_file, tokenizer, max_length=512, build_index=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.corpus_file = corpus_file

        print("Loading queries...")
        # Load queries
        self.queries = {}
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line.strip())
                self.queries[query['_id']] = query['text']
        print(f"Loaded {len(self.queries)} queries")

        print("Loading qrels...")
        # Load qrels
        self.qrels = {}
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    query_id, doc_id, relevance = parts
                    try:
                        if int(relevance) > 0:
                            if query_id not in self.qrels:
                                self.qrels[query_id] = []
                            self.qrels[query_id].append(doc_id)
                    except ValueError:
                        continue
        print(f"Loaded {len(self.qrels)} query-document relevance pairs")

        # Create pairs
        print("Creating training pairs...")
        self.pairs = []
        for query_id, doc_ids in self.qrels.items():
            if query_id in self.queries:
                for doc_id in doc_ids:
                    self.pairs.append((query_id, doc_id))
        print(f"Created {len(self.pairs)} query-document pairs")

        # Build file index: {doc_id: byte_offset}
        if build_index:
            self._build_corpus_index()
        else:
            self.doc_index = {}
            print("Skipping index building (will search linearly - SLOW)")

    def _build_corpus_index(self):
        """Build index mapping doc_id to file byte offset"""
        print(f"Building corpus index from {self.corpus_file}...")
        self.doc_index = {}

        # Tìm tất cả doc_ids cần thiết
        needed_doc_ids = set()
        for query_id, doc_ids in self.qrels.items():
            needed_doc_ids.update(doc_ids)
        print(f"Need to index {len(needed_doc_ids)} documents")

        count = 0
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            offset = 0
            for line in f:
                doc = json.loads(line.strip())
                doc_id = str(doc.get('_id', ''))

                if doc_id in needed_doc_ids:
                    self.doc_index[doc_id] = offset
                    count += 1

                if count % 100000 == 0:
                    print(f"Indexed {count} documents")

                # Early exit
                if count >= len(needed_doc_ids):
                    break

                # Update offset to next line
                offset += len(line.encode('utf-8'))

        print(f"Index built: {len(self.doc_index)} documents indexed")

        # Filter pairs
        original_count = len(self.pairs)
        self.pairs = [(q_id, d_id) for q_id, d_id in self.pairs if d_id in self.doc_index]
        print(f"Filtered pairs: {original_count} -> {len(self.pairs)}")

    def _get_document(self, doc_id):
        """Read document from file by doc_id"""
        if doc_id not in self.doc_index:
            return None

        offset = self.doc_index[doc_id]
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            doc = json.loads(line.strip())
            return {
                'title': doc.get('title', '').strip(),
                'abstract': doc.get('text', '').strip()
            }

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query_id, pos_doc_id = self.pairs[idx]

        # Encode query
        query_text = self.queries[query_id]
        query_encoding = self.tokenizer(
            query_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get document from file (lazy load)
        pos_doc = self._get_document(pos_doc_id)
        if pos_doc is None:
            # Fallback to empty doc if not found
            pos_doc_text = ""
        else:
            pos_doc_text = f"{pos_doc['title']} {pos_doc['abstract']}".strip()

        pos_doc_encoding = self.tokenizer(
            pos_doc_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'pos_doc_input_ids': pos_doc_encoding['input_ids'].squeeze(0),
            'pos_doc_attention_mask': pos_doc_encoding['attention_mask'].squeeze(0),
            'query_id': query_id,
            'pos_doc_id': pos_doc_id
        }

class BiEncoder(nn.Module):
    def __init__(self, model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 embedding_dim=768, pooling='cls'):
        super(BiEncoder, self).__init__()

        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.doc_encoder = AutoModel.from_pretrained(model_name)

        self.pooling = pooling
        self.embedding_dim = embedding_dim

        # Optional projection layer (uncomment if you want different embedding dim)
        # self.projection = nn.Linear(768, embedding_dim)

    def pool_embeddings(self, last_hidden_state, attention_mask):
        if self.pooling == 'cls':
            # Use [CLS] token embedding
            return last_hidden_state[:, 0, :]

        elif self.pooling == 'mean':
            # Mean pooling with attention mask
            token_embeddings = last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        elif self.pooling == 'max':
            # Max pooling
            token_embeddings = last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9
            return torch.max(token_embeddings, 1)[0]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def encode_query(self, input_ids, attention_mask):
        # Encode query (patient summary)
        outputs = self.query_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        pooled = self.pool_embeddings(outputs.last_hidden_state, attention_mask)

        # Optional: apply projection
        # pooled = self.projection(pooled)

        # L2 normalization for cosine similarity
        return F.normalize(pooled, p=2, dim=1)

    def encode_doc(self, input_ids, attention_mask):
        # Encode document (article title + abstract)
        outputs = self.doc_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        pooled = self.pool_embeddings(outputs.last_hidden_state, attention_mask)

        # Optional: apply projection
        # pooled = self.projection(pooled)

        # L2 normalization for cosine similarity
        return F.normalize(pooled, p=2, dim=1)

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        query_embeddings = self.encode_query(query_input_ids, query_attention_mask)
        doc_embeddings = self.encode_doc(doc_input_ids, doc_attention_mask)
        return query_embeddings, doc_embeddings


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings):
        batch_size = query_embeddings.size(0)

        similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T) / self.temperature

        labels = torch.arange(batch_size, device=query_embeddings.device)

        loss_q2d = self.criterion(similarity_matrix, labels)
        loss_d2q = self.criterion(similarity_matrix.T, labels)

        loss = (loss_q2d + loss_d2q) / 2.0

        return loss

def save_checkpoint(filepath, epoch, step, model, optimizer, loss):
    print(f"Đang lưu checkpoint: epoch {epoch}, step {step}, loss {loss:.4f}")
    # Nếu model là DataParallel, lấy module gốc
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    last_path = os.path.join(filepath, "last_model.pt")
    torch.save(checkpoint, last_path)
    best_path = os.path.join(filepath, "best_model.pt")

    if not os.path.exists(best_path):
        torch.save(checkpoint, best_path)
        print("Đã lưu best_model.pt (lần đầu)")
    else:
        try:
            best_loss = torch.load(best_path, map_location='cpu')['loss']
            if loss < best_loss:
                torch.save(checkpoint, best_path)
                print(f"Đã cập nhật best_model.pt: {best_loss:.4f} → {loss:.4f}")
        except Exception as e:
            print(f"Lỗi khi đọc best_model.pt: {e} → lưu đè")
            torch.save(checkpoint, best_path)


def load_checkpoint(filepath, model, optimizer, device):
    last_path = "/mnt/par/data_warehouse/Data Warehouse/PAR/checkpoints/best_model.pt"
    best_path = "/mnt/par/data_warehouse/Data Warehouse/PAR/checkpoints/best_model.pt"

    start_epoch = 0
    start_step = 0
    best_loss = float('inf')
    loaded = False

    # Xác định model để load state_dict (xử lý DataParallel)
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model

    if os.path.isfile(last_path):
        try:
            checkpoint = torch.load(last_path, map_location=device)
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            start_step = checkpoint.get('step', 0)
            loaded = True
            print(f"Loaded từ last_model.pt epoch {checkpoint['epoch']}")
        except Exception as e:
            print(f"Lỗi khi load last_model.pt: {e}")

    if not loaded and os.path.isfile(best_path):
        try:
            checkpoint = torch.load(best_path, map_location=device)
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            start_step = checkpoint.get('step', 0)
            print(f"Loaded từ best_model.pt epoch {checkpoint['epoch']}")
            loaded = True
        except Exception as e:
            print(f"Lỗi khi load best_model.pt: {e}")

    if not loaded:
        print("Không tìm thấy hoặc không load được checkpoint, bắt đầu từ đầu.")
        start_epoch = 0
        start_step = 0

    if os.path.isfile(best_path):
        try:
            best_loss = torch.load(best_path, map_location=device)['loss']
        except:
            best_loss = float('inf')
    else:
        best_loss = float('inf')

    return start_epoch, start_step, best_loss


def train():
    batch_size = Config.BATCH_SIZE
    lr = Config.LEARNING_RATE
    epochs = Config.NUM_EPOCHS
    weight_decay = Config.WEIGHT_DECAY
    warmup_ratio = Config.WARMUP_RATIO
    max_grad_norm = Config.MAX_GRAD_NORM
    temperature = Config.TEMPERATURE
    log_path = Config.LOG_DIR
    checkpoint_path = Config.CHECKPOINT_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(batch_size)
    # Check số lượng GPU
    n_gpus = torch.cuda.device_count()
    print(f"Số lượng GPU có sẵn: {n_gpus}")
    if n_gpus > 1:
        print(f"Sử dụng DataParallel trên {n_gpus} GPU")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = BiEncoder(
        model_name=Config.MODEL_NAME,
        embedding_dim=Config.EMBEDDING_DIM,
        pooling=Config.POOLING
    ).to(device)

    # Wrap model với DataParallel nếu có nhiều GPU
    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Model wrapped với DataParallel trên GPU: {list(range(n_gpus))}")

    train_dataset = PARDatasetOptimized(
        queries_file=Config.TRAIN_QUERIES,
        qrels_file=Config.TRAIN_QRELS,
        corpus_file=Config.CORPUS_FILE,
        tokenizer=tokenizer,
        max_length=Config.MAX_LENGTH
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        shuffle=True,
        drop_last=False,
        pin_memory=True if device.type == 'cuda' else False
    )

    dev_dataset = PARDatasetOptimized(
        queries_file=Config.DEV_QUERIES,
        qrels_file=Config.DEV_QRELS,
        corpus_file=Config.CORPUS_FILE,
        tokenizer=tokenizer,
        max_length=Config.MAX_LENGTH
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        shuffle=False,
        drop_last=False,
        pin_memory=True if device.type == 'cuda' else False
    )

    criterion = InfoNCELoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Create directories and TensorBoard writer
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    writer = SummaryWriter(log_path)

    start_epoch, global_step, best_loss = load_checkpoint(checkpoint_path, model, optimizer, device)
    dev_loss = best_loss  # Initialize dev_loss for intermediate checkpoints

    # ==== TRAINING LOOP ====
    for epoch in range(start_epoch, epochs):
        model.train()
        all_losses = []

        progress_bar = tqdm(train_dataloader, colour="BLUE")
        for i, batch in enumerate(progress_bar):
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            pos_doc_input_ids = batch['pos_doc_input_ids'].to(device)
            pos_doc_attention_mask = batch['pos_doc_attention_mask'].to(device)

            # Forward pass
            query_embeddings, doc_embeddings = model(
                query_input_ids,
                query_attention_mask,
                pos_doc_input_ids,
                pos_doc_attention_mask
            )

            loss = criterion(query_embeddings, doc_embeddings)

            progress_bar.set_description(
                "Epoch {}/{}. Loss {:0.4f}. LR {:0.2e}".format(
                    epoch + 1, epochs, loss.item(), scheduler.get_last_lr()[0]
                )
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            all_losses.append(loss.item())

            writer.add_scalar("Train/loss", loss.item(), global_step)
            writer.add_scalar("Train/learning_rate", scheduler.get_last_lr()[0], global_step)

            global_step += 1
            if i %  100 == 0 and i > 0:
                save_checkpoint(checkpoint_path, epoch, global_step, model, optimizer, dev_loss)

        train_loss = sum(all_losses) / len(all_losses)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

        # ==== VALIDATION ====
        model.eval()
        all_dev_losses = []

        with torch.no_grad():
            for batch in tqdm(dev_dataloader, desc="Evaluating", colour="GREEN"):
                query_input_ids = batch['query_input_ids'].to(device)
                query_attention_mask = batch['query_attention_mask'].to(device)
                pos_doc_input_ids = batch['pos_doc_input_ids'].to(device)
                pos_doc_attention_mask = batch['pos_doc_attention_mask'].to(device)

                query_embeddings, doc_embeddings = model(
                    query_input_ids,
                    query_attention_mask,
                    pos_doc_input_ids,
                    pos_doc_attention_mask
                )

                loss = criterion(query_embeddings, doc_embeddings)
                all_dev_losses.append(loss.item())

        dev_loss = sum(all_dev_losses) / len(all_dev_losses)
        print(f"Epoch {epoch + 1}/{epochs} - Dev Loss: {dev_loss:.4f}")

        writer.add_scalar("Dev/loss", dev_loss, epoch)
        writer.add_scalar("Train/epoch_loss", train_loss, epoch)

        save_checkpoint(checkpoint_path, epoch, global_step, model, optimizer, dev_loss)

    writer.close()


if __name__ == '__main__':
    train()