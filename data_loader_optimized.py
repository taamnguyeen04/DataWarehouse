import json
import torch
from torch.utils.data import Dataset
from config import Config


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


if __name__ == "__main__":
    from transformers import AutoTokenizer

    print("Testing PARDatasetOptimized...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    dataset = PARDatasetOptimized(
        queries_file=Config.TRAIN_QUERIES,
        qrels_file=Config.TRAIN_QRELS,
        corpus_file=Config.CORPUS_FILE,
        tokenizer=tokenizer,
        max_length=Config.MAX_LENGTH
    )

    print(f"\nDataset size: {len(dataset)}")
    print("First sample:")
    sample = dataset[0]
    print(f"Query ID: {sample['query_id']}")
    print(f"Positive Doc ID: {sample['pos_doc_id']}")
    print(f"Query input shape: {sample['query_input_ids'].shape}")
    print(f"Doc input shape: {sample['pos_doc_input_ids'].shape}")