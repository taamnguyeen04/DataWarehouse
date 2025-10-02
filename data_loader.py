import json
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from config import Config

class Mesh(Dataset):
    def __init__(self, file_path):
        self.mesh_path = file_path
        self.mesh_data = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.mesh_data = json.load(f)
        
        self.pmids = list(self.mesh_data.keys())

    def __len__(self):
        return len(self.pmids)

    def __getitem__(self, idx):
        pmid = self.pmids[idx]
        mesh_labels = self.mesh_data[pmid]
        
        return {
            'pmid': pmid,
            'mesh_labels': mesh_labels
        }

    def __getitembyid__(self, pmid):
        if pmid in self.mesh_data:
            return {
                'pmid': pmid,
                'mesh_labels': self.mesh_data[pmid]
            }
        else:
            raise KeyError(f"PMID {pmid} không tồn tại trong dữ liệu MeSH.")


class CorpusDataset(Dataset):
    def __init__(self, file_path):
        self.corpus_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
        # Đếm số dòng trong file để biết length
        self.length = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                self.length += 1
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Đọc đến dòng thứ idx
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == idx:
                    paper = json.loads(line.strip())
                    break
        
        text = f"[CLS] {paper['title']} [SEP] {paper['text']} [SEP]"
        # print("encoding")
        encoding = self.tokenizer(
            text,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        # print("encode xong")
        return text, encoding

    def __getitembyid__(self, pmid):
        """Lấy paper theo PMID thay vì index"""
        # Tìm paper có PMID tương ứng
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line.strip())
                if str(paper.get('_id', '')) == str(pmid):
                    text = f"[CLS] {paper['title']} [SEP] {paper['text']} [SEP]"
                    return text
        
        # Nếu không tìm thấy PMID
        return None

# class PMCdataset(Dataset):
#     def __init__(self):
#         self.corpus = CorpusDataset(Config.CORPUS_FILE)
#         self.mesh = Mesh(Config.PMID_MESH_FILE)
#
#     def __len__(self):
#         return 1
#
#     def __getitem__(self, idx):
#         return self.corpus[idx], self.mesh[idx]

def build_common_ids(corpus_file, mesh_file, output_file):
    # Đọc mesh IDs (thường file nhỏ hơn, có thể load hết vào RAM)
    with open(mesh_file, "r", encoding="utf-8") as f:
        mesh_data = json.load(f)
    mesh_pmids = set(map(str, mesh_data.keys()))
    print(f"Loaded {len(mesh_pmids)} mesh IDs")

    # Đọc corpus.jsonl line-by-line và tìm giao
    common_pmids = []
    count = 0
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            count += 1
            paper = json.loads(line.strip())
            pmid = str(paper.get("_id", ""))
            if pmid in mesh_pmids:
                common_pmids.append(pmid)

            # In tiến độ để theo dõi
            if count % 1_000_000 == 0:
                print(f"Processed {count} lines, found {len(common_pmids)} common IDs")

    # Lưu ra file txt
    with open(output_file, "w", encoding="utf-8") as f:
        for pmid in common_pmids:
            f.write(pmid + "\n")

    print(f"✅ Done! Found {len(common_pmids)} common IDs. Saved to {output_file}")


class StreamingContrastiveDataset(torch.utils.data.IterableDataset):
    def __init__(self, corpus_file, tokenizer, max_length=512):
        self.corpus_file = corpus_file
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return 11713201

    def __iter__(self):
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    paper = json.loads(line.strip())
                    title = paper.get('title', '').strip()
                    abstract = paper.get('text', '').strip()

                    if len(title) < 10 or len(abstract) < 50:
                        continue

                    title_encoding = self.tokenizer(
                        title,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )

                    abstract_encoding = self.tokenizer(
                        abstract,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )

                    yield {
                        'title_input_ids': title_encoding['input_ids'].squeeze(0),
                        'title_attention_mask': title_encoding['attention_mask'].squeeze(0),
                        'abstract_input_ids': abstract_encoding['input_ids'].squeeze(0),
                        'abstract_attention_mask': abstract_encoding['attention_mask'].squeeze(0),
                    }

                except (json.JSONDecodeError, KeyError) as e:
                    continue


class ContrastivePairDataset(Dataset):
    def __init__(self, corpus_file, max_samples=None):
        self.corpus_file = corpus_file
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.max_length = Config.MAX_LENGTH

        self.data = []
        count = 0

        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break

                paper = json.loads(line.strip())
                title = paper.get('title', '').strip()
                abstract = paper.get('text', '').strip()

                if title and abstract:
                    self.data.append({
                        'title': title,
                        'abstract': abstract,
                        'pmid': paper.get('_id', str(count))
                    })
                count += 1

                if count % 100000 == 0:
                    print(f"Loaded {count} papers, kept {len(self.data)} valid pairs")

        print(f"Final dataset size: {len(self.data)} (title, abstract) pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        title_encoding = self.tokenizer(
            item['title'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        abstract_encoding = self.tokenizer(
            item['abstract'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'title_input_ids': title_encoding['input_ids'].squeeze(0),
            'title_attention_mask': title_encoding['attention_mask'].squeeze(0),
            'abstract_input_ids': abstract_encoding['input_ids'].squeeze(0),
            'abstract_attention_mask': abstract_encoding['attention_mask'].squeeze(0),
            'pmid': item['pmid']
        }


class PARDataset(Dataset):
    """Dataset for PAR (Patient Article Retrieval) bi-encoder training"""
    def __init__(self, queries_file, qrels_file, corpus_file, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load queries
        self.queries = {}
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line.strip())
                # query format: {"_id": "...", "text": "patient summary..."}
                self.queries[query['_id']] = query['text']

        # Load qrels (query-document relevance)
        self.qrels = {}  # {query_id: [list of relevant doc_ids]}
        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 4:
                    query_id, doc_id, relevance = parts
                    if int(relevance) > 0:  # only positive relevance
                        if query_id not in self.qrels:
                            self.qrels[query_id] = []
                        self.qrels[query_id].append(doc_id)

        # Build corpus index: {doc_id: {"title": ..., "abstract": ...}}
        self.corpus = {}
        print(f"Loading corpus from {corpus_file}...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = str(doc.get('_id', ''))
                title = doc.get('title', '').strip()
                abstract = doc.get('text', '').strip()
                if title or abstract:
                    self.corpus[doc_id] = {
                        'title': title,
                        'abstract': abstract
                    }

        print(f"Loaded {len(self.corpus)} documents")

        # Create training pairs: (query_id, positive_doc_id)
        self.pairs = []
        for query_id, doc_ids in self.qrels.items():
            if query_id in self.queries:
                for doc_id in doc_ids:
                    if doc_id in self.corpus:
                        self.pairs.append((query_id, doc_id))

        print(f"Created {len(self.pairs)} query-document pairs")

        # All doc_ids for random negative sampling
        self.all_doc_ids = list(self.corpus.keys())

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


if __name__ == "__main__":
    corpus = CorpusDataset(Config.CORPUS_FILE)
    print(corpus[0])
    # print(corpus.__getitembyid__(15555068))
    # mesh = Mesh(Config.PMID_MESH_FILE)
    # print(mesh[0])
    # corpus_file = Config.CORPUS_FILE
    # mesh_file = Config.PMID_MESH_FILE
    # output_file = "common_pmids.txt"
    #
    # build_common_ids(corpus_file, mesh_file, output_file)