import json
import os
import torch
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from transformers import AutoModel
from config import Config
from data_loader import CorpusDataset
from tqdm import tqdm

class MilvusInserter:
    def __init__(self, checkpoint_file='milvus_checkpoint.json'):
        self.checkpoint_file = checkpoint_file
        self.last_processed_idx = self.load_checkpoint()

        connections.connect(
            alias="default",
            host=Config.MILVUS_HOST,
            port=Config.MILVUS_PORT,
            user=Config.MILVUS_USER,
            password=Config.MILVUS_PASSWORD
        )

        self.model = AutoModel.from_pretrained(Config.MODEL_NAME)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.collection = self.setup_collection()

        self.corpus_dataset = CorpusDataset(Config.CORPUS_FILE)

    def load_checkpoint(self):
        """Load checkpoint để biết đã xử lý đến đâu"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                return data.get('last_processed_idx', 0)
        return 0

    def save_checkpoint(self, idx):
        """Lưu checkpoint"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump({'last_processed_idx': idx}, f)

    def setup_collection(self):
        """Tạo hoặc load collection trong Milvus"""
        if utility.has_collection(Config.COLLECTION_NAME):
            print(f"Collection '{Config.COLLECTION_NAME}' đã tồn tại, sử dụng collection này")
            collection = Collection(Config.COLLECTION_NAME)
        else:
            print(f"Tạo collection mới: '{Config.COLLECTION_NAME}'")
            fields = [
                FieldSchema(name="pmid", dtype=DataType.VARCHAR, max_length=50, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
            ]
            schema = CollectionSchema(fields=fields, description="PMC Papers Embeddings")
            collection = Collection(name=Config.COLLECTION_NAME, schema=schema)

            # Tạo index cho vector search
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            print("Đã tạo index cho collection")

        return collection

    def get_embedding(self, encoding):
        """Tạo embedding từ encoding đã có"""
        with torch.no_grad():
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            outputs = self.model(**encoding)
            # Lấy embedding từ [CLS] token
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        return embedding.tolist()

    def insert_batch(self, batch_data):
        """Insert một batch vào Milvus"""
        pmids = []
        embeddings = []

        for item in batch_data:
            pmids.append(item['pmid'])
            embeddings.append(item['embedding'])

        entities = [pmids, embeddings]
        self.collection.insert(entities)

    def process_single_item(self, paper_data):
        """Xử lý một document"""
        try:
            title = paper_data.get('title', '').strip()
            text = paper_data.get('text', '').strip()
            pmid = str(paper_data.get('_id', ''))

            if not title or not text:
                return None

            # Tạo text và encoding
            full_text = f"[CLS] {title} [SEP] {text} [SEP]"
            encoding = self.corpus_dataset.tokenizer(
                full_text,
                max_length=Config.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            embedding = self.get_embedding(encoding)

            return {
                'pmid': pmid,
                'embedding': embedding
            }
        except Exception as e:
            print(f"\nError processing document: {e}")
            return None

    def process_corpus(self, batch_size=100, save_interval=1000, start_idx=None, end_idx=None):
        """
        Xử lý corpus và insert vào Milvus

        Args:
            batch_size: số lượng document trong mỗi batch insert
            save_interval: sau bao nhiêu document thì save checkpoint
            start_idx: index bắt đầu (None để dùng checkpoint)
            end_idx: index kết thúc (None để xử lý hết file)
        """
        # Dùng start_idx nếu được cung cấp, không thì dùng checkpoint
        if start_idx is not None:
            current_idx = start_idx
        else:
            current_idx = self.last_processed_idx

        print(f"Bắt đầu từ index: {current_idx}")

        total_docs = len(self.corpus_dataset)
        if end_idx is not None:
            total_docs = min(end_idx, total_docs)

        print(f"Xử lý từ {current_idx} đến {total_docs}")

        batch_data = []
        line_idx = 0

        pbar = tqdm(total=total_docs - current_idx, initial=0)

        # Đọc file và xử lý tuần tự
        with open(Config.CORPUS_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip các dòng trước start index
                if line_idx < current_idx:
                    line_idx += 1
                    continue

                # Dừng nếu đã đến end index
                if end_idx is not None and line_idx >= end_idx:
                    break

                try:
                    paper = json.loads(line.strip())
                    result = self.process_single_item(paper)

                    if result:
                        batch_data.append(result)

                    # Insert batch khi đủ số lượng
                    if len(batch_data) >= batch_size:
                        self.insert_batch(batch_data)
                        batch_data = []

                    # Save checkpoint
                    if line_idx % save_interval == 0:
                        self.save_checkpoint(line_idx)
                        self.collection.flush()
                        pbar.set_description(f"Checkpoint: {line_idx}")

                except Exception as e:
                    print(f"\nError at line {line_idx}: {e}")

                line_idx += 1
                pbar.update(1)

        # Insert batch cuối cùng
        if batch_data:
            self.insert_batch(batch_data)

        self.save_checkpoint(line_idx)
        self.collection.flush()
        pbar.close()

        print(f"\n✅ Hoàn thành! Đã xử lý {line_idx - current_idx} documents")
        print(f"Số lượng entities trong collection: {self.collection.num_entities}")

    def load_index(self):
        """Load collection vào memory để search"""
        self.collection.load()
        print("Collection đã được load vào memory")


if __name__ == "__main__":
    inserter = MilvusInserter(checkpoint_file='milvus_checkpoint.json')

    # 11713201
    inserter.process_corpus(batch_size=100, save_interval=1000, start_idx=43000, end_idx=44000)

    inserter.load_index()