import json
import pandas as pd
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
        print("encoding")
        encoding = self.tokenizer(
            text,
            max_length=Config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        print("encode xong")
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


if __name__ == "__main__":
    corpus = CorpusDataset(Config.CORPUS_FILE)
    print(corpus[2])
    # print(corpus.__getitembyid__(15555068))
    # mesh = Mesh(Config.PMID_MESH_FILE)
    # print(mesh[0])
    corpus_file = Config.CORPUS_FILE
    mesh_file = Config.PMID_MESH_FILE
    output_file = "common_pmids.txt"

    build_common_ids(corpus_file, mesh_file, output_file)