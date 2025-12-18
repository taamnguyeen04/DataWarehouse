import json
import os
import pickle
from transformers import AutoTokenizer
from config import Config

class IndexedCorpusDataset:
    def __init__(self, file_path, index_path=None):
        self.corpus_path = file_path
        self.index_path = index_path if index_path else Config.CORPUS_INDEX_FILE
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.index = {} # pmid -> offset
        
        if os.path.exists(self.index_path):
            self._load_index()
        else:
            self._build_index()
            self._save_index()
        
    def _build_index(self):
        """
        Xây dựng index (PMID -> Offset) bằng cách quét file một lần.
        """
        print(f"Building index for {self.corpus_path}...")
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                try:
                    doc = json.loads(line)
                    self.index[str(doc['_id'])] = offset
                except:
                    pass
        print(f"Index built. {len(self.index)} documents indexed.")

    def _save_index(self):
        print(f"Saving index to {self.index_path}...")
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        print("Index saved.")

    def _load_index(self):
        print(f"Loading index from {self.index_path}...")
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        print(f"Index loaded. {len(self.index)} documents indexed.")

    def __getitembyid__(self, pmid):
        """
        Lấy paper theo PMID sử dụng index.
        Thời gian: O(1) ~ 0.001s
        """
        offset = self.index.get(str(pmid))
        if offset is None:
            return None
            
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            paper = json.loads(line.strip())
            
            text = f"[CLS] {paper['title']} [SEP] {paper['text']} [SEP]"
            return text

if __name__ == "__main__":
    # Test
    dataset = IndexedCorpusDataset(Config.CORPUS_FILE)
    pmid = 30860276
    print(f"Searching for PMID: {pmid}")
    result = dataset.__getitembyid__(pmid)
    print("Result found!" if result else "Not found")
    # print(result[:100] + "..." if result else "")
    print(result)
