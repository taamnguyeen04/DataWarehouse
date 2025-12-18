import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from config import Config
from transformers import AutoTokenizer

# 1. User's Original Implementation (Linear Scan)
class CorpusDataset:
    def __init__(self, file_path):
        self.corpus_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        
        # Đếm số dòng trong file để biết length (Optional for this test, but kept for compatibility)
        # self.length = 0
        # with open(file_path, 'r', encoding='utf-8') as f:
        #     for _ in f:
        #         self.length += 1
    
    def __getitembyid__(self, pmid):
        """Lấy paper theo PMID thay vì index"""
        # Tìm paper có PMID tương ứng
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                paper = json.loads(line.strip())
                if str(paper.get('_id', '')) == str(pmid):
                    text = f"[CLS] {paper['title']} [SEP] {paper['text']} [SEP]"
                    return text
        return None

# 2. Multi-threaded Implementation (Parallel Scan)
class MultiThreadedCorpusDataset:
    def __init__(self, file_path, num_threads=8):
        self.corpus_path = file_path
        self.num_threads = num_threads
        self.file_size = os.path.getsize(file_path)
        self.found_result = None
        self.stop_event = threading.Event()

    def _search_chunk(self, start_byte, end_byte, pmid):
        if self.stop_event.is_set():
            return

        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            f.seek(start_byte)
            
            # If not at start of file, discard partial line
            if start_byte > 0:
                f.readline()
            
            while True:
                # Check stop condition
                if self.stop_event.is_set():
                    return
                
                # Check position
                current_pos = f.tell()
                if current_pos >= end_byte:
                    break
                
                line = f.readline()
                if not line:
                    break
                
                try:
                    # Fast check: check if pmid string is in line before parsing json
                    # This is a heuristic to speed up. JSON parsing is slow.
                    # Assuming "_id": "PMID" or "_id": PMID format
                    if str(pmid) not in line: 
                        continue

                    paper = json.loads(line.strip())
                    if str(paper.get('_id', '')) == str(pmid):
                        self.found_result = f"[CLS] {paper['title']} [SEP] {paper['text']} [SEP]"
                        self.stop_event.set()
                        return
                except ValueError:
                    continue

    def __getitembyid__(self, pmid):
        self.found_result = None
        self.stop_event.clear()
        
        chunk_size = self.file_size // self.num_threads
        futures = []
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for i in range(self.num_threads):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < self.num_threads - 1 else self.file_size
                futures.append(executor.submit(self._search_chunk, start, end, pmid))
        
        return self.found_result

# 3. Indexed Implementation (Best for repeated lookups)
class IndexedCorpusDataset:
    def __init__(self, file_path):
        self.corpus_path = file_path
        self.index = {} # pmid -> offset
        self._build_index()
        
    def _build_index(self):
        print("Building index (PMID -> Offset)...")
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                try:
                    # Quick extraction of ID without full JSON parse if possible
                    # But for safety we parse. To optimize, we could use regex or string find.
                    # For now, let's just parse.
                    doc = json.loads(line)
                    self.index[str(doc['_id'])] = offset
                except:
                    pass
        print(f"Index built. {len(self.index)} documents indexed.")

    def __getitembyid__(self, pmid):
        offset = self.index.get(str(pmid))
        if offset is None:
            return None
            
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            line = f.readline()
            paper = json.loads(line.strip())
            text = f"[CLS] {paper['title']} [SEP] {paper['text']} [SEP]"
            return text

def main():
    target_pmid = 23296716
    file_path = Config.CORPUS_FILE
    
    print(f"Target PMID: {target_pmid}")
    print(f"Corpus File: {file_path}")
    
    # 1. Test Baseline
    print("\n--- Testing Baseline (Linear Scan) ---")
    dataset = CorpusDataset(file_path)
    start_time = time.time()
    result = dataset.__getitembyid__(target_pmid)
    end_time = time.time()
    print(f"Result found: {result is not None}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    # 2. Test Multi-threaded
    print("\n--- Testing Multi-threaded (Parallel Scan) ---")
    mt_dataset = MultiThreadedCorpusDataset(file_path, num_threads=4)
    start_time = time.time()
    result = mt_dataset.__getitembyid__(target_pmid)
    end_time = time.time()
    print(f"Result found: {result is not None}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")

    # 3. Test Indexed (Hash Map)
    print("\n--- Testing Indexed (Hash Map) ---")
    start_build = time.time()
    indexed_dataset = IndexedCorpusDataset(file_path)
    end_build = time.time()
    print(f"Index build time: {end_build - start_build:.4f} seconds")
    
    start_time = time.time()
    result = indexed_dataset.__getitembyid__(target_pmid)
    end_time = time.time()
    print(f"Result found: {result is not None}")
    print(f"Lookup Time taken: {end_time - start_time:.6f} seconds")

if __name__ == "__main__":
    main()
