import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from model import BiEncoder
from config import Config
import argparse
import math

class CorpusDataset(Dataset):
    def __init__(self, corpus_file, tokenizer, max_length=384, start_idx=0, end_idx=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lines = []
        
        print(f"Loading corpus from line {start_idx} to {end_idx if end_idx else 'end'}...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            # Skip to start_idx
            for _ in range(start_idx):
                next(f, None)
            
            # Read until end_idx
            count = 0
            limit = end_idx - start_idx if end_idx else float('inf')
            
            for line in f:
                if count >= limit:
                    break
                self.lines.append(line)
                count += 1
        print(f"Loaded {len(self.lines)} documents into memory.")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        try:
            doc = json.loads(line)
            doc_id = str(doc['_id'])
            text = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
            
            enc = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'doc_id': doc_id,
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0)
            }
        except Exception as e:
            print(f"Error processing line {idx}: {e}")
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    
    return {
        'doc_id': [b['doc_id'] for b in batch],
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch])
    }

def generate_embeddings():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="best_model.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="embeddings_output")
    
    # Arguments for distributed processing (sharding)
    parser.add_argument("--total_shards", type=int, default=1500, help="Total number of machines/notebooks running in parallel")
    parser.add_argument("--shard_id", type=int, default=0, help="ID of the current shard (0 to total_shards-1)")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")

    # 1. Calculate Shard Range
    # First count total lines in file (efficiently)
    print("Counting total documents in corpus...")
    total_docs = 0
    with open(Config.CORPUS_FILE, 'r', encoding='utf-8') as f:
        for _ in f:
            total_docs += 1
    print(f"Total documents: {total_docs}")
    
    docs_per_shard = math.ceil(total_docs / args.total_shards)
    start_idx = args.shard_id * docs_per_shard
    end_idx = min((args.shard_id + 1) * docs_per_shard, total_docs)
    
    print(f"Shard {args.shard_id}/{args.total_shards}: Processing documents {start_idx} to {end_idx}")
    
    if start_idx >= total_docs:
        print("Shard ID out of range. Nothing to do.")
        return

    # 2. Load Model
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = BiEncoder(Config.MODEL_NAME, Config.EMBEDDING_DIM, Config.POOLING)
    
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    
    # Enable DataParallel if multiple GPUs are available
    if num_gpus > 1:
        print(f"Enabling DataParallel on {num_gpus} GPUs")
        model = nn.DataParallel(model)
        
    model.eval()

    # 3. Prepare Data
    dataset = CorpusDataset(Config.CORPUS_FILE, tokenizer, start_idx=start_idx, end_idx=end_idx)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        collate_fn=collate_fn,
        pin_memory=True
    )

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"corpus_embeddings_shard_{args.shard_id}.jsonl")

    print(f"Generating embeddings to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        with torch.no_grad():
            for batch in tqdm(dataloader):
                if batch is None: continue
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                doc_ids = batch['doc_id']

                # Encode
                if num_gpus > 1:
                    # DataParallel wraps the model, so we call forward directly or access module
                    # BiEncoder.forward handles 'mode' argument
                    embeddings = model(doc_input_ids=input_ids, doc_attention_mask=attention_mask, mode='doc')
                else:
                    embeddings = model.encode_doc(input_ids, attention_mask)
                
                embeddings = embeddings.cpu().numpy()

                # Write to file
                for doc_id, emb in zip(doc_ids, embeddings):
                    record = {
                        "pmid": doc_id,
                        "embedding": emb.tolist()
                    }
                    f.write(json.dumps(record) + "\n")

    print(f"✅ Done! Shard {args.shard_id} saved to {output_file}")

if __name__ == "__main__":
    generate_embeddings()
