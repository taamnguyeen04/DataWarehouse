import json
import random
import torch
import os
import logging
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample, losses, evaluation
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from config import Config
import math

# Set up logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(queries_file, qrels_file, corpus_file):
    logger.info("Loading queries...")
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            query = json.loads(line.strip())
            queries[query['_id']] = query['text']

    logger.info("Loading corpus...")
    corpus = {}
    # We only load documents that are in qrels to save memory, 
    # but for negative sampling we might need more. 
    # For simplicity in this script, we'll load the needed ones + some random ones if possible,
    # or just load the ones referenced in qrels and pick negatives from there.
    # Better approach for large corpus: Lazy load or use the existing PARDataset logic if adapted.
    # Here we will load all referenced docs first.
    
    needed_doc_ids = set()
    qrels = {}
    
    logger.info("Loading qrels...")
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                query_id, doc_id, relevance = parts
                try:
                    if int(relevance) > 0:
                        if query_id not in qrels:
                            qrels[query_id] = []
                        qrels[query_id].append(doc_id)
                        needed_doc_ids.add(doc_id)
                except ValueError:
                    continue

    logger.info(f"Loading {len(needed_doc_ids)} documents from corpus...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line.strip())
            doc_id = str(doc.get('_id', ''))
            if doc_id in needed_doc_ids:
                corpus[doc_id] = f"{doc.get('title', '')} {doc.get('text', '')}".strip()
    
    return queries, qrels, corpus

def create_training_examples(queries, qrels, corpus, num_negatives=4):
    train_samples = []
    logger.info("Creating training examples...")
    
    doc_ids = list(corpus.keys())
    
    for query_id, pos_doc_ids in qrels.items():
        if query_id not in queries:
            continue
            
        query_text = queries[query_id]
        
        for pos_doc_id in pos_doc_ids:
            if pos_doc_id not in corpus:
                continue
                
            # Positive example
            pos_doc_text = corpus[pos_doc_id]
            train_samples.append(InputExample(texts=[query_text, pos_doc_text], label=1.0))
            
            # Negative examples
            # Random sampling from the loaded corpus (which contains relevant docs for other queries)
            # This is "in-batch" style negative sampling but explicit
            for _ in range(num_negatives):
                neg_doc_id = random.choice(doc_ids)
                # Ensure negative is not a positive
                while neg_doc_id in pos_doc_ids:
                    neg_doc_id = random.choice(doc_ids)
                
                neg_doc_text = corpus[neg_doc_id]
                train_samples.append(InputExample(texts=[query_text, neg_doc_text], label=0.0))
                
    return train_samples

def main():
    # Configuration
    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    train_batch_size = 8
    num_epochs = 4
    model_save_path = 'output/cross-encoder-pubmedbert'
    
    # Load data
    queries, qrels, corpus = load_data(Config.TRAIN_QUERIES, Config.TRAIN_QRELS, Config.CORPUS_FILE)
    
    # Create examples
    train_samples = create_training_examples(queries, qrels, corpus, num_negatives=4)
    logger.info(f"Created {len(train_samples)} training samples")
    
    # DataLoader
    # num_workers=4 để load dữ liệu đa luồng, pin_memory=True để chuyển dữ liệu sang GPU nhanh hơn
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, num_workers=4, pin_memory=True)
    
    # Initialize CrossEncoder
    logger.info(f"Initializing CrossEncoder with {model_name}")
    # Thêm max_length=512 để tự động cắt ngắn các câu quá dài, tránh lỗi indexing
    model = CrossEncoder(model_name, num_labels=1, max_length=512)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        # Chỉ wrap phần transformer model bên trong, không wrap cả CrossEncoder
        # Tuy nhiên, CrossEncoder của sentence-transformers không hỗ trợ DataParallel trực tiếp theo cách này
        # Cách tốt nhất là để Trainer tự xử lý hoặc dùng accelerator nếu dùng bản mới.
        # Nhưng để fix lỗi attribute 'config' bị ẩn, ta tạm thời bỏ DataParallel thủ công
        # vì sentence-transformers v3 tự hỗ trợ multi-gpu khi train.
        pass 
        
    # Load dev data for evaluation
    logger.info("Loading dev data for evaluation...")
    dev_queries, dev_qrels, dev_corpus = load_data(Config.DEV_QUERIES, Config.DEV_QRELS, Config.CORPUS_FILE)
    
    # Create dev examples
    # For evaluation, we can use CEBinaryClassificationEvaluator which expects InputExample
    dev_samples = create_training_examples(dev_queries, dev_qrels, dev_corpus, num_negatives=1) # Less negatives for dev to save time
    logger.info(f"Created {len(dev_samples)} dev samples")
    
    # Initialize Evaluator
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='dev')

    # Train
    logger.info("Starting training...")
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) # 10% of train data for warm-up
    
    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              evaluation_steps=1000, # Evaluate every 1000 steps
              save_best_model=True,
              show_progress_bar=True)
              
    logger.info(f"Training finished. Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
