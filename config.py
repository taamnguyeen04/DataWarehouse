import os

class Config:
    # Paths
    DATA_DIR = "C:/Users/tam/Documents/data/Data Warehouse"
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
    MAX_LENGTH = 512
    EMBEDDING_DIM = 768
    POOLING = "mean"  # 'cls', 'mean', or 'max'

    # Training Hyperparameters
    BATCH_SIZE = 2  # Large batch for in-batch negatives
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
    CHECKPOINT_DIR = "./PAR/checkpoints"
    LOG_DIR = "./PAR/logs"
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

