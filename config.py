import os

class Config:
    # Paths
    DATA_DIR = "C:/Users/tam/Desktop/Data/Data Warehouse"
    CORPUS_FILE = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/corpus.jsonl")
    CORPUS_INDEX_FILE = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/corpus_index.pkl")
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
    PAIRS_TRAIN_FILE = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/gold/pairs_train.jsonl")

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
    MILVUS_HOST = "127.0.0.1"
    # MILVUS_PORT = "19530"
    MILVUS_HOST_SERVER = "0.tcp.ap.ngrok.io"
    MILVUS_PORT_SERVER_1 = "13292"
    MILVUS_PORT_SERVER_2 = "17358"
    # MILVUS_HOST = "100.98.10.24"
    MILVUS_PORT = "19530"
    MILVUS_USER = "root"
    MILVUS_URI = "https://in05-0c7fef863ca43c4.serverless.aws-eu-central-1.cloud.zilliz.com"
    MILVUS_TOKEN = "5c9300395f1ed7bb3ca8d90b913eb87fe72666c8f1ed941b86a22bf3a33c33604e24f7168a89f68ff3e3b0bb609465dcc48eaf60"
    MILVUS_PASSWORD = "aiostorm"
    COLLECTION_NAME = "pmc_papers_v2"

    # Tasks weights
    CLASSIFICATION_WEIGHT = 1.0
    LINK_PREDICTION_WEIGHT = 1.0
    RETRIEVAL_WEIGHT = 1.0

