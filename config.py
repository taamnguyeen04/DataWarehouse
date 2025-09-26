import os

class Config:
    # Paths
    DATA_DIR = "C:/Users/tam/Documents/data/Data Warehouse"
    CORPUS_FILE = os.path.join(DATA_DIR, "ReCDS_benchmark/PAR/corpus.jsonl")
    PMID_MESH_FILE = os.path.join(DATA_DIR, "meta_data/PMID2MeSH.json")
    PATIENTS_FILE = os.path.join(DATA_DIR, "PMC-Patients.json")
    RELEVANCE_FILE = os.path.join(DATA_DIR, "patient2article_relevance.json")
    
    # Model
    MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    
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

