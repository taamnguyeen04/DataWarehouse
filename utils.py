import json
import pickle
import numpy as np
from typing import List, Dict, Any

def save_embeddings(embeddings: np.ndarray, filepath: str):
    """Save embeddings to file"""
    np.save(filepath, embeddings)

def load_embeddings(filepath: str) -> np.ndarray:
    """Load embeddings from file"""
    return np.load(filepath)

def save_json(data: Dict[str, Any], filepath: str):
    """Save dictionary to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_mesh_vocabulary(mesh_labels: Dict[str, List[str]]) -> Dict[str, int]:
    """Create MeSH term to index mapping"""
    all_mesh_terms = set()
    for mesh_list in mesh_labels.values():
        all_mesh_terms.update(mesh_list)
    
    return {term: idx for idx, term in enumerate(sorted(all_mesh_terms))}

def create_mesh_multihot(mesh_terms: List[str], mesh_vocab: Dict[str, int]) -> np.ndarray:
    """Convert MeSH terms to multi-hot encoding"""
    vector = np.zeros(len(mesh_vocab))
    for term in mesh_terms:
        if term in mesh_vocab:
            vector[mesh_vocab[term]] = 1
    return vector

def evaluate_retrieval(predicted_pmids: List[str], relevant_pmids: List[str]) -> Dict[str, float]:
    """Evaluate retrieval performance"""
    predicted_set = set(predicted_pmids)
    relevant_set = set(relevant_pmids)
    
    if len(relevant_set) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    true_positives = len(predicted_set & relevant_set)
    
    precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0.0
    recall = true_positives / len(relevant_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

class ProgressTracker:
    """Simple progress tracking for training"""
    def __init__(self):
        self.metrics = {
            'train_loss': [],
            'val_accuracy': [],
            'val_f1': []
        }
    
    def update(self, train_loss: float, val_accuracy: float, val_f1: float):
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_accuracy'].append(val_accuracy)
        self.metrics['val_f1'].append(val_f1)
    
    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.metrics, f)
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.metrics = pickle.load(f)