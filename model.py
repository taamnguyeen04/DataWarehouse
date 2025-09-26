import torch
import torch.nn as nn
from transformers import AutoModel
from config import Config

class MultiTaskPARModel(nn.Module):
    def __init__(self, num_mesh_classes, num_patients):
        super().__init__()
        
        # PubMedBERT backbone
        self.backbone = AutoModel.from_pretrained(Config.MODEL_NAME)
        hidden_size = self.backbone.config.hidden_size
        
        # Task heads
        self.classification_head = nn.Linear(hidden_size, num_mesh_classes)
        self.link_prediction_head = nn.Linear(hidden_size * 2, 1)  # paper + patient embeddings
        self.retrieval_head = nn.Linear(hidden_size, hidden_size)  # for similarity computation
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask, task='classification'):
        # Get paper embeddings
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        paper_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        paper_embeddings = self.dropout(paper_embeddings)
        
        if task == 'classification':
            return self.classification_head(paper_embeddings)
        
        elif task == 'link_prediction':
            # Cần thêm patient embeddings ở đây
            return self.link_prediction_head(paper_embeddings)
        
        elif task == 'retrieval':
            return self.retrieval_head(paper_embeddings)
        
        else:
            return paper_embeddings  # Raw embeddings