import torch
import torch.nn as nn
import torch.nn.functional as F
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
            return paper_embeddings


class ContrastiveModel(nn.Module):
    def __init__(self, projection_dim=768, temperature=0.07):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(Config.MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim),
        )
        self.temperature = temperature

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]

        projections = self.projection_head(embeddings)
        projections = F.normalize(projections, p=2, dim=1)
        return projections

    def compute_contrastive_loss(self, title_embeddings, abstract_embeddings):
        batch_size = title_embeddings.shape[0]

        logits = torch.matmul(title_embeddings, abstract_embeddings.T) / self.temperature

        labels = torch.arange(batch_size, device=logits.device)

        loss_title_to_abstract = F.cross_entropy(logits, labels)
        loss_abstract_to_title = F.cross_entropy(logits.T, labels)
        total_loss = (loss_title_to_abstract + loss_abstract_to_title) / 2

        pred_title = torch.argmax(logits, dim=1)
        pred_abstract = torch.argmax(logits.T, dim=1)
        acc_title = (pred_title == labels).float().mean()
        acc_abstract = (pred_abstract == labels).float().mean()
        accuracy = (acc_title + acc_abstract) / 2

        return total_loss, accuracy