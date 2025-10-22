import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class BiEncoder(nn.Module):
    def __init__(self, model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 embedding_dim=768, pooling='cls'):
        super(BiEncoder, self).__init__()

        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.doc_encoder = AutoModel.from_pretrained(model_name)

        self.pooling = pooling
        self.embedding_dim = embedding_dim

        # Optional projection layer (uncomment if you want different embedding dim)
        # self.projection = nn.Linear(768, embedding_dim)

    def pool_embeddings(self, last_hidden_state, attention_mask):
        if self.pooling == 'cls':
            # Use [CLS] token embedding
            return last_hidden_state[:, 0, :]

        elif self.pooling == 'mean':
            # Mean pooling with attention mask
            token_embeddings = last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        elif self.pooling == 'max':
            # Max pooling
            token_embeddings = last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9
            return torch.max(token_embeddings, 1)[0]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def encode_query(self, input_ids, attention_mask):
        # Encode query (patient summary)
        outputs = self.query_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        pooled = self.pool_embeddings(outputs.last_hidden_state, attention_mask)

        # Optional: apply projection
        # pooled = self.projection(pooled)

        # L2 normalization for cosine similarity
        return F.normalize(pooled, p=2, dim=1)

    def encode_doc(self, input_ids, attention_mask):
        # Encode document (article title + abstract)
        outputs = self.doc_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        pooled = self.pool_embeddings(outputs.last_hidden_state, attention_mask)

        # Optional: apply projection
        # pooled = self.projection(pooled)

        # L2 normalization for cosine similarity
        return F.normalize(pooled, p=2, dim=1)

    def forward(self, query_input_ids=None, query_attention_mask=None,
                doc_input_ids=None, doc_attention_mask=None, mode='dual'):
        """
        Forward với multiple modes để support DataParallel
        mode='dual': training mode, trả về (query_emb, doc_emb)
        mode='doc': inference mode, chỉ encode documents
        mode='query': inference mode, chỉ encode queries
        """
        if mode == 'dual':
            query_embeddings = self.encode_query(query_input_ids, query_attention_mask)
            doc_embeddings = self.encode_doc(doc_input_ids, doc_attention_mask)
            return query_embeddings, doc_embeddings
        elif mode == 'doc':
            return self.encode_doc(doc_input_ids, doc_attention_mask)
        elif mode == 'query':
            return self.encode_query(query_input_ids, query_attention_mask)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, query_embeddings, doc_embeddings):
        batch_size = query_embeddings.size(0)

        similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T) / self.temperature

        labels = torch.arange(batch_size, device=query_embeddings.device)

        loss_q2d = self.criterion(similarity_matrix, labels)
        loss_d2q = self.criterion(similarity_matrix.T, labels)

        loss = (loss_q2d + loss_d2q) / 2.0

        return loss


if __name__ == "__main__":
    model = BiEncoder()
    batch_size = 4
    seq_len = 128

    query_input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    query_attention_mask = torch.ones(batch_size, seq_len)
    doc_input_ids = torch.randint(0, 30000, (batch_size, seq_len))
    doc_attention_mask = torch.ones(batch_size, seq_len)

    query_emb, doc_emb = model(query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask)

    print(f"Query embeddings shape: {query_emb.shape}")
    print(f"Document embeddings shape: {doc_emb.shape}")

    # Test loss
    loss_fn = InfoNCELoss()
    loss = loss_fn(query_emb, doc_emb)
    print(f"InfoNCE loss: {loss.item()}")