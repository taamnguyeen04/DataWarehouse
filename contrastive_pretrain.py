import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModel
from config import Config
import random
import os
from torch.utils.tensorboard import SummaryWriter
import time

class ContrastivePairDataset(Dataset):
    """Dataset for contrastive pretraining using (title, abstract) pairs"""
    def __init__(self, corpus_file, max_samples=None):
        self.corpus_file = corpus_file
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.max_length = Config.MAX_LENGTH

        self.data = []
        count = 0

        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break

                paper = json.loads(line.strip())
                title = paper.get('title', '').strip()
                abstract = paper.get('text', '').strip()

                if title and abstract:
                    self.data.append({
                        'title': title,
                        'abstract': abstract,
                        'pmid': paper.get('_id', str(count))
                    })
                count += 1

                if count % 100000 == 0:
                    print(f"Loaded {count} papers, kept {len(self.data)} valid pairs")

        print(f"Final dataset size: {len(self.data)} (title, abstract) pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        title_encoding = self.tokenizer(
            item['title'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        abstract_encoding = self.tokenizer(
            item['abstract'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'title_input_ids': title_encoding['input_ids'].squeeze(0),
            'title_attention_mask': title_encoding['attention_mask'].squeeze(0),
            'abstract_input_ids': abstract_encoding['input_ids'].squeeze(0),
            'abstract_attention_mask': abstract_encoding['attention_mask'].squeeze(0),
            'pmid': item['pmid']
        }


class ContrastiveModel(nn.Module):
    """Contrastive learning model with PubMedBERT backbone"""
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
        """Compute InfoNCE/SimCLR loss"""
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


def create_contrastive_dataloader(corpus_file, batch_size=128, max_samples=None, num_workers=4):
    """Create DataLoader for contrastive pretraining"""
    dataset = ContrastivePairDataset(corpus_file, max_samples)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader


def evaluate_embeddings(model, device, num_samples=1000):
    """Evaluate embedding quality using similarity metrics"""
    model.eval()
    eval_dataloader = create_contrastive_dataloader(
        corpus_file=Config.CORPUS_FILE,
        batch_size=64,
        max_samples=num_samples
    )

    title_embeddings = []
    abstract_embeddings = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            if batch_idx * eval_dataloader.batch_size >= num_samples:
                break

            title_input_ids = batch['title_input_ids'].to(device)
            title_attention_mask = batch['title_attention_mask'].to(device)
            abstract_input_ids = batch['abstract_input_ids'].to(device)
            abstract_attention_mask = batch['abstract_attention_mask'].to(device)

            title_emb = model(title_input_ids, title_attention_mask)
            abstract_emb = model(abstract_input_ids, abstract_attention_mask)

            title_embeddings.append(title_emb.cpu())
            abstract_embeddings.append(abstract_emb.cpu())

    title_embeddings = torch.cat(title_embeddings, dim=0)
    abstract_embeddings = torch.cat(abstract_embeddings, dim=0)

    # Compute similarity metrics
    similarities = torch.cosine_similarity(title_embeddings, abstract_embeddings, dim=1)

    # Compute retrieval accuracy
    similarity_matrix = torch.matmul(title_embeddings, abstract_embeddings.T)
    top1_correct = (torch.argmax(similarity_matrix, dim=1) == torch.arange(len(similarity_matrix))).float().mean()
    top5_correct = torch.topk(similarity_matrix, k=5, dim=1).indices
    top5_acc = (top5_correct == torch.arange(len(similarity_matrix)).unsqueeze(1)).any(dim=1).float().mean()

    return {
        'mean_similarity': similarities.mean().item(),
        'std_similarity': similarities.std().item(),
        'top1_accuracy': top1_correct.item(),
        'top5_accuracy': top5_acc.item()
    }


def train():
    corpus_file = Config.CORPUS_FILE
    output_dir = "./contrastive_checkpoints"
    batch_size = 256
    learning_rate = 1e-4
    epochs = 10
    max_samples = None
    save_every = 1000
    eval_every = 100
    eval_epochs = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model = ContrastiveModel(projection_dim=768, temperature=0.07)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    dataloader = create_contrastive_dataloader(
        corpus_file,
        batch_size=batch_size,
        max_samples=max_samples
    )

    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard_logs'))

    hparams = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'projection_dim': 768,
        'temperature': 0.07
    }
    writer.add_hparams(hparams, {})

    model.train()
    step = 0
    global_step = 0
    total_loss = 0
    total_accuracy = 0
    start_time = time.time()


    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs}")

        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            title_input_ids = batch['title_input_ids'].to(device)
            title_attention_mask = batch['title_attention_mask'].to(device)
            abstract_input_ids = batch['abstract_input_ids'].to(device)
            abstract_attention_mask = batch['abstract_attention_mask'].to(device)

            title_embeddings = model(title_input_ids, title_attention_mask)
            abstract_embeddings = model(abstract_input_ids, abstract_attention_mask)

            loss, accuracy = model.compute_contrastive_loss(title_embeddings, abstract_embeddings)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_accuracy += accuracy.item()
            epoch_loss += loss.item()
            epoch_acc += accuracy.item()
            step += 1
            global_step += 1
            num_batches += 1

            writer.add_scalar('Training/Loss_Step', loss.item(), global_step)
            writer.add_scalar('Training/Accuracy_Step', accuracy.item(), global_step)
            writer.add_scalar('Training/Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

            if step % eval_every == 0:
                avg_loss = total_loss / eval_every
                avg_acc = total_accuracy / eval_every

                elapsed = time.time() - start_time
                speed = step * batch_size / elapsed

                print(f"Step {step}: Loss = {avg_loss:.4f}, Accuracy = {avg_acc:.4f}, Speed = {speed:.0f} samples/s")

                writer.add_scalar('Training/Loss_Avg', avg_loss, global_step)
                writer.add_scalar('Training/Accuracy_Avg', avg_acc, global_step)
                writer.add_scalar('Training/Speed', speed, global_step)

                total_loss = 0
                total_accuracy = 0

            if step % save_every == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'accuracy': accuracy.item()
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        avg_epoch_loss = epoch_loss / num_batches
        avg_epoch_acc = epoch_acc / num_batches
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"Epoch {epoch + 1} - Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_acc:.4f}")

        writer.add_scalar('Training/Epoch_Loss', avg_epoch_loss, epoch + 1)
        writer.add_scalar('Training/Epoch_Accuracy', avg_epoch_acc, epoch + 1)
        writer.add_scalar('Training/Epoch_Time', epoch_time, epoch + 1)

        if (epoch + 1) % eval_epochs == 0:
            print(f"Running evaluation at epoch {epoch + 1}...")
            eval_metrics = evaluate_embeddings(model, device, num_samples=2000)
            writer.add_scalar('Evaluation/Mean_Similarity', eval_metrics['mean_similarity'], epoch + 1)
            writer.add_scalar('Evaluation/Std_Similarity', eval_metrics['std_similarity'], epoch + 1)
            writer.add_scalar('Evaluation/Top1_Accuracy', eval_metrics['top1_accuracy'], epoch + 1)
            writer.add_scalar('Evaluation/Top5_Accuracy', eval_metrics['top5_accuracy'], epoch + 1)

            model.train()

        epoch_path = os.path.join(output_dir, f"epoch_{epoch + 1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_metrics': eval_metrics if (epoch + 1) % eval_epochs == 0 else None
        }, epoch_path)
        print(f"Saved epoch checkpoint: {epoch_path}")

    final_eval = evaluate_embeddings(model, device, num_samples=5000)
    writer.add_scalar('Final/Mean_Similarity', final_eval['mean_similarity'], epochs)
    writer.add_scalar('Final/Top1_Accuracy', final_eval['top1_accuracy'], epochs)
    writer.add_scalar('Final/Top5_Accuracy', final_eval['top5_accuracy'], epochs)

    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_eval': final_eval,
        'hyperparameters': hparams,
        'total_steps': step
    }, final_path)
    print(f"Final model saved: {final_path}")

    writer.close()
    return model


if __name__ == "__main__":
    trained_model = train()
