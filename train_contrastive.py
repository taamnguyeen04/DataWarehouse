"""
Optimized contrastive pretraining script for 11M papers
Supports streaming data loading, mixed precision, and large batch training
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from config import Config
import os
import time
import logging
from tqdm import tqdm


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamingContrastiveDataset(IterableDataset):
    """
    Streaming dataset for large corpus files
    Avoids loading entire dataset into memory
    """

    def __init__(self, corpus_file, tokenizer, max_length=512):
        self.corpus_file = corpus_file
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        with open(self.corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    paper = json.loads(line.strip())
                    title = paper.get('title', '').strip()
                    abstract = paper.get('text', '').strip()

                    # Skip if either title or abstract is empty or too short
                    if len(title) < 10 or len(abstract) < 50:
                        continue

                    # Tokenize
                    title_encoding = self.tokenizer(
                        title,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )

                    abstract_encoding = self.tokenizer(
                        abstract,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    )

                    yield {
                        'title_input_ids': title_encoding['input_ids'].squeeze(0),
                        'title_attention_mask': title_encoding['attention_mask'].squeeze(0),
                        'abstract_input_ids': abstract_encoding['input_ids'].squeeze(0),
                        'abstract_attention_mask': abstract_encoding['attention_mask'].squeeze(0),
                    }

                except (json.JSONDecodeError, KeyError) as e:
                    continue


class ContrastiveModel(nn.Module):
    """Optimized contrastive model with gradient checkpointing"""

    def __init__(self, projection_dim=768, temperature=0.07):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(Config.MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size

        # Enable gradient checkpointing for memory efficiency
        self.encoder.gradient_checkpointing_enable()

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim),
        )

        self.temperature = temperature

    def forward(self, input_ids, attention_mask):
        # Get [CLS] embeddings
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Project and normalize
        projections = self.projection_head(embeddings)
        projections = F.normalize(projections, p=2, dim=1)

        return projections

    def compute_contrastive_loss(self, title_embeddings, abstract_embeddings):
        """Compute InfoNCE loss with numerical stability"""
        batch_size = title_embeddings.shape[0]

        # Compute similarity matrix
        logits = torch.matmul(title_embeddings, abstract_embeddings.T) / self.temperature

        # Create labels
        labels = torch.arange(batch_size, device=logits.device)

        # Compute loss in both directions
        loss_t2a = F.cross_entropy(logits, labels)
        loss_a2t = F.cross_entropy(logits.T, labels)
        total_loss = (loss_t2a + loss_a2t) / 2

        # Compute accuracy
        with torch.no_grad():
            pred_t2a = torch.argmax(logits, dim=1)
            pred_a2t = torch.argmax(logits.T, dim=1)
            acc_t2a = (pred_t2a == labels).float().mean()
            acc_a2t = (pred_a2t == labels).float().mean()
            accuracy = (acc_t2a + acc_a2t) / 2

        return total_loss, accuracy, acc_t2a, acc_a2t


def create_dataloader(corpus_file, batch_size=512, max_length=512, num_workers=4):
    """Create optimized streaming dataloader"""
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    dataset = StreamingContrastiveDataset(
        corpus_file,
        tokenizer,
        max_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader, tokenizer


def train_contrastive_model(
    corpus_file=None,
    batch_size=512,
    learning_rate=1e-4,
    epochs=5,
    projection_dim=768,
    temperature=0.07,
    output_dir="./contrastive_checkpoints",
    mixed_precision=True,
    warmup_steps=1000,
    save_every=1000,
    eval_every=100,
    max_length=512,
    num_workers=4
):
    """Main training function with direct parameters"""

    # Use config defaults if not provided
    if corpus_file is None:
        corpus_file = Config.CORPUS_FILE

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training on device: {device}")

    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Auto-adjust batch size based on GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory >= 40:  # A100 40GB
            recommended_batch = 1024
        elif gpu_memory >= 24:  # RTX 4090, A6000
            recommended_batch = 512
        elif gpu_memory >= 16:  # RTX 4080
            recommended_batch = 256
        elif gpu_memory >= 12:  # RTX 3080Ti
            recommended_batch = 128
        else:
            recommended_batch = 64

        if batch_size > recommended_batch:
            logger.warning(f"Reducing batch size from {batch_size} to {recommended_batch}")
            batch_size = recommended_batch

    # Create model
    model = ContrastiveModel(projection_dim=projection_dim, temperature=temperature)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    # Setup mixed precision
    scaler = GradScaler() if mixed_precision else None

    # Create dataloader
    dataloader, tokenizer = create_dataloader(corpus_file, batch_size, max_length, num_workers)

    # Setup scheduler
    total_steps = epochs * 10000  # Estimate
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_info = {
        'corpus_file': corpus_file,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'projection_dim': projection_dim,
        'temperature': temperature,
        'max_length': max_length,
        'mixed_precision': mixed_precision,
        'model_name': Config.MODEL_NAME
    }

    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")

    # Training loop
    model.train()
    step = 0
    epoch = 0
    running_loss = 0
    running_accuracy = 0
    running_acc_t2a = 0
    running_acc_a2t = 0
    start_time = time.time()

    logger.info("=== Training Configuration ===")
    logger.info(f"Corpus: {corpus_file}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Projection dim: {projection_dim}")
    logger.info(f"Temperature: {temperature}")
    logger.info("=" * 30)

    try:
        while epoch < epochs:
            epoch_start_time = time.time()
            logger.info(f"Starting epoch {epoch + 1}/{epochs}")

            for batch in dataloader:
                # Move to device
                title_input_ids = batch['title_input_ids'].to(device, non_blocking=True)
                title_attention_mask = batch['title_attention_mask'].to(device, non_blocking=True)
                abstract_input_ids = batch['abstract_input_ids'].to(device, non_blocking=True)
                abstract_attention_mask = batch['abstract_attention_mask'].to(device, non_blocking=True)

                # Forward pass with mixed precision
                if mixed_precision:
                    with autocast():
                        title_embeddings = model(title_input_ids, title_attention_mask)
                        abstract_embeddings = model(abstract_input_ids, abstract_attention_mask)
                        loss, accuracy, acc_t2a, acc_a2t = model.compute_contrastive_loss(
                            title_embeddings, abstract_embeddings
                        )
                else:
                    title_embeddings = model(title_input_ids, title_attention_mask)
                    abstract_embeddings = model(abstract_input_ids, abstract_attention_mask)
                    loss, accuracy, acc_t2a, acc_a2t = model.compute_contrastive_loss(
                        title_embeddings, abstract_embeddings
                    )

                # Backward pass
                optimizer.zero_grad()

                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                scheduler.step()

                # Update metrics
                running_loss += loss.item()
                running_accuracy += accuracy.item()
                running_acc_t2a += acc_t2a.item()
                running_acc_a2t += acc_a2t.item()
                step += 1

                # Logging
                if step % eval_every == 0:
                    avg_loss = running_loss / eval_every
                    avg_accuracy = running_accuracy / eval_every
                    avg_acc_t2a = running_acc_t2a / eval_every
                    avg_acc_a2t = running_acc_a2t / eval_every

                    elapsed = time.time() - start_time
                    speed = step / elapsed * batch_size

                    logger.info(
                        f"Step {step:6d} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"Acc: {avg_accuracy:.4f} | "
                        f"T2A: {avg_acc_t2a:.4f} | "
                        f"A2T: {avg_acc_a2t:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                        f"Speed: {speed:.0f} samples/s"
                    )

                    running_loss = 0
                    running_accuracy = 0
                    running_acc_t2a = 0
                    running_acc_a2t = 0

                # Save checkpoint
                if step % save_every == 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
                    torch.save({
                        'step': step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                        'accuracy': accuracy.item(),
                        'config': config_info
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint: {checkpoint_path}")

                # Memory cleanup
                if step % 100 == 0:
                    torch.cuda.empty_cache()

            # End of epoch
            epoch += 1
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Completed epoch {epoch} in {epoch_time:.1f}s")

            # Save epoch checkpoint
            epoch_path = os.path.join(output_dir, f"epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config_info
            }, epoch_path)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    # Save final model
    final_path = os.path.join(output_dir, "final_model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config_info,
        'total_steps': step
    }, final_path)

    logger.info(f"Training completed! Final model saved to: {final_path}")
    return model


def main():
    """Main function"""

    # Configuration parameters
    corpus_file = Config.CORPUS_FILE
    batch_size = 512  # Large batch size for contrastive learning
    learning_rate = 1e-4
    epochs = 5
    projection_dim = 768  # Same as PubMedBERT hidden size
    temperature = 0.07  # Standard for contrastive learning

    # Train model with direct parameters
    model = train_contrastive_model(
        corpus_file=corpus_file,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        projection_dim=projection_dim,
        temperature=temperature,
        mixed_precision=True,
        warmup_steps=1000,
        save_every=1000,
        eval_every=100
    )


if __name__ == "__main__":
    main()