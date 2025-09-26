import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer
from config import Config
import os
from torch.utils.tensorboard import SummaryWriter
import time
from tqdm.autonotebook import tqdm
from model import ContrastiveModel
from data_loader import StreamingContrastiveDataset
import warnings

warnings.filterwarnings("ignore")



def create_contrastive_dataloader(corpus_file, batch_size=128, num_workers=4):
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    dataset = StreamingContrastiveDataset(corpus_file, tokenizer, max_length=512)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader


def evaluate_embeddings(model, device, num_samples=1000):
    model.eval()
    eval_dataloader = create_contrastive_dataloader( corpus_file=Config.CORPUS_FILE, batch_size=64)

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

    similarities = torch.cosine_similarity(title_embeddings, abstract_embeddings, dim=1)

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
    batch_size = 8
    learning_rate = 1e-4
    epochs = 10
    max_samples = None  # None for full 11M dataset
    projection_dim = 768
    temperature = 0.07
    save_every = 1000
    eval_epochs = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = ContrastiveModel(projection_dim=projection_dim, temperature=temperature)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    dataloader = create_contrastive_dataloader(corpus_file, batch_size=batch_size)

    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard_logs'))

    hparams = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'projection_dim': projection_dim,
        'temperature': temperature
    }
    writer.add_hparams(hparams, {})

    best_acc = 0.0
    step = 0
    global_step = 0

    # Start training
    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()

        all_losses = []
        all_accuracies = []

        progress_bar = tqdm(dataloader, colour="BLUE")

        for i, batch in enumerate(progress_bar):
            title_input_ids = batch['title_input_ids'].to(device)
            title_attention_mask = batch['title_attention_mask'].to(device)
            abstract_input_ids = batch['abstract_input_ids'].to(device)
            abstract_attention_mask = batch['abstract_attention_mask'].to(device)

            title_embeddings = model(title_input_ids, title_attention_mask)
            abstract_embeddings = model(abstract_input_ids, abstract_attention_mask)
            loss, accuracy = model.compute_contrastive_loss(title_embeddings, abstract_embeddings)

            progress_bar.set_description(
                f"Epoch {epoch + 1}/{epochs}. Loss {loss:.4f}. Acc {accuracy:.4f}"
            )

            writer.add_scalar("Train/loss", loss, global_step)
            writer.add_scalar("Train/accuracy", accuracy, global_step)
            writer.add_scalar("Train/learning_rate", optimizer.param_groups[0]['lr'], global_step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            all_losses.append(loss.item())
            all_accuracies.append(accuracy.item())
            step += 1
            global_step += 1

            if step % save_every == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{step}.pt")
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "step": step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                    "accuracy": accuracy.item()
                }
                torch.save(checkpoint, checkpoint_path)

        avg_epoch_loss = np.mean(all_losses)
        avg_epoch_acc = np.mean(all_accuracies)
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch + 1}/{epochs}. Loss {avg_epoch_loss:.4f}. Acc {avg_epoch_acc:.4f}. Time {epoch_time:.1f}s")

        writer.add_scalar("Train/epoch_loss", avg_epoch_loss, epoch)
        writer.add_scalar("Train/epoch_accuracy", avg_epoch_acc, epoch)
        writer.add_scalar("Train/epoch_time", epoch_time, epoch)

        if (epoch + 1) % eval_epochs == 0:
            print(f"Running evaluation at epoch {epoch + 1}...")
            eval_metrics = evaluate_embeddings(model, device, num_samples=2000)

            print(f"Evaluation Results (Epoch {epoch + 1}):")
            print(f"  Mean similarity: {eval_metrics['mean_similarity']:.4f}")
            print(f"  Std similarity: {eval_metrics['std_similarity']:.4f}")
            print(f"  Top-1 accuracy: {eval_metrics['top1_accuracy']:.4f}")
            print(f"  Top-5 accuracy: {eval_metrics['top5_accuracy']:.4f}")

            # Log evaluation metrics
            writer.add_scalar('Eval/mean_similarity', eval_metrics['mean_similarity'], epoch)
            writer.add_scalar('Eval/std_similarity', eval_metrics['std_similarity'], epoch)
            writer.add_scalar('Eval/top1_accuracy', eval_metrics['top1_accuracy'], epoch)
            writer.add_scalar('Eval/top5_accuracy', eval_metrics['top5_accuracy'], epoch)

            if eval_metrics['top1_accuracy'] > best_acc:
                best_checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_metrics": eval_metrics
                }
                torch.save(best_checkpoint, os.path.join(output_dir, "best.pt"))
                best_acc = eval_metrics['top1_accuracy']
                print(f"New best model saved! Top-1 accuracy: {best_acc:.4f}")

        epoch_checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "avg_loss": avg_epoch_loss,
            "avg_accuracy": avg_epoch_acc
        }
        torch.save(epoch_checkpoint, os.path.join(output_dir, "last.pt"))

if __name__ == "__main__":
    train()

