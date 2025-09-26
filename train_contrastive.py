import torch
from torch.utils.data import DataLoader, IterableDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from config import Config
import os
from tqdm import tqdm
from model import ContrastiveModel
from data_loader import StreamingContrastiveDataset


def train():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus_file = Config.CORPUS_FILE
    batch_size = 8
    learning_rate = 1e-4
    epochs = 5
    projection_dim = 768
    temperature = 0.07
    output_dir = "./contrastive_checkpoints"
    mixed_precision = True
    warmup_steps = 1000
    save_every = 1000
    max_length = 512
    num_workers = 4

    os.makedirs(output_dir, exist_ok=True)

    model = ContrastiveModel(projection_dim=projection_dim, temperature=temperature)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )

    scaler = GradScaler() if mixed_precision else None

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    dataset = StreamingContrastiveDataset(corpus_file, tokenizer, max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    total_steps = epochs * 10000
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

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

    best_acc = 0.0

    try:
        for epoch in range(epochs):
            model.train()

            all_losses = []
            all_accuracies = []

            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), colour="BLUE", desc=f"Epoch {epoch + 1}/{epochs}")

            for i, batch in progress_bar:
                title_input_ids = batch['title_input_ids'].to(device)
                title_attention_mask = batch['title_attention_mask'].to(device)
                abstract_input_ids = batch['abstract_input_ids'].to(device)
                abstract_attention_mask = batch['abstract_attention_mask'].to(device)

                if mixed_precision:
                    with autocast():
                        title_embeddings = model(title_input_ids, title_attention_mask)
                        abstract_embeddings = model(abstract_input_ids, abstract_attention_mask)
                        loss, accuracy = model.compute_contrastive_loss(title_embeddings, abstract_embeddings)
                else:
                    title_embeddings = model(title_input_ids, title_attention_mask)
                    abstract_embeddings = model(abstract_input_ids, abstract_attention_mask)
                    loss, accuracy = model.compute_contrastive_loss(title_embeddings, abstract_embeddings)

                progress_bar.set_description(
                    f"Epoch {epoch + 1}/{epochs}. Loss {loss:.4f}. Acc {accuracy:.4f}"
                )

                optimizer.zero_grad()

                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                scheduler.step()

                all_losses.append(loss.item())
                all_accuracies.append(accuracy.item())

                if i % save_every == 0 and i > 0:
                    checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{i}.pt")
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "step": i,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.item(),
                        "accuracy": accuracy.item()
                    }
                    torch.save(checkpoint, checkpoint_path)

            avg_epoch_loss = np.mean(all_losses)
            avg_epoch_acc = np.mean(all_accuracies)

            print(f"Epoch {epoch + 1}/{epochs}. Loss {avg_epoch_loss:.4f}. Acc {avg_epoch_acc:.4f}")

            if avg_epoch_acc > best_acc:
                best_checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "accuracy": avg_epoch_acc
                }
                torch.save(best_checkpoint, os.path.join(output_dir, "best.pt"))
                best_acc = avg_epoch_acc
                print(f"New best model saved! Accuracy: {best_acc:.4f}")

            epoch_checkpoint = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_epoch_loss,
                "avg_accuracy": avg_epoch_acc
            }
            torch.save(epoch_checkpoint, os.path.join(output_dir, "last.pt"))

    except KeyboardInterrupt:
        print("Training interrupted by user")

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_info,
        "best_accuracy": best_acc
    }
    torch.save(final_checkpoint, os.path.join(output_dir, "final.pt"))

    print(f"Training completed! Best accuracy: {best_acc:.4f}")
    return model


if __name__ == "__main__":
    train()