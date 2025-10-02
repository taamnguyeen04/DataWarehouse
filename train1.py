import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
import json
import shutil
import warnings
warnings.filterwarnings("ignore")

from model import BiEncoder, InfoNCELoss
from data_loader import PARDataset
from config import Config


def save_checkpoint(filepath, epoch, step, model, optimizer, loss):
    print(f"Đang lưu checkpoint: epoch {epoch}, step {step}, loss {loss:.4f}")
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    last_path = os.path.join(filepath, "last_model.pt")
    torch.save(checkpoint, last_path)
    best_path = os.path.join(filepath, "best_model.pt")

    if not os.path.exists(best_path):
        torch.save(checkpoint, best_path)
        print("Đã lưu best_model.pt (lần đầu)")
    else:
        try:
            best_loss = torch.load(best_path, map_location='cpu')['loss']
            if loss < best_loss:
                torch.save(checkpoint, best_path)
                print(f"Đã cập nhật best_model.pt: {best_loss:.4f} → {loss:.4f}")
        except Exception as e:
            print(f"Lỗi khi đọc best_model.pt: {e} → lưu đè")
            torch.save(checkpoint, best_path)


def load_checkpoint(filepath, model, optimizer, device):
    last_path = os.path.join(filepath, "last_model.pt")
    best_path = os.path.join(filepath, "best_model.pt")

    start_epoch = 0
    start_step = 0
    best_loss = float('inf')
    loaded = False

    if os.path.isfile(last_path):
        try:
            checkpoint = torch.load(last_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            start_step = checkpoint.get('step', 0)
            loaded = True
            print(f"Loaded từ last_model.pt epoch {checkpoint['epoch']}")
        except Exception as e:
            print(f"Lỗi khi load last_model.pt: {e}")

    if not loaded and os.path.isfile(best_path):
        try:
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            start_step = checkpoint.get('step', 0)
            print(f"Loaded từ best_model.pt epoch {checkpoint['epoch']}")
            loaded = True
        except Exception as e:
            print(f"Lỗi khi load best_model.pt: {e}")

    if not loaded:
        print("Không tìm thấy hoặc không load được checkpoint, bắt đầu từ đầu.")
        start_epoch = 0
        start_step = 0

    if os.path.isfile(best_path):
        try:
            best_loss = torch.load(best_path, map_location=device)['loss']
        except:
            best_loss = float('inf')
    else:
        best_loss = float('inf')

    return start_epoch, start_step, best_loss


def train():
    batch_size = Config.BATCH_SIZE
    lr = Config.LEARNING_RATE
    epochs = Config.NUM_EPOCHS
    weight_decay = Config.WEIGHT_DECAY
    warmup_ratio = Config.WARMUP_RATIO
    max_grad_norm = Config.MAX_GRAD_NORM
    temperature = Config.TEMPERATURE
    log_path = Config.LOG_DIR
    checkpoint_path = Config.CHECKPOINT_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = BiEncoder(
        model_name=Config.MODEL_NAME,
        embedding_dim=Config.EMBEDDING_DIM,
        pooling=Config.POOLING
    ).to(device)

    train_dataset = PARDataset(
        queries_file=Config.TRAIN_QUERIES,
        qrels_file=Config.TRAIN_QRELS,
        corpus_file=Config.CORPUS_FILE,
        tokenizer=tokenizer,
        max_length=Config.MAX_LENGTH
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        shuffle=True,
        drop_last=False,
        pin_memory=True if device.type == 'cuda' else False
    )

    dev_dataset = PARDataset(
        queries_file=Config.DEV_QUERIES,
        qrels_file=Config.DEV_QRELS,
        corpus_file=Config.CORPUS_FILE,
        tokenizer=tokenizer,
        max_length=Config.MAX_LENGTH
    )
    dev_dataloader = DataLoader(
        dataset=dev_dataset,
        batch_size=batch_size,
        num_workers=Config.NUM_WORKERS,
        shuffle=False,
        drop_last=False,
        pin_memory=True if device.type == 'cuda' else False
    )

    criterion = InfoNCELoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Create directories and TensorBoard writer
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)

    writer = SummaryWriter(log_path)

    start_epoch, global_step, best_loss = load_checkpoint(checkpoint_path, model, optimizer, device)

    # ==== TRAINING LOOP ====
    for epoch in range(start_epoch, epochs):
        model.train()
        all_losses = []

        progress_bar = tqdm(train_dataloader, colour="BLUE")
        for i, batch in enumerate(progress_bar):
            query_input_ids = batch['query_input_ids'].to(device)
            query_attention_mask = batch['query_attention_mask'].to(device)
            pos_doc_input_ids = batch['pos_doc_input_ids'].to(device)
            pos_doc_attention_mask = batch['pos_doc_attention_mask'].to(device)

            # Forward pass
            query_embeddings, doc_embeddings = model(
                query_input_ids,
                query_attention_mask,
                pos_doc_input_ids,
                pos_doc_attention_mask
            )

            loss = criterion(query_embeddings, doc_embeddings)

            progress_bar.set_description(
                "Epoch {}/{}. Loss {:0.4f}. LR {:0.2e}".format(
                    epoch + 1, epochs, loss.item(), scheduler.get_last_lr()[0]
                )
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step()

            all_losses.append(loss.item())

            writer.add_scalar("Train/loss", loss.item(), global_step)
            writer.add_scalar("Train/learning_rate", scheduler.get_last_lr()[0], global_step)

            global_step += 1

        train_loss = sum(all_losses) / len(all_losses)
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

        # ==== VALIDATION ====
        model.eval()
        all_dev_losses = []

        with torch.no_grad():
            for batch in tqdm(dev_dataloader, desc="Evaluating", colour="GREEN"):
                query_input_ids = batch['query_input_ids'].to(device)
                query_attention_mask = batch['query_attention_mask'].to(device)
                pos_doc_input_ids = batch['pos_doc_input_ids'].to(device)
                pos_doc_attention_mask = batch['pos_doc_attention_mask'].to(device)

                query_embeddings, doc_embeddings = model(
                    query_input_ids,
                    query_attention_mask,
                    pos_doc_input_ids,
                    pos_doc_attention_mask
                )

                loss = criterion(query_embeddings, doc_embeddings)
                all_dev_losses.append(loss.item())

        dev_loss = sum(all_dev_losses) / len(all_dev_losses)
        print(f"Epoch {epoch + 1}/{epochs} - Dev Loss: {dev_loss:.4f}")

        writer.add_scalar("Dev/loss", dev_loss, epoch)
        writer.add_scalar("Train/epoch_loss", train_loss, epoch)

        save_checkpoint(checkpoint_path, epoch, global_step, model, optimizer, dev_loss)

    writer.close()


if __name__ == '__main__':
    train()