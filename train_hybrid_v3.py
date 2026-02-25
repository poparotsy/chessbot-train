import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import gc
import time
import signal
import sys
from torch.utils.data import DataLoader, TensorDataset

# ============ SYSTEM STABILITY ============
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ============ PRODUCTION CONFIG ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()
# Conservative batching for 14GB-16GB GPUs
BATCH_SIZE = 256 * max(1, GPU_COUNT)
LEARNING_RATE = 1e-4
EPOCHS = 50
DATA_DIR = "tensors_v3"
MODEL_SAVE_PATH = "models/model_hybrid_beast.pt"
CHECKPOINT_DIR = "models/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Global flag for graceful shutdown
INTERRUPTED = False

def signal_handler(sig, frame):
    global INTERRUPTED
    print("\n⚠️  Interrupt received. Saving checkpoint...")
    INTERRUPTED = True

signal.signal(signal.SIGINT, signal_handler)

# ============ MODEL ARCHITECTURE ============


class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(
                64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(
                128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(
                256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(
                512), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 13)
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


def save_checkpoint(epoch, model, optimizer, scheduler, accuracy, loss, path):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'accuracy': accuracy,
        'loss': loss
    }
    torch.save(checkpoint, path)


def train():
    global INTERRUPTED
    print(f"\n🚀 STARTING BEAST MODE TRAINING")
    print(f"💻 Hardware: {DEVICE} ({GPU_COUNT} GPUs) | Batch Size: {BATCH_SIZE}")

    model = ChessCNN()
    if GPU_COUNT > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE)

    weights = torch.tensor([0.7] + [1.3] * 12).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    train_files = sorted(glob.glob(f"{DATA_DIR}/train_*.pt"))
    val_files = sorted(glob.glob(f"{DATA_DIR}/val_*.pt"))

    # Resume from checkpoint if exists
    start_epoch = 0
    checkpoint_path = f"{CHECKPOINT_DIR}/latest.pt"
    if os.path.exists(checkpoint_path):
        print(f"📂 Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if GPU_COUNT > 1:
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        if INTERRUPTED:
            break
            
        model.train()
        epoch_start = time.time()
        total_loss = 0

        # --- TRAINING PHASE ---
        for f_idx, f in enumerate(train_files):
            if INTERRUPTED:
                break
                
            data = torch.load(f, map_location='cpu')
            x = (data['x'].float() / 127.5) - 1.0
            y = data['y'].long()

            loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)

            chunk_loss = 0
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                chunk_loss += loss.item()

            total_loss += (chunk_loss / len(loader))
            if (f_idx + 1) % 2 == 0:
                print(f"   Epoch {epoch + 1} | Chunk {f_idx + 1}/{len(train_files)} | Loss: {chunk_loss / len(loader):.4f}")

            del data, x, y, loader
            gc.collect()
            torch.cuda.empty_cache()

        if INTERRUPTED:
            break

        # --- VALIDATION PHASE (ALL CHUNKS) ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for v_file in val_files:
                v_data = torch.load(v_file, map_location='cpu')
                vx_all = (v_data['x'].float() / 127.5) - 1.0
                vy_all = v_data['y'].long()

                v_loader = DataLoader(TensorDataset(vx_all, vy_all), batch_size=BATCH_SIZE)
                for vbx, vby in v_loader:
                    vbx, vby = vbx.to(DEVICE), vby.to(DEVICE)
                    v_out = model(vbx)
                    preds = torch.argmax(v_out, dim=1)
                    correct += (preds == vby).sum().item()
                    total += vby.size(0)

                del v_data, vx_all, vy_all, v_loader
                gc.collect()
                torch.cuda.empty_cache()

        # --- SUMMARY ---
        duration = time.time() - epoch_start
        accuracy = correct / total
        current_lr = scheduler.get_last_lr()[0]
        print(f"✅ EPOCH {epoch + 1:02d} | Loss: {total_loss / len(train_files):.4f} | Val Acc: {accuracy:.4f} | LR: {current_lr:.2e} | Time: {duration:.1f}s")

        # Save checkpoint
        save_obj = model.module.state_dict() if GPU_COUNT > 1 else model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state': save_obj,
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'accuracy': accuracy,
            'loss': total_loss / len(train_files)
        }
        torch.save(checkpoint, checkpoint_path)
        torch.save(save_obj, f"{CHECKPOINT_DIR}/epoch_{epoch + 1:02d}.pt")
        
        # Save best model
        if epoch == 0 or accuracy > train.best_acc:
            train.best_acc = accuracy
            torch.save(save_obj, MODEL_SAVE_PATH)
            print(f"   💾 Best model saved (acc: {accuracy:.4f})")

        scheduler.step()

    if INTERRUPTED:
        print(f"\n✅ Training interrupted. Checkpoint saved at epoch {epoch + 1}")
    else:
        print(f"\n🎉 Training complete!")

train.best_acc = 0.0


if __name__ == "__main__":
    train()
