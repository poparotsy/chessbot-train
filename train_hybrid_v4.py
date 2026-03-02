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
LEARNING_RATE = 1e-5
EPOCHS = 300
DATA_DIR = "tensors_v4"
MODEL_SAVE_PATH = "models/model_hybrid_v4_300e_best.pt"
FINAL_MODEL_SAVE_PATH = "models/model_hybrid_v4_300e_final.pt"
CHECKPOINT_DIR = "models/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/latest.pt"
BASE_MODEL_PATH = "models/model_hybrid_v4_250e_best.pt"

# Global flag for graceful shutdown
INTERRUPTED = False

def signal_handler(sig, frame):
    global INTERRUPTED
    print("\n⚠️  Interrupt received. Saving checkpoint...")
    INTERRUPTED = True

signal.signal(signal.SIGINT, signal_handler)

# ============ MODEL ARCHITECTURE ============

class FocalLoss(nn.Module):
    """Focal Loss - focuses on hard examples"""
    def __init__(self, alpha=1, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 13)
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

    # Warm-start from completed v3 model for Phase 3 continuation.
    if os.path.exists(BASE_MODEL_PATH):
        base_state = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
        if GPU_COUNT > 1:
            model.module.load_state_dict(base_state)
        else:
            model.load_state_dict(base_state)
        print(f"📦 Loaded base model: {BASE_MODEL_PATH}")
    else:
        print(f"⚠️ Base model not found: {BASE_MODEL_PATH} (training from scratch)")

    weights = torch.tensor([0.7] + [1.3] * 12).to(DEVICE)
    # Use Focal Loss instead of CrossEntropy - focuses on hard examples
    criterion = FocalLoss(alpha=1, gamma=2, weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    train_files = sorted(glob.glob(f"{DATA_DIR}/train_*.pt"))
    val_files = sorted(glob.glob(f"{DATA_DIR}/val_*.pt"))

    # Resume v4-specific checkpoint if available
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"📂 Resuming from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if GPU_COUNT > 1:
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        train.best_acc = checkpoint.get('accuracy', 0.0)
        print(f"✅ Resumed from epoch {start_epoch}")
        # Ensure there is always at least one exported model file on resume.
        if not os.path.exists(MODEL_SAVE_PATH):
            save_obj = model.module.state_dict() if GPU_COUNT > 1 else model.state_dict()
            torch.save(save_obj, MODEL_SAVE_PATH)
            print(f"💾 Seeded best-model file from resumed checkpoint: {MODEL_SAVE_PATH}")

    if not train_files or not val_files:
        raise RuntimeError(f"Missing dataset chunks in {DATA_DIR}. Found train={len(train_files)}, val={len(val_files)}")

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
        torch.save(checkpoint, CHECKPOINT_PATH)
        
        # Save best model only
        if epoch == 0 or accuracy > train.best_acc:
            train.best_acc = accuracy
            torch.save(save_obj, MODEL_SAVE_PATH)
            print(f"   💾 Best model saved (acc: {accuracy:.4f})")

        scheduler.step()

    if INTERRUPTED:
        print(f"\n✅ Training interrupted. Checkpoint saved at epoch {epoch + 1}")
    else:
        # Always export final model snapshot regardless of best-accuracy improvements.
        final_obj = model.module.state_dict() if GPU_COUNT > 1 else model.state_dict()
        torch.save(final_obj, FINAL_MODEL_SAVE_PATH)
        print(f"\n🎉 Training complete! Final model saved: {FINAL_MODEL_SAVE_PATH}")

train.best_acc = 0.0


if __name__ == "__main__":
    train()
