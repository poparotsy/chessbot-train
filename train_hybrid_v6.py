#!/usr/bin/env python3
"""Standalone v6 trainer (tile classifier) with val-only best selection."""

import gc
import glob
import os
import signal
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_REV = "v6-train-2026-03-12-r1"

# ============ HUMAN CONFIG (SAFE TO EDIT) ============
DATA_DIR = "tensors_v6"
MODEL_SAVE_PATH = "models/model_hybrid_v6_latest_best.pt"
FINAL_MODEL_SAVE_PATH = "models/model_hybrid_v6_final.pt"
CHECKPOINT_DIR = "models/checkpoints_v6"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")
BASE_MODEL_PATH = "models/model_hybrid_v5_latest_best.pt"
EPOCHS = 200
LEARNING_RATE = 3e-6
BATCH_SIZE_PER_GPU = 256
RESUME_FROM_CHECKPOINT = True
MIN_ACC_IMPROVEMENT = 1e-6
BACKUP_EXISTING_BEST_MODEL = True

# ============ ADVANCED / INTERNAL ============
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()
BATCH_SIZE = BATCH_SIZE_PER_GPU * max(1, GPU_COUNT)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

INTERRUPTED = False


def signal_handler(sig, frame):
    del sig, frame
    global INTERRUPTED
    INTERRUPTED = True
    print("\n⚠️ Interrupt received. Saving checkpoint at epoch boundary...", flush=True)


signal.signal(signal.SIGINT, signal_handler)


class FocalLoss(nn.Module):
    """Focal loss for class imbalance and hard-example focus."""

    def __init__(self, alpha=1.0, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1.0 - pt) ** self.gamma * ce_loss
        return focal.mean()


class ChessCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 13),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def extract_model_state(payload):
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"], "checkpoint"
    return payload, "state_dict"


def save_checkpoint(epoch, model, optimizer, scheduler, accuracy, loss, path):
    save_obj = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    checkpoint = {
        "epoch": int(epoch),
        "model_state": save_obj,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "accuracy": float(accuracy),
        "loss": float(loss),
    }
    torch.save(checkpoint, path)


def evaluate_model_accuracy(model, val_files):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for v_file in val_files:
            v_data = torch.load(v_file, map_location="cpu")
            vx_all = (v_data["x"].float() / 127.5) - 1.0
            vy_all = v_data["y"].long()
            v_loader = DataLoader(TensorDataset(vx_all, vy_all), batch_size=BATCH_SIZE)
            for vbx, vby in v_loader:
                vbx, vby = vbx.to(DEVICE), vby.to(DEVICE)
                preds = torch.argmax(model(vbx), dim=1)
                correct += (preds == vby).sum().item()
                total += vby.size(0)
            del v_data, vx_all, vy_all, v_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return (correct / total) if total else 0.0


def train():
    global INTERRUPTED

    print(f"🧾 Script revision: {SCRIPT_REV}")
    print("\n🚀 STARTING BEAST MODE TRAINING")
    print(f"💻 Hardware: {DEVICE} ({GPU_COUNT} GPUs) | Batch Size: {BATCH_SIZE}")
    print(f"📚 Data Dir: {DATA_DIR} | Base Model: {BASE_MODEL_PATH}")

    model = ChessCNN()
    if GPU_COUNT > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE)

    weights = torch.tensor([0.7] + [1.3] * 12).to(DEVICE)
    criterion = FocalLoss(alpha=1.0, gamma=2.0, weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    train_files = sorted(glob.glob(f"{DATA_DIR}/train_*.pt"))
    val_files = sorted(glob.glob(f"{DATA_DIR}/val_*.pt"))
    if not train_files or not val_files:
        raise RuntimeError(
            f"Missing dataset chunks in {DATA_DIR}. Found train={len(train_files)}, val={len(val_files)}"
        )

    best_acc = 0.0
    start_epoch = 0
    resumed = False

    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_acc = float(checkpoint.get("accuracy", 0.0))
        resumed = True
        print(f"📂 Resumed from checkpoint: {CHECKPOINT_PATH} @ epoch {start_epoch}")
    elif RESUME_FROM_CHECKPOINT:
        print(f"ℹ️ No checkpoint found at {CHECKPOINT_PATH}; using base model warm-start.")

    if not resumed:
        if os.path.exists(BASE_MODEL_PATH):
            payload = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
            base_state, base_kind = extract_model_state(payload)
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(base_state)
            else:
                model.load_state_dict(base_state)
            print(f"📦 Loaded base model ({base_kind}): {BASE_MODEL_PATH}")
        else:
            print(f"⚠️ Base model not found: {BASE_MODEL_PATH} (training from scratch)")

    if os.path.exists(MODEL_SAVE_PATH):
        if BACKUP_EXISTING_BEST_MODEL:
            backup_path = MODEL_SAVE_PATH.replace(".pt", "_pretrain_backup.pt")
            torch.save(torch.load(MODEL_SAVE_PATH, map_location="cpu"), backup_path)
            print(f"🛡️ Backed up existing best model: {backup_path}")

        try:
            eval_model = ChessCNN()
            if GPU_COUNT > 1:
                eval_model = nn.DataParallel(eval_model)
            eval_model.to(DEVICE)
            payload = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            state, _ = extract_model_state(payload)
            if isinstance(eval_model, nn.DataParallel):
                eval_model.module.load_state_dict(state)
            else:
                eval_model.load_state_dict(state)
            existing_acc = evaluate_model_accuracy(eval_model, val_files)
            best_acc = max(best_acc, existing_acc)
            print(f"📌 Protected best baseline from {MODEL_SAVE_PATH}: val_acc={existing_acc:.4f}")
            del eval_model, payload, state
        except Exception as exc:
            print(f"⚠️ Could not evaluate existing best model baseline: {exc}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    last_completed_epoch = start_epoch - 1
    last_loss = 0.0
    last_acc = best_acc

    for epoch in range(start_epoch, EPOCHS):
        if INTERRUPTED:
            break
        model.train()
        epoch_start = time.time()
        total_loss = 0.0

        for file_idx, fpath in enumerate(train_files):
            if INTERRUPTED:
                break
            data = torch.load(fpath, map_location="cpu")
            x = (data["x"].float() / 127.5) - 1.0
            y = data["y"].long()
            loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True)

            chunk_loss = 0.0
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                chunk_loss += float(loss.item())

            avg_chunk_loss = chunk_loss / max(1, len(loader))
            total_loss += avg_chunk_loss
            if (file_idx + 1) % 2 == 0:
                print(f"   Epoch {epoch + 1:03d} | Chunk {file_idx + 1}/{len(train_files)} | Loss: {avg_chunk_loss:.4f}")

            del data, x, y, loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if INTERRUPTED:
            break

        accuracy = evaluate_model_accuracy(model, val_files)
        avg_loss = total_loss / max(1, len(train_files))
        duration = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"✅ EPOCH {epoch + 1:03d} | Loss: {avg_loss:.4f} | Val Acc: {accuracy:.4f} "
            f"| LR: {current_lr:.2e} | Time: {duration:.1f}s"
        )

        save_checkpoint(epoch, model, optimizer, scheduler, accuracy, avg_loss, CHECKPOINT_PATH)
        last_completed_epoch = epoch
        last_loss = avg_loss
        last_acc = accuracy

        if accuracy > (best_acc + MIN_ACC_IMPROVEMENT):
            best_acc = accuracy
            save_obj = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(save_obj, MODEL_SAVE_PATH)
            print(f"   💾 Best model saved (val_acc: {accuracy:.4f})")

        scheduler.step()

    # Final checkpoint on graceful interrupt to avoid progress loss inside current run.
    if INTERRUPTED:
        save_checkpoint(
            last_completed_epoch,
            model,
            optimizer,
            scheduler,
            last_acc,
            last_loss,
            CHECKPOINT_PATH,
        )
        print(f"\n✅ Training interrupted. Checkpoint saved: {CHECKPOINT_PATH}")
    else:
        final_obj = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(final_obj, FINAL_MODEL_SAVE_PATH)
        print(f"\n🎉 Training complete! Final model saved: {FINAL_MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
