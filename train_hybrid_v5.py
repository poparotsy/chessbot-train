import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import gc
import time
import signal
import sys
import subprocess
import shutil
import json
import re
from torch.utils.data import DataLoader, TensorDataset

# ============ SYSTEM STABILITY ============
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


# ============ HUMAN CONFIG (SAFE TO EDIT) ============
DATA_DIR = "tensors_v5"
MODEL_SAVE_PATH = "models/model_hybrid_v5_latest_best.pt"
FINAL_MODEL_SAVE_PATH = "models/model_hybrid_v5_final.pt"
CHECKPOINT_DIR = "models/checkpoints_v5"
BASE_MODEL_PATH = "models/model_hybrid_v5_latest_best.pt"
EPOCHS = 333
LEARNING_RATE = 3e-6
BATCH_SIZE_PER_GPU = 256
RESUME_FROM_CHECKPOINT = False
RUN_HARDSET_EVAL_ON_BEST = True
HARDSET_EVAL_SCRIPT = "scripts/rank_models_hardset.py"
HARDSET_TRUTH_JSON = "images_4_test/truth_verified.json"
HARDSET_COMPARE_FULL_FEN = False
HARDSET_REQUIRE_NON_REGRESSION = True
HARDSET_MAX_CONSECUTIVE_REGRESSIONS = 6
MIN_ACC_IMPROVEMENT = 1e-6
BACKUP_EXISTING_BEST_MODEL = True

# ============ ADVANCED / INTERNAL (USUALLY DON'T TOUCH) ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()
BATCH_SIZE = BATCH_SIZE_PER_GPU * max(1, GPU_COUNT)
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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


def extract_model_state(payload):
    """Accept either raw state_dict or full checkpoint dict."""
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"], "checkpoint"
    return payload, "state_dict"


def run_rank_eval(
    model_path,
    truth_json,
    images_dir=None,
    compare_full_fen=False,
    label="benchmark",
):
    """Run ranking script and return parsed score metadata."""
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), HARDSET_EVAL_SCRIPT)
    if not os.path.exists(script_path):
        print(f"   ⚠️ {label} eval script not found: {script_path}")
        return None
    if not os.path.exists(truth_json):
        print(f"   ⚠️ {label} truth file not found: {truth_json}")
        return None

    cmd = [
        sys.executable,
        script_path,
        "--models-glob",
        model_path,
        "--truth-json",
        truth_json,
    ]
    if images_dir:
        cmd.extend(["--images-dir", images_dir])
    if compare_full_fen:
        cmd.append("--compare-full-fen")

    print(f"   📊 Running {label} ranking on candidate...")
    try:
        proc = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.stderr:
            print(proc.stderr.rstrip())
        if proc.returncode != 0:
            print(f"   ⚠️ {label} ranking exited with code {proc.returncode}")
            return None
        output = proc.stdout or ""
        marker = "=== HARD-SET RANKING ==="
        if marker in output:
            tail = output.split(marker, 1)[1].strip()
            try:
                summary = json.loads(tail)
                model_abs = os.path.abspath(model_path)
                for row in summary.get("ranked", []):
                    row_model_abs = os.path.abspath(row.get("model", ""))
                    if row_model_abs == model_abs or os.path.basename(row_model_abs) == os.path.basename(model_abs):
                        return {
                            "passed": int(row["passed"]),
                            "total": int(row["total"]),
                            "avg_confidence": float(row.get("avg_confidence", 0.0)),
                        }
            except Exception:
                pass

        pattern = re.compile(r"->\s*(\d+)/(\d+)\s*\(avg_conf=([0-9.]+)\)")
        matches = pattern.findall(output)
        if matches:
            last = matches[-1]
            return {
                "passed": int(last[0]),
                "total": int(last[1]),
                "avg_confidence": float(last[2]),
            }
    except Exception as exc:
        print(f"   ⚠️ {label} ranking failed: {exc}")
    return None


def run_hardset_eval_on_best(model_path):
    """Run canonical hardset ranking."""
    if not RUN_HARDSET_EVAL_ON_BEST:
        return None
    return run_rank_eval(
        model_path=model_path,
        truth_json=HARDSET_TRUTH_JSON,
        images_dir=None,
        compare_full_fen=HARDSET_COMPARE_FULL_FEN,
        label="hardset",
    )


def evaluate_model_accuracy(model, val_files):
    """Compute validation accuracy across all val chunks."""
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
                preds = torch.argmax(model(vbx), dim=1)
                correct += (preds == vby).sum().item()
                total += vby.size(0)
            del v_data, vx_all, vy_all, v_loader
            gc.collect()
            torch.cuda.empty_cache()
    return (correct / total) if total else 0.0


def train():
    global INTERRUPTED
    print(f"\n🚀 STARTING BEAST MODE TRAINING")
    print(f"💻 Hardware: {DEVICE} ({GPU_COUNT} GPUs) | Batch Size: {BATCH_SIZE}")
    print(f"📚 Data Dir: {DATA_DIR} | Base Model: {BASE_MODEL_PATH}")

    model = ChessCNN()
    if GPU_COUNT > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE)

    weights = torch.tensor([0.7] + [1.3] * 12).to(DEVICE)
    # Use Focal Loss instead of CrossEntropy - focuses on hard examples
    criterion = FocalLoss(alpha=1, gamma=2, weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    train_files = sorted(glob.glob(f"{DATA_DIR}/train_*.pt"))
    val_files = sorted(glob.glob(f"{DATA_DIR}/val_*.pt"))
    train.consecutive_hardset_regressions = 0

    # Explicit precedence:
    # 1) resume checkpoint when enabled and present
    # 2) otherwise warm-start from base model
    start_epoch = 0
    resumed = False
    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        print(f"📂 Resuming from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if GPU_COUNT > 1:
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch'] + 1
        train.best_acc = checkpoint.get('accuracy', 0.0)
        resumed = True
        print(f"✅ Resumed from epoch {start_epoch}")
        # Ensure there is always at least one exported model file on resume.
        if not os.path.exists(MODEL_SAVE_PATH):
            save_obj = model.module.state_dict() if GPU_COUNT > 1 else model.state_dict()
            torch.save(save_obj, MODEL_SAVE_PATH)
            print(f"💾 Seeded best-model file from resumed checkpoint: {MODEL_SAVE_PATH}")
    else:
        if RESUME_FROM_CHECKPOINT:
            print(f"ℹ️ No checkpoint found at {CHECKPOINT_PATH}; using base model warm-start.")
        elif os.path.exists(CHECKPOINT_PATH):
            print(f"ℹ️ Checkpoint exists at {CHECKPOINT_PATH} but RESUME_FROM_CHECKPOINT=False; ignoring it.")

    if not resumed:
        if os.path.exists(BASE_MODEL_PATH):
            raw_payload = torch.load(BASE_MODEL_PATH, map_location=DEVICE)
            base_state, base_kind = extract_model_state(raw_payload)
            if GPU_COUNT > 1:
                model.module.load_state_dict(base_state)
            else:
                model.load_state_dict(base_state)
            print(f"📦 Loaded base model ({base_kind}): {BASE_MODEL_PATH}")
        else:
            print(f"⚠️ Base model not found: {BASE_MODEL_PATH} (training from scratch)")

    train.best_hardset = None
    if not train_files or not val_files:
        raise RuntimeError(f"Missing dataset chunks in {DATA_DIR}. Found train={len(train_files)}, val={len(val_files)}")

    # Protect existing exported best model from accidental downgrade.
    if os.path.exists(MODEL_SAVE_PATH):
        if BACKUP_EXISTING_BEST_MODEL:
            backup_path = MODEL_SAVE_PATH.replace(".pt", "_pretrain_backup.pt")
            shutil.copy2(MODEL_SAVE_PATH, backup_path)
            print(f"🛡️ Backed up existing best model: {backup_path}")
        try:
            eval_model = ChessCNN()
            if GPU_COUNT > 1:
                eval_model = nn.DataParallel(eval_model)
            eval_model.to(DEVICE)

            existing_payload = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            existing_state, _ = extract_model_state(existing_payload)
            if GPU_COUNT > 1:
                eval_model.module.load_state_dict(existing_state)
            else:
                eval_model.load_state_dict(existing_state)
            existing_acc = evaluate_model_accuracy(eval_model, val_files)
            train.best_acc = max(train.best_acc, existing_acc)
            print(f"📌 Protected best baseline from {MODEL_SAVE_PATH}: val_acc={existing_acc:.4f}")
            del eval_model, existing_payload, existing_state
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"⚠️ Could not evaluate existing best model baseline: {exc}")

        baseline_hardset = run_hardset_eval_on_best(MODEL_SAVE_PATH)
        if baseline_hardset is not None:
            train.best_hardset = baseline_hardset
            print(
                f"📌 Hardset baseline protected: "
                f"{baseline_hardset['passed']}/{baseline_hardset['total']}"
            )
        else:
            print("⚠️ Hardset baseline unavailable; save gating will use val_acc only.")

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
        accuracy = evaluate_model_accuracy(model, val_files)

        # --- SUMMARY ---
        duration = time.time() - epoch_start
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
        if accuracy > (train.best_acc + MIN_ACC_IMPROVEMENT):
            candidate_path = MODEL_SAVE_PATH.replace(".pt", "_candidate.pt")
            torch.save(save_obj, candidate_path)
            candidate_hardset = run_hardset_eval_on_best(candidate_path)

            hardset_allows_save = True
            if HARDSET_REQUIRE_NON_REGRESSION and train.best_hardset is not None and candidate_hardset is not None:
                if candidate_hardset["passed"] < train.best_hardset["passed"]:
                    hardset_allows_save = False
                    print(
                        "   ⛔ Rejecting candidate due to hardset regression: "
                        f"{candidate_hardset['passed']}/{candidate_hardset['total']} < "
                        f"{train.best_hardset['passed']}/{train.best_hardset['total']}"
                    )
                    train.consecutive_hardset_regressions += 1
                    print(
                        f"   ⏱️ Consecutive hardset regressions: "
                        f"{train.consecutive_hardset_regressions}/{HARDSET_MAX_CONSECUTIVE_REGRESSIONS}"
                    )
                else:
                    train.consecutive_hardset_regressions = 0

            if hardset_allows_save:
                train.best_acc = accuracy
                os.replace(candidate_path, MODEL_SAVE_PATH)
                print(f"   💾 Best model saved (acc: {accuracy:.4f})")
                if candidate_hardset is not None:
                    train.best_hardset = candidate_hardset
                train.consecutive_hardset_regressions = 0
            else:
                try:
                    os.remove(candidate_path)
                except OSError:
                    pass
                if train.consecutive_hardset_regressions >= HARDSET_MAX_CONSECUTIVE_REGRESSIONS:
                    print(
                        "🛑 Early stop: repeated hardset regressions despite val_acc improvements."
                    )
                    if train.best_hardset is not None:
                        print(
                            f"   Protected hardset best: {train.best_hardset['passed']}/{train.best_hardset['total']}"
                        )
                    break

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
