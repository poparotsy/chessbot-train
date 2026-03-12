#!/usr/bin/env python3
"""Train multi-head v7 model (pieces + geometry + POV + STM)."""

from __future__ import annotations

import gc
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw is not None else default


# ============ HUMAN CONFIG (EDIT THESE) ============
DATA_DIR = "tensors_v7"
MODEL_SAVE_PATH = "models/model_hybrid_v7_scratch_latest_best.pt"
FINAL_MODEL_SAVE_PATH = "models/model_hybrid_v7_scratch_final.pt"
CHECKPOINT_DIR = "models/checkpoints_v7_scratch"
BASE_MODEL_PATH = ""
EPOCHS = 120
LEARNING_RATE = 3e-5
BATCH_SIZE_PER_GPU = 20
RESUME_FROM_CHECKPOINT = False
W_PIECE = 1.0
W_GEOM = 0.35
W_BOARD = 0.25
W_POV = 0.20
W_STM = 0.15
RUN_HARDSET_EVAL_ON_BEST = True
HARDSET_EVAL_SCRIPT = "scripts/rank_models_hardset.py"
HARDSET_TRUTH_JSON = "images_4_test/truth_verified.json"
HARDSET_COMPARE_FULL_FEN = False
HARDSET_REQUIRE_NON_REGRESSION = True
HARDSET_MAX_CONSECUTIVE_REGRESSIONS = 6
MIN_SCORE_IMPROVEMENT = 1e-6
BACKUP_EXISTING_BEST_MODEL = False
USE_EXISTING_MODEL_SAVE_BASELINE = False

# Optional: allow env overrides for Kaggle/automation runs.
APPLY_ENV_OVERRIDES = True
if APPLY_ENV_OVERRIDES:
    DATA_DIR = env_str("DATA_DIR", DATA_DIR)
    MODEL_SAVE_PATH = env_str("MODEL_SAVE_PATH", MODEL_SAVE_PATH)
    FINAL_MODEL_SAVE_PATH = env_str("FINAL_MODEL_SAVE_PATH", FINAL_MODEL_SAVE_PATH)
    CHECKPOINT_DIR = env_str("CHECKPOINT_DIR", CHECKPOINT_DIR)
    BASE_MODEL_PATH = env_str("BASE_MODEL_PATH", BASE_MODEL_PATH)
    EPOCHS = env_int("EPOCHS", EPOCHS)
    LEARNING_RATE = env_float("LEARNING_RATE", LEARNING_RATE)
    BATCH_SIZE_PER_GPU = env_int("BATCH_SIZE_PER_GPU", BATCH_SIZE_PER_GPU)
    RESUME_FROM_CHECKPOINT = env_str(
        "RESUME_FROM_CHECKPOINT",
        "1" if RESUME_FROM_CHECKPOINT else "0",
    ) == "1"
    W_PIECE = env_float("W_PIECE", W_PIECE)
    W_GEOM = env_float("W_GEOM", W_GEOM)
    W_BOARD = env_float("W_BOARD", W_BOARD)
    W_POV = env_float("W_POV", W_POV)
    W_STM = env_float("W_STM", W_STM)

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")


# ============ INTERNAL ============
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()
BATCH_SIZE = BATCH_SIZE_PER_GPU * max(1, GPU_COUNT)


class MultiHeadChessNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Keep same conv stem as v5/v4 for warm-start compatibility.
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
        )

        self.piece_head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(128, 13, 1),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.geom_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 8),
        )
        self.board_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.pov_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # white/black/unknown
        )
        self.stm_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # w/b/unknown
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.features(x)
        piece_logits = self.piece_head(feat)  # [B, 13, 8, 8]
        piece_logits = piece_logits.permute(0, 2, 3, 1).reshape(x.size(0), 64, 13)

        g = self.global_pool(feat).flatten(1)
        corners = self.geom_head(g)
        board_logit = self.board_head(g).squeeze(1)
        pov_logits = self.pov_head(g)
        stm_logits = self.stm_head(g)
        return {
            "piece_logits": piece_logits,
            "corners": corners,
            "board_logit": board_logit,
            "pov_logits": pov_logits,
            "stm_logits": stm_logits,
        }


def _clean_state_dict_keys(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def load_warmstart(model: MultiHeadChessNet, model_path: str) -> None:
    if not model_path:
        print("📦 Warm-start disabled (training from scratch).")
        return
    if not os.path.exists(model_path):
        print(f"⚠️ Base model not found: {model_path} (training backbone from scratch)")
        return
    payload = torch.load(model_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        payload = payload["model_state"]
    src = _clean_state_dict_keys(payload)
    dst = model.state_dict()

    # Try full warm-start first (true continuation when architecture matches).
    full_loaded = 0
    for k, v in src.items():
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v
            full_loaded += 1
    full_ratio = full_loaded / max(1, len(dst))

    if full_ratio >= 0.90:
        model.load_state_dict(dst, strict=False)
        print(
            f"📦 Warm-started FULL model from {model_path} | "
            f"matched={full_loaded}/{len(dst)} ({full_ratio:.1%})"
        )
        return

    # Fallback: backbone-only warm-start (v5 -> v7 transfer).
    dst = model.state_dict()
    feat_loaded = 0
    for k, v in src.items():
        if not k.startswith("features."):
            continue
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v
            feat_loaded += 1
    model.load_state_dict(dst, strict=False)
    print(
        f"📦 Warm-started BACKBONE from {model_path} | "
        f"matched features={feat_loaded}"
    )


def run_rank_eval(
    model_path: str,
    truth_json: str,
    recognizer_module: str = "recognizer_v7",
    images_dir: str | None = None,
    compare_full_fen: bool = False,
    label: str = "hardset",
) -> Dict[str, float] | None:
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
        "--recognizer-module",
        recognizer_module,
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


def run_hardset_eval_on_model(model_path: str) -> Dict[str, float] | None:
    if not RUN_HARDSET_EVAL_ON_BEST:
        return None
    model_name = os.path.basename(model_path).lower()
    if "v5" in model_name or "v4" in model_name:
        recognizer_module = "recognizer_v5"
    else:
        recognizer_module = "recognizer_v7"
    return run_rank_eval(
        model_path=model_path,
        truth_json=HARDSET_TRUTH_JSON,
        recognizer_module=recognizer_module,
        images_dir=None,
        compare_full_fen=HARDSET_COMPARE_FULL_FEN,
        label="hardset",
    )


def _make_loader(data_file: str, batch_size: int, shuffle: bool) -> DataLoader:
    d = torch.load(data_file, map_location="cpu")
    x = (d["x"].float() / 127.5) - 1.0
    piece = d["piece"].long()
    piece_mask = d["piece_mask"].float()
    corners = d["corners"].float()
    board_present = d["board_present"].float()
    pov = d["pov"].long()
    stm = d["stm"].long()
    ds = TensorDataset(x, piece, piece_mask, corners, board_present, pov, stm)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _compute_losses(
    out: Dict[str, torch.Tensor],
    piece: torch.Tensor,
    piece_mask: torch.Tensor,
    corners: torch.Tensor,
    board_present: torch.Tensor,
    pov: torch.Tensor,
    stm: torch.Tensor,
    ce_piece: nn.CrossEntropyLoss,
    ce_aux: nn.CrossEntropyLoss,
    bce: nn.BCEWithLogitsLoss,
    smooth_l1: nn.SmoothL1Loss,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    bsz = piece.size(0)
    piece_logits = out["piece_logits"].reshape(-1, 13)
    piece_target = piece.reshape(-1)
    # tile-level CE -> sample-level mask by board_present/piece_mask
    tile_loss = ce_piece(piece_logits, piece_target).reshape(bsz, 64).mean(dim=1)
    denom = piece_mask.sum().clamp(min=1.0)
    piece_loss = (tile_loss * piece_mask).sum() / denom

    geom_raw = smooth_l1(out["corners"], corners).mean(dim=1)
    geom_loss = (geom_raw * board_present).sum() / board_present.sum().clamp(min=1.0)

    board_loss = bce(out["board_logit"], board_present)
    pov_loss = ce_aux(out["pov_logits"], pov)
    stm_loss = ce_aux(out["stm_logits"], stm)

    total = (
        W_PIECE * piece_loss
        + W_GEOM * geom_loss
        + W_BOARD * board_loss
        + W_POV * pov_loss
        + W_STM * stm_loss
    )
    metrics = {
        "piece_loss": float(piece_loss.item()),
        "geom_loss": float(geom_loss.item()),
        "board_loss": float(board_loss.item()),
        "pov_loss": float(pov_loss.item()),
        "stm_loss": float(stm_loss.item()),
        "total_loss": float(total.item()),
    }
    return total, metrics


@torch.no_grad()
def evaluate(model: nn.Module, val_files: list[str]) -> Dict[str, float]:
    model.eval()
    ce_piece = nn.CrossEntropyLoss(reduction="none")
    ce_aux = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss(reduction="none")

    total_samples = 0
    agg = {
        "loss": 0.0,
        "tile_correct": 0.0,
        "tile_total": 0.0,
        "board_correct": 0.0,
        "pov_correct": 0.0,
        "stm_correct": 0.0,
    }
    for vf in val_files:
        loader = _make_loader(vf, batch_size=BATCH_SIZE, shuffle=False)
        for bx, piece, piece_mask, corners, board_present, pov, stm in loader:
            bx = bx.to(DEVICE)
            piece = piece.to(DEVICE)
            piece_mask = piece_mask.to(DEVICE)
            corners = corners.to(DEVICE)
            board_present = board_present.to(DEVICE)
            pov = pov.to(DEVICE)
            stm = stm.to(DEVICE)
            out = model(bx)

            total, _ = _compute_losses(
                out=out,
                piece=piece,
                piece_mask=piece_mask,
                corners=corners,
                board_present=board_present,
                pov=pov,
                stm=stm,
                ce_piece=ce_piece,
                ce_aux=ce_aux,
                bce=bce,
                smooth_l1=smooth_l1,
            )
            agg["loss"] += float(total.item()) * bx.size(0)

            pred_piece = out["piece_logits"].argmax(dim=-1)
            match = (pred_piece == piece).float().mean(dim=1)
            agg["tile_correct"] += float((match * piece_mask).sum().item()) * 64.0
            agg["tile_total"] += float(piece_mask.sum().item()) * 64.0

            pred_board = (torch.sigmoid(out["board_logit"]) >= 0.5).long()
            agg["board_correct"] += float((pred_board == board_present.long()).sum().item())
            agg["pov_correct"] += float((out["pov_logits"].argmax(dim=1) == pov).sum().item())
            agg["stm_correct"] += float((out["stm_logits"].argmax(dim=1) == stm).sum().item())
            total_samples += bx.size(0)

        del loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if total_samples == 0:
        return {"loss": 0.0, "piece_acc": 0.0, "board_acc": 0.0, "pov_acc": 0.0, "stm_acc": 0.0, "score": 0.0}

    piece_acc = agg["tile_correct"] / max(1.0, agg["tile_total"])
    board_acc = agg["board_correct"] / total_samples
    pov_acc = agg["pov_correct"] / total_samples
    stm_acc = agg["stm_correct"] / total_samples
    loss = agg["loss"] / total_samples
    score = piece_acc + 0.25 * board_acc + 0.15 * pov_acc + 0.10 * stm_acc
    return {
        "loss": loss,
        "piece_acc": piece_acc,
        "board_acc": board_acc,
        "pov_acc": pov_acc,
        "stm_acc": stm_acc,
        "score": score,
    }


def train() -> None:
    train_files = sorted(glob.glob(os.path.join(DATA_DIR, "train_*.pt")))
    val_files = sorted(glob.glob(os.path.join(DATA_DIR, "val_*.pt")))
    if not train_files or not val_files:
        raise RuntimeError(f"Missing v7 tensors in {DATA_DIR}. train={len(train_files)} val={len(val_files)}")

    print("\n🚀 STARTING BEAST MODE TRAINING")
    print(f"💻 Hardware: {DEVICE} ({GPU_COUNT} GPUs) | Batch Size: {BATCH_SIZE}")
    print(f"📚 Data Dir: {DATA_DIR} | Base Model: {BASE_MODEL_PATH or '(none)'}")

    model = MultiHeadChessNet()
    load_warmstart(model, BASE_MODEL_PATH)
    if GPU_COUNT > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE)

    ce_piece = nn.CrossEntropyLoss(reduction="none")
    ce_aux = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_score = -1e9
    train.best_hardset = None
    train.consecutive_hardset_regressions = 0
    start_epoch = 0
    if RESUME_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        if GPU_COUNT > 1:
            model.module.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", -1e9))
        print(f"📂 Resumed from checkpoint: {CHECKPOINT_PATH} @ epoch {start_epoch}")
    elif RESUME_FROM_CHECKPOINT:
        print(f"ℹ️ No checkpoint found at {CHECKPOINT_PATH}; using base model warm-start.")

    if USE_EXISTING_MODEL_SAVE_BASELINE and os.path.exists(MODEL_SAVE_PATH) and BACKUP_EXISTING_BEST_MODEL:
        backup_path = MODEL_SAVE_PATH.replace(".pt", "_pretrain_backup.pt")
        if not os.path.exists(backup_path):
            shutil.copy2(MODEL_SAVE_PATH, backup_path)
            print(f"🛡️ Backed up existing best model: {backup_path}")

    baseline_sources = []
    baseline_candidates = []
    if USE_EXISTING_MODEL_SAVE_BASELINE and MODEL_SAVE_PATH:
        baseline_candidates.append(MODEL_SAVE_PATH)
    if BASE_MODEL_PATH:
        baseline_candidates.append(BASE_MODEL_PATH)
    for p in baseline_candidates:
        if p and os.path.exists(p) and p not in baseline_sources:
            baseline_sources.append(p)

    best_baseline = None
    best_baseline_src = None
    for src in baseline_sources:
        score = run_hardset_eval_on_model(src)
        if score is None:
            continue
        if best_baseline is None:
            best_baseline = score
            best_baseline_src = src
            continue
        lhs = (int(score["passed"]), float(score.get("avg_confidence", 0.0)))
        rhs = (int(best_baseline["passed"]), float(best_baseline.get("avg_confidence", 0.0)))
        if lhs > rhs:
            best_baseline = score
            best_baseline_src = src

    if best_baseline is not None:
        train.best_hardset = best_baseline
        print(
            f"📌 Hardset baseline protected from {best_baseline_src}: "
            f"{best_baseline['passed']}/{best_baseline['total']}"
        )
        if best_baseline_src != MODEL_SAVE_PATH:
            shutil.copy2(best_baseline_src, MODEL_SAVE_PATH)
            print(f"📦 Promoted best baseline to save path: {MODEL_SAVE_PATH}")
    else:
        print("⚠️ Hardset baseline unavailable; save gating will use val score only.")

    print(f"🧠 Multi-head v7 config | epochs={EPOCHS} lr={LEARNING_RATE:.2e}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_start = time.time()
        loss_sum = 0.0
        seen = 0

        for i, tf in enumerate(train_files):
            loader = _make_loader(tf, batch_size=BATCH_SIZE, shuffle=True)
            for bx, piece, piece_mask, corners, board_present, pov, stm in loader:
                bx = bx.to(DEVICE)
                piece = piece.to(DEVICE)
                piece_mask = piece_mask.to(DEVICE)
                corners = corners.to(DEVICE)
                board_present = board_present.to(DEVICE)
                pov = pov.to(DEVICE)
                stm = stm.to(DEVICE)

                optimizer.zero_grad()
                out = model(bx)
                total, _metrics = _compute_losses(
                    out=out,
                    piece=piece,
                    piece_mask=piece_mask,
                    corners=corners,
                    board_present=board_present,
                    pov=pov,
                    stm=stm,
                    ce_piece=ce_piece,
                    ce_aux=ce_aux,
                    bce=bce,
                    smooth_l1=smooth_l1,
                )
                total.backward()
                optimizer.step()
                loss_sum += float(total.item()) * bx.size(0)
                seen += bx.size(0)
            del loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (i + 1) % 2 == 0:
                print(f"   Epoch {epoch + 1:03d} | Chunk {i + 1}/{len(train_files)}")

        val = evaluate(model, val_files)
        train_loss = loss_sum / max(1, seen)
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - epoch_start
        print(
            f"✅ EPOCH {epoch + 1:03d} | train_loss={train_loss:.4f} val_loss={val['loss']:.4f} "
            f"piece={val['piece_acc']:.4f} board={val['board_acc']:.4f} pov={val['pov_acc']:.4f} "
            f"stm={val['stm_acc']:.4f} score={val['score']:.4f} lr={lr:.2e} t={elapsed:.1f}s"
        )

        state = model.module.state_dict() if GPU_COUNT > 1 else model.state_dict()
        torch.save(
            {
                "epoch": epoch,
                "model_state": state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
                "val": val,
            },
            CHECKPOINT_PATH,
        )

        if val["score"] > (best_score + MIN_SCORE_IMPROVEMENT):
            best_score = val["score"]
            candidate_path = MODEL_SAVE_PATH.replace(".pt", "_candidate.pt")
            torch.save(state, candidate_path)

            candidate_hardset = run_hardset_eval_on_model(candidate_path)
            accept_candidate = True
            if (
                HARDSET_REQUIRE_NON_REGRESSION
                and train.best_hardset is not None
                and candidate_hardset is not None
                and candidate_hardset["passed"] < train.best_hardset["passed"]
            ):
                accept_candidate = False
                train.consecutive_hardset_regressions += 1
                print(
                    "   ⛔ Rejecting candidate due to hardset regression: "
                    f"{candidate_hardset['passed']}/{candidate_hardset['total']} < "
                    f"{train.best_hardset['passed']}/{train.best_hardset['total']}"
                )
                print(
                    "   ⏱️ Consecutive hardset regressions: "
                    f"{train.consecutive_hardset_regressions}/{HARDSET_MAX_CONSECUTIVE_REGRESSIONS}"
                )

            if accept_candidate:
                os.replace(candidate_path, MODEL_SAVE_PATH)
                if candidate_hardset is not None:
                    train.best_hardset = candidate_hardset
                train.consecutive_hardset_regressions = 0
                print(f"   💾 Best model saved (score: {best_score:.4f})")
            else:
                if os.path.exists(candidate_path):
                    os.remove(candidate_path)
                if train.consecutive_hardset_regressions >= HARDSET_MAX_CONSECUTIVE_REGRESSIONS:
                    print("🛑 Early stop: repeated hardset regressions despite val score improvements.")
                    break

        scheduler.step()

    final_state = model.module.state_dict() if GPU_COUNT > 1 else model.state_dict()
    torch.save(final_state, FINAL_MODEL_SAVE_PATH)
    print(f"\n🎉 Training complete! Final model saved: {FINAL_MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
