#!/usr/bin/env python3
"""Train multi-head v7 model (pieces + geometry + POV + STM)."""

from __future__ import annotations

import gc
import glob
import os
import shutil
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_REV = "v7-train-2026-03-12-r7"


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
MODEL_SAVE_PATH = "models/model_hybrid_v7_latest_best.pt"
FINAL_MODEL_SAVE_PATH = "models/model_hybrid_v7_final.pt"
CHECKPOINT_DIR = "models/checkpoints_v7"
BASE_MODEL_PATH = ""
EPOCHS = 120
LEARNING_RATE = 3e-5
BATCH_SIZE_PER_GPU = 20
RESUME_FROM_CHECKPOINT = True
W_PIECE = 1.0
W_GEOM = 0.35
W_BOARD = 0.25
W_POV = 0.20
W_STM = 0.15
NONEMPTY_TILE_BONUS = 2.5
MIN_SCORE_IMPROVEMENT = 1e-6

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
    NONEMPTY_TILE_BONUS = env_float("NONEMPTY_TILE_BONUS", NONEMPTY_TILE_BONUS)

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
    piece_logits_raw = out["piece_logits"].reshape(-1, 13)
    piece_target = piece.reshape(-1)
    # tile-level CE -> upweight non-empty tiles to avoid empty-dominant optimization.
    tile_ce = ce_piece(piece_logits_raw, piece_target).reshape(bsz, 64)
    nonempty_mask = (piece != 0).float()
    tile_w = 1.0 + NONEMPTY_TILE_BONUS * nonempty_mask
    tile_loss = (tile_ce * tile_w).sum(dim=1) / tile_w.sum(dim=1).clamp(min=1.0)
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
    piece_class_weights = torch.tensor(
        [0.55] + [1.45] * 12,
        dtype=torch.float32,
        device=DEVICE,
    )
    ce_piece = nn.CrossEntropyLoss(reduction="none", weight=piece_class_weights)
    ce_aux = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss(reduction="none")

    total_samples = 0
    agg = {
        "loss": 0.0,
        "tile_correct": 0.0,
        "tile_total": 0.0,
        "nonempty_correct": 0.0,
        "nonempty_total": 0.0,
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
            nonempty = (piece != 0).float()
            nonempty_match = ((pred_piece == piece).float() * nonempty).sum()
            agg["nonempty_correct"] += float(nonempty_match.item())
            agg["nonempty_total"] += float(nonempty.sum().item())

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
        return {
            "loss": 0.0,
            "piece_acc": 0.0,
            "nonempty_acc": 0.0,
            "board_acc": 0.0,
            "pov_acc": 0.0,
            "stm_acc": 0.0,
            "score": 0.0,
        }

    piece_acc = agg["tile_correct"] / max(1.0, agg["tile_total"])
    nonempty_acc = agg["nonempty_correct"] / max(1.0, agg["nonempty_total"])
    board_acc = agg["board_correct"] / total_samples
    pov_acc = agg["pov_correct"] / total_samples
    stm_acc = agg["stm_correct"] / total_samples
    loss = agg["loss"] / total_samples
    score = 0.40 * piece_acc + 0.60 * nonempty_acc + 0.25 * board_acc + 0.15 * pov_acc + 0.10 * stm_acc
    return {
        "loss": loss,
        "piece_acc": piece_acc,
        "nonempty_acc": nonempty_acc,
        "board_acc": board_acc,
        "pov_acc": pov_acc,
        "stm_acc": stm_acc,
        "score": score,
    }


def train() -> int:
    train_files = sorted(glob.glob(os.path.join(DATA_DIR, "train_*.pt")))
    val_files = sorted(glob.glob(os.path.join(DATA_DIR, "val_*.pt")))
    if not train_files or not val_files:
        raise RuntimeError(f"Missing v7 tensors in {DATA_DIR}. train={len(train_files)} val={len(val_files)}")

    print(f"\n🧾 Script revision: {SCRIPT_REV}")
    print("\n🚀 STARTING BEAST MODE TRAINING")
    print(f"💻 Hardware: {DEVICE} ({GPU_COUNT} GPUs) | Batch Size: {BATCH_SIZE}")
    print(f"📚 Data Dir: {DATA_DIR} | Base Model: {BASE_MODEL_PATH or '(none)'}")

    model = MultiHeadChessNet()
    load_warmstart(model, BASE_MODEL_PATH)
    if GPU_COUNT > 1:
        model = nn.DataParallel(model)
    model.to(DEVICE)

    piece_class_weights = torch.tensor(
        [0.55] + [1.45] * 12,
        dtype=torch.float32,
        device=DEVICE,
    )
    ce_piece = nn.CrossEntropyLoss(reduction="none", weight=piece_class_weights)
    ce_aux = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    smooth_l1 = nn.SmoothL1Loss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_score = -1e9
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

    print(f"🧠 Multi-head v7 config | epochs={EPOCHS} lr={LEARNING_RATE:.2e}")

    last_completed_epoch = start_epoch - 1

    try:
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
                f"piece={val['piece_acc']:.4f} nonempty={val['nonempty_acc']:.4f} "
                f"board={val['board_acc']:.4f} pov={val['pov_acc']:.4f} "
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
            last_completed_epoch = epoch

            if val["score"] > (best_score + MIN_SCORE_IMPROVEMENT):
                best_score = val["score"]
                candidate_path = MODEL_SAVE_PATH.replace(".pt", "_candidate.pt")
                torch.save(state, candidate_path)
                os.replace(candidate_path, MODEL_SAVE_PATH)
                print(f"   💾 Best model saved (score: {best_score:.4f})")

            scheduler.step()
    except KeyboardInterrupt:
        state = model.module.state_dict() if GPU_COUNT > 1 else model.state_dict()
        torch.save(
            {
                "epoch": last_completed_epoch,
                "model_state": state,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
                "interrupted": True,
            },
            CHECKPOINT_PATH,
        )
        print("\n🛑 Interrupted by user (Ctrl+C). Checkpoint saved. Exiting cleanly.")
        return 130

    if os.path.exists(MODEL_SAVE_PATH):
        shutil.copy2(MODEL_SAVE_PATH, FINAL_MODEL_SAVE_PATH)
        print(
            f"\n🎉 Training complete! Final model saved from best val: "
            f"{FINAL_MODEL_SAVE_PATH}"
        )
    else:
        final_state = model.module.state_dict() if GPU_COUNT > 1 else model.state_dict()
        torch.save(final_state, FINAL_MODEL_SAVE_PATH)
        print(
            f"\n🎉 Training complete! Final model saved from last epoch state: "
            f"{FINAL_MODEL_SAVE_PATH}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
