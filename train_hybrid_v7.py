#!/usr/bin/env python3
"""Train multi-head v7 model (pieces + geometry + POV + STM)."""

from __future__ import annotations

import gc
import glob
import os
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


# ============ HUMAN CONFIG ============
DATA_DIR = env_str("DATA_DIR", "tensors_v7")
MODEL_SAVE_PATH = env_str("MODEL_SAVE_PATH", "models/model_hybrid_v7_latest_best.pt")
FINAL_MODEL_SAVE_PATH = env_str("FINAL_MODEL_SAVE_PATH", "models/model_hybrid_v7_final.pt")
CHECKPOINT_DIR = env_str("CHECKPOINT_DIR", "models/checkpoints_v7")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.pt")
BASE_MODEL_PATH = env_str("BASE_MODEL_PATH", "models/model_hybrid_v5_latest_best.pt")
EPOCHS = env_int("EPOCHS", 120)
LEARNING_RATE = env_float("LEARNING_RATE", 3e-5)
BATCH_SIZE_PER_GPU = env_int("BATCH_SIZE_PER_GPU", 20)
RESUME_FROM_CHECKPOINT = env_str("RESUME_FROM_CHECKPOINT", "0") == "1"
W_PIECE = env_float("W_PIECE", 1.0)
W_GEOM = env_float("W_GEOM", 0.35)
W_BOARD = env_float("W_BOARD", 0.25)
W_POV = env_float("W_POV", 0.20)
W_STM = env_float("W_STM", 0.15)


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


def load_backbone_from_v5(model: MultiHeadChessNet, model_path: str) -> None:
    if not os.path.exists(model_path):
        print(f"⚠️ Base model not found: {model_path} (training backbone from scratch)")
        return
    payload = torch.load(model_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        payload = payload["model_state"]
    src = _clean_state_dict_keys(payload)
    dst = model.state_dict()
    loaded = 0
    for k, v in src.items():
        if not k.startswith("features."):
            continue
        if k in dst and dst[k].shape == v.shape:
            dst[k] = v
            loaded += 1
    model.load_state_dict(dst, strict=False)
    print(f"📦 Warm-started backbone from {model_path} | loaded feature tensors={loaded}")


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

    model = MultiHeadChessNet()
    load_backbone_from_v5(model, BASE_MODEL_PATH)
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
        print(f"📂 Resumed from {CHECKPOINT_PATH} @ epoch {start_epoch}")

    print(
        f"🚀 v7 train start | device={DEVICE} gpus={GPU_COUNT} batch={BATCH_SIZE} "
        f"epochs={EPOCHS} data={DATA_DIR}"
    )

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

        if val["score"] > best_score:
            best_score = val["score"]
            torch.save(state, MODEL_SAVE_PATH)
            print(f"   💾 best updated -> {MODEL_SAVE_PATH} score={best_score:.4f}")

        scheduler.step()

    final_state = model.module.state_dict() if GPU_COUNT > 1 else model.state_dict()
    torch.save(final_state, FINAL_MODEL_SAVE_PATH)
    print(f"🎉 v7 training complete -> {FINAL_MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
