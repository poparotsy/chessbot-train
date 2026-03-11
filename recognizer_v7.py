#!/usr/bin/env python3
"""Recognizer v7: multi-head model inference (pieces + geometry + POV + STM)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import recognizer_v6 as v6


FEN_CHARS = "1PNBRQKpnbrqk"
DEFAULT_MODEL_CANDIDATES = [
    "models/model_hybrid_v7_latest_best.pt",
    "models/model_hybrid_v7_final.pt",
    "model_hybrid_v7_latest_best.pt",
    "model_hybrid_v7_final.pt",
]

INFER_SIZE = 320
BOARD_PRESENT_THRESHOLD = 0.42
CORNERS_AREA_MIN = 0.08
DEBUG_MODE = False


def _resolve_default_model() -> str | None:
    for p in DEFAULT_MODEL_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


MODEL_PATH = _resolve_default_model()


class MultiHeadChessNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
            nn.Linear(128, 3),
        )
        self.stm_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.features(x)
        piece_logits = self.piece_head(feat).permute(0, 2, 3, 1).reshape(x.size(0), 64, 13)
        g = self.global_pool(feat).flatten(1)
        return {
            "piece_logits": piece_logits,
            "corners": self.geom_head(g),
            "board_logit": self.board_head(g).squeeze(1),
            "pov_logits": self.pov_head(g),
            "stm_logits": self.stm_head(g),
        }


_MODEL_CACHE = {"path": None, "model": None, "device": None}


def _clean_state_dict_keys(state: dict) -> dict:
    out = {}
    for k, v in state.items():
        out[k[7:] if k.startswith("module.") else k] = v
    return out


def _get_model(model_path: str) -> Tuple[nn.Module, torch.device]:
    global _MODEL_CACHE
    if _MODEL_CACHE["path"] == model_path and _MODEL_CACHE["model"] is not None:
        return _MODEL_CACHE["model"], _MODEL_CACHE["device"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadChessNet().to(device)
    payload = torch.load(model_path, map_location=device)
    if isinstance(payload, dict) and "model_state" in payload:
        payload = payload["model_state"]
    model.load_state_dict(_clean_state_dict_keys(payload), strict=False)
    model.eval()

    _MODEL_CACHE = {"path": model_path, "model": model, "device": device}
    return model, device


def _prepare_image(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.resize((INFER_SIZE, INFER_SIZE), Image.LANCZOS), dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    t = torch.from_numpy(arr).float().unsqueeze(0)
    return t


def _labels_to_fen(label_ids: np.ndarray) -> str:
    rows = []
    for r in range(8):
        row = []
        for c in range(8):
            idx = int(label_ids[r * 8 + c])
            row.append(FEN_CHARS[idx])
        rows.append("".join(row))
    fen = "/".join(rows)
    # compress empties
    out_rows = []
    for row in fen.split("/"):
        comp = []
        run = 0
        for ch in row:
            if ch == "1":
                run += 1
            else:
                if run:
                    comp.append(str(run))
                    run = 0
                comp.append(ch)
        if run:
            comp.append(str(run))
        out_rows.append("".join(comp))
    return "/".join(out_rows)


def _predict_once(model: nn.Module, device: torch.device, img: Image.Image) -> dict:
    x = _prepare_image(img).to(device)
    with torch.no_grad():
        out = model(x)
        piece_logits = out["piece_logits"][0]
        piece_probs = torch.softmax(piece_logits, dim=-1)
        piece_ids = torch.argmax(piece_probs, dim=-1).detach().cpu().numpy()
        piece_conf = float(torch.max(piece_probs, dim=-1).values.mean().item())
        fen = _labels_to_fen(piece_ids)

        board_prob = float(torch.sigmoid(out["board_logit"][0]).item())
        corners_norm = out["corners"][0].detach().cpu().numpy().reshape(4, 2)
        corners_norm = np.clip(corners_norm, 0.0, 1.0)
        corners_px = corners_norm.copy()
        corners_px[:, 0] *= (INFER_SIZE - 1)
        corners_px[:, 1] *= (INFER_SIZE - 1)
        corners_px = v6.order_corners(corners_px.astype(np.float32))

        pov_logits = out["pov_logits"][0].detach().cpu().numpy()
        stm_logits = out["stm_logits"][0].detach().cpu().numpy()
        pov_cls = int(np.argmax(pov_logits))
        stm_cls = int(np.argmax(stm_logits))
    return {
        "fen": fen,
        "piece_conf": piece_conf,
        "board_prob": board_prob,
        "corners_px": corners_px,
        "pov_cls": pov_cls,
        "stm_cls": stm_cls,
    }


def _candidate_score(fen: str, conf: float, board_prob: float, geom_quality: float) -> float:
    plaus = v6.board_plausibility_score(fen)
    return float(0.50 * plaus + 0.35 * conf + 0.15 * board_prob + 0.10 * geom_quality)


def predict_board(image_path: str, model_path: str | None = None, board_perspective: str = "auto") -> Tuple[str, float]:
    resolved = model_path or MODEL_PATH
    if not resolved:
        raise FileNotFoundError("No default v7 model found. Provide --model-path.")
    model, device = _get_model(resolved)

    img = Image.open(image_path).convert("RGB")
    img_320 = img.resize((INFER_SIZE, INFER_SIZE), Image.LANCZOS)

    first = _predict_once(model, device, img_320)
    candidates = []
    candidates.append(
        {
            "tag": "full_v7",
            "fen": first["fen"],
            "conf": first["piece_conf"],
            "board_prob": first["board_prob"],
            "geom_quality": 0.0,
            "pov_cls": first["pov_cls"],
            "stm_cls": first["stm_cls"],
        }
    )

    if first["board_prob"] >= BOARD_PRESENT_THRESHOLD:
        metrics = v6.compute_quad_metrics(first["corners_px"], INFER_SIZE, INFER_SIZE)
        if metrics["area_ratio"] >= CORNERS_AREA_MIN:
            warped = v6.perspective_transform(img_320, first["corners_px"])
            second = _predict_once(model, device, warped)
            candidates.append(
                {
                    "tag": "geom_warp_v7",
                    "fen": second["fen"],
                    "conf": second["piece_conf"],
                    "board_prob": second["board_prob"],
                    "geom_quality": v6.warp_geometry_quality(metrics),
                    "pov_cls": second["pov_cls"],
                    "stm_cls": second["stm_cls"],
                }
            )
            inset = v6.inset_board(warped, 2)
            third = _predict_once(model, device, inset)
            candidates.append(
                {
                    "tag": "geom_warp_v7_inset2",
                    "fen": third["fen"],
                    "conf": third["piece_conf"],
                    "board_prob": third["board_prob"],
                    "geom_quality": v6.warp_geometry_quality(metrics),
                    "pov_cls": third["pov_cls"],
                    "stm_cls": third["stm_cls"],
                }
            )

    best = max(candidates, key=lambda c: _candidate_score(c["fen"], c["conf"], c["board_prob"], c["geom_quality"]))

    if board_perspective == "black":
        final_fen = v6.rotate_fen_180(best["fen"])
    elif board_perspective == "white":
        final_fen = best["fen"]
    else:
        # auto: use POV head, fallback to v6 label/piece heuristic.
        if best["pov_cls"] == 1:
            final_fen = v6.rotate_fen_180(best["fen"])
        elif best["pov_cls"] == 0:
            final_fen = best["fen"]
        else:
            fallback = v6.infer_board_perspective_from_piece_distribution(best["fen"])
            final_fen = v6.rotate_fen_180(best["fen"]) if fallback == "black" else best["fen"]

    if DEBUG_MODE:
        print(
            f"DEBUG: v7 selected={best['tag']} conf={best['conf']:.4f} "
            f"board_prob={best['board_prob']:.4f} model={resolved}",
            file=sys.stderr,
        )
    return final_fen, float(best["conf"])


def infer_side_to_move_from_checks(fen_board: str) -> Tuple[str, str]:
    # compatibility for existing rank/eval scripts.
    return v6.infer_side_to_move_from_checks(fen_board)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize chess position from image (v7 multi-head)")
    parser.add_argument("image", help="Path to chess board image")
    parser.add_argument("--model-path", default=None, help="Override v7 model path")
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Infer perspective automatically, or force White/Black at the bottom",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose debug logging")
    args = parser.parse_args()

    DEBUG_MODE = args.debug
    try:
        fen, conf = predict_board(
            args.image,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
        )
        side_to_move, side_source = infer_side_to_move_from_checks(fen)
        print(
            json.dumps(
                {
                    "success": True,
                    "fen": f"{fen} {side_to_move} - - 0 1",
                    "confidence": round(conf, 4),
                    "side_to_move": side_to_move,
                    "side_to_move_source": side_source,
                }
            )
        )
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}))
