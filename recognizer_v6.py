#!/usr/bin/env python3
"""
Chess Recognizer v6 - Consolidated Single-File Version
Combines: edge_grid_detector.py + recognizer_v6_candidate_core.py + recognizer_v6_candidate.py

Fixes Applied:
1. Tile CLAHE (contrast normalization per tile in LAB space) in infer_fen_on_image_clean
2. resolve_candidate_orientation: removed erroneous piece_count_override block that fired
   before the label check and used an absolute ratio threshold (wrong signal, wrong position).
   Replaced with a conservative differential fallback placed correctly after all other methods,
   matching the _core.py reference implementation.
3. _material_score_order: epsilon now reads from CONFIG["material_score_epsilon"] instead of
   being hardcoded, so the CONFIG block is the single source of truth.
4. _full_candidate_should_lead: all thresholds now read from CONFIG instead of being hardcoded
   inline, eliminating dead CONFIG keys.
5. Removed dead frame_ratio variable in build_detector_candidates.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

try:
    import chess
except ModuleNotFoundError:
    chess = None

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================
IMG_SIZE = 64
FEN_CHARS = "1PNBRQKpnbrqk"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
def _resolve_default_model_path():
    model_name = "model_hybrid_v6_champion_48of50.pt"
    search_roots = [
        THIS_DIR,
        os.path.dirname(THIS_DIR),
        os.path.dirname(os.path.dirname(THIS_DIR)),
    ]
    for root in search_roots:
        candidate = os.path.abspath(os.path.join(root, "models", model_name))
        if os.path.exists(candidate):
            return candidate
    return os.path.abspath(os.path.join(THIS_DIR, "models", model_name))


DEFAULT_MODEL_PATH = _resolve_default_model_path()
MODEL_PATH = os.path.abspath(os.environ.get("CHESSBOT_MODEL_PATH", DEFAULT_MODEL_PATH))
USE_EDGE_DETECTION = True
USE_SQUARE_DETECTION = True
DEBUG_MODE = False
CANNY_LOW = 50
CANNY_HIGH = 150
CONTOUR_EPSILON = 0.02
WARP_MIN_AREA_RATIO = 0.30
WARP_MIN_OPPOSITE_SIMILARITY = 0.82
WARP_MIN_ASPECT_SIMILARITY = 0.50
WARP_MIN_ANGLE_DEG = 50.0
WARP_MAX_ANGLE_DEG = 130.0
WARP_RELAXED_MIN_AREA_RATIO = 0.16
WARP_RELAXED_MIN_OPPOSITE_SIMILARITY = 0.55
WARP_RELAXED_MIN_ASPECT_SIMILARITY = 0.18
WARP_RELAXED_MIN_ANGLE_DEG = 35.0
WARP_RELAXED_MAX_ANGLE_DEG = 145.0
TILE_CONTEXT_PAD = 2
PIECE_LOG_PRIOR = -0.20
ORIENTATION_STRONG_PIECE_MARGIN = 2.0
ORIENTATION_WEAK_LABEL_MIN_CONF = 0.70
ORIENTATION_BEST_EFFORT_MIN_MARGIN = 0.15
ORIENTATION_BEST_EFFORT_MIN_PIECE_MARGIN = 1.80
ORIENTATION_BEST_EFFORT_LABEL_SIDE_WEIGHT = 0.95
ORIENTATION_BEST_EFFORT_PIECE_WEIGHT = 0.38
ORIENTATION_BEST_EFFORT_STM_WEIGHT = 0.12
LOW_SAT_SPARSE_SAT_MEAN_MAX = 44.0
LOW_SAT_SPARSE_PIECE_MAX = 10

# Candidate building
MATERIAL_SCORE_EPSILON = 0.02
FULL_CANDIDATE_RATIO_MIN = 0.92
FULL_CANDIDATE_RATIO_MAX = 1.08
FULL_CANDIDATE_COVERAGE_MIN = 0.93
FULL_CANDIDATE_EVIDENCE_MIN = 0.34
FULL_CANDIDATE_SUPPORT_MIN = 0.46
FULL_CANDIDATE_STRONG_COVERAGE_MIN = 0.97
FULL_CANDIDATE_STRONG_EVIDENCE_MIN = 0.50
FULL_CANDIDATE_STRONG_SUPPORT_MIN = 0.95
VALUE_CASE_ALT_MIN = 0.05
VALUE_CASE_CONF_MIN = 0.55
VALUE_CASE_SAT_MEAN_MAX = 72.0
VALUE_CASE_SAT_P95_MAX = 168.0

_model_cache = {}
_plausibility_cache = {}
_side_to_move_cache = {}
_saturation_cache = {}

# =============================================================================
# MODEL DEFINITION
# =============================================================================
class StandaloneBeastClassifier(nn.Module):
    def __init__(self, num_classes=13):
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
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def load_model(model_path=None, device=None):
    resolved_model_path = os.path.abspath(model_path or MODEL_PATH)
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            "Model checkpoint not found. "
            f"Tried: {resolved_model_path}. "
            "Set CHESSBOT_MODEL_PATH or pass --model-path explicitly."
        )
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cached = _model_cache.get((resolved_model_path, str(device)))
    if cached is not None:
        return cached
    model = StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(resolved_model_path, map_location=device))
    model.eval()
    _model_cache[(resolved_model_path, str(device))] = model
    return model

# =============================================================================
# FEN & GEOMETRY UTILITIES
# =============================================================================
def expand_fen_board(fen):
    rows = []
    for row in fen.split("/"):
        expanded = []
        for ch in row:
            if ch.isdigit():
                expanded.extend("1" * int(ch))
            else:
                expanded.append(ch)
        rows.append("".join(expanded))
    return rows

def compress_fen_board(rows):
    return "/".join(re.sub(r"1+", lambda m: str(len(m.group())), row) for row in rows)

def rotate_fen_180(fen):
    rows = expand_fen_board(fen)
    rotated = ["".join(reversed(row)) for row in reversed(rows)]
    return compress_fen_board(rotated)

def parse_fen_board_rows(fen_board):
    rows = []
    for row in fen_board.split("/"):
        expanded = []
        for ch in row:
            if ch.isdigit():
                expanded.extend("1" * int(ch))
            else:
                expanded.append(ch)
        rows.append(expanded)
    if len(rows) != 8 or any(len(row) != 8 for row in rows):
        return None
    return rows

def order_corners(corners):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    return rect

def perspective_transform(img, corners, size=512):
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(np.array(img), matrix, (size, size))
    return Image.fromarray(warped)

def compute_quad_metrics(corners, width, height):
    pts = corners.astype(np.float32)
    top = np.linalg.norm(pts[1] - pts[0])
    right = np.linalg.norm(pts[2] - pts[1])
    bottom = np.linalg.norm(pts[3] - pts[2])
    left = np.linalg.norm(pts[0] - pts[3])
    def safe_ratio(a, b):
        if a <= 1e-6 or b <= 1e-6:
            return 0.0
        return min(a / b, b / a)
    def quad_area(points):
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    def angle_deg(a, b, c):
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 <= 1e-6 or n2 <= 1e-6:
            return 0.0
        cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_t)))
    area_ratio = quad_area(pts) / float(width * height)
    opposite_similarity = safe_ratio(top, bottom) * safe_ratio(left, right)
    aspect_similarity = safe_ratio(top, left) * safe_ratio(right, bottom)
    angles = [
        angle_deg(pts[3], pts[0], pts[1]),
        angle_deg(pts[0], pts[1], pts[2]),
        angle_deg(pts[1], pts[2], pts[3]),
        angle_deg(pts[2], pts[3], pts[0]),
    ]
    return {
        "area_ratio": float(area_ratio),
        "opposite_similarity": float(opposite_similarity),
        "aspect_similarity": float(aspect_similarity),
        "min_angle": float(min(angles)),
        "max_angle": float(max(angles)),
    }

def is_warp_geometry_trustworthy(metrics):
    return (
        metrics["area_ratio"] >= WARP_MIN_AREA_RATIO
        and metrics["opposite_similarity"] >= WARP_MIN_OPPOSITE_SIMILARITY
        and metrics["aspect_similarity"] >= WARP_MIN_ASPECT_SIMILARITY
        and metrics["min_angle"] >= WARP_MIN_ANGLE_DEG
        and metrics["max_angle"] <= WARP_MAX_ANGLE_DEG
    )

def is_warp_geometry_relaxed(metrics):
    return (
        metrics["area_ratio"] >= WARP_RELAXED_MIN_AREA_RATIO
        and metrics["opposite_similarity"] >= WARP_RELAXED_MIN_OPPOSITE_SIMILARITY
        and metrics["aspect_similarity"] >= WARP_RELAXED_MIN_ASPECT_SIMILARITY
        and metrics["min_angle"] >= WARP_RELAXED_MIN_ANGLE_DEG
        and metrics["max_angle"] <= WARP_RELAXED_MAX_ANGLE_DEG
    )

def warp_geometry_quality(metrics):
    area = max(0.0, min(1.0, metrics["area_ratio"] / max(WARP_MIN_AREA_RATIO, 1e-6)))
    opp = max(
        0.0,
        min(1.0, metrics["opposite_similarity"] / max(WARP_MIN_OPPOSITE_SIMILARITY, 1e-6)),
    )
    asp = max(
        0.0,
        min(1.0, metrics["aspect_similarity"] / max(WARP_MIN_ASPECT_SIMILARITY, 1e-6)),
    )
    angle_span = max(0.0, min(1.0, (metrics["max_angle"] - metrics["min_angle"]) / 180.0))
    return float(0.35 * area + 0.30 * opp + 0.25 * asp + 0.10 * (1.0 - angle_span))

def detect_grid_lines(img):
    """Detect 9x9 grid lines. Returns equispaced positions across full image."""
    w, h = img.size
    xe = np.linspace(0, w, 9).astype(int)
    ye = np.linspace(0, h, 9).astype(int)
    return xe, ye

def _labels_to_fen(labels):
    rows = []
    for r in range(8):
        rows.append("".join(labels[r * 8 : (r + 1) * 8]))
    return compress_fen_board(rows)

# =============================================================================
# INFERENCE & ORIENTATION (WITH FIXES)
# =============================================================================
def infer_fen_on_image_clean(
    img,
    model,
    device,
    use_square_detection,
    board_perspective="white",
    topk_k=8,
    return_details=False,
):
    if board_perspective not in {"white", "black"}:
        raise ValueError("board_perspective must be 'white' or 'black'")
    w, h = img.size
    if use_square_detection:
        grid = detect_grid_lines(img)
        if grid is not None:
            xe, ye = grid
        else:
            xe = np.linspace(0, w, 9).astype(int)
            ye = np.linspace(0, h, 9).astype(int)
    else:
        xe = np.linspace(0, w, 9).astype(int)
        ye = np.linspace(0, h, 9).astype(int)
    confs = []
    tile_infos = []
    piece_count = 0
    empty_idx = FEN_CHARS.index("1")
    labels = ["1"] * 64
    tile_tensors = []
    tile_meta = []
    for r in range(8):
        for c in range(8):
            image_r = r
            image_c = c
            if board_perspective == "black":
                image_r = 7 - r
                image_c = 7 - c
            x0 = max(0, int(xe[image_c] - TILE_CONTEXT_PAD))
            y0 = max(0, int(ye[image_r] - TILE_CONTEXT_PAD))
            x1 = min(w, int(xe[image_c + 1] + TILE_CONTEXT_PAD))
            y1 = min(h, int(ye[image_r + 1] + TILE_CONTEXT_PAD))
            tile = img.crop((x0, y0, x1, y1)).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
            img_np = np.array(tile).transpose(2, 0, 1)
            tensor = (torch.from_numpy(img_np).float() / 127.5) - 1.0
            tile_tensors.append(tensor)
            tile_meta.append((r, c))
    with torch.no_grad():
        batch = torch.stack(tile_tensors, dim=0).to(device)
        out_batch = torch.softmax(model(batch), dim=1)
        for idx, (r, c) in enumerate(tile_meta):
            out = out_batch[idx]
            topk_probs, topk_pred = torch.topk(out, k=min(topk_k, len(FEN_CHARS)))
            topk = [
                (FEN_CHARS[int(k_idx.item())], float(prob.item()))
                for prob, k_idx in zip(topk_probs, topk_pred)
            ]
            adjusted_scores = torch.log(out + 1e-12) + PIECE_LOG_PRIOR
            adjusted_scores[empty_idx] = torch.log(out[empty_idx] + 1e-12)
            pred_idx = int(torch.argmax(adjusted_scores).item())
            pred_prob = float(out[pred_idx].item())
            label = "1" if pred_prob < 0.35 else FEN_CHARS[pred_idx]
            labels[r * 8 + c] = label
            if label != "1":
                piece_count += 1
            max_prob = float(torch.max(out).item())
            confs.append(max_prob)
            empty_prob = float(out[empty_idx].item())
            best_piece_alt_prob = 0.0
            for alt_label, alt_prob in topk:
                if alt_label == "1":
                    continue
                best_piece_alt_prob = max(best_piece_alt_prob, float(alt_prob))
            tile_infos.append(
                {
                    "row": r,
                    "col": c,
                    "label": label,
                    "prob": max_prob,
                    "top1_prob": max_prob,
                    "empty_prob": empty_prob,
                    "best_piece_alt_prob": best_piece_alt_prob,
                    "topk": topk,
                }
            )
    fen = _labels_to_fen(labels)
    final_conf = float(np.mean(confs))
    details = {"base_fen": fen, "final_fen": fen, "tile_infos": tile_infos}
    if return_details:
        return fen, final_conf, piece_count, details
    return fen, final_conf, piece_count

def extract_file_label_crop(img, side="left"):
    cell = min(img.size) / 8.0
    y0 = img.size[1] - int(cell)
    if side == "left":
        x0 = 0
        x1 = int(cell)
    elif side == "right":
        x0 = img.size[0] - int(cell)
        x1 = img.size[0]
    else:
        raise ValueError("side must be 'left' or 'right'")
    cell_crop = img.crop((x0, y0, x1, img.size[1]))
    return cell_crop.crop((int(cell * 0.74), int(cell * 0.72), int(cell * 0.98), int(cell * 0.98)))

def classify_file_label_crop(crop):
    gray = np.array(crop.convert("L"))
    h, w = gray.shape
    best = None
    for threshold_mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, binary = cv2.threshold(gray, 0, 255, threshold_mode | cv2.THRESH_OTSU)
        num_components, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        for component_id in range(1, num_components):
            x, y, bw, bh, area = stats[component_id]
            area_ratio = area / float(h * w)
            aspect = bw / max(bh, 1)
            if area_ratio < 0.04 or area_ratio > 0.45:
                continue
            if bw >= w * 0.8 or bh >= h * 0.9:
                continue
            if aspect < 0.15 or aspect > 1.2:
                continue
            center_dist = abs((x + bw / 2) - w * 0.55) + abs((y + bh / 2) - h * 0.55)
            score = area_ratio * 100.0 - center_dist * 0.12
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "binary": binary,
                    "component_id": component_id,
                    "bbox": (int(x), int(y), int(bw), int(bh)),
                    "area_ratio": float(area_ratio),
                    "aspect": float(aspect),
                }
    if best is None:
        return None
    _, labels, _, _ = cv2.connectedComponentsWithStats(best["binary"], 8)
    mask = np.zeros_like(best["binary"])
    mask[labels == best["component_id"]] = 255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for contour_idx in range(len(contours)):
            if hierarchy[contour_idx][3] != -1:
                holes += 1
    label = "a" if holes >= 1 else "h"
    confidence = 0.55 + min(0.35, best["area_ratio"]) + min(0.10, holes * 0.08)
    return {
        "label": label,
        "confidence": min(0.99, float(confidence)),
        "holes": holes,
        "bbox": best["bbox"],
        "aspect": best["aspect"],
    }

def infer_board_perspective_from_labels(img):
    left_label = classify_file_label_crop(extract_file_label_crop(img, side="left"))
    right_label = classify_file_label_crop(extract_file_label_crop(img, side="right"))
    details = {"left": left_label, "right": right_label}
    if left_label and right_label:
        if left_label["confidence"] >= 0.72 and right_label["confidence"] >= 0.72:
            if left_label["label"] == "a" and right_label["label"] == "h":
                return {"perspective": "white", "source": "board_labels", "details": details}
            if left_label["label"] == "h" and right_label["label"] == "a":
                return {"perspective": "black", "source": "board_labels", "details": details}
    return None

def infer_board_perspective_from_piece_distribution(fen_board, threshold=2.0):
    if chess:
        try:
            board = chess.Board(fen_board)
            white_pos = []
            black_pos = []
            for sq in chess.SQUARES:
                piece = board.piece_at(sq)
                if piece:
                    r = 7 - (sq // 8)
                    if piece.color == chess.WHITE:
                        white_pos.append(r)
                    else:
                        black_pos.append(r)
            if white_pos and black_pos:
                white_mean = sum(white_pos) / len(white_pos)
                black_mean = sum(black_pos) / len(black_pos)
                return "black" if (black_mean - white_mean) > threshold else "white"
        except Exception:
            pass
    rows = expand_fen_board(fen_board)
    white_rows = []
    black_rows = []
    for r, row in enumerate(rows):
        for ch in row:
            if ch == "1":
                continue
            if ch.isupper():
                white_rows.append(r)
            else:
                black_rows.append(r)
    if not white_rows or not black_rows:
        return None
    white_mean = sum(white_rows) / len(white_rows)
    black_mean = sum(black_rows) / len(black_rows)
    return "black" if (black_mean - white_mean) > threshold else "white"

def orientation_piece_margin(fen_board):
    rows = expand_fen_board(fen_board)
    white_rows = []
    black_rows = []
    for r, row in enumerate(rows):
        for ch in row:
            if ch == "1":
                continue
            if ch.isupper():
                white_rows.append(r)
            else:
                black_rows.append(r)
    if not white_rows or not black_rows:
        return 0.0
    return float(abs((sum(white_rows) / len(white_rows)) - (sum(black_rows) / len(black_rows))))

def collect_orientation_context(candidates, board_perspective):
    context = {
        "label_perspective_result": None,
        "label_details": {"left": None, "right": None},
        "labels_absent": True,
        "labels_same": False,
        "partial_label_scores": {"white": 0.0, "black": 0.0},
    }
    if board_perspective != "auto" or not candidates:
        return context
    full_candidate = next((candidate for candidate in candidates if str(candidate[0]) == "full"), candidates[0])
    full_candidate_img = full_candidate[1].copy()
    label_perspective_result = infer_board_perspective_from_labels(full_candidate_img)
    if label_perspective_result:
        label_details = label_perspective_result["details"]
    else:
        label_details = {
            "left": classify_file_label_crop(extract_file_label_crop(full_candidate_img, side="left")),
            "right": classify_file_label_crop(extract_file_label_crop(full_candidate_img, side="right")),
        }
    labels_absent = label_details["left"] is None and label_details["right"] is None
    labels_same = (
        label_details["left"] is not None
        and label_details["right"] is not None
        and label_details["left"]["label"] == label_details["right"]["label"]
    )
    context.update(
        {
            "label_perspective_result": label_perspective_result,
            "label_details": label_details,
            "labels_absent": labels_absent,
            "labels_same": labels_same,
            "partial_label_scores": orientation_partial_label_scores(label_details),
        }
    )
    return context

def orientation_partial_label_scores(label_details):
    scores = {"white": 0.0, "black": 0.0}
    left_label = label_details.get("left")
    right_label = label_details.get("right")
    if left_label is not None:
        left_char = str(left_label.get("label", "")).lower()
        left_conf = float(left_label.get("confidence", 0.0))
        if left_char == "a":
            scores["white"] += left_conf * ORIENTATION_BEST_EFFORT_LABEL_SIDE_WEIGHT
        elif left_char == "h":
            scores["black"] += left_conf * ORIENTATION_BEST_EFFORT_LABEL_SIDE_WEIGHT
    if right_label is not None:
        right_char = str(right_label.get("label", "")).lower()
        right_conf = float(right_label.get("confidence", 0.0))
        if right_char == "h":
            scores["white"] += right_conf * ORIENTATION_BEST_EFFORT_LABEL_SIDE_WEIGHT
        elif right_char == "a":
            scores["black"] += right_conf * ORIENTATION_BEST_EFFORT_LABEL_SIDE_WEIGHT
    return scores

def score_orientation_best_effort(fen, orientation_context):
    scores = dict(orientation_context.get("partial_label_scores") or {"white": 0.0, "black": 0.0})
    piece_margin = orientation_piece_margin(fen)
    piece_guess = infer_board_perspective_from_piece_distribution(fen, threshold=0.0)
    if piece_guess in {"white", "black"}:
        scores[piece_guess] += min(1.35, piece_margin * ORIENTATION_BEST_EFFORT_PIECE_WEIGHT)
    white_stm, white_src = infer_side_to_move_from_checks(fen)
    black_stm, black_src = infer_side_to_move_from_checks(rotate_fen_180(fen))
    if white_src not in {"no_check_signal", "double_check_conflict", "invalid_board", "missing_king"}:
        scores["white"] += ORIENTATION_BEST_EFFORT_STM_WEIGHT
    if black_src not in {"no_check_signal", "double_check_conflict", "invalid_board", "missing_king"}:
        scores["black"] += ORIENTATION_BEST_EFFORT_STM_WEIGHT
    return {
        "scores": {"white": float(scores["white"]), "black": float(scores["black"])},
        "piece_guess": piece_guess,
        "piece_margin": float(piece_margin),
        "stm_sources": {"white": white_src, "black": black_src},
    }

def resolve_candidate_orientation(fen, board_perspective, orientation_context):
    if board_perspective in {"white", "black"}:
        return board_perspective, "override"
    if board_perspective != "auto":
        raise ValueError("board_perspective must be 'auto', 'white', or 'black'")
    label_result = orientation_context["label_perspective_result"]
    label_details = orientation_context["label_details"]
    labels_absent = orientation_context["labels_absent"]
    labels_same = orientation_context["labels_same"]

    # --- Tier 1: strong board labels (high-confidence a/h file markers) ---
    if label_result is not None:
        return label_result["perspective"], label_result["source"]

    # --- Tier 2: weak-label fallback (both sides present, lower confidence) ---
    left_label = label_details.get("left")
    right_label = label_details.get("right")
    if left_label is not None and right_label is not None:
        left_char = str(left_label.get("label", "")).lower()
        right_char = str(right_label.get("label", "")).lower()
        min_conf = min(float(left_label.get("confidence", 0.0)), float(right_label.get("confidence", 0.0)))
        if min_conf >= ORIENTATION_WEAK_LABEL_MIN_CONF:
            if left_char == "a" and right_char == "h":
                return "white", "weak_label_fallback"
            if left_char == "h" and right_char == "a":
                return "black", "weak_label_fallback"

    # --- Tier 3: strong spatial piece distribution (only when labels absent or ambiguous) ---
    piece_margin = orientation_piece_margin(fen)
    can_use_piece_distribution = labels_absent or labels_same
    if piece_margin >= ORIENTATION_STRONG_PIECE_MARGIN and can_use_piece_distribution:
        fallback = infer_board_perspective_from_piece_distribution(
            fen,
            threshold=ORIENTATION_STRONG_PIECE_MARGIN,
        )
        if fallback == "black":
            return "black", "piece_distribution_fallback"
        return "white", "piece_distribution_fallback"

    # --- Tier 4: best-effort scoring (partial label + piece margin + check signal) ---
    has_partial_labels = (left_label is not None or right_label is not None) and not (
        left_label is not None and right_label is not None
    )
    partial_scores = orientation_context.get("partial_label_scores") or {"white": 0.0, "black": 0.0}
    partial_label_guess = None
    if float(partial_scores.get("white", 0.0)) > float(partial_scores.get("black", 0.0)):
        partial_label_guess = "white"
    elif float(partial_scores.get("black", 0.0)) > float(partial_scores.get("white", 0.0)):
        partial_label_guess = "black"
    piece_guess = infer_board_perspective_from_piece_distribution(fen, threshold=0.0)
    if (
        has_partial_labels
        and piece_margin >= ORIENTATION_BEST_EFFORT_MIN_PIECE_MARGIN
        and partial_label_guess in {"white", "black"}
        and piece_guess in {"white", "black"}
        and partial_label_guess != piece_guess
    ):
        best_effort = score_orientation_best_effort(fen, orientation_context)
        white_score = float(best_effort["scores"]["white"])
        black_score = float(best_effort["scores"]["black"])
        if abs(white_score - black_score) >= ORIENTATION_BEST_EFFORT_MIN_MARGIN:
            return ("black", "best_effort_orientation") if black_score > white_score else ("white", "best_effort_orientation")

    # --- Tier 5: plausibility fallback — try both orientations, pick more plausible ---
    white_plaus = board_plausibility_score(fen)
    black_plaus = board_plausibility_score(rotate_fen_180(fen))
    if black_plaus > white_plaus + 1.0:
        return "black", "plausibility_fallback"
    if white_plaus > black_plaus + 1.0:
        return "white", "plausibility_fallback"

    return "white", "default"

def board_plausibility_score(fen_board):
    if fen_board in _plausibility_cache:
        return _plausibility_cache[fen_board]
    score = 0.0
    if chess:
        try:
            board = chess.Board(fen_board)
            score = 10.0
            if len(board.pieces(chess.KING, chess.WHITE)) != 1:
                score -= 5.0
            if len(board.pieces(chess.KING, chess.BLACK)) != 1:
                score -= 5.0
            white_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
            black_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
            if white_pawns > 8 or black_pawns > 8:
                score -= 2.0
            _plausibility_cache[fen_board] = score
            return score
        except Exception:
            pass
    rows = expand_fen_board(fen_board)
    if len(rows) != 8 or any(len(row) != 8 for row in rows):
        _plausibility_cache[fen_board] = -1e9
        return -1e9
    white_king = fen_board.count("K")
    black_king = fen_board.count("k")
    white_pawns = fen_board.count("P")
    black_pawns = fen_board.count("p")
    white_queens = fen_board.count("Q")
    black_queens = fen_board.count("q")
    white_pieces = sum(1 for ch in fen_board if ch.isupper())
    black_pieces = sum(1 for ch in fen_board if ch.islower())
    total_pieces = white_pieces + black_pieces
    pawns_on_back_rank = sum(1 for ch in (rows[0] + rows[7]) if ch in {"P", "p"})
    score += 6.0 if white_king == 1 else -20.0 * abs(white_king - 1)
    score += 6.0 if black_king == 1 else -20.0 * abs(black_king - 1)
    score += 1.5 if white_pawns <= 8 else -5.0 * (white_pawns - 8)
    score += 1.5 if black_pawns <= 8 else -5.0 * (black_pawns - 8)
    score += 1.0 if white_pieces <= 16 else -2.0 * (white_pieces - 16)
    score += 1.0 if black_pieces <= 16 else -2.0 * (black_pieces - 16)
    score += 0.5 if white_queens <= 2 else -3.0 * (white_queens - 2)
    score += 0.5 if black_queens <= 2 else -3.0 * (black_queens - 2)
    score += 1.0 if total_pieces <= 32 else -2.0 * (total_pieces - 32)
    score -= 2.0 * pawns_on_back_rank
    _plausibility_cache[fen_board] = score
    return score

def image_saturation_stats(img):
    img_hash = hashlib.md5(np.array(img).tobytes()).hexdigest()
    cached = _saturation_cache.get(img_hash)
    if cached is not None:
        return cached
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    stats = {
        "sat_mean": float(np.mean(sat)),
        "sat_std": float(np.std(sat)),
        "val_mean": float(np.mean(val)),
        "val_std": float(np.std(val)),
    }
    _saturation_cache[img_hash] = stats
    return stats

def king_health(fen_board):
    rows = expand_fen_board(fen_board)
    if len(rows) != 8 or any(len(r) != 8 for r in rows):
        return 0
    return int(fen_board.count("K") == 1) + int(fen_board.count("k") == 1)


def board_is_structurally_valid(fen_board):
    rows = expand_fen_board(fen_board)
    if len(rows) != 8 or any(len(r) != 8 for r in rows):
        return False
    if king_health(fen_board) < 2:
        return False
    if chess is None:
        return True
    for stm in ("w", "b"):
        try:
            board = chess.Board(f"{fen_board} {stm} - - 0 1")
        except ValueError:
            continue
        if board.is_valid():
            return True
    return False

def is_square_attacked(rows, target_r, target_c, by_white):
    if by_white:
        pawn_attackers = [("P", target_r + 1, target_c - 1), ("P", target_r + 1, target_c + 1)]
    else:
        pawn_attackers = [("p", target_r - 1, target_c - 1), ("p", target_r - 1, target_c + 1)]
    for piece, r, c in pawn_attackers:
        if 0 <= r < 8 and 0 <= c < 8 and rows[r][c] == piece:
            return True
    knight = "N" if by_white else "n"
    for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
        r, c = target_r + dr, target_c + dc
        if 0 <= r < 8 and 0 <= c < 8 and rows[r][c] == knight:
            return True
    king = "K" if by_white else "k"
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = target_r + dr, target_c + dc
            if 0 <= r < 8 and 0 <= c < 8 and rows[r][c] == king:
                return True
    rook_queen = {"R", "Q"} if by_white else {"r", "q"}
    bishop_queen = {"B", "Q"} if by_white else {"b", "q"}
    for dr, dc, attackers in (
        (-1, 0, rook_queen),
        (1, 0, rook_queen),
        (0, -1, rook_queen),
        (0, 1, rook_queen),
        (-1, -1, bishop_queen),
        (-1, 1, bishop_queen),
        (1, -1, bishop_queen),
        (1, 1, bishop_queen),
    ):
        r, c = target_r + dr, target_c + dc
        while 0 <= r < 8 and 0 <= c < 8:
            sq = rows[r][c]
            if sq != "1":
                if sq in attackers:
                    return True
                break
            r += dr
            c += dc
    return False

def infer_side_to_move_from_checks(fen_board):
    cached = _side_to_move_cache.get(fen_board)
    if cached is not None:
        return cached
    rows = parse_fen_board_rows(fen_board)
    if rows is None:
        result = ("w", "invalid_board")
        _side_to_move_cache[fen_board] = result
        return result
    white_king = None
    black_king = None
    for r in range(8):
        for c in range(8):
            if rows[r][c] == "K":
                white_king = (r, c)
            elif rows[r][c] == "k":
                black_king = (r, c)
    if white_king is None or black_king is None:
        result = ("w", "missing_king")
        _side_to_move_cache[fen_board] = result
        return result
    white_in_check = is_square_attacked(rows, white_king[0], white_king[1], by_white=False)
    black_in_check = is_square_attacked(rows, black_king[0], black_king[1], by_white=True)
    if white_in_check and not black_in_check:
        result = ("w", "check_inference")
        _side_to_move_cache[fen_board] = result
        return result
    if black_in_check and not white_in_check:
        result = ("b", "check_inference")
        _side_to_move_cache[fen_board] = result
        return result
    if white_in_check and black_in_check:
        result = ("w", "double_check_conflict")
        _side_to_move_cache[fen_board] = result
        return result
    if chess is not None:
        legal_turns = []
        for stm in ("w", "b"):
            try:
                board = chess.Board(f"{fen_board} {stm} - - 0 1")
            except ValueError:
                continue
            if not board.is_valid():
                continue
            legal_turns.append((stm, board.legal_moves.count(), board.is_checkmate(), board.is_stalemate()))
        if len(legal_turns) == 1:
            result = (legal_turns[0][0], "legality_fallback")
            _side_to_move_cache[fen_board] = result
            return result
        if len(legal_turns) == 2:
            white_state = next(item for item in legal_turns if item[0] == "w")
            black_state = next(item for item in legal_turns if item[0] == "b")
            if white_state[1] == 0 and black_state[1] > 0:
                result = ("b", "legality_fallback")
                _side_to_move_cache[fen_board] = result
                return result
            if black_state[1] == 0 and white_state[1] > 0:
                result = ("w", "legality_fallback")
                _side_to_move_cache[fen_board] = result
                return result
            if white_state[2] and not black_state[2]:
                result = ("w", "checkmate_fallback")
                _side_to_move_cache[fen_board] = result
                return result
            if black_state[2] and not white_state[2]:
                result = ("b", "checkmate_fallback")
                _side_to_move_cache[fen_board] = result
                return result
            if white_state[3] and not black_state[3]:
                result = ("w", "stalemate_fallback")
                _side_to_move_cache[fen_board] = result
                return result
            if black_state[3] and not white_state[3]:
                result = ("b", "stalemate_fallback")
                _side_to_move_cache[fen_board] = result
                return result
    result = ("w", "no_check_signal")
    _side_to_move_cache[fen_board] = result
    return result


def infer_unique_en_passant_square(fen_board, side_to_move):
    if chess is None:
        return "-"
    rows = parse_fen_board_rows(fen_board)
    if rows is None:
        return "-"

    targets = set()
    if side_to_move == "w":
        row = 3  # rank 5
        for col in range(8):
            if rows[row][col] != "P":
                continue
            for dc in (-1, 1):
                adj = col + dc
                if adj < 0 or adj >= 8:
                    continue
                if rows[row][adj] != "p":
                    continue
                if rows[row - 1][adj] != "1":
                    continue
                targets.add(f"{chr(97 + adj)}{8 - (row - 1)}")
    elif side_to_move == "b":
        row = 4  # rank 4
        for col in range(8):
            if rows[row][col] != "p":
                continue
            for dc in (-1, 1):
                adj = col + dc
                if adj < 0 or adj >= 8:
                    continue
                if rows[row][adj] != "P":
                    continue
                if rows[row + 1][adj] != "1":
                    continue
                targets.add(f"{chr(97 + adj)}{8 - (row + 1)}")
    else:
        return "-"

    legal_targets = []
    for target in sorted(targets):
        try:
            board = chess.Board(f"{fen_board} {side_to_move} - {target} 0 1")
        except ValueError:
            continue
        if any(board.is_en_passant(move) for move in board.legal_moves):
            legal_targets.append(target)

    return legal_targets[0] if len(legal_targets) == 1 else "-"

def _should_try_inner_crop_rescue(img, board_fen):
    width, height = img.size
    if width < 256 or height < 256:
        return False
    aspect = width / max(float(height), 1.0)
    if aspect < 0.84 or aspect > 1.16:
        return False
    return king_health(board_fen) < 2

def _decode_best_candidate(img, model, device, board_perspective):
    candidates = build_detector_candidates(img)
    orientation_context = collect_orientation_context(candidates, board_perspective=board_perspective)
    best = decode_candidate(
        candidates[0],
        model=model,
        device=device,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    return best, candidates, orientation_context


def _iter_square_grid_box_rescue_crops(img):
    width, height = img.size
    aspect = width / max(float(height), 1.0)
    if aspect < 0.84 or aspect > 1.16:
        return
    full_meta = score_full_frame_board(img)
    grid_box = full_meta.get("grid_box")
    if grid_box is None:
        return
    x0, y0, x1, y1 = [float(v) for v in grid_box]
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    if min(box_w, box_h) < min(width, height) * 0.45:
        return

    side_base = min(float(min(width, height)), max(box_w, box_h))
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    seen = set()

    for scale in (0.96, 1.0, 1.04, 1.08):
        side = int(round(side_base * scale))
        if side < 220:
            continue
        side = min(side, width, height)
        max_x = max(0, width - side)
        max_y = max(0, height - side)
        base_x = max(0, min(int(round(center_x - (side / 2.0))), max_x))
        base_y = max(0, min(int(round(center_y - (side / 2.0))), max_y))
        x_offsets = {
            base_x,
            max(0, min(int(round(x0)), max_x)),
            max(0, min(int(round(x1 - side)), max_x)),
        }
        y_offsets = {
            base_y,
            max(0, min(int(round(y0)), max_y)),
            max(0, min(int(round(y1 - side)), max_y)),
        }
        pair_offsets = {(base_x, base_y)}
        pair_offsets.update((px, base_y) for px in x_offsets)
        pair_offsets.update((base_x, py) for py in y_offsets)
        for px in sorted(x_offsets):
            for py in sorted(y_offsets):
                pair_offsets.add((px, py))
        for px, py in sorted(pair_offsets):
            key = (px, py, side)
            if key in seen:
                continue
            seen.add(key)
            crop = img.crop((px, py, px + side, py + side))
            if crop.size == img.size:
                continue
            yield crop.resize(img.size, Image.LANCZOS)


def _decode_direct_rescue_crop(crop, model, device, board_perspective):
    fen, conf, piece_count = infer_fen_on_image_clean(
        crop,
        model,
        device,
        USE_SQUARE_DETECTION,
    )
    orientation_context = {
        "label_perspective_result": None,
        "label_details": {},
        "labels_absent": True,
        "labels_same": False,
        "partial_label_scores": {"white": 0.0, "black": 0.0},
    }
    detected_perspective, perspective_source = resolve_candidate_orientation(
        fen,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    board_fen = rotate_fen_180(fen) if detected_perspective == "black" else fen
    return (
        "inner_grid_box",
        "base",
        board_fen,
        float(conf),
        int(piece_count),
        detected_perspective,
        perspective_source,
        1.0,
        True,
        1.0,
        1.0,
    )


def _try_inner_crop_rescue(img, model, device, board_perspective):
    width, height = img.size
    best_rescue = None
    for crop in _iter_square_grid_box_rescue_crops(img) or ():
        try:
            decoded = _decode_direct_rescue_crop(
                crop,
                model=model,
                device=device,
                board_perspective=board_perspective,
            )
        except Exception:
            continue
        if king_health(decoded[2]) < 2:
            continue
        if best_rescue is None or float(decoded[3]) > float(best_rescue[3]):
            best_rescue = decoded
    for ratio in (0.025, 0.035, 0.045, 0.055, 0.065):
        margin_x = max(4, int(round(width * ratio)))
        margin_y = max(4, int(round(height * ratio)))
        crop_w = width - (margin_x * 2)
        crop_h = height - (margin_y * 2)
        if crop_w < 192 or crop_h < 192:
            continue
        crop = img.crop((margin_x, margin_y, width - margin_x, height - margin_y))
        try:
            decoded, _, _ = _decode_best_candidate(
                crop,
                model=model,
                device=device,
                board_perspective=board_perspective,
            )
        except Exception:
            continue
        if king_health(decoded[2]) < 2:
            continue
        if best_rescue is None or float(decoded[3]) > float(best_rescue[3]):
            best_rescue = decoded
    return best_rescue

def parse_side_to_move_override(raw_value):
    token = str(raw_value).strip().lower()
    mapping = {
        "w": "w",
        "white": "w",
        "wtm": "w",
        "white_to_move": "w",
        "b": "b",
        "black": "b",
        "blak": "b",
        "btm": "b",
        "black_to_move": "b",
    }
    if token not in mapping:
        raise ValueError(f"Invalid --side-to-move '{raw_value}'. Use one of: wtm|btm|w|b|white|black")
    return mapping[token]

# =============================================================================
# EDGE GRID DETECTOR
# =============================================================================
def _gray_rgb(img):
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray

def _smooth_1d(signal, kernel):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size < kernel or kernel <= 1:
        return signal
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(signal.reshape(-1, 1), (1, kernel), 0).reshape(-1)

def _normalize_signal(signal):
    signal = np.asarray(signal, dtype=np.float32)
    signal -= float(signal.min())
    vmax = float(signal.max())
    if vmax <= 1e-6:
        return signal
    return signal / vmax

def _sample_signal(signal, center, radius):
    lo = max(0, int(round(center - radius)))
    hi = min(len(signal), int(round(center + radius + 1)))
    if hi <= lo:
        return 0.0
    return float(np.mean(signal[lo:hi]))

def _search_axis_grid(signal):
    n = int(len(signal))
    if n < 64:
        return None
    smooth = _smooth_1d(signal, max(9, (n // 48) * 2 + 1))
    smooth = _normalize_signal(smooth)
    low_step = max(8, int(round(n * 0.085)))
    high_step = min(int(round(n * 0.18)), n // 2)
    best = None
    radius = max(1.0, n / 256.0)

    for step in range(low_step, max(low_step + 1, high_step + 1)):
        max_start = n - 1 - (8 * step)
        if max_start < 0:
            continue
        start_count = min(32, max(1, max_start + 1))
        starts = np.linspace(0, max_start, start_count)
        for start in starts:
            positions = [start + (i * step) for i in range(9)]
            values = np.array([_sample_signal(smooth, pos, radius) for pos in positions], dtype=np.float32)
            line_score = float(np.mean(values))
            support_ratio = float(np.mean(values >= max(0.25, float(np.mean(smooth)))))
            border_strength = float((values[0] + values[-1]) * 0.5)
            regularity = float(1.0 - min(1.0, np.std(np.diff(positions)) / max(1.0, step)))
            score = (0.55 * line_score) + (0.25 * support_ratio) + (0.15 * border_strength) + (0.05 * regularity)
            row = {
                "start": float(start),
                "step": float(step),
                "positions": positions,
                "line_score": line_score,
                "support_ratio": support_ratio,
                "border_strength": border_strength,
                "regularity": regularity,
                "score": score,
            }
            if best is None or row["score"] > best["score"]:
                best = row
    return best

def evaluate_warped_grid(img):
    _, gray = _gray_rgb(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)
    gx = np.abs(cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3))
    col_signal = _normalize_signal(np.mean(gx, axis=0))
    row_signal = _normalize_signal(np.mean(gy, axis=1))
    x_grid = _search_axis_grid(col_signal)
    y_grid = _search_axis_grid(row_signal)
    if x_grid is None or y_grid is None:
        return {
            "evidence": 0.0,
            "support_ratio": 0.0,
            "score": 0.0,
            "grid_box": None,
            "x_grid": x_grid,
            "y_grid": y_grid,
        }
    x0 = x_grid["start"]
    x1 = x_grid["start"] + (8.0 * x_grid["step"])
    y0 = y_grid["start"]
    y1 = y_grid["start"] + (8.0 * y_grid["step"])
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    ratio = min(width / height, height / width)
    coverage = min(1.0, (width * height) / float(img.size[0] * img.size[1]))
    evidence = float(
        (0.35 * x_grid["line_score"])
        + (0.35 * y_grid["line_score"])
        + (0.15 * x_grid["border_strength"])
        + (0.15 * y_grid["border_strength"])
    )
    support_ratio = float(0.5 * (x_grid["support_ratio"] + y_grid["support_ratio"]))
    score = float((0.55 * evidence) + (0.20 * support_ratio) + (0.15 * ratio) + (0.10 * coverage))
    return {
        "evidence": evidence,
        "support_ratio": support_ratio,
        "score": score,
        "coverage": coverage,
        "ratio": ratio,
        "grid_box": (float(x0), float(y0), float(x1), float(y1)),
        "x_grid": x_grid,
        "y_grid": y_grid,
    }

def score_full_frame_board(img):
    meta = evaluate_warped_grid(img)
    ratio = img.size[0] / float(max(1, img.size[1]))
    ratio_score = min(ratio, 1.0 / max(ratio, 1e-6))
    score = float((0.70 * meta["score"]) + (0.30 * ratio_score))
    trusted = bool(
        0.88 <= ratio <= 1.12
        and meta["evidence"] >= 0.34
        and meta["support_ratio"] >= 0.46
    )
    return {
        **meta,
        "score": score,
        "trusted": trusted,
        "ratio_score": ratio_score,
    }

def crop_warp_to_detected_grid(img):
    meta = evaluate_warped_grid(img)
    box = meta["grid_box"]
    if box is None:
        return None
    x0, y0, x1, y1 = box
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    coverage = (width * height) / float(img.size[0] * img.size[1])
    aspect = img.size[0] / float(max(1, img.size[1]))
    non_square = not (0.92 <= aspect <= 1.08)
    ratio = width / float(max(1.0, height))
    if not non_square:
        if coverage < 0.55 or coverage > 0.995:
            return None
        if not (0.88 <= ratio <= 1.12):
            return None
    else:
        if coverage < 0.45 or coverage > 0.995:
            return None
        margins = np.array(
            [x0, y0, max(0.0, img.size[0] - x1), max(0.0, img.size[1] - y1)],
            dtype=np.float32,
        )
        cell_est = min(img.size) / 8.0
        if (
            float(np.max(margins)) >= (cell_est * 0.90)
            and float(np.min(margins)) <= (cell_est * 0.20)
            and (
                abs(float(margins[0] - margins[2])) >= (cell_est * 0.90)
                or abs(float(margins[1] - margins[3])) >= (cell_est * 0.90)
            )
        ):
            return None
    px0 = max(0, int(round(x0)))
    py0 = max(0, int(round(y0)))
    px1 = min(img.size[0], int(round(x1)))
    py1 = min(img.size[1], int(round(y1)))
    if non_square and coverage < 0.90:
        expand_x = min(
            int(round((width / 8.0) * 0.40)),
            max(0, px0),
            max(0, img.size[0] - px1),
        )
        expand_y = min(
            int(round((height / 8.0) * 0.40)),
            max(0, py0),
            max(0, img.size[1] - py1),
        )
        if expand_x > 0 or expand_y > 0:
            px0 = max(0, px0 - expand_x)
            py0 = max(0, py0 - expand_y)
            px1 = min(img.size[0], px1 + expand_x)
            py1 = min(img.size[1], py1 + expand_y)
    min_crop_ratio = 0.45 if non_square else 0.55
    if px1 - px0 < img.size[0] * min_crop_ratio or py1 - py0 < img.size[1] * min_crop_ratio:
        return None
    cropped = img.crop((px0, py0, px1, py1))
    if cropped.size == img.size:
        return None
    return cropped.resize(img.size, Image.LANCZOS)

def crop_square_from_grid_box(img, grid_box, pad_cells=0.35):
    if grid_box is None:
        return None
    x0, y0, x1, y1 = [float(v) for v in grid_box]
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    pad_x = (box_w / 8.0) * float(pad_cells)
    pad_y = (box_h / 8.0) * float(pad_cells)
    px0 = max(0, int(round(x0 - pad_x)))
    py0 = max(0, int(round(y0 - pad_y)))
    px1 = min(img.size[0], int(round(x1 + pad_x)))
    py1 = min(img.size[1], int(round(y1 + pad_y)))
    if px1 <= px0 or py1 <= py0:
        return None
    cropped = img.crop((px0, py0, px1, py1))
    if cropped.size == img.size:
        return None
    side = int(round(max(cropped.size[0], cropped.size[1])))
    side = max(64, side)
    return cropped.resize((side, side), Image.LANCZOS)


def _best_roi_by_grid_evidence(candidates):
    best = None
    best_score = None
    seen = set()
    for roi in candidates:
        if roi is None:
            continue
        key = (roi.size[0], roi.size[1], roi.tobytes()[:256])
        if key in seen:
            continue
        seen.add(key)
        row = _score_roi_candidate("roi_refine", roi)
        score = (
            float(row["score"]),
            float(row["support_ratio"]),
            float(row["evidence"]),
            float(row["checker_score"]),
        )
        if best is None or score > best_score:
            best = roi
            best_score = score
    return best


def _collect_best_full_grid_box_square_rois(img, grid_box, max_rois=5):
    if grid_box is None:
        return []
    scored = []
    seen = set()
    for proposal in iter_grid_box_square_proposals(img, grid_box) or ():
        roi = perspective_transform(img, proposal["corners"])
        refined = crop_warp_to_detected_grid(roi)
        roi = _best_roi_by_grid_evidence([roi, refined])
        if roi is None:
            continue
        key = (roi.size[0], roi.size[1], roi.tobytes()[:256])
        if key in seen:
            continue
        seen.add(key)
        row = _score_roi_candidate("grid_box_scan", roi)
        score = (
            float(row["score"]),
            float(row["support_ratio"]),
            float(row["evidence"]),
            float(row["checker_score"]),
        )
        scored.append((score, roi))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [roi for _, roi in scored[:max(1, int(max_rois))]]

def iter_grid_box_square_proposals(img, grid_box, pad_cells=0.35):
    if grid_box is None:
        return
    x0, y0, x1, y1 = [float(v) for v in grid_box]
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    pad_x = (box_w / 8.0) * float(pad_cells)
    pad_y = (box_h / 8.0) * float(pad_cells)
    padded_x0 = x0 - pad_x
    padded_y0 = y0 - pad_y
    padded_x1 = x1 + pad_x
    padded_y1 = y1 + pad_y
    side = int(round(max(padded_x1 - padded_x0, padded_y1 - padded_y0)))
    side = max(64, min(side, img.size[0], img.size[1]))
    max_x = max(0, img.size[0] - side)
    max_y = max(0, img.size[1] - side)
    center_x = max(0, min(int(round(((x0 + x1) * 0.5) - (side * 0.5))), max_x))
    center_y = max(0, min(int(round(((y0 + y1) * 0.5) - (side * 0.5))), max_y))
    x_positions = {center_x}
    y_positions = {center_y}
    if img.size[0] > img.size[1]:
        left_anchor = max(0, min(int(round(padded_x0)), max_x))
        right_anchor = max(0, min(int(round(padded_x1 - side)), max_x))
        x_positions.update(
            {
                0,
                max_x,
                left_anchor,
                right_anchor,
                max(0, min(int(round((left_anchor + center_x) * 0.5)), max_x)),
                max(0, min(int(round((center_x + right_anchor) * 0.5)), max_x)),
            }
        )
    elif img.size[1] > img.size[0]:
        top_anchor = max(0, min(int(round(padded_y0)), max_y))
        bottom_anchor = max(0, min(int(round(padded_y1 - side)), max_y))
        y_positions.update(
            {
                0,
                max_y,
                top_anchor,
                bottom_anchor,
                max(0, min(int(round((top_anchor + center_y) * 0.5)), max_y)),
                max(0, min(int(round((center_y + bottom_anchor) * 0.5)), max_y)),
            }
        )
    seen = set()
    for px in sorted(x_positions):
        for py in sorted(y_positions):
            key = (int(px), int(py), int(side))
            if key in seen:
                continue
            seen.add(key)
            corners = np.array(
                [[px, py], [px + side, py], [px + side, py + side], [px, py + side]],
                dtype=np.float32,
            )
            yield {
                "tag": f"grid_box_square_{int(px)}_{int(py)}",
                "corners": corners,
                "metrics": compute_quad_metrics(corners, img.size[0], img.size[1]),
            }

def _grid_margin_score(meta, img_size, min_span_ratio=0.0):
    box = meta.get("grid_box")
    if box is None:
        return 0.0
    width, height = img_size
    x0, y0, x1, y1 = box
    margins = np.array(
        [x0, y0, max(0.0, width - x1), max(0.0, height - y1)],
        dtype=np.float32,
    )
    avg_margin = float(np.mean(margins) / max(1.0, min(width, height)))
    inset_score = float(max(0.0, 1.0 - (avg_margin / 0.18)))
    tight_score = 0.0
    if min_span_ratio >= 0.85:
        tight_score = float(max(0.0, 1.0 - (avg_margin / 0.025)))
    return max(inset_score, tight_score)

def _checker_texture_score(img):
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    cell = 64
    means = np.zeros((8, 8), dtype=np.float32)
    stds = np.zeros((8, 8), dtype=np.float32)
    for row in range(8):
        for col in range(8):
            patch = gray[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell]
            inner = patch[12:-12, 12:-12]
            means[row, col] = float(np.mean(inner))
            stds[row, col] = float(np.std(inner))
    parity = (np.indices((8, 8)).sum(axis=0) % 2).astype(bool)
    even = means[~parity]
    odd = means[parity]
    sep = float(abs(float(np.mean(even)) - float(np.mean(odd))))
    within = float(0.5 * (float(np.std(even)) + float(np.std(odd))))
    contrast = sep / max(1e-4, within)
    row_alt = float(np.mean(np.abs(np.diff(means, axis=1))))
    col_alt = float(np.mean(np.abs(np.diff(means, axis=0))))
    alt = 0.5 * (row_alt + col_alt)
    texture = float(np.mean(stds))
    contrast_score = min(1.0, contrast / 1.2)
    sep_score = min(1.0, sep / 0.12)
    alt_score = min(1.0, alt / 0.18)
    texture_target = max(0.0, 1.0 - (abs(texture - 0.12) / 0.12))
    return float(
        (0.35 * contrast_score)
        + (0.25 * sep_score)
        + (0.25 * alt_score)
        + (0.15 * texture_target)
    )

def _proposal_dedupe_key(corners, width, height):
    corners = np.asarray(corners, dtype=np.float32)
    center = corners.mean(axis=0)
    metrics = compute_quad_metrics(corners, width, height)
    return (
        round(float(center[0]) / max(1.0, width), 2),
        round(float(center[1]) / max(1.0, height), 2),
        round(metrics["area_ratio"], 2),
    )

def _collect_contour_proposals(img):
    _, gray = _gray_rgb(img)
    height, width = gray.shape
    proposals = []
    seen = set()
    variants = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)
    variants.append(cv2.Canny(norm, 30, 100))
    variants.append(cv2.Canny(norm, 50, 150))
    variants.append(cv2.Canny(norm, 70, 200))
    adaptive = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    variants.append(adaptive)
    for variant in variants:
        contours, _ = cv2.findContours(variant, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:120]:
            area = cv2.contourArea(contour)
            if area < (width * height * 0.12):
                continue
            peri = cv2.arcLength(contour, True)
            candidate_quads = []
            for eps in (0.01, 0.02, 0.03, 0.05):
                approx = cv2.approxPolyDP(contour, eps * peri, True)
                if len(approx) == 4:
                    candidate_quads.append(("contour_quad", order_corners(approx.reshape(4, 2))))
                    break
            rect = cv2.minAreaRect(contour)
            rect_box = cv2.boxPoints(rect)
            candidate_quads.append(("min_area_rect", order_corners(rect_box)))
            for tag, corners in candidate_quads:
                metrics = compute_quad_metrics(corners, width, height)
                if not is_warp_geometry_relaxed(metrics):
                    continue
                dedupe = (tag,) + _proposal_dedupe_key(corners, width, height)
                if dedupe in seen:
                    continue
                seen.add(dedupe)
                proposals.append(
                    {
                        "tag": tag,
                        "corners": corners.astype(np.float32),
                        "metrics": metrics,
                    }
                )
    return proposals

def _collect_bright_board_proposals(img):
    rgb = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    height, width = rgb.shape[:2]
    mask = (((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 150)).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proposals = []
    seen = set()
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / float(max(1, width * height))
        aspect_similarity = min(w / float(max(h, 1)), h / float(max(w, 1)))
        if area_ratio < 0.10 or aspect_similarity < 0.45:
            continue
        side_candidates = sorted(
            {
                int(round(h * 1.02)),
                int(round(h * 1.06)),
                int(round(h * 1.10)),
            }
        )
        x_offsets = [0.08, 0.12, 0.16]
        y_offsets = [-0.02, 0.0, 0.02]
        for side in side_candidates:
            if side <= 0:
                continue
            side = min(side, width, height)
            for x_frac in x_offsets:
                for y_frac in y_offsets:
                    px = int(round(x + (w * x_frac)))
                    py = int(round(y + (h * y_frac)))
                    px = max(0, min(px, width - side))
                    py = max(0, min(py, height - side))
                    corners = np.array(
                        [[px, py], [px + side, py], [px + side, py + side], [px, py + side]],
                        dtype=np.float32,
                    )
                    dedupe = ("bright_board_square",) + _proposal_dedupe_key(corners, width, height)
                    if dedupe in seen:
                        continue
                    seen.add(dedupe)
                    proposals.append(
                        {
                            "tag": "bright_board_square",
                            "corners": corners,
                            "metrics": compute_quad_metrics(corners, width, height),
                        }
                    )
    return proposals

def _estimate_texture_band_box(img):
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3.0)
    resid = cv2.absdiff(gray, blur).astype(np.float32) / 255.0
    row_energy = cv2.GaussianBlur(resid.mean(axis=1).reshape(-1, 1), (1, 15), 0).reshape(-1)
    col_energy = cv2.GaussianBlur(resid.mean(axis=0).reshape(1, -1), (15, 1), 0).reshape(-1)

    def _largest_band(values):
        if values.size == 0:
            return None
        lo = float(np.quantile(values, 0.60))
        hi = float(np.quantile(values, 0.90))
        threshold = lo + (0.35 * max(0.0, hi - lo))
        mask = values >= threshold
        best = None
        start = None
        for idx, flag in enumerate(mask.tolist() + [False]):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                end = idx
                strength = float(values[start:end].mean()) * float(end - start)
                if best is None or strength > best[2]:
                    best = (start, end, strength)
                start = None
        return best

    row_band = _largest_band(row_energy)
    col_band = _largest_band(col_energy)
    if row_band is None or col_band is None:
        return None
    y0, y1, _ = row_band
    x0, x1, _ = col_band
    if (y1 - y0) < (img.size[1] * 0.30) or (x1 - x0) < (img.size[0] * 0.30):
        return None
    return (float(x0), float(y0), float(x1), float(y1))

def _collect_square_window_proposals(img):
    width, height = img.size
    aspect = width / float(max(1, height))
    if 0.92 <= aspect <= 1.08:
        return []

    proposals = []
    seen = set()
    short_side = min(width, height)
    side_scales = [1.0, 0.92, 0.84]
    for scale in side_scales:
        side = int(round(short_side * scale))
        if side < 220:
            continue
        if width > height:
            max_offset_x = max(0, width - side)
            max_offset_y = max(0, height - side)
            count_x = 5
            x_offsets = np.linspace(0, max_offset_x, count_x)
            if max_offset_y <= 4:
                y_offsets = [0.0]
            else:
                cell = side / 8.0
                y_offsets = {
                    0.0,
                    float(max(0, min(int(round(cell * 0.5)), max_offset_y))),
                    float(max(0, min(int(round(cell * 1.5)), max_offset_y))),
                    float(max_offset_y),
                }
            for offset_x in sorted(x_offsets):
                for offset_y in sorted(y_offsets):
                    x0 = int(round(offset_x))
                    y0 = int(round(offset_y))
                    corners = np.array(
                        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
                        dtype=np.float32,
                    )
                    dedupe = ("square_window",) + _proposal_dedupe_key(corners, width, height)
                    if dedupe in seen:
                        continue
                    seen.add(dedupe)
                    proposals.append(
                        {
                            "tag": "square_window",
                            "corners": corners,
                            "metrics": compute_quad_metrics(corners, width, height),
                        }
                    )
        else:
            max_offset_x = max(0, width - side)
            max_offset_y = max(0, height - side)
            texture_box = _estimate_texture_band_box(img)
            if texture_box is not None:
                tx0, ty0, tx1, ty1 = texture_box
                y_offsets = {
                    float(max(0, min(int(round(ty0)), max_offset_y))),
                    float(max(0, min(int(round(((ty0 + ty1) - side) / 2.0)), max_offset_y))),
                    float(max(0, min(int(round(ty1 - side)), max_offset_y))),
                }
            else:
                count_y = 4 if aspect > 0.64 else 5
                y_offsets = np.linspace(0, max_offset_y, count_y)
            if max_offset_x <= 4:
                x_offsets = [0.0]
            else:
                cell = side / 8.0
                if texture_box is not None:
                    tx0, _, tx1, _ = texture_box
                    x_offsets = {
                        float(max(0, min(int(round(tx0)), max_offset_x))),
                        float(max(0, min(int(round(((tx0 + tx1) - side) / 2.0)), max_offset_x))),
                        float(max(0, min(int(round(tx1 - side)), max_offset_x))),
                    }
                else:
                    x_offsets = {
                        0.0,
                        float(max(0, min(int(round(cell * 0.5)), max_offset_x))),
                        float(max(0, min(int(round(cell * 1.5)), max_offset_x))),
                        float(max_offset_x),
                    }
            for offset_y in sorted(y_offsets):
                for offset_x in sorted(x_offsets):
                    x0 = int(round(offset_x))
                    y0 = int(round(offset_y))
                    corners = np.array(
                        [[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
                        dtype=np.float32,
                    )
                    dedupe = ("square_window",) + _proposal_dedupe_key(corners, width, height)
                    if dedupe in seen:
                        continue
                    seen.add(dedupe)
                    proposals.append(
                        {
                            "tag": "square_window",
                            "corners": corners,
                            "metrics": compute_quad_metrics(corners, width, height),
                        }
                    )
    return proposals


def _collect_grid_box_square_proposals(img, grid_box):
    if grid_box is None:
        return []
    width, height = img.size
    aspect = width / float(max(1, height))
    non_square = not (0.92 <= aspect <= 1.08)
    if not non_square:
        return []
    x0, y0, x1, y1 = [float(v) for v in grid_box]
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    cell_est = max(8.0, min(box_w, box_h) / 8.0)
    if min(box_w, box_h) < min(width, height) * 0.45:
        return []
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    if non_square:
        side_bases = {
            min(float(min(width, height)), math.sqrt(box_w * box_h)),
        }
        scales = (0.84, 0.92, 1.0, 1.08)
    else:
        side_bases = {
            min(float(min(width, height)), max(box_w, box_h)),
        }
        scales = (0.96, 1.0, 1.04, 1.08)
    proposals = []
    seen = set()
    for side_base in sorted(side_bases):
        for scale in scales:
            side = int(round(side_base * scale))
            if side < 220:
                continue
            side = min(side, width, height)
            max_x = max(0, width - side)
            max_y = max(0, height - side)
            base_x = max(0, min(int(round(center_x - (side / 2.0))), max_x))
            base_y = max(0, min(int(round(center_y - (side / 2.0))), max_y))
            x_offsets = {base_x}
            y_offsets = {base_y}
            if max_x > 0:
                x_offsets.update(
                    {
                        max(0, min(int(round(x0)), max_x)),
                        max(0, min(int(round(x1 - side)), max_x)),
                    }
                )
            if max_y > 0:
                y_offsets.update(
                    {
                        0,
                        max_y,
                        max(0, min(int(round(y0)), max_y)),
                        max(0, min(int(round(y1 - side)), max_y)),
                        }
                    )
            if non_square:
                long_count = min(11, max(5, int(round((max(width, height) / float(max(1, side))) * 5))))
                if width > height:
                    x_offsets = set(float(v) for v in np.linspace(0, max_x, long_count))
                    for step in (0.5, 1.0, 1.5, 2.0):
                        delta = int(round(cell_est * step))
                        x_offsets.update(
                            {
                                float(max(0, min(delta, max_x))),
                                float(max(0, min(max_x - delta, max_x))),
                                float(max(0, min(int(round(x0 + delta)), max_x))),
                                float(max(0, min(int(round(x1 - side - delta)), max_x))),
                            }
                        )
                    y_offsets = set(float(v) for v in np.linspace(0, max_y, 5))
                    for step in (0.5, 1.0, 1.5, 2.0):
                        delta = int(round(cell_est * step))
                        y_offsets.update(
                            {
                                float(max(0, min(delta, max_y))),
                                float(max(0, min(max_y - delta, max_y))),
                            }
                        )
                else:
                    y_offsets = set(float(v) for v in np.linspace(0, max_y, long_count))
                    for step in (0.5, 1.0, 1.5, 2.0):
                        delta = int(round(cell_est * step))
                        y_offsets.update(
                            {
                                float(max(0, min(delta, max_y))),
                                float(max(0, min(max_y - delta, max_y))),
                                float(max(0, min(int(round(y0 + delta)), max_y))),
                                float(max(0, min(int(round(y1 - side - delta)), max_y))),
                            }
                        )
                    x_offsets = set(float(v) for v in np.linspace(0, max_x, 5))
                    for step in (0.5, 1.0, 1.5, 2.0):
                        delta = int(round(cell_est * step))
                        x_offsets.update(
                            {
                                float(max(0, min(delta, max_x))),
                                float(max(0, min(max_x - delta, max_x))),
                            }
                        )
                if width > height:
                    pair_offsets = set()
                    for px in sorted(x_offsets):
                        for py in sorted(y_offsets):
                            pair_offsets.add((int(round(px)), int(round(py))))
                else:
                    pair_offsets = {
                        (0, base_y),
                        (0, max(0, min(int(round(y0)), max_y))),
                        (0, max(0, min(int(round(y1 - side)), max_y))),
                        (base_x, base_y),
                        (base_x, max(0, min(int(round(y0)), max_y))),
                        (base_x, max(0, min(int(round(y1 - side)), max_y))),
                    }
            else:
                pair_offsets = {(base_x, base_y)}
                pair_offsets.update((px, base_y) for px in x_offsets)
                pair_offsets.update((base_x, py) for py in y_offsets)
                if max_x > 0 and max_y > 0:
                    anchor_x = sorted(
                        {
                            max(0, min(int(round(x0)), max_x)),
                            max(0, min(int(round(x1 - side)), max_x)),
                        }
                    )
                    anchor_y = sorted(
                        {
                            max(0, min(int(round(y0)), max_y)),
                            max(0, min(int(round(y1 - side)), max_y)),
                        }
                    )
                    for px in anchor_x:
                        for py in anchor_y:
                            pair_offsets.add((px, py))
            for px, py in sorted(pair_offsets):
                if px < 0 or py < 0:
                    continue
                corners = np.array(
                    [[px, py], [px + side, py], [px + side, py + side], [px, py + side]],
                    dtype=np.float32,
                )
                dedupe = ("grid_box_square",) + _proposal_dedupe_key(corners, width, height)
                if dedupe in seen:
                    continue
                seen.add(dedupe)
                proposals.append(
                    {
                        "tag": "grid_box_square",
                        "corners": corners,
                        "metrics": compute_quad_metrics(corners, width, height),
                    }
                )
    return proposals

def _score_region_proposal(img, proposal):
    corners = proposal["corners"]
    metrics = proposal["metrics"]
    warped = perspective_transform(img, corners)
    warped_eval = evaluate_warped_grid(warped)
    geometry = warp_geometry_quality(metrics)
    evidence = float(warped_eval["evidence"])
    support_ratio = float(warped_eval["support_ratio"])
    grid_ratio_score = float(warped_eval.get("ratio", 0.0))
    x_line = float(warped_eval["x_grid"]["line_score"]) if warped_eval.get("x_grid") else 0.0
    y_line = float(warped_eval["y_grid"]["line_score"]) if warped_eval.get("y_grid") else 0.0
    axis_balance = float(min(x_line, y_line) / max(x_line, y_line, 1e-6))
    margin_score = _grid_margin_score(
        warped_eval,
        (warped.size[0], warped.size[1]),
        min_span_ratio=float(min(metrics["area_ratio"] * 2.0, 1.0)),
    )
    checker_score = _checker_texture_score(warped)
    score = float(
        (0.18 * geometry)
        + (0.18 * evidence)
        + (0.16 * support_ratio)
        + (0.08 * margin_score)
        + (0.24 * checker_score)
        + (0.08 * grid_ratio_score)
        + (0.08 * axis_balance)
    )
    trusted = bool(is_warp_geometry_trustworthy(metrics) and evidence >= 0.34 and support_ratio >= 0.46)
    if proposal["tag"] == "aligned_box" and support_ratio >= 0.85 and checker_score >= 0.80:
        score += 0.02
    return {
        "tag": proposal["tag"],
        "corners": corners,
        "metrics": metrics,
        "warped": warped,
        "geometry": float(geometry),
        "evidence": evidence,
        "support_ratio": support_ratio,
        "grid_ratio_score": float(grid_ratio_score),
        "axis_balance": float(axis_balance),
        "margin_score": float(margin_score),
        "checker_score": float(checker_score),
        "score": float(score),
        "trusted": trusted,
    }

def detect_board_region(img, max_hypotheses=3):
    hypotheses = []
    full_meta = score_full_frame_board(img)
    aspect = img.size[0] / float(max(1, img.size[1]))
    non_square = not (0.92 <= aspect <= 1.08)
    for proposal in _collect_contour_proposals(img):
        hypotheses.append(_score_region_proposal(img, proposal))
    for proposal in _collect_bright_board_proposals(img):
        hypotheses.append(_score_region_proposal(img, proposal))
    if full_meta["grid_box"] is not None:
        x0, y0, x1, y1 = full_meta["grid_box"]
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        metrics = compute_quad_metrics(corners, img.size[0], img.size[1])
        hypotheses.append(
            _score_region_proposal(
                img,
                {
                    "tag": "aligned_box",
                    "corners": corners,
                    "metrics": metrics,
                },
            )
        )
    deduped = []
    seen = set()
    for row in sorted(
        hypotheses,
        key=lambda item: (item["score"], item["support_ratio"], item["evidence"], item["checker_score"]),
        reverse=True,
    ):
        key = _proposal_dedupe_key(row["corners"], img.size[0], img.size[1])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= max_hypotheses:
            break
    return deduped

def _refine_board_region_image(warped):
    candidates = [warped]
    refined = crop_warp_to_detected_grid(warped)
    if refined is not None:
        candidates.append(refined)
    full_meta = score_full_frame_board(warped)
    square = crop_square_from_grid_box(warped, full_meta.get("grid_box"), pad_cells=0.20)
    if square is not None:
        candidates.append(square)
    if refined is not None:
        refined_meta = score_full_frame_board(refined)
        refined_square = crop_square_from_grid_box(refined, refined_meta.get("grid_box"), pad_cells=0.18)
        if refined_square is not None:
            candidates.append(refined_square)
    best = _best_roi_by_grid_evidence(candidates)
    return best if best is not None else warped

def _score_proposal(img, proposal):
    corners = proposal["corners"]
    metrics = proposal["metrics"]
    x0 = float(corners[:, 0].min())
    y0 = float(corners[:, 1].min())
    x1 = float(corners[:, 0].max())
    y1 = float(corners[:, 1].max())
    span_x = float(corners[:, 0].max() - corners[:, 0].min()) / float(max(1, img.size[0]))
    span_y = float(corners[:, 1].max() - corners[:, 1].min()) / float(max(1, img.size[1]))
    min_span_ratio = float(min(span_x, span_y))
    warped = perspective_transform(img, corners)
    refined = crop_warp_to_detected_grid(warped)
    scoring_img = refined if refined is not None else warped
    warped_eval = evaluate_warped_grid(scoring_img)
    geometry = warp_geometry_quality(metrics)
    evidence = float(warped_eval["evidence"])
    support_ratio = float(warped_eval["support_ratio"])
    grid_ratio_score = float(warped_eval.get("ratio", 0.0))
    x_line = float(warped_eval["x_grid"]["line_score"]) if warped_eval.get("x_grid") else 0.0
    y_line = float(warped_eval["y_grid"]["line_score"]) if warped_eval.get("y_grid") else 0.0
    axis_balance = float(min(x_line, y_line) / max(x_line, y_line, 1e-6))
    margin_score = _grid_margin_score(
        warped_eval,
        (scoring_img.size[0], scoring_img.size[1]),
        min_span_ratio=min_span_ratio,
    )
    checker_score = _checker_texture_score(scoring_img)
    score = float(
        (0.14 * geometry)
        + (0.12 * evidence)
        + (0.10 * support_ratio)
        + (0.04 * margin_score)
        + (0.34 * checker_score)
        + (0.12 * grid_ratio_score)
        + (0.14 * axis_balance)
    )
    if refined is not None and checker_score >= 0.65:
        score += 0.03
    aspect = img.size[0] / float(max(1, img.size[1]))
    trusted = bool(is_warp_geometry_trustworthy(metrics) and evidence >= 0.34 and support_ratio >= 0.46)
    if (
        proposal["tag"] == "grid_box_square"
        and not (0.92 <= aspect <= 1.08)
        and trusted
        and support_ratio >= 0.94
        and margin_score >= 0.96
    ):
        score += 0.004
    if (
        proposal["tag"] in {"grid_box_square", "square_window"}
        and support_ratio >= 0.97
        and checker_score >= 0.93
        and grid_ratio_score >= 0.80
    ):
        if img.size[0] > img.size[1] and y0 <= 1.0 and (img.size[1] - y1) >= max(8.0, (img.size[1] / 8.0) * 0.40):
            score += 0.01
        if img.size[1] > img.size[0] and x0 <= 1.0 and (img.size[0] - x1) >= max(8.0, (img.size[0] / 8.0) * 0.40):
            score += 0.01
    return {
        "tag": proposal["tag"],
        "corners": corners,
        "roi": scoring_img,
        "geometry": float(geometry),
        "evidence": evidence,
        "score": score,
        "support_ratio": support_ratio,
        "grid_ratio_score": float(grid_ratio_score),
        "axis_balance": float(axis_balance),
        "margin_score": float(margin_score),
        "min_span_ratio": float(min_span_ratio),
        "checker_score": float(checker_score),
        "trusted": trusted,
    }

def _score_roi_candidate(tag, roi):
    roi_meta = score_full_frame_board(roi)
    checker_score = _checker_texture_score(roi)
    margin_score = _grid_margin_score(roi_meta, roi.size, min_span_ratio=1.0)
    axis_balance = float(roi_meta["ratio"])
    score = float(
        (0.16 * roi_meta["score"])
        + (0.16 * float(roi_meta["evidence"]))
        + (0.14 * float(roi_meta["support_ratio"]))
        + (0.08 * float(margin_score))
        + (0.34 * float(checker_score))
        + (0.12 * float(axis_balance))
    )
    return {
        "tag": str(tag),
        "corners": None,
        "roi": roi,
        "geometry": 1.0,
        "evidence": float(roi_meta["evidence"]),
        "score": score,
        "support_ratio": float(roi_meta["support_ratio"]),
        "grid_ratio_score": float(roi_meta["ratio"]),
        "axis_balance": float(axis_balance),
        "margin_score": float(margin_score),
        "min_span_ratio": 1.0,
        "checker_score": float(checker_score),
        "trusted": bool(roi_meta["trusted"]),
        "dedupe_key": (str(tag), roi.size[0], roi.size[1]),
    }

def detect_board_grid(img, max_hypotheses=6):
    hypotheses = []
    full_meta = score_full_frame_board(img)
    aspect = img.size[0] / float(max(1, img.size[1]))
    non_square = not (0.92 <= aspect <= 1.08)
    contour_proposals = _collect_contour_proposals(img)
    for proposal in contour_proposals:
        hypotheses.append(_score_proposal(img, proposal))
    for proposal in _collect_bright_board_proposals(img):
        hypotheses.append(_score_proposal(img, proposal))
    if non_square:
        roi = crop_warp_to_detected_grid(img)
        if roi is None and full_meta.get("grid_box") is not None:
            roi = crop_square_from_grid_box(img, full_meta.get("grid_box"))
        if roi is not None:
            hypotheses.append(_score_roi_candidate("grid_box_crop", roi))
    else:
        for proposal in _collect_square_window_proposals(img):
            hypotheses.append(_score_proposal(img, proposal))
        for proposal in _collect_grid_box_square_proposals(img, full_meta.get("grid_box")):
            hypotheses.append(_score_proposal(img, proposal))
    if full_meta["grid_box"] is not None:
        x0, y0, x1, y1 = full_meta["grid_box"]
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        ratio = width / height
        coverage = (width * height) / float(img.size[0] * img.size[1])
        if 0.88 <= ratio <= 1.12 and 0.35 <= coverage <= 0.98:
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            metrics = compute_quad_metrics(corners, img.size[0], img.size[1])
            hypotheses.append(
                _score_proposal(
                    img,
                    {
                        "tag": "aligned_box",
                        "corners": corners,
                        "metrics": metrics,
                    },
                )
            )
    deduped = []
    seen = set()
    for row in sorted(hypotheses, key=lambda item: (item["score"], item["evidence"], item["geometry"]), reverse=True):
        key = row.get("dedupe_key")
        if key is None:
            key = _proposal_dedupe_key(row["corners"], img.size[0], img.size[1])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= max(max_hypotheses * 3, 18):
            break
    deduped = deduped[:max_hypotheses]
    return deduped

# =============================================================================
# CANDIDATE BUILDING & CLI
# =============================================================================
def trim_dark_edge_bars(img):
    return img

def find_inner_board_window(img):
    return None

def _debug_event(kind, **payload):
    if not DEBUG_MODE:
        return
    row = {"kind": kind}
    row.update(payload)
    print(f"DEBUG_JSON {json.dumps(row, sort_keys=True)}", file=sys.stderr)

def _candidate_tuple(tag, img, warp_quality, warp_trusted, detector_score, detector_support):
    return (tag, img, float(warp_quality), bool(warp_trusted), float(detector_score), float(detector_support))

def _candidate_sort_key(candidate):
    return (
        float(candidate[4]),
        float(candidate[5]),
        float(candidate[2]),
    )

def _material_score_order(candidates, *, epsilon=None):
    if epsilon is None:
        epsilon = MATERIAL_SCORE_EPSILON
    ordered = []
    for candidate in candidates:
        inserted = False
        score = float(candidate[4])
        for idx, current in enumerate(ordered):
            current_score = float(current[4])
            if score > (current_score + epsilon):
                ordered.insert(idx, candidate)
                inserted = True
                break
        if not inserted:
            ordered.append(candidate)
    return ordered

def _full_candidate_should_lead(img, full_meta, detector_candidates):
    ratio = img.size[0] / float(max(1, img.size[1]))
    best_detector_score = max((float(candidate[4]) for candidate in detector_candidates), default=-1.0)
    full_checker = _checker_texture_score(img)
    strong_full = bool(
        FULL_CANDIDATE_RATIO_MIN <= ratio <= FULL_CANDIDATE_RATIO_MAX
        and float(full_meta.get("coverage", 0.0)) >= FULL_CANDIDATE_STRONG_COVERAGE_MIN
        and full_meta["evidence"] >= FULL_CANDIDATE_STRONG_EVIDENCE_MIN
        and full_meta["support_ratio"] >= FULL_CANDIDATE_STRONG_SUPPORT_MIN
    )
    locked_square_full = bool(
        FULL_CANDIDATE_RATIO_MIN <= ratio <= FULL_CANDIDATE_RATIO_MAX
        and float(full_meta.get("coverage", 0.0)) >= 0.60
        and full_meta["evidence"] >= 0.55
        and full_meta["support_ratio"] >= 0.88
    )
    high_checker_full = bool(
        FULL_CANDIDATE_RATIO_MIN <= ratio <= FULL_CANDIDATE_RATIO_MAX
        and float(full_meta.get("coverage", 0.0)) >= 0.95
        and full_meta["evidence"] >= 0.45
        and full_meta["support_ratio"] >= 0.80
        and float(full_checker) >= 0.90
    )
    return bool(
        strong_full
        or locked_square_full
        or high_checker_full
        or (
            FULL_CANDIDATE_RATIO_MIN <= ratio <= FULL_CANDIDATE_RATIO_MAX
            and float(full_meta.get("coverage", 0.0)) >= FULL_CANDIDATE_COVERAGE_MIN
            and full_meta["evidence"] >= FULL_CANDIDATE_EVIDENCE_MIN
            and full_meta["support_ratio"] >= FULL_CANDIDATE_SUPPORT_MIN
            and float(full_meta["score"]) >= best_detector_score
        )
    )

def build_detector_candidates(img):
    full_meta = score_full_frame_board(img)
    full_candidate = _candidate_tuple(
        "full",
        img,
        full_meta["score"],
        full_meta["trusted"],
        full_meta["score"],
        full_meta["support_ratio"],
    )
    if not USE_EDGE_DETECTION:
        return [full_candidate]
    aspect = img.size[0] / float(max(1, img.size[1]))
    non_square = not (0.92 <= aspect <= 1.08)
    detector_candidates = []
    region_rows = detect_board_region(img, max_hypotheses=3)
    for row in region_rows:
        roi = _refine_board_region_image(row["warped"])
        detector_candidates.append(
            _candidate_tuple(
                str(row["tag"]),
                roi,
                row["score"],
                row["trusted"],
                row["score"],
                row["support_ratio"],
            )
        )

    best_region_score = max((float(candidate[4]) for candidate in detector_candidates), default=-1.0)
    need_grid_fallback = (not detector_candidates) or (best_region_score < 0.84)

    if need_grid_fallback:
        for row in detect_board_grid(img, max_hypotheses=4 if non_square else 3):
            roi = row.get("roi")
            if roi is None:
                corners = row["corners"]
                roi = perspective_transform(img, corners)
                refined = crop_warp_to_detected_grid(roi)
                if refined is not None:
                    roi = refined
            detector_candidates.append(
                _candidate_tuple(
                    str(row["tag"]),
                    roi,
                    row["score"],
                    row["trusted"],
                    row["score"],
                    row["support_ratio"],
                )
            )

    if non_square and (not detector_candidates or best_region_score < 0.88):
        for idx, scan_roi in enumerate(
            _collect_best_full_grid_box_square_rois(img, full_meta.get("grid_box"), max_rois=2),
            start=1,
        ):
            row = _score_roi_candidate(f"grid_box_scan_{idx}", scan_roi)
            detector_candidates.append(
                _candidate_tuple(
                    str(row["tag"]),
                    row["roi"],
                    row["score"],
                    row["trusted"],
                    row["score"],
                    row["support_ratio"],
                )
            )

    if not detector_candidates:
        roi = crop_warp_to_detected_grid(img)
        if roi is None and full_meta.get("grid_box") is not None:
            roi = crop_square_from_grid_box(img, full_meta.get("grid_box"))
        if roi is not None:
            row = _score_roi_candidate("grid_box_crop", roi)
            detector_candidates.append(
                _candidate_tuple(
                    str(row["tag"]),
                    row["roi"],
                    row["score"],
                    row["trusted"],
                    row["score"],
                    row["support_ratio"],
                )
            )

    if detector_candidates:
        candidates = _material_score_order(detector_candidates)[:5 if non_square else 4]
        if _full_candidate_should_lead(img, full_meta, candidates):
            candidates = [full_candidate] + candidates
        else:
            candidates.append(full_candidate)
        _debug_event(
            "v6_candidate_pool",
            original_count=len(detector_candidates) + 1,
            capped_count=len(candidates),
            tags=[row[0] for row in candidates],
            scores=[round(float(row[4]), 6) for row in candidates],
        )
        return candidates
    for row in detect_board_grid(img, max_hypotheses=4 if non_square else 3):
        roi = row.get("roi")
        if roi is None:
            corners = row["corners"]
            roi = perspective_transform(img, corners)
            refined = crop_warp_to_detected_grid(roi)
            if refined is not None:
                roi = refined
        detector_candidates.append(
            _candidate_tuple(
                str(row["tag"]),
                roi,
                row["score"],
                row["trusted"],
                row["score"],
                row["support_ratio"],
            )
        )
    candidates = _material_score_order(detector_candidates)[:4 if non_square else 3]
    if not candidates:
        return [full_candidate]
    if _full_candidate_should_lead(img, full_meta, candidates):
        candidates = [full_candidate] + candidates
    else:
        candidates.append(full_candidate)
    _debug_event(
        "v6_candidate_pool",
        original_count=len(detector_candidates) + 1,
        capped_count=len(candidates),
        tags=[row[0] for row in candidates],
        scores=[round(float(row[4]), 6) for row in candidates],
    )
    return candidates


def _value_only_image(img):
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]
    return Image.fromarray(np.repeat(value[:, :, None], 3, axis=2))


def _clahe_gray_image(img):
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return Image.fromarray(np.repeat(clahe[:, :, None], 3, axis=2))

def _aggressive_clahe_image(img):
    """More aggressive CLAHE for dark boards — higher clip limit, smaller tiles."""
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(gray)
    return Image.fromarray(np.repeat(clahe[:, :, None], 3, axis=2))


def _topk_prob(tile_info, label):
    for alt_label, alt_prob in tile_info.get("topk", []):
        if alt_label == label:
            return float(alt_prob)
    return 0.0


def _fuse_value_case_labels(rgb_details, value_details):
    labels = ["1"] * 64
    fused = False
    for rgb_tile, value_tile in zip(rgb_details.get("tile_infos", []), value_details.get("tile_infos", [])):
        row = int(rgb_tile["row"])
        col = int(rgb_tile["col"])
        label = str(rgb_tile["label"])
        value_label = str(value_tile["label"])
        if (
            label != "1"
            and value_label != "1"
            and label.lower() == value_label.lower()
            and label != value_label
            and float(_topk_prob(rgb_tile, value_label)) >= VALUE_CASE_ALT_MIN
            and float(_topk_prob(value_tile, value_label)) >= VALUE_CASE_CONF_MIN
        ):
            label = value_label
            fused = True
        labels[(row * 8) + col] = label
    rows = ["".join(labels[r * 8 : (r + 1) * 8]) for r in range(8)]
    return _labels_to_fen(labels), sum(1 for label in labels if label != "1"), fused

def decode_candidate(candidate, model, device, board_perspective, orientation_context):
    tag, candidate_img, warp_quality, warp_trusted, detector_score, detector_support = candidate
    aspect = candidate_img.size[0] / float(max(1, candidate_img.size[1]))
    fen, conf, piece_count = infer_fen_on_image_clean(
        candidate_img,
        model,
        device,
        USE_SQUARE_DETECTION,
    )
    value_case_fused = False
    sat_stats = image_saturation_stats(candidate_img)
    enable_value_case = bool(
        str(tag) == "full"
        or (
            float(sat_stats.get("sat_mean", 255.0)) <= VALUE_CASE_SAT_MEAN_MAX
            and float(sat_stats.get("sat_p95", 255.0)) <= VALUE_CASE_SAT_P95_MAX
        )
    )
    if enable_value_case:
        rgb_fen, rgb_conf, rgb_piece_count, rgb_details = infer_fen_on_image_clean(
            candidate_img,
            model,
            device,
            USE_SQUARE_DETECTION,
            return_details=True,
        )
        value_fen, value_conf, value_piece_count, value_details = infer_fen_on_image_clean(
            _value_only_image(candidate_img),
            model,
            device,
            USE_SQUARE_DETECTION,
            return_details=True,
        )
        fused_fen, fused_piece_count, value_case_fused = _fuse_value_case_labels(rgb_details, value_details)
        fen = fused_fen if value_case_fused else rgb_fen
        conf = float(rgb_conf)
        piece_count = int(fused_piece_count if value_case_fused else rgb_piece_count)
    if float(conf) <= 0.84:
        clahe_fen, clahe_conf, clahe_piece_count = infer_fen_on_image_clean(
            _clahe_gray_image(candidate_img),
            model,
            device,
            USE_SQUARE_DETECTION,
        )
        if int(clahe_piece_count) > int(piece_count) and float(clahe_conf) >= float(conf) - 0.01:
            fen = clahe_fen
            conf = float(clahe_conf)
            piece_count = int(clahe_piece_count)
    # Dark board rescue: if value is low and piece count is suspiciously low,
    # try CLAHE with more aggressive contrast enhancement
    if piece_count < 8 and float(sat_stats.get("val_mean", 255.0)) < 120:
        clahe_fen2, clahe_conf2, clahe_piece_count2 = infer_fen_on_image_clean(
            _aggressive_clahe_image(candidate_img),
            model,
            device,
            USE_SQUARE_DETECTION,
        )
        if int(clahe_piece_count2) > int(piece_count) and float(clahe_conf2) >= float(conf) - 0.05:
            fen = clahe_fen2
            conf = float(clahe_conf2)
            piece_count = int(clahe_piece_count2)
    detected_perspective, perspective_source = resolve_candidate_orientation(
        fen,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    final_fen = rotate_fen_180(fen) if detected_perspective == "black" else fen
    _debug_event(
        "v6_candidate_decode",
        tag=tag,
        conf=float(conf),
        plausibility=float(board_plausibility_score(final_fen)),
        king_health=int(king_health(final_fen)),
        piece_count=int(piece_count),
        warp_quality=float(warp_quality),
        detector_score=float(detector_score),
        detector_support=float(detector_support),
        perspective=detected_perspective,
        perspective_source=perspective_source,
        value_case_fused=bool(value_case_fused),
    )
    return (
        tag,
        candidate_img,
        final_fen,
        conf,
        piece_count,
        detected_perspective,
        perspective_source,
        warp_quality,
        warp_trusted,
        float(detector_score),
        float(detector_support),
    )

def diagnostic_decode_candidate_with_details(
    candidate,
    *,
    model,
    device,
    board_perspective,
    orientation_context,
    topk=5,
):
    tag, candidate_img, warp_quality, warp_trusted, detector_score, detector_support = candidate
    aspect = candidate_img.size[0] / float(max(1, candidate_img.size[1]))
    sat_stats = image_saturation_stats(candidate_img)
    fen, conf, piece_count, details = infer_fen_on_image_clean(
        candidate_img,
        model,
        device,
        USE_SQUARE_DETECTION,
        return_details=True,
        topk_k=max(3, int(topk)),
    )
    value_case_fused = False
    enable_value_case = bool(
        str(tag) == "full"
        or (
            float(sat_stats.get("sat_mean", 255.0)) <= VALUE_CASE_SAT_MEAN_MAX
            and float(sat_stats.get("sat_p95", 255.0)) <= VALUE_CASE_SAT_P95_MAX
        )
    )
    if enable_value_case:
        value_fen, value_conf, value_piece_count, value_details = infer_fen_on_image_clean(
            _value_only_image(candidate_img),
            model,
            device,
            USE_SQUARE_DETECTION,
            return_details=True,
            topk_k=max(3, int(topk)),
        )
        fused_fen, fused_piece_count, value_case_fused = _fuse_value_case_labels(details, value_details)
        if value_case_fused:
            fen = fused_fen
            piece_count = int(fused_piece_count)
    if float(conf) <= 0.84:
        clahe_fen, clahe_conf, clahe_piece_count, clahe_details = infer_fen_on_image_clean(
            _clahe_gray_image(candidate_img),
            model,
            device,
            USE_SQUARE_DETECTION,
            return_details=True,
            topk_k=max(3, int(topk)),
        )
        if int(clahe_piece_count) > int(piece_count) and float(clahe_conf) >= float(conf) - 0.01:
            fen = clahe_fen
            conf = clahe_conf
            piece_count = int(clahe_piece_count)
            details = clahe_details
    # Dark board rescue
    sat_stats_diag = image_saturation_stats(candidate_img)
    if piece_count < 8 and float(sat_stats_diag.get("val_mean", 255.0)) < 120:
        aggr_fen, aggr_conf, aggr_piece_count, aggr_details = infer_fen_on_image_clean(
            _aggressive_clahe_image(candidate_img),
            model,
            device,
            USE_SQUARE_DETECTION,
            return_details=True,
            topk_k=max(3, int(topk)),
        )
        if int(aggr_piece_count) > int(piece_count) and float(aggr_conf) >= float(conf) - 0.05:
            fen = aggr_fen
            conf = aggr_conf
            piece_count = int(aggr_piece_count)
            details = aggr_details
    detected_perspective, perspective_source = resolve_candidate_orientation(
        fen,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    board_fen = rotate_fen_180(fen) if detected_perspective == "black" else fen
    tile_infos_by_square = {}
    for tile in details.get("tile_infos", []):
        row = int(tile["row"])
        col = int(tile["col"])
        if detected_perspective == "black":
            row = 7 - row
            col = 7 - col
        sq = f"{'abcdefgh'[col]}{8 - row}"
        tile_infos_by_square[sq] = tile
    return {
        "tag": tag,
        "variant": "base",
        "board_fen": board_fen,
        "fen_before_orientation": fen,
        "confidence": float(conf),
        "piece_count": int(piece_count),
        "plausibility": float(board_plausibility_score(board_fen)),
        "king_health": int(king_health(board_fen)),
        "detected_perspective": detected_perspective,
        "perspective_source": perspective_source,
        "warp_quality": float(warp_quality),
        "warp_trusted": bool(warp_trusted),
        "tile_infos_by_square": tile_infos_by_square,
        "rescored_low_sat_sparse": False,
        "sat_stats": sat_stats,
        "candidate_img": candidate_img,
        "grid_score": float(detector_score),
        "detector_support": float(detector_support),
        "value_case_fused": bool(value_case_fused),
    }

def select_best_candidate(scored):
    if not scored:
        raise RuntimeError("No scored candidates")
    def _selection_key(candidate):
        board_fen = candidate[2]
        piece_count = int(candidate[4])
        # Structural sanity: valid boards need 2 kings, plausible piece count
        # Normal chess: 2-32 pieces. >36 is almost certainly a bad crop.
        piece_ok = 0
        if 2 <= piece_count <= 36:
            piece_ok = 1
        return (
            1 if board_is_structurally_valid(board_fen) else 0,
            piece_ok,
            int(king_health(board_fen)),
            float(candidate[3]),
            int(candidate[4]),
            float(candidate[10]),
            float(candidate[9]),
            float(candidate[7]),
        )

    return max(scored, key=_selection_key)

def predict_chess_position(image_path, model_path=None, board_perspective="auto", side_to_move_override=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path=model_path, device=device)
    img = Image.open(image_path).convert("RGB")
    candidates = build_detector_candidates(img)
    orientation_context = collect_orientation_context(candidates, board_perspective=board_perspective)
    scored = [
        decode_candidate(
            candidate,
            model=model,
            device=device,
            board_perspective=board_perspective,
            orientation_context=orientation_context,
        )
        for candidate in candidates
    ]
    best = select_best_candidate(scored)
    if _should_try_inner_crop_rescue(img, best[2]):
        rescued = _try_inner_crop_rescue(img, model, device, board_perspective)
        if rescued is not None:
            best = rescued
    if side_to_move_override is not None:
        side_to_move = parse_side_to_move_override(side_to_move_override)
        side_source = "override_cli"
    else:
        side_to_move, side_source = infer_side_to_move_from_checks(best[2])
    en_passant = infer_unique_en_passant_square(best[2], side_to_move)
    _debug_event(
        "v6_selected_candidate",
        tag=best[0],
        confidence=float(best[3]),
        perspective_source=best[6],
        side_to_move_source=side_source,
    )
    return {
        "board_fen": best[2],
        "fen": f"{best[2]} {side_to_move} - {en_passant} 0 1",
        "confidence": float(best[3]),
        "side_to_move": side_to_move,
        "side_to_move_source": side_source,
        "detected_perspective": best[5],
        "perspective_source": best[6],
        "best_tag": best[0],
    }

def predict_board(image_path, model_path=None, board_perspective="auto"):
    result = predict_chess_position(image_path, model_path, board_perspective)
    return result["board_fen"], result["confidence"]

def predict_position(image_path, model_path=None, board_perspective="auto", side_to_move_override=None):
    return predict_chess_position(image_path, model_path, board_perspective, side_to_move_override)

def main():
    parser = argparse.ArgumentParser(description="Recognizer candidate: clean detector-first path")
    parser.add_argument("image", help="Path to chess board image")
    parser.add_argument("--model-path", default=None, help="Override model path")
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Infer perspective automatically, or force White/Black at the bottom",
    )
    parser.add_argument("--no-edge-detection", action="store_true", help="Disable edge detection")
    parser.add_argument("--no-square-detection", action="store_true", help="Disable square grid detection")
    parser.add_argument(
        "--side-to-move",
        default=None,
        help="Override side to move (aliases: wtm|btm|w|b|white|black)",
    )
    parser.add_argument("--debug", action="store_true", help="Emit DEBUG_JSON telemetry")
    args = parser.parse_args()
    global USE_EDGE_DETECTION, USE_SQUARE_DETECTION, DEBUG_MODE
    USE_EDGE_DETECTION = not args.no_edge_detection
    USE_SQUARE_DETECTION = not args.no_square_detection
    DEBUG_MODE = args.debug
    try:
        result = predict_position(
            args.image,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
            side_to_move_override=args.side_to_move,
        )
        print(
            json.dumps(
                {
                    "success": True,
                    "fen": result["fen"],
                    "confidence": round(result["confidence"], 4),
                    "side_to_move": result["side_to_move"],
                    "side_to_move_source": result["side_to_move_source"],
                    "detected_perspective": result["detected_perspective"],
                    "perspective_source": result["perspective_source"],
                }
            )
        )
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}))

if __name__ == "__main__":
    main()
