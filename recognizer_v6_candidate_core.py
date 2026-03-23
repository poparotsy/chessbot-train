#!/usr/bin/env python3
import hashlib
import os
import re
import sys
from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

try:
    import chess
except ModuleNotFoundError:
    chess = None


IMG_SIZE = 64
FEN_CHARS = "1PNBRQKpnbrqk"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(THIS_DIR, "models", "model_hybrid_v6_champion_48of50.pt")
)
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

_model_cache = {}
_plausibility_cache = {}
_side_to_move_cache = {}
_saturation_cache = {}


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
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=img.size[0] // 3,
        maxLineGap=20,
    )
    if lines is None:
        return None
    h_lines = []
    v_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        if angle < 15 or angle > 165:
            h_lines.append((y1 + y2) // 2)
        elif 75 < angle < 105:
            v_lines.append((x1 + x2) // 2)
    if not h_lines or not v_lines:
        return None

    def cluster_lines(values, threshold=10):
        values = sorted(values)
        clusters = []
        current = [values[0]]
        for value in values[1:]:
            if value - current[-1] < threshold:
                current.append(value)
            else:
                clusters.append(int(np.mean(current)))
                current = [value]
        clusters.append(int(np.mean(current)))
        return clusters

    h_lines = cluster_lines(h_lines)
    v_lines = cluster_lines(v_lines)
    if 7 <= len(h_lines) <= 11 and 7 <= len(v_lines) <= 11:
        if len(h_lines) > 9:
            h_lines = h_lines[:9]
        if len(v_lines) > 9:
            v_lines = v_lines[:9]
        if len(h_lines) == 9 and len(v_lines) == 9:
            return v_lines, h_lines
    return None


def _labels_to_fen(labels):
    rows = []
    for r in range(8):
        rows.append("".join(labels[r * 8 : (r + 1) * 8]))
    return compress_fen_board(rows)


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
    if label_result is not None:
        return label_result["perspective"], label_result["source"]

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
