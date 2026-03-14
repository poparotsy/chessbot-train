#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn


IMG_SIZE = 64
FEN_CHARS = "1PNBRQKpnbrqk"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(THIS_DIR, "models", "model_hybrid_v5_latest_best.pt"))
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
ORIENTATION_STRONG_PIECE_MARGIN = 2.0
ORIENTATION_WEAK_LABEL_MIN_CONF = 0.70
PIECE_LOG_PRIOR = -0.20

LOW_SAT_SPARSE_SAT_MEAN_MAX = 44.0
LOW_SAT_SPARSE_PIECE_MAX = 10
LOW_SAT_SPARSE_EMPTY_ALT_MIN = 0.015
LOW_SAT_SPARSE_ALT_OPTIONS = 2
LOW_SAT_EDGE_ROOK_OBJECTIVE_BONUS = 0.40

GRADIENT_PROJ_MIN_AREA_RATIO = 0.16
GRADIENT_PROJ_MIN_SIDE_RATIO = 0.24
GRADIENT_PROJ_MAX_SIDE_RATIO = 0.94

FAMILY_ORDER = ["full", "contour", "lattice", "gradient_projection", "panel_split"]
FAMILY_BUDGETS = {
    "full": 1,
    "contour": 2,
    "lattice": 2,
    "gradient_projection": 1,
    "panel_split": 1,
}


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


def debug(msg):
    if DEBUG_MODE:
        print(f"DEBUG: {msg}", file=sys.stderr)


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


def perspective_transform(img, corners):
    width = height = 512
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(np.array(img), matrix, (width, height))
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
    opp = max(0.0, min(1.0, metrics["opposite_similarity"] / max(WARP_MIN_OPPOSITE_SIMILARITY, 1e-6)))
    asp = max(0.0, min(1.0, metrics["aspect_similarity"] / max(WARP_MIN_ASPECT_SIMILARITY, 1e-6)))
    angle_span = max(0.0, min(1.0, (metrics["max_angle"] - metrics["min_angle"]) / 180.0))
    angle = 1.0 - angle_span
    return float(0.35 * area + 0.30 * opp + 0.25 * asp + 0.10 * angle)


def find_board_corners(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    img_area = float(h * w)

    def quad_area(corners):
        x = corners[:, 0]
        y = corners[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def score_quad(corners):
        top = np.linalg.norm(corners[1] - corners[0])
        right = np.linalg.norm(corners[2] - corners[1])
        bottom = np.linalg.norm(corners[3] - corners[2])
        left = np.linalg.norm(corners[0] - corners[3])
        min_side = min(top, right, bottom, left)
        if min_side <= 0:
            return -1.0
        area_ratio = quad_area(corners) / img_area
        if area_ratio < 0.20:
            return -1.0
        opposite_similarity = min(top / bottom, bottom / top) * min(left / right, right / left)
        aspect_similarity = min(top / left, left / top) * min(right / bottom, bottom / right)
        xs = corners[:, 0]
        ys = corners[:, 1]
        margin = min(xs.min(), w - xs.max(), ys.min(), h - ys.max()) / max(w, h)
        return area_ratio * 8.0 + opposite_similarity * 6.0 + aspect_similarity * 7.0 + margin * 4.0

    candidates = []
    param_sets = [
        (CANNY_LOW, CANNY_HIGH, None, cv2.RETR_EXTERNAL),
        (30, 100, None, cv2.RETR_EXTERNAL),
        (20, 80, None, cv2.RETR_EXTERNAL),
        (70, 200, None, cv2.RETR_EXTERNAL),
        (CANNY_LOW, CANNY_HIGH, 3, cv2.RETR_LIST),
        (30, 100, 3, cv2.RETR_LIST),
        (20, 80, 3, cv2.RETR_LIST),
    ]

    for low, high, dilate_kernel, retrieval in param_sets:
        edges = cv2.Canny(gray, low, high)
        if dilate_kernel:
            kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, retrieval, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            peri = cv2.arcLength(contour, True)
            for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
                approx = cv2.approxPolyDP(contour, eps * peri, True)
                if len(approx) != 4:
                    continue
                corners = order_corners(approx.reshape(4, 2))
                score = score_quad(corners)
                if score > 0:
                    candidates.append((score, corners))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def find_board_corners_legacy(img):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, CONTOUR_EPSILON * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            image_area = img.size[0] * img.size[1]
            if area > image_area * 0.25:
                return order_corners(approx.reshape(4, 2))
    return None


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
    if (
        left_label
        and right_label
        and left_label["confidence"] >= 0.72
        and right_label["confidence"] >= 0.72
    ):
        if left_label["label"] == "a" and right_label["label"] == "h":
            return {"perspective": "white", "source": "board_labels", "details": details}
        if left_label["label"] == "h" and right_label["label"] == "a":
            return {"perspective": "black", "source": "board_labels", "details": details}
    return None


def infer_board_perspective_from_piece_distribution(fen_board, threshold=2.0):
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
    if (black_mean - white_mean) > threshold:
        return "black"
    return "white"


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
    white_mean = sum(white_rows) / len(white_rows)
    black_mean = sum(black_rows) / len(black_rows)
    return float(abs(white_mean - black_mean))


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
    rows = parse_fen_board_rows(fen_board)
    if rows is None:
        return "w", "default_invalid_board"

    white_king = None
    black_king = None
    for r in range(8):
        for c in range(8):
            if rows[r][c] == "K":
                white_king = (r, c)
            elif rows[r][c] == "k":
                black_king = (r, c)

    if white_king is None or black_king is None:
        return "w", "default_missing_king"

    white_in_check = is_square_attacked(rows, white_king[0], white_king[1], by_white=False)
    black_in_check = is_square_attacked(rows, black_king[0], black_king[1], by_white=True)

    if white_in_check and not black_in_check:
        return "w", "check_inference"
    if black_in_check and not white_in_check:
        return "b", "check_inference"
    if white_in_check and black_in_check:
        return "w", "default_double_check_conflict"
    return "w", "default_no_check_signal"


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
        allowed = "wtm|btm|w|b|white|black"
        raise ValueError(f"Invalid --side-to-move '{raw_value}'. Use one of: {allowed}")
    return mapping[token]


def board_plausibility_score(fen_board):
    rows = expand_fen_board(fen_board)
    if len(rows) != 8 or any(len(row) != 8 for row in rows):
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

    score = 0.0
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
    return score


def image_saturation_stats(img):
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    return {
        "sat_mean": float(np.mean(sat)),
        "sat_std": float(np.std(sat)),
        "val_mean": float(np.mean(val)),
        "val_std": float(np.std(val)),
    }


def trim_dark_edge_bars(img):
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    h, w = gray.shape
    if h < 64 or w < 64:
        return img

    mid = gray[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)]
    if mid.size == 0:
        return img
    ref = float(np.median(mid))
    dark_thr = ref * 0.70

    max_trim_y = int(h * 0.18)
    max_trim_x = int(w * 0.12)
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)

    top = 0
    while top < max_trim_y and row_mean[top] < dark_thr:
        top += 1
    bottom = 0
    while bottom < max_trim_y and row_mean[h - 1 - bottom] < dark_thr:
        bottom += 1
    left = 0
    while left < max_trim_x and col_mean[left] < dark_thr:
        left += 1
    right = 0
    while right < max_trim_x and col_mean[w - 1 - right] < dark_thr:
        right += 1

    if top < int(h * 0.02):
        top = 0
    if bottom < int(h * 0.02):
        bottom = 0
    if left < int(w * 0.015):
        left = 0
    if right < int(w * 0.015):
        right = 0

    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return img

    x0, y0 = left, top
    x1, y1 = w - right, h - bottom
    if x1 - x0 < int(w * 0.65) or y1 - y0 < int(h * 0.65):
        return img

    cropped = img.crop((x0, y0, x1, y1))
    return cropped.resize((w, h), Image.LANCZOS)


def _label_prob_from_topk(tile, label):
    for alt_label, alt_prob in tile["topk"]:
        if alt_label == label:
            return float(alt_prob)
    return 1e-12


def _labels_to_fen(labels):
    rows = []
    for r in range(8):
        rows.append("".join(labels[r * 8 : (r + 1) * 8]))
    return compress_fen_board(rows)


def _build_fen_from_tile_infos(tile_infos):
    rows = []
    for row_idx in range(8):
        row = "".join(tile_infos[row_idx * 8 + col_idx]["label"] for col_idx in range(8))
        rows.append(row)
    return compress_fen_board(rows)


def _king_health_from_labels(labels):
    fen = _labels_to_fen(labels)
    return int(fen.count("K") == 1) + int(fen.count("k") == 1)


def _edge_rook_count_from_labels(labels):
    count = 0
    for idx, label in enumerate(labels):
        if label not in {"R", "r"}:
            continue
        row = idx // 8
        col = idx % 8
        if row in (0, 7) or col in (0, 7):
            count += 1
    return count


def enforce_king_counts_from_topk(tile_infos):
    repaired = [dict(tile, topk=list(tile["topk"])) for tile in tile_infos]

    for king_label in ("K", "k"):
        king_tiles = [tile for tile in repaired if tile["label"] == king_label]
        if len(king_tiles) <= 1:
            continue
        king_tiles.sort(key=lambda tile: tile["prob"], reverse=True)
        for extra_tile in king_tiles[1:]:
            replacement_label = None
            replacement_prob = -1.0
            empty_prob = float(extra_tile.get("empty_prob", _label_prob_from_topk(extra_tile, "1")))
            if empty_prob > 0.0:
                replacement_label = "1"
                replacement_prob = empty_prob
            for alt_label, alt_prob in extra_tile["topk"]:
                if alt_label in {"K", "k"}:
                    continue
                if alt_label == "1":
                    continue
                # Prefer empty squares over hallucinating a replacement piece
                # when demoting an extra king from noise/logo artifacts.
                if replacement_label != "1" and float(alt_prob) > replacement_prob:
                    replacement_label = alt_label
                    replacement_prob = float(alt_prob)
            if replacement_label is None:
                replacement_label = "1"
                replacement_prob = float(extra_tile.get("empty_prob", 0.0))
            extra_tile["label"] = replacement_label
            extra_tile["prob"] = replacement_prob

    white_kings = sum(1 for tile in repaired if tile["label"] == "K")
    black_kings = sum(1 for tile in repaired if tile["label"] == "k")
    if white_kings != 1 or black_kings != 1:
        return None

    piece_count = sum(1 for tile in repaired if tile["label"] != "1")
    mean_conf = float(np.mean([tile["prob"] for tile in repaired])) if repaired else 0.0
    return _build_fen_from_tile_infos(repaired), mean_conf, piece_count


def repair_missing_king_from_topk(tile_infos, base_fen, base_conf):
    if not tile_infos:
        return None

    base_labels = [tile["label"] for tile in tile_infos]
    missing = [king for king in ("K", "k") if sum(1 for label in base_labels if label == king) == 0]
    if len(missing) != 1:
        return None

    target_king = missing[0]
    base_piece_count = sum(1 for label in base_labels if label != "1")
    base_kh = _king_health_from_labels(base_labels)
    base_obj = board_plausibility_score(base_fen) + 3.0 * base_kh + 0.12 * base_piece_count
    best = (base_obj, base_fen, base_conf, base_piece_count)

    for idx, tile in enumerate(tile_infos):
        current = tile["label"]
        if current in {"K", "k"} and current != target_king:
            continue

        trial_labels = list(base_labels)
        trial_labels[idx] = target_king
        trial_fen = _labels_to_fen(trial_labels)
        trial_kh = _king_health_from_labels(trial_labels)
        if trial_kh < 2:
            continue

        king_prob = _label_prob_from_topk(tile, target_king)
        if current == "1":
            current_prob = float(tile.get("empty_prob", _label_prob_from_topk(tile, "1")))
        else:
            current_prob = _label_prob_from_topk(tile, current)

        trial_piece_count = sum(1 for label in trial_labels if label != "1")
        trial_plaus = board_plausibility_score(trial_fen)
        trial_probs = np.array(
            [_label_prob_from_topk(tile_infos[i], trial_labels[i]) for i in range(64)],
            dtype=np.float32,
        )
        trial_conf = float(np.mean(trial_probs))
        trial_obj = (
            trial_plaus
            + 3.0 * trial_kh
            + 0.12 * trial_piece_count
            + 0.35 * float(np.log(max(king_prob, 1e-12)))
            - 0.10 * float(np.log(max(current_prob, 1e-12)))
        )
        if trial_obj > best[0]:
            best = (trial_obj, trial_fen, trial_conf, trial_piece_count)

    if best[0] <= base_obj + 2.0 or best[1] == base_fen:
        return None
    return best[1], float(best[2]), int(best[3])


def promote_sparse_pieces_from_topk(tile_infos, base_fen, base_conf):
    if not tile_infos:
        return None

    base_labels = [tile["label"] for tile in tile_infos]
    base_piece_count = sum(1 for label in base_labels if label != "1")
    if base_piece_count > 8:
        return None

    uncertain = []
    for idx, tile in enumerate(tile_infos):
        if tile["label"] != "1":
            continue
        empty_prob = float(tile.get("empty_prob", _label_prob_from_topk(tile, "1")))
        options = []
        for alt_label, alt_prob in tile["topk"]:
            if alt_label == "1":
                continue
            prob = float(alt_prob)
            if prob < max(0.18, empty_prob + 0.05):
                continue
            options.append((alt_label, prob))
            if len(options) >= 2:
                break
        if not options:
            continue
        best_gain = max(prob - empty_prob for _, prob in options)
        uncertain.append((best_gain, idx, options))

    if not uncertain:
        return None

    uncertain.sort(reverse=True)
    uncertain = uncertain[:8]
    beam = [(base_labels, 0.0)]

    for _gain, idx, options in uncertain:
        next_beam = {}
        for labels, log_bonus in beam:
            key_keep = tuple(labels)
            prev_keep = next_beam.get(key_keep)
            if prev_keep is None or log_bonus > prev_keep:
                next_beam[key_keep] = log_bonus

            changes = sum(1 for left, right in zip(labels, base_labels) if left != right)
            if changes >= 2:
                continue
            for alt_label, alt_prob in options:
                new_labels = list(labels)
                new_labels[idx] = alt_label
                key_new = tuple(new_labels)
                new_bonus = log_bonus + float(np.log(max(alt_prob, 1e-12)))
                prev_new = next_beam.get(key_new)
                if prev_new is None or new_bonus > prev_new:
                    next_beam[key_new] = new_bonus

        scored = []
        for labels_tuple, log_bonus in next_beam.items():
            labels = list(labels_tuple)
            fen = _labels_to_fen(labels)
            plaus = board_plausibility_score(fen)
            k_health = _king_health_from_labels(labels)
            piece_count = sum(1 for label in labels if label != "1")
            objective = plaus + 1.5 * k_health + 0.30 * piece_count + 0.35 * (log_bonus / 8.0)
            scored.append((objective, labels, log_bonus))
        scored.sort(key=lambda row: row[0], reverse=True)
        beam = [(labels, log_bonus) for _objective, labels, log_bonus in scored[:24]]

    base_obj = board_plausibility_score(base_fen) + 1.5 * _king_health_from_labels(base_labels) + 0.30 * base_piece_count
    best = (base_obj, base_fen, base_conf, base_piece_count)
    for labels, log_bonus in beam:
        fen = _labels_to_fen(labels)
        piece_count = sum(1 for label in labels if label != "1")
        probs = np.array([_label_prob_from_topk(tile_infos[i], labels[i]) for i in range(64)], dtype=np.float32)
        conf = float(np.mean(probs))
        obj = board_plausibility_score(fen) + 1.5 * _king_health_from_labels(labels) + 0.30 * piece_count + 0.35 * (log_bonus / 8.0)
        if obj > best[0]:
            best = (obj, fen, conf, piece_count)

    if best[0] <= base_obj + 0.12 or best[1] == base_fen:
        return None
    return best[1], float(best[2]), int(best[3])


def rescore_low_saturation_sparse_from_topk(tile_infos, base_fen, base_conf):
    if not tile_infos:
        return None

    base_labels = [tile["label"] for tile in tile_infos]
    base_piece_count = sum(1 for x in base_labels if x != "1")
    if base_piece_count > LOW_SAT_SPARSE_PIECE_MAX:
        return None

    uncertain = []
    for idx, tile in enumerate(tile_infos):
        if tile["label"] != "1":
            continue
        empty_prob = float(tile.get("empty_prob", _label_prob_from_topk(tile, "1")))
        options = []
        for alt_label, alt_prob in tile["topk"]:
            if alt_label == "1":
                continue
            prob = float(alt_prob)
            if prob < LOW_SAT_SPARSE_EMPTY_ALT_MIN:
                continue
            options.append((alt_label, prob))
            if len(options) >= LOW_SAT_SPARSE_ALT_OPTIONS:
                break
        if not options:
            continue
        best_gain = max(prob - empty_prob for _, prob in options)
        uncertain.append((best_gain, idx, empty_prob, options))
    if not uncertain:
        return None

    uncertain.sort(reverse=True)
    uncertain = uncertain[:12]
    beam = [(base_labels, 0, 0.0)]

    for _gain, idx, _empty_prob, options in uncertain:
        next_beam = {}
        for labels, changes, log_bonus in beam:
            key_keep = tuple(labels)
            prev_keep = next_beam.get(key_keep)
            if prev_keep is None or log_bonus > prev_keep[2]:
                next_beam[key_keep] = (labels, changes, log_bonus)
            if changes >= 2:
                continue
            for alt_label, alt_prob in options:
                new_labels = list(labels)
                new_labels[idx] = alt_label
                new_log_bonus = log_bonus + float(np.log(max(alt_prob, 1e-12)))
                key_new = tuple(new_labels)
                prev_new = next_beam.get(key_new)
                if prev_new is None or new_log_bonus > prev_new[2]:
                    next_beam[key_new] = (new_labels, changes + 1, new_log_bonus)

        scored = []
        for labels, changes, log_bonus in next_beam.values():
            fen = _labels_to_fen(labels)
            plaus = board_plausibility_score(fen)
            k_health = _king_health_from_labels(labels)
            piece_count = sum(1 for x in labels if x != "1")
            edge_rook_count = _edge_rook_count_from_labels(labels)
            objective = (
                plaus
                + 1.5 * k_health
                + LOW_SAT_EDGE_ROOK_OBJECTIVE_BONUS * edge_rook_count
                + 0.5 * (log_bonus / 8.0)
            )
            scored.append((objective, labels, changes, log_bonus))
        scored.sort(key=lambda row: row[0], reverse=True)
        beam = [(labels, changes, log_bonus) for _obj, labels, changes, log_bonus in scored[:24]]

    def promote_missing_edge_rook(labels):
        out = list(labels)

        def score_current_label(tile, current_label):
            if current_label == "1":
                return float(tile.get("empty_prob", _label_prob_from_topk(tile, "1")))
            return _label_prob_from_topk(tile, current_label)

        def try_color(target_rook, opposite_rook, same_color_check):
            if sum(1 for x in out if x == target_rook) > 0:
                return
            if sum(1 for x in out if x == opposite_rook) == 0:
                return

            best_idx = None
            best_score = 0.0
            for idx, current in enumerate(out):
                if current in {"K", "k"}:
                    continue
                if current != "1" and not same_color_check(current):
                    continue
                row = idx // 8
                col = idx % 8
                if row not in (0, 7) and col not in (0, 7):
                    continue
                rook_prob = _label_prob_from_topk(tile_infos[idx], target_rook)
                if rook_prob < 0.03:
                    continue
                cur_prob = score_current_label(tile_infos[idx], current)
                edge_bonus = 0.03 if row in (0, 7) else 0.0
                empty_bonus = 0.02 if current == "1" else 0.0
                score = rook_prob - 0.10 * cur_prob + edge_bonus + empty_bonus
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None and best_score > 0.0:
                out[best_idx] = target_rook

        try_color("r", "R", lambda ch: ch.islower())
        try_color("R", "r", lambda ch: ch.isupper())
        return out

    def eval_labels(labels, log_bonus):
        labels = promote_missing_edge_rook(labels)
        fen = _labels_to_fen(labels)
        plaus = board_plausibility_score(fen)
        k_health = _king_health_from_labels(labels)
        piece_count = sum(1 for x in labels if x != "1")
        edge_rook_count = _edge_rook_count_from_labels(labels)
        probs = np.array([_label_prob_from_topk(tile_infos[i], labels[i]) for i in range(64)], dtype=np.float32)
        conf = float(np.mean(probs))
        objective = (
            plaus
            + 1.5 * k_health
            + LOW_SAT_EDGE_ROOK_OBJECTIVE_BONUS * edge_rook_count
            + 0.5 * (log_bonus / 8.0)
        )
        return objective, fen, conf, piece_count, k_health

    base_obj, _, _, _, base_kh = eval_labels(base_labels, 0.0)
    best = (base_obj, base_fen, base_conf, base_piece_count, base_kh)
    for labels, _changes, log_bonus in beam:
        obj, fen, conf, piece_count, k_health = eval_labels(labels, log_bonus)
        if k_health < base_kh:
            continue
        if piece_count < base_piece_count:
            continue
        if obj > best[0]:
            best = (obj, fen, conf, piece_count, k_health)

    improved = best[0] > (base_obj + 0.15)
    if not improved or best[1] == base_fen:
        return None
    return best[1], float(best[2]), int(best[3])


def suppress_sparse_false_positives(tile_infos, base_fen, base_conf):
    if not tile_infos:
        return None

    base_labels = [tile["label"] for tile in tile_infos]
    base_piece_count = sum(1 for label in base_labels if label != "1")
    if base_piece_count > 12:
        return None

    uncertain = []
    for idx, tile in enumerate(tile_infos):
        current = tile["label"]
        if current in {"1", "K", "k"}:
            continue
        current_prob = _label_prob_from_topk(tile, current)
        empty_prob = float(tile.get("empty_prob", _label_prob_from_topk(tile, "1")))
        if empty_prob < 0.04:
            continue
        closeness = empty_prob - 0.75 * current_prob
        uncertain.append((closeness, idx, current_prob, empty_prob))

    if not uncertain:
        return None

    uncertain.sort(reverse=True)
    uncertain = uncertain[:16]
    beam = [(base_labels, 0, 0.0)]

    for _gain, idx, current_prob, empty_prob in uncertain:
        next_beam = {}
        for labels, changes, log_bonus in beam:
            key_keep = tuple(labels)
            prev_keep = next_beam.get(key_keep)
            if prev_keep is None or log_bonus > prev_keep[2]:
                next_beam[key_keep] = (labels, changes, log_bonus)

            if changes >= 3:
                continue
            new_labels = list(labels)
            new_labels[idx] = "1"
            new_log_bonus = log_bonus + float(np.log(max(empty_prob, 1e-12))) - float(np.log(max(current_prob, 1e-12)))
            key_new = tuple(new_labels)
            prev_new = next_beam.get(key_new)
            if prev_new is None or new_log_bonus > prev_new[2]:
                next_beam[key_new] = (new_labels, changes + 1, new_log_bonus)

        scored = []
        for labels, changes, log_bonus in next_beam.values():
            fen = _labels_to_fen(labels)
            plaus = board_plausibility_score(fen)
            k_health = _king_health_from_labels(labels)
            piece_count = sum(1 for x in labels if x != "1")
            objective = plaus + 1.5 * k_health - 0.10 * piece_count + 0.5 * log_bonus
            scored.append((objective, labels, changes, log_bonus))
        scored.sort(key=lambda row: row[0], reverse=True)
        beam = [(labels, changes, log_bonus) for _obj, labels, changes, log_bonus in scored[:24]]

    def eval_labels(labels, log_bonus):
        fen = _labels_to_fen(labels)
        plaus = board_plausibility_score(fen)
        k_health = _king_health_from_labels(labels)
        piece_count = sum(1 for x in labels if x != "1")
        probs = np.array([_label_prob_from_topk(tile_infos[i], labels[i]) for i in range(64)], dtype=np.float32)
        conf = float(np.mean(probs))
        objective = plaus + 1.5 * k_health - 0.10 * piece_count + 0.5 * log_bonus
        return objective, fen, conf, piece_count, k_health

    base_obj, _, _, _, base_kh = eval_labels(base_labels, 0.0)
    best = (base_obj, base_fen, base_conf, base_piece_count, base_kh)
    for labels, _changes, log_bonus in beam:
        obj, fen, conf, piece_count, k_health = eval_labels(labels, log_bonus)
        if k_health < base_kh:
            continue
        if obj > best[0]:
            best = (obj, fen, conf, piece_count, k_health)

    if best[0] <= base_obj + 0.08 or best[1] == base_fen:
        return None
    return best[1], float(best[2]), int(best[3])


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
        topk = [(FEN_CHARS[int(k_idx.item())], float(prob.item())) for prob, k_idx in zip(topk_probs, topk_pred)]

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
    final_fen = fen
    final_conf = float(np.mean(confs))
    final_piece_count = piece_count

    repaired_kings = enforce_king_counts_from_topk(tile_infos)
    if repaired_kings is not None and repaired_kings[0] != final_fen:
        final_fen, final_conf, final_piece_count = repaired_kings

    sparse_cleanup = suppress_sparse_false_positives(tile_infos, base_fen=final_fen, base_conf=final_conf)
    if sparse_cleanup is not None and sparse_cleanup[0] != final_fen:
        final_fen, final_conf, final_piece_count = sparse_cleanup

    details = {"base_fen": fen, "final_fen": final_fen, "tile_infos": tile_infos}

    if return_details:
        return final_fen, final_conf, final_piece_count, details
    return final_fen, final_conf, final_piece_count


class BoardDetector:
    def __init__(self, debug=False):
        self.debug = debug

    def _log(self, msg):
        if self.debug:
            print(f"DEBUG: {msg}", file=sys.stderr)

    def _gray(self, img):
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    def _clip01(self, value):
        return float(max(0.0, min(1.0, float(value))))

    def _lens_confidence(self, metrics, trusted, relaxed, extra=0.0):
        area = self._clip01(metrics["area_ratio"] / 0.70)
        opp = self._clip01(metrics["opposite_similarity"])
        asp = self._clip01(metrics["aspect_similarity"])
        min_angle = self._clip01(metrics["min_angle"] / 90.0)
        max_angle_penalty = self._clip01((180.0 - metrics["max_angle"]) / 90.0)
        conf = (
            0.35 * warp_geometry_quality(metrics)
            + 0.25 * area
            + 0.20 * opp
            + 0.15 * asp
            + 0.05 * min(min_angle, max_angle_penalty)
            + float(extra)
        )
        if trusted:
            conf += 0.08
        elif relaxed:
            conf += 0.03
        return self._clip01(conf)

    def _lens_candidate(self, tag, corners, img, extra_conf=0.0, raw_score=None):
        metrics = compute_quad_metrics(corners, img.size[0], img.size[1])
        trusted = is_warp_geometry_trustworthy(metrics)
        relaxed = is_warp_geometry_relaxed(metrics)
        return {
            "tag": tag,
            "corners": order_corners(corners),
            "metrics": metrics,
            "trusted": trusted,
            "relaxed": relaxed,
            "warp_quality": warp_geometry_quality(metrics),
            "lens_confidence": self._lens_confidence(metrics, trusted, relaxed, extra=extra_conf),
            "raw_score": None if raw_score is None else float(raw_score),
        }

    def _angular_distance_deg(self, a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    def _dominant_orthogonal_orientations(self, angles, weights):
        if not angles:
            return None
        hist = np.zeros(180, dtype=np.float32)
        for angle, weight in zip(angles, weights):
            hist[int(round(angle)) % 180] += max(float(weight), 1e-3)
        primary = int(np.argmax(hist))
        secondary_target = (primary + 90) % 180
        search_radius = 18
        best_secondary = None
        best_secondary_val = -1.0
        for delta in range(-search_radius, search_radius + 1):
            idx = (secondary_target + delta) % 180
            if hist[idx] > best_secondary_val:
                best_secondary_val = float(hist[idx])
                best_secondary = idx
        if best_secondary is None or best_secondary_val <= 0.0:
            return None
        return float(primary), float(best_secondary)

    def _orientation_hypotheses(self, angles, weights, top_n=5, min_sep_deg=14, sec_radius=24):
        if not angles:
            return []
        hist = np.zeros(180, dtype=np.float32)
        for angle, weight in zip(angles, weights):
            hist[int(round(angle)) % 180] += max(float(weight), 1e-3)

        primaries = []
        order = np.argsort(hist)[::-1]
        for idx in order:
            idx = int(idx)
            if hist[idx] <= 0.0:
                continue
            if any(self._angular_distance_deg(idx, p) < min_sep_deg for p in primaries):
                continue
            primaries.append(idx)
            if len(primaries) >= top_n:
                break

        pairs = []
        seen = set()
        for primary in primaries:
            secondary_target = (primary + 90) % 180
            best_secondary = None
            best_secondary_val = -1.0
            for delta in range(-sec_radius, sec_radius + 1):
                idx = int((secondary_target + delta) % 180)
                val = float(hist[idx])
                if val > best_secondary_val:
                    best_secondary_val = val
                    best_secondary = idx
            if best_secondary is None or best_secondary_val <= 0.0:
                continue
            a, b = sorted((int(primary), int(best_secondary)))
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((float(primary), float(best_secondary), float(hist[primary] + best_secondary_val)))
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs

    def _line_intersection(self, n1, rho1, n2, rho2):
        a = np.array([[n1[0], n1[1]], [n2[0], n2[1]]], dtype=np.float32)
        b = np.array([rho1, rho2], dtype=np.float32)
        det = float(np.linalg.det(a))
        if abs(det) < 1e-6:
            return None
        pt = np.linalg.solve(a, b)
        return float(pt[0]), float(pt[1])

    def _cluster_axis(self, items, threshold=10):
        if not items:
            return []
        items = sorted(items, key=lambda item: item[0])
        clusters = [[items[0]]]
        for item in items[1:]:
            if item[0] - clusters[-1][-1][0] <= threshold:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        merged = []
        for cluster in clusters:
            positions = np.array([item[0] for item in cluster], dtype=np.float32)
            weights = np.array([item[1] for item in cluster], dtype=np.float32)
            merged.append({
                "pos": float(np.average(positions, weights=np.maximum(weights, 1e-3))),
                "weight": float(weights.sum()),
                "count": len(cluster),
            })
        return merged

    def _best_line_window(self, clustered, span):
        if len(clustered) < span:
            return None
        best = None
        for start in range(0, len(clustered) - span + 1):
            window = clustered[start : start + span]
            positions = np.array([line["pos"] for line in window], dtype=np.float32)
            diffs = np.diff(positions)
            if np.any(diffs <= 1):
                continue
            spacing_mean = float(np.mean(diffs))
            spacing_std = float(np.std(diffs))
            regularity = 1.0 / (1.0 + (spacing_std / max(spacing_mean, 1e-6)))
            strength = float(sum(line["weight"] for line in window))
            score = regularity * 1000.0 + strength
            candidate = {
                "positions": positions,
                "regularity": regularity,
                "strength": strength,
                "spacing_mean": spacing_mean,
                "score": score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best

    def _periodicity_score(self, signal, min_lag=6):
        arr = np.asarray(signal, dtype=np.float32)
        n = arr.shape[0]
        if n < max(24, min_lag + 4):
            return 0.0
        arr = arr - float(arr.mean())
        denom = float(np.dot(arr, arr))
        if denom <= 1e-6:
            return 0.0
        acf = np.correlate(arr, arr, mode="full")[n - 1 :]
        lag_lo = min_lag
        lag_hi = max(lag_lo + 1, min(n // 3, n - 1))
        if lag_hi <= lag_lo:
            return 0.0
        peak = float(np.max(acf[lag_lo:lag_hi]))
        return float(max(0.0, min(1.0, peak / denom)))

    def _tile_acf_score(self, signal, tile_period):
        arr = np.asarray(signal, dtype=np.float32)
        n = arr.shape[0]
        if n < tile_period * 4:
            return 0.0
        arr = arr - float(arr.mean())
        denom = float(np.dot(arr, arr))
        if denom <= 1e-6:
            return 0.0
        acf = np.correlate(arr, arr, mode="full")[n - 1 :]
        peaks = []
        for k in range(1, 5):
            center = int(round(k * tile_period))
            if center + 3 >= n:
                break
            lo = max(1, center - 3)
            hi = min(n - 1, center + 4)
            peaks.append(float(np.max(acf[lo:hi])) / denom)
        if not peaks:
            return 0.0
        return float(max(0.0, min(1.0, np.mean(peaks))))

    def _periodicity_2d_score(self, gray, grad, x0, y0, x1, y1):
        roi_grad = grad[y0 : y1 + 1, x0 : x1 + 1]
        roi_gray = gray[y0 : y1 + 1, x0 : x1 + 1]
        hh, ww = roi_gray.shape[:2]
        if hh < 120 or ww < 120:
            return 0.0, 0.0, 0.0
        x_proj = roi_grad.mean(axis=0)
        y_proj = roi_grad.mean(axis=1)
        px = self._periodicity_score(x_proj, min_lag=max(5, ww // 42))
        py = self._periodicity_score(y_proj, min_lag=max(5, hh // 42))
        periodicity = 0.5 * (px + py)
        texture = float(np.mean(roi_grad))
        contrast = float(np.std(roi_gray))
        return float(periodicity), texture, contrast

    def detect_gradient_projection(self, img):
        gray = self._gray(img)
        h, w = gray.shape
        if min(h, w) < 200:
            return None
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
        grad = 0.5 * gx + 0.5 * gy

        col_energy = gx.mean(axis=0)
        row_energy = gy.mean(axis=1)
        if w >= 31:
            col_energy = cv2.GaussianBlur(col_energy.reshape(1, -1), (1, 31), 0).reshape(-1)
        if h >= 31:
            row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (31, 1), 0).reshape(-1)

        def projection_peak_pair(signal, min_span_ratio=0.24):
            arr = np.asarray(signal, dtype=np.float32)
            n = int(arr.shape[0])
            if n < 40:
                return None
            lo = int(n * 0.06)
            hi = int(n * 0.94)
            mid = (lo + hi) // 2
            if hi - lo < 20:
                return None
            left_band = arr[lo:mid]
            right_band = arr[mid:hi]
            if left_band.size == 0 or right_band.size == 0:
                return None
            li = int(np.argmax(left_band)) + lo
            ri = int(np.argmax(right_band)) + mid
            if ri <= li or (ri - li) < int(n * min_span_ratio):
                return None
            mean_signal = float(np.mean(arr) + 1e-6)
            peak_gain = float((arr[li] + arr[ri]) * 0.5 / mean_signal)
            return li, ri, peak_gain

        x_pair = projection_peak_pair(col_energy, min_span_ratio=GRADIENT_PROJ_MIN_SIDE_RATIO)
        y_pair = projection_peak_pair(row_energy, min_span_ratio=GRADIENT_PROJ_MIN_SIDE_RATIO)
        if x_pair is None or y_pair is None:
            return None

        lx, rx, x_gain = x_pair
        ty, by, y_gain = y_pair
        pad_x = max(2, int(w * 0.015))
        pad_y = max(2, int(h * 0.015))
        x0 = max(0, lx - pad_x)
        x1 = min(w - 1, rx + pad_x)
        y0 = max(0, ty - pad_y)
        y1 = min(h - 1, by + pad_y)
        ww = x1 - x0 + 1
        hh = y1 - y0 + 1
        if ww <= 4 or hh <= 4:
            return None

        width_ratio = ww / float(w)
        height_ratio = hh / float(h)
        ratio = ww / float(max(1, hh))
        area_ratio = (ww * hh) / float(w * h)
        if (
            width_ratio < GRADIENT_PROJ_MIN_SIDE_RATIO
            or height_ratio < GRADIENT_PROJ_MIN_SIDE_RATIO
            or width_ratio > GRADIENT_PROJ_MAX_SIDE_RATIO
            or height_ratio > GRADIENT_PROJ_MAX_SIDE_RATIO
            or ratio < 0.52
            or ratio > 1.95
            or area_ratio < GRADIENT_PROJ_MIN_AREA_RATIO
        ):
            return None

        periodicity, texture_raw, contrast_raw = self._periodicity_2d_score(gray, grad, x0, y0, x1, y1)
        if periodicity <= 0.03:
            return None

        edge_gain = 0.5 * (x_gain + y_gain)
        area_pref = 1.0 - min(1.0, abs(area_ratio - 0.42) / 0.42)
        square_pref = 1.0 - min(1.0, abs(1.0 - ratio) / 0.8)
        texture = float(texture_raw / (float(np.mean(grad)) + 1e-6))
        contrast = float(contrast_raw / (float(np.std(gray)) + 1e-6))
        score = 2.6 * periodicity + 0.70 * max(0.0, edge_gain - 1.0) + 0.40 * texture + 0.30 * contrast + 0.25 * area_pref + 0.25 * square_pref
        return {
            "tag": "gradient_projection",
            "corners": np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32),
            "score": float(score),
            "periodicity": float(periodicity),
        }

    def _warp_grid_score(self, img, corners, size=256):
        arr = np.array(img)
        src = corners.astype(np.float32)
        dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
        mat = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(arr, mat, (size, size))
        gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        x_proj = gx.mean(axis=0)
        y_proj = gy.mean(axis=1)
        tile = max(6, size // 8)
        sx = self._tile_acf_score(x_proj, tile)
        sy = self._tile_acf_score(y_proj, tile)

        line_pos = [int(round(i * size / 8.0)) for i in range(9)]
        edge_band = 2
        line_x = []
        off_x = []
        line_y = []
        off_y = []
        for p in line_pos:
            lo = max(0, p - edge_band)
            hi = min(size, p + edge_band + 1)
            line_x.append(float(np.mean(x_proj[lo:hi])))
            line_y.append(float(np.mean(y_proj[lo:hi])))
        for p in [int(round((i + 0.5) * size / 8.0)) for i in range(8)]:
            lo = max(0, p - edge_band)
            hi = min(size, p + edge_band + 1)
            off_x.append(float(np.mean(x_proj[lo:hi])))
            off_y.append(float(np.mean(y_proj[lo:hi])))
        lx = float(np.mean(line_x) / max(1e-6, np.mean(off_x) if off_x else np.mean(line_x)))
        ly = float(np.mean(line_y) / max(1e-6, np.mean(off_y) if off_y else np.mean(line_y)))
        line_sep = max(0.0, min(1.0, 0.5 * (lx + ly) - 0.6))
        return float(max(0.0, min(1.0, 0.55 * (sx + sy) * 0.5 + 0.45 * line_sep)))

    def _refine_corners_grid_fit(self, img, corners):
        h, w = img.size[1], img.size[0]
        cur = order_corners(corners.copy().astype(np.float32))
        base_metrics = compute_quad_metrics(cur, w, h)
        base_score = 0.70 * self._warp_grid_score(img, cur) + 0.30 * warp_geometry_quality(base_metrics)
        best_score = float(base_score)

        for frac in (0.04, 0.02):
            step = max(2.0, min(w, h) * frac)
            improved_any = False
            for idx in range(4):
                local_best = cur.copy()
                local_score = best_score
                for dx in (-step, 0.0, step):
                    for dy in (-step, 0.0, step):
                        if dx == 0.0 and dy == 0.0:
                            continue
                        trial = cur.copy()
                        trial[idx, 0] = np.clip(trial[idx, 0] + dx, 0, w - 1)
                        trial[idx, 1] = np.clip(trial[idx, 1] + dy, 0, h - 1)
                        trial = order_corners(trial)
                        metrics = compute_quad_metrics(trial, w, h)
                        if metrics["area_ratio"] < 0.16:
                            continue
                        gscore = self._warp_grid_score(img, trial)
                        score = 0.70 * gscore + 0.30 * warp_geometry_quality(metrics)
                        if score > local_score + 1e-4:
                            local_score = float(score)
                            local_best = trial
                if local_score > best_score + 1e-4:
                    best_score = local_score
                    cur = local_best
                    improved_any = True
            if not improved_any:
                break
        return cur, float(best_score)

    def detect_panel_split(self, img):
        gray = self._gray(img)
        h, w = gray.shape
        if min(h, w) < 200:
            return None

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.convertScaleAbs(0.5 * np.abs(gx) + 0.5 * np.abs(gy)).astype(np.float32)
        candidates = []

        col_energy = grad.mean(axis=0)
        row_energy = grad.mean(axis=1)
        if w >= 31:
            col_energy = cv2.GaussianBlur(col_energy.reshape(1, -1), (1, 31), 0).reshape(-1)
        if h >= 31:
            row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (31, 1), 0).reshape(-1)

        total_col = float(np.mean(col_energy) + 1e-6)
        total_row = float(np.mean(row_energy) + 1e-6)

        def score_rect(tag, x0, y0, x1, y1, edge_gain, total_mean):
            ww = x1 - x0 + 1
            hh = y1 - y0 + 1
            ratio = ww / float(max(1, hh))
            if ww < int(w * 0.28) or hh < int(h * 0.34) or ratio < 0.52 or ratio > 1.95:
                return
            roi_grad = grad[y0 : y1 + 1, x0 : x1 + 1]
            roi_gray = gray[y0 : y1 + 1, x0 : x1 + 1]
            x_proj = roi_grad.mean(axis=0)
            y_proj = roi_grad.mean(axis=1)
            px = self._periodicity_score(x_proj, min_lag=max(5, ww // 40))
            py = self._periodicity_score(y_proj, min_lag=max(5, hh // 40))
            periodicity = 0.5 * (px + py)
            texture = float(np.mean(roi_grad) / (total_mean * 2.0))
            contrast = float(np.std(roi_gray) / 64.0)
            score = 2.6 * periodicity + 0.95 * max(0.0, edge_gain) + 0.45 * texture + 0.25 * contrast
            candidates.append({
                "tag": tag,
                "corners": np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32),
                "score": float(score),
                "periodicity": float(periodicity),
            })

        x_lo = int(w * 0.18)
        x_hi = int(w * 0.82)
        x_step = max(4, w // 160)
        for split in range(x_lo, x_hi, x_step):
            left_mean = float(np.mean(col_energy[:split])) if split > 8 else 0.0
            right_mean = float(np.mean(col_energy[split:])) if split < w - 8 else 0.0

            x0, x1 = split, w - 1
            roi_rows = grad[:, x0 : x1 + 1].mean(axis=1)
            thr = float(np.quantile(roi_rows, 0.35))
            active = np.where(roi_rows >= thr)[0]
            if active.size >= int(h * 0.45):
                y0 = max(0, int(active[0]) - max(3, h // 120))
                y1 = min(h - 1, int(active[-1]) + max(3, h // 120))
            else:
                y0, y1 = 0, h - 1
            score_rect("panel_split_right", x0, y0, x1, y1, (right_mean - left_mean) / total_col, total_col)

            x0, x1 = 0, split
            roi_rows = grad[:, x0 : x1 + 1].mean(axis=1)
            thr = float(np.quantile(roi_rows, 0.35))
            active = np.where(roi_rows >= thr)[0]
            if active.size >= int(h * 0.45):
                y0 = max(0, int(active[0]) - max(3, h // 120))
                y1 = min(h - 1, int(active[-1]) + max(3, h // 120))
            else:
                y0, y1 = 0, h - 1
            score_rect("panel_split_left", x0, y0, x1, y1, (left_mean - right_mean) / total_col, total_col)

        y_lo = int(h * 0.14)
        y_hi = int(h * 0.86)
        y_step = max(4, h // 160)
        for split in range(y_lo, y_hi, y_step):
            top_mean = float(np.mean(row_energy[:split])) if split > 8 else 0.0
            bottom_mean = float(np.mean(row_energy[split:])) if split < h - 8 else 0.0

            y0, y1 = split, h - 1
            roi_cols = grad[y0 : y1 + 1, :].mean(axis=0)
            thr = float(np.quantile(roi_cols, 0.35))
            active = np.where(roi_cols >= thr)[0]
            if active.size >= int(w * 0.45):
                x0 = max(0, int(active[0]) - max(3, w // 120))
                x1 = min(w - 1, int(active[-1]) + max(3, w // 120))
            else:
                x0, x1 = 0, w - 1
            score_rect("panel_split_bottom", x0, y0, x1, y1, (bottom_mean - top_mean) / total_row, total_row)

            y0, y1 = 0, split
            roi_cols = grad[y0 : y1 + 1, :].mean(axis=0)
            thr = float(np.quantile(roi_cols, 0.35))
            active = np.where(roi_cols >= thr)[0]
            if active.size >= int(w * 0.45):
                x0 = max(0, int(active[0]) - max(3, w // 120))
                x1 = min(w - 1, int(active[-1]) + max(3, w // 120))
            else:
                x0, x1 = 0, w - 1
            score_rect("panel_split_top", x0, y0, x1, y1, (top_mean - bottom_mean) / total_row, total_row)

        if not candidates:
            return None
        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]
        if best["score"] < 0.60:
            return None
        return best

    def detect_lattice(self, img):
        gray = self._gray(img)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120, apertureSize=3)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=60,
            minLineLength=min(img.size) // 5,
            maxLineGap=25,
        )
        if lines is None:
            return None

        segments = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in line]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < min(img.size) * 0.18:
                continue
            angle = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            segments.append((angle, length, mx, my))
        if len(segments) < 18:
            return None

        angles = [seg[0] for seg in segments]
        strengths = [seg[1] for seg in segments]
        orientation_pairs = self._orientation_hypotheses(angles, strengths)
        if not orientation_pairs:
            dominant = self._dominant_orthogonal_orientations(angles, strengths)
            if dominant is not None:
                orientation_pairs = [(dominant[0], dominant[1], 0.0)]
        if not orientation_pairs:
            return None

        best_result = None
        for theta_a, theta_b, pair_strength in orientation_pairs:
            family_a = []
            family_b = []
            for angle, length, mx, my in segments:
                da = self._angular_distance_deg(angle, theta_a)
                db = self._angular_distance_deg(angle, theta_b)
                if da <= 18 and da <= db:
                    na = np.deg2rad(theta_a + 90.0)
                    pos = mx * np.cos(na) + my * np.sin(na)
                    family_a.append((pos, length))
                elif db <= 18:
                    nb = np.deg2rad(theta_b + 90.0)
                    pos = mx * np.cos(nb) + my * np.sin(nb)
                    family_b.append((pos, length))

            a_clusters = self._cluster_axis(family_a, threshold=max(8, img.size[0] // 80))
            b_clusters = self._cluster_axis(family_b, threshold=max(8, img.size[1] // 80))
            a_best = self._best_line_window(a_clusters, 9)
            b_best = self._best_line_window(b_clusters, 9)
            if a_best is None or b_best is None:
                continue

            n_a = np.array([np.cos(np.deg2rad(theta_a + 90.0)), np.sin(np.deg2rad(theta_a + 90.0))], dtype=np.float32)
            n_b = np.array([np.cos(np.deg2rad(theta_b + 90.0)), np.sin(np.deg2rad(theta_b + 90.0))], dtype=np.float32)

            grid_points = []
            valid = True
            for rho_a in a_best["positions"]:
                row_pts = []
                for rho_b in b_best["positions"]:
                    pt = self._line_intersection(n_a, float(rho_a), n_b, float(rho_b))
                    if pt is None:
                        valid = False
                        break
                    row_pts.append(pt)
                if not valid:
                    break
                grid_points.append(row_pts)
            if not valid:
                continue

            corners = np.array([grid_points[0][0], grid_points[0][-1], grid_points[-1][-1], grid_points[-1][0]], dtype=np.float32)
            metrics = compute_quad_metrics(corners, img.size[0], img.size[1])
            geom_quality = 300.0 * max(0.0, metrics["area_ratio"]) + 220.0 * max(0.0, metrics["opposite_similarity"]) + 200.0 * max(0.0, metrics["aspect_similarity"])
            score = a_best["score"] + b_best["score"] + geom_quality + pair_strength
            candidate = {
                "tag": "lattice",
                "corners": order_corners(corners),
                "score": float(score),
                "regularity": (a_best["regularity"], b_best["regularity"]),
            }
            if best_result is None or candidate["score"] > best_result["score"]:
                best_result = candidate
        return best_result

    def lens_hypotheses(self, img):
        hypotheses = []

        gradient = self.detect_gradient_projection(img)
        if gradient is not None:
            grad_bonus = min(0.12, 0.08 * max(0.0, gradient.get("periodicity", 0.0)))
            hypotheses.append(self._lens_candidate(gradient["tag"], gradient["corners"], img, extra_conf=grad_bonus, raw_score=gradient.get("score")))

        panel = self.detect_panel_split(img)
        if panel is not None:
            panel_bonus = min(0.12, 0.08 * max(0.0, panel.get("periodicity", 0.0)))
            hypotheses.append(self._lens_candidate(panel["tag"], panel["corners"], img, extra_conf=panel_bonus, raw_score=panel.get("score")))

        lattice = self.detect_lattice(img)
        if lattice is not None:
            reg = lattice.get("regularity")
            reg_bonus = 0.0 if not reg else 0.04 * float(np.mean(reg))
            hypotheses.append(self._lens_candidate("lattice", lattice["corners"], img, extra_conf=reg_bonus, raw_score=lattice.get("score")))

        robust = find_board_corners(img)
        if robust is not None:
            hypotheses.append(self._lens_candidate("contour_robust", robust, img))

        legacy = find_board_corners_legacy(img)
        if legacy is not None:
            hypotheses.append(self._lens_candidate("contour_legacy", legacy, img))

        hypotheses.sort(
            key=lambda item: (
                int(item["trusted"]),
                int(item["relaxed"]),
                float(item["lens_confidence"]),
                float(item["warp_quality"]),
                float(item["metrics"]["area_ratio"]),
            ),
            reverse=True,
        )
        return hypotheses


def build_detector_candidates(img):
    detector = BoardDetector(debug=DEBUG_MODE)
    raw = [("full", img, 0.0, True)]
    if USE_EDGE_DETECTION:
        for result in detector.lens_hypotheses(img):
            if not result["trusted"] and not result["relaxed"]:
                continue
            tag = result["tag"] if result["trusted"] else f"{result['tag']}_relaxed"
            corners = result["corners"]
            warp_quality = result["warp_quality"]
            warp_trusted = result["trusted"]
            refined_corners, grid_fit_score = detector._refine_corners_grid_fit(img, corners)
            refined_metrics = compute_quad_metrics(refined_corners, img.size[0], img.size[1])
            refined_trusted = is_warp_geometry_trustworthy(refined_metrics)
            refined_relaxed = is_warp_geometry_relaxed(refined_metrics)
            if refined_trusted or refined_relaxed:
                refined_warp_q = max(
                    warp_geometry_quality(refined_metrics),
                    0.65 * warp_geometry_quality(refined_metrics) + 0.35 * grid_fit_score,
                )
                if grid_fit_score >= 0.28 and refined_warp_q >= warp_quality:
                    corners = refined_corners
                    warp_quality = refined_warp_q
                    warp_trusted = refined_trusted
                    tag = result["tag"] if refined_trusted else f"{result['tag']}_relaxed"
            warped = perspective_transform(img, corners)
            raw.append((tag, warped, warp_quality, warp_trusted))

    def candidate_family(tag):
        text = str(tag)
        if text.startswith("full"):
            return "full"
        if text.startswith("contour_"):
            return "contour"
        if text.startswith("lattice"):
            return "lattice"
        if text.startswith("gradient_projection"):
            return "gradient_projection"
        if text.startswith("panel_split_"):
            return "panel_split"
        return None

    def candidate_key(item):
        tag, _img, warp_quality, warp_trusted = item
        return (
            int(bool(warp_trusted)),
            float(warp_quality),
            int(not str(tag).endswith("_relaxed")),
            str(tag),
        )

    grouped = {family: [] for family in FAMILY_ORDER}
    for item in raw:
        family = candidate_family(item[0])
        if family is None:
            continue
        grouped[family].append(item)

    selected = []
    for family in FAMILY_ORDER:
        ranked = sorted(grouped[family], key=candidate_key, reverse=True)
        selected.extend(ranked[: FAMILY_BUDGETS[family]])
    return selected


def find_inner_board_window(img):
    arr = np.array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    h, w = gray.shape
    if min(h, w) < 320:
        return None

    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    col_energy = gx.mean(axis=0)
    row_energy = gy.mean(axis=1)
    if w >= 31:
        col_energy = cv2.GaussianBlur(col_energy.reshape(1, -1), (1, 31), 0).reshape(-1)
    if h >= 31:
        row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (31, 1), 0).reshape(-1)

    def peak_pair(signal, left_band_frac, right_band_frac):
        n = int(signal.shape[0])
        left_hi = max(8, int(n * left_band_frac))
        right_lo = min(n - 8, int(n * right_band_frac))
        if left_hi >= right_lo:
            return None
        left_idx = int(np.argmax(signal[:left_hi]))
        right_idx = int(np.argmax(signal[right_lo:])) + right_lo
        if right_idx <= left_idx:
            return None
        mean_signal = float(np.mean(signal) + 1e-6)
        gain = float((signal[left_idx] + signal[right_idx]) * 0.5 / mean_signal)
        return left_idx, right_idx, gain

    x_pair = peak_pair(col_energy, left_band_frac=0.35, right_band_frac=0.65)
    y_pair = peak_pair(row_energy, left_band_frac=0.35, right_band_frac=0.65)
    if x_pair is None or y_pair is None:
        return None

    left, right, x_gain = x_pair
    top, bottom, y_gain = y_pair
    pad_x = max(4, int(w * 0.015))
    pad_y = max(4, int(h * 0.015))
    x0 = max(0, left)
    x1 = min(w - 1, right + pad_x)
    y0 = max(0, top)
    y1 = min(h - 1, bottom + pad_y)
    ww = x1 - x0 + 1
    hh = y1 - y0 + 1
    width_ratio = ww / float(w)
    height_ratio = hh / float(h)
    ratio = ww / float(max(1, hh))
    if (
        width_ratio < 0.55
        or height_ratio < 0.55
        or width_ratio > 0.98
        or height_ratio > 0.98
        or ratio < 0.74
        or ratio > 1.26
        or min(x_gain, y_gain) < 2.2
    ):
        return None

    cropped = img.crop((x0, y0, x1 + 1, y1 + 1)).resize((w, h), Image.LANCZOS)
    return cropped


def collect_orientation_context(candidates, board_perspective):
    context = {
        "label_perspective_result": None,
        "label_details": {"left": None, "right": None},
        "labels_absent": True,
        "labels_same": False,
    }
    if board_perspective != "auto":
        return context

    full_candidate_img = trim_dark_edge_bars(candidates[0][1].copy())
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
    context.update({
        "label_perspective_result": label_perspective_result,
        "label_details": label_details,
        "labels_absent": labels_absent,
        "labels_same": labels_same,
    })
    return context


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
        fallback = infer_board_perspective_from_piece_distribution(fen, threshold=ORIENTATION_STRONG_PIECE_MARGIN)
        if fallback == "black":
            return "black", "piece_distribution_fallback"
        return "white", "piece_distribution_fallback"

    return "white", "default"


def king_health(fen_board):
    rows = expand_fen_board(fen_board)
    if len(rows) != 8 or any(len(r) != 8 for r in rows):
        return 0
    return int(fen_board.count("K") == 1) + int(fen_board.count("k") == 1)


def decode_candidate(candidate, model, device, board_perspective, orientation_context):
    tag, candidate_img, warp_quality, warp_trusted = candidate
    candidate_img = trim_dark_edge_bars(candidate_img)

    def decode_variant(variant_img):
        fen, conf, piece_count, details = infer_fen_on_image_clean(
            variant_img,
            model,
            device,
            USE_SQUARE_DETECTION,
            return_details=True,
        )
        sat_stats = image_saturation_stats(variant_img)
        if sat_stats["sat_mean"] <= LOW_SAT_SPARSE_SAT_MEAN_MAX and piece_count <= LOW_SAT_SPARSE_PIECE_MAX:
            rescored = rescore_low_saturation_sparse_from_topk(details.get("tile_infos", []), base_fen=fen, base_conf=conf)
            if rescored is not None:
                fen, conf, piece_count = rescored

        missing_king = repair_missing_king_from_topk(details.get("tile_infos", []), base_fen=fen, base_conf=conf)
        if missing_king is not None:
            fen, conf, piece_count = missing_king

        sparse_piece_repair = promote_sparse_pieces_from_topk(details.get("tile_infos", []), base_fen=fen, base_conf=conf)
        if sparse_piece_repair is not None:
            fen, conf, piece_count = sparse_piece_repair

        return variant_img, fen, conf, piece_count

    decoded_variants = [decode_variant(candidate_img)]
    if str(tag).startswith("contour_"):
        inner_board = find_inner_board_window(candidate_img)
        if inner_board is not None:
            decoded_variants.append(decode_variant(inner_board))

    candidate_img, fen, conf, piece_count = max(
        decoded_variants,
        key=lambda row: (
            board_plausibility_score(row[1]),
            king_health(row[1]),
            row[3] if row[3] <= LOW_SAT_SPARSE_PIECE_MAX else 0,
            row[2],
        ),
    )

    detected_perspective, perspective_source = resolve_candidate_orientation(
        fen,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    final_fen = rotate_fen_180(fen) if detected_perspective == "black" else fen
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
    )


def select_best_candidate(scored):
    best_raw_conf = max(float(item[3]) for item in scored) if scored else 0.0
    enriched = []
    for item in scored:
        fen_board = item[2]
        plausibility = board_plausibility_score(fen_board)
        k_health = king_health(fen_board)
        piece_count = int(item[4])
        geom_q = float(item[7])
        _, stm_src = infer_side_to_move_from_checks(fen_board)
        has_double_check_conflict = stm_src == "default_double_check_conflict"
        conf_adj = float(item[3])
        stats = image_saturation_stats(item[1])
        low_sat_sparse = stats["sat_mean"] <= LOW_SAT_SPARSE_SAT_MEAN_MAX and piece_count <= LOW_SAT_SPARSE_PIECE_MAX
        edge_rook_bonus = _edge_rook_count_from_labels(list("".join(expand_fen_board(fen_board)))) if low_sat_sparse else 0
        if "gradient_projection" in str(item[0]):
            conf_adj -= 0.010
        if "panel_split" in str(item[0]):
            conf_adj -= 0.030
        sparse_bonus_enabled = low_sat_sparse and float(item[3]) >= (best_raw_conf - 0.025)
        edge_rook_bonus = edge_rook_bonus if sparse_bonus_enabled else 0
        sparse_piece_bonus = piece_count if sparse_bonus_enabled else 0
        enriched.append((item, plausibility, k_health, edge_rook_bonus, sparse_piece_bonus, has_double_check_conflict, geom_q, conf_adj))

    return max(enriched, key=lambda pair: (pair[1], pair[2], pair[3], pair[4], -int(pair[5]), pair[7], pair[6]))[0]


def predict_board(image_path, model_path=None, board_perspective="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_path = model_path or MODEL_PATH
    if not resolved_model_path or not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            "Model checkpoint not found. "
            f"Tried: {resolved_model_path}. "
            "Set CHESSBOT_MODEL_PATH or pass --model-path explicitly."
        )

    model = StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(resolved_model_path, map_location=device))
    model.eval()

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
    return best[2], best[3]


def main():
    parser = argparse.ArgumentParser(description="Recognize chess position from image (v6 clean)")
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
    parser.add_argument("--debug", action="store_true", help="Verbose debugging")
    args = parser.parse_args()

    global USE_EDGE_DETECTION, USE_SQUARE_DETECTION, DEBUG_MODE
    USE_EDGE_DETECTION = not args.no_edge_detection
    USE_SQUARE_DETECTION = not args.no_square_detection
    DEBUG_MODE = args.debug

    try:
        fen, conf = predict_board(args.image, model_path=args.model_path, board_perspective=args.board_perspective)
        if args.side_to_move is None:
            side_to_move, side_source = infer_side_to_move_from_checks(fen)
        else:
            side_to_move = parse_side_to_move_override(args.side_to_move)
            side_source = "override_cli"
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


if __name__ == "__main__":
    main()
