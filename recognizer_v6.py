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

IMG_SIZE, FEN_CHARS = 64, "1PNBRQKpnbrqk"

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# User-facing runtime settings
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(THIS_DIR, "models", "model_hybrid_v5_latest_best.pt"))
MODEL_PATH = os.path.abspath(os.environ.get("CHESSBOT_MODEL_PATH", DEFAULT_MODEL_PATH))
USE_EDGE_DETECTION = True
USE_SQUARE_DETECTION = True
DEBUG_MODE = False

# Internal algorithm thresholds (advanced; usually do not change)
CANNY_LOW = 50
CANNY_HIGH = 150
CONTOUR_EPSILON = 0.02
WARP_MIN_AREA_RATIO = 0.30
WARP_MIN_OPPOSITE_SIMILARITY = 0.82
WARP_MIN_ASPECT_SIMILARITY = 0.50
WARP_MIN_ANGLE_DEG = 50.0
WARP_MAX_ANGLE_DEG = 130.0
# Relaxed acceptance band (v6): keep borderline warps as candidates and let
# downstream plausibility decide, instead of hard-dropping them.
WARP_RELAXED_MIN_AREA_RATIO = 0.16
WARP_RELAXED_MIN_OPPOSITE_SIMILARITY = 0.55
WARP_RELAXED_MIN_ASPECT_SIMILARITY = 0.18
WARP_RELAXED_MIN_ANGLE_DEG = 35.0
WARP_RELAXED_MAX_ANGLE_DEG = 145.0
WARP_PIECE_COVERAGE_RATIO = 0.45
WARP_PIECE_COVERAGE_MIN_FULL = 8
FULL_CONF_FOR_COVERAGE_GUARD = 0.95
TILE_CONTEXT_PAD = 2
ORIENTATION_STRONG_PIECE_MARGIN = 2.0
ORIENTATION_BEST_GUESS_MIN_MARGIN = 0.6
ORIENTATION_BEST_GUESS_MIN_PIECES = 10
ORIENTATION_BEST_GUESS_MIN_PAWNS_PER_COLOR = 1
# Keep disabled by default: this heuristic is useful for diagnostics but can
# flip otherwise-correct auto orientation on real social screenshots.
ORIENTATION_BEST_GUESS_ENABLED = False
ORIENTATION_WEAK_LABEL_MIN_CONF = 0.70
PIECE_LOG_PRIOR = -0.20
MAX_DECODE_CANDIDATES = 12
LOW_SAT_SPARSE_SAT_MEAN_MAX = 44.0
LOW_SAT_SPARSE_PIECE_MAX = 10
LOW_SAT_SPARSE_EMPTY_ALT_MIN = 0.015
PANEL_DIRECTIONAL_TRIM_FRAC = 0.08
LOW_SAT_ENHANCE_SAT_MAX = 52.0
LOW_SAT_ENHANCE_VAL_STD_MAX = 62.0
LOW_SAT_ENHANCE_CLAHE_CLIP = 2.2
LOW_SAT_ENHANCE_UNSHARP = 0.35
LOW_SAT_SPARSE_ALT_OPTIONS = 2
LOW_SAT_EDGE_ROOK_OBJECTIVE_BONUS = 0.40
GRADIENT_PROJ_MIN_AREA_RATIO = 0.16
GRADIENT_PROJ_MIN_SIDE_RATIO = 0.24
GRADIENT_PROJ_MAX_SIDE_RATIO = 0.94


class StandaloneBeastClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(StandaloneBeastClassifier, self).__init__()
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


def order_corners(corners):
    """Order corners as: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]
    rect[2] = corners[np.argmax(s)]
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]
    rect[3] = corners[np.argmax(diff)]
    return rect


def find_board_corners(img):
    """Find chessboard corners with multi-pass edge search and geometric scoring."""
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

        # Favor board-like quadrilaterals; keep margin only as a weak tie-breaker.
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

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_corners = candidates[0][1]
        if DEBUG_MODE:
            print(f"DEBUG: Selected board corners score={candidates[0][0]:.3f}", file=sys.stderr)
        return best_corners

    return None


def find_board_corners_legacy(img):
    """Original corner detector kept as fallback candidate."""
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


def perspective_transform(img, corners):
    """Deskew board to a square image."""
    width = height = 512
    dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(np.array(img), matrix, (width, height))
    return Image.fromarray(warped)


def compute_quad_metrics(corners, width, height):
    """Compute geometric quality metrics for a detected board quadrilateral."""
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
    # Normalized blend used only as a tie-breaker after board plausibility.
    area = max(0.0, min(1.0, metrics["area_ratio"] / max(WARP_MIN_AREA_RATIO, 1e-6)))
    opp = max(0.0, min(1.0, metrics["opposite_similarity"] / max(WARP_MIN_OPPOSITE_SIMILARITY, 1e-6)))
    asp = max(0.0, min(1.0, metrics["aspect_similarity"] / max(WARP_MIN_ASPECT_SIMILARITY, 1e-6)))
    angle_span = max(0.0, min(1.0, (metrics["max_angle"] - metrics["min_angle"]) / 180.0))
    angle = 1.0 - angle_span
    return float(0.35 * area + 0.30 * opp + 0.25 * asp + 0.10 * angle)


def detect_grid_lines(img):
    """Detect chessboard grid lines to infer exact tile boundaries."""
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

    def cluster_lines(lines, threshold=10):
        lines = sorted(lines)
        clusters = []
        current = [lines[0]]
        for line in lines[1:]:
            if line - current[-1] < threshold:
                current.append(line)
            else:
                clusters.append(int(np.mean(current)))
                current = [line]
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
    return cell_crop.crop(
        (
            int(cell * 0.74),
            int(cell * 0.72),
            int(cell * 0.98),
            int(cell * 0.98),
        )
    )


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
    # Conservative: only auto-orient when both side labels are present,
    # high-confidence, and mutually consistent.
    if (
        left_label
        and right_label
        and left_label["confidence"] >= 0.72
        and right_label["confidence"] >= 0.72
    ):
        if left_label["label"] == "a" and right_label["label"] == "h":
            return {
                "perspective": "white",
                "source": "board_labels",
                "details": details,
            }
        if left_label["label"] == "h" and right_label["label"] == "a":
            return {
                "perspective": "black",
                "source": "board_labels",
                "details": details,
            }
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
    # If black pieces sit significantly lower in image coordinates,
    # the screenshot is likely black POV and should be rotated.
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


def _orientation_features(oriented_fen):
    rows = expand_fen_board(oriented_fen)
    white_rows = []
    black_rows = []
    white_pawn_rows = []
    black_pawn_rows = []
    white_king = None
    black_king = None

    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "1":
                continue
            if ch.isupper():
                white_rows.append(r)
            else:
                black_rows.append(r)
            if ch == "P":
                white_pawn_rows.append(r)
            elif ch == "p":
                black_pawn_rows.append(r)
            elif ch == "K":
                white_king = (r, c)
            elif ch == "k":
                black_king = (r, c)

    piece_delta = 0.0
    if white_rows and black_rows:
        piece_delta = (sum(white_rows) / len(white_rows)) - (sum(black_rows) / len(black_rows))

    pawn_delta = 0.0
    if white_pawn_rows and black_pawn_rows:
        pawn_delta = (sum(white_pawn_rows) / len(white_pawn_rows)) - (
            sum(black_pawn_rows) / len(black_pawn_rows)
        )

    king_delta = 0.0
    if white_king is not None and black_king is not None:
        king_delta = float(white_king[0] - black_king[0])

    wk_in_check = False
    bk_in_check = False
    if white_king is not None:
        wk_in_check = is_square_attacked(rows, white_king[0], white_king[1], by_white=False)
    if black_king is not None:
        bk_in_check = is_square_attacked(rows, black_king[0], black_king[1], by_white=True)

    # Lightweight legality-style signals: both kings in check is heavily suspicious;
    # exactly one checked king is informative for orientation in tactical screenshots.
    check_score = -1.5 if (wk_in_check and bk_in_check) else (0.25 if (wk_in_check or bk_in_check) else 0.0)

    # Encourage both kings being present once.
    king_count_score = 0.0
    wk_count = oriented_fen.count("K")
    bk_count = oriented_fen.count("k")
    if wk_count != 1 or bk_count != 1:
        king_count_score = -2.0

    # Scores are clipped to keep this as a last-resort prior, not an override hammer.
    piece_term = max(-3.0, min(3.0, piece_delta))
    pawn_term = max(-3.0, min(3.0, pawn_delta))
    king_term = max(-3.0, min(3.0, king_delta))

    score = 0.45 * piece_term + 0.35 * pawn_term + 0.20 * king_term + check_score + king_count_score
    return {
        "score": float(score),
        "white_piece_count": int(len(white_rows)),
        "black_piece_count": int(len(black_rows)),
        "total_piece_count": int(len(white_rows) + len(black_rows)),
        "white_pawn_count": int(len(white_pawn_rows)),
        "black_pawn_count": int(len(black_pawn_rows)),
        "piece_delta": float(piece_delta),
        "pawn_delta": float(pawn_delta),
        "king_delta": float(king_delta),
        "wk_in_check": bool(wk_in_check),
        "bk_in_check": bool(bk_in_check),
        "wk_count": int(wk_count),
        "bk_count": int(bk_count),
    }


def orientation_best_guess(fen_board, min_margin=ORIENTATION_BEST_GUESS_MIN_MARGIN):
    white_features = _orientation_features(fen_board)
    black_board = rotate_fen_180(fen_board)
    black_features = _orientation_features(black_board)

    total_pieces = int(white_features.get("total_piece_count", 0))
    white_pawns = int(white_features.get("white_pawn_count", 0))
    black_pawns = int(white_features.get("black_pawn_count", 0))
    if total_pieces < ORIENTATION_BEST_GUESS_MIN_PIECES:
        return None, {
            "reason": "insufficient_piece_signal",
            "total_piece_count": total_pieces,
            "min_piece_count": int(ORIENTATION_BEST_GUESS_MIN_PIECES),
            "white": white_features,
            "black": black_features,
        }
    if min(white_pawns, black_pawns) < ORIENTATION_BEST_GUESS_MIN_PAWNS_PER_COLOR:
        return None, {
            "reason": "insufficient_pawn_signal",
            "white_pawn_count": white_pawns,
            "black_pawn_count": black_pawns,
            "min_pawns_per_color": int(ORIENTATION_BEST_GUESS_MIN_PAWNS_PER_COLOR),
            "white": white_features,
            "black": black_features,
        }

    white_score = float(white_features["score"])
    black_score = float(black_features["score"])
    margin = abs(white_score - black_score)
    if margin < float(min_margin):
        return None, {
            "margin": margin,
            "white": white_features,
            "black": black_features,
            "min_margin": float(min_margin),
        }

    perspective = "white" if white_score >= black_score else "black"
    return perspective, {
        "margin": margin,
        "white": white_features,
        "black": black_features,
        "min_margin": float(min_margin),
    }


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


def build_fen_from_tile_infos(tile_infos):
    rows = []
    for row_idx in range(8):
        row = "".join(tile_infos[row_idx * 8 + col_idx]["label"] for col_idx in range(8))
        rows.append(row)
    fen = "/".join(rows)
    return "/".join(re.sub(r"1+", lambda m: str(len(m.group())), row) for row in fen.split("/"))


def repair_duplicate_kings(tile_infos):
    repaired = [dict(tile, topk=list(tile["topk"])) for tile in tile_infos]

    for king_label in ("K", "k"):
        king_tiles = [tile for tile in repaired if tile["label"] == king_label]
        if len(king_tiles) <= 1:
            continue

        king_tiles.sort(key=lambda tile: tile["prob"], reverse=True)
        for extra_tile in king_tiles[1:]:
            replacement_label = "1"
            replacement_prob = extra_tile["prob"]
            found_empty = False
            for alt_label, alt_prob in extra_tile["topk"]:
                if alt_label == "1":
                    replacement_label = alt_label
                    replacement_prob = alt_prob
                    found_empty = True
                    break
            if not found_empty:
                for alt_label, alt_prob in extra_tile["topk"]:
                    if alt_label == king_label:
                        continue
                    replacement_label = alt_label
                    replacement_prob = alt_prob
                    break
            extra_tile["label"] = replacement_label
            extra_tile["prob"] = replacement_prob

    white_kings = sum(1 for tile in repaired if tile["label"] == "K")
    black_kings = sum(1 for tile in repaired if tile["label"] == "k")
    if white_kings != 1 or black_kings != 1:
        return None

    piece_count = sum(1 for tile in repaired if tile["label"] != "1")
    mean_conf = float(np.mean([tile["prob"] for tile in repaired])) if repaired else 0.0
    return build_fen_from_tile_infos(repaired), mean_conf, piece_count


def infer_fen_on_image(
    img,
    model,
    device,
    use_square_detection,
    board_perspective="white",
):
    """Run tile inference on a prepared board image and return fen stats."""
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

    fen, confs = [], []
    tile_infos = []
    piece_count = 0
    with torch.no_grad():
        for r in range(8):
            row = ""
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
                tensor = torch.from_numpy(img_np).float().to(device)
                tensor = (tensor / 127.5) - 1.0
                tensor = tensor.unsqueeze(0)

                out = torch.softmax(model(tensor), dim=1)
                topk_probs, topk_pred = torch.topk(out[0], k=min(3, len(FEN_CHARS)))
                topk = [(FEN_CHARS[int(idx.item())], float(prob.item())) for prob, idx in zip(topk_probs, topk_pred)]
                prob, pred = torch.max(out, 1)
                if prob.item() < 0.35:
                    label = "1"
                else:
                    label = FEN_CHARS[pred.item()]
                row += label
                if label != "1":
                    piece_count += 1
                confs.append(prob.item())
                tile_infos.append(
                    {
                        "row": r,
                        "col": c,
                        "label": label,
                        "prob": float(prob.item()),
                        "topk": topk,
                    }
                )
            fen.append(row)

    result = "/".join(fen)
    result = "/".join([re.sub(r"1+", lambda m: str(len(m.group())), row) for row in result.split("/")])
    repaired = repair_duplicate_kings(tile_infos)
    if repaired is not None:
        repaired_fen, repaired_conf, repaired_piece_count = repaired
        if DEBUG_MODE and repaired_fen != result:
            print(f"DEBUG: repaired duplicate kings {result} -> {repaired_fen}", file=sys.stderr)
        result = repaired_fen
        return result, repaired_conf, repaired_piece_count

    return result, float(np.mean(confs)), piece_count


def inset_board(img, px):
    w, h = img.size
    if w <= 2 * px or h <= 2 * px:
        return img
    return img.crop((px, px, w - px, h - px)).resize((w, h), Image.LANCZOS)


def trim_dark_edge_bars(img):
    """Trim dark screenshot bars on edges (top/bottom/left/right) before 8x8 slicing."""
    arr = np.array(img)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

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

    # Ignore tiny/noisy trims.
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

    if DEBUG_MODE:
        print(
            f"DEBUG: trim_dark_edge_bars top={top} bottom={bottom} left={left} right={right}",
            file=sys.stderr,
        )

    cropped = img.crop((x0, y0, x1, y1))
    return cropped.resize((w, h), Image.LANCZOS)


# Expose local helpers/constants through the historical v4 alias used below.
v4 = sys.modules[__name__]


def debug_event(kind, **fields):
    if not DEBUG_MODE:
        return
    payload = {"kind": kind}
    payload.update(fields)
    print(f"DEBUG_JSON {json.dumps(payload, sort_keys=True)}", file=sys.stderr)


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
    """Heuristic structural score: prioritize board sanity over raw model confidence."""
    rows = v4.expand_fen_board(fen_board)
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

    if white_king == 1:
        score += 6.0
    else:
        score -= 20.0 * abs(white_king - 1)

    if black_king == 1:
        score += 6.0
    else:
        score -= 20.0 * abs(black_king - 1)

    if white_pawns <= 8:
        score += 1.5
    else:
        score -= 5.0 * (white_pawns - 8)

    if black_pawns <= 8:
        score += 1.5
    else:
        score -= 5.0 * (black_pawns - 8)

    if white_pieces <= 16:
        score += 1.0
    else:
        score -= 2.0 * (white_pieces - 16)

    if black_pieces <= 16:
        score += 1.0
    else:
        score -= 2.0 * (black_pieces - 16)

    if white_queens <= 2:
        score += 0.5
    else:
        score -= 3.0 * (white_queens - 2)

    if black_queens <= 2:
        score += 0.5
    else:
        score -= 3.0 * (black_queens - 2)

    if total_pieces <= 32:
        score += 1.0
    else:
        score -= 2.0 * (total_pieces - 32)

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


def enhance_low_saturation_image(img):
    """
    Global low-saturation enhancement for scan/book-like boards.
    Returns None when the image does not match the target profile.
    """
    stats = image_saturation_stats(img)
    if stats["sat_mean"] > LOW_SAT_ENHANCE_SAT_MAX:
        return None
    if stats["val_std"] > LOW_SAT_ENHANCE_VAL_STD_MAX:
        return None

    rgb = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=LOW_SAT_ENHANCE_CLAHE_CLIP, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)

    # Mild unsharp mask to strengthen thin rook/edge signals.
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharp = cv2.addWeighted(enhanced, 1.0 + LOW_SAT_ENHANCE_UNSHARP, blur, -LOW_SAT_ENHANCE_UNSHARP, 0)
    return Image.fromarray(sharp.astype(np.uint8))


def directional_trim_resize(img, direction, frac):
    if img is None:
        return None
    w, h = img.size
    if w <= 2 or h <= 2:
        return None
    frac = float(max(0.0, min(0.2, frac)))
    if frac <= 0.0:
        return img.copy()

    trim_x = int(round(w * frac))
    trim_y = int(round(h * frac))
    x0, y0, x1, y1 = 0, 0, w, h
    if direction == "top":
        y0 = min(h - 2, trim_y)
    elif direction == "bottom":
        y1 = max(2, h - trim_y)
    elif direction == "left":
        x0 = min(w - 2, trim_x)
    elif direction == "right":
        x1 = max(2, w - trim_x)
    else:
        return None

    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return img.crop((x0, y0, x1, y1)).resize((w, h), Image.LANCZOS)


def tile_confidence_summary(tile_infos):
    if not tile_infos:
        return {
            "tiles": 0,
            "mean_top1": 0.0,
            "p10_top1": 0.0,
            "mean_empty_prob": 0.0,
            "mean_piece_alt_prob": 0.0,
        }
    top1 = np.array([float(tile.get("top1_prob", tile.get("prob", 0.0))) for tile in tile_infos], dtype=np.float32)
    empty = np.array([float(tile.get("empty_prob", 0.0)) for tile in tile_infos], dtype=np.float32)
    piece_alt = np.array([float(tile.get("best_piece_alt_prob", 0.0)) for tile in tile_infos], dtype=np.float32)
    return {
        "tiles": len(tile_infos),
        "mean_top1": float(np.mean(top1)),
        "p10_top1": float(np.percentile(top1, 10)),
        "mean_empty_prob": float(np.mean(empty)),
        "mean_piece_alt_prob": float(np.mean(piece_alt)),
    }


def _label_prob_from_topk(tile, label):
    for alt_label, alt_prob in tile["topk"]:
        if alt_label == label:
            return float(alt_prob)
    return 1e-12


def _labels_to_fen(labels):
    rows = []
    for r in range(8):
        row = "".join(labels[r * 8 : (r + 1) * 8])
        rows.append(row)
    board = "/".join(rows)
    return "/".join(re.sub(r"1+", lambda m: str(len(m.group())), row) for row in board.split("/"))


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


def rescore_low_saturation_sparse_from_topk(tile_infos, base_fen, base_conf):
    """
    Global low-saturation sparse-board rescue.
    Uses only tile top-k probabilities + board plausibility; no image IDs/per-square constants.
    """
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

    # Beam over at most 2 promotions from empty->piece.
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
                + 0.25 * piece_count
                + LOW_SAT_EDGE_ROOK_OBJECTIVE_BONUS * edge_rook_count
                + 0.5 * (log_bonus / 8.0)
            )
            scored.append((objective, labels, changes, log_bonus))
        scored.sort(key=lambda row: row[0], reverse=True)
        beam = [(labels, changes, log_bonus) for _obj, labels, changes, log_bonus in scored[:24]]

    def promote_missing_edge_rook(labels):
        out = list(labels)

        def try_color(target_rook, opposite_rook, is_same_color):
            if sum(1 for x in out if x == target_rook) > 0:
                return
            if sum(1 for x in out if x == opposite_rook) == 0:
                return

            best_idx = None
            best_score = 0.0
            for idx, current in enumerate(out):
                if not is_same_color(current):
                    continue
                if current in {"K", "k", target_rook}:
                    continue
                row = idx // 8
                col = idx % 8
                if row not in (0, 7) and col not in (0, 7):
                    continue
                rook_prob = _label_prob_from_topk(tile_infos[idx], target_rook)
                if rook_prob < 0.03:
                    continue
                cur_prob = _label_prob_from_topk(tile_infos[idx], current)
                edge_bonus = 0.03 if (row in (0, 7)) else 0.0
                score = rook_prob - 0.10 * cur_prob + edge_bonus
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None and best_score > 0.0:
                out[best_idx] = target_rook

        try_color("r", "R", lambda ch: ch.islower())
        try_color("R", "r", lambda ch: ch.isupper())
        return out

    def eval_labels(labels, log_bonus):
        promoted_labels = promote_missing_edge_rook(labels)
        fen = _labels_to_fen(promoted_labels)
        plaus = board_plausibility_score(fen)
        k_health = _king_health_from_labels(promoted_labels)
        piece_count = sum(1 for x in promoted_labels if x != "1")
        edge_rook_count = _edge_rook_count_from_labels(promoted_labels)
        probs = np.array(
            [_label_prob_from_topk(tile_infos[i], promoted_labels[i]) for i in range(64)],
            dtype=np.float32,
        )
        conf = float(np.mean(probs))
        objective = (
            plaus
            + 1.5 * k_health
            + 0.25 * piece_count
            + LOW_SAT_EDGE_ROOK_OBJECTIVE_BONUS * edge_rook_count
            + 0.5 * (log_bonus / 8.0)
        )
        return objective, fen, conf, piece_count, k_health, plaus

    base_obj, _, _, _, base_kh, _base_plaus = eval_labels(base_labels, 0.0)
    best = (base_obj, base_fen, base_conf, base_piece_count, base_kh)
    for labels, _changes, log_bonus in beam:
        obj, fen, conf, piece_count, k_health, _plaus = eval_labels(labels, log_bonus)
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


def repair_missing_kings_from_topk(tile_infos):
    """
    If a king is missing, promote the highest-probability king alternative from tile top-k.
    Keeps opposite king fixed and only runs when a king count is zero.
    """
    repaired = [dict(tile, topk=list(tile["topk"])) for tile in tile_infos]

    def count(label):
        return sum(1 for tile in repaired if tile["label"] == label)

    for king_label in ("K", "k"):
        if count(king_label) > 0:
            continue

        best_idx = None
        best_score = -1.0
        best_prob = 0.0
        for idx, tile in enumerate(repaired):
            # Never overwrite the opposite king.
            if tile["label"] in {"K", "k"} and tile["label"] != king_label:
                continue

            alt_prob = None
            for alt_label, alt_p in tile["topk"]:
                if alt_label == king_label:
                    alt_prob = float(alt_p)
                    break
            if alt_prob is None:
                continue

            # Favor true king alternatives over already-confident current labels.
            score = alt_prob - 0.60 * float(tile["prob"])
            if score > best_score:
                best_score = score
                best_idx = idx
                best_prob = alt_prob

        if best_idx is not None:
            repaired[best_idx]["label"] = king_label
            repaired[best_idx]["prob"] = best_prob

    white_kings = sum(1 for tile in repaired if tile["label"] == "K")
    black_kings = sum(1 for tile in repaired if tile["label"] == "k")
    if white_kings != 1 or black_kings != 1:
        return None

    piece_count = sum(1 for tile in repaired if tile["label"] != "1")
    mean_conf = float(np.mean([tile["prob"] for tile in repaired])) if repaired else 0.0
    return build_fen_from_tile_infos(repaired), mean_conf, piece_count


def repair_sparse_missing_pawn(tile_infos):
    """
    Sparse-board structural repair:
    if one side has zero pawns and the other has >=2, promote one empty square to a
    pawn when pawn probability is strong enough.
    """
    repaired = [dict(tile, topk=list(tile["topk"])) for tile in tile_infos]
    labels = [tile["label"] for tile in repaired]
    wk = sum(1 for x in labels if x == "K")
    bk = sum(1 for x in labels if x == "k")
    if wk != 1 or bk != 1:
        return None

    piece_count = sum(1 for x in labels if x != "1")
    if piece_count > 8:
        return None

    white_pawns = sum(1 for x in labels if x == "P")
    black_pawns = sum(1 for x in labels if x == "p")
    if not ((white_pawns == 0 and black_pawns >= 2) or (black_pawns == 0 and white_pawns >= 2)):
        return None

    target = "P" if white_pawns == 0 else "p"
    best_idx = None
    best_score = -1e9
    best_prob = 0.0
    for idx, tile in enumerate(repaired):
        if tile["label"] != "1":
            continue
        pawn_prob = None
        for alt_label, alt_p in tile["topk"]:
            if alt_label == target:
                pawn_prob = float(alt_p)
                break
        if pawn_prob is None:
            continue
        empty_prob = float(tile["prob"])
        # Require meaningful pawn evidence while allowing empty to stay slightly higher.
        if pawn_prob < 0.14 or empty_prob > 0.44:
            continue
        score = pawn_prob - 0.45 * empty_prob
        if score > best_score:
            best_score = score
            best_idx = idx
            best_prob = pawn_prob

    if best_idx is None:
        return None

    repaired[best_idx]["label"] = target
    repaired[best_idx]["prob"] = best_prob

    piece_count = sum(1 for tile in repaired if tile["label"] != "1")
    mean_conf = float(np.mean([tile["prob"] for tile in repaired])) if repaired else 0.0
    return build_fen_from_tile_infos(repaired), mean_conf, piece_count


def infer_fen_on_image_deep_topk(
    img,
    model,
    device,
    use_square_detection,
    board_perspective="white",
    topk_k=8,
    return_details=False,
):
    """v5 tile inference with deeper per-tile top-k for cleaner structural repair."""
    if board_perspective not in {"white", "black"}:
        raise ValueError("board_perspective must be 'white' or 'black'")

    w, h = img.size
    if use_square_detection:
        grid = v4.detect_grid_lines(img)
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
    empty_idx = v4.FEN_CHARS.index("1")
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

            x0 = max(0, int(xe[image_c] - v4.TILE_CONTEXT_PAD))
            y0 = max(0, int(ye[image_r] - v4.TILE_CONTEXT_PAD))
            x1 = min(w, int(xe[image_c + 1] + v4.TILE_CONTEXT_PAD))
            y1 = min(h, int(ye[image_r + 1] + v4.TILE_CONTEXT_PAD))
            tile = img.crop((x0, y0, x1, y1)).resize((v4.IMG_SIZE, v4.IMG_SIZE), Image.LANCZOS)
            img_np = np.array(tile).transpose(2, 0, 1)
            tensor = (torch.from_numpy(img_np).float() / 127.5) - 1.0
            tile_tensors.append(tensor)
            tile_meta.append((r, c))

    with torch.no_grad():
        batch = torch.stack(tile_tensors, dim=0).to(device)
        out_batch = torch.softmax(model(batch), dim=1)

    for idx, (r, c) in enumerate(tile_meta):
        out = out_batch[idx]
        topk_probs, topk_pred = torch.topk(out, k=min(topk_k, len(v4.FEN_CHARS)))
        topk = [
            (v4.FEN_CHARS[int(k_idx.item())], float(prob.item()))
            for prob, k_idx in zip(topk_probs, topk_pred)
        ]

        # Global sparsity prior: empty squares are slightly favored unless
        # a piece probability is clearly stronger.
        adjusted_scores = torch.log(out + 1e-12) + PIECE_LOG_PRIOR
        adjusted_scores[empty_idx] = torch.log(out[empty_idx] + 1e-12)
        pred_idx = int(torch.argmax(adjusted_scores).item())
        pred_prob = float(out[pred_idx].item())

        if pred_prob < 0.35:
            label = "1"
        else:
            label = v4.FEN_CHARS[pred_idx]

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

    fen_rows = ["".join(labels[r * 8 : (r + 1) * 8]) for r in range(8)]

    result = "/".join(fen_rows)
    result = "/".join(
        [re.sub(r"1+", lambda m: str(len(m.group())), row) for row in result.split("/")]
    )
    repaired = v4.repair_duplicate_kings(tile_infos)
    final_fen = result
    final_conf = float(np.mean(confs))
    final_piece_count = piece_count
    if repaired is not None:
        repaired_fen, repaired_conf, repaired_piece_count = repaired
        # Only short-circuit when duplicate-king repair actually changed the board.
        if repaired_fen != result:
            final_fen, final_conf, final_piece_count = repaired_fen, repaired_conf, repaired_piece_count
            details = {
                "base_fen": result,
                "final_fen": final_fen,
                "tile_infos": tile_infos,
                "confidence_summary": tile_confidence_summary(tile_infos),
            }
            if return_details:
                return final_fen, final_conf, final_piece_count, details
            return final_fen, final_conf, final_piece_count
    repaired_missing = repair_missing_kings_from_topk(tile_infos)
    if repaired_missing is not None:
        repaired_fen, repaired_conf, repaired_piece_count = repaired_missing
        if repaired_fen != result:
            if DEBUG_MODE:
                print(f"DEBUG: repaired missing king {result} -> {repaired_fen}", file=sys.stderr)
            final_fen, final_conf, final_piece_count = repaired_fen, repaired_conf, repaired_piece_count
            details = {
                "base_fen": result,
                "final_fen": final_fen,
                "tile_infos": tile_infos,
                "confidence_summary": tile_confidence_summary(tile_infos),
            }
            if return_details:
                return final_fen, final_conf, final_piece_count, details
            return final_fen, final_conf, final_piece_count
    repaired_sparse_pawn = repair_sparse_missing_pawn(tile_infos)
    if repaired_sparse_pawn is not None:
        repaired_fen, repaired_conf, repaired_piece_count = repaired_sparse_pawn
        if DEBUG_MODE and repaired_fen != result:
            print(f"DEBUG: repaired sparse pawn {result} -> {repaired_fen}", file=sys.stderr)
        final_fen, final_conf, final_piece_count = repaired_fen, repaired_conf, repaired_piece_count
        details = {
            "base_fen": result,
            "final_fen": final_fen,
            "tile_infos": tile_infos,
            "confidence_summary": tile_confidence_summary(tile_infos),
        }
        if return_details:
            return final_fen, final_conf, final_piece_count, details
        return final_fen, final_conf, final_piece_count

    details = {
        "base_fen": result,
        "final_fen": final_fen,
        "tile_infos": tile_infos,
        "confidence_summary": tile_confidence_summary(tile_infos),
    }
    if return_details:
        return final_fen, final_conf, final_piece_count, details
    return final_fen, final_conf, final_piece_count


class BoardDetector:
    """Modular board detector that combines contour and lattice hypotheses."""

    def __init__(self, debug=False):
        self.debug = debug

    def _log(self, msg):
        if self.debug:
            print(f"DEBUG: {msg}", file=sys.stderr)

    def _gray(self, img):
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    def _clip01(self, v):
        return float(max(0.0, min(1.0, float(v))))

    def _lens_confidence(self, metrics, trusted, relaxed, extra=0.0):
        area = self._clip01(metrics["area_ratio"] / 0.70)
        opp = self._clip01(metrics["opposite_similarity"])
        asp = self._clip01(metrics["aspect_similarity"])
        min_angle = self._clip01(metrics["min_angle"] / 90.0)
        max_angle_penalty = self._clip01((180.0 - metrics["max_angle"]) / 90.0)
        conf = (
            0.35 * v4.warp_geometry_quality(metrics)
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
        iw, ih = img.size[0], img.size[1]
        metrics = v4.compute_quad_metrics(corners, iw, ih)
        trusted = v4.is_warp_geometry_trustworthy(metrics)
        relaxed = v4.is_warp_geometry_relaxed(metrics)
        candidate = {
            "tag": tag,
            "corners": corners,
            "metrics": metrics,
            "trusted": trusted,
            "relaxed": relaxed,
            "warp_quality": v4.warp_geometry_quality(metrics),
            "lens_confidence": self._lens_confidence(
                metrics,
                trusted=trusted,
                relaxed=relaxed,
                extra=extra_conf,
            ),
            "raw_score": None if raw_score is None else float(raw_score),
        }
        debug_event(
            "v6_geometry_candidate",
            tag=tag,
            trusted=bool(trusted),
            relaxed=bool(relaxed),
            area=round(float(metrics["area_ratio"]), 6),
            opp=round(float(metrics["opposite_similarity"]), 6),
            asp=round(float(metrics["aspect_similarity"]), 6),
            min_angle=round(float(metrics["min_angle"]), 4),
            max_angle=round(float(metrics["max_angle"]), 4),
            warp_quality=round(float(candidate["warp_quality"]), 6),
            lens_confidence=round(float(candidate["lens_confidence"]), 6),
            raw_score=None if raw_score is None else round(float(raw_score), 6),
        )
        return candidate

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
            # Canonical ordering to dedupe
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
            merged.append(
                {
                    "pos": float(np.average(positions, weights=np.maximum(weights, 1e-3))),
                    "weight": float(weights.sum()),
                    "count": len(cluster),
                }
            )
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

    def detect_axis_grid_windows(self, img, topk=8):
        gray = self._gray(img)
        h, w = gray.shape
        if min(h, w) < 200:
            return []

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.convertScaleAbs(0.5 * np.abs(gx) + 0.5 * np.abs(gy)).astype(np.float32)

        col_energy = grad.mean(axis=0)
        row_energy = grad.mean(axis=1)
        if w >= 31:
            col_energy = cv2.GaussianBlur(col_energy.reshape(1, -1), (1, 31), 0).reshape(-1)
        if h >= 31:
            row_energy = cv2.GaussianBlur(row_energy.reshape(-1, 1), (31, 1), 0).reshape(-1)

        dcol = np.abs(np.gradient(col_energy))
        drow = np.abs(np.gradient(row_energy))
        x_ids = np.argsort(dcol)[::-1]
        y_ids = np.argsort(drow)[::-1]

        x_cuts = [int(w * 0.5)]
        y_cuts = [int(h * 0.5)]
        for idx in x_ids:
            idx = int(idx)
            if idx < int(w * 0.12) or idx > int(w * 0.88):
                continue
            if any(abs(idx - p) < max(12, w // 40) for p in x_cuts):
                continue
            x_cuts.append(idx)
            if len(x_cuts) >= 7:
                break
        for idx in y_ids:
            idx = int(idx)
            if idx < int(h * 0.10) or idx > int(h * 0.90):
                continue
            if any(abs(idx - p) < max(10, h // 40) for p in y_cuts):
                continue
            y_cuts.append(idx)
            if len(y_cuts) >= 7:
                break

        x_cuts = sorted(set([0, w] + x_cuts))
        y_cuts = sorted(set([0, h] + y_cuts))
        candidates = []
        global_tex = float(np.mean(grad) + 1e-6)
        global_ctr = float(np.std(gray) + 1e-6)

        x_pairs = []
        for xa in x_cuts:
            for xb in x_cuts:
                if xb <= xa:
                    continue
                x_pairs.append((xa, xb))
        y_pairs = []
        for ya in y_cuts:
            for yb in y_cuts:
                if yb <= ya:
                    continue
                y_pairs.append((ya, yb))

        for xa, xb in x_pairs:
            for ya, yb in y_pairs:
                x0, x1 = int(xa), int(xb - 1)
                y0, y1 = int(ya), int(yb - 1)
                ww = x1 - x0 + 1
                hh = y1 - y0 + 1
                if ww < int(w * 0.24) or hh < int(h * 0.24):
                    continue
                ratio = ww / float(max(1, hh))
                if ratio < 0.52 or ratio > 1.95:
                    continue
                periodicity, texture_raw, contrast_raw = self._periodicity_2d_score(gray, grad, x0, y0, x1, y1)
                if periodicity <= 0.06:
                    continue

                texture = float(texture_raw / global_tex)
                contrast = float(contrast_raw / global_ctr)
                area_ratio = (ww * hh) / float(w * h)
                area_pref = 1.0 - min(1.0, abs(area_ratio - 0.42) / 0.42)
                square_pref = 1.0 - min(1.0, abs(1.0 - ratio) / 0.8)
                score = (
                    3.2 * periodicity
                    + 0.60 * texture
                    + 0.35 * contrast
                    + 0.30 * area_pref
                    + 0.30 * square_pref
                )
                candidates.append(
                    {
                        "tag": "axis_grid_window",
                        "corners": np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32),
                        "score": float(score),
                        "periodicity": float(periodicity),
                        "ratio": float(ratio),
                        "area_ratio": float(area_ratio),
                    }
                )

        if not candidates:
            return []
        candidates.sort(key=lambda c: c["score"], reverse=True)
        # Deduplicate heavily overlapping rectangles.
        chosen = []
        for cand in candidates:
            c = cand["corners"]
            x0, y0 = c[0]
            x1, y1 = c[2]
            keep = True
            for prev in chosen:
                p = prev["corners"]
                px0, py0 = p[0]
                px1, py1 = p[2]
                ix0, iy0 = max(x0, px0), max(y0, py0)
                ix1, iy1 = min(x1, px1), min(y1, py1)
                if ix1 >= ix0 and iy1 >= iy0:
                    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
                    a = (x1 - x0 + 1) * (y1 - y0 + 1)
                    b = (px1 - px0 + 1) * (py1 - py0 + 1)
                    iou = inter / float(max(1.0, a + b - inter))
                    if iou > 0.82:
                        keep = False
                        break
            if keep:
                chosen.append(cand)
            if len(chosen) >= topk:
                break
        if self.debug and chosen:
            best = chosen[0]
            self._log(
                f"axis_grid best score={best['score']:.3f} per={best['periodicity']:.3f} "
                f"ratio={best['ratio']:.3f} area={best['area_ratio']:.3f}"
            )
        return chosen

    def _projection_peak_pair(self, signal, min_span_ratio=0.24):
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
        if ri <= li:
            return None
        if (ri - li) < int(n * min_span_ratio):
            return None

        mean_signal = float(np.mean(arr) + 1e-6)
        peak_gain = float((arr[li] + arr[ri]) * 0.5 / mean_signal)
        return li, ri, peak_gain

    def detect_gradient_projection(self, img):
        """
        Projection-based board bounds (Sobel gradients) inspired by classic
        CV pipelines: detect strong paired edge responses on x/y projections.
        """
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

        x_pair = self._projection_peak_pair(col_energy, min_span_ratio=GRADIENT_PROJ_MIN_SIDE_RATIO)
        y_pair = self._projection_peak_pair(row_energy, min_span_ratio=GRADIENT_PROJ_MIN_SIDE_RATIO)
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
        if width_ratio < GRADIENT_PROJ_MIN_SIDE_RATIO or height_ratio < GRADIENT_PROJ_MIN_SIDE_RATIO:
            return None
        if width_ratio > GRADIENT_PROJ_MAX_SIDE_RATIO or height_ratio > GRADIENT_PROJ_MAX_SIDE_RATIO:
            return None

        ratio = ww / float(max(1, hh))
        if ratio < 0.52 or ratio > 1.95:
            return None

        area_ratio = (ww * hh) / float(w * h)
        if area_ratio < GRADIENT_PROJ_MIN_AREA_RATIO:
            return None

        periodicity, texture_raw, contrast_raw = self._periodicity_2d_score(gray, grad, x0, y0, x1, y1)
        if periodicity <= 0.03:
            return None

        edge_gain = 0.5 * (x_gain + y_gain)
        area_pref = 1.0 - min(1.0, abs(area_ratio - 0.42) / 0.42)
        square_pref = 1.0 - min(1.0, abs(1.0 - ratio) / 0.8)
        texture = float(texture_raw / (float(np.mean(grad)) + 1e-6))
        contrast = float(contrast_raw / (float(np.std(gray)) + 1e-6))
        score = (
            2.6 * periodicity
            + 0.70 * max(0.0, edge_gain - 1.0)
            + 0.40 * texture
            + 0.30 * contrast
            + 0.25 * area_pref
            + 0.25 * square_pref
        )

        candidate = {
            "tag": "gradient_projection",
            "corners": np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32),
            "score": float(score),
            "periodicity": float(periodicity),
            "edge_gain": float(edge_gain),
            "ratio": float(ratio),
            "area_ratio": float(area_ratio),
        }
        self._log(
            "gradient_projection "
            f"score={candidate['score']:.3f} per={candidate['periodicity']:.3f} "
            f"edge_gain={candidate['edge_gain']:.3f} ratio={candidate['ratio']:.3f} "
            f"area={candidate['area_ratio']:.3f}"
        )
        return candidate

    def _warp_grid_score(self, img, corners, size=256):
        arr = np.array(img)
        src = corners.astype(np.float32)
        dst = np.array(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype=np.float32,
        )
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
        cur = v4.order_corners(corners.copy().astype(np.float32))
        base_metrics = v4.compute_quad_metrics(cur, w, h)
        base_score = 0.70 * self._warp_grid_score(img, cur) + 0.30 * v4.warp_geometry_quality(base_metrics)
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
                        trial = v4.order_corners(trial)
                        metrics = v4.compute_quad_metrics(trial, w, h)
                        if metrics["area_ratio"] < 0.16:
                            continue
                        gscore = self._warp_grid_score(img, trial)
                        score = 0.70 * gscore + 0.30 * v4.warp_geometry_quality(metrics)
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
            if ww < int(w * 0.28) or hh < int(h * 0.34):
                return
            ratio = ww / float(max(1, hh))
            if ratio < 0.52 or ratio > 1.95:
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
            candidates.append(
                {
                    "tag": tag,
                    "corners": np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32),
                    "score": float(score),
                    "periodicity": float(periodicity),
                    "edge_gain": float(edge_gain),
                    "ratio": float(ratio),
                }
            )

        x_lo = int(w * 0.18)
        x_hi = int(w * 0.82)
        x_step = max(4, w // 160)
        for split in range(x_lo, x_hi, x_step):
            left_mean = float(np.mean(col_energy[:split])) if split > 8 else 0.0
            right_mean = float(np.mean(col_energy[split:])) if split < w - 8 else 0.0

            # Right panel
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

            # Left panel
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

            # Bottom panel
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

            # Top panel
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
        self._log(
            "panel_split "
            f"tag={best['tag']} score={best['score']:.3f} per={best['periodicity']:.3f} "
            f"edge_gain={best['edge_gain']:.3f} ratio={best['ratio']:.3f}"
        )
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

            n_a = np.array(
                [np.cos(np.deg2rad(theta_a + 90.0)), np.sin(np.deg2rad(theta_a + 90.0))],
                dtype=np.float32,
            )
            n_b = np.array(
                [np.cos(np.deg2rad(theta_b + 90.0)), np.sin(np.deg2rad(theta_b + 90.0))],
                dtype=np.float32,
            )

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

            corners = np.array(
                [
                    grid_points[0][0],
                    grid_points[0][-1],
                    grid_points[-1][-1],
                    grid_points[-1][0],
                ],
                dtype=np.float32,
            )
            metrics = v4.compute_quad_metrics(corners, img.size[0], img.size[1])
            geom_quality = (
                300.0 * max(0.0, metrics["area_ratio"])
                + 220.0 * max(0.0, metrics["opposite_similarity"])
                + 200.0 * max(0.0, metrics["aspect_similarity"])
            )
            score = a_best["score"] + b_best["score"] + geom_quality + pair_strength
            candidate = {
                "tag": "lattice",
                "corners": v4.order_corners(corners),
                "score": float(score),
                "grid": (a_best["positions"], b_best["positions"]),
                "metrics": metrics,
                "theta": (theta_a, theta_b),
                "regularity": (a_best["regularity"], b_best["regularity"]),
                "strength": (a_best["strength"], b_best["strength"]),
            }
            if best_result is None or candidate["score"] > best_result["score"]:
                best_result = candidate

        if best_result is None:
            return None

        self._log(
            "lattice "
            f"a_reg={best_result['regularity'][0]:.3f} b_reg={best_result['regularity'][1]:.3f} "
            f"a_strength={best_result['strength'][0]:.1f} b_strength={best_result['strength'][1]:.1f} "
            f"theta=({best_result['theta'][0]:.1f},{best_result['theta'][1]:.1f}) "
            f"area={best_result['metrics']['area_ratio']:.3f} "
            f"opp={best_result['metrics']['opposite_similarity']:.3f} "
            f"asp={best_result['metrics']['aspect_similarity']:.3f}"
        )
        return best_result

    def detect_contour(self, img):
        corners = v4.find_board_corners(img)
        if corners is None:
            return None
        return {"tag": "robust", "corners": corners}

    def lens_hypotheses(self, img):
        hypotheses = []

        for axis in self.detect_axis_grid_windows(img, topk=6):
            axis_bonus = min(0.18, 0.12 * max(0.0, axis.get("periodicity", 0.0)))
            hypotheses.append(
                self._lens_candidate(
                    axis["tag"],
                    axis["corners"],
                    img,
                    extra_conf=axis_bonus,
                    raw_score=axis.get("score"),
                )
            )

        gradient = self.detect_gradient_projection(img)
        if gradient is not None:
            grad_bonus = min(0.12, 0.08 * max(0.0, gradient.get("periodicity", 0.0)))
            hypotheses.append(
                self._lens_candidate(
                    gradient["tag"],
                    gradient["corners"],
                    img,
                    extra_conf=grad_bonus,
                    raw_score=gradient.get("score"),
                )
            )

        panel = self.detect_panel_split(img)
        if panel is not None:
            panel_bonus = min(0.12, 0.08 * max(0.0, panel.get("periodicity", 0.0)))
            hypotheses.append(
                self._lens_candidate(
                    panel["tag"],
                    panel["corners"],
                    img,
                    extra_conf=panel_bonus,
                    raw_score=panel.get("score"),
                )
            )

        lattice = self.detect_lattice(img)
        if lattice is not None:
            reg = lattice.get("regularity")
            reg_bonus = 0.0
            if reg:
                reg_bonus = 0.04 * float(np.mean(reg))
            hypotheses.append(
                self._lens_candidate(
                    "lattice",
                    lattice["corners"],
                    img,
                    extra_conf=reg_bonus,
                    raw_score=lattice.get("score"),
                )
            )

        robust = v4.find_board_corners(img)
        if robust is not None:
            hypotheses.append(self._lens_candidate("contour_robust", robust, img))

        legacy = v4.find_board_corners_legacy(img)
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

        if self.debug and hypotheses:
            best = hypotheses[0]
            self._log(
                "lens "
                f"best={best['tag']} conf={best['lens_confidence']:.3f} "
                f"trusted={best['trusted']} relaxed={best['relaxed']} "
                f"area={best['metrics']['area_ratio']:.3f} "
                f"opp={best['metrics']['opposite_similarity']:.3f} "
                f"asp={best['metrics']['aspect_similarity']:.3f}"
            )
        return hypotheses

    def candidate_images(self, img):
        candidates = [("full", img, 0.0, True)]
        for idx, result in enumerate(self.lens_hypotheses(img)):
            corners = result["corners"]
            metrics = result["metrics"]
            trusted = result["trusted"]
            relaxed = result["relaxed"]
            if not trusted and not relaxed:
                self._log(
                    f"reject {result['tag']} area={metrics['area_ratio']:.3f} "
                    f"opp={metrics['opposite_similarity']:.3f} asp={metrics['aspect_similarity']:.3f}"
                )
                continue
            quality = result["warp_quality"]
            warped = v4.perspective_transform(img, corners)
            tag = result["tag"] if trusted else f"{result['tag']}_relaxed"
            candidates.append((tag, warped, quality, trusted))
            candidates.append((f"{tag}_inset2", v4.inset_board(warped, 2), quality, trusted))

            # Grid-fit corner refinement on top hypotheses to lock full 8x8 extent.
            if idx < 4:
                refined_corners, grid_fit_score = self._refine_corners_grid_fit(img, corners)
                refined_metrics = v4.compute_quad_metrics(refined_corners, img.size[0], img.size[1])
                refined_trusted = v4.is_warp_geometry_trustworthy(refined_metrics)
                refined_relaxed = v4.is_warp_geometry_relaxed(refined_metrics)
                if refined_trusted or refined_relaxed:
                    refined_warp_q = max(
                        v4.warp_geometry_quality(refined_metrics),
                        0.65 * v4.warp_geometry_quality(refined_metrics) + 0.35 * grid_fit_score,
                    )
                    if grid_fit_score >= 0.28:
                        rtag_base = f"{tag}_gfit"
                        rwarped = v4.perspective_transform(img, refined_corners)
                        candidates.append((rtag_base, rwarped, refined_warp_q, refined_trusted))
                        candidates.append((f"{rtag_base}_inset2", v4.inset_board(rwarped, 2), refined_warp_q, refined_trusted))
                        self._log(
                            f"gfit {rtag_base} score={grid_fit_score:.3f} area={refined_metrics['area_ratio']:.3f} "
                            f"opp={refined_metrics['opposite_similarity']:.3f} asp={refined_metrics['aspect_similarity']:.3f}"
                        )
        return candidates


def build_detector_candidates(img):
    detector = BoardDetector(debug=DEBUG_MODE)
    base = detector.candidate_images(img) if USE_EDGE_DETECTION else [("full", img, 0.0, True)]

    # Optional global enhancement path for low-saturation scan/book images.
    enhanced_source = enhance_low_saturation_image(img)
    if enhanced_source is not None:
        debug_event("v6_low_sat_enhance", applied=True)
        if USE_EDGE_DETECTION:
            enh_candidates = detector.candidate_images(enhanced_source)
        else:
            enh_candidates = [("full", enhanced_source, 0.0, True)]
        enh_candidates = [
            (f"{tag}_enhsrc", candidate_img, warp_quality, warp_trusted)
            for (tag, candidate_img, warp_quality, warp_trusted) in enh_candidates
        ]
        # Keep original "full" first for deterministic orientation context.
        base = base + [("full_enhsrc", enhanced_source, 0.0, True)] + enh_candidates
    else:
        debug_event("v6_low_sat_enhance", applied=False)

    augmented = []
    for tag, candidate_img, warp_quality, warp_trusted in base:
        augmented.append((tag, candidate_img, warp_quality, warp_trusted))
        if isinstance(tag, str):
            tag_for_direction = tag[:-7] if tag.endswith("_enhsrc") else tag
        else:
            tag_for_direction = ""
        if tag_for_direction.startswith("panel_split_"):
            direction = tag_for_direction.split("_")[-1]
            trimmed = directional_trim_resize(candidate_img, direction, PANEL_DIRECTIONAL_TRIM_FRAC)
            if trimmed is not None:
                trim_pct = int(PANEL_DIRECTIONAL_TRIM_FRAC * 100)
                augmented.append((f"{tag}_trim{trim_pct}", trimmed, warp_quality, warp_trusted))

    if len(augmented) <= MAX_DECODE_CANDIDATES:
        debug_event(
            "v6_candidate_pool",
            count=len(augmented),
            tags=[str(item[0]) for item in augmented],
            capped=False,
            max_decode_candidates=MAX_DECODE_CANDIDATES,
        )
        return augmented

    full = [augmented[0]]
    others = sorted(
        augmented[1:],
        key=lambda item: (
            int(item[3]),  # trusted first
            float(item[2]),  # higher warp quality first
            0 if str(item[0]).endswith("_inset2") else 1,  # prefer non-inset on ties
        ),
        reverse=True,
    )
    capped = full + others[: max(0, MAX_DECODE_CANDIDATES - 1)]
    debug_event(
        "v6_candidate_cap",
        original_count=len(full) + len(others),
        capped_count=len(capped),
        max_decode_candidates=MAX_DECODE_CANDIDATES,
    )
    debug_event(
        "v6_candidate_pool",
        count=len(capped),
        tags=[str(item[0]) for item in capped],
        capped=True,
        max_decode_candidates=MAX_DECODE_CANDIDATES,
    )
    return capped


def collect_orientation_context(candidates, board_perspective):
    context = {
        "label_perspective_result": None,
        "label_details": {"left": None, "right": None},
        "labels_absent": True,
        "labels_same": False,
    }
    if board_perspective != "auto":
        return context

    full_candidate_img = v4.trim_dark_edge_bars(candidates[0][1].copy())
    label_perspective_result = v4.infer_board_perspective_from_labels(full_candidate_img)
    label_details = label_perspective_result["details"] if label_perspective_result else {
        "left": v4.classify_file_label_crop(v4.extract_file_label_crop(full_candidate_img, side="left")),
        "right": v4.classify_file_label_crop(v4.extract_file_label_crop(full_candidate_img, side="right")),
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
        }
    )
    return context


def resolve_candidate_orientation(fen, board_perspective, orientation_context):
    label_perspective_result = orientation_context["label_perspective_result"]
    label_details = orientation_context["label_details"]
    labels_absent = orientation_context["labels_absent"]
    labels_same = orientation_context["labels_same"]

    if board_perspective == "auto":
        if label_perspective_result is None:
            detected_perspective = "white"
            perspective_source = "default"
            weak_label_perspective = None
            left_label = label_details.get("left")
            right_label = label_details.get("right")
            if left_label is not None and right_label is not None:
                left_char = str(left_label.get("label", "")).lower()
                right_char = str(right_label.get("label", "")).lower()
                left_conf = float(left_label.get("confidence", 0.0))
                right_conf = float(right_label.get("confidence", 0.0))
                min_conf = min(left_conf, right_conf)
                if min_conf >= ORIENTATION_WEAK_LABEL_MIN_CONF:
                    if left_char == "a" and right_char == "h":
                        weak_label_perspective = "white"
                    elif left_char == "h" and right_char == "a":
                        weak_label_perspective = "black"
            if weak_label_perspective is not None:
                detected_perspective = weak_label_perspective
                perspective_source = "weak_label_fallback"
                debug_event(
                    "v6_orientation_fallback",
                    strategy="weak_label_fallback",
                    selected=detected_perspective,
                    min_conf=round(
                        min(
                            float(left_label.get("confidence", 0.0)),
                            float(right_label.get("confidence", 0.0)),
                        ),
                        4,
                    ),
                    labels_absent=bool(labels_absent),
                    labels_same=bool(labels_same),
                )
            piece_margin = orientation_piece_margin(fen)
            can_use_piece_distribution = labels_absent or labels_same
            if (
                weak_label_perspective is None
                and piece_margin >= ORIENTATION_STRONG_PIECE_MARGIN
                and can_use_piece_distribution
            ):
                fallback = v4.infer_board_perspective_from_piece_distribution(
                    fen,
                    threshold=ORIENTATION_STRONG_PIECE_MARGIN,
                )
                if fallback == "black":
                    detected_perspective = "black"
                perspective_source = "piece_distribution_fallback"
                debug_event(
                    "v6_orientation_fallback",
                    strategy="piece_distribution",
                    margin=round(float(piece_margin), 4),
                    selected=detected_perspective,
                    labels_absent=bool(labels_absent),
                    labels_same=bool(labels_same),
                )
            else:
                if weak_label_perspective is not None:
                    pass
                elif piece_margin >= ORIENTATION_STRONG_PIECE_MARGIN and not can_use_piece_distribution:
                    debug_event(
                        "v6_orientation_fallback",
                        strategy="skip_piece_distribution_due_label_signal",
                        margin=round(float(piece_margin), 4),
                        selected=detected_perspective,
                        labels_absent=bool(labels_absent),
                        labels_same=bool(labels_same),
                    )
                elif ORIENTATION_BEST_GUESS_ENABLED:
                    best_guess, guess_details = orientation_best_guess(fen)
                    if best_guess in {"white", "black"}:
                        detected_perspective = best_guess
                        perspective_source = "orientation_best_guess"
                    debug_event(
                        "v6_orientation_fallback",
                        strategy="orientation_best_guess",
                        piece_margin=round(float(piece_margin), 4),
                        selected=detected_perspective,
                        guessed=best_guess,
                        guess_details=guess_details,
                    )
                else:
                    debug_event(
                        "v6_orientation_fallback",
                        strategy="disabled_orientation_best_guess",
                        piece_margin=round(float(piece_margin), 4),
                        selected=detected_perspective,
                    )
        else:
            detected_perspective = label_perspective_result["perspective"]
            perspective_source = label_perspective_result["source"]
    elif board_perspective not in {"white", "black"}:
        raise ValueError("board_perspective must be 'auto', 'white', or 'black'")
    else:
        detected_perspective = board_perspective
        perspective_source = "override"

    return detected_perspective, perspective_source


def decode_candidate(candidate, model, device, board_perspective, orientation_context):
    tag, candidate_img, warp_quality, warp_trusted = candidate
    candidate_img = v4.trim_dark_edge_bars(candidate_img)
    fen, conf, piece_count, details = infer_fen_on_image_deep_topk(
        candidate_img,
        model,
        device,
        USE_SQUARE_DETECTION,
        return_details=True,
    )
    sat_stats = image_saturation_stats(candidate_img)
    low_sat_sparse = (
        sat_stats["sat_mean"] <= LOW_SAT_SPARSE_SAT_MEAN_MAX
        and piece_count <= LOW_SAT_SPARSE_PIECE_MAX
    )
    rescore_applied = False
    if low_sat_sparse:
        rescored = rescore_low_saturation_sparse_from_topk(
            details.get("tile_infos", []),
            base_fen=fen,
            base_conf=conf,
        )
        if rescored is not None:
            fen, conf, piece_count = rescored
            rescore_applied = True
            details["final_fen"] = fen

    detected_perspective, perspective_source = resolve_candidate_orientation(
        fen,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    final_fen = v4.rotate_fen_180(fen) if detected_perspective == "black" else fen
    plausibility = board_plausibility_score(final_fen)
    k_health = king_health(final_fen)
    _, stm_source = infer_side_to_move_from_checks(final_fen)
    conf_adj = float(conf) - (0.003 if str(tag).endswith("_inset2") else 0.0)
    if "gradient_projection" in str(tag):
        conf_adj -= 0.010
    decoded = (
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
    debug_event(
        "v6_candidate_decode",
        tag=tag,
        fen_board=final_fen,
        conf=round(float(conf), 6),
        conf_adj=round(float(conf_adj), 6),
        piece_count=int(piece_count),
        plausibility=round(float(plausibility), 4),
        king_health=int(k_health),
        stm_source=stm_source,
        perspective=detected_perspective,
        perspective_source=perspective_source,
        warp_quality=round(float(warp_quality), 6),
        warp_trusted=bool(warp_trusted),
        sat_mean=round(float(sat_stats["sat_mean"]), 4),
        sat_std=round(float(sat_stats["sat_std"]), 4),
        low_sat_sparse_mode=bool(low_sat_sparse),
        rescore_applied=bool(rescore_applied),
        confidence_summary=details.get("confidence_summary", {}),
    )
    return decoded


def king_health(fen_board):
    rows = v4.expand_fen_board(fen_board)
    if len(rows) != 8 or any(len(r) != 8 for r in rows):
        return 0
    wk = fen_board.count("K")
    bk = fen_board.count("k")
    return int(wk == 1) + int(bk == 1)


def select_best_candidate(scored):
    full_piece_count = next((item[4] for item in scored if item[0] == "full"), 0)
    full_conf = next((item[3] for item in scored if item[0] == "full"), 0.0)
    full_fen = next((item[2] for item in scored if item[0] == "full"), "")
    full_plaus = board_plausibility_score(full_fen) if full_fen else -1e9
    full_king = king_health(full_fen) if full_fen else 0

    full_strong = full_king == 2 and full_plaus >= 0.0 and full_conf >= 0.94
    if full_strong:
        filtered = [item for item in scored if "_relaxed" not in item[0]]
    else:
        filtered = []
        for item in scored:
            tag, _, _, _, piece_count, _, _, _, _ = item
            if (
                tag != "full"
                and full_conf >= v4.FULL_CONF_FOR_COVERAGE_GUARD
                and full_piece_count >= v4.WARP_PIECE_COVERAGE_MIN_FULL
                and piece_count <= int(full_piece_count * v4.WARP_PIECE_COVERAGE_RATIO)
            ):
                continue
            filtered.append(item)
    if not filtered:
        filtered = scored

    sparse_override = None
    if full_piece_count <= 8 and full_conf < 0.90:
        trusted_warps = [
            item
            for item in filtered
            if item[0] != "full"
            and item[8]
            and item[3] >= full_conf + 0.06
            and item[4] >= full_piece_count + 6
            and infer_side_to_move_from_checks(item[2])[1] != "default_double_check_conflict"
        ]
        if trusted_warps:
            trusted_warps.sort(
                key=lambda item: (
                    item[3],
                    item[7],
                    board_plausibility_score(item[2]),
                ),
                reverse=True,
            )
            cand = trusted_warps[0]
            cand_plaus = board_plausibility_score(cand[2])
            if cand_plaus >= full_plaus - 4.0:
                sparse_override = cand
                if DEBUG_MODE:
                    print(
                        f"DEBUG: sparse override full(conf={full_conf:.4f},pieces={full_piece_count}) "
                        f"-> {cand[0]}(conf={cand[3]:.4f},pieces={cand[4]})",
                        file=sys.stderr,
                    )

    enriched = []
    for item in filtered:
        plausibility = board_plausibility_score(item[2])
        k_health = king_health(item[2])
        geom_q = float(item[7])
        _, stm_src = infer_side_to_move_from_checks(item[2])
        has_double_check_conflict = stm_src == "default_double_check_conflict"
        conf_adj = float(item[3]) - (0.003 if str(item[0]).endswith("_inset2") else 0.0)
        if "gradient_projection" in str(item[0]):
            conf_adj -= 0.010
        enriched.append((item, plausibility, k_health, has_double_check_conflict, geom_q, conf_adj))
        if DEBUG_MODE:
            print(
                f"DEBUG: V6 Candidate={item[0]} plausibility={plausibility:.2f} "
                f"king_health={k_health} double_check_conflict={has_double_check_conflict} "
                f"warp_q={geom_q:.3f} conf={item[3]:.4f} conf_adj={conf_adj:.4f}",
                file=sys.stderr,
            )

    if sparse_override is not None:
        return sparse_override
    return max(enriched, key=lambda pair: (pair[1], pair[2], -int(pair[3]), pair[5], pair[4]))[0]


def predict_board(image_path, model_path=None, board_perspective="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_path = model_path or v4.MODEL_PATH
    if not resolved_model_path or not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            "Model checkpoint not found. "
            f"Tried: {resolved_model_path}. "
            "Set CHESSBOT_MODEL_PATH or pass --model-path explicitly."
        )

    model = v4.StandaloneBeastClassifier(num_classes=13).to(device)
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
    (
        best_tag,
        _,
        best_fen,
        best_conf,
        _,
        best_perspective,
        best_perspective_source,
        _,
        _,
    ) = select_best_candidate(scored)
    if DEBUG_MODE:
        print(
            f"DEBUG: V6 Selected candidate={best_tag} confidence={best_conf:.4f} "
            f"perspective={best_perspective} source={best_perspective_source} "
            f"model={resolved_model_path}",
            file=sys.stderr,
        )
    debug_event(
        "v6_selected_candidate",
        tag=best_tag,
        confidence=round(float(best_conf), 6),
        perspective=best_perspective,
        perspective_source=best_perspective_source,
        model=str(resolved_model_path),
    )
    return best_fen, best_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize chess position from image (v6 edge-first)")
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

    USE_EDGE_DETECTION = not args.no_edge_detection
    USE_SQUARE_DETECTION = not args.no_square_detection
    DEBUG_MODE = args.debug

    try:
        fen, conf = predict_board(
            args.image,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
        )
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
