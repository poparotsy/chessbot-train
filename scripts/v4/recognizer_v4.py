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
MODEL_FILENAMES = [
    "model_hybrid_v4_300e_last_best.pt",
    "model_hybrid_v4_300e_best.pt",
    "model_hybrid_v4_250e_final.pt",
    "model_hybrid_v4_final.pt",
    "model_hybrid_v4_150e.pt",
]
MODEL_SEARCH_DIRS = [
    os.path.join(THIS_DIR, "models"),
    os.path.join(THIS_DIR, "..", "models"),
    THIS_DIR,
    os.path.join(THIS_DIR, ".."),
]
MODEL_CANDIDATES = []
for model_dir in MODEL_SEARCH_DIRS:
    for filename in MODEL_FILENAMES:
        candidate = os.path.abspath(os.path.join(model_dir, filename))
        if candidate not in MODEL_CANDIDATES:
            MODEL_CANDIDATES.append(candidate)
MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.exists(path)), None)

# Edge detection parameters
CANNY_LOW = 50
CANNY_HIGH = 150
CONTOUR_EPSILON = 0.02
USE_EDGE_DETECTION = True
USE_SQUARE_DETECTION = False
DEBUG_MODE = False
WARP_MIN_AREA_RATIO = 0.30
WARP_MIN_OPPOSITE_SIMILARITY = 0.82
WARP_MIN_ASPECT_SIMILARITY = 0.50
WARP_MIN_ANGLE_DEG = 50.0
WARP_MAX_ANGLE_DEG = 130.0
WARP_PIECE_COVERAGE_RATIO = 0.45
WARP_PIECE_COVERAGE_MIN_FULL = 8
FULL_CONF_FOR_COVERAGE_GUARD = 0.95
TILE_CONTEXT_PAD = 2


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

        # Favor board-like quadrilaterals with a margin over full-frame rectangles.
        return area_ratio * 10.0 + opposite_similarity * 5.0 + aspect_similarity * 5.0 + margin * 20.0

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
            for eps in (0.01, 0.02, CONTOUR_EPSILON, 0.05, 0.08):
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


def predict_board(image_path, model_path=None, board_perspective="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_path = model_path or MODEL_PATH
    if not resolved_model_path:
        raise FileNotFoundError(
            "No default model found. Expected models/model_hybrid_v4_150e.pt. "
            "Use --model-path to provide an explicit checkpoint."
        )

    model = StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(resolved_model_path, map_location=device))
    model.eval()

    debug_dir = None
    if DEBUG_MODE:
        debug_dir = os.path.join(os.path.dirname(image_path), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.basename(image_path).rsplit(".", 1)[0]

    img = Image.open(image_path).convert("RGB")
    original_img = img.copy()
    candidates = [("full", img)]

    if USE_EDGE_DETECTION:
        iw, ih = original_img.size
        robust_corners = find_board_corners(original_img)
        if robust_corners is not None:
            robust_metrics = compute_quad_metrics(robust_corners, iw, ih)
            robust_ok = is_warp_geometry_trustworthy(robust_metrics)
            if DEBUG_MODE:
                print(
                    "DEBUG: robust metrics "
                    f"area={robust_metrics['area_ratio']:.3f} "
                    f"opp={robust_metrics['opposite_similarity']:.3f} "
                    f"asp={robust_metrics['aspect_similarity']:.3f} "
                    f"angle=[{robust_metrics['min_angle']:.1f},{robust_metrics['max_angle']:.1f}] "
                    f"trusted={robust_ok}",
                    file=sys.stderr,
                )
            if robust_ok:
                robust_img = perspective_transform(original_img, robust_corners)
                candidates.append(("robust", robust_img))
                candidates.append(("robust_inset2", inset_board(robust_img, 2)))

        legacy_corners = find_board_corners_legacy(original_img)
        if legacy_corners is not None:
            legacy_metrics = compute_quad_metrics(legacy_corners, iw, ih)
            legacy_ok = is_warp_geometry_trustworthy(legacy_metrics)
            if DEBUG_MODE:
                print(
                    "DEBUG: legacy metrics "
                    f"area={legacy_metrics['area_ratio']:.3f} "
                    f"opp={legacy_metrics['opposite_similarity']:.3f} "
                    f"asp={legacy_metrics['aspect_similarity']:.3f} "
                    f"angle=[{legacy_metrics['min_angle']:.1f},{legacy_metrics['max_angle']:.1f}] "
                    f"trusted={legacy_ok}",
                    file=sys.stderr,
                )
            if legacy_ok:
                legacy_img = perspective_transform(original_img, legacy_corners)
                candidates.append(("legacy", legacy_img))
                candidates.append(("legacy_inset2", inset_board(legacy_img, 2)))

    full_candidate_img = trim_dark_edge_bars(candidates[0][1].copy())
    label_perspective_result = None
    label_details = {"left": None, "right": None}
    labels_absent = True
    labels_same = False
    if board_perspective == "auto":
        label_perspective_result = infer_board_perspective_from_labels(full_candidate_img)
        label_details = label_perspective_result["details"] if label_perspective_result else {
            "left": classify_file_label_crop(extract_file_label_crop(full_candidate_img, side="left")),
            "right": classify_file_label_crop(extract_file_label_crop(full_candidate_img, side="right")),
        }
        labels_absent = label_details["left"] is None and label_details["right"] is None
        labels_same = (
            label_details["left"] is not None
            and label_details["right"] is not None
            and label_details["left"]["label"] == label_details["right"]["label"]
        )

    scored = []
    for tag, candidate_img in candidates:
        candidate_img = trim_dark_edge_bars(candidate_img)
        raw_fen, conf, piece_count = infer_fen_on_image(
            candidate_img,
            model,
            device,
            USE_SQUARE_DETECTION,
        )
        perspective_source = "override"
        if board_perspective == "auto":
            if label_perspective_result is None:
                detected_perspective = "white"
                perspective_source = "default"
                if labels_absent or labels_same:
                    fallback_perspective = infer_board_perspective_from_piece_distribution(raw_fen)
                    if fallback_perspective == "black":
                        detected_perspective = "black"
                        perspective_source = "piece_distribution_fallback"
            else:
                detected_perspective = label_perspective_result["perspective"]
                perspective_source = label_perspective_result["source"]
        elif board_perspective not in {"white", "black"}:
            raise ValueError("board_perspective must be 'auto', 'white', or 'black'")
        else:
            detected_perspective = board_perspective

        final_fen = rotate_fen_180(raw_fen) if detected_perspective == "black" else raw_fen
        scored.append((tag, candidate_img, final_fen, conf, piece_count, detected_perspective, perspective_source))
        if DEBUG_MODE:
            print(
                f"DEBUG: Candidate={tag} conf={conf:.4f} pieces={piece_count} "
                f"perspective={detected_perspective} source={perspective_source}",
                file=sys.stderr,
            )

    full_piece_count = next((item[4] for item in scored if item[0] == "full"), 0)
    full_conf = next((item[3] for item in scored if item[0] == "full"), 0.0)
    filtered = []
    for item in scored:
        tag, _, _, _, piece_count, _, _ = item
        if (
            tag != "full"
            and full_conf >= FULL_CONF_FOR_COVERAGE_GUARD
            and full_piece_count >= WARP_PIECE_COVERAGE_MIN_FULL
            and piece_count <= int(full_piece_count * WARP_PIECE_COVERAGE_RATIO)
        ):
            if DEBUG_MODE:
                print(
                    f"DEBUG: Rejecting candidate={tag} low_piece_coverage "
                    f"{piece_count}/{full_piece_count}",
                    file=sys.stderr,
                )
            continue
        filtered.append(item)

    if not filtered:
        filtered = scored

    best_tag, best_img, best_fen, best_conf, _, best_perspective, best_perspective_source = max(
        filtered, key=lambda item: item[3]
    )

    if DEBUG_MODE:
        print(f"DEBUG: Using model={resolved_model_path}", file=sys.stderr)
        print(f"DEBUG: Selected candidate={best_tag} confidence={best_conf:.4f}", file=sys.stderr)
        print(
            f"DEBUG: Selected perspective={best_perspective} source={best_perspective_source}",
            file=sys.stderr,
        )
        preprocessed_path = os.path.join(debug_dir, f"{base_name}_preprocessed.png")
        best_img.save(preprocessed_path)
        print(f"DEBUG: Selected board image saved to {preprocessed_path}", file=sys.stderr)

    return best_fen, best_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize chess position from image")
    parser.add_argument("image", help="Path to chess board image")
    parser.add_argument("--model-path", default=None, help="Override model path")
    parser.add_argument("--no-edge-detection", action="store_true", help="Disable edge detection")
    parser.add_argument("--detect-squares", action="store_true", help="Enable square grid detection")
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Infer perspective automatically, or force White/Black at the bottom",
    )
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    args = parser.parse_args()

    USE_EDGE_DETECTION = not args.no_edge_detection
    USE_SQUARE_DETECTION = args.detect_squares
    DEBUG_MODE = args.debug

    try:
        fen, conf = predict_board(
            args.image,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
        )
        result = {
            "success": True,
            "fen": f"{fen} w - - 0 1",
            "confidence": round(conf, 4),
        }
        if conf < 0.67:
            result["warning"] = "Low confidence - prediction may be incorrect"
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
