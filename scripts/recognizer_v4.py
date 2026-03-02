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

MODEL_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "models", "model_hybrid_v4_250e_final.pt"),
    os.path.join(os.path.dirname(__file__), "..", "models", "model_hybrid_v4_final.pt"),
    os.path.join(os.path.dirname(__file__), "..", "models", "model_hybrid_v4_150e.pt"),
    "models/model_hybrid_v4_250e_final.pt",
    "models/model_hybrid_v4_final.pt",
    "models/model_hybrid_v4_150e.pt",
]
MODEL_PATH = next((path for path in MODEL_CANDIDATES if os.path.exists(path)), None)

# Edge detection parameters
CANNY_LOW = 50
CANNY_HIGH = 150
CONTOUR_EPSILON = 0.02
USE_EDGE_DETECTION = True
USE_SQUARE_DETECTION = False
DEBUG_MODE = False
WARP_MIN_AREA_RATIO = 0.30
WARP_MIN_OPPOSITE_SIMILARITY = 0.50
WARP_MIN_ASPECT_SIMILARITY = 0.50
WARP_MIN_ANGLE_DEG = 50.0
WARP_MAX_ANGLE_DEG = 130.0
WARP_PIECE_COVERAGE_RATIO = 0.45
WARP_PIECE_COVERAGE_MIN_FULL = 8
FULL_CONF_FOR_COVERAGE_GUARD = 0.95


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


def infer_fen_on_image(img, model, device, use_square_detection):
    """Run tile inference on a prepared board image and return fen stats."""
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
    piece_count = 0
    with torch.no_grad():
        for r in range(8):
            row = ""
            for c in range(8):
                tile = img.crop((xe[c], ye[r], xe[c + 1], ye[r + 1])).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                img_np = np.array(tile).transpose(2, 0, 1)
                tensor = torch.from_numpy(img_np).float().to(device)
                tensor = (tensor / 127.5) - 1.0
                tensor = tensor.unsqueeze(0)

                out = torch.softmax(model(tensor), dim=1)
                prob, pred = torch.max(out, 1)
                if prob.item() < 0.35:
                    row += "1"
                else:
                    label = FEN_CHARS[pred.item()]
                    row += label
                    if label != "1":
                        piece_count += 1
                confs.append(prob.item())
            fen.append(row)

    result = "/".join(fen)
    result = "/".join([re.sub(r"1+", lambda m: str(len(m.group())), row) for row in result.split("/")])
    return result, float(np.mean(confs)), piece_count


def inset_board(img, px):
    w, h = img.size
    if w <= 2 * px or h <= 2 * px:
        return img
    return img.crop((px, px, w - px, h - px)).resize((w, h), Image.LANCZOS)


def predict_board(image_path, model_path=None):
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

    scored = []
    for tag, candidate_img in candidates:
        fen, conf, piece_count = infer_fen_on_image(candidate_img, model, device, USE_SQUARE_DETECTION)
        scored.append((tag, candidate_img, fen, conf, piece_count))
        if DEBUG_MODE:
            print(
                f"DEBUG: Candidate={tag} conf={conf:.4f} pieces={piece_count}",
                file=sys.stderr,
            )

    full_piece_count = next((item[4] for item in scored if item[0] == "full"), 0)
    full_conf = next((item[3] for item in scored if item[0] == "full"), 0.0)
    filtered = []
    for item in scored:
        tag, _, _, _, piece_count = item
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

    best_tag, best_img, best_fen, best_conf, _ = max(filtered, key=lambda item: item[3])

    if DEBUG_MODE:
        print(f"DEBUG: Using model={resolved_model_path}", file=sys.stderr)
        print(f"DEBUG: Selected candidate={best_tag} confidence={best_conf:.4f}", file=sys.stderr)
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
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    args = parser.parse_args()

    USE_EDGE_DETECTION = not args.no_edge_detection
    USE_SQUARE_DETECTION = args.detect_squares
    DEBUG_MODE = args.debug

    try:
        fen, conf = predict_board(args.image, model_path=args.model_path)
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
