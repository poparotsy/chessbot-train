import sys
import json
import re
import torch
import numpy as np
import cv2
from torch import nn
from PIL import Image

IMG_SIZE, FEN_CHARS = 64, "1PNBRQKpnbrqk"
import os
#_MODEL_CANDIDATES = [
#    os.path.join(os.path.dirname(__file__), "..", "models", "model_hybrid_v4_150e.pt"),
#    os.path.join(os.path.dirname(__file__), "..", "models", "model_hybrid_100e.pt"),
#    "models/model_hybrid_v4_150e.pt",
#    "models/model_hybrid_100e.pt",
#]
#MODEL_PATH = next((path for path in _MODEL_CANDIDATES if os.path.exists(path)), _MODEL_CANDIDATES[0])

MODEL_PATH = "models/model_hybrid_v4_150e.pt"

# Edge detection parameters (tune these if needed)
CANNY_LOW = 50
CANNY_HIGH = 150
CONTOUR_EPSILON = 0.02  # Lower = more precise, Higher = more tolerant
USE_EDGE_DETECTION = True  # Global flag, overridden by command-line
USE_SQUARE_DETECTION = False  # Detect individual squares (experimental)
DEBUG_MODE = False  # Save debug images


############
class StandaloneBeastClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(StandaloneBeastClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 13)
        )

    def forward(self, x):
        x = self.features(x)
        # Flatten here so it doesn't mess up the classifier indexing
        x = torch.flatten(x, 1)
        return self.classifier(x)

############
# Board Detection Functions


def order_corners(corners):
    """Order corners as: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]  # top-left (smallest sum)
    rect[2] = corners[np.argmax(s)]  # bottom-right (largest sum)
    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]  # top-right
    rect[3] = corners[np.argmax(diff)]  # bottom-left
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

        # Favor board-like quadrilaterals with decent margin over full-frame boxes.
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
    """Original v3 corner detector kept as fallback candidate."""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, CONTOUR_EPSILON * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            img_area = img.size[0] * img.size[1]
            if area > img_area * 0.25:
                return order_corners(approx.reshape(4, 2))
    return None




def perspective_transform(img, corners):
    """Deskew board to perfect square"""
    width = height = 512
    dst = np.array([[0, 0], [width, 0], [width, height],
                   [0, height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(np.array(img), M, (width, height))
    return Image.fromarray(warped)


def detect_grid_lines(img):
    """Detect chessboard grid lines to find exact square boundaries"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=img.size[0] // 3,
        maxLineGap=20)

    if lines is None:
        return None

    h_lines = []  # Horizontal lines (y positions)
    v_lines = []  # Vertical lines (x positions)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

        # Horizontal line (angle close to 0 or 180)
        if angle < 15 or angle > 165:
            h_lines.append((y1 + y2) // 2)
        # Vertical line (angle close to 90)
        elif 75 < angle < 105:
            v_lines.append((x1 + x2) // 2)

    if not h_lines or not v_lines:
        return None

    # Cluster nearby lines (within 10 pixels)
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

    # We need exactly 9 lines (or close to it)
    if 7 <= len(h_lines) <= 11 and 7 <= len(v_lines) <= 11:
        # If we have more than 9, take the most evenly spaced 9
        if len(h_lines) > 9:
            h_lines = h_lines[:9]
        if len(v_lines) > 9:
            v_lines = v_lines[:9]

        # If we have less than 9, fall back to uniform
        if len(h_lines) == 9 and len(v_lines) == 9:
            return v_lines, h_lines

    return None

############
# Preprocessing


def preprocess_image(img):
    """Apply aggressive preprocessing to improve recognition"""
    from PIL import ImageEnhance, ImageFilter
    import cv2

    # Convert to numpy for OpenCV processing
    img_np = np.array(img)

    # 1. Adaptive Histogram Equalization (CLAHE) - Critical for brightness
    # normalization
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = Image.fromarray(img_np)

    # 2. Denoising - helps with scanned/printed images
    img_np = cv2.fastNlMeansDenoisingColored(
        np.array(img), None, 10, 10, 7, 21)
    img = Image.fromarray(img_np)

    # 3. Sharpening - enhance edges
    img = img.filter(
        ImageFilter.UnsharpMask(
            radius=1,
            percent=150,
            threshold=3))

    # 4. Contrast enhancement
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)

    return img


def infer_fen_on_image(img, model, device, use_square_detection):
    """Run tile inference on a prepared board image and return fen + mean confidence."""
    w, h = img.size
    if use_square_detection:
        grid = detect_grid_lines(img)
        if grid is not None:
            xe, ye = grid
        else:
            xe, ye = np.linspace(0, w, 9).astype(int), np.linspace(0, h, 9).astype(int)
    else:
        xe, ye = np.linspace(0, w, 9).astype(int), np.linspace(0, h, 9).astype(int)

    fen, confs = [], []
    with torch.no_grad():
        for r in range(8):
            row = ""
            for c in range(8):
                tile = img.crop((xe[c], ye[r], xe[c + 1], ye[r + 1])).resize((64, 64), Image.LANCZOS)
                img_np = np.array(tile).transpose(2, 0, 1)
                it = torch.from_numpy(img_np).float().to(device)
                it = (it / 127.5) - 1.0
                it = it.unsqueeze(0)

                out = torch.softmax(model(it), dim=1)
                prob, pred = torch.max(out, 1)
                if prob.item() < 0.35:
                    row += '1'
                else:
                    row += FEN_CHARS[pred.item()]
                confs.append(prob.item())
            fen.append(row)

    res = "/".join(fen)
    res = "/".join([re.sub(r'1+', lambda m: str(len(m.group())), r) for r in res.split('/')])
    return res, float(np.mean(confs))


def inset_board(img, px):
    w, h = img.size
    if w <= 2 * px or h <= 2 * px:
        return img
    return img.crop((px, px, w - px, h - px)).resize((w, h), Image.LANCZOS)


def predict_board(image_path):
    global USE_EDGE_DETECTION, USE_SQUARE_DETECTION  # Declare global variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Setup debug directory
    debug_dir = None
    if DEBUG_MODE:
        import os
        debug_dir = os.path.join(os.path.dirname(image_path), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        base_name = os.path.basename(image_path).rsplit('.', 1)[0]

    # Load image
    img = Image.open(image_path).convert("RGB")
    original_img = img.copy()
    candidates = [("full", img)]

    if USE_EDGE_DETECTION:
        robust_corners = find_board_corners(original_img)
        if robust_corners is not None:
            robust_img = perspective_transform(original_img, robust_corners)
            candidates.append(("robust", robust_img))
            candidates.append(("robust_inset2", inset_board(robust_img, 2)))
        legacy_corners = find_board_corners_legacy(original_img)
        if legacy_corners is not None:
            legacy_img = perspective_transform(original_img, legacy_corners)
            candidates.append(("legacy", legacy_img))
            candidates.append(("legacy_inset2", inset_board(legacy_img, 2)))

    best_fen = None
    best_conf = -1.0
    best_tag = "full"
    for tag, candidate_img in candidates:
        fen, conf = infer_fen_on_image(candidate_img, model, device, USE_SQUARE_DETECTION)
        if conf > best_conf:
            best_fen = fen
            best_conf = conf
            best_tag = tag

    if DEBUG_MODE:
        print(f"DEBUG: Selected candidate={best_tag} confidence={best_conf:.4f}", file=sys.stderr)
        selected_img = None
        for tag, img_candidate in candidates:
            if tag == best_tag:
                selected_img = img_candidate
                break
        if selected_img is None:
            selected_img = original_img
        preprocessed_path = os.path.join(debug_dir, f"{base_name}_preprocessed.png")
        selected_img.save(preprocessed_path)
        print(f"DEBUG: Selected board image saved to {preprocessed_path}", file=sys.stderr)

    return best_fen, best_conf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Recognize chess position from image')
    parser.add_argument('image', help='Path to chess board image')
    parser.add_argument('--no-edge-detection', action='store_true',
                        help='Disable edge detection (use full image)')
    parser.add_argument('--detect-squares', action='store_true',
                        help='Enable square grid detection (experimental)')
    parser.add_argument('--debug', action='store_true',
                        help='Save debug images to debug/ folder')
    args = parser.parse_args()

    # Global flags
    USE_EDGE_DETECTION = not args.no_edge_detection
    USE_SQUARE_DETECTION = args.detect_squares
    DEBUG_MODE = args.debug

    try:
        f, c = predict_board(args.image)
        result = {
            "success": True,
            "fen": f"{f} w - - 0 1",
            "confidence": round(c, 4)
        }

        # Add warning for low confidence predictions
        if c < 0.67:
            result["warning"] = "Low confidence - prediction may be incorrect"

        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
