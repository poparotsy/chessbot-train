import sys
import json
import re
import torch
import numpy as np
import cv2
from torch import nn
from PIL import Image

IMG_SIZE, FEN_CHARS, MODEL_PATH = 64, "1PNBRQKpnbrqk", "./models/model_hybrid_beast.pt"

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
        x = torch.flatten(x, 1) # Flatten here so it doesn't mess up the classifier indexing
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
    """Find chessboard using edge detection"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find largest quadrilateral
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, CONTOUR_EPSILON * peri, True)
        if len(approx) == 4:
            # Check if contour is large enough (at least 25% of image)
            area = cv2.contourArea(approx)
            img_area = img.size[0] * img.size[1]
            if area > img_area * 0.25:
                return order_corners(approx.reshape(4, 2))
    return None

def perspective_transform(img, corners):
    """Deskew board to perfect square"""
    width = height = 512
    dst = np.array([[0,0], [width,0], [width,height], [0,height]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(np.array(img), M, (width, height))
    return Image.fromarray(warped)

def detect_grid_lines(img):
    """Detect chessboard grid lines to find exact square boundaries"""
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 30, 100, apertureSize=3)
    
    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=img.size[0]//3, maxLineGap=20)
    
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
    
    # 1. Adaptive Histogram Equalization (CLAHE) - Critical for brightness normalization
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = Image.fromarray(img_np)
    
    # 2. Denoising - helps with scanned/printed images
    img_np = cv2.fastNlMeansDenoisingColored(np.array(img), None, 10, 10, 7, 21)
    img = Image.fromarray(img_np)
    
    # 3. Sharpening - enhance edges
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    # 4. Contrast enhancement
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    return img

def predict_board(image_path):
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
    edge_detection_used = False
    
    # Try edge detection (if enabled)
    if USE_EDGE_DETECTION:
        corners = find_board_corners(img)
        if corners is not None:
            print(f"DEBUG: Board corners detected, applying perspective transform", file=sys.stderr)
            
            # Save image with detected corners
            if DEBUG_MODE:
                debug_img = np.array(original_img.copy())
                cv2.polylines(debug_img, [corners.astype(np.int32)], True, (0, 255, 0), 3)
                debug_path = os.path.join(debug_dir, f"{base_name}_detected_board.png")
                Image.fromarray(debug_img).save(debug_path)
                print(f"DEBUG: Detected board outline saved to {debug_path}", file=sys.stderr)
            
            img = perspective_transform(img, corners)
            edge_detection_used = True
        else:
            print(f"DEBUG: No board corners detected, using full image", file=sys.stderr)
    else:
        print(f"DEBUG: Edge detection disabled, using full image", file=sys.stderr)
    
    # Apply preprocessing
    img = preprocess_image(img)
    
    # Save preprocessed image
    if DEBUG_MODE:
        preprocessed_path = os.path.join(debug_dir, f"{base_name}_preprocessed.png")
        img.save(preprocessed_path)
        suffix = " (with edge detection)" if edge_detection_used else " (no edge detection)"
        print(f"DEBUG: Preprocessed image saved to {preprocessed_path}{suffix}", file=sys.stderr)
    
    w, h = img.size
    
    # Try to detect grid lines for precise square boundaries
    grid_detected = False
    if USE_SQUARE_DETECTION:
        grid = detect_grid_lines(img)
        if grid is not None:
            xe, ye = grid
            grid_detected = True
            print(f"DEBUG: Grid lines detected: {len(xe)}x{len(ye)}", file=sys.stderr)
        else:
            print(f"DEBUG: Grid detection failed, using uniform division", file=sys.stderr)
            xe, ye = np.linspace(0, w, 9).astype(int), np.linspace(0, h, 9).astype(int)
    else:
        xe, ye = np.linspace(0, w, 9).astype(int), np.linspace(0, h, 9).astype(int)
    
    # Save grid visualization (always in debug mode)
    if DEBUG_MODE:
        grid_img = np.array(img.copy())
        for x in xe:
            cv2.line(grid_img, (x, 0), (x, h), (0, 0, 255), 2)  # Red vertical lines
        for y in ye:
            cv2.line(grid_img, (0, y), (w, y), (0, 0, 255), 2)  # Red horizontal lines
        grid_path = os.path.join(debug_dir, f"{base_name}_grid.png")
        Image.fromarray(grid_img).save(grid_path)
        grid_type = "detected" if grid_detected else "uniform"
        print(f"DEBUG: Grid visualization ({grid_type}) saved to {grid_path}", file=sys.stderr)
    
    fen, confs = [], []
    with torch.no_grad():
        for r in range(8):
            row = ""
            for c in range(8):
                # Crop and resize with preprocessing
                tile = img.crop((xe[c], ye[r], xe[c + 1], ye[r + 1])).resize((64, 64), Image.LANCZOS)
                
                # Per-tile normalization (critical for varying brightness)
                tile_np = np.array(tile)
                tile_np = cv2.normalize(tile_np, None, 0, 255, cv2.NORM_MINMAX)
                tile = Image.fromarray(tile_np)
                
                # Apply "THE LAW" Normalization exactly as in Trainer
                img_np = np.array(tile).transpose(2, 0, 1)
                it = torch.from_numpy(img_np).float().to(device)
                it = (it / 127.5) - 1.0 
                it = it.unsqueeze(0)
                
                # Inference
                out = torch.softmax(model(it), dim=1)
                prob, pred = torch.max(out, 1)
                
                # Empty square confidence threshold (prevent hallucination)
                if prob.item() < 0.35:
                    row += '1'  # Low confidence = empty square
                else:
                    row += FEN_CHARS[pred.item()]
                
                confs.append(prob.item())
            fen.append(row)
    # ... rest of your FEN compression logic ...
    res = "/".join(fen)
    res = "/".join([re.sub(r'1+', lambda m: str(len(m.group())), r)
                   for r in res.split('/')])
    return res, np.mean(confs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Recognize chess position from image')
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
