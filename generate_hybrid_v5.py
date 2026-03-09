import os, io, random, chess, torch, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont

# THE GLOBAL LABEL LAW (Aligned with audit_dataset.py)
FEN_CHARS = "1PNBRQKpnbrqk" 
# 0:1, 1:P, 2:N, 3:B, 4:R, 5:Q, 6:K, 7:p, 8:n, 9:b, 10:r, 11:q, 12:k

IMG_SIZE = 64
BOARDS_PER_CHUNK = 1000
CHUNKS_TRAIN, CHUNKS_VAL = 10, 2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOARD_THEMES_DIR = os.path.join(BASE_DIR, "board_themes")
PIECE_SETS_DIR = os.path.join(BASE_DIR, "piece_sets")
OUTPUT_DIR = os.path.join(BASE_DIR, "tensors_v5")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def env_float(name, default):
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def env_int(name, default):
    raw = os.getenv(name)
    return int(raw) if raw is not None else default

BASE_CONFIG = {
    "ENABLE_WATERMARK_AUG": os.getenv("ENABLE_WATERMARK_AUG", "1") != "0",
    "WATERMARK_BOARD_PROB": env_float("WATERMARK_BOARD_PROB", 0.3),
    "WATERMARK_MIN_PER_BOARD": env_int("WATERMARK_MIN_PER_BOARD", 1),
    "WATERMARK_MAX_PER_BOARD": env_int("WATERMARK_MAX_PER_BOARD", 2),
    "WATERMARK_SCALE_MIN": env_float("WATERMARK_SCALE_MIN", 1.0),
    "WATERMARK_SCALE_MAX": env_float("WATERMARK_SCALE_MAX", 1.5),
    "WATERMARK_FULL_KING_WORDMARK_PROB": env_float("WATERMARK_FULL_KING_WORDMARK_PROB", 0.55),
    "LABELS_PROB": env_float("LABELS_PROB", 0.60),
    "TRIM_CAPTURE_PROB": env_float("TRIM_CAPTURE_PROB", 0.35),
    "ARTIFACT_EMPTY_TILE_PROB": env_float("ARTIFACT_EMPTY_TILE_PROB", 0.25),
    "HIGHLIGHT_BOARD_PROB": env_float("HIGHLIGHT_BOARD_PROB", 0.70),
    "ARROW_BOARD_PROB": env_float("ARROW_BOARD_PROB", 0.65),
    "TACTICAL_MARKER_PROB": env_float("TACTICAL_MARKER_PROB", 0.80),
    "HARD_EDGE_ROOK_PROB": env_float("HARD_EDGE_ROOK_PROB", 0.30),
    "SPARSE_BOARD_PROB": env_float("SPARSE_BOARD_PROB", 0.18),
    "SCREENSHOT_CLUTTER_PROB": env_float("SCREENSHOT_CLUTTER_PROB", 0.15),
    "MIN_PLIES": env_int("MIN_PLIES", 5),
    "MAX_PLIES": env_int("MAX_PLIES", 65),
}

PROFILE_OVERRIDES = {
    "default": {},
    "screenshot_clutter": {
        "LABELS_PROB": 1.0,
        "TRIM_CAPTURE_PROB": 0.10,
        "ARTIFACT_EMPTY_TILE_PROB": 0.10,
        "HIGHLIGHT_BOARD_PROB": 0.08,
        "ARROW_BOARD_PROB": 0.04,
        "TACTICAL_MARKER_PROB": 0.06,
        "WATERMARK_BOARD_PROB": 1.0,
        "WATERMARK_MIN_PER_BOARD": 1,
        "WATERMARK_MAX_PER_BOARD": 4,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.92,
        "HARD_EDGE_ROOK_PROB": 0.15,
        "SPARSE_BOARD_PROB": 0.90,
        "SCREENSHOT_CLUTTER_PROB": 1.0,
        "MIN_PLIES": 0,
        "MAX_PLIES": 10,
    },
    "edge_rook": {
        "LABELS_PROB": 0.85,
        "TRIM_CAPTURE_PROB": 0.25,
        "WATERMARK_BOARD_PROB": 0.35,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.70,
        "HARD_EDGE_ROOK_PROB": 1.0,
        "SPARSE_BOARD_PROB": 0.55,
        "SCREENSHOT_CLUTTER_PROB": 0.55,
        "MIN_PLIES": 0,
        "MAX_PLIES": 18,
    },
    "hard_mix": {
        "LABELS_PROB": 0.90,
        "TRIM_CAPTURE_PROB": 0.45,
        "ARTIFACT_EMPTY_TILE_PROB": 0.45,
        "HIGHLIGHT_BOARD_PROB": 0.45,
        "ARROW_BOARD_PROB": 0.35,
        "TACTICAL_MARKER_PROB": 0.40,
        "WATERMARK_BOARD_PROB": 0.80,
        "WATERMARK_MIN_PER_BOARD": 1,
        "WATERMARK_MAX_PER_BOARD": 3,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.85,
        "HARD_EDGE_ROOK_PROB": 0.75,
        "SPARSE_BOARD_PROB": 0.60,
        "SCREENSHOT_CLUTTER_PROB": 0.60,
        "MIN_PLIES": 0,
        "MAX_PLIES": 20,
    },
}

# Targeted default mix for current failure modes:
# - puzzle-00024: empty-square logos/text misread as kings (screenshot_clutter)
# - puzzle-00028: missed edge rook (edge_rook)
DEFAULT_PROFILE_WEIGHTS = [
    ("default", 0.20),
    ("screenshot_clutter", 0.45),
    ("edge_rook", 0.30),
    ("hard_mix", 0.05),
]


def get_profile_config(profile):
    if profile not in PROFILE_OVERRIDES:
        raise ValueError(f"Unknown generation profile: {profile}")
    cfg = dict(BASE_CONFIG)
    cfg.update(PROFILE_OVERRIDES[profile])
    return cfg


def choose_profile(profile=None):
    if profile:
        return profile
    names = [name for name, _ in DEFAULT_PROFILE_WEIGHTS]
    weights = [weight for _, weight in DEFAULT_PROFILE_WEIGHTS]
    return random.choices(names, weights=weights, k=1)[0]


def augment_image(img):
    """Realistic augmentation mix for robust tile classification."""
    import cv2

    # 1. Optional grayscale conversion (keep low so color cues remain primary).
    if random.random() < 0.15:
        img = img.convert('L').convert('RGB')

    # 2. Micro-rotations.
    if random.random() < 0.25:
        angle = random.uniform(-2.5, 2.5)
        img = img.rotate(angle, fillcolor=(128, 128, 128))

    # 3. Mild perspective jitter to simulate camera skew.
    if random.random() < 0.20:
        arr = np.array(img)
        h, w = arr.shape[:2]
        max_j = max(2, int(min(h, w) * 0.02))
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = src + np.float32([
            [random.randint(-max_j, max_j), random.randint(-max_j, max_j)],
            [random.randint(-max_j, max_j), random.randint(-max_j, max_j)],
            [random.randint(-max_j, max_j), random.randint(-max_j, max_j)],
            [random.randint(-max_j, max_j), random.randint(-max_j, max_j)],
        ])
        mat = cv2.getPerspectiveTransform(src, dst)
        arr = cv2.warpPerspective(arr, mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        img = Image.fromarray(arr)

    # 4. CLAHE (randomized settings), not always-on.
    if random.random() < 0.50:
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clip = random.uniform(1.6, 2.4)
        grid = random.choice([(6, 6), (8, 8), (10, 10)])
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        img = Image.fromarray(img_np)

    # 5. Moderate brightness/contrast/saturation jitter.
    if random.random() < 0.55:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.78, 1.22))
    if random.random() < 0.55:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.80, 1.25))
    if random.random() < 0.25:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.90, 1.10))

    # 6. Mild sharpening, occasional.
    if random.random() < 0.20:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))

    # 7. Slight Gaussian noise.
    if random.random() < 0.15:
        arr = np.array(img)
        noise = np.random.normal(0, random.uniform(3.0, 10.0), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # 8. Slight blur for compression/scan artifacts.
    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.9)))

    return img

def draw_arrow(draw, start_square, end_square, color, ts):
    """Draw a proper chess arrow with arrowhead"""
    # Calculate center points
    start_x = start_square[1] * ts + ts // 2
    start_y = start_square[0] * ts + ts // 2
    end_x = end_square[1] * ts + ts // 2
    end_y = end_square[0] * ts + ts // 2
    
    # Draw thick line
    draw.line([start_x, start_y, end_x, end_y], fill=color, width=random.randint(8, 15))
    
    # Draw arrowhead (simple triangle)
    import math
    angle = math.atan2(end_y - start_y, end_x - start_x)
    arrow_size = 20
    
    # Three points of triangle
    p1 = (end_x, end_y)
    p2 = (end_x - arrow_size * math.cos(angle - math.pi/6),
          end_y - arrow_size * math.sin(angle - math.pi/6))
    p3 = (end_x - arrow_size * math.cos(angle + math.pi/6),
          end_y - arrow_size * math.sin(angle + math.pi/6))
    
    draw.polygon([p1, p2, p3], fill=color)


def draw_annotation_marker(draw, square_rc, ts):
    """Draw tactical annotation marker: small ring, !, or !! near tile corners."""
    r, c = square_rc
    x0, y0 = c * ts, r * ts

    # Corner-biased placement, often overlapping piece/tile boundary like puzzle UIs.
    corner = random.choice(("tr", "tl", "br", "bl"))
    offset = max(6, ts // 8)
    jitter = 4
    if corner == "tr":
        center_x = x0 + ts - offset + random.randint(-jitter, jitter)
        center_y = y0 + offset + random.randint(-jitter, jitter)
    elif corner == "tl":
        center_x = x0 + offset + random.randint(-jitter, jitter)
        center_y = y0 + offset + random.randint(-jitter, jitter)
    elif corner == "br":
        center_x = x0 + ts - offset + random.randint(-jitter, jitter)
        center_y = y0 + ts - offset + random.randint(-jitter, jitter)
    else:
        center_x = x0 + offset + random.randint(-jitter, jitter)
        center_y = y0 + ts - offset + random.randint(-jitter, jitter)

    radius = random.randint(9, 12)
    marker = random.choices(["ring", "!", "!!"], weights=[0.20, 0.45, 0.35], k=1)[0]

    if marker == "ring":
        draw.ellipse(
            [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            outline=(255, 0, 0, random.randint(140, 200)),
            width=3)
        return

    # Chess.com-style brilliant marker: turquoise round badge + thick white exclamation bars.
    badge = random.choice([(37, 176, 176, 230), (29, 166, 167, 230), (42, 183, 184, 225)])
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        fill=badge,
        outline=(18, 120, 120, 170),
        width=1)

    # Subtle inner highlight for glossy look.
    inner_r = max(3, radius - 3)
    draw.ellipse(
        [center_x - inner_r, center_y - inner_r, center_x + inner_r, center_y + inner_r],
        outline=(180, 245, 245, 60),
        width=1)

    count = 2 if marker == "!!" else 1
    bar_h = max(6, int(radius * 0.95))
    bar_w = max(2, int(radius * 0.32))
    dot_h = max(2, int(radius * 0.28))
    gap = max(2, int(bar_w * 0.6))
    total_w = count * bar_w + (count - 1) * gap
    start_x = center_x - total_w // 2
    top_y = center_y - int(radius * 0.52)

    for idx in range(count):
        x0 = int(start_x + idx * (bar_w + gap))
        x1 = int(x0 + bar_w)
        y0 = int(top_y)
        y1 = int(y0 + bar_h)
        # Tiny shadow
        draw.rounded_rectangle([x0 + 1, y0 + 1, x1 + 1, y1 + 1], radius=2, fill=(0, 0, 0, 45))
        # Main white stroke
        draw.rounded_rectangle([x0, y0, x1, y1], radius=2, fill=(244, 244, 244, 255))

        dot_y0 = int(y1 + max(1, radius * 0.08))
        dot_y1 = int(dot_y0 + dot_h)
        draw.rounded_rectangle([x0 + 1, dot_y0 + 1, x1 + 1, dot_y1 + 1], radius=1, fill=(0, 0, 0, 45))
        draw.rounded_rectangle([x0, dot_y0, x1, dot_y1], radius=1, fill=(244, 244, 244, 255))


def draw_empty_artifact(draw, square_rc, ts):
    """Draw artifact fragments on an empty square without adding piece-like silhouettes."""
    r, c = square_rc
    x0, y0 = c * ts, r * ts

    mode = random.choice(("marker", "short_arrow", "small_highlight"))
    if mode == "marker":
        draw_annotation_marker(draw, square_rc, ts)
        return

    if mode == "small_highlight":
        pad = random.randint(10, 18)
        color = random.choice([(255, 230, 0, 70), (0, 190, 255, 65), (120, 255, 120, 65)])
        draw.rectangle([x0 + pad, y0 + pad, x0 + ts - pad, y0 + ts - pad], fill=color)
        return

    # Short local arrow-like stroke confined inside one tile.
    cx, cy = x0 + ts // 2, y0 + ts // 2
    dx, dy = random.randint(-12, 12), random.randint(-12, 12)
    sx, sy = cx - dx, cy - dy
    ex, ey = cx + dx, cy + dy
    color = random.choice([(255, 165, 0, 110), (0, 255, 0, 105), (0, 150, 255, 105)])
    draw.line([sx, sy, ex, ey], fill=color, width=random.randint(4, 7))


def draw_watermark_overlay(img, draw, square_rc, ts, cfg):
    """Draw puzzle-site style watermark: piece-like silhouette + letter mark."""
    r, c = square_rc
    x0, y0 = c * ts, r * ts

    def _safe_text_size(text, font, size_hint):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
        except Exception:
            # Fallback for PIL/font edge cases.
            width = max(1, int(len(text) * size_hint * 0.62))
            height = max(1, int(size_hint * 1.05))
            return width, height

    def _safe_draw_text(xy, text, fill, font):
        try:
            draw.text(xy, text, fill=fill, font=font)
        except Exception:
            fallback_font = ImageFont.load_default()
            draw.text(xy, text, fill=fill, font=fallback_font)

    def _draw_wordmark_text(x, y, size, alpha=205, style="mono"):
        wordmark = random.choice(("SLC", "S.C.C", "LCC", "CHS", "ICC", "C.L.C"))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except Exception:
            font = ImageFont.load_default()
        if style != "crest":
            tw, th = _safe_text_size(wordmark, font, size)
            tx = x - tw / 2 + random.randint(-1, 1)
            ty = y - th / 2 + random.randint(-1, 1)
            _safe_draw_text((tx + 1, ty + 1), wordmark, (30, 30, 30, 90), font)
            _safe_draw_text((tx, ty), wordmark, (248, 248, 248, alpha), font)
            return

        # Crest style: 3 caps with larger center glyph (similar to puzzle logos).
        letters = random.choice((("S", "L", "C"), ("S", "C", "C"), ("C", "L", "C"), ("I", "C", "C")))
        side_size = max(8, int(size * 0.88))
        mid_size = max(10, int(size * 1.30))
        try:
            side_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", side_size)
            mid_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", mid_size)
        except Exception:
            side_font = ImageFont.load_default()
            mid_font = ImageFont.load_default()

        side_w, _ = _safe_text_size(letters[0], side_font, side_size)
        mid_w, _ = _safe_text_size(letters[1], mid_font, mid_size)
        side2_w, _ = _safe_text_size(letters[2], side_font, side_size)
        total_w = side_w + mid_w + side2_w + max(2, int(size * 0.10)) * 2
        tx = x - total_w / 2
        baseline_y = y - side_size / 2

        gap = max(2, int(size * 0.10))
        # Shadow layer
        _safe_draw_text((tx + 1, baseline_y + 1), letters[0], (25, 25, 25, 110), side_font)
        _safe_draw_text((tx + side_w + gap + 1, baseline_y - int(size * 0.20) + 1), letters[1], (25, 25, 25, 110), mid_font)
        _safe_draw_text((tx + side_w + gap + mid_w + gap + 1, baseline_y + 1), letters[2], (25, 25, 25, 110), side_font)
        # Main layer
        fg = (18, 18, 18, alpha)
        _safe_draw_text((tx, baseline_y), letters[0], fg, side_font)
        _safe_draw_text((tx + side_w + gap, baseline_y - int(size * 0.20)), letters[1], fg, mid_font)
        _safe_draw_text((tx + side_w + gap + mid_w + gap, baseline_y), letters[2], fg, side_font)

    # High-fidelity "logo tile" style: full king + 3-cap wordmark.
    if random.random() < cfg["WATERMARK_FULL_KING_WORDMARK_PROB"]:
        # Build a badge-like overlay inside tile (closer to puzzle-site logos).
        bw = int(ts * random.uniform(0.80, 1.03))
        bh = int(ts * random.uniform(0.74, 0.98))
        bx0 = x0 + random.randint(-int(ts * 0.18), int(ts * 0.06))
        by0 = y0 + random.randint(-int(ts * 0.04), int(ts * 0.18))
        bx0 = max(0, min(img.size[0] - bw, bx0))
        by0 = max(0, min(img.size[1] - bh, by0))
        bx1, by1 = bx0 + bw, by0 + bh
        cx = (bx0 + bx1) // 2

        badge_fill = random.choice([(236, 233, 216, 210), (232, 228, 208, 220), (238, 236, 221, 200)])
        badge_edge = random.choice([(142, 132, 110, 160), (120, 112, 95, 150)])
        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=max(4, int(ts * 0.08)), fill=badge_fill, outline=badge_edge, width=1)

        # Top arc-ish text line (generic, not copied).
        top_line = random.choice(("STRATEGIC CHESS", "TACTICAL CHESS", "CLASSIC CHESS", "CHESS SOCIETY"))
        try:
            top_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(7, int(ts * 0.11)))
        except Exception:
            top_font = ImageFont.load_default()
        tw, _ = _safe_text_size(top_line, top_font, max(7, int(ts * 0.11)))
        tx = cx - tw / 2 + random.randint(-2, 2)
        ty = by0 + int(bh * 0.08)
        _safe_draw_text((tx, ty), top_line, (25, 25, 25, random.randint(120, 190)), top_font)

        # Central king silhouette (dark) to trigger king-like confusion robustly.
        king_col = (22, 22, 22, random.randint(160, 230))
        kcx = cx + random.randint(-2, 2)
        kcy = by0 + int(bh * 0.56)
        kw = int(bw * 0.34)
        kh = int(bh * 0.36)
        draw.rounded_rectangle([kcx - kw // 2, kcy - int(kh * 0.14), kcx + kw // 2, kcy + kh // 2], radius=3, fill=king_col)
        draw.ellipse([kcx - int(kw * 0.18), kcy - int(kh * 0.58), kcx + int(kw * 0.18), kcy - int(kh * 0.20)], fill=king_col)
        cw = max(1, kw // 12)
        ch = max(2, kh // 6)
        draw.rectangle([kcx - cw, kcy - int(kh * 0.70), kcx + cw, kcy - int(kh * 0.45)], fill=king_col)
        draw.rectangle([kcx - int(cw * 2.6), kcy - int(kh * 0.58), kcx + int(cw * 2.6), kcy - int(kh * 0.50)], fill=king_col)

        # Centered 3-cap crest letters.
        _draw_wordmark_text(cx, by0 + int(bh * 0.60), max(10, int(ts * 0.21)), alpha=random.randint(170, 235), style="crest")

        # Bottom tiny tag.
        bottom_line = random.choice(("CLUB", "CHESS CLUB", "SOCIETY"))
        try:
            bottom_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(6, int(ts * 0.10)))
        except Exception:
            bottom_font = ImageFont.load_default()
        btw, _ = _safe_text_size(bottom_line, bottom_font, max(6, int(ts * 0.10)))
        _safe_draw_text((cx - btw / 2, by1 - int(bh * 0.18)), bottom_line, (30, 30, 30, random.randint(120, 195)), bottom_font)
        return

    # Bottom-corner bias matches common puzzle-site logo placement.
    anchor = random.choice(("bl", "br", "bc"))
    if anchor == "bl":
        cx = x0 + random.randint(10, 18)
        cy = y0 + ts - random.randint(10, 18)
    elif anchor == "br":
        cx = x0 + ts - random.randint(10, 18)
        cy = y0 + ts - random.randint(10, 18)
    else:
        cx = x0 + ts // 2 + random.randint(-6, 6)
        cy = y0 + ts - random.randint(10, 16)

    mark_size = int(ts * random.uniform(cfg["WATERMARK_SCALE_MIN"], cfg["WATERMARK_SCALE_MAX"]))
    half = max(8, mark_size // 2)

    # Subtle blob base.
    base = random.choice([(126, 126, 126, 88), (102, 102, 102, 100), (142, 142, 142, 76)])
    draw.ellipse([cx - half, cy - half, cx + half, cy + half], fill=base)

    # Piece-like silhouette (rook/king) inside blob.
    sil = random.choice(("rook", "king"))
    sil_col = (236, 236, 236, random.randint(105, 150))
    bx = cx - int(half * 0.42)
    by = cy - int(half * 0.48)
    bw = int(half * 0.84)
    bh = int(half * 0.98)

    if sil == "rook":
        # Rook body
        draw.rounded_rectangle([bx, by + int(bh * 0.28), bx + bw, by + bh], radius=2, fill=sil_col)
        # Rook top crenels
        tw = max(2, bw // 4)
        gap = max(1, tw // 3)
        top_y0 = by + int(bh * 0.08)
        top_y1 = by + int(bh * 0.32)
        for i in range(3):
            x0t = bx + i * (tw + gap)
            x1t = x0t + tw
            draw.rectangle([x0t, top_y0, x1t, top_y1], fill=sil_col)
    else:
        # King body
        draw.rounded_rectangle([bx, by + int(bh * 0.23), bx + bw, by + bh], radius=2, fill=sil_col)
        # Crown/head
        draw.ellipse([cx - int(bw * 0.22), by + int(bh * 0.02), cx + int(bw * 0.22), by + int(bh * 0.3)], fill=sil_col)
        # Tiny cross hint
        cxw = max(1, bw // 10)
        cyh = max(2, bh // 9)
        draw.rectangle([cx - cxw, by - cyh, cx + cxw, by + cyh], fill=sil_col)
        draw.rectangle([cx - int(cxw * 2.4), by, cx + int(cxw * 2.4), by + cyh], fill=sil_col)

    # Wordmark over the piece (club/site style).
    _draw_wordmark_text(cx, cy, max(10, int(half * 0.72)), alpha=random.randint(145, 210))


def draw_board_labels(draw, ts, pov):
    if pov == "black":
        files = "hgfedcba"
        ranks = [str(i) for i in range(1, 9)]
    else:
        files = "abcdefgh"
        ranks = [str(8 - i) for i in range(8)]

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = None

    for i, letter in enumerate(files):
        x = i * ts + ts - 12
        y = 512 - 16
        draw.text((x, y), letter, fill=(128, 128, 128, 180), font=font)

    x = 4 if random.random() < 0.5 else 512 - 14
    for i, rank in enumerate(ranks):
        y = i * ts + 4
        draw.text((x, y), rank, fill=(128, 128, 128, 180), font=font)


def vandalize(img, grid, cfg):
    """Add arrows/highlights like chess sites with artifact-on-empty oversampling."""
    draw = ImageDraw.Draw(img, "RGBA")
    w = img.size[0]
    ts = w // 8
    empty_squares = [(rr, cc) for rr in range(8) for cc in range(8) if not grid[rr][cc]]

    # Square highlights are common but not universal.
    if random.random() < cfg["HIGHLIGHT_BOARD_PROB"]:
        for _ in range(random.randint(1, 2)):
            r, c = random.randint(0, 7), random.randint(0, 7)
            color = random.choice([(0, 255, 0, 65), (255, 255, 0, 65), (255, 0, 0, 65), (0, 150, 255, 65)])
            draw.rectangle([c * ts, r * ts, (c + 1) * ts, (r + 1) * ts], fill=color)

    # Arrows appear often, but not on every board.
    if random.random() < cfg["ARROW_BOARD_PROB"]:
        for _ in range(random.randint(1, 2)):
            start_r, start_c = random.randint(0, 7), random.randint(0, 7)
            end_r = max(0, min(7, start_r + random.randint(-3, 3)))
            end_c = max(0, min(7, start_c + random.randint(-3, 3)))
            if (start_r, start_c) != (end_r, end_c):
                color = random.choice([(0, 255, 0, 110), (255, 165, 0, 110), (255, 0, 0, 110), (0, 150, 255, 110)])
                draw_arrow(draw, (start_r, start_c), (end_r, end_c), color, ts)

    # Tactical markers are frequent for robustness against ! and !! overlays.
    if random.random() < cfg["TACTICAL_MARKER_PROB"]:
        for _ in range(random.choices([1, 2, 3], weights=[0.60, 0.30, 0.10], k=1)[0]):
            r, c = random.randint(0, 7), random.randint(0, 7)
            draw_annotation_marker(draw, (r, c), ts)

    # Targeted hardening: artifact-only empty tiles to reduce empty->piece hallucination.
    if empty_squares and random.random() < cfg["ARTIFACT_EMPTY_TILE_PROB"]:
        random.shuffle(empty_squares)
        for sq in empty_squares[:random.randint(2, 4)]:
            draw_empty_artifact(draw, sq, ts)

    # Watermark-like overlays on empty squares (controlled via top-level vars).
    if cfg["ENABLE_WATERMARK_AUG"] and empty_squares and random.random() < cfg["WATERMARK_BOARD_PROB"]:
        # Bias watermark injection toward bottom ranks where logos commonly sit.
        bottom_pref = [sq for sq in empty_squares if sq[0] >= 6]
        pool = bottom_pref if bottom_pref else empty_squares
        random.shuffle(pool)
        n = random.randint(cfg["WATERMARK_MIN_PER_BOARD"], cfg["WATERMARK_MAX_PER_BOARD"])
        for sq in pool[:n]:
            draw_watermark_overlay(img, draw, sq, ts, cfg)

    return img


def simulate_trimmed_capture(img, cfg):
    """Simulate imperfect screenshots where board edges/labels are partially clipped."""
    if random.random() >= cfg["TRIM_CAPTURE_PROB"]:
        return img

    w, h = img.size
    max_crop_x = max(4, int(w * 0.05))
    max_crop_y = max(4, int(h * 0.05))

    # Asymmetric side trimming is common in mobile/browser screenshots.
    left = random.randint(0, max_crop_x)
    right = random.randint(0, max_crop_x)
    top = random.randint(0, max_crop_y)
    bottom = random.randint(0, max_crop_y)

    # Keep at least 80% of each dimension before resizing back.
    if (left + right) > int(w * 0.20):
        right = max(0, int(w * 0.20) - left)
    if (top + bottom) > int(h * 0.20):
        bottom = max(0, int(h * 0.20) - top)

    cropped = img.crop((left, top, w - right, h - bottom))
    return cropped.resize((w, h), Image.LANCZOS)

def render_board(fen, return_meta=False, profile=None):
    profile = choose_profile(profile)
    cfg = get_profile_config(profile)
    board_file = random.choice(os.listdir(BOARD_THEMES_DIR))
    background = Image.open(os.path.join(BOARD_THEMES_DIR, board_file)).convert("RGB").resize((512, 512))
    p_set = random.choice(os.listdir(PIECE_SETS_DIR))
    ts = 64
    
    grid = [[None]*8 for _ in range(8)]
    for r, row_str in enumerate(fen.split('/')):
        c = 0
        for char in row_str:
            if char.isdigit(): c += int(char)
            else: grid[r][c] = char; c += 1

    for r in range(8):
        for c in range(8):
            char = grid[r][c]
            if char:
                # Standard Lichess filename mapping: wP, bK, etc.
                p_name = f"{'w' if char.isupper() else 'b'}{char.upper()}.png"
                p_img = Image.open(os.path.join(PIECE_SETS_DIR, p_set, p_name)).convert("RGBA").resize((ts, ts))
                background.paste(p_img, (c*ts, r*ts), p_img)

    # Add coordinate labels - randomize position to match real-world variety
    label_choice = random.random()
    label_pov = None
    if label_choice < cfg["LABELS_PROB"]:
        draw = ImageDraw.Draw(background)
        label_pov = random.choice(("white", "black")) if profile != "default" else "white"
        draw_board_labels(draw, ts, label_pov)
    background = vandalize(background, grid, cfg)
    
    # Apply augmentation BEFORE compression
    background = augment_image(background)
    
    # JPEG compression artifacts are common in user screenshots.
    if random.random() < 0.60:
        buf = io.BytesIO()
        background.save(buf, "JPEG", quality=random.randint(45, 92))
        background = Image.open(buf).copy()

    # Simulate trimmed screenshot captures before tile slicing.
    background = simulate_trimmed_capture(background, cfg)

    tiles, labels = [], []
    for r in range(8):
        for c in range(8):
            tile = background.crop((c*ts, r*ts, (c+1)*ts, (r+1)*ts))
            tiles.append(np.array(tile, dtype=np.uint8).transpose(2,0,1))
            labels.append(FEN_CHARS.index(grid[r][c]) if grid[r][c] else 0)
    if return_meta:
        return tiles, labels, {
            "board_theme": board_file,
            "piece_set": p_set,
            "label_pov": label_pov,
            "profile": profile,
        }
    return tiles, labels


def board_has_edge_rook(board):
    edge_squares = []
    for sq in chess.SQUARES:
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        if file_idx in (0, 7) or rank_idx in (0, 7):
            edge_squares.append(sq)
    for sq in edge_squares:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.ROOK:
            return True
    return False


def is_sparse_board(board):
    return len(board.piece_map()) <= 10


def _random_non_adjacent_square(taken, forbidden_neighbors=()):
    candidates = []
    forbidden = set(forbidden_neighbors)
    for sq in chess.SQUARES:
        if sq in taken or sq in forbidden:
            continue
        candidates.append(sq)
    return random.choice(candidates) if candidates else None


def _king_neighborhood(square):
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    blocked = set()
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            nf, nr = file_idx + df, rank_idx + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                blocked.add(chess.square(nf, nr))
    return blocked


def build_screenshot_clutter_board(force_edge_rook=False):
    """Build sparse screenshot-style boards with lots of empty squares."""
    board = chess.Board(None)
    taken = set()

    white_king = random.choice([
        chess.C1, chess.G1, chess.B2, chess.F2, chess.C2, chess.G2
    ])
    taken.add(white_king)
    black_king = _random_non_adjacent_square(taken, _king_neighborhood(white_king))
    if black_king is None:
        black_king = chess.E8
    taken.add(black_king)

    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))

    optional_pieces = [
        (chess.QUEEN, chess.WHITE), (chess.QUEEN, chess.BLACK),
        (chess.ROOK, chess.WHITE), (chess.ROOK, chess.BLACK),
        (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.BLACK),
        (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.BLACK),
        (chess.PAWN, chess.WHITE), (chess.PAWN, chess.BLACK),
        (chess.PAWN, chess.WHITE), (chess.PAWN, chess.BLACK),
    ]
    random.shuffle(optional_pieces)

    if force_edge_rook:
        edge_targets = [chess.A7, chess.H7, chess.A2, chess.H2, chess.A4, chess.H5]
        sq = random.choice([cand for cand in edge_targets if cand not in taken])
        color = random.choice([chess.WHITE, chess.BLACK])
        board.set_piece_at(sq, chess.Piece(chess.ROOK, color))
        taken.add(sq)

    extra_count = random.randint(1, 5)
    for piece_type, color in optional_pieces:
        if extra_count <= 0:
            break
        candidates = []
        for sq in chess.SQUARES:
            if sq in taken:
                continue
            if piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            candidates.append(sq)
        if not candidates:
            continue
        sq = random.choice(candidates)
        board.set_piece_at(sq, chess.Piece(piece_type, color))
        taken.add(sq)
        extra_count -= 1

    return board


def random_training_board(profile=None):
    profile = choose_profile(profile)
    cfg = get_profile_config(profile)
    if random.random() < cfg["SCREENSHOT_CLUTTER_PROB"]:
        force_edge_rook = random.random() < cfg["HARD_EDGE_ROOK_PROB"]
        return build_screenshot_clutter_board(force_edge_rook=force_edge_rook)

    for _ in range(200):
        board = chess.Board()
        target_sparse = random.random() < cfg["SPARSE_BOARD_PROB"]
        target_edge_rook = random.random() < cfg["HARD_EDGE_ROOK_PROB"]
        plies = random.randint(cfg["MIN_PLIES"], cfg["MAX_PLIES"])
        for _ in range(plies):
            if board.is_game_over():
                break
            board.push(random.choice(list(board.legal_moves)))

        if target_sparse and not is_sparse_board(board):
            continue
        if target_edge_rook and not board_has_edge_rook(board):
            continue
        return board

    board = chess.Board()
    for _ in range(random.randint(cfg["MIN_PLIES"], cfg["MAX_PLIES"])):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board

if __name__ == "__main__":
    for name in [f"val_{i}" for i in range(CHUNKS_VAL)] + [f"train_{i}" for i in range(CHUNKS_TRAIN)]:
        all_x, all_y = [], []
        profile_counts = {k: 0 for k in PROFILE_OVERRIDES.keys()}
        for _ in range(BOARDS_PER_CHUNK):
            profile = choose_profile()
            profile_counts[profile] += 1
            b = random_training_board(profile=profile)
            t, l = render_board(b.fen().split()[0], profile=profile)
            all_x.extend(t); all_y.extend(l)
        torch.save({'x': torch.from_numpy(np.stack(all_x)), 'y': torch.tensor(all_y)}, os.path.join(OUTPUT_DIR, f"{name}.pt"))
        mix = ", ".join(f"{k}:{v}" for k, v in profile_counts.items())
        print(f"✅ Created {name} | mix[{mix}]")
