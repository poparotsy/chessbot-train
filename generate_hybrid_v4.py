import os, io, random, chess, torch, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont

# THE GLOBAL LABEL LAW (Aligned with audit_dataset.py)
FEN_CHARS = "1PNBRQKpnbrqk" 
# 0:1, 1:P, 2:N, 3:B, 4:R, 5:Q, 6:K, 7:p, 8:n, 9:b, 10:r, 11:q, 12:k

IMG_SIZE = 64
BOARDS_PER_CHUNK = 1000
CHUNKS_TRAIN, CHUNKS_VAL = 10, 2
OUTPUT_DIR = "tensors_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Watermark hard-negative settings (empty squares only)
ENABLE_WATERMARK_AUG = True
WATERMARK_BOARD_PROB = 0.3
WATERMARK_MIN_PER_BOARD = 1
WATERMARK_MAX_PER_BOARD = 2
WATERMARK_SCALE_MIN = 1.0
WATERMARK_SCALE_MAX = 1.5

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


def draw_watermark_overlay(draw, square_rc, ts):
    """Draw puzzle-site style watermark: piece-like silhouette + letter mark."""
    r, c = square_rc
    x0, y0 = c * ts, r * ts

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

    mark_size = int(ts * random.uniform(WATERMARK_SCALE_MIN, WATERMARK_SCALE_MAX))
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

    # Letter mark over the piece (club/site initial).
    letter = random.choice(("C", "D", "H", "L", "M", "S"))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(11, int(half * 0.9)))
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), letter, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = cx - tw / 2 + random.randint(-1, 1)
    ty = cy - th / 2 + random.randint(-1, 1)
    draw.text((tx + 1, ty + 1), letter, fill=(20, 20, 20, 70), font=font)
    draw.text((tx, ty), letter, fill=(248, 248, 248, random.randint(145, 210)), font=font)


def vandalize(img, grid):
    """Add arrows/highlights like chess sites with artifact-on-empty oversampling."""
    draw = ImageDraw.Draw(img, "RGBA")
    w = img.size[0]
    ts = w // 8
    empty_squares = [(rr, cc) for rr in range(8) for cc in range(8) if not grid[rr][cc]]

    # Square highlights are common but not universal.
    if random.random() < 0.70:
        for _ in range(random.randint(1, 2)):
            r, c = random.randint(0, 7), random.randint(0, 7)
            color = random.choice([(0, 255, 0, 65), (255, 255, 0, 65), (255, 0, 0, 65), (0, 150, 255, 65)])
            draw.rectangle([c * ts, r * ts, (c + 1) * ts, (r + 1) * ts], fill=color)

    # Arrows appear often, but not on every board.
    if random.random() < 0.65:
        for _ in range(random.randint(1, 2)):
            start_r, start_c = random.randint(0, 7), random.randint(0, 7)
            end_r = max(0, min(7, start_r + random.randint(-3, 3)))
            end_c = max(0, min(7, start_c + random.randint(-3, 3)))
            if (start_r, start_c) != (end_r, end_c):
                color = random.choice([(0, 255, 0, 110), (255, 165, 0, 110), (255, 0, 0, 110), (0, 150, 255, 110)])
                draw_arrow(draw, (start_r, start_c), (end_r, end_c), color, ts)

    # Tactical markers are frequent for robustness against ! and !! overlays.
    if random.random() < 0.80:
        for _ in range(random.choices([1, 2, 3], weights=[0.60, 0.30, 0.10], k=1)[0]):
            r, c = random.randint(0, 7), random.randint(0, 7)
            draw_annotation_marker(draw, (r, c), ts)

    # Targeted hardening: artifact-only empty tiles to reduce empty->piece hallucination.
    if empty_squares and random.random() < 0.25:
        random.shuffle(empty_squares)
        for sq in empty_squares[:random.randint(2, 4)]:
            draw_empty_artifact(draw, sq, ts)

    # Watermark-like overlays on empty squares (controlled via top-level vars).
    if ENABLE_WATERMARK_AUG and empty_squares and random.random() < WATERMARK_BOARD_PROB:
        # Bias watermark injection toward bottom ranks where logos commonly sit.
        bottom_pref = [sq for sq in empty_squares if sq[0] >= 6]
        pool = bottom_pref if bottom_pref else empty_squares
        random.shuffle(pool)
        n = random.randint(WATERMARK_MIN_PER_BOARD, WATERMARK_MAX_PER_BOARD)
        for sq in pool[:n]:
            draw_watermark_overlay(draw, sq, ts)

    return img


def simulate_trimmed_capture(img):
    """Simulate imperfect screenshots where board edges/labels are partially clipped."""
    if random.random() >= 0.35:
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

def render_board(fen):
    board_file = random.choice(os.listdir("board_themes"))
    background = Image.open(f"board_themes/{board_file}").convert("RGB").resize((512, 512))
    p_set = random.choice(os.listdir("piece_sets"))
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
                p_img = Image.open(f"piece_sets/{p_set}/{p_name}").convert("RGBA").resize((ts, ts))
                background.paste(p_img, (c*ts, r*ts), p_img)

    # Add coordinate labels - randomize position to match real-world variety
    label_choice = random.random()
    if label_choice > 0.4:  # 60% chance of having labels
        draw = ImageDraw.Draw(background)
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = None
        
        # File labels (a-h) at bottom
        for i, letter in enumerate('abcdefgh'):
            x = i * ts + ts - 12
            y = 512 - 16
            draw.text((x, y), letter, fill=(128, 128, 128, 180), font=font)
        
        # Rank labels on left (most common) or right
        if label_choice > 0.7:  # 30% on left
            x = 4
        else:  # 30% on right
            x = 512 - 14
        
        for i in range(8):
            y = i * ts + 4
            draw.text((x, y), str(8-i), fill=(128, 128, 128, 180), font=font)
    # 40% have no labels at all

    background = vandalize(background, grid)
    
    # Apply augmentation BEFORE compression
    background = augment_image(background)
    
    # JPEG compression artifacts are common in user screenshots.
    if random.random() < 0.60:
        buf = io.BytesIO()
        background.save(buf, "JPEG", quality=random.randint(45, 92))
        background = Image.open(buf).copy()

    # Simulate trimmed screenshot captures before tile slicing.
    background = simulate_trimmed_capture(background)

    tiles, labels = [], []
    for r in range(8):
        for c in range(8):
            tile = background.crop((c*ts, r*ts, (c+1)*ts, (r+1)*ts))
            tiles.append(np.array(tile, dtype=np.uint8).transpose(2,0,1))
            labels.append(FEN_CHARS.index(grid[r][c]) if grid[r][c] else 0)
    return tiles, labels

if __name__ == "__main__":
    for name in [f"val_{i}" for i in range(CHUNKS_VAL)] + [f"train_{i}" for i in range(CHUNKS_TRAIN)]:
        all_x, all_y = [], []
        for _ in range(BOARDS_PER_CHUNK):
            b = chess.Board()
            for _ in range(random.randint(5, 65)):
                if not b.is_game_over(): b.push(random.choice(list(b.legal_moves)))
            t, l = render_board(b.fen().split()[0])
            all_x.extend(t); all_y.extend(l)
        torch.save({'x': torch.from_numpy(np.stack(all_x)), 'y': torch.tensor(all_y)}, f"{OUTPUT_DIR}/{name}.pt")
        print(f"✅ Created {name}")
