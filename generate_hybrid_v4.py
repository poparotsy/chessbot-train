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

    radius = random.randint(8, 11)
    marker = random.choices(["ring", "!", "!!"], weights=[0.20, 0.45, 0.35], k=1)[0]

    if marker == "ring":
        draw.ellipse(
            [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            outline=(255, 0, 0, random.randint(140, 200)),
            width=3)
        return

    # Highlight badge for ! / !!
    fill = random.choice([(255, 219, 77, 215), (113, 227, 122, 215), (255, 168, 84, 215)])
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        fill=fill,
        outline=(30, 30, 30, 190),
        width=2)

    try:
        font_size = 16 if marker == "!" else 13
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Center text manually for broad PIL compatibility.
    bbox = draw.textbbox((0, 0), marker, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        (center_x - text_w / 2, center_y - text_h / 2 - 1),
        marker,
        fill=(20, 20, 20, 255),
        font=font)


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
