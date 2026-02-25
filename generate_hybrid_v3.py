import os, io, random, chess, torch, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

# THE GLOBAL LABEL LAW (Aligned with audit_dataset.py)
FEN_CHARS = "1PNBRQKpnbrqk" 
# 0:1, 1:P, 2:N, 3:B, 4:R, 5:Q, 6:K, 7:p, 8:n, 9:b, 10:r, 11:q, 12:k

IMG_SIZE = 64
BOARDS_PER_CHUNK = 1000
CHUNKS_TRAIN, CHUNKS_VAL = 10, 2
OUTPUT_DIR = "tensors_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def augment_image(img):
    """Balanced augmentation: variety without destroying visibility"""
    # Grayscale conversion (25% chance) - for B&W diagrams
    if random.random() > 0.75:
        img = img.convert('L').convert('RGB')  # Convert to grayscale but keep 3 channels
    
    # Brightness - wider range
    if random.random() > 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
    
    # Contrast - wider range
    if random.random() > 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    
    # Color saturation - wider range (only if not grayscale)
    if random.random() > 0.5:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.15))
    
    # Gaussian noise - moderate
    if random.random() > 0.5:
        arr = np.array(img)
        noise = np.random.normal(0, random.randint(5, 12), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    
    # Light blur occasionally
    if random.random() > 0.8:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
    
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

def vandalize(img):
    """Add arrows/highlights like real chess sites - but keep pieces visible"""
    draw = ImageDraw.Draw(img, "RGBA")
    w = img.size[0]
    ts = w // 8
    
    # Square highlights (1-3 squares) - moderate opacity
    for _ in range(random.randint(1, 3)):
        r, c = random.randint(0, 7), random.randint(0, 7)
        # Opacity 70 - visible but not overwhelming
        color = random.choice([(0,255,0,70), (255,255,0,70), (255,0,0,70), (0,150,255,70)])
        draw.rectangle([c*ts, r*ts, (c+1)*ts, (r+1)*ts], fill=color)
    
    # Proper chess arrows (1-3) - with arrowheads - ALWAYS have at least 1
    for _ in range(random.randint(1, 3)):
        start_r, start_c = random.randint(0, 7), random.randint(0, 7)
        # Arrow goes 1-3 squares away
        end_r = max(0, min(7, start_r + random.randint(-3, 3)))
        end_c = max(0, min(7, start_c + random.randint(-3, 3)))
        
        if (start_r, start_c) != (end_r, end_c):  # Don't draw arrow to same square
            color = random.choice([(0,255,0,120), (255,165,0,120), (255,0,0,120), (0,150,255,120)])
            draw_arrow(draw, (start_r, start_c), (end_r, end_c), color, ts)
    
    # Red circles (0-1) - like puzzle highlights
    if random.random() > 0.8:
        r, c = random.randint(0, 7), random.randint(0, 7)
        center_x, center_y = c*ts + ts//2, r*ts + ts//2
        radius = ts // 2 - 5
        draw.ellipse([center_x-radius, center_y-radius, center_x+radius, center_y+radius], 
                     outline=(255,0,0,150), width=4)
    
    return img

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

    # Add coordinate labels (30% chance) - like real chess boards
    if random.random() > 0.7:
        draw = ImageDraw.Draw(background)
        # Try to use a font, fallback to default if not available
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
        
        # Rank labels (1-8) on right
        for i in range(8):
            x = 512 - 14
            y = i * ts + 4
            draw.text((x, y), str(8-i), fill=(128, 128, 128, 180), font=font)

    background = vandalize(background)
    
    # Apply augmentation BEFORE compression
    background = augment_image(background)
    
    # JPEG compression - wider range for variety
    if random.random() > 0.3:
        buf = io.BytesIO()
        background.save(buf, "JPEG", quality=random.randint(30, 90))
        background = Image.open(buf).copy()

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

