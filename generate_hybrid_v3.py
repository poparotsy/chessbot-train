import os, io, random, chess, torch, numpy as np
from PIL import Image, ImageDraw, ImageFilter

# THE GLOBAL LABEL LAW (Aligned with audit_dataset.py)
FEN_CHARS = "1PNBRQKpnbrqk" 
# 0:1, 1:P, 2:N, 3:B, 4:R, 5:Q, 6:K, 7:p, 8:n, 9:b, 10:r, 11:q, 12:k

IMG_SIZE = 64
BOARDS_PER_CHUNK = 1000
CHUNKS_TRAIN, CHUNKS_VAL = 10, 2
OUTPUT_DIR = "tensors_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def vandalize(img):
    draw = ImageDraw.Draw(img, "RGBA")
    w = img.size[0]
    for _ in range(random.randint(0, 5)):
        ts = w // 8
        r, c = random.randint(0, 7), random.randint(0, 7)
        color = random.choice([(0,255,0,100), (255,255,0,100), (255,0,0,100)])
        draw.rectangle([c*ts, r*ts, (c+1)*ts, (r+1)*ts], fill=color)
    for _ in range(random.randint(0, 3)):
        c = random.choice([(0,0,255,160), (255,165,0,160), (0,255,0,160)])
        draw.line([random.randint(0,w) for _ in range(4)], fill=c, width=random.randint(4,12))
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

    background = vandalize(background)
    if random.random() > 0.5:
        buf = io.BytesIO(); background.save(buf, "JPEG", quality=random.randint(30, 80))
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

