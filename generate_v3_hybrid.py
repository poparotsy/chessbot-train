import os
import time
import random
import chess
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps

# ============ CONFIG ============
IMG_SIZE, CHUNKS, BOARDS_PER_CHUNK = 64, 10, 1000
FEN_CHARS = "1RNBQKPrnbqkp"
PIECE_SETS = [
    'alpha',
    'cburnett',
    'merida',
    'california',
    'cardinal',
    'dubrovny',
    'gioco',
    'maestro']

# 1. EXPANDED THEMES (Including Newspaper/B&W)
THEMES = [
    ("#f0d9b5", "#b58863"),  # Wood
    ("#ffffff", "#769656"),  # Green
    ("#dee3e6", "#8ca2ad"),  # Marble
    ("#ffffff", "#999999"),  # Grayscale Light
    ("#cccccc", "#777777"),  # Grayscale Dark
    ("#f9f6f2", "#b8836f")  # Newspaper Sepia
]


def get_piece(p_set, p_name, size):
    path = os.path.join("piece_sets", p_set, f"{p_name}.png")
    return Image.open(path).convert("RGBA").resize(
        (size, size), Image.LANCZOS) if os.path.exists(path) else None


def render_board_hybrid(fen):
    # 2. THE 50/50 RULE (Clean vs Twitter Noise)
    is_noisy = random.random() > 0.5
    theme, ts = random.choice(THEMES), random.choice([64, 72, 80])
    img = Image.new("RGB", (8 * ts, 8 * ts), theme[0])
    draw = ImageDraw.Draw(img)

    # 3. NOISE: Highlights & Internal Coordinates
    hl_sqs = random.sample(range(64), 2) if (
        is_noisy and random.random() > 0.4) else []
    for r in range(8):
        for c in range(8):
            color = theme[0] if (r + c) % 2 == 0 else theme[1]
            if (r * 8 + c) in hl_sqs:
                color = random.choice(["#f7f769", "#64b5f6"])
            draw.rectangle([c * ts, r * ts, (c + 1) *
                           ts, (r + 1) * ts], fill=color)
            if is_noisy and random.random() > 0.9:
                draw.text(
                    (c * ts + 2,
                     r * ts + 2),
                    random.choice("abcdefgh123"),
                    fill="#444")

    # 4. PIECES: Mapping & Parsing
    p_map = {
        'P': 'wP',
        'N': 'wN',
        'B': 'wB',
        'R': 'wR',
        'Q': 'wQ',
        'K': 'wK',
        'p': 'bP',
        'n': 'bN',
        'b': 'bB',
        'r': 'bR',
        'q': 'bQ',
        'k': 'bK'}
    grid_rows = fen.split()[0].split('/')
    parsed_grid = []
    for row in grid_rows:
        r_list = []
        for c in row:
            if c.isdigit():
                r_list.extend([None] * int(c))
            else:
                r_list.append(c)
        parsed_grid.append(r_list)

    p_set = random.choice(PIECE_SETS)
    for r in range(8):
        for c in range(8):
            char = parsed_grid[r][c]
            if char:
                p_img = get_piece(p_set, p_map[char], ts)
                if p_img:
                    # Random Jitter (Translation)
                    off = (random.randint(-4, 4), random.randint(-4, 4)
                           ) if is_noisy else (0, 0)
                    img.paste(p_img, (c * ts + off[0], r * ts + off[1]), p_img)

    # 5. GLOBAL AUGMENTATION (The B&W Fix)
    if is_noisy:
        if random.random() > 0.5:
            img = ImageEnhance.Brightness(
                img).enhance(random.uniform(0.8, 1.2))

        # 25% chance of purely Black & White training board
        if random.random() > 0.75:
            img = ImageOps.grayscale(img).convert("RGB")

        if random.random() > 0.5:
            img = img.filter(
                ImageFilter.GaussianBlur(
                    radius=random.uniform(
                        0.1, 0.5)))

    # 6. TILE EXTRACTION (Save as CHW uint8)
    tiles, labels = [], []
    for r in range(8):
        for c in range(8):
            tile = img.crop((c * ts, r * ts, (c + 1) * ts,
                            (r + 1) * ts)).resize((IMG_SIZE, IMG_SIZE))
            tiles.append(
                np.transpose(
                    np.array(
                        tile, dtype=np.uint8), (2, 0, 1)))
            labels.append(
                FEN_CHARS.index(
                    parsed_grid[r][c]) if parsed_grid[r][c] else 0)
    return np.stack(tiles), np.array(labels, dtype=np.uint8)


def main():
    os.makedirs("tensors_v3", exist_ok=True)
    print(f"🚀 Generating {CHUNKS * BOARDS_PER_CHUNK:,} boards...")
    for i in range(CHUNKS):
        start = time.time()
        all_x, all_y = [], []
        for _ in range(BOARDS_PER_CHUNK):
            board = chess.Board()
            for _ in range(random.randint(5, 50)):
                if not board.is_game_over():
                    board.push(random.choice(list(board.legal_moves)))
            xb, yb = render_board_hybrid(board.fen())
            all_x.append(xb)
            all_y.append(yb)
        torch.save({'x': torch.from_numpy(np.concatenate(all_x)), 'y': torch.from_numpy(
            np.concatenate(all_y))}, f"tensors_v3/train_chunk_{i}.pt")
        print(f"✅ Chunk {i} saved | Time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
