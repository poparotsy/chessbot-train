"""
🎨 V3 BEAST GENERATOR - PRO VERSION
Generates 64x64 RGB Chess Tiles with real-world noise.
Supports RGB themes and B&W Newspaper themes.
"""

import os
import time
import random

# pylint: disable=import-error
import chess
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps

# ============ CONFIG ============
IMG_SIZE = 64
CHUNKS_TRAIN = 10
BOARDS_PER_CHUNK = 1500
FEN_CHARS = "1RNBQKPrnbqkp"

THEMES = [
    ("#f0d9b5", "#b58863"), ("#ffffff", "#769656"), ("#dee3e6", "#8ca2ad"),
    ("#ffffff", "#999999"), ("#f0e9c5", "#c5a55a"), ("#faf0d7", "#8b6343")
]
HIGHLIGHTS = ["#f7f769", "#b9ca43", "#64b5f6", "#f44336", None]
PIECE_SETS = [
    'alpha', 'cburnett', 'merida', 'california',
    'cardinal', 'dubrovny', 'gioco', 'maestro'
]


def get_piece_image(p_set, p_name, size):
    """Loads and resizes a chess piece image."""
    path = os.path.join("piece_sets", p_set, f"{p_name}.png")
    if os.path.exists(path):
        img = Image.open(path).convert("RGBA")
        return img.resize((size, size), Image.LANCZOS)
    return None


def apply_jitter(img):
    """Applies random brightness, contrast, and B&W conversion."""
    if random.random() > 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    if random.random() > 0.5:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    # THE GRAYSCALE FIX: 25% chance to make the board grayscale (Newspaper mode)
    if random.random() > 0.75:
        img = ImageOps.grayscale(img).convert("RGB")
    return img


def paste_pieces(img, fen_grid, p_set, tile_size):
    """Pastes pieces onto the board image."""
    p_map = {
        'P': 'wP', 'N': 'wN', 'B': 'wB', 'R': 'wR', 'Q': 'wQ', 'K': 'wK',
        'p': 'bP', 'n': 'bN', 'b': 'bB', 'r': 'bR', 'q': 'bQ', 'k': 'bK'
    }
    for r in range(8):
        for c in range(8):
            char = fen_grid[r][c]
            if char in p_map:
                p_img = get_piece_image(p_set, p_map[char], tile_size)
                if p_img:
                    off_x, off_y = random.randint(-2, 2), random.randint(-2, 2)
                    img.paste(p_img, (c * tile_size + off_x, r * tile_size + off_y), p_img)


def _draw_grid(draw, theme, tile_size):
    """Draws the 8x8 squares and highlights."""
    hl_sqs = random.sample(range(64), 2) if random.random() > 0.4 else []
    hl_color = random.choice(HIGHLIGHTS)
    for r in range(8):
        for c in range(8):
            color = theme[0] if (r + c) % 2 == 0 else theme[1]
            if (r * 8 + c) in hl_sqs and hl_color:
                color = hl_color
            x_0, y_0 = c * tile_size, r * tile_size
            x_1, y_1 = x_0 + tile_size, y_0 + tile_size
            draw.rectangle([x_0, y_0, x_1, y_1], fill=color)


def _parse_fen(fen):
    """Parses FEN string into an 8x8 grid."""
    grid = []
    for row in fen.split()[0].split('/'):
        r_list = []
        for char in row:
            if char.isdigit():
                r_list.extend([None] * int(char))
            else:
                r_list.append(char)
        grid.append(r_list)
    return grid


def render_board_v3(fen):
    """Renders a full board and returns 64 extracted tiles."""
    theme, ts = random.choice(THEMES), random.choice([64, 72, 80])
    img = Image.new("RGB", (8 * ts, 8 * ts), theme[0])
    draw = ImageDraw.Draw(img)

    _draw_grid(draw, theme, ts)
    grid = _parse_fen(fen)

    paste_pieces(img, grid, random.choice(PIECE_SETS), ts)
    img = apply_jitter(img)
    if random.random() > 0.5:
        radius = random.uniform(0.1, 0.6)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    tiles, labels = [], []
    for r in range(8):
        for c in range(8):
            tile = img.crop((c*ts, r*ts, (c+1)*ts, (r+1)*ts)).resize((IMG_SIZE, IMG_SIZE))
            tiles.append(np.transpose(np.array(tile, dtype=np.uint8), (2, 0, 1)))
            label = FEN_CHARS.index(grid[r][c]) if grid[r][c] else 0
            labels.append(label)
    return np.stack(tiles), np.array(labels, dtype=np.uint8)


def main():
    """Main generation loop."""
    os.makedirs("tensors_v3", exist_ok=True)
    for i in range(CHUNKS_TRAIN):
        start = time.time()
        all_x, all_y = [], []
        for _ in range(BOARDS_PER_CHUNK):
            board = chess.Board()
            for _ in range(random.randint(5, 60)):
                if not board.is_game_over():
                    board.push(random.choice(list(board.legal_moves)))
            x_b, y_b = render_board_v3(board.fen())
            all_x.append(x_b)
            all_y.append(y_b)
        torch.save({
            'x': torch.from_numpy(np.concatenate(all_x)),
            'y': torch.from_numpy(np.concatenate(all_y))
        }, f"tensors_v3/train_chunk_{i}.pt")
        print(f"✅ Chunk {i} | Time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
