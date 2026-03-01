#!/usr/bin/env python3
"""Generate visual sample boards using v4 augmentation pipeline."""

import argparse
import io
import random
from pathlib import Path

import chess
from PIL import Image

import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generate_hybrid_v4 as gen4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sample board images with v4 data augmentations.")
    parser.add_argument("--count", type=int, default=12, help="Number of samples to generate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "sample_boards_v4",
        help="Directory to save sample images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--min-plies", type=int, default=10, help="Min random plies per board.")
    parser.add_argument("--max-plies", type=int, default=40, help="Max random plies per board.")
    return parser.parse_args()


def board_to_grid(fen_board):
    grid = [[None] * 8 for _ in range(8)]
    for r, row_str in enumerate(fen_board.split("/")):
        c = 0
        for char in row_str:
            if char.isdigit():
                c += int(char)
            else:
                grid[r][c] = char
                c += 1
    return grid


def render_board_image(fen_board):
    board_theme = random.choice(list((ROOT_DIR / "board_themes").iterdir())).name
    piece_set = random.choice(list((ROOT_DIR / "piece_sets").iterdir())).name

    background = Image.open(ROOT_DIR / "board_themes" / board_theme).convert("RGB").resize((512, 512))
    grid = board_to_grid(fen_board)
    tile_size = 64

    for r in range(8):
        for c in range(8):
            char = grid[r][c]
            if not char:
                continue
            piece_name = f"{'w' if char.isupper() else 'b'}{char.upper()}.png"
            piece = Image.open(ROOT_DIR / "piece_sets" / piece_set / piece_name).convert("RGBA").resize((tile_size, tile_size))
            background.paste(piece, (c * tile_size, r * tile_size), piece)

    background = gen4.vandalize(background)
    background = gen4.augment_image(background)

    if random.random() > 0.3:
        buf = io.BytesIO()
        background.save(buf, "JPEG", quality=random.randint(30, 90))
        background = Image.open(buf).copy()

    return background, board_theme, piece_set


def random_position(min_plies, max_plies):
    board = chess.Board()
    for _ in range(random.randint(min_plies, max_plies)):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board.fen().split()[0]


def main():
    args = parse_args()
    random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    print(f"Generating {args.count} v4 sample boards into: {args.output_dir}")
    for i in range(args.count):
        fen_board = random_position(args.min_plies, args.max_plies)
        image, board_theme, piece_set = render_board_image(fen_board)
        out_path = args.output_dir / f"sample_v4_{i + 1:03d}.png"
        image.save(out_path)
        manifest.append(
            {
                "file": out_path.name,
                "fen_board": fen_board,
                "board_theme": board_theme,
                "piece_set": piece_set,
            })
        print(f"  ✅ {out_path.name} | theme={board_theme} | pieces={piece_set}")

    manifest_path = args.output_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(
                f"{item['file']} | fen={item['fen_board']} | theme={item['board_theme']} | pieces={item['piece_set']}\n")

    print(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
