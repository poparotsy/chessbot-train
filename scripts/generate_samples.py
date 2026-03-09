#!/usr/bin/env python3
"""Generate visual sample boards using the v5 training distribution."""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generate_hybrid_v5 as gen5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sample board images with the v5 training distribution.")
    parser.add_argument("--count", type=int, default=12, help="Number of samples to generate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "sample_boards_v5",
        help="Directory to save sample images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--profile",
        choices=("default", "screenshot-clutter", "edge-rook", "hard-mix"),
        default="default",
        help="Sampling profile to preview the training distribution.")
    return parser.parse_args()


def render_board_image(fen_board, profile=None):
    tiles, _labels, meta = gen5.base.render_board(fen_board, return_meta=True, profile=profile)
    rows = []
    for row_idx in range(8):
        row_tiles = []
        for col_idx in range(8):
            tile = tiles[row_idx * 8 + col_idx].transpose(1, 2, 0)
            row_tiles.append(tile)
        rows.append(np.concatenate(row_tiles, axis=1))
    background = Image.fromarray(np.concatenate(rows, axis=0))
    return background, meta["board_theme"], meta["piece_set"], meta.get("label_pov")


def apply_profile(profile):
    mapping = {
        "default": None,
        "screenshot-clutter": "screenshot_clutter",
        "edge-rook": "edge_rook",
        "hard-mix": "hard_mix",
    }
    return mapping[profile]


def main():
    args = parse_args()
    random.seed(args.seed)
    profile = apply_profile(args.profile)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    print(f"Generating {args.count} v5 sample boards into: {args.output_dir}")
    print(
        "Profile="
        f"{args.profile} "
        f"(mode={'mixed-default' if profile is None else profile})"
    )
    for i in range(args.count):
        fen_board = gen5.base.random_training_board(profile=profile).board_fen()
        image, board_theme, piece_set, label_pov = render_board_image(fen_board, profile=profile)
        out_path = args.output_dir / f"sample_v5_{i + 1:03d}.png"
        image.save(out_path)
        manifest.append(
            {
                "file": out_path.name,
                "fen_board": fen_board,
                "board_theme": board_theme,
                "piece_set": piece_set,
                "profile": args.profile,
                "label_pov": label_pov,
            })
        print(
            f"  ✅ {out_path.name} | theme={board_theme} | pieces={piece_set} | "
            f"labels={label_pov or 'none'}"
        )

    manifest_path = args.output_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(
                f"{item['file']} | fen={item['fen_board']} | theme={item['board_theme']} | "
                f"pieces={item['piece_set']} | profile={item['profile']} | labels={item['label_pov']}\n")

    print(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
