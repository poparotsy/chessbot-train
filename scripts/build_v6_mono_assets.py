#!/usr/bin/env python3
"""Generate dedicated v6 mono/print board themes and piece sets."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
BOARD_THEMES_DIR = ROOT / "board_themes"
PIECE_SETS_DIR = ROOT / "piece_sets"
SOURCE_SET = PIECE_SETS_DIR / "cburnett"
PIECE_NAMES = ["P", "N", "B", "R", "Q", "K"]
SEED = 1337


def rng(seed_offset: int) -> random.Random:
    return random.Random(SEED + seed_offset)


def build_noise(width: int, height: int, side: int, low: float, high: float, blur: float) -> np.ndarray:
    small_h = max(4, height // side)
    small_w = max(4, width // side)
    arr = np.random.uniform(low, high, (small_h, small_w)).astype(np.float32)
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "L")
    img = img.resize((width, height), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur))
    return np.asarray(img, dtype=np.float32) / 255.0


def save_board_theme(name: str, dark: int, light: int, paper: tuple[float, float], stripes: bool = False) -> None:
    size = 512
    tile = size // 8
    board = np.zeros((size, size), dtype=np.float32)
    for row in range(8):
        for col in range(8):
            val = light if (row + col) % 2 == 0 else dark
            board[row * tile:(row + 1) * tile, col * tile:(col + 1) * tile] = float(val)

    texture = build_noise(size, size, side=20, low=paper[0], high=paper[1], blur=1.0)
    fibers_x = build_noise(size, size, side=72, low=0.96, high=1.04, blur=0.8)
    fibers_y = build_noise(size, size, side=72, low=0.96, high=1.04, blur=0.8).T
    fibers_y = fibers_y[:size, :size]
    board = board * texture + (fibers_x - 1.0) * 20.0 + (fibers_y - 1.0) * 16.0

    if stripes:
        for y in range(0, size, 6):
            board[y:y + 1, :] -= np.random.uniform(2.0, 6.0)

    board = np.clip(board, 0, 255).astype(np.uint8)
    out = Image.fromarray(board, "L").convert("RGB")
    out.save(BOARD_THEMES_DIR / name)


def degrade_alpha(alpha: Image.Image, style: str, edge_bias: bool = False) -> Image.Image:
    r = rng(10 if style == "scan" else 20)
    down = alpha.resize((18, 18), Image.BICUBIC).resize((64, 64), Image.BICUBIC)
    arr = np.asarray(down, dtype=np.float32) / 255.0

    damage = build_noise(64, 64, side=12, low=0.45, high=1.0, blur=0.6)
    arr *= damage

    if edge_bias:
        grad = np.tile(np.linspace(0.55, 1.0, 64, dtype=np.float32), (64, 1))
        arr *= grad

    for _ in range(3 if style == "faded" else 2):
        y = r.randrange(64)
        x = r.randrange(64)
        h = r.randrange(4, 12)
        w = r.randrange(4, 12)
        arr[max(0, y - h):min(64, y + h), max(0, x - w):min(64, x + w)] *= r.uniform(0.0, 0.5)

    out = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), "L")
    out = out.filter(ImageFilter.GaussianBlur(radius=0.7 if style == "faded" else 0.45))
    return out


def render_piece(alpha: Image.Image, piece_color: str, style: str) -> Image.Image:
    edge_bias = piece_color == "b" and style == "faded"
    alpha = degrade_alpha(alpha, style=style, edge_bias=edge_bias)
    alpha_arr = np.asarray(alpha, dtype=np.float32) / 255.0
    outline = alpha.filter(ImageFilter.MaxFilter(3))
    outline_arr = np.asarray(outline, dtype=np.float32) / 255.0
    outline_arr = np.clip(outline_arr - alpha_arr, 0.0, 1.0)

    if piece_color == "w":
        fill = 220 if style == "scan" else 198
        line = 92 if style == "scan" else 82
    else:
        fill = 34 if style == "scan" else 56
        line = 98 if style == "scan" else 112

    canvas = np.zeros((64, 64, 4), dtype=np.uint8)
    base = alpha_arr
    if piece_color == "w":
        rgb = np.clip(fill * base + line * outline_arr, 0, 255)
    else:
        rgb = np.clip(fill * base + line * outline_arr * 0.35, 0, 255)

    if style == "faded":
        rgb = rgb * build_noise(64, 64, side=18, low=0.84, high=1.08, blur=0.5)

    canvas[:, :, 0] = np.clip(rgb, 0, 255).astype(np.uint8)
    canvas[:, :, 1] = np.clip(rgb, 0, 255).astype(np.uint8)
    canvas[:, :, 2] = np.clip(rgb, 0, 255).astype(np.uint8)
    canvas[:, :, 3] = np.clip((alpha_arr * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(canvas, "RGBA")


def build_piece_set(name: str, style: str) -> None:
    out_dir = PIECE_SETS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    for color in ("w", "b"):
        for piece in PIECE_NAMES:
            src = Image.open(SOURCE_SET / f"{color}{piece}.png").convert("RGBA").resize((64, 64), Image.LANCZOS)
            alpha = src.getchannel("A")
            rendered = render_piece(alpha, color, style=style)
            rendered.save(out_dir / f"{color}{piece}.png")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)

    save_board_theme("mono_paper_scan_light.png", dark=140, light=172, paper=(0.90, 1.10), stripes=True)
    save_board_theme("mono_paper_scan_mid.png", dark=118, light=150, paper=(0.88, 1.08), stripes=True)
    save_board_theme("mono_heather_print.png", dark=92, light=118, paper=(0.92, 1.06), stripes=False)

    build_piece_set("mono_print_scan", style="scan")
    build_piece_set("mono_print_faded", style="faded")

    print("Generated mono/print v6 assets:")
    print("  board_themes/mono_paper_scan_light.png")
    print("  board_themes/mono_paper_scan_mid.png")
    print("  board_themes/mono_heather_print.png")
    print("  piece_sets/mono_print_scan/")
    print("  piece_sets/mono_print_faded/")


if __name__ == "__main__":
    main()
