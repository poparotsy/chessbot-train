#!/usr/bin/env python3
"""Generate multi-head training tensors for v7."""

from __future__ import annotations

import io
import json
import math
import os
import random
import signal
import sys
from pathlib import Path

import chess
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


FEN_CHARS = "1PNBRQKpnbrqk"  # 13 classes
POV_UNKNOWN = 2
STM_UNKNOWN = 2


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw is not None else default


# ============ HUMAN CONFIG ============
IMG_SIZE = env_int("IMG_SIZE", 320)
BOARD_RENDER_SIZE = env_int("BOARD_RENDER_SIZE", 512)
CANVAS_W = env_int("CANVAS_W", 1280)
CANVAS_H = env_int("CANVAS_H", 720)
MIN_PLIES = env_int("MIN_PLIES", 6)
MAX_PLIES = env_int("MAX_PLIES", 42)
SEED = env_int("SEED", 1337)

# Stable default preset for Kaggle/local consistency.
DATASET_PRESETS = {
    "smoke_v1": {
        "BOARDS_PER_CHUNK": 16,
        "CHUNKS_TRAIN": 1,
        "CHUNKS_VAL": 1,
        "BOARD_ABSENT_PROB": 0.08,
        "RECIPE_NAME": "boot_v1",
    },
    "kaggle_safe_v1": {
        "BOARDS_PER_CHUNK": 640,
        "CHUNKS_TRAIN": 20,
        "CHUNKS_VAL": 5,
        "BOARD_ABSENT_PROB": 0.08,
        "RECIPE_NAME": "boot_v1",
    },
}
DATASET_PRESET = env_str("DATASET_PRESET", "kaggle_safe_v1")
if DATASET_PRESET not in DATASET_PRESETS:
    raise ValueError(
        f"Unknown DATASET_PRESET={DATASET_PRESET}. "
        f"Valid: {', '.join(sorted(DATASET_PRESETS))}"
    )
PRESET = DATASET_PRESETS[DATASET_PRESET]
BOARDS_PER_CHUNK = int(PRESET["BOARDS_PER_CHUNK"])
CHUNKS_TRAIN = int(PRESET["CHUNKS_TRAIN"])
CHUNKS_VAL = int(PRESET["CHUNKS_VAL"])
BOARD_ABSENT_PROB = float(PRESET["BOARD_ABSENT_PROB"])
RECIPE_NAME = str(PRESET["RECIPE_NAME"])

# ============ PATHS ============
BASE_DIR = Path(__file__).resolve().parent
BOARD_THEMES_DIR = BASE_DIR / "board_themes"
PIECE_SETS_DIR = BASE_DIR / "piece_sets"
OUTPUT_DIR = BASE_DIR / env_str("OUTPUT_DIR", "tensors_v7")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PROFILE_RECIPES = {
    "boot_v1": [
        ("clean", 0.55),
        ("banner_side", 0.16),
        ("footer_heavy", 0.12),
        ("trimmed", 0.08),
        ("mono_scan", 0.06),
        ("combo", 0.03),
    ],
    "stable_v1": [
        ("clean", 0.26),
        ("banner_side", 0.22),
        ("footer_heavy", 0.20),
        ("trimmed", 0.14),
        ("mono_scan", 0.10),
        ("combo", 0.08),
    ],
    "targeted_v1": [
        ("clean", 0.12),
        ("banner_side", 0.22),
        ("footer_heavy", 0.26),
        ("trimmed", 0.14),
        ("mono_scan", 0.10),
        ("combo", 0.16),
    ],
}
DEFAULT_PROFILE_WEIGHTS = PROFILE_RECIPES.get(RECIPE_NAME, PROFILE_RECIPES["stable_v1"])
SEVERITY_WEIGHTS_BY_RECIPE = {
    "boot_v1": [(1, 0.58), (2, 0.27), (3, 0.11), (4, 0.04)],
    "stable_v1": [(1, 0.35), (2, 0.30), (3, 0.22), (4, 0.13)],
    "targeted_v1": [(1, 0.22), (2, 0.33), (3, 0.27), (4, 0.18)],
}
SEVERITY_WEIGHTS = SEVERITY_WEIGHTS_BY_RECIPE.get(RECIPE_NAME, SEVERITY_WEIGHTS_BY_RECIPE["stable_v1"])
INTERRUPTED = False

# Board annotation augmentation (enabled by default for robustness)
ANNOTATION_BOARD_PROB = 0.60
ANNOTATION_MIN_ACTIONS = 1
ANNOTATION_MAX_ACTIONS = 3
ARROW_ACTION_PROB = 0.55
HIGHLIGHT_ACTION_PROB = 0.50
MARKER_ACTION_PROB = 0.45
LOGO_ACTION_PROB = 0.40
FULL_LOGO_PROB = 0.35
PIECE_OCCLUSION_PROB = 0.16
LOCAL_PIECE_TILT_PROB = 0.14
LOCAL_PIECE_TILT_MAX_DEG = 18.0


def _signal_handler(sig, frame):
    del sig, frame
    global INTERRUPTED
    INTERRUPTED = True
    print("\n⚠️ Interrupt received. Finishing current step then stopping...", flush=True)


def choose_profile() -> str:
    names = [n for n, _ in DEFAULT_PROFILE_WEIGHTS]
    weights = [w for _, w in DEFAULT_PROFILE_WEIGHTS]
    return random.choices(names, weights=weights, k=1)[0]


def choose_severity() -> int:
    levels = [lv for lv, _ in SEVERITY_WEIGHTS]
    weights = [w for _, w in SEVERITY_WEIGHTS]
    return int(random.choices(levels, weights=weights, k=1)[0])


def draw_board_labels(draw: ImageDraw.ImageDraw, ts: int, pov_black: bool) -> None:
    files = "hgfedcba" if pov_black else "abcdefgh"
    ranks = [str(i) for i in range(1, 9)] if pov_black else [str(8 - i) for i in range(8)]
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()

    edge = BOARD_RENDER_SIZE
    for i, letter in enumerate(files):
        x = i * ts + ts - 12
        y = edge - 16
        draw.text((x, y), letter, fill=(128, 128, 128, 180), font=font)

    x = 4 if random.random() < 0.5 else edge - 14
    for i, rank in enumerate(ranks):
        y = i * ts + 4
        draw.text((x, y), rank, fill=(128, 128, 128, 180), font=font)


def _draw_arrow(draw: ImageDraw.ImageDraw, start_square: tuple[int, int], end_square: tuple[int, int], color: tuple[int, int, int, int], ts: int) -> None:
    sx = start_square[1] * ts + ts // 2
    sy = start_square[0] * ts + ts // 2
    ex = end_square[1] * ts + ts // 2
    ey = end_square[0] * ts + ts // 2
    draw.line([sx, sy, ex, ey], fill=color, width=random.randint(8, 14))
    angle = math.atan2(ey - sy, ex - sx)
    ah = max(10, ts // 3)
    p1 = (ex, ey)
    p2 = (ex - ah * math.cos(angle - math.pi / 6), ey - ah * math.sin(angle - math.pi / 6))
    p3 = (ex - ah * math.cos(angle + math.pi / 6), ey - ah * math.sin(angle + math.pi / 6))
    draw.polygon([p1, p2, p3], fill=color)


def _draw_marker(draw: ImageDraw.ImageDraw, square_rc: tuple[int, int], ts: int) -> None:
    r, c = square_rc
    x0, y0 = c * ts, r * ts
    cx = x0 + random.randint(int(ts * 0.2), int(ts * 0.8))
    cy = y0 + random.randint(int(ts * 0.2), int(ts * 0.8))
    radius = random.randint(max(6, ts // 8), max(10, ts // 5))
    style = random.choice(("ring", "!", "!!"))
    if style == "ring":
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(255, 0, 0, 185), width=3)
        return
    badge = random.choice([(37, 176, 176, 230), (29, 166, 167, 230), (42, 183, 184, 225)])
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=badge, outline=(18, 120, 120, 170), width=1)
    count = 2 if style == "!!" else 1
    bw = max(2, radius // 3)
    bh = max(6, int(radius * 0.9))
    gap = max(2, bw // 2)
    total = count * bw + (count - 1) * gap
    sx = cx - total // 2
    sy = cy - int(radius * 0.55)
    for i in range(count):
        x1 = sx + i * (bw + gap)
        draw.rounded_rectangle([x1, sy, x1 + bw, sy + bh], radius=1, fill=(245, 245, 245, 255))
        dy = sy + bh + 2
        draw.rounded_rectangle([x1, dy, x1 + bw, dy + max(2, bw)], radius=1, fill=(245, 245, 245, 255))


def _draw_logo_overlay(img: Image.Image, draw: ImageDraw.ImageDraw, square_rc: tuple[int, int], ts: int) -> None:
    r, c = square_rc
    x0, y0 = c * ts, r * ts
    if random.random() < FULL_LOGO_PROB:
        # Full tile-ish logo with king silhouette + 3-cap wordmark
        bw = int(ts * random.uniform(0.78, 1.02))
        bh = int(ts * random.uniform(0.72, 0.98))
        bx0 = max(0, min(img.size[0] - bw, x0 + random.randint(-int(ts * 0.15), int(ts * 0.08))))
        by0 = max(0, min(img.size[1] - bh, y0 + random.randint(-int(ts * 0.04), int(ts * 0.14))))
        bx1, by1 = bx0 + bw, by0 + bh
        cx = (bx0 + bx1) // 2
        draw.rounded_rectangle(
            [bx0, by0, bx1, by1],
            radius=max(4, int(ts * 0.08)),
            fill=random.choice([(236, 233, 216, 210), (232, 228, 208, 220), (238, 236, 221, 200)]),
            outline=random.choice([(142, 132, 110, 160), (120, 112, 95, 150)]),
            width=1,
        )
        kcol = (24, 24, 24, random.randint(160, 230))
        kw = int(bw * 0.34)
        kh = int(bh * 0.36)
        kcy = by0 + int(bh * 0.56)
        draw.rounded_rectangle([cx - kw // 2, kcy - int(kh * 0.14), cx + kw // 2, kcy + kh // 2], radius=3, fill=kcol)
        draw.ellipse([cx - int(kw * 0.18), kcy - int(kh * 0.58), cx + int(kw * 0.18), kcy - int(kh * 0.20)], fill=kcol)
        cw = max(1, kw // 12)
        draw.rectangle([cx - cw, kcy - int(kh * 0.70), cx + cw, kcy - int(kh * 0.45)], fill=kcol)
        draw.rectangle([cx - int(cw * 2.6), kcy - int(kh * 0.58), cx + int(cw * 2.6), kcy - int(kh * 0.50)], fill=kcol)
        label = random.choice(("SLC", "S.C.C", "LCC", "ICC"))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(10, int(ts * 0.20)))
        except Exception:
            font = ImageFont.load_default()
        draw.text((cx - int(ts * 0.18), by1 - int(bh * 0.35)), label, fill=(18, 18, 18, random.randint(170, 235)), font=font)
        return

    # Small corner watermark logo
    cx = x0 + random.randint(int(ts * 0.20), int(ts * 0.80))
    cy = y0 + random.randint(int(ts * 0.20), int(ts * 0.80))
    half = max(8, int(ts * random.uniform(0.22, 0.34)))
    draw.ellipse([cx - half, cy - half, cx + half, cy + half], fill=(120, 120, 120, random.randint(70, 120)))
    sil = random.choice(("rook", "king"))
    scol = (236, 236, 236, random.randint(105, 150))
    bx = cx - int(half * 0.45)
    by = cy - int(half * 0.50)
    bw = int(half * 0.9)
    bh = int(half * 1.0)
    if sil == "rook":
        draw.rounded_rectangle([bx, by + int(bh * 0.28), bx + bw, by + bh], radius=2, fill=scol)
        tw = max(2, bw // 4)
        gap = max(1, tw // 3)
        for i in range(3):
            tx0 = bx + i * (tw + gap)
            draw.rectangle([tx0, by + int(bh * 0.08), tx0 + tw, by + int(bh * 0.32)], fill=scol)
    else:
        draw.rounded_rectangle([bx, by + int(bh * 0.23), bx + bw, by + bh], radius=2, fill=scol)
        draw.ellipse([cx - int(bw * 0.22), by + int(bh * 0.02), cx + int(bw * 0.22), by + int(bh * 0.3)], fill=scol)


def apply_board_annotations(img: Image.Image, rows: list[list[str]]) -> Image.Image:
    if random.random() >= ANNOTATION_BOARD_PROB:
        return img
    draw = ImageDraw.Draw(img, "RGBA")
    ts = img.size[0] // 8
    occupied = [(r, c) for r in range(8) for c in range(8) if rows[r][c] != "1"]
    empties = [(r, c) for r in range(8) for c in range(8) if rows[r][c] == "1"]
    actions = random.randint(ANNOTATION_MIN_ACTIONS, ANNOTATION_MAX_ACTIONS)
    for _ in range(actions):
        modes = []
        if random.random() < ARROW_ACTION_PROB:
            modes.append("arrow")
        if random.random() < HIGHLIGHT_ACTION_PROB:
            modes.append("highlight")
        if random.random() < MARKER_ACTION_PROB:
            modes.append("marker")
        if random.random() < LOGO_ACTION_PROB:
            modes.append("logo")
        if not modes:
            modes = ["highlight"]
        mode = random.choice(modes)
        if mode == "arrow":
            s = random.randint(0, 7), random.randint(0, 7)
            e = max(0, min(7, s[0] + random.randint(-3, 3))), max(0, min(7, s[1] + random.randint(-3, 3)))
            if s != e:
                col = random.choice([(0, 255, 0, 110), (255, 165, 0, 110), (255, 0, 0, 110), (0, 150, 255, 110)])
                _draw_arrow(draw, s, e, col, ts)
        elif mode == "highlight":
            r, c = random.randint(0, 7), random.randint(0, 7)
            col = random.choice([(0, 255, 0, 65), (255, 255, 0, 65), (255, 0, 0, 65), (0, 150, 255, 65)])
            draw.rectangle([c * ts, r * ts, (c + 1) * ts, (r + 1) * ts], fill=col)
        elif mode == "marker":
            sq = random.choice(occupied if occupied else [(random.randint(0, 7), random.randint(0, 7))])
            _draw_marker(draw, sq, ts)
        else:  # logo
            pool = empties if empties else [(random.randint(0, 7), random.randint(0, 7))]
            sq = random.choice(pool)
            _draw_logo_overlay(img, draw, sq, ts)
    return img


def apply_piece_occlusion_overlay(img: Image.Image, rows: list[list[str]]) -> Image.Image:
    if random.random() >= PIECE_OCCLUSION_PROB:
        return img
    out = img.copy()
    draw = ImageDraw.Draw(out, "RGBA")
    ts = out.size[0] // 8
    occupied = [(r, c) for r in range(8) for c in range(8) if rows[r][c] != "1"]
    if not occupied:
        return out
    for r, c in random.sample(occupied, k=min(len(occupied), random.randint(1, 3))):
        x0, y0 = c * ts, r * ts
        x1, y1 = x0 + ts, y0 + ts
        style = random.choice(("bar", "corner", "ellipse"))
        col = random.choice([(38, 38, 38, 110), (245, 245, 245, 95), (18, 140, 220, 85), (200, 30, 30, 80)])
        if style == "bar":
            h = random.randint(max(6, ts // 8), max(10, ts // 4))
            yb = random.randint(y0, max(y0, y1 - h))
            draw.rectangle([x0, yb, x1, yb + h], fill=col)
        elif style == "corner":
            w = random.randint(max(8, ts // 5), max(12, ts // 2))
            h = random.randint(max(8, ts // 5), max(12, ts // 2))
            cx = x0 if random.random() < 0.5 else x1 - w
            cy = y0 if random.random() < 0.5 else y1 - h
            draw.rectangle([cx, cy, cx + w, cy + h], fill=col)
        else:
            pad = random.randint(5, max(6, ts // 4))
            draw.ellipse([x0 + pad, y0 + pad, x1 - pad, y1 - pad], fill=col)
    return out


def apply_local_piece_tilt(img: Image.Image, rows: list[list[str]]) -> Image.Image:
    if random.random() >= LOCAL_PIECE_TILT_PROB:
        return img
    ts = img.size[0] // 8
    occupied = [(r, c) for r in range(8) for c in range(8) if rows[r][c] != "1"]
    if not occupied:
        return img
    out = img.copy()
    for r, c in random.sample(occupied, k=min(len(occupied), random.randint(1, 2))):
        x0, y0 = c * ts, r * ts
        tile = out.crop((x0, y0, x0 + ts, y0 + ts))
        arr = np.asarray(tile, dtype=np.uint8)
        fill = tuple(int(v) for v in np.median(arr.reshape(-1, 3), axis=0))
        angle = random.uniform(-LOCAL_PIECE_TILT_MAX_DEG, LOCAL_PIECE_TILT_MAX_DEG)
        tilted = tile.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill)
        out.paste(tilted, (x0, y0))
    return out


def _expand_fen_board(fen_board: str) -> list[list[str]]:
    rows = []
    for row in fen_board.split("/"):
        cells = []
        for ch in row:
            if ch.isdigit():
                cells.extend(["1"] * int(ch))
            else:
                cells.append(ch)
        rows.append(cells)
    return rows


def _flatten_labels(rows: list[list[str]]) -> list[int]:
    flat = []
    for r in range(8):
        for c in range(8):
            flat.append(FEN_CHARS.index(rows[r][c]))
    return flat


def _rotate_labels_180(labels: list[int]) -> list[int]:
    grid = [labels[i * 8 : (i + 1) * 8] for i in range(8)]
    grid = [list(reversed(row)) for row in reversed(grid)]
    out = []
    for row in grid:
        out.extend(row)
    return out


def random_training_board() -> chess.Board:
    board = chess.Board()
    plies = random.randint(MIN_PLIES, MAX_PLIES)
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(random.choice(moves))
    return board


def render_board_image(board: chess.Board, pov_black: bool) -> tuple[Image.Image, list[int]]:
    board_theme = random.choice(sorted([p.name for p in BOARD_THEMES_DIR.iterdir() if p.is_file()]))
    piece_set = random.choice(sorted([p.name for p in PIECE_SETS_DIR.iterdir() if p.is_dir()]))

    out = Image.open(BOARD_THEMES_DIR / board_theme).convert("RGB").resize((BOARD_RENDER_SIZE, BOARD_RENDER_SIZE))
    ts = BOARD_RENDER_SIZE // 8
    rows = _expand_fen_board(board.board_fen())
    piece_labels = _flatten_labels(rows)

    for r in range(8):
        for c in range(8):
            ch = rows[r][c]
            if ch == "1":
                continue
            fname = f"{'w' if ch.isupper() else 'b'}{ch.upper()}.png"
            piece = Image.open(PIECE_SETS_DIR / piece_set / fname).convert("RGBA").resize((ts, ts))
            out.paste(piece, (c * ts, r * ts), piece)

    if random.random() < 0.70:
        draw = ImageDraw.Draw(out)
        draw_board_labels(draw, ts, pov_black=False)

    out = apply_board_annotations(out, rows)
    out = apply_piece_occlusion_overlay(out, rows)
    out = apply_local_piece_tilt(out, rows)

    if pov_black:
        out = out.rotate(180, resample=Image.BICUBIC, expand=False)
        piece_labels = _rotate_labels_180(piece_labels)
    return out, piece_labels


def _draw_banner(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    fill = random.choice([(14, 30, 92), (16, 36, 86), (34, 34, 34)])
    draw.rectangle([x0, y0, x1, y1], fill=fill)
    for _ in range(26):
        yy = random.randint(y0, y1)
        draw.line([x0, yy, x1, yy], fill=random.choice([(68, 100, 180), (80, 120, 190), (60, 60, 60)]), width=1)
    draw.text((x0 + 16, y0 + 28), random.choice(("Chess", "Opening", "Strategy")), fill=(245, 245, 245))
    draw.text((x0 + 16, y0 + 70), random.choice(("Analysis", "Puzzle", "Trainer")), fill=(230, 230, 230))


def _draw_footer(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int) -> None:
    fill = random.choice([(240, 240, 240), (250, 250, 250), (228, 232, 238)])
    draw.rectangle([x0, y0, x1, y1], fill=fill)
    for _ in range(16):
        xx = random.randint(x0, x1)
        draw.line([xx, y0, xx, y1], fill=(190, 190, 190), width=1)
    draw.text((x0 + 12, y0 + 8), random.choice(("Move list", "Engine eval", "Analysis")), fill=(35, 35, 35))


def _clip_corners(corners: np.ndarray, w: int, h: int) -> np.ndarray:
    c = corners.copy()
    c[:, 0] = np.clip(c[:, 0], 0, w - 1)
    c[:, 1] = np.clip(c[:, 1], 0, h - 1)
    return c


def _apply_trim(canvas: Image.Image, corners: np.ndarray, severity: int) -> tuple[Image.Image, np.ndarray]:
    w, h = canvas.size
    base_ratio = 0.02 + 0.02 * severity
    max_x = int(w * base_ratio)
    max_y = int(h * base_ratio)
    min_x = int(np.min(corners[:, 0]))
    max_xc = int(np.max(corners[:, 0]))
    min_y = int(np.min(corners[:, 1]))
    max_yc = int(np.max(corners[:, 1]))

    left = random.randint(0, max(0, min(max_x, min_x - 6)))
    right = random.randint(0, max(0, min(max_x, (w - 1 - max_xc) - 6)))
    top = random.randint(0, max(0, min(max_y, min_y - 6)))
    bottom = random.randint(0, max(0, min(max_y, (h - 1 - max_yc) - 6)))

    x0, y0 = left, top
    x1, y1 = max(x0 + 16, w - right), max(y0 + 16, h - bottom)
    cropped = canvas.crop((x0, y0, x1, y1))
    cw, ch = cropped.size
    sx = w / float(cw)
    sy = h / float(ch)
    new_corners = corners.copy()
    new_corners[:, 0] = (new_corners[:, 0] - x0) * sx
    new_corners[:, 1] = (new_corners[:, 1] - y0) * sy
    return cropped.resize((w, h), Image.LANCZOS), _clip_corners(new_corners, w, h)


def _apply_perspective(canvas: Image.Image, corners: np.ndarray, severity: int) -> tuple[Image.Image, np.ndarray]:
    arr = np.array(canvas)
    h, w = arr.shape[:2]
    max_j = max(2, int(min(h, w) * (0.0025 + 0.0018 * severity)))
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src + np.float32(
        [
            [random.randint(-max_j, max_j), random.randint(-max_j, max_j)],
            [random.randint(-max_j, max_j), random.randint(-max_j, max_j)],
            [random.randint(-max_j, max_j), random.randint(-max_j, max_j)],
            [random.randint(-max_j, max_j), random.randint(-max_j, max_j)],
        ]
    )
    mat = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    pts = cv2.perspectiveTransform(corners.reshape(1, 4, 2).astype(np.float32), mat).reshape(4, 2)
    return Image.fromarray(warped), _clip_corners(pts, w, h)


def _apply_mono(canvas: Image.Image, severity: int) -> Image.Image:
    gray = canvas.convert("L")
    contrast = max(0.45, 0.95 - 0.12 * severity)
    gray = ImageEnhance.Contrast(gray).enhance(contrast)
    gray = ImageEnhance.Brightness(gray).enhance(random.uniform(0.92, 1.05))
    out = gray.convert("RGB")
    if severity >= 2:
        out = out.filter(ImageFilter.GaussianBlur(radius=0.25 + 0.25 * (severity - 1)))
    return out


def synth_scene(board_img: Image.Image, family: str, severity: int) -> tuple[Image.Image, np.ndarray]:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), random.choice([(242, 242, 242), (24, 30, 40), (230, 232, 238)]))
    draw = ImageDraw.Draw(canvas)

    if family == "clean":
        bw = int(CANVAS_W * random.uniform(0.78, 0.92))
        bh = int(CANVAS_H * random.uniform(0.82, 0.96))
        x = (CANVAS_W - bw) // 2
        y = (CANVAS_H - bh) // 2
    elif family == "banner_side":
        bw = int(CANVAS_W * random.uniform(0.56 - 0.05 * (severity - 1), 0.72 - 0.03 * (severity - 1)))
        bh = int(CANVAS_H * random.uniform(0.88, 1.0))
        if random.random() < 0.5:
            x = CANVAS_W - bw
            y = 0
            _draw_banner(draw, 0, 0, x, CANVAS_H)
        else:
            x = 0
            y = 0
            _draw_banner(draw, bw, 0, CANVAS_W, CANVAS_H)
    elif family == "footer_heavy":
        bw = int(CANVAS_W * random.uniform(0.82, 0.96))
        bh = int(CANVAS_H * random.uniform(0.56 - 0.05 * (severity - 1), 0.74 - 0.04 * (severity - 1)))
        x = (CANVAS_W - bw) // 2
        y = 0
        _draw_footer(draw, 0, y + bh, CANVAS_W, CANVAS_H)
    elif family == "trimmed":
        bw = int(CANVAS_W * random.uniform(0.84, 0.96))
        bh = int(CANVAS_H * random.uniform(0.84, 0.96))
        x = (CANVAS_W - bw) // 2
        y = (CANVAS_H - bh) // 2
    elif family == "mono_scan":
        bw = int(CANVAS_W * random.uniform(0.82, 0.96))
        bh = int(CANVAS_H * random.uniform(0.84, 0.98))
        x = (CANVAS_W - bw) // 2
        y = (CANVAS_H - bh) // 2
    elif family == "combo":
        bw = int(CANVAS_W * random.uniform(0.50 - 0.03 * (severity - 1), 0.66 - 0.02 * (severity - 1)))
        bh = int(CANVAS_H * random.uniform(0.62 - 0.04 * (severity - 1), 0.82 - 0.03 * (severity - 1)))
        if random.random() < 0.5:
            x = CANVAS_W - bw
            _draw_banner(draw, 0, 0, x, CANVAS_H)
        else:
            x = 0
            _draw_banner(draw, bw, 0, CANVAS_W, CANVAS_H)
        y = 0
        _draw_footer(draw, 0, bh, CANVAS_W, CANVAS_H)
    else:
        bw = int(CANVAS_W * 0.88)
        bh = int(CANVAS_H * 0.90)
        x = (CANVAS_W - bw) // 2
        y = (CANVAS_H - bh) // 2

    board_scaled = board_img.resize((bw, bh), Image.LANCZOS)
    canvas.paste(board_scaled, (x, y))
    corners = np.array([[x, y], [x + bw - 1, y], [x + bw - 1, y + bh - 1], [x, y + bh - 1]], dtype=np.float32)

    if family in {"mono_scan", "combo"}:
        canvas = _apply_mono(canvas, severity)

    if random.random() < (0.50 + 0.10 * max(0, severity - 1)):
        canvas, corners = _apply_trim(canvas, corners, severity=severity)

    canvas, corners = _apply_perspective(canvas, corners, severity=severity)

    if random.random() < 0.55:
        buf = io.BytesIO()
        canvas.save(buf, "JPEG", quality=random.randint(42, 92))
        canvas = Image.open(buf).convert("RGB")

    canvas = canvas.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    sx = IMG_SIZE / float(CANVAS_W)
    sy = IMG_SIZE / float(CANVAS_H)
    corners[:, 0] *= sx
    corners[:, 1] *= sy
    corners = _clip_corners(corners, IMG_SIZE, IMG_SIZE)
    return canvas, corners


def synth_absent_scene() -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), random.choice([(18, 28, 56), (230, 230, 230), (32, 32, 32)]))
    draw = ImageDraw.Draw(canvas)
    if random.random() < 0.8:
        _draw_banner(draw, 0, 0, random.randint(220, 520), CANVAS_H)
    if random.random() < 0.8:
        _draw_footer(draw, 0, random.randint(int(CANVAS_H * 0.68), int(CANVAS_H * 0.86)), CANVAS_W, CANVAS_H)
    for _ in range(random.randint(10, 30)):
        x0 = random.randint(0, CANVAS_W - 60)
        y0 = random.randint(0, CANVAS_H - 40)
        x1 = min(CANVAS_W, x0 + random.randint(30, 220))
        y1 = min(CANVAS_H, y0 + random.randint(18, 90))
        draw.rectangle([x0, y0, x1, y1], fill=random.choice([(70, 90, 140), (180, 180, 180), (55, 55, 55)]))
    canvas = _apply_mono(canvas, severity=random.randint(1, 4)) if random.random() < 0.35 else canvas
    canvas, _ = _apply_perspective(canvas, np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32), severity=random.randint(1, 4))
    return canvas.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


def make_sample(forced_family: str | None = None, forced_severity: int | None = None) -> dict:
    if random.random() < BOARD_ABSENT_PROB:
        img = synth_absent_scene()
        piece_labels = [0] * 64
        corners = np.zeros((4, 2), dtype=np.float32)
        board_present = 0.0
        pov = POV_UNKNOWN
        stm = STM_UNKNOWN
        family = "absent"
        severity = 0
    else:
        board = random_training_board()
        pov_black = random.random() < 0.50
        board_img, piece_labels = render_board_image(board, pov_black=pov_black)
        family = forced_family if forced_family is not None else choose_profile()
        severity = int(forced_severity) if forced_severity is not None else choose_severity()
        img, corners = synth_scene(board_img, family=family, severity=severity)
        board_present = 1.0
        pov = 1 if pov_black else 0
        stm = 0 if board.turn == chess.WHITE else 1

    x = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
    corners_norm = corners.copy()
    corners_norm[:, 0] /= float(max(1, IMG_SIZE - 1))
    corners_norm[:, 1] /= float(max(1, IMG_SIZE - 1))
    corners_norm = np.clip(corners_norm, 0.0, 1.0).reshape(-1)
    return {
        "x": x,
        "piece": np.array(piece_labels, dtype=np.int64),
        "piece_mask": np.float32(board_present),
        "corners": corners_norm.astype(np.float32),
        "board_present": np.float32(board_present),
        "pov": np.int64(pov),
        "stm": np.int64(stm),
        "family": family,
        "severity": severity,
    }


def build_profile_plan(total: int) -> list[str]:
    raw = [(name, w * total) for name, w in DEFAULT_PROFILE_WEIGHTS]
    floored = [(name, int(v)) for name, v in raw]
    used = sum(v for _, v in floored)
    rem = total - used
    remainders = sorted(((name, rawv - int(rawv)) for name, rawv in raw), key=lambda p: p[1], reverse=True)
    counts = {name: n for name, n in floored}
    for i in range(max(0, rem)):
        counts[remainders[i % len(remainders)][0]] += 1
    plan = []
    for name, _ in DEFAULT_PROFILE_WEIGHTS:
        plan.extend([name] * counts[name])
    random.shuffle(plan)
    return plan


def save_chunk(split: str, idx: int, count: int) -> bool:
    xs = []
    pieces = []
    piece_masks = []
    corners = []
    board_present = []
    pov = []
    stm = []
    family_counts: dict[str, int] = {}
    severity_counts: dict[int, int] = {}

    plan = build_profile_plan(count)
    for i in range(count):
        if INTERRUPTED:
            break
        sample = make_sample(forced_family=plan[i])

        xs.append(sample["x"])
        pieces.append(sample["piece"])
        piece_masks.append(sample["piece_mask"])
        corners.append(sample["corners"])
        board_present.append(sample["board_present"])
        pov.append(sample["pov"])
        stm.append(sample["stm"])
        family_counts[sample["family"]] = family_counts.get(sample["family"], 0) + 1
        severity_counts[sample["severity"]] = severity_counts.get(sample["severity"], 0) + 1

    if not xs:
        print(f"Stopped before writing {split}_{idx} (no samples).")
        return False

    payload = {
        "x": torch.from_numpy(np.stack(xs, axis=0)),
        "piece": torch.from_numpy(np.stack(pieces, axis=0)),
        "piece_mask": torch.from_numpy(np.array(piece_masks, dtype=np.float32)),
        "corners": torch.from_numpy(np.stack(corners, axis=0)),
        "board_present": torch.from_numpy(np.array(board_present, dtype=np.float32)),
        "pov": torch.from_numpy(np.array(pov, dtype=np.int64)),
        "stm": torch.from_numpy(np.array(stm, dtype=np.int64)),
    }
    out_path = OUTPUT_DIR / f"{split}_{idx}.pt"
    torch.save(payload, out_path)

    mix = ", ".join(f"{k}:{family_counts[k]}" for k in sorted(family_counts))
    sev = ", ".join(f"L{k}:{severity_counts[k]}" for k in sorted(severity_counts))
    print(f"Created {split}_{idx} | n={len(xs)} | mix[{mix}] | severity[{sev}]")
    return not INTERRUPTED


def main() -> None:
    signal.signal(signal.SIGINT, _signal_handler)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    total_samples = BOARDS_PER_CHUNK * (CHUNKS_TRAIN + CHUNKS_VAL)
    # rough lower-bound estimate (x tensor only + light metadata overhead)
    bytes_per_sample = (3 * IMG_SIZE * IMG_SIZE) + 320
    est_gb = (total_samples * bytes_per_sample) / (1024 ** 3)
    manifest = {
        "dataset_preset": DATASET_PRESET,
        "img_size": IMG_SIZE,
        "board_render_size": BOARD_RENDER_SIZE,
        "canvas_w": CANVAS_W,
        "canvas_h": CANVAS_H,
        "boards_per_chunk": BOARDS_PER_CHUNK,
        "chunks_train": CHUNKS_TRAIN,
        "chunks_val": CHUNKS_VAL,
        "min_plies": MIN_PLIES,
        "max_plies": MAX_PLIES,
        "board_absent_prob": BOARD_ABSENT_PROB,
        "seed": SEED,
        "recipe_name": RECIPE_NAME,
        "total_samples": total_samples,
        "estimated_disk_gb": round(est_gb, 3),
    }
    (OUTPUT_DIR / "generation_manifest_v7.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        f"Generating v7 tensors -> {OUTPUT_DIR} | IMG={IMG_SIZE} | "
        f"train_chunks={CHUNKS_TRAIN} val_chunks={CHUNKS_VAL} boards_per_chunk={BOARDS_PER_CHUNK} "
        f"| preset={DATASET_PRESET} | total_samples≈{total_samples} | est_disk≈{est_gb:.2f}GB (+torch overhead)"
    )
    try:
        for i in range(CHUNKS_TRAIN):
            if not save_chunk("train", i, BOARDS_PER_CHUNK):
                break
        if not INTERRUPTED:
            for i in range(CHUNKS_VAL):
                if not save_chunk("val", i, BOARDS_PER_CHUNK):
                    break
    except KeyboardInterrupt:
        print("\n⚠️ KeyboardInterrupt: stopping generation.")
    if INTERRUPTED:
        print("Stopped by user interrupt.")
        sys.exit(130)
    print("Done.")


if __name__ == "__main__":
    main()
