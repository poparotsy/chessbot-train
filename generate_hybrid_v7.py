#!/usr/bin/env python3
"""Generate multi-head training tensors for v7."""

from __future__ import annotations

import io
import os
import random
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
BOARDS_PER_CHUNK = env_int("BOARDS_PER_CHUNK", 256)
CHUNKS_TRAIN = env_int("CHUNKS_TRAIN", 10)
CHUNKS_VAL = env_int("CHUNKS_VAL", 2)
MIN_PLIES = env_int("MIN_PLIES", 6)
MAX_PLIES = env_int("MAX_PLIES", 42)
BOARD_ABSENT_PROB = env_float("BOARD_ABSENT_PROB", 0.08)
SEED = env_int("SEED", 1337)
RECIPE_NAME = env_str("RECIPE_NAME", "boot_v1")

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


def make_sample() -> dict:
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
        family = choose_profile()
        severity = choose_severity()
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


def save_chunk(split: str, idx: int, count: int) -> None:
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
        sample = make_sample()
        # Encourage configured family mix for present boards by overriding family draw.
        if sample["board_present"] > 0.5:
            forced_family = plan[i]
            if sample["family"] != forced_family:
                board = random_training_board()
                pov_black = random.random() < 0.50
                board_img, piece_labels = render_board_image(board, pov_black=pov_black)
                sev = choose_severity()
                img, crn = synth_scene(board_img, family=forced_family, severity=sev)
                sample["x"] = np.array(img, dtype=np.uint8).transpose(2, 0, 1)
                sample["piece"] = np.array(piece_labels, dtype=np.int64)
                sample["pov"] = np.int64(1 if pov_black else 0)
                sample["stm"] = np.int64(0 if board.turn == chess.WHITE else 1)
                crn[:, 0] /= float(max(1, IMG_SIZE - 1))
                crn[:, 1] /= float(max(1, IMG_SIZE - 1))
                sample["corners"] = np.clip(crn, 0.0, 1.0).reshape(-1).astype(np.float32)
                sample["family"] = forced_family
                sample["severity"] = sev

        xs.append(sample["x"])
        pieces.append(sample["piece"])
        piece_masks.append(sample["piece_mask"])
        corners.append(sample["corners"])
        board_present.append(sample["board_present"])
        pov.append(sample["pov"])
        stm.append(sample["stm"])
        family_counts[sample["family"]] = family_counts.get(sample["family"], 0) + 1
        severity_counts[sample["severity"]] = severity_counts.get(sample["severity"], 0) + 1

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
    print(f"Created {split}_{idx} | mix[{mix}] | severity[{sev}]")


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(
        f"Generating v7 tensors -> {OUTPUT_DIR} | IMG={IMG_SIZE} | "
        f"train_chunks={CHUNKS_TRAIN} val_chunks={CHUNKS_VAL} boards_per_chunk={BOARDS_PER_CHUNK}"
    )
    for i in range(CHUNKS_TRAIN):
        save_chunk("train", i, BOARDS_PER_CHUNK)
    for i in range(CHUNKS_VAL):
        save_chunk("val", i, BOARDS_PER_CHUNK)
    print("Done.")


if __name__ == "__main__":
    main()
