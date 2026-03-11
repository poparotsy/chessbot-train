#!/usr/bin/env python3
"""Generate boundary stress test images for board/corner detection."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import chess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import generate_hybrid_v5 as gen  # noqa: E402


CANVAS_W = 1920
CANVAS_H = 1080
BOARD_SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate boundary stress suite for recognizer geometry.")
    parser.add_argument(
        "--output-dir",
        default=str(TRAIN_DIR / "images_boundary_v6"),
        help="Where boundary images will be written.",
    )
    parser.add_argument(
        "--truth-json",
        default=str(TRAIN_DIR / "images_boundary_v6" / "truth_boundary_v6.json"),
        help="Truth output JSON mapping image->full FEN.",
    )
    parser.add_argument(
        "--manifest-json",
        default=str(TRAIN_DIR / "images_boundary_v6" / "manifest_boundary_v6.json"),
        help="Manifest output JSON with family/severity metadata.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument(
        "--boards",
        type=int,
        default=12,
        help="Unique base boards to synthesize from.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=4,
        help="Difficulty levels per family.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=88,
        help="JPEG save quality.",
    )
    return parser.parse_args()


def build_seed_boards(count: int, rng: random.Random) -> list[str]:
    boards: list[str] = []
    for _ in range(count):
        b = chess.Board()
        plies = rng.randint(8, 34)
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        boards.append(b.board_fen())
    return boards


def _choose_theme_and_set(rng: random.Random) -> tuple[str, str]:
    theme = rng.choice(sorted([p.name for p in Path(gen.BOARD_THEMES_DIR).iterdir() if p.is_file()]))
    pset = rng.choice(sorted([p.name for p in Path(gen.PIECE_SETS_DIR).iterdir() if p.is_dir()]))
    return theme, pset


def render_base_board(fen_board: str, rng: random.Random) -> tuple[Image.Image, dict]:
    theme, pset = _choose_theme_and_set(rng)
    board = Image.open(Path(gen.BOARD_THEMES_DIR) / theme).convert("RGB").resize((BOARD_SIZE, BOARD_SIZE))
    ts = BOARD_SIZE // 8

    rows = []
    for row in fen_board.split("/"):
        cells = []
        for ch in row:
            if ch.isdigit():
                cells.extend(["."] * int(ch))
            else:
                cells.append(ch)
        rows.append(cells)

    for r in range(8):
        for c in range(8):
            ch = rows[r][c]
            if ch == ".":
                continue
            fname = f"{'w' if ch.isupper() else 'b'}{ch.upper()}.png"
            piece = Image.open(Path(gen.PIECE_SETS_DIR) / pset / fname).convert("RGBA").resize((ts, ts))
            board.paste(piece, (c * ts, r * ts), piece)

    draw = ImageDraw.Draw(board)
    if rng.random() < 0.65:
        pov = rng.choice(("white", "black"))
        gen.draw_board_labels(draw, ts, pov)
    else:
        pov = None

    meta = {"theme": theme, "piece_set": pset, "label_pov": pov}
    return board, meta


def _add_banner(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, rng: random.Random) -> None:
    fill = rng.choice([(14, 30, 92), (16, 36, 86), (34, 34, 34)])
    draw.rectangle([x0, y0, x1, y1], fill=fill)
    for _ in range(28):
        yy = rng.randint(y0, y1)
        alpha_col = rng.choice([(65, 100, 180), (80, 120, 190), (60, 60, 60)])
        draw.line([x0, yy, x1, yy], fill=alpha_col, width=1)
    tx = x0 + 18
    ty = y0 + rng.randint(20, 160)
    draw.text((tx, ty), rng.choice(("Chess", "Opening", "Strategy")), fill=(245, 245, 245))
    draw.text((tx, ty + 44), rng.choice(("Analysis", "Trainer", "Puzzle")), fill=(230, 230, 230))


def _add_footer(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int, rng: random.Random) -> None:
    fill = rng.choice([(240, 240, 240), (250, 250, 250), (230, 232, 238)])
    draw.rectangle([x0, y0, x1, y1], fill=fill)
    for _ in range(18):
        xx = rng.randint(x0, x1)
        draw.line([xx, y0, xx, y1], fill=(190, 190, 190), width=1)
    draw.text((x0 + 16, y0 + 10), rng.choice(("Engine eval", "Move list", "Analysis")), fill=(30, 30, 30))


def _place_board_on_canvas(
    board_img: Image.Image,
    rng: random.Random,
    board_w_ratio: float,
    board_h_ratio: float,
    anchor: str,
    background=(235, 235, 235),
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), background)
    bw = int(CANVAS_W * board_w_ratio)
    bh = int(CANVAS_H * board_h_ratio)
    bw = max(280, min(CANVAS_W, bw))
    bh = max(280, min(CANVAS_H, bh))
    board = board_img.resize((bw, bh), Image.LANCZOS)

    if anchor == "right":
        x = CANVAS_W - bw
        y = 0
    elif anchor == "left":
        x = 0
        y = 0
    elif anchor == "bottom":
        x = (CANVAS_W - bw) // 2
        y = CANVAS_H - bh
    elif anchor == "top":
        x = (CANVAS_W - bw) // 2
        y = 0
    else:
        x = (CANVAS_W - bw) // 2
        y = (CANVAS_H - bh) // 2

    canvas.paste(board, (x, y))
    return canvas, (x, y, x + bw, y + bh)


def _apply_trim_resize(img: Image.Image, trim_ratio: float, rng: random.Random) -> Image.Image:
    w, h = img.size
    max_x = int(w * trim_ratio)
    max_y = int(h * trim_ratio)
    l = rng.randint(0, max_x)
    r = rng.randint(0, max_x)
    t = rng.randint(0, max_y)
    b = rng.randint(0, max_y)
    cropped = img.crop((l, t, max(l + 10, w - r), max(t + 10, h - b)))
    return cropped.resize((w, h), Image.LANCZOS)


def _apply_mono_scan(img: Image.Image, severity: int, rng: random.Random) -> Image.Image:
    gray = img.convert("L")
    contrast = 0.95 - 0.12 * severity
    contrast = max(0.45, contrast)
    gray = ImageEnhance.Contrast(gray).enhance(contrast)
    gray = ImageEnhance.Brightness(gray).enhance(0.95 + 0.03 * rng.random())
    out = gray.convert("RGB")
    if severity >= 2:
        out = out.filter(ImageFilter.GaussianBlur(radius=0.3 + 0.25 * (severity - 1)))
    if severity >= 3:
        arr = np.array(out, dtype=np.float32)
        noise = np.random.normal(0, 5.0 + 1.8 * severity, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        out = Image.fromarray(arr)
    return out


def _apply_perspective_jitter(img: Image.Image, severity: int, rng: random.Random) -> Image.Image:
    arr = np.array(img)
    h, w = arr.shape[:2]
    max_j = max(2, int(min(h, w) * (0.0025 + 0.002 * severity)))
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src + np.float32(
        [
            [rng.randint(-max_j, max_j), rng.randint(-max_j, max_j)],
            [rng.randint(-max_j, max_j), rng.randint(-max_j, max_j)],
            [rng.randint(-max_j, max_j), rng.randint(-max_j, max_j)],
            [rng.randint(-max_j, max_j), rng.randint(-max_j, max_j)],
        ]
    )
    mat = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(arr, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(out)


def synth_case(
    family: str,
    severity: int,
    board_img: Image.Image,
    rng: random.Random,
) -> tuple[Image.Image, dict]:
    sev = int(max(1, min(4, severity)))
    info: dict = {"family": family, "severity": sev}

    if family == "banner_right":
        bw = 0.62 - 0.08 * (sev - 1)
        canvas, rect = _place_board_on_canvas(board_img, rng, board_w_ratio=bw, board_h_ratio=1.0, anchor="right", background=(20, 35, 88))
        draw = ImageDraw.Draw(canvas)
        _add_banner(draw, 0, 0, rect[0], CANVAS_H, rng)
        img = canvas
    elif family == "banner_left":
        bw = 0.62 - 0.08 * (sev - 1)
        canvas, rect = _place_board_on_canvas(board_img, rng, board_w_ratio=bw, board_h_ratio=1.0, anchor="left", background=(28, 28, 30))
        draw = ImageDraw.Draw(canvas)
        _add_banner(draw, rect[2], 0, CANVAS_W, CANVAS_H, rng)
        img = canvas
    elif family == "thin_frame":
        bw = 0.52 + 0.02 * (4 - sev)
        bh = 0.84 - 0.06 * (sev - 1)
        canvas, rect = _place_board_on_canvas(board_img, rng, board_w_ratio=bw, board_h_ratio=bh, anchor="center", background=(245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        fw = max(1, 4 - sev)
        draw.rectangle([rect[0] - 1, rect[1] - 1, rect[2] + 1, rect[3] + 1], outline=(255, 255, 255), width=fw)
        _add_footer(draw, 0, int(CANVAS_H * 0.87), CANVAS_W, CANVAS_H, rng)
        img = canvas
    elif family == "footer_heavy":
        bh = 0.78 - 0.08 * (sev - 1)
        canvas, rect = _place_board_on_canvas(board_img, rng, board_w_ratio=0.90, board_h_ratio=bh, anchor="top", background=(238, 238, 238))
        draw = ImageDraw.Draw(canvas)
        _add_footer(draw, 0, rect[3], CANVAS_W, CANVAS_H, rng)
        img = canvas
    elif family == "trimmed":
        canvas, _ = _place_board_on_canvas(board_img, rng, board_w_ratio=0.86, board_h_ratio=0.94, anchor="center", background=(210, 214, 220))
        img = _apply_trim_resize(canvas, trim_ratio=0.035 + 0.03 * sev, rng=rng)
    elif family == "mono_scan":
        canvas, _ = _place_board_on_canvas(board_img, rng, board_w_ratio=0.84, board_h_ratio=0.92, anchor="center", background=(220, 220, 220))
        img = _apply_mono_scan(canvas, sev, rng)
    elif family == "combo":
        bw = 0.60 - 0.07 * (sev - 1)
        canvas, rect = _place_board_on_canvas(board_img, rng, board_w_ratio=bw, board_h_ratio=0.92, anchor=rng.choice(("right", "left")), background=(30, 34, 40))
        draw = ImageDraw.Draw(canvas)
        if rect[0] > 0:
            _add_banner(draw, 0, 0, rect[0], CANVAS_H, rng)
        if rect[2] < CANVAS_W:
            _add_banner(draw, rect[2], 0, CANVAS_W, CANVAS_H, rng)
        _add_footer(draw, 0, int(CANVAS_H * (0.90 - 0.03 * sev)), CANVAS_W, CANVAS_H, rng)
        img = _apply_trim_resize(canvas, trim_ratio=0.03 + 0.02 * sev, rng=rng)
        img = _apply_mono_scan(img, max(1, sev - 1), rng)
    else:
        raise ValueError(f"unknown family: {family}")

    if sev >= 2:
        img = _apply_perspective_jitter(img, sev, rng)
    info["size"] = [img.size[0], img.size[1]]
    return img, info


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth_path = Path(args.truth_json)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_json)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    base_fens = build_seed_boards(args.boards, rng)
    families = [
        "banner_right",
        "banner_left",
        "thin_frame",
        "footer_heavy",
        "trimmed",
        "mono_scan",
        "combo",
    ]
    levels = list(range(1, max(1, args.levels) + 1))

    truth: dict[str, str] = {}
    manifest: list[dict] = []
    idx = 0
    for fen_idx, fen_board in enumerate(base_fens):
        base_img, base_meta = render_base_board(fen_board, rng)
        for family in families:
            for level in levels:
                idx += 1
                img, scene_meta = synth_case(family, level, base_img, rng)
                name = f"boundary-{idx:05d}.jpeg"
                out_path = out_dir / name
                img.save(out_path, format="JPEG", quality=int(max(35, min(95, args.jpeg_quality))))
                truth[name] = f"{fen_board} w - - 0 1"
                manifest.append(
                    {
                        "image": name,
                        "fen_board": fen_board,
                        "family": family,
                        "severity": level,
                        "base_index": fen_idx,
                        "base_meta": base_meta,
                        "scene_meta": scene_meta,
                    }
                )

    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    family_counts = {}
    for row in manifest:
        key = f"{row['family']}:L{row['severity']}"
        family_counts[key] = family_counts.get(key, 0) + 1

    print(f"generated={len(manifest)} images -> {out_dir}")
    print(f"truth_json={truth_path}")
    print(f"manifest_json={manifest_path}")
    for key in sorted(family_counts):
        print(f"{key}={family_counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
