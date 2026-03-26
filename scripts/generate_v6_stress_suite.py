#!/usr/bin/env python3
"""Build a realistic v6 stress suite by swapping fresh boards into images_4_test-style templates."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import generate_hybrid_v6 as gen
import recognizer_v6 as rec


DEFAULT_OUTPUT_DIR = TRAIN_DIR / "generated" / "v6_stress_suite"
DEFAULT_TEMPLATE_DIR = TRAIN_DIR / "images_4_test"
WOOD_THEME_CHOICES = [
    "burled_wood.png",
    "dark_wood.png",
    "walnut.png",
    "wood.jpg",
    "wood2.jpg",
    "wood3.jpg",
    "wood4.jpg",
]
WOOD_PROFILE_NAMES = ["wood_3d_arrow_clean"]
PRINT_PROFILE_NAMES = ["book_page_reference", "shirt_print_reference", "mono_scan"]
DARK_PROFILE_NAMES = ["dark_anchor_clean", "broadcast_dark_sparse"]
NEUTRAL_PROFILE_NAMES = ["clean", "dark_anchor_clean", "digital_overlay_clean", "wood_3d_arrow_clean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a realistic v6 stress suite from images_4_test templates.")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    return parser.parse_args()


def _template_category(img: Image.Image, name: str) -> str:
    rgb = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    val_mean = float(np.mean(hsv[:, :, 2]))
    ratio = img.size[0] / float(max(1, img.size[1]))

    if name == "puzzle-00026.jpeg":
        return "wood_neo_reference"
    if sat_mean < 24.0 and val_mean > 115.0:
        return "print_scan_reference"
    if val_mean < 96.0:
        return "dark_broadcast_reference"
    if ratio > 1.18:
        return "wide_ui_reference"
    if ratio < 0.84:
        return "tall_ui_reference"
    return "square_reference"


def _load_template_entries(template_dir: Path):
    entries = []
    for path in sorted(template_dir.iterdir()):
        if path.name == "truth_verified.json" or not path.is_file():
            continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue

        img = Image.open(path).convert("RGB")
        proposals = rec.detect_board_grid(img, max_hypotheses=1)
        if not proposals:
            continue
        best = proposals[0]
        entries.append(
            {
                "name": path.name,
                "path": path,
                "size": list(img.size),
                "corners": np.asarray(best["corners"], dtype=np.float32),
                "tag": str(best["tag"]),
                "score": float(best["score"]),
                "support_ratio": float(best["support_ratio"]),
                "category": _template_category(img, path.name),
            }
        )
    if not entries:
        raise SystemExit(f"No usable templates found in {template_dir}")
    return entries


def _fen_grid(fen_board: str):
    grid = [[None] * 8 for _ in range(8)]
    for row_idx, row_str in enumerate(fen_board.split("/")):
        col_idx = 0
        for char in row_str:
            if char.isdigit():
                col_idx += int(char)
            else:
                grid[row_idx][col_idx] = char
                col_idx += 1
    return grid


def _resolve_piece_asset(piece_set: str, char: str) -> Path:
    piece_dir = TRAIN_DIR / "piece_sets" / piece_set
    if piece_set == "chesscom_neo":
        name = f"{'w' if char.isupper() else 'b'}{char.lower()}.png"
    else:
        name = f"{'w' if char.isupper() else 'b'}{char.upper()}.png"
    path = piece_dir / name
    if path.exists():
        return path
    lowered = {child.name.lower(): child for child in piece_dir.iterdir() if child.is_file()}
    match = lowered.get(name.lower())
    if match is None:
        raise FileNotFoundError(f"Missing asset for {piece_set}:{char} -> {name}")
    return match


def _render_board_from_tiles(fen_board: str, profile: str, sample_seed: int):
    old_state = random.getstate()
    old_np_state = np.random.get_state()
    random.seed(sample_seed)
    np.random.seed(sample_seed % (2**32))
    try:
        tiles, _labels, meta = gen.render_board(fen_board, return_meta=True, profile=profile)
    finally:
        random.setstate(old_state)
        np.random.set_state(old_np_state)

    rows = []
    for row_idx in range(8):
        row_tiles = []
        for col_idx in range(8):
            tile = tiles[row_idx * 8 + col_idx].transpose(1, 2, 0)
            row_tiles.append(tile)
        rows.append(np.concatenate(row_tiles, axis=1))
    return Image.fromarray(np.concatenate(rows, axis=0)), meta


def _render_chesscom_neo_board(fen_board: str, sample_seed: int):
    rng = random.Random(sample_seed)
    grid = _fen_grid(fen_board)
    board_name = rng.choice(WOOD_THEME_CHOICES)
    board = Image.open(TRAIN_DIR / "board_themes" / board_name).convert("RGB").resize((512, 512), Image.LANCZOS)
    ts = 64
    for row_idx in range(8):
        for col_idx in range(8):
            char = grid[row_idx][col_idx]
            if not char:
                continue
            piece = Image.open(_resolve_piece_asset("chesscom_neo", char)).convert("RGBA").resize((ts, ts), Image.LANCZOS)
            board.paste(piece, (col_idx * ts, row_idx * ts), piece)

    cfg = gen.get_profile_config("wood_3d_arrow_clean")
    label_rng = random.Random(sample_seed ^ 0xA5A5A5)
    if label_rng.random() < cfg["LABELS_PROB"]:
        draw = gen.ImageDraw.Draw(board)
        gen.draw_board_labels(draw, ts, label_rng.choice(("white", "black")))

    old_state = random.getstate()
    old_np_state = np.random.get_state()
    random.seed(sample_seed ^ 0x5A5A5A)
    np.random.seed((sample_seed ^ 0x5A5A5A) % (2**32))
    try:
        board = gen.vandalize(board, grid, cfg)
        board = gen.apply_piece_occlusion_overlay(board, grid, cfg)
        board = gen.apply_local_piece_tilt(board, grid, cfg)
    finally:
        random.setstate(old_state)
        np.random.set_state(old_np_state)

    return board, {
        "board_theme": board_name,
        "piece_set": "chesscom_neo",
        "label_pov": None,
        "profile": "wood_neo_reference",
    }


def _choose_render_profile(category: str, sample_seed: int) -> str:
    rng = random.Random(sample_seed)
    if category == "wood_neo_reference":
        return "wood_neo_reference"
    if category == "print_scan_reference":
        return rng.choice(PRINT_PROFILE_NAMES)
    if category == "dark_broadcast_reference":
        return rng.choice(DARK_PROFILE_NAMES)
    if category == "wide_ui_reference":
        return rng.choice(["clean", "dark_anchor_clean", "digital_overlay_clean"])
    if category == "tall_ui_reference":
        return rng.choice(["clean", "dark_anchor_clean", "wood_3d_arrow_clean"])
    return rng.choice(NEUTRAL_PROFILE_NAMES)


def _render_reference_board(fen_board: str, category: str, sample_seed: int):
    profile = _choose_render_profile(category, sample_seed)
    if profile == "wood_neo_reference":
        return _render_chesscom_neo_board(fen_board, sample_seed)
    return _render_board_from_tiles(fen_board, profile, sample_seed)


def _warp_board_into_template(template_img: Image.Image, board_img: Image.Image, corners: np.ndarray) -> Image.Image:
    template = np.array(template_img.convert("RGB"))
    board = np.array(board_img.convert("RGB"))
    height, width = template.shape[:2]
    bh, bw = board.shape[:2]
    src = np.array([[0, 0], [bw - 1, 0], [bw - 1, bh - 1], [0, bh - 1]], dtype=np.float32)
    dst = rec.order_corners(np.asarray(corners, dtype=np.float32))
    matrix = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(board, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)
    mask = cv2.warpPerspective(
        np.full((bh, bw), 255, dtype=np.uint8),
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255.0
    blended = (template.astype(np.float32) * (1.0 - mask[:, :, None])) + (warped.astype(np.float32) * mask[:, :, None])
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def _build_template_plan(entries, count: int, seed: int):
    rng = random.Random(seed)
    pool = list(entries)
    plan = []
    while len(plan) < count:
        rng.shuffle(pool)
        plan.extend(pool)
    return plan[:count]


def main() -> int:
    args = parse_args()
    template_entries = _load_template_entries(args.template_dir)
    template_plan = _build_template_plan(template_entries, int(args.count), int(args.seed))

    output_dir = args.output_dir
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    truth = {}
    categories = {}
    manifest_items = []
    category_counts = Counter()

    for idx, template in enumerate(template_plan, start=1):
        sample_seed = int(args.seed) + (idx * 104729)
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32))

        template_img = Image.open(template["path"]).convert("RGB")
        profile_for_board = _choose_render_profile(template["category"], sample_seed)
        board = gen.random_training_board(profile="clean" if profile_for_board == "wood_neo_reference" else profile_for_board)
        fen_board = board.board_fen()
        board_img, render_meta = _render_reference_board(fen_board, template["category"], sample_seed)
        composed = _warp_board_into_template(template_img, board_img, template["corners"])

        file_name = f"stress-{idx:05d}.png"
        out_path = images_dir / file_name
        composed.save(out_path)

        truth[file_name] = f"{fen_board} w - - 0 1"
        categories.setdefault(template["category"], []).append(file_name)
        category_counts[template["category"]] += 1
        manifest_items.append(
            {
                "file": file_name,
                "fen_board": fen_board,
                "source_template": template["name"],
                "source_category": template["category"],
                "source_size": template["size"],
                "detector_tag": template["tag"],
                "detector_score": template["score"],
                "detector_support_ratio": template["support_ratio"],
                "render_profile": render_meta.get("profile", profile_for_board),
                "board_theme": render_meta.get("board_theme"),
                "piece_set": render_meta.get("piece_set"),
                "label_pov": render_meta.get("label_pov"),
            }
        )

    truth_path = output_dir / "truth.json"
    categories_path = output_dir / "categories.json"
    manifest_path = output_dir / "manifest.json"
    truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    categories_path.write_text(json.dumps(categories, indent=2), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "seed": int(args.seed),
                "count": int(args.count),
                "generator": "images_4_test_template_swap",
                "template_count": len(template_entries),
                "category_counts": dict(category_counts),
                "items": manifest_items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"images_dir={images_dir}")
    print(f"truth_json={truth_path}")
    print(f"categories_json={categories_path}")
    print(f"manifest_json={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
