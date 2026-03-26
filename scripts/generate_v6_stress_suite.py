#!/usr/bin/env python3
"""Generate a deterministic v6 stress suite from current assets and profiles."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import generate_hybrid_v6 as gen


DEFAULT_OUTPUT_DIR = ROOT / "generated" / "v6_stress_suite"
DEFAULT_WEIGHTS = [
    ("edge_frame", 0.24),
    ("mono_scan", 0.20),
    ("wood_3d_arrow_clean", 0.18),
    ("shirt_print_reference", 0.16),
    ("book_page_reference", 0.12),
    ("dark_anchor", 0.10),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deterministic v6 stress suite.")
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def render_board_image(fen_board: str, profile: str):
    tiles, _labels, meta = gen.render_board(fen_board, return_meta=True, profile=profile)
    rows = []
    for row_idx in range(8):
        row_tiles = []
        for col_idx in range(8):
            tile = tiles[row_idx * 8 + col_idx].transpose(1, 2, 0)
            row_tiles.append(tile)
        rows.append(np.concatenate(row_tiles, axis=1))
    background = np.concatenate(rows, axis=0)
    return background, meta


def build_profile_plan(count: int):
    weights = np.array([weight for _, weight in DEFAULT_WEIGHTS], dtype=np.float64)
    weights = weights / np.sum(weights)
    exact = weights * count
    base = np.floor(exact).astype(int)
    remainder = count - int(base.sum())
    order = np.argsort(-(exact - base))
    for idx in order[:remainder]:
        base[idx] += 1
    plan = []
    for idx, (name, _weight) in enumerate(DEFAULT_WEIGHTS):
        plan.extend([name] * int(base[idx]))
    return plan


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))

    output_dir = args.output_dir
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    truth = {}
    categories = {name: [] for name, _weight in DEFAULT_WEIGHTS}
    manifest = []

    plan = build_profile_plan(int(args.count))
    random.Random(args.seed).shuffle(plan)
    counts = Counter(plan)

    for idx, profile in enumerate(plan, start=1):
        fen_board = gen.random_training_board(profile=profile).board_fen()
        image, meta = render_board_image(fen_board, profile)
        file_name = f"stress-{idx:05d}.png"
        out_path = images_dir / file_name
        from PIL import Image

        Image.fromarray(image).save(out_path)
        truth[file_name] = f"{fen_board} w - - 0 1"
        categories.setdefault(profile, []).append(file_name)
        manifest.append(
            {
                "file": file_name,
                "fen_board": fen_board,
                "profile": profile,
                "board_theme": meta.get("board_theme"),
                "piece_set": meta.get("piece_set"),
                "label_pov": meta.get("label_pov"),
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
                "seed": args.seed,
                "count": args.count,
                "profile_counts": dict(counts),
                "items": manifest,
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
