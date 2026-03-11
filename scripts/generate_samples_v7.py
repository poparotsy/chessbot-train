#!/usr/bin/env python3
"""Generate visual sample scenes using v7 distribution (with geometry labels)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import generate_hybrid_v7 as gen7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sample v7 training scenes.")
    parser.add_argument("--count", type=int, default=12, help="Number of samples.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "sample_boards_v7",
        help="Output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed (default: random each run).",
    )
    parser.add_argument(
        "--present-only",
        action="store_true",
        help="Skip board-absent samples.",
    )
    parser.add_argument(
        "--draw-corners",
        action="store_true",
        help="Overlay corner labels on saved samples.",
    )
    return parser.parse_args()


def _draw_corners(img: Image.Image, corners_flat: np.ndarray) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    pts = corners_flat.reshape(4, 2).copy()
    pts[:, 0] *= (img.size[0] - 1)
    pts[:, 1] *= (img.size[1] - 1)
    p = [(float(x), float(y)) for x, y in pts.tolist()]
    draw.line([p[0], p[1], p[2], p[3], p[0]], fill=(0, 255, 0), width=2)
    for i, (x, y) in enumerate(p):
        r = 3
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 80, 20))
        draw.text((x + 4, y + 2), str(i), fill=(255, 255, 255))
    return out


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    print(f"Generating {args.count} v7 samples into: {args.output_dir}")
    made = 0
    attempts = 0
    while made < args.count and attempts < args.count * 30:
        attempts += 1
        sample = gen7.make_sample()
        present = float(sample["board_present"]) > 0.5
        if args.present_only and not present:
            continue

        img = Image.fromarray(sample["x"].transpose(1, 2, 0))
        if args.draw_corners and present:
            img = _draw_corners(img, sample["corners"])
        out_name = f"sample_v7_{made + 1:03d}.png"
        out_path = args.output_dir / out_name
        img.save(out_path)

        manifest.append(
            {
                "file": out_name,
                "board_present": int(present),
                "family": sample["family"],
                "severity": int(sample["severity"]),
                "pov": int(sample["pov"]),
                "stm": int(sample["stm"]),
                "corners": [round(float(v), 5) for v in sample["corners"].tolist()],
            }
        )
        print(
            f"  ✅ {out_name} | present={int(present)} | family={sample['family']} "
            f"| severity={sample['severity']} | pov={sample['pov']} | stm={sample['stm']}"
        )
        made += 1

    manifest_path = args.output_dir / "manifest_v7.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
