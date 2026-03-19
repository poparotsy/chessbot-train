#!/usr/bin/env python3
"""Normalize palette PNG assets to RGBA to avoid PIL transparency warnings."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


DEFAULT_PIECE_DIRS = [
    "bases",
    "club",
    "dash",
    "glass",
    "gothic",
    "lolz",
    "nature",
    "neon",
    "sky",
    "tigers",
    "tournament",
    "vintage",
]


def convert_png(path: Path) -> bool:
    img = Image.open(path)
    if img.mode == "RGBA" and "transparency" not in img.info:
        return False
    img.convert("RGBA").save(path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--board", action="append", default=["glass.png"])
    parser.add_argument("--piece-set", action="append", dest="piece_sets", default=[])
    args = parser.parse_args()

    root = Path(args.root)
    piece_sets = args.piece_sets or DEFAULT_PIECE_DIRS
    converted = 0

    for name in piece_sets:
        for path in (root / "piece_sets" / name).glob("*.png"):
            converted += int(convert_png(path))

    for board_name in args.board:
        path = root / "board_themes" / board_name
        if path.exists():
            converted += int(convert_png(path))

    print(f"Converted {converted} PNG files to RGBA")


if __name__ == "__main__":
    main()
