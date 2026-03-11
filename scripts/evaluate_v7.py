#!/usr/bin/env python3
"""Evaluate recognizer_v7 on puzzle truth files (board/full-FEN reports)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import recognizer_v7 as rec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recognizer_v7 on puzzle truth files.")
    parser.add_argument("--images-dir", default=str(TRAIN_DIR / "images_4_test"))
    parser.add_argument("--truth-json", default=str(TRAIN_DIR / "images_4_test" / "truth_verified.json"))
    parser.add_argument("--model-path", default=None, help="Optional model path override.")
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Board perspective mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = json.loads(Path(args.truth_json).read_text(encoding="utf-8"))
    images_dir = Path(args.images_dir)
    names = sorted([name for name in truth if (images_dir / name).exists()])

    board_pass = 0
    full_pass = 0
    board_failures = []
    full_failures = []

    for name in names:
        expected_full = truth[name]
        expected_board = expected_full.split()[0]
        image_path = str(images_dir / name)

        board_fen, conf = rec.predict_board(
            image_path,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
        )
        side_to_move, side_source = rec.infer_side_to_move_from_checks(board_fen)
        predicted_full = f"{board_fen} {side_to_move} - - 0 1"

        if board_fen == expected_board:
            board_pass += 1
        else:
            board_failures.append(
                {
                    "image": name,
                    "predicted_board": board_fen,
                    "expected_board": expected_board,
                    "confidence": round(float(conf), 4),
                }
            )

        if predicted_full == expected_full:
            full_pass += 1
        else:
            full_failures.append(
                {
                    "image": name,
                    "predicted_full": predicted_full,
                    "expected_full": expected_full,
                    "confidence": round(float(conf), 4),
                    "side_to_move_source": side_source,
                }
            )

    print(f"images={len(names)}")
    print(f"board_pass={board_pass}/{len(names)}")
    print(f"full_pass={full_pass}/{len(names)}")

    if board_failures:
        print("\n=== BOARD FAILURES ===")
        for item in board_failures:
            print(json.dumps(item, ensure_ascii=True))

    if full_failures:
        print("\n=== FULL-FEN FAILURES ===")
        for item in full_failures:
            print(json.dumps(item, ensure_ascii=True))


if __name__ == "__main__":
    main()
