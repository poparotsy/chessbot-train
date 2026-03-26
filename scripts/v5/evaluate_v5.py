#!/usr/bin/env python3
"""Evaluate recognizer_v5 on puzzle sets with clear board-vs-full-FEN reporting."""

import argparse
import json
import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

import recognizer_v5 as rec


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate recognizer_v5 on puzzle truth files.")
    parser.add_argument(
        "--images-dir",
        default=str(TRAIN_DIR / "images_4_test"),
        help="Directory containing puzzle images.",
    )
    parser.add_argument(
        "--truth-json",
        default=str(TRAIN_DIR / "images_4_test" / "truth_known.json"),
        help="JSON map of image file -> full FEN.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional checkpoint override.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.truth_json, "r", encoding="utf-8") as handle:
        truth = json.load(handle)

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

        board_fen, conf = rec.predict_board(image_path, model_path=args.model_path)
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
