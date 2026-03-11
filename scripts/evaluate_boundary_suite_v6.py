#!/usr/bin/env python3
"""Evaluate recognizer against boundary stress suite with family/severity breakdown."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate boundary stress suite.")
    parser.add_argument(
        "--images-dir",
        default=str(TRAIN_DIR / "images_boundary_v6"),
        help="Boundary image directory.",
    )
    parser.add_argument(
        "--truth-json",
        default=str(TRAIN_DIR / "images_boundary_v6" / "truth_boundary_v6.json"),
        help="Truth JSON map image->full FEN.",
    )
    parser.add_argument(
        "--manifest-json",
        default=str(TRAIN_DIR / "images_boundary_v6" / "manifest_boundary_v6.json"),
        help="Manifest JSON from generator (family/severity).",
    )
    parser.add_argument(
        "--recognizer-module",
        default="recognizer_v6",
        help="Recognizer module from train root.",
    )
    parser.add_argument(
        "--model-path",
        default=str(TRAIN_DIR / "models" / "model_hybrid_v5_latest_best.pt"),
        help="Model path.",
    )
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Board perspective mode.",
    )
    parser.add_argument("--show-fails", type=int, default=20, help="How many failing rows to print.")
    return parser.parse_args()


def board_part(full_fen: str) -> str:
    return full_fen.split()[0].strip()


def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_square_match(rec, fen_a: str, fen_b: str) -> int:
    rows_a = rec.expand_fen_board(fen_a)
    rows_b = rec.expand_fen_board(fen_b)
    if len(rows_a) != 8 or len(rows_b) != 8:
        return 0
    if any(len(r) != 8 for r in rows_a) or any(len(r) != 8 for r in rows_b):
        return 0
    return sum(1 for r in range(8) for c in range(8) if rows_a[r][c] == rows_b[r][c])


def main() -> int:
    args = parse_args()
    rec = importlib.import_module(args.recognizer_module)
    rec.DEBUG_MODE = False

    truth = load_json(args.truth_json)
    manifest = load_json(args.manifest_json) if Path(args.manifest_json).exists() else []
    by_image = {row["image"]: row for row in manifest if isinstance(row, dict) and "image" in row}

    images_dir = Path(args.images_dir)
    rows = []
    for image_name, expected_full in sorted(truth.items()):
        image_path = images_dir / image_name
        expected_board = board_part(expected_full)
        if not image_path.exists():
            rows.append(
                {
                    "image": image_name,
                    "ok": False,
                    "error": "missing_image",
                    "expected_board": expected_board,
                    "pred_board": None,
                    "confidence": 0.0,
                    "square_match": 0,
                    "family": by_image.get(image_name, {}).get("family", "unknown"),
                    "severity": by_image.get(image_name, {}).get("severity", 0),
                }
            )
            continue
        try:
            pred_board, conf = rec.predict_board(
                str(image_path),
                model_path=args.model_path,
                board_perspective=args.board_perspective,
            )
            ok = pred_board == expected_board
            rows.append(
                {
                    "image": image_name,
                    "ok": ok,
                    "expected_board": expected_board,
                    "pred_board": pred_board,
                    "confidence": float(conf),
                    "square_match": int(safe_square_match(rec, pred_board, expected_board)),
                    "family": by_image.get(image_name, {}).get("family", "unknown"),
                    "severity": by_image.get(image_name, {}).get("severity", 0),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "image": image_name,
                    "ok": False,
                    "error": str(exc),
                    "expected_board": expected_board,
                    "pred_board": None,
                    "confidence": 0.0,
                    "square_match": 0,
                    "family": by_image.get(image_name, {}).get("family", "unknown"),
                    "severity": by_image.get(image_name, {}).get("severity", 0),
                }
            )

    total = len(rows)
    passed = sum(1 for r in rows if r["ok"])
    avg_conf = sum(r["confidence"] for r in rows) / max(1, total)

    bucket = defaultdict(lambda: {"total": 0, "pass": 0, "avg_square_match": 0.0})
    for r in rows:
        key = (str(r["family"]), int(r["severity"]))
        bucket[key]["total"] += 1
        bucket[key]["pass"] += int(r["ok"])
        bucket[key]["avg_square_match"] += float(r["square_match"])
    for key in bucket:
        bucket[key]["avg_square_match"] /= max(1, bucket[key]["total"])

    print(f"images={total}")
    print(f"board_pass={passed}/{total}")
    print(f"avg_conf={avg_conf:.4f}")
    print("\n=== BREAKDOWN (family, severity) ===")
    for family, severity in sorted(bucket.keys(), key=lambda x: (x[0], x[1])):
        b = bucket[(family, severity)]
        print(
            f"{family:14s} L{severity} "
            f"pass={b['pass']:3d}/{b['total']:3d} "
            f"match={b['avg_square_match']:.1f}/64"
        )

    print("\n=== FAILURES ===")
    fails = [r for r in rows if not r["ok"]]
    fails.sort(key=lambda r: (r["family"], r["severity"], r["image"]))
    for r in fails[: max(0, args.show_fails)]:
        print(
            json.dumps(
                {
                    "image": r["image"],
                    "family": r["family"],
                    "severity": r["severity"],
                    "square_match": r["square_match"],
                    "confidence": round(float(r["confidence"]), 4),
                    "predicted_board": r["pred_board"],
                    "expected_board": r["expected_board"],
                    **({"error": r["error"]} if "error" in r else {}),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
