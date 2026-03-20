#!/usr/bin/env python3
"""Evaluate recognizer_v6 on tiny domain-focused hardset slices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_v6_hardset as eval_v6


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v6 on a tiny domain regression suite.")
    parser.add_argument("--images-dir", default=str(TRAIN_DIR / "images_4_test"))
    parser.add_argument("--truth-json", default=str(TRAIN_DIR / "images_4_test" / "truth_verified.json"))
    parser.add_argument(
        "--suite-json",
        default=str(TRAIN_DIR / "scripts" / "testdata" / "v6_domain_cases.json"),
        help="JSON map: category -> [image names].",
    )
    parser.add_argument("--model-path", default=None, help="Optional model checkpoint override.")
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Board perspective mode passed through to recognizer_v6.",
    )
    parser.add_argument("--timeout-sec", type=float, default=45.0, help="Per-image timeout in seconds.")
    parser.add_argument(
        "--script",
        default=str(TRAIN_DIR / "recognizer_v6.py"),
        help="Path to recognizer script module.",
    )
    parser.add_argument(
        "--compare-full-fen",
        action="store_true",
        help="Primary score uses full FEN instead of board-only.",
    )
    parser.add_argument("--debug", action="store_true", help="Pass debug mode through to recognizer_v6.")
    parser.add_argument("--show-progress", action="store_true", help="Show per-image progress inside each category.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def load_suite(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"suite JSON must be a non-empty object: {path}")
    suite: dict[str, list[str]] = {}
    for category, names in raw.items():
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"invalid category name in suite JSON: {path}")
        if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.strip() for n in names):
            raise ValueError(f"suite category '{category}' must be a non-empty string list")
        suite[category] = names
    return suite


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir)
    truth = eval_v6.load_truth(Path(args.truth_json))
    suite = load_suite(Path(args.suite_json))

    categories = []
    total_images = 0
    total_primary_pass = 0
    total_conf_sum = 0.0
    total_conf_count = 0

    for category, names in suite.items():
        subset = {name: truth[name] for name in names if name in truth}
        missing_truth = [name for name in names if name not in truth]
        if missing_truth:
            raise KeyError(f"suite category '{category}' references unknown truth images: {missing_truth}")

        summary = eval_v6.evaluate_hardset(
            truth=subset,
            images_dir=images_dir,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
            timeout_sec=float(args.timeout_sec),
            recognizer_script=args.script,
            compare_full_fen=bool(args.compare_full_fen),
            debug=bool(args.debug),
            reports_dir=None,
            write_reports=False,
            show_progress=bool(args.show_progress),
        )
        passed = int(summary["full_pass"] if args.compare_full_fen else summary["board_pass"])
        misses = [
            row["image"]
            for row in summary.get("results", [])
            if not row.get("full_ok" if args.compare_full_fen else "board_ok", False)
        ]
        category_row = {
            "category": category,
            "passed": passed,
            "total": int(summary["images"]),
            "avg_confidence": float(summary.get("avg_confidence", 0.0)),
            "misses": misses,
        }
        categories.append(category_row)
        total_images += int(summary["images"])
        total_primary_pass += passed
        total_conf_sum += float(summary.get("avg_confidence", 0.0)) * int(summary["images"])
        total_conf_count += int(summary["images"])

    categories.sort(key=lambda row: (row["passed"] / max(1, row["total"]), row["avg_confidence"]), reverse=True)
    aggregate = {
        "evaluator": "evaluate_v6_domain_suite",
        "model_path": args.model_path,
        "truth_json": args.truth_json,
        "suite_json": args.suite_json,
        "images": total_images,
        "primary_pass": total_primary_pass,
        "avg_confidence": (total_conf_sum / total_conf_count) if total_conf_count else 0.0,
        "compare_full_fen": bool(args.compare_full_fen),
        "categories": categories,
    }

    print("=== DOMAIN SUITE ===")
    print(f"images={aggregate['images']}")
    print(f"pass={aggregate['primary_pass']}/{aggregate['images']}")
    print(f"avg_conf={aggregate['avg_confidence']:.4f}")
    print()
    for row in categories:
        print(
            f"{row['category']:18s} {row['passed']}/{row['total']} "
            f"(avg_conf={row['avg_confidence']:.4f})"
        )
        if row["misses"]:
            print(f"  misses: {', '.join(row['misses'])}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
