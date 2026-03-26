#!/usr/bin/env python3
"""Evaluate recognizer_v6 on the generated stress suite."""

from __future__ import annotations

import argparse
from pathlib import Path

import evaluate_v6_domain_suite as eval_domain


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
DEFAULT_ROOT = TRAIN_DIR / "generated" / "v6_stress_suite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recognizer_v6 on the generated stress suite.")
    parser.add_argument("--images-dir", default=str(DEFAULT_ROOT / "images"))
    parser.add_argument("--truth-json", default=str(DEFAULT_ROOT / "truth.json"))
    parser.add_argument("--suite-json", default=str(DEFAULT_ROOT / "categories.json"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
    )
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    parser.add_argument("--script", default=str(TRAIN_DIR / "recognizer_v6.py"))
    parser.add_argument("--compare-full-fen", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return eval_domain.main.__wrapped__(args) if hasattr(eval_domain.main, "__wrapped__") else _run(args)


def _run(args: argparse.Namespace) -> int:
    # Minimal inline clone to keep stress defaults separate while reusing domain evaluator behavior.
    import evaluate_v6_hardset as eval_v6
    import json

    images_dir = Path(args.images_dir)
    truth = eval_v6.load_truth(Path(args.truth_json))
    suite = eval_domain.load_suite(Path(args.suite_json))

    categories = []
    total_images = 0
    total_primary_pass = 0
    total_conf_sum = 0.0
    total_conf_count = 0

    for category, names in suite.items():
        subset = {name: truth[name] for name in names if name in truth}
        summary = eval_v6.evaluate_hardset(
            truth=subset,
            images_dir=images_dir,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
            timeout_sec=float(args.timeout_sec),
            recognizer_script=Path(args.script),
            compare_full_fen=bool(args.compare_full_fen),
            debug=bool(args.debug),
            reports_dir=None,
            write_reports=False,
            show_progress=bool(args.show_progress),
        )
        passed = int(summary["full_pass"] if args.compare_full_fen else summary["board_pass"])
        categories.append(
            {
                "category": category,
                "passed": passed,
                "total": int(summary["images"]),
                "avg_confidence": float(summary.get("avg_confidence", 0.0)),
            }
        )
        total_images += int(summary["images"])
        total_primary_pass += passed
        total_conf_sum += float(summary.get("avg_confidence", 0.0)) * int(summary["images"])
        total_conf_count += int(summary["images"])

    aggregate = {
        "images": total_images,
        "primary_pass": total_primary_pass,
        "avg_confidence": (total_conf_sum / total_conf_count) if total_conf_count else 0.0,
        "categories": categories,
    }
    print(f"images={aggregate['images']}")
    print(f"pass={aggregate['primary_pass']}/{aggregate['images']}")
    print(f"avg_conf={aggregate['avg_confidence']:.4f}")
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"output_json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
