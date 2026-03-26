#!/usr/bin/env python3
"""Analyze confidence deltas across recognizer scripts for identical board outputs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import evaluate_v6_hardset as eval_v6


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze confidence variance across recognizer scripts.")
    parser.add_argument("--images-dir", default=str(TRAIN_DIR / "images_4_test"))
    parser.add_argument("--truth-json", default=str(TRAIN_DIR / "images_4_test" / "truth_verified.json"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
    )
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    parser.add_argument(
        "--script-label",
        action="append",
        required=True,
        help="Recognizer label and path in the form label=path. Repeat for each recognizer.",
    )
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def parse_script_labels(rows: list[str]) -> list[tuple[str, str]]:
    items = []
    for row in rows:
        if "=" not in row:
            path = str(Path(row).resolve())
            items.append((Path(path).stem, path))
            continue
        label, path = row.split("=", 1)
        items.append((label.strip(), str(Path(path.strip()).resolve())))
    return items


def main() -> int:
    args = parse_args()
    truth = eval_v6.load_truth(Path(args.truth_json))
    images_dir = Path(args.images_dir)

    reports = {}
    for label, script_path in parse_script_labels(args.script_label):
        summary = eval_v6.evaluate_hardset(
            truth=truth,
            images_dir=images_dir,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
            timeout_sec=float(args.timeout_sec),
            recognizer_script=Path(script_path),
            compare_full_fen=False,
            debug=False,
            reports_dir=None,
            write_reports=False,
            show_progress=False,
        )
        reports[label] = {row["image"]: row for row in summary["results"]}

    labels = list(reports.keys())
    rows = []
    pairwise = []
    for image in sorted(truth):
        image_rows = [reports[label][image] for label in labels]
        boards = {str(row.get("predicted_board")) for row in image_rows}
        if len(boards) != 1:
            continue
        row = {"image": image, "predicted_board": image_rows[0].get("predicted_board"), "scripts": {}}
        for label in labels:
            item = reports[label][image]
            row["scripts"][label] = {
                "confidence": item.get("confidence"),
                "best_tag": item.get("best_tag"),
                "detector_score": item.get("detector_score"),
                "detector_support": item.get("detector_support"),
                "perspective_source": item.get("perspective_source"),
                "value_case_fused": item.get("value_case_fused"),
            }
        rows.append(row)
        for left, right in itertools.combinations(labels, 2):
            left_conf = float(reports[left][image].get("confidence", 0.0))
            right_conf = float(reports[right][image].get("confidence", 0.0))
            pairwise.append(
                {
                    "image": image,
                    "left": left,
                    "right": right,
                    "delta": round(left_conf - right_conf, 6),
                }
            )

    output = {
        "scripts": labels,
        "identical_board_images": len(rows),
        "images": rows,
        "pairwise_deltas": pairwise,
    }
    print(f"identical_board_images={len(rows)}")
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"output_json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
