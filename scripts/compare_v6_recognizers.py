#!/usr/bin/env python3
"""Compare multiple recognizer scripts on the same truth set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_v6_domain_suite as eval_domain
import evaluate_v6_hardset as eval_v6


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare recognizer scripts on a shared suite.")
    parser.add_argument("--images-dir", default=str(TRAIN_DIR / "images_4_test"))
    parser.add_argument("--truth-json", default=str(TRAIN_DIR / "images_4_test" / "truth_verified.json"))
    parser.add_argument(
        "--suite-json",
        default=None,
        help="Optional category -> [image] JSON to compare a named subset.",
    )
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
    parser.add_argument("--compare-full-fen", action="store_true")
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


def build_subset(truth: dict[str, str], suite_json: str | None):
    if not suite_json:
        return truth, {}
    suite = eval_domain.load_suite(Path(suite_json))
    names = []
    categories_by_image: dict[str, list[str]] = {}
    for category, images in suite.items():
        for image in images:
            names.append(image)
            categories_by_image.setdefault(image, []).append(category)
    subset = {name: truth[name] for name in sorted(dict.fromkeys(names)) if name in truth}
    return subset, categories_by_image


def summarise_categories(results: list[dict], categories_by_image: dict[str, list[str]], compare_full_fen: bool):
    if not categories_by_image:
        return {}
    grouped: dict[str, dict[str, int]] = {}
    ok_field = "full_ok" if compare_full_fen else "board_ok"
    for row in results:
        image = row["image"]
        for category in categories_by_image.get(image, []):
            bucket = grouped.setdefault(category, {"passed": 0, "total": 0})
            bucket["total"] += 1
            if row.get(ok_field, False):
                bucket["passed"] += 1
    return grouped


def main() -> int:
    args = parse_args()
    truth = eval_v6.load_truth(Path(args.truth_json))
    subset, categories_by_image = build_subset(truth, args.suite_json)
    images_dir = Path(args.images_dir)

    reports = {}
    for label, script_path in parse_script_labels(args.script_label):
        summary = eval_v6.evaluate_hardset(
            truth=subset,
            images_dir=images_dir,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
            timeout_sec=float(args.timeout_sec),
            recognizer_script=Path(script_path),
            compare_full_fen=bool(args.compare_full_fen),
            debug=False,
            reports_dir=None,
            write_reports=False,
            show_progress=False,
        )
        reports[label] = {
            "script": script_path,
            "summary": {
                "images": summary["images"],
                "board_pass": summary["board_pass"],
                "full_pass": summary["full_pass"],
                "avg_confidence": round(float(summary.get("avg_confidence", 0.0)), 6),
            },
            "categories": summarise_categories(summary["results"], categories_by_image, bool(args.compare_full_fen)),
            "results": summary["results"],
        }

    images = sorted(subset)
    by_image = []
    for image in images:
        row = {
            "image": image,
            "expected_board": subset[image].split()[0],
            "expected_full": subset[image],
            "categories": categories_by_image.get(image, []),
            "scripts": {},
        }
        boards = set()
        fulls = set()
        for label, payload in reports.items():
            result = next(item for item in payload["results"] if item["image"] == image)
            row["scripts"][label] = {
                "board_ok": result.get("board_ok"),
                "full_ok": result.get("full_ok"),
                "predicted_board": result.get("predicted_board"),
                "predicted_full": result.get("predicted_full"),
                "confidence": result.get("confidence"),
                "best_tag": result.get("best_tag"),
                "detector_score": result.get("detector_score"),
                "detector_support": result.get("detector_support"),
                "value_case_fused": result.get("value_case_fused"),
                "perspective_source": result.get("perspective_source"),
            }
            boards.add(str(result.get("predicted_board")))
            fulls.add(str(result.get("predicted_full")))
        row["all_same_board"] = len(boards) == 1
        row["all_same_full"] = len(fulls) == 1
        by_image.append(row)

    diffs = [row for row in by_image if not row["all_same_board"] or not row["all_same_full"]]
    output = {
        "scripts": {
            label: {
                "script": payload["script"],
                "summary": payload["summary"],
                "categories": payload["categories"],
            }
            for label, payload in reports.items()
        },
        "images": by_image,
        "diffs": diffs,
    }

    print(json.dumps(output["scripts"], indent=2))
    print(f"diff_images={len(diffs)}")
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"output_json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
