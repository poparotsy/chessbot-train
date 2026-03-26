#!/usr/bin/env python3
"""Stress-check recognizer_v6_fullguard against expected full-candidate behavior.

This is the gate before making fullguard the default recognizer.

It answers two questions:
1. Did fullguard help where `full` should win?
2. Did fullguard start hallucinating `full` where it should not win?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import deep_diagnostic_v6 as deep_diag


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
ARCHIVE_DIR = TRAIN_DIR / "archive" / "recognizer_legacy"
DEFAULT_IMAGES_DIR = TRAIN_DIR / "images_4_test"
DEFAULT_TRUTH_JSON = DEFAULT_IMAGES_DIR / "truth_verified.json"
DEFAULT_SUITE_JSON = SCRIPT_DIR / "testdata" / "v6_fullguard_cases.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress-check fullguard recognizer behavior.")
    parser.add_argument("--model-path", required=True, help="Checkpoint to evaluate.")
    parser.add_argument(
        "--recognizer-path",
        default=str(ARCHIVE_DIR / "recognizer_v6_fullguard.py"),
        help="Recognizer module under test.",
    )
    parser.add_argument("--suite-json", default=str(DEFAULT_SUITE_JSON))
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--truth-json", default=str(DEFAULT_TRUTH_JSON))
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _load_suite(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid suite JSON: {path}")
    normalized: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(value, list) or not value:
            raise ValueError(f"Invalid suite bucket {key!r} in {path}")
        normalized[str(key)] = [str(item) for item in value]
    return normalized


def _classify_case(bucket: str, row: dict) -> tuple[bool, str]:
    chosen = row["current_model"]["selected"]
    chosen_tag = chosen["tag"] if chosen else None
    chosen_ok = bool(row["current_model"]["chosen_board_ok"])

    if bucket == "full_should_win":
        ok = chosen_ok and chosen_tag == "full"
        reason = "full selected and correct" if ok else f"selected={chosen_tag} chosen_ok={chosen_ok}"
        return ok, reason

    if bucket == "full_should_not_win":
        ok = chosen_ok and chosen_tag != "full"
        reason = "non-full selected and correct" if ok else f"selected={chosen_tag} chosen_ok={chosen_ok}"
        return ok, reason

    if bucket == "full_neutral_ok":
        ok = chosen_ok and chosen_tag == "full"
        if chosen_ok:
            reason = f"board correct via {chosen_tag}"
            return True, reason
        reason = f"selected={chosen_tag} chosen_ok={chosen_ok}"
        return ok, reason

    if bucket == "full_localization_cases":
        chosen_fen = chosen["board_fen"] if chosen else None
        reason = f"selected={chosen_tag} chosen_ok={chosen_ok} fen={chosen_fen}"
        return True, reason

    raise ValueError(f"Unsupported suite bucket: {bucket}")


def main() -> int:
    args = parse_args()
    truth = deep_diag.load_truth(Path(args.truth_json))
    suite = _load_suite(Path(args.suite_json))
    images_dir = Path(args.images_dir)

    bucket_results = []
    total_cases = 0
    total_failures = 0

    for bucket, names in suite.items():
        cases = []
        passed = 0
        for name in names:
            image_path = deep_diag.resolve_image_path(name, images_dir)
            truth_full = truth.get(image_path.name)
            row = deep_diag.analyze_image(
                image_path=image_path,
                truth_full=truth_full,
                recognizer_path=args.recognizer_path,
                model_path=args.model_path,
                baseline_model_path=None,
                board_perspective=args.board_perspective,
                topk=args.topk,
                detail_level="human",
                save_crops_dir=None,
            )
            ok, reason = _classify_case(bucket, row)
            total_cases += 1
            if ok:
                passed += 1
            else:
                total_failures += 1
            current = row["current_model"]
            cases.append(
                {
                    "image": image_path.name,
                    "ok": ok,
                    "reason": reason,
                    "verdict": current["verdict"],
                    "selected_tag": current["selected"]["tag"] if current["selected"] else None,
                    "selected_board_ok": current["chosen_board_ok"],
                    "full_board_ok": current["full_board_ok"],
                    "selected_board_fen": current["selected"]["board_fen"] if current["selected"] else None,
                    "full_board_fen": current["full"]["board_fen"] if current["full"] else None,
                }
            )
        bucket_results.append(
            {
                "bucket": bucket,
                "passed": passed,
                "total": len(names),
                "cases": cases,
            }
        )

    report = {
        "tool": "evaluate_fullguard_stress_v6",
        "model_path": args.model_path,
        "recognizer_path": str(Path(args.recognizer_path).resolve()),
        "suite_json": str(Path(args.suite_json).resolve()),
        "summary": {
            "passed": total_cases - total_failures,
            "total": total_cases,
            "failures": total_failures,
        },
        "buckets": bucket_results,
    }

    for bucket in bucket_results:
        print(f"{bucket['bucket']}: {bucket['passed']}/{bucket['total']}")
        failures = [case for case in bucket["cases"] if not case["ok"]]
        for case in failures:
            print(
                f"  - {case['image']}: {case['reason']} "
                f"verdict={case['verdict']} selected={case['selected_tag']}"
            )

    print(
        f"summary: {report['summary']['passed']}/{report['summary']['total']} "
        f"(failures={report['summary']['failures']})"
    )

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report_json: {out_path}")

    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
