#!/usr/bin/env python3
"""Rank models using board-first scoring across hardset, domain suite, and stress suite."""

from __future__ import annotations

import argparse
import glob as pyglob
import json
import statistics
import tempfile
from pathlib import Path

import torch

import evaluate_v6_domain_suite as eval_domain
import evaluate_v6_hardset as eval_v6


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
DEFAULT_STRESS_ROOT = TRAIN_DIR / "generated" / "v6_stress_suite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank v6 models using hardset, domain, and stress suites.")
    parser.add_argument("--models-glob", default="models/*.pt")
    parser.add_argument("--script", default=str(TRAIN_DIR / "recognizer_v6.py"))
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
    )
    parser.add_argument("--hard-images-dir", default=str(TRAIN_DIR / "images_4_test"))
    parser.add_argument("--hard-truth-json", default=str(TRAIN_DIR / "images_4_test" / "truth_verified.json"))
    parser.add_argument("--domain-suite-json", default=str(TRAIN_DIR / "scripts" / "testdata" / "v6_domain_cases.json"))
    parser.add_argument("--stress-images-dir", default=str(DEFAULT_STRESS_ROOT / "images"))
    parser.add_argument("--stress-truth-json", default=str(DEFAULT_STRESS_ROOT / "truth.json"))
    parser.add_argument("--stress-suite-json", default=str(DEFAULT_STRESS_ROOT / "categories.json"))
    parser.add_argument("--with-debug", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def extract_state_dict(payload):
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    return payload


def is_v6_compatible_state(state: dict):
    if not isinstance(state, dict) or not state:
        return False, "state_dict is empty or invalid"
    keys = set(state.keys())
    if any(k.startswith("piece_head.") for k in keys) or any(k.startswith("geom_head.") for k in keys):
        return False, "multi-head checkpoint not v6-compatible"
    if "classifier.1.weight" not in keys and "module.classifier.1.weight" not in keys:
        return False, "missing v6 classifier weights"
    return True, "ok"


def materialize_state_dict(model_path: Path):
    payload = torch.load(model_path, map_location="cpu")
    state = extract_state_dict(payload)
    ok, reason = is_v6_compatible_state(state)
    if not ok:
        return None, None, reason
    if isinstance(payload, dict) and "model_state" in payload:
        tmpdir = tempfile.TemporaryDirectory(prefix="v6_true_rank_")
        out = Path(tmpdir.name) / f"{model_path.stem}_state.pt"
        torch.save(state, out)
        return out, tmpdir, None
    return model_path, None, None


def category_primary_pass(summary: dict, suite: dict[str, list[str]], compare_full_fen: bool = False):
    ok_field = "full_ok" if compare_full_fen else "board_ok"
    results = {row["image"]: row for row in summary["results"]}
    passed = 0
    total = 0
    for names in suite.values():
        for image in names:
            if image not in results:
                continue
            total += 1
            if results[image].get(ok_field, False):
                passed += 1
    return passed, total


def confidence_stats(*summaries):
    values = []
    for summary in summaries:
        for row in summary.get("results", []):
            if row.get("board_ok", False) and row.get("confidence") is not None:
                values.append(float(row["confidence"]))
    if not values:
        return 0.0, 0.0
    return float(sum(values) / len(values)), float(statistics.pstdev(values) if len(values) > 1 else 0.0)


def main() -> int:
    args = parse_args()
    hard_truth = eval_v6.load_truth(Path(args.hard_truth_json))
    domain_suite = eval_domain.load_suite(Path(args.domain_suite_json))
    stress_truth = eval_v6.load_truth(Path(args.stress_truth_json)) if Path(args.stress_truth_json).exists() else {}
    stress_suite = eval_domain.load_suite(Path(args.stress_suite_json)) if Path(args.stress_suite_json).exists() else {}

    model_files = sorted(Path(p) for p in pyglob.glob(args.models_glob))
    if not model_files:
        model_files = sorted(Path(".").glob(args.models_glob))
    if not model_files:
        raise SystemExit(f"No model files found for glob: {args.models_glob}")

    reports = []
    for model_file in model_files:
        state_path, tmpdir, err = materialize_state_dict(model_file)
        if err is not None or state_path is None:
            print(f"{model_file} -> ERROR: {err}")
            continue
        try:
            hard = eval_v6.evaluate_hardset(
                truth=hard_truth,
                images_dir=Path(args.hard_images_dir),
                model_path=str(state_path),
                board_perspective=args.board_perspective,
                timeout_sec=float(args.timeout_sec),
                recognizer_script=Path(args.script),
                compare_full_fen=False,
                debug=False,
                reports_dir=None,
                write_reports=False,
                show_progress=False,
            )
            domain_pass, domain_total = category_primary_pass(hard, domain_suite, compare_full_fen=False)
            domain_full_pass, _ = category_primary_pass(hard, domain_suite, compare_full_fen=True)

            stress = {
                "board_pass": 0,
                "full_pass": 0,
                "images": 0,
                "results": [],
            }
            stress_pass = 0
            stress_total = 0
            stress_full_pass = 0
            if stress_truth:
                stress = eval_v6.evaluate_hardset(
                    truth=stress_truth,
                    images_dir=Path(args.stress_images_dir),
                    model_path=str(state_path),
                    board_perspective=args.board_perspective,
                    timeout_sec=float(args.timeout_sec),
                    recognizer_script=Path(args.script),
                    compare_full_fen=False,
                    debug=False,
                    reports_dir=None,
                    write_reports=False,
                    show_progress=False,
                )
                stress_pass, stress_total = category_primary_pass(stress, stress_suite, compare_full_fen=False)
                stress_full_pass, _ = category_primary_pass(stress, stress_suite, compare_full_fen=True)

            avg_conf, conf_std = confidence_stats(hard, stress)
            report = {
                "model": str(model_file),
                "hard_board_pass": int(hard["board_pass"]),
                "hard_total": int(hard["images"]),
                "domain_board_pass": int(domain_pass),
                "domain_total": int(domain_total),
                "stress_board_pass": int(stress_pass),
                "stress_total": int(stress_total),
                "full_pass_total": int(hard["full_pass"]) + int(domain_full_pass) + int(stress_full_pass),
                "avg_confidence": avg_conf,
                "confidence_std": conf_std,
                "misses": {
                    "hard": [row["image"] for row in hard["results"] if not row.get("board_ok", False)],
                    "domain": [
                        image
                        for names in domain_suite.values()
                        for image in names
                        if image in {row["image"] for row in hard["results"] if not row.get("board_ok", False)}
                    ],
                    "stress": [row["image"] for row in stress["results"] if not row.get("board_ok", False)],
                },
            }
            reports.append(report)
            print(
                f"{model_file} -> hard={report['hard_board_pass']}/{report['hard_total']} "
                f"domain={report['domain_board_pass']}/{report['domain_total']} "
                f"stress={report['stress_board_pass']}/{report['stress_total']}"
            )
            if args.with_debug:
                print(json.dumps(report["misses"], indent=2))
        finally:
            if tmpdir is not None:
                tmpdir.cleanup()

    if not reports:
        raise SystemExit("No successful evaluations.")

    reports.sort(
        key=lambda row: (
            -row["hard_board_pass"],
            -row["domain_board_pass"],
            -row["stress_board_pass"],
            -row["full_pass_total"],
            row["confidence_std"],
            -row["avg_confidence"],
        )
    )
    output = {
        "best_model": reports[0]["model"],
        "ranked": reports,
    }
    print("\n=== TRUE TEST RANKING ===")
    print(json.dumps(output, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"output_json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
