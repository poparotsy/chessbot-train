#!/usr/bin/env python3
"""Rank v6-compatible models on hardset using recognizer_v6 only."""

from __future__ import annotations

import argparse
import glob as pyglob
import json
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent

import evaluate_v6_hardset as eval_v6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank v6-compatible model files on hardset + stress suite.")
    parser.add_argument("--models-glob", default="models/*.pt", help="Glob for model/checkpoint files.")
    parser.add_argument("--images-dir", default=str(TRAIN_DIR / "images_4_test"))
    parser.add_argument("--truth-json", default=str(TRAIN_DIR / "images_4_test" / "truth_verified.json"))
    parser.add_argument("--stress-images", default=str(TRAIN_DIR / "generated/v6_stress_suite/images"),
                        help="Stress suite image directory.")
    parser.add_argument("--stress-truth", default=str(TRAIN_DIR / "generated/v6_stress_suite/truth.json"),
                        help="Stress suite truth JSON.")
    parser.add_argument(
        "--script",
        default=str(TRAIN_DIR / "recognizer_v6.py"),
        help="Recognizer script path (default: recognizer_v6.py).",
    )
    parser.add_argument("--timeout-sec", type=float, default=45.0, help="Per-image timeout for recognizer calls.")
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Board perspective mode passed through to recognizer_v6.",
    )
    parser.add_argument(
        "--compare-full-fen",
        action="store_true",
        help="Score against full FEN (default: board-only).",
    )
    parser.add_argument("--with-debug", action="store_true", help="Print failure details per model.")
    parser.add_argument("--skip-stress", action="store_true", help="Skip stress suite evaluation.")
    return parser.parse_args()


def extract_state_dict(payload):
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    return payload


def is_v6_compatible_state(state: dict) -> Tuple[bool, str]:
    if not isinstance(state, dict) or not state:
        return False, "state_dict is empty or invalid"
    keys = set(state.keys())
    if any(k.startswith("piece_head.") for k in keys):
        return False, "multi-head checkpoint (piece_head.*) not v6-compatible"
    if any(k.startswith("geom_head.") for k in keys):
        return False, "multi-head checkpoint (geom_head.*) not v6-compatible"
    if "classifier.1.weight" not in keys and "module.classifier.1.weight" not in keys:
        return False, "missing v6 classifier weights"
    return True, "ok"


def materialize_state_dict(model_path: Path) -> Tuple[Path | None, tempfile.TemporaryDirectory | None, str | None]:
    try:
        payload = torch.load(model_path, map_location="cpu")
    except Exception as exc:
        return None, None, f"torch.load failed: {exc}"

    state = extract_state_dict(payload)
    ok, reason = is_v6_compatible_state(state)
    if not ok:
        return None, None, reason

    if isinstance(payload, dict) and "model_state" in payload:
        tmpdir = tempfile.TemporaryDirectory(prefix="v6_rank_")
        out = Path(tmpdir.name) / f"{model_path.stem}_state.pt"
        torch.save(state, out)
        return out, tmpdir, None
    return model_path, None, None


def main() -> int:
    args = parse_args()
    truth = eval_v6.load_truth(Path(args.truth_json))
    images_dir = Path(args.images_dir)

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
            # Hardset evaluation
            summary = eval_v6.evaluate_hardset(
                truth=truth,
                images_dir=images_dir,
                model_path=str(state_path),
                board_perspective=args.board_perspective,
                timeout_sec=float(args.timeout_sec),
                recognizer_script=args.script,
                compare_full_fen=bool(args.compare_full_fen),
                debug=False,
                reports_dir=None,
                write_reports=False,
                show_progress=True,
            )
            hardset_pass = int(summary["full_pass"] if args.compare_full_fen else summary["board_pass"])
            hardset_total = int(summary["images"])
            hardset_conf = float(summary.get("avg_confidence", 0.0))

            # Stress suite evaluation
            stress_pass = 0
            stress_total = 0
            stress_conf = 0.0
            if not args.skip_stress:
                stress_truth = eval_v6.load_truth(Path(args.stress_truth))
                stress_summary = eval_v6.evaluate_hardset(
                    truth=stress_truth,
                    images_dir=Path(args.stress_images),
                    model_path=str(state_path),
                    board_perspective=args.board_perspective,
                    timeout_sec=float(args.timeout_sec),
                    recognizer_script=args.script,
                    compare_full_fen=bool(args.compare_full_fen),
                    debug=False,
                    reports_dir=None,
                    write_reports=False,
                    show_progress=True,
                )
                stress_pass = int(stress_summary["full_pass"] if args.compare_full_fen else stress_summary["board_pass"])
                stress_total = int(stress_summary["images"])
                stress_conf = float(stress_summary.get("avg_confidence", 0.0))

            passed = hardset_pass + stress_pass
            total = hardset_total + stress_total
            avg_conf = (hardset_conf * hardset_total + stress_conf * stress_total) / max(1, total)
            failures = [row for row in summary.get("results", []) if not row.get("board_ok", False)]
            if args.compare_full_fen:
                failures = [row for row in summary.get("results", []) if not row.get("full_ok", False)]
            misses = [str(row["image"]) for row in failures]
            reports.append(
                {
                    "model": str(model_file),
                    "hardset_passed": hardset_pass,
                    "hardset_total": hardset_total,
                    "stress_passed": stress_pass,
                    "stress_total": stress_total,
                    "passed": passed,
                    "total": total,
                    "avg_confidence": avg_conf,
                    "misses": misses,
                    "summary": summary,
                }
            )
            label = f"{hardset_pass}/{hardset_total} hardset"
            if not args.skip_stress:
                label += f", {stress_pass}/{stress_total} stress"
            print(f"{model_file} -> {label} (avg_conf={avg_conf:.4f})")
            if misses:
                print(f"  misses: {', '.join(misses)}")
            if args.with_debug:
                for item in failures:
                    print(f"  - {item['image']}: pred={item.get('predicted_full', item.get('error'))}")
        finally:
            if tmpdir is not None:
                tmpdir.cleanup()

    if not reports:
        raise SystemExit("No successful evaluations.")

    reports.sort(key=lambda r: (r["passed"], r["avg_confidence"]), reverse=True)
    best = reports[0]
    summary = {
        "best_model": best["model"],
        "best_score": f"{best['passed']}/{best['total']}",
        "ranked": [
            {
                "model": row["model"],
                "passed": row["passed"],
                "total": row["total"],
                "avg_confidence": round(row["avg_confidence"], 6),
                "misses": row["misses"],
            }
            for row in reports
        ],
    }
    print("\n=== HARD-SET RANKING ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
