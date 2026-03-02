#!/usr/bin/env python3
"""Rank model/checkpoint files on a fixed hard puzzle set."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import torch

import recognizer_v4 as rec


DEFAULT_TRUTH = {
    "puzzle-00003.jpeg": "7R/8/8/8/6pq/7k/4Np1r/5KbQ w - - 0 1",
    "puzzle-00020.jpeg": "8/1r2kp2/1P3Rb1/2K5/1P5P/8/8/8 w - - 0 1",
    "puzzle-00021.jpeg": "8/8/8/8/K7/8/pp1Q4/k7 w - - 0 1",
    "puzzle-00023.jpeg": "6k1/p2rN1p1/1p4P1/2n5/8/4P3/PPq5/K4R1R w - - 0 1",
    "puzzle-00024.jpeg": "4Rnk1/pr3ppp/1p3q1N/6Q1/2p5/8/P4PPP/1K4K1 w - - 0 1",
    "puzzle-00025.jpeg": "2r1n1Qk/p1p3pr/1p3p2/8/P1N5/5PP1/B1P1q2P/7K w - - 0 1",
    "puzzle-00026.jpeg": "nrb5/k1P1R3/1p6/p7/K7/3Q4/8/8 w - - 0 1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate model/checkpoint files against hard puzzle set."
    )
    parser.add_argument(
        "--models-glob",
        default="models/model_hybrid*.pt",
        help="Glob for model/checkpoint files (default: models/model_hybrid*.pt)",
    )
    parser.add_argument(
        "--images-dir",
        default="images_4_test",
        help="Directory containing puzzle images (default: images_4_test)",
    )
    parser.add_argument(
        "--truth-json",
        default=None,
        help="Optional path to JSON file: {\"puzzle-xxxxx.jpeg\": \"fen ...\"}",
    )
    parser.add_argument(
        "--with-debug",
        action="store_true",
        help="Print mismatches for each model.",
    )
    return parser.parse_args()


def load_truth(truth_json: str | None) -> Dict[str, str]:
    if not truth_json:
        return dict(DEFAULT_TRUTH)
    with open(truth_json, "r", encoding="utf-8") as handle:
        return json.load(handle)


def materialize_state_dict(model_path: Path) -> Tuple[Path, tempfile.TemporaryDirectory | None]:
    """Return a state_dict file path usable by recognizer_v4."""
    payload = torch.load(model_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        tmpdir = tempfile.TemporaryDirectory(prefix="hardset_")
        out = Path(tmpdir.name) / f"{model_path.stem}_state.pt"
        torch.save(payload["model_state"], out)
        return out, tmpdir
    return model_path, None


def evaluate_model(model_file: Path, truth: Dict[str, str], images_dir: Path) -> Dict:
    state_path, tmpdir = materialize_state_dict(model_file)
    passed = 0
    total = len(truth)
    mismatches = []
    conf_sum = 0.0

    try:
        for image_name, expected_fen in truth.items():
            image_path = images_dir / image_name
            if not image_path.exists():
                mismatches.append(
                    {"image": image_name, "reason": f"missing image: {image_path}"}
                )
                continue

            fen, conf = rec.predict_board(str(image_path), model_path=str(state_path))
            predicted_fen = f"{fen} w - - 0 1"
            conf_sum += float(conf)

            if predicted_fen == expected_fen:
                passed += 1
            else:
                mismatches.append(
                    {
                        "image": image_name,
                        "expected": expected_fen,
                        "predicted": predicted_fen,
                        "confidence": round(float(conf), 4),
                    }
                )
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()

    return {
        "model": str(model_file),
        "passed": passed,
        "total": total,
        "score": passed / total if total else 0.0,
        "avg_confidence": conf_sum / total if total else 0.0,
        "mismatches": mismatches,
    }


def main() -> None:
    args = parse_args()
    truth = load_truth(args.truth_json)
    images_dir = Path(args.images_dir)

    model_files = sorted(Path(".").glob(args.models_glob))
    if not model_files:
        raise SystemExit(f"No model files found for glob: {args.models_glob}")

    rec.USE_EDGE_DETECTION = True
    rec.USE_SQUARE_DETECTION = False
    rec.DEBUG_MODE = False

    reports = []
    for model_file in model_files:
        try:
            report = evaluate_model(model_file, truth, images_dir)
            reports.append(report)
            print(
                f"{model_file} -> {report['passed']}/{report['total']} "
                f"(avg_conf={report['avg_confidence']:.4f})"
            )
            if args.with_debug and report["mismatches"]:
                for mm in report["mismatches"]:
                    print(f"  - {mm['image']}: {mm.get('predicted', mm.get('reason'))}")
        except Exception as exc:
            print(f"{model_file} -> ERROR: {exc}")

    if not reports:
        raise SystemExit("No successful evaluations.")

    reports.sort(key=lambda r: (r["passed"], r["avg_confidence"]), reverse=True)
    best = reports[0]

    summary = {
        "best_model": best["model"],
        "best_score": f"{best['passed']}/{best['total']}",
        "ranked": [
            {
                "model": r["model"],
                "passed": r["passed"],
                "total": r["total"],
                "avg_confidence": round(r["avg_confidence"], 6),
            }
            for r in reports
        ],
    }
    print("\n=== HARD-SET RANKING ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
