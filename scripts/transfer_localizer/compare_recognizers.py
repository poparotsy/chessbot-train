#!/usr/bin/env python3
"""Compare production v6 against the transfer-localizer candidate recognizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent.parent
PARENT_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(PARENT_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_SCRIPTS_DIR))

import evaluate_v6_domain_suite as eval_domain
import evaluate_v6_hardset as eval_v6
import evaluate_v6_temp_canaries as eval_canaries

DEFAULT_COMPARE_OUTPUT = TRAIN_DIR / "reports" / "transfer_localizer_v1" / "recognizer_compare.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare recognizer_v6 against recognizer_transfer_localizer.")
    parser.add_argument("--baseline-script", default=str(TRAIN_DIR / "recognizer_v6.py"))
    parser.add_argument("--candidate-script", default=str(TRAIN_DIR / "recognizer_transfer_localizer.py"))
    parser.add_argument("--images-dir", default=str(TRAIN_DIR / "images_4_test"))
    parser.add_argument("--truth-json", default=str(TRAIN_DIR / "images_4_test" / "truth_verified.json"))
    parser.add_argument("--domain-suite-json", default=str(TRAIN_DIR / "scripts" / "testdata" / "v6_domain_cases.json"))
    parser.add_argument("--stress-images-dir", default=str(TRAIN_DIR / "generated" / "v6_stress_suite" / "images"))
    parser.add_argument("--stress-truth-json", default=str(TRAIN_DIR / "generated" / "v6_stress_suite" / "truth.json"))
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    parser.add_argument("--board-perspective", choices=["auto", "white", "black"], default="auto")
    parser.add_argument("--output-json", default=str(DEFAULT_COMPARE_OUTPUT))
    parser.add_argument("--temp-canary-json", default=str(TRAIN_DIR / "scripts" / "testdata" / "v6_temp_canaries.json"))
    return parser.parse_args()


def _evaluate_script(script_path: str, args: argparse.Namespace) -> dict:
    truth = eval_v6.load_truth(Path(args.truth_json))
    hard = eval_v6.evaluate_hardset(
        truth=truth,
        images_dir=Path(args.images_dir),
        model_path=None,
        board_perspective=args.board_perspective,
        timeout_sec=float(args.timeout_sec),
        recognizer_script=script_path,
        compare_full_fen=False,
        debug=False,
        reports_dir=None,
        write_reports=False,
        show_progress=False,
    )
    domain_suite = eval_domain.load_suite(Path(args.domain_suite_json))
    hard_failures = {row["image"] for row in hard["results"] if not row.get("board_ok", False)}
    domain_pass = sum(1 for names in domain_suite.values() for image in names if image not in hard_failures)
    stress = None
    stress_path = Path(args.stress_truth_json)
    if stress_path.exists():
        stress_truth = eval_v6.load_truth(stress_path)
        stress = eval_v6.evaluate_hardset(
            truth=stress_truth,
            images_dir=Path(args.stress_images_dir),
            model_path=None,
            board_perspective=args.board_perspective,
            timeout_sec=float(args.timeout_sec),
            recognizer_script=script_path,
            compare_full_fen=False,
            debug=False,
            reports_dir=None,
            write_reports=False,
            show_progress=False,
        )
    canary_manifest = eval_canaries.load_manifest(Path(args.temp_canary_json))
    canary_locked = eval_canaries._normalize_locked_cases(canary_manifest.get("locked_cases", []))
    blocker_failures = []
    blocker_pass = 0
    for row in canary_locked:
        payload, _elapsed = eval_v6.run_recognizer_once(
            image_path=TRAIN_DIR / row["managed_path"],
            model_path=None,
            board_perspective=args.board_perspective,
            timeout_sec=float(args.timeout_sec),
            debug=False,
            script_path=script_path,
            preserve_full_fen=True,
        )
        full_ok = bool(payload.get("success", False)) and str(payload.get("fen") or "") == str(row["truth_fen"])
        if full_ok:
            blocker_pass += 1
        else:
            blocker_failures.append(row["blocker_id"])
    return {
        "script": script_path,
        "hard": hard,
        "domain_pass": int(domain_pass),
        "domain_total": sum(len(v) for v in domain_suite.values()),
        "stress": stress,
        "blocker_pass": blocker_pass,
        "blocker_total": len(canary_locked),
        "blocker_failures": blocker_failures,
    }


def _compare(baseline: dict, candidate: dict) -> dict:
    baseline_hard = {row["image"]: row for row in baseline["hard"]["results"]}
    candidate_hard = {row["image"]: row for row in candidate["hard"]["results"]}
    preserved = []
    new_wins = []
    regressions = []
    for image, base_row in baseline_hard.items():
        cand_row = candidate_hard.get(image)
        if cand_row is None:
            continue
        base_ok = bool(base_row.get("board_ok", False))
        cand_ok = bool(cand_row.get("board_ok", False))
        if base_ok and cand_ok:
            preserved.append(image)
        elif (not base_ok) and cand_ok:
            new_wins.append(image)
        elif base_ok and (not cand_ok):
            regressions.append(image)
    return {
        "preserved_wins": preserved,
        "new_wins": new_wins,
        "regressions": regressions,
        "baseline_blocker_failures": baseline["blocker_failures"],
        "candidate_blocker_failures": candidate["blocker_failures"],
    }


def main() -> int:
    args = parse_args()
    baseline = _evaluate_script(args.baseline_script, args)
    candidate = _evaluate_script(args.candidate_script, args)
    payload = {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": _compare(baseline, candidate),
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["comparison"], indent=2))
    print(f"output_json={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
