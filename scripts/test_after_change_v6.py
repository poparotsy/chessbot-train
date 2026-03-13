#!/usr/bin/env python3
"""Single post-change gate for recognizer_v6 (quick by default, full on demand)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, TRAIN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import benchmark_v6
import evaluate_v6_hardset as eval_v6
import recognizer_v6 as rec_v6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v6 post-change gate.")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--images-dir", default=str(TRAIN_DIR / "images_4_test"))
    parser.add_argument("--model-path", default=str(TRAIN_DIR / "models" / "model_hybrid_v5_latest_best.pt"))
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    parser.add_argument("--quick-truth-json", default=str(SCRIPT_DIR / "testdata" / "v6_quick_cases.json"))
    parser.add_argument("--orientation-truth-json", default=str(SCRIPT_DIR / "testdata" / "v6_orientation_cases.json"))
    parser.add_argument("--full-truth-json", default=str(TRAIN_DIR / "images_4_test" / "truth_verified.json"))
    parser.add_argument("--reports-dir", default=str(TRAIN_DIR / "reports"))
    parser.add_argument("--quick-baseline-json", default=str(TRAIN_DIR / "reports" / "v6_quick_baseline.json"))
    parser.add_argument("--update-quick-baseline", action="store_true")
    parser.add_argument("--update-perf-baseline", action="store_true")
    parser.add_argument("--median-ratio-max", type=float, default=1.25)
    parser.add_argument("--p95-ratio-max", type=float, default=1.35)
    return parser.parse_args()


def truth_signature(truth: dict) -> str:
    payload = json.dumps(truth, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def run_internal_tests() -> None:
    cmd = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(SCRIPT_DIR / "tests_v6"),
        "-p",
        "test_*.py",
        "-v",
    ]
    print(f"RUN: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(TRAIN_DIR))
    if proc.returncode != 0:
        raise SystemExit("Internal tests failed.")


def run_orientation_regression(
    orientation_truth: dict,
    images_dir: Path,
    model_path: str,
) -> None:
    failures = []
    for image_name, expected_full in sorted(orientation_truth.items()):
        truth_board = expected_full.split()[0]
        truth_board_rot = rec_v6.v4.rotate_fen_180(truth_board)
        image_path = images_dir / image_name

        pred_white, _ = rec_v6.predict_board(
            str(image_path),
            model_path=model_path,
            board_perspective="white",
        )
        pred_black, _ = rec_v6.predict_board(
            str(image_path),
            model_path=model_path,
            board_perspective="black",
        )
        pred_auto, _ = rec_v6.predict_board(
            str(image_path),
            model_path=model_path,
            board_perspective="auto",
        )
        white_black_consistent = pred_black == rec_v6.v4.rotate_fen_180(pred_white)
        truth_matches_forced = truth_board in {pred_white, pred_black}
        auto_is_forced = pred_auto in {pred_white, pred_black}
        ok = (
            white_black_consistent
            and truth_matches_forced
            and auto_is_forced
        )
        status = "PASS" if ok else "FAIL"
        print(f"[orientation] {image_name} {status}")
        if not ok:
            failures.append(
                {
                    "image": image_name,
                    "truth_board": truth_board,
                    "truth_board_rot": truth_board_rot,
                    "pred_white": pred_white,
                    "pred_black": pred_black,
                    "pred_auto": pred_auto,
                    "white_black_consistent": white_black_consistent,
                    "truth_matches_forced": truth_matches_forced,
                    "auto_is_forced": auto_is_forced,
                }
            )
    if failures:
        raise SystemExit(f"Orientation regression failed on {len(failures)} image(s): {json.dumps(failures, indent=2)}")


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def enforce_quick_baseline(
    quick_truth: dict,
    quick_summary: dict,
    baseline_path: Path,
    quick_truth_json_path: str,
    update_baseline: bool,
) -> None:
    signature = truth_signature(quick_truth)
    baseline = load_json(baseline_path, default=None)
    current_pass = int(quick_summary["board_pass"])
    total = int(quick_summary["images"])

    if baseline is None:
        if not update_baseline:
            raise SystemExit(
                f"Quick baseline missing at {baseline_path}. "
                "Run once with --update-quick-baseline."
            )
        baseline = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "required_pass": current_pass,
            "total": total,
            "truth_signature": signature,
            "truth_json": str(Path(quick_truth_json_path)),
        }
        save_json(baseline_path, baseline)
        print(f"Quick baseline initialized: {current_pass}/{total}")
        return

    if baseline.get("truth_signature") != signature:
        if not update_baseline:
            raise SystemExit(
                "Quick baseline truth signature mismatch. "
                "Update baseline with --update-quick-baseline."
            )
        baseline["truth_signature"] = signature
        baseline["truth_json"] = str(Path(quick_truth_json_path))

    required_pass = int(baseline.get("required_pass", 0))
    if current_pass < required_pass:
        raise SystemExit(
            f"Quick regression: {current_pass}/{total} < baseline {required_pass}/{baseline.get('total', total)}"
        )

    if update_baseline:
        baseline["required_pass"] = current_pass
        baseline["total"] = total
        baseline["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(baseline_path, baseline)
        print(f"Quick baseline updated: {current_pass}/{total}")
    else:
        print(f"Quick baseline check passed: {current_pass}/{total} (required >= {required_pass})")


def run_perf_gate(args: argparse.Namespace) -> None:
    perf_args = argparse.Namespace(
        images_dir=args.images_dir,
        truth_json=args.quick_truth_json,
        model_path=args.model_path,
        board_perspective="auto",
        timeout_sec=float(args.timeout_sec),
        reports_dir=args.reports_dir,
        baseline_json=str(Path(args.reports_dir) / "v6_perf_baselines.json"),
        report_json=str(Path(args.reports_dir) / "v6_benchmark_latest.json"),
        median_ratio_max=float(args.median_ratio_max),
        p95_ratio_max=float(args.p95_ratio_max),
        update_baseline=bool(args.update_perf_baseline),
    )

    truth = eval_v6.load_truth(Path(perf_args.truth_json))
    fingerprint = benchmark_v6.compute_machine_fingerprint()
    run = benchmark_v6.benchmark_quick_set(
        truth=truth,
        images_dir=Path(perf_args.images_dir),
        model_path=perf_args.model_path,
        board_perspective=perf_args.board_perspective,
        timeout_sec=perf_args.timeout_sec,
    )

    baseline_path = Path(perf_args.baseline_json)
    report_path = Path(perf_args.report_json)
    baselines = benchmark_v6.load_json(baseline_path, default={})
    fid = fingerprint["fingerprint_id"]

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "metrics": run["metrics"],
        "results": run["results"],
        "baseline_check": None,
    }
    if perf_args.update_baseline:
        baselines[fid] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "metrics": run["metrics"],
            "truth_json": perf_args.truth_json,
        }
        benchmark_v6.save_json(baseline_path, baselines)
        report["baseline_check"] = {"mode": "updated", "fingerprint_id": fid}
        benchmark_v6.save_json(report_path, report)
        print(f"Perf baseline updated for fingerprint {fid}")
        return

    if fid not in baselines:
        benchmark_v6.save_json(report_path, report)
        raise SystemExit(
            "Performance baseline missing for this machine. "
            "Run once with --update-perf-baseline."
        )

    passed, checks = benchmark_v6.evaluate_against_baseline(
        metrics=run["metrics"],
        baseline_metrics=baselines[fid]["metrics"],
        median_ratio_max=perf_args.median_ratio_max,
        p95_ratio_max=perf_args.p95_ratio_max,
    )
    report["baseline_check"] = {
        "mode": "compare",
        "fingerprint_id": fid,
        "baseline_metrics": baselines[fid]["metrics"],
        "checks": checks,
        "passed": bool(passed),
    }
    benchmark_v6.save_json(report_path, report)
    if not passed:
        raise SystemExit(
            "Performance regression: "
            f"median_ratio={checks['median_ratio']} p95_ratio={checks['p95_ratio']} "
            f"timeouts_ok={checks['timeouts_ok']}"
        )
    print(
        "Perf baseline check passed: "
        f"median_ratio={checks['median_ratio']} p95_ratio={checks['p95_ratio']}"
    )


def run_full_validators(args: argparse.Namespace) -> None:
    print("Running full hardset evaluator...")
    full_truth = eval_v6.load_truth(Path(args.full_truth_json))
    full_summary = eval_v6.evaluate_hardset(
        truth=full_truth,
        images_dir=Path(args.images_dir),
        model_path=args.model_path,
        board_perspective="auto",
        timeout_sec=float(args.timeout_sec),
        compare_full_fen=False,
        debug=False,
        reports_dir=Path(args.reports_dir),
        write_reports=True,
        show_progress=True,
    )
    print(
        f"Full hardset: board_pass={full_summary['board_pass']}/{full_summary['images']} "
        f"avg_conf={full_summary['avg_confidence']:.4f}"
    )

    print("Running pipeline stage validator...")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "test_pipeline_v6.py"),
        "--truth-json",
        args.full_truth_json,
        "--model-path",
        args.model_path,
    ]
    proc = subprocess.run(cmd, cwd=str(TRAIN_DIR))
    if proc.returncode != 0:
        raise SystemExit("test_pipeline_v6.py failed.")


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"profile={args.profile}")
    print("Step 1/4: internal tests")
    run_internal_tests()

    print("Step 2/4: orientation regression")
    orientation_truth = eval_v6.load_truth(Path(args.orientation_truth_json))
    run_orientation_regression(
        orientation_truth=orientation_truth,
        images_dir=images_dir,
        model_path=args.model_path,
    )

    print("Step 3/4: quick functional baseline gate")
    quick_truth = eval_v6.load_truth(Path(args.quick_truth_json))
    quick_summary = eval_v6.evaluate_hardset(
        truth=quick_truth,
        images_dir=images_dir,
        model_path=args.model_path,
        board_perspective="auto",
        timeout_sec=float(args.timeout_sec),
        compare_full_fen=False,
        debug=False,
        reports_dir=reports_dir,
        write_reports=True,
        show_progress=True,
    )
    print(
        f"Quick hardset: board_pass={quick_summary['board_pass']}/{quick_summary['images']} "
        f"avg_conf={quick_summary['avg_confidence']:.4f}"
    )
    enforce_quick_baseline(
        quick_truth=quick_truth,
        quick_summary=quick_summary,
        baseline_path=Path(args.quick_baseline_json),
        quick_truth_json_path=args.quick_truth_json,
        update_baseline=bool(args.update_quick_baseline),
    )

    print("Step 4/4: performance gate")
    run_perf_gate(args)

    if args.profile == "full":
        run_full_validators(args)

    print("v6 post-change gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
