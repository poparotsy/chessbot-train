#!/usr/bin/env python3
"""Performance benchmark and relative-baseline gate for recognizer_v6 quick set."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_v6_hardset as eval_v6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark recognizer_v6 latency on quick test set.")
    parser.add_argument("--images-dir", default=str(TRAIN_DIR / "images_4_test"))
    parser.add_argument("--truth-json", default=str(SCRIPT_DIR / "testdata" / "v6_quick_cases.json"))
    parser.add_argument("--model-path", default=str(TRAIN_DIR / "models" / "model_hybrid_v5_latest_best.pt"))
    parser.add_argument("--board-perspective", choices=["auto", "white", "black"], default="auto")
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    parser.add_argument("--reports-dir", default=str(TRAIN_DIR / "reports"))
    parser.add_argument("--baseline-json", default=str(TRAIN_DIR / "reports" / "v6_perf_baselines.json"))
    parser.add_argument("--report-json", default=str(TRAIN_DIR / "reports" / "v6_benchmark_latest.json"))
    parser.add_argument("--median-ratio-max", type=float, default=1.25)
    parser.add_argument("--p95-ratio-max", type=float, default=1.35)
    parser.add_argument("--update-baseline", action="store_true")
    return parser.parse_args()


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = max(0.0, min(100.0, float(pct))) / 100.0 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return float(ordered[lo] + frac * (ordered[hi] - ordered[lo]))


def compute_machine_fingerprint() -> Dict:
    gpu_names = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(idx))
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": gpu_names,
    }
    payload_json = json.dumps(payload, sort_keys=True)
    payload["fingerprint_id"] = hashlib.sha1(payload_json.encode("utf-8")).hexdigest()[:16]
    return payload


def benchmark_quick_set(
    truth: Dict[str, str],
    images_dir: Path,
    model_path: str,
    board_perspective: str,
    timeout_sec: float,
) -> Dict:
    results = []
    latencies = []
    timeouts = 0
    board_pass = 0
    for idx, image_name in enumerate(sorted(truth.keys()), start=1):
        expected_board = truth[image_name].split()[0]
        image_path = images_dir / image_name
        payload, elapsed = eval_v6.run_recognizer_once(
            image_path=image_path,
            model_path=model_path,
            board_perspective=board_perspective,
            timeout_sec=timeout_sec,
            debug=False,
        )
        row = {
            "image": image_name,
            "elapsed_sec": round(float(elapsed), 4),
            "success": bool(payload.get("success", False)),
            "error": payload.get("error"),
            "confidence": float(payload.get("confidence", 0.0)) if payload.get("success") else 0.0,
        }
        if payload.get("success", False):
            pred_board = str(payload.get("fen", "")).split()[0]
            board_ok = pred_board == expected_board
            row["predicted_board"] = pred_board
            row["board_ok"] = bool(board_ok)
            board_pass += int(board_ok)
        else:
            board_ok = False
            if "timeout" in str(payload.get("error", "")).lower():
                timeouts += 1
        results.append(row)
        latencies.append(float(elapsed))
        print(
            f"[{idx:02d}/{len(truth)}] {image_name} "
            f"{'PASS' if board_ok else 'FAIL'} t={elapsed:.3f}s",
            flush=True,
        )

    metrics = {
        "images": len(truth),
        "board_pass": board_pass,
        "timeouts": timeouts,
        "median_sec": round(float(statistics.median(latencies)) if latencies else 0.0, 6),
        "p95_sec": round(percentile(latencies, 95.0), 6),
        "max_sec": round(max(latencies) if latencies else 0.0, 6),
    }
    return {"metrics": metrics, "results": results}


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_against_baseline(
    metrics: Dict,
    baseline_metrics: Dict,
    median_ratio_max: float,
    p95_ratio_max: float,
) -> Tuple[bool, Dict]:
    median_ratio = float("inf")
    p95_ratio = float("inf")
    baseline_median = float(baseline_metrics.get("median_sec", 0.0))
    baseline_p95 = float(baseline_metrics.get("p95_sec", 0.0))
    if baseline_median > 0:
        median_ratio = float(metrics["median_sec"]) / baseline_median
    if baseline_p95 > 0:
        p95_ratio = float(metrics["p95_sec"]) / baseline_p95
    checks = {
        "timeouts_ok": int(metrics["timeouts"]) == 0,
        "median_ok": median_ratio <= float(median_ratio_max),
        "p95_ok": p95_ratio <= float(p95_ratio_max),
        "median_ratio": round(median_ratio, 6) if median_ratio != float("inf") else None,
        "p95_ratio": round(p95_ratio, 6) if p95_ratio != float("inf") else None,
    }
    passed = checks["timeouts_ok"] and checks["median_ok"] and checks["p95_ok"]
    return passed, checks


def main() -> int:
    args = parse_args()
    truth = eval_v6.load_truth(Path(args.truth_json))
    images_dir = Path(args.images_dir)
    report_path = Path(args.report_json)
    baseline_path = Path(args.baseline_json)
    fingerprint = compute_machine_fingerprint()

    run = benchmark_quick_set(
        truth=truth,
        images_dir=images_dir,
        model_path=args.model_path,
        board_perspective=args.board_perspective,
        timeout_sec=float(args.timeout_sec),
    )

    run_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "inputs": {
            "truth_json": str(Path(args.truth_json)),
            "images_dir": str(images_dir),
            "model_path": str(args.model_path),
            "board_perspective": args.board_perspective,
            "timeout_sec": float(args.timeout_sec),
        },
        "metrics": run["metrics"],
        "results": run["results"],
        "baseline_check": None,
    }

    baselines = load_json(baseline_path, default={})
    fid = fingerprint["fingerprint_id"]

    if args.update_baseline:
        baselines[fid] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "metrics": run["metrics"],
            "truth_json": str(Path(args.truth_json)),
        }
        save_json(baseline_path, baselines)
        run_report["baseline_check"] = {"mode": "updated", "fingerprint_id": fid}
        save_json(report_path, run_report)
        print(f"Updated baseline for fingerprint {fid}")
        return 0

    if fid not in baselines:
        run_report["baseline_check"] = {
            "mode": "missing",
            "fingerprint_id": fid,
            "error": "No baseline for current machine fingerprint. Run with --update-baseline first.",
        }
        save_json(report_path, run_report)
        print("ERROR: missing performance baseline for this machine. Run with --update-baseline.")
        return 2

    baseline_metrics = baselines[fid]["metrics"]
    passed, checks = evaluate_against_baseline(
        metrics=run["metrics"],
        baseline_metrics=baseline_metrics,
        median_ratio_max=float(args.median_ratio_max),
        p95_ratio_max=float(args.p95_ratio_max),
    )
    run_report["baseline_check"] = {
        "mode": "compare",
        "fingerprint_id": fid,
        "baseline_metrics": baseline_metrics,
        "checks": checks,
        "passed": bool(passed),
    }
    save_json(report_path, run_report)

    print(
        f"metrics: median={run['metrics']['median_sec']:.3f}s "
        f"p95={run['metrics']['p95_sec']:.3f}s max={run['metrics']['max_sec']:.3f}s "
        f"timeouts={run['metrics']['timeouts']}"
    )
    print(
        "baseline-check: "
        f"median_ratio={checks['median_ratio']} p95_ratio={checks['p95_ratio']} "
        f"passed={passed}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
