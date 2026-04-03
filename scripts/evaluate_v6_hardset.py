#!/usr/bin/env python3
"""Deterministic hardset evaluator for recognizer_v6."""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
RECOGNIZER_SCRIPT = TRAIN_DIR / "recognizer_v6.py"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
_RECOGNIZER_MODULE_CACHE = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recognizer_v6 on a truth JSON hardset.")
    parser.add_argument("--images-dir", default=str(TRAIN_DIR / "images_4_test"))
    parser.add_argument("--truth-json", default=str(TRAIN_DIR / "images_4_test" / "truth_verified.json"))
    parser.add_argument("--model-path", default=None, help="Optional model checkpoint override.")
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Board perspective mode passed through to recognizer_v6.",
    )
    parser.add_argument("--timeout-sec", type=float, default=45.0, help="Per-image timeout in seconds.")
    parser.add_argument("--reports-dir", default=str(TRAIN_DIR / "reports"))
    parser.add_argument(
        "--script",
        default=str(RECOGNIZER_SCRIPT),
        help="Path to recognizer script module (default: recognizer_v6.py).",
    )
    parser.add_argument(
        "--compare-full-fen",
        action="store_true",
        help="Primary score uses full FEN (including side-to-move) instead of board only.",
    )
    parser.add_argument("--debug", action="store_true", help="Pass --debug to recognizer_v6 subprocess.")
    return parser.parse_args()


def _load_recognizer_module(script_path: str):
    script = str(Path(script_path).resolve())
    cached = _RECOGNIZER_MODULE_CACHE.get(script)
    if cached is not None:
        return cached
    module_name = f"recognizer_dynamic_{abs(hash(script))}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load recognizer module from: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _RECOGNIZER_MODULE_CACHE[script] = module
    return module


def load_truth(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"truth JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"truth JSON must be a non-empty object: {path}")
    return data


def _recognizer_worker(image_path, model_path, board_perspective, debug, script_path, preserve_full_fen, queue):
    try:
        rec = _load_recognizer_module(script_path)
        rec.DEBUG_MODE = bool(debug)
        rec.USE_EDGE_DETECTION = True
        rec.USE_SQUARE_DETECTION = True
        if hasattr(rec, "predict_position"):
            result = rec.predict_position(
                str(image_path),
                model_path=model_path,
                board_perspective=board_perspective,
            )
            board_fen = result["board_fen"]
            predicted_full_fen = str(result.get("fen") or "")
            conf = float(result["confidence"])
            side_to_move = result["side_to_move"]
            side_source = result["side_to_move_source"]
            detected_perspective = result.get("detected_perspective")
            perspective_source = result.get("perspective_source")
        else:
            board_fen, conf = rec.predict_board(
                str(image_path),
                model_path=model_path,
                board_perspective=board_perspective,
            )
            side_to_move, side_source = rec.infer_side_to_move_from_checks(board_fen)
            detected_perspective = None
            perspective_source = None
        queue.put(
            {
                "success": True,
                "fen": predicted_full_fen if preserve_full_fen and predicted_full_fen else f"{board_fen} {side_to_move} - - 0 1",
                "confidence": float(conf),
                "side_to_move": side_to_move,
                "side_to_move_source": side_source,
                "detected_perspective": detected_perspective,
                "perspective_source": perspective_source,
                "best_tag": result.get("best_tag"),
                "detector_score": result.get("detector_score"),
                "detector_support": result.get("detector_support"),
                "value_case_fused": result.get("value_case_fused"),
            }
        )
    except Exception as exc:
        queue.put({"success": False, "error": str(exc)})


def run_recognizer_once(
    image_path: Path,
    model_path: str | None,
    board_perspective: str,
    timeout_sec: float,
    debug: bool,
    script_path: str = str(RECOGNIZER_SCRIPT),
    preserve_full_fen: bool = False,
) -> Tuple[dict, float]:
    start = time.perf_counter()
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_recognizer_worker,
        args=(str(image_path), model_path, board_perspective, bool(debug), str(script_path), bool(preserve_full_fen), queue),
    )
    proc.daemon = True
    proc.start()
    try:
        proc.join(timeout=max(0.1, float(timeout_sec)))
        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)
            payload = {"success": False, "error": f"timeout after {timeout_sec:.2f}s"}
        else:
            if queue.empty():
                payload = {"success": False, "error": "recognizer worker exited without payload"}
            else:
                payload = queue.get()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)

    elapsed = time.perf_counter() - start
    return payload, elapsed


def evaluate_hardset(
    truth: Dict[str, str],
    images_dir: Path,
    model_path: str | None,
    board_perspective: str,
    timeout_sec: float,
    recognizer_script: Path | str = RECOGNIZER_SCRIPT,
    compare_full_fen: bool = False,
    debug: bool = False,
    reports_dir: Path | None = None,
    write_reports: bool = True,
    show_progress: bool = False,
) -> dict:
    names = sorted([name for name in truth if (images_dir / name).exists()])
    total = len(names)
    board_pass = 0
    full_pass = 0
    total_conf = 0.0
    count_conf = 0
    timeout_count = 0
    error_count = 0
    per_image = []
    failures = []

    for idx, name in enumerate(names, start=1):
        expected_full = truth[name]
        expected_board = expected_full.split()[0]
        image_path = images_dir / name

        payload, elapsed = run_recognizer_once(
            image_path=image_path,
            model_path=model_path,
            board_perspective=board_perspective,
            timeout_sec=timeout_sec,
            debug=debug,
            script_path=str(recognizer_script),
        )

        record = {
            "image": name,
            "elapsed_sec": round(float(elapsed), 4),
            "success": bool(payload.get("success", False)),
        }

        if not payload.get("success", False):
            err = str(payload.get("error", "recognizer_failed"))
            record.update(
                {
                    "error": err,
                    "expected_full": expected_full,
                    "expected_board": expected_board,
                }
            )
            failures.append(record)
            per_image.append(record)
            if "timeout" in err:
                timeout_count += 1
            else:
                error_count += 1
            if show_progress:
                print(
                    f"[{idx:02d}/{total}] {name} ERROR {err} t={record['elapsed_sec']}s",
                    flush=True,
                )
            continue

        predicted_full = str(payload.get("fen", ""))
        predicted_board = predicted_full.split()[0] if predicted_full else ""
        conf = float(payload.get("confidence", 0.0))
        total_conf += conf
        count_conf += 1

        board_ok = predicted_board == expected_board
        full_ok = predicted_full == expected_full
        if board_ok:
            board_pass += 1
        if full_ok:
            full_pass += 1

        record.update(
            {
                "confidence": round(conf, 4),
                "predicted_board": predicted_board,
                "predicted_full": predicted_full,
                "expected_board": expected_board,
                "expected_full": expected_full,
                "board_ok": board_ok,
                "full_ok": full_ok,
                "side_to_move_source": payload.get("side_to_move_source"),
                "best_tag": payload.get("best_tag"),
                "detector_score": payload.get("detector_score"),
                "detector_support": payload.get("detector_support"),
                "value_case_fused": payload.get("value_case_fused"),
                "detected_perspective": payload.get("detected_perspective"),
                "perspective_source": payload.get("perspective_source"),
            }
        )
        per_image.append(record)
        primary_ok = full_ok if compare_full_fen else board_ok
        if not primary_ok:
            failures.append(record)
        if show_progress:
            status = "PASS" if primary_ok else "FAIL"
            print(
                f"[{idx:02d}/{total}] {name} {status} conf={record['confidence']:.4f} t={record['elapsed_sec']}s",
                flush=True,
            )

    summary = {
        "evaluator": "evaluate_v6_hardset",
        "recognizer_script": str(Path(recognizer_script)),
        "images_dir": str(images_dir),
        "truth_json": None,
        "model_path": model_path or "(script-default)",
        "board_perspective": board_perspective,
        "timeout_sec": timeout_sec,
        "compare_full_fen": compare_full_fen,
        "images": total,
        "board_pass": board_pass,
        "full_pass": full_pass,
        "primary_pass": full_pass if compare_full_fen else board_pass,
        "avg_confidence": (total_conf / count_conf) if count_conf else 0.0,
        "timeouts": timeout_count,
        "errors": error_count,
        "results": per_image,
    }

    if write_reports and reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        eval_path = reports_dir / "v6_eval_latest.json"
        fail_path = reports_dir / "v6_failures_latest.json"
        eval_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        fail_payload = {
            "images": total,
            "primary_pass": summary["primary_pass"],
            "board_pass": board_pass,
            "full_pass": full_pass,
            "compare_full_fen": compare_full_fen,
            "failures": failures,
        }
        fail_path.write_text(json.dumps(fail_payload, indent=2), encoding="utf-8")
        summary["report_eval"] = str(eval_path)
        summary["report_failures"] = str(fail_path)

    return summary


def main() -> int:
    args = parse_args()
    truth_path = Path(args.truth_json)
    truth = load_truth(truth_path)
    images_dir = Path(args.images_dir)
    reports_dir = Path(args.reports_dir)

    summary = evaluate_hardset(
        truth=truth,
        images_dir=images_dir,
        model_path=args.model_path,
        board_perspective=args.board_perspective,
        timeout_sec=float(args.timeout_sec),
        recognizer_script=Path(args.script),
        compare_full_fen=bool(args.compare_full_fen),
        debug=bool(args.debug),
        reports_dir=reports_dir,
        write_reports=True,
        show_progress=True,
    )
    summary["truth_json"] = str(truth_path)
    if "report_eval" in summary:
        eval_path = Path(summary["report_eval"])
        eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
        eval_data["truth_json"] = str(truth_path)
        eval_path.write_text(json.dumps(eval_data, indent=2), encoding="utf-8")

    total = summary["images"]
    print(f"images={total}")
    print(f"board_pass={summary['board_pass']}/{total}")
    print(f"full_pass={summary['full_pass']}/{total}")
    if summary.get("report_eval"):
        print(f"report_eval={summary['report_eval']}")
    if summary.get("report_failures"):
        print(f"report_failures={summary['report_failures']}")

    failures_payload = json.loads(Path(summary["report_failures"]).read_text(encoding="utf-8"))
    failures = failures_payload.get("failures", [])
    if failures:
        print("\n=== FAILURES ===")
        for item in failures:
            print(json.dumps(item, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
