#!/usr/bin/env python3
"""Evaluate real-world temp blockers for recognizer_v6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_v6_hardset as eval_v6


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
DEFAULT_MANIFEST = TRAIN_DIR / "scripts" / "testdata" / "v6_temp_canaries.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate real-world temp blockers for recognizer_v6.")
    parser.add_argument("--manifest-json", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--model-path", default=None, help="Optional model checkpoint override.")
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Board perspective mode passed through to recognizer_v6.",
    )
    parser.add_argument("--timeout-sec", type=float, default=45.0, help="Per-image timeout in seconds.")
    parser.add_argument(
        "--script",
        default=str(TRAIN_DIR / "recognizer_v6.py"),
        help="Path to recognizer script module.",
    )
    parser.add_argument("--show-progress", action="store_true", help="Show per-image progress for locked blockers.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"manifest JSON must be an object: {path}")
    if "locked_cases" not in raw or "pending_cases" not in raw:
        raise ValueError(f"manifest JSON must contain locked_cases and pending_cases: {path}")
    return raw


def _normalize_locked_cases(raw_cases: list[dict]) -> list[dict]:
    cases = []
    seen = set()
    for row in raw_cases:
        blocker_id = str(row["blocker_id"])
        if blocker_id in seen:
            raise ValueError(f"duplicate blocker_id in locked_cases: {blocker_id}")
        seen.add(blocker_id)
        cases.append(
            {
                "blocker_id": blocker_id,
                "source_path": str(row["source_path"]),
                "truth_fen": str(row["truth_fen"]),
                "note": str(row.get("note", "")),
            }
        )
    return cases


def _normalize_pending_cases(raw_cases: list[dict]) -> list[dict]:
    cases = []
    seen = set()
    for row in raw_cases:
        blocker_id = str(row["blocker_id"])
        if blocker_id in seen:
            raise ValueError(f"duplicate blocker_id in pending_cases: {blocker_id}")
        seen.add(blocker_id)
        cases.append(
            {
                "blocker_id": blocker_id,
                "source_path": str(row["source_path"]),
                "note": str(row.get("note", "")),
            }
        )
    return cases


def main() -> int:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest_json))
    locked_cases = _normalize_locked_cases(manifest.get("locked_cases", []))
    pending_cases = _normalize_pending_cases(manifest.get("pending_cases", []))

    locked_results = []
    locked_pass = 0
    total_conf = 0.0
    count_conf = 0
    for idx, row in enumerate(locked_cases, start=1):
        image_path = TRAIN_DIR / row["source_path"]
        payload, elapsed = eval_v6.run_recognizer_once(
            image_path=image_path,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
            timeout_sec=float(args.timeout_sec),
            debug=False,
            script_path=args.script,
            preserve_full_fen=True,
        )
        result = {
            "blocker_id": row["blocker_id"],
            "source_path": row["source_path"],
            "truth_fen": row["truth_fen"],
            "note": row["note"],
            "elapsed_sec": round(float(elapsed), 4),
            "success": bool(payload.get("success", False)),
        }
        if not payload.get("success", False):
            result["error"] = payload.get("error")
            locked_results.append(result)
            if args.show_progress:
                print(
                    f"[{idx:02d}/{len(locked_cases)}] {row['blocker_id']} ERROR {result['error']}",
                    flush=True,
                )
            continue

        predicted_full = str(payload.get("fen") or "")
        predicted_board = predicted_full.split()[0] if predicted_full else ""
        expected_full = row["truth_fen"]
        expected_board = expected_full.split()[0]
        conf = float(payload.get("confidence", 0.0))
        total_conf += conf
        count_conf += 1
        full_ok = predicted_full == expected_full
        board_ok = predicted_board == expected_board
        if full_ok:
            locked_pass += 1
        result.update(
            {
                "confidence": round(conf, 4),
                "predicted_full": predicted_full,
                "predicted_board": predicted_board,
                "expected_full": expected_full,
                "expected_board": expected_board,
                "full_ok": full_ok,
                "board_ok": board_ok,
                "side_to_move_source": payload.get("side_to_move_source"),
                "best_tag": payload.get("best_tag"),
                "detector_score": payload.get("detector_score"),
                "detector_support": payload.get("detector_support"),
                "value_case_fused": payload.get("value_case_fused"),
                "detected_perspective": payload.get("detected_perspective"),
                "perspective_source": payload.get("perspective_source"),
            }
        )
        locked_results.append(result)
        if args.show_progress:
            status = "PASS" if full_ok else "FAIL"
            print(
                f"[{idx:02d}/{len(locked_cases)}] {row['blocker_id']} {status} "
                f"conf={result['confidence']:.4f} t={result['elapsed_sec']}s",
                flush=True,
            )

    pending_results = []
    for row in pending_cases:
        image_path = TRAIN_DIR / row["source_path"]
        payload, elapsed = eval_v6.run_recognizer_once(
            image_path=image_path,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
            timeout_sec=float(args.timeout_sec),
            debug=False,
            script_path=args.script,
            preserve_full_fen=True,
        )
        pending_results.append(
            {
                "blocker_id": row["blocker_id"],
                "source_path": row["source_path"],
                "note": row["note"],
                "elapsed_sec": round(float(elapsed), 4),
                "success": bool(payload.get("success", False)),
                "fen": payload.get("fen"),
                "confidence": payload.get("confidence"),
                "error": payload.get("error"),
            }
        )

    locked_failures = [row for row in locked_results if not row.get("full_ok", False)]
    locked_summary = {
        "images": len(locked_cases),
        "full_pass": locked_pass,
        "avg_confidence": (total_conf / count_conf) if count_conf else 0.0,
        "results": locked_results,
        "failures": locked_failures,
    }
    print("=== TEMP BLOCKERS ===")
    print(
        f"locked_pass={locked_summary['full_pass']}/{locked_summary['images']} "
        f"(avg_conf={locked_summary.get('avg_confidence', 0.0):.4f})"
    )
    if locked_failures:
        print("locked_failures:")
        for row in locked_failures:
            print(
                f"  {row['blocker_id']} {row['source_path']}: "
                f"predicted={row.get('predicted_full')} expected={row.get('expected_full')}"
            )

    if pending_results:
        print("pending_cases:")
        for row in pending_results:
            if row["success"]:
                print(
                    f"  {row['blocker_id']} {row['source_path']}: "
                    f"{row['fen']} conf={float(row['confidence'] or 0.0):.4f}"
                )
            else:
                print(f"  {row['blocker_id']} {row['source_path']}: ERROR {row['error']}")

    aggregate = {
        "manifest_json": args.manifest_json,
        "locked_summary": locked_summary,
        "pending_results": pending_results,
    }
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
