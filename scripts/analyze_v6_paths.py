#!/usr/bin/env python3
"""Per-image execution path analysis for recognizer_v6."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
DEFAULT_IMAGES_DIR = TRAIN_DIR / "images_4_test"
DEFAULT_TRUTH_JSON = DEFAULT_IMAGES_DIR / "truth_verified.json"
DEFAULT_MODEL_PATH = TRAIN_DIR / "models" / "model_hybrid_v5_latest_best.pt"
DEFAULT_RECOGNIZER = TRAIN_DIR / "recognizer_v6.py"
DEFAULT_REPORT_JSON = TRAIN_DIR / "reports" / "v6_path_analysis_latest.json"
DEFAULT_REPORT_MD = TRAIN_DIR / "reports" / "v6_path_analysis_latest.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze recognizer_v6 path image-by-image.")
    parser.add_argument("--images-dir", default=str(DEFAULT_IMAGES_DIR))
    parser.add_argument("--truth-json", default=str(DEFAULT_TRUTH_JSON))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--recognizer-path", default=str(DEFAULT_RECOGNIZER))
    parser.add_argument("--board-perspective", choices=["auto", "white", "black"], default="auto")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--output-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_REPORT_MD))
    return parser.parse_args()


def load_truth(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid truth mapping: {path}")
    return data


def tag_family(tag: str) -> str:
    if not tag:
        return "unknown"
    core = tag
    for suffix in ("_enhsrc", "_inset2", "_trim8", "_gfit", "_relaxed"):
        core = core.replace(suffix, "")
    if core.startswith("full"):
        return "full"
    if core.startswith("axis_"):
        return "axis_grid"
    if core.startswith("gradient_projection"):
        return "gradient_projection"
    if core.startswith("panel_split_"):
        return "panel_split"
    if core.startswith("lattice"):
        return "lattice"
    if core.startswith("contour"):
        return "contour"
    return core.split("_")[0]


def parse_debug_events(stderr_text: str) -> list[dict]:
    events: list[dict] = []
    for line in stderr_text.splitlines():
        line = line.strip()
        if not line.startswith("DEBUG_JSON "):
            continue
        payload = line[len("DEBUG_JSON ") :]
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def safe_float(v, fallback=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(fallback)


def run_one(
    recognizer_path: Path,
    image_path: Path,
    model_path: Path,
    board_perspective: str,
    timeout_sec: float,
) -> dict:
    cmd = [
        sys.executable,
        str(recognizer_path),
        str(image_path),
        "--model-path",
        str(model_path),
        "--board-perspective",
        board_perspective,
        "--debug",
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - t0
        return {
            "success": False,
            "error": f"timeout after {timeout_sec}s",
            "elapsed_sec": elapsed,
            "events": [],
            "raw_stdout": "",
            "raw_stderr": "",
        }

    elapsed = time.perf_counter() - t0
    events = parse_debug_events(proc.stderr or "")
    output_line = ""
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line:
            output_line = line
    payload = None
    if output_line:
        try:
            payload = json.loads(output_line)
        except json.JSONDecodeError:
            payload = None
    if not isinstance(payload, dict):
        payload = {
            "success": False,
            "error": f"invalid recognizer output (rc={proc.returncode})",
        }

    return {
        "success": bool(payload.get("success", False)),
        "payload": payload,
        "elapsed_sec": elapsed,
        "events": events,
        "raw_stdout": proc.stdout or "",
        "raw_stderr": proc.stderr or "",
    }


def analyze_image(
    image_name: str,
    expected_full: str,
    run_result: dict,
) -> dict:
    expected_board = expected_full.split()[0]
    expected_stm = expected_full.split()[1] if len(expected_full.split()) > 1 else None

    events = run_result.get("events", [])
    decode_events = [e for e in events if e.get("kind") == "v6_candidate_decode"]
    selected_events = [e for e in events if e.get("kind") == "v6_selected_candidate"]
    orientation_events = [e for e in events if e.get("kind") == "v6_orientation_fallback"]
    cap_events = [e for e in events if e.get("kind") == "v6_candidate_cap"]
    pool_events = [e for e in events if e.get("kind") == "v6_candidate_pool"]
    enhance_events = [e for e in events if e.get("kind") == "v6_low_sat_enhance"]

    payload = run_result.get("payload", {})
    fen_full = payload.get("fen") if isinstance(payload, dict) else None
    predicted_board = None
    predicted_stm = None
    if isinstance(fen_full, str) and fen_full.strip():
        parts = fen_full.split()
        if parts:
            predicted_board = parts[0]
        if len(parts) > 1:
            predicted_stm = parts[1]

    selected_tag = selected_events[-1].get("tag") if selected_events else None
    selected_decode = next((d for d in decode_events if d.get("tag") == selected_tag), None)

    candidate_conf = [safe_float(d.get("conf")) for d in decode_events]
    top_conf = max(candidate_conf) if candidate_conf else 0.0
    second_conf = sorted(candidate_conf, reverse=True)[1] if len(candidate_conf) > 1 else 0.0
    selected_conf = safe_float(selected_events[-1].get("confidence")) if selected_events else 0.0
    conf_margin = selected_conf - second_conf

    def candidate_key(event: dict) -> tuple[float, int, int, float, float]:
        stm_conflict = str(event.get("stm_source")) == "default_double_check_conflict"
        return (
            safe_float(event.get("plausibility"), -1e9),
            int(event.get("king_health", 0)),
            -int(stm_conflict),
            safe_float(event.get("conf_adj"), safe_float(event.get("conf"))),
            safe_float(event.get("warp_quality")),
        )

    ranked = sorted(decode_events, key=candidate_key, reverse=True)
    selected_rank = None
    if selected_tag:
        for idx, ev in enumerate(ranked, start=1):
            if ev.get("tag") == selected_tag:
                selected_rank = idx
                break

    dropped_candidates = 0
    if cap_events:
        last_cap = cap_events[-1]
        dropped_candidates = max(
            0,
            int(last_cap.get("original_count", 0)) - int(last_cap.get("capped_count", 0)),
        )

    selected_family = tag_family(selected_tag or "")

    risks: list[str] = []
    if selected_conf < 0.90:
        risks.append("low_selected_confidence")
    if len(decode_events) >= 2 and selected_conf < 0.99 and conf_margin < 0.01:
        risks.append("near_tie_low_conf")
    if selected_events and selected_events[-1].get("perspective_source") in {
        "piece_distribution_fallback",
        "weak_label_fallback",
    }:
        risks.append("orientation_nonlabel_heuristic")
    if cap_events:
        cap = cap_events[-1]
        if int(cap.get("original_count", 0)) > int(cap.get("capped_count", 0)) and dropped_candidates >= 6:
            if dropped_candidates >= 20:
                risks.append("candidate_cap_high_pressure")
    if selected_decode and selected_decode.get("rescore_applied"):
        risks.append("special_path_dependency")
    if selected_tag and "relaxed" in selected_tag:
        risks.append("special_path_dependency")
    if selected_tag and ("_gfit" in selected_tag or "_trim8" in selected_tag or "_inset2" in selected_tag):
        risks.append("special_path_dependency")
    if selected_family in {"gradient_projection", "panel_split", "axis_grid"}:
        risks.append("special_path_dependency")
    if selected_rank not in (None, 1):
        top_adj = safe_float(ranked[0].get("conf_adj")) if ranked else 0.0
        if (top_adj - safe_float(selected_decode.get("conf_adj") if selected_decode else selected_conf)) <= 0.005:
            risks.append("selection_inversion_low_delta")
    if not decode_events:
        risks.append("no_decode_events")
    # Keep risk list stable and unique.
    risks = list(dict.fromkeys(risks))

    board_ok = predicted_board == expected_board
    stm_ok = expected_stm is None or predicted_stm == expected_stm

    function_path = ["build_detector_candidates"]
    if enhance_events and bool(enhance_events[-1].get("applied")):
        function_path.append("enhance_low_saturation_image")
    if selected_family == "axis_grid":
        function_path.append("detect_axis_grid_windows")
    elif selected_family == "gradient_projection":
        function_path.append("detect_gradient_projection")
    elif selected_family == "panel_split":
        function_path.append("detect_panel_split")
    elif selected_family == "lattice":
        function_path.append("detect_lattice")
    elif selected_family == "contour":
        function_path.append("detect_contour")
    if selected_tag and "_gfit" in selected_tag:
        function_path.append("_refine_corners_grid_fit")
    if selected_tag and "_trim8" in selected_tag:
        function_path.append("directional_trim_resize")
    if selected_tag and "_inset2" in selected_tag:
        function_path.append("inset_board")
    function_path.extend(
        [
            "infer_fen_on_image_deep_topk",
            "rescore_low_saturation_sparse_from_topk" if (selected_decode and selected_decode.get("rescore_applied")) else "decode_no_rescore",
            "resolve_candidate_orientation",
            "select_best_candidate",
            "infer_side_to_move_from_checks",
        ]
    )

    return {
        "image": image_name,
        "elapsed_sec": round(safe_float(run_result.get("elapsed_sec")), 4),
        "expected": {
            "full": expected_full,
            "board": expected_board,
            "stm": expected_stm,
        },
        "predicted": {
            "full": fen_full,
            "board": predicted_board,
            "stm": predicted_stm,
            "confidence": safe_float(payload.get("confidence")),
            "side_to_move_source": payload.get("side_to_move_source"),
        },
        "pass": {
            "board": bool(board_ok),
            "stm": bool(stm_ok),
            "full": bool(board_ok and stm_ok),
        },
        "path": {
            "selected_tag": selected_tag,
            "selected_family": selected_family,
            "function_path": function_path,
            "selected_perspective": selected_events[-1].get("perspective") if selected_events else None,
            "selected_perspective_source": selected_events[-1].get("perspective_source") if selected_events else None,
            "candidate_count_decoded": len(decode_events),
            "candidate_tags": [d.get("tag") for d in decode_events],
            "low_sat_enhancement_applied": bool(
                enhance_events and bool(enhance_events[-1].get("applied"))
            ),
            "candidate_pool": pool_events[-1] if pool_events else None,
            "candidate_cap": cap_events[-1] if cap_events else None,
            "orientation_events": orientation_events,
        },
        "selection": {
            "selected_conf": round(selected_conf, 6),
            "top_conf": round(top_conf, 6),
            "second_conf": round(second_conf, 6),
            "conf_margin_vs_second": round(conf_margin, 6),
            "selected_rank_by_selection_key": selected_rank,
            "selection_key_top3": [
                {
                    "tag": ev.get("tag"),
                    "plausibility": ev.get("plausibility"),
                    "king_health": ev.get("king_health"),
                    "stm_source": ev.get("stm_source"),
                    "conf_adj": ev.get("conf_adj"),
                    "warp_quality": ev.get("warp_quality"),
                }
                for ev in ranked[:3]
            ],
            "selected_decode_event": selected_decode,
        },
        "risk_flags": risks,
        "error": None if run_result.get("success", False) else run_result.get("error"),
    }


def build_summary(rows: list[dict]) -> dict:
    total = len(rows)
    board_pass = sum(1 for r in rows if r["pass"]["board"])
    stm_pass = sum(1 for r in rows if r["pass"]["stm"])
    full_pass = sum(1 for r in rows if r["pass"]["full"])
    by_family: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    elapsed = []
    board_failures = []
    full_failures = []
    special_dependency_cases = []
    cap_pressure_cases = []
    perf_outliers = []
    for row in rows:
        fam = row["path"]["selected_family"]
        by_family[fam] = by_family.get(fam, 0) + 1
        for rf in row.get("risk_flags", []):
            risk_counts[rf] = risk_counts.get(rf, 0) + 1
        t = safe_float(row.get("elapsed_sec"))
        elapsed.append(t)
        if not row["pass"]["board"]:
            board_failures.append(row["image"])
        if not row["pass"]["full"]:
            full_failures.append(row["image"])
        if "special_path_dependency" in row.get("risk_flags", []):
            special_dependency_cases.append(row["image"])
        if "candidate_cap_high_pressure" in row.get("risk_flags", []):
            cap_pressure_cases.append(row["image"])

    med = statistics.median(elapsed) if elapsed else 0.0
    outlier_thr = med * 1.8 if med > 0 else 0.0
    for row in rows:
        if safe_float(row.get("elapsed_sec")) >= outlier_thr and outlier_thr > 0:
            perf_outliers.append(row["image"])
    return {
        "images": total,
        "board_pass": board_pass,
        "full_pass": full_pass,
        "stm_pass": stm_pass,
        "selected_family_counts": dict(sorted(by_family.items(), key=lambda kv: (-kv[1], kv[0]))),
        "risk_flag_counts": dict(sorted(risk_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "timing_sec": {
            "median": round(med, 4) if elapsed else 0.0,
            "p95": round(sorted(elapsed)[int(max(0, (len(elapsed) - 1) * 0.95))], 4) if elapsed else 0.0,
            "max": round(max(elapsed), 4) if elapsed else 0.0,
        },
        "actionable": {
            "board_failures": board_failures,
            "full_fen_failures": full_failures,
            "special_dependency_cases": special_dependency_cases,
            "candidate_cap_high_pressure_cases": cap_pressure_cases,
            "perf_outliers": perf_outliers,
        },
    }


def write_markdown(path: Path, report: dict) -> None:
    summary = report["summary"]
    rows = report["images"]
    fails = [r for r in rows if not r["pass"]["board"]]
    lines = []
    lines.append("# v6 Path Analysis")
    lines.append("")
    lines.append(f"- images: {summary['images']}")
    lines.append(f"- board_pass: {summary['board_pass']}/{summary['images']}")
    lines.append(f"- full_pass: {summary['full_pass']}/{summary['images']}")
    lines.append(f"- median_sec: {summary['timing_sec']['median']}")
    lines.append(f"- p95_sec: {summary['timing_sec']['p95']}")
    lines.append("")
    lines.append("## Selected Path Families")
    for fam, count in summary["selected_family_counts"].items():
        lines.append(f"- {fam}: {count}")
    lines.append("")
    lines.append("## Risk Flags (Actionable)")
    for flag, count in summary["risk_flag_counts"].items():
        lines.append(f"- {flag}: {count}")
    lines.append("")
    lines.append("## What This Means")
    lines.append(f"- Board recognition is stable: {summary['board_pass']}/{summary['images']}.")
    lines.append(f"- Full FEN misses are side-to-move only: {summary['full_pass']}/{summary['images']}.")
    lines.append("")
    act = summary.get("actionable", {})
    lines.append("## Critical Cases")
    lines.append(f"- full_fen_failures: {', '.join(act.get('full_fen_failures', [])) or 'none'}")
    lines.append(
        f"- special_dependency_cases: {', '.join(act.get('special_dependency_cases', [])) or 'none'}"
    )
    lines.append(
        f"- candidate_cap_high_pressure_cases: {', '.join(act.get('candidate_cap_high_pressure_cases', [])) or 'none'}"
    )
    lines.append(f"- perf_outliers: {', '.join(act.get('perf_outliers', [])) or 'none'}")
    lines.append("")
    lines.append("## Board Failures")
    if not fails:
        lines.append("- none")
    else:
        for row in fails:
            lines.append(
                f"- {row['image']}: selected={row['path']['selected_tag']} "
                f"pred={row['predicted']['board']} expected={row['expected']['board']} "
                f"risks={','.join(row['risk_flags']) or 'none'}"
            )
    lines.append("")
    lines.append("## High-Risk Per-Image")
    high_risk = [r for r in rows if r.get("risk_flags")]
    if not high_risk:
        lines.append("- none")
    else:
        for row in sorted(high_risk, key=lambda r: (-len(r.get("risk_flags", [])), r["image"])):
            lines.append(
                f"- {row['image']}: board={'PASS' if row['pass']['board'] else 'FAIL'} "
                f"tag={row['path']['selected_tag']} "
                f"family={row['path']['selected_family']} "
                f"conf={row['selection']['selected_conf']:.4f} "
                f"t={row['elapsed_sec']:.3f}s "
                f"risks={','.join(row['risk_flags'])}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    truth_json = Path(args.truth_json)
    model_path = Path(args.model_path)
    recognizer_path = Path(args.recognizer_path)

    truth = load_truth(truth_json)
    rows = []
    for image_name, expected_full in sorted(truth.items()):
        image_path = images_dir / image_name
        if not image_path.exists():
            rows.append(
                {
                    "image": image_name,
                    "elapsed_sec": 0.0,
                    "expected": {"full": expected_full, "board": expected_full.split()[0], "stm": None},
                    "predicted": {"full": None, "board": None, "stm": None, "confidence": 0.0, "side_to_move_source": None},
                    "pass": {"board": False, "stm": False, "full": False},
                    "path": {
                        "selected_tag": None,
                        "selected_family": "missing_image",
                        "selected_perspective": None,
                        "selected_perspective_source": None,
                        "candidate_count_decoded": 0,
                        "candidate_tags": [],
                        "low_sat_enhancement_applied": False,
                        "candidate_cap": None,
                        "orientation_events": [],
                    },
                    "selection": {
                        "selected_conf": 0.0,
                        "top_conf": 0.0,
                        "second_conf": 0.0,
                        "conf_margin_vs_second": 0.0,
                        "selected_decode_event": None,
                    },
                    "risk_flags": ["missing_image"],
                    "error": "missing image",
                }
            )
            print(f"{image_name}: MISSING")
            continue

        run_result = run_one(
            recognizer_path=recognizer_path,
            image_path=image_path,
            model_path=model_path,
            board_perspective=args.board_perspective,
            timeout_sec=args.timeout_sec,
        )
        row = analyze_image(image_name=image_name, expected_full=expected_full, run_result=run_result)
        rows.append(row)
        print(
            f"{image_name}: board={'PASS' if row['pass']['board'] else 'FAIL'} "
            f"tag={row['path']['selected_tag']} conf={row['selection']['selected_conf']:.4f} "
            f"t={row['elapsed_sec']:.3f}s"
        )

    summary = build_summary(rows)
    report = {
        "recognizer_path": str(recognizer_path),
        "model_path": str(model_path),
        "images_dir": str(images_dir),
        "truth_json": str(truth_json),
        "board_perspective": args.board_perspective,
        "timeout_sec": args.timeout_sec,
        "summary": summary,
        "images": rows,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    write_markdown(output_md, report)

    print("")
    print(f"images={summary['images']}")
    print(f"board_pass={summary['board_pass']}/{summary['images']}")
    print(f"full_pass={summary['full_pass']}/{summary['images']}")
    print(f"report_json={output_json}")
    print(f"report_md={output_md}")


if __name__ == "__main__":
    main()
