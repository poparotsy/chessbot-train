#!/usr/bin/env python3
"""Stage-by-stage validation for recognizer_v6 pipeline."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate recognizer_v6 pipeline stage by stage.")
    parser.add_argument(
        "--recognizer-module",
        default="recognizer_v6",
        help="Recognizer module to import (default: recognizer_v6).",
    )
    parser.add_argument(
        "--images-dir",
        default=str(TRAIN_DIR / "images_4_test"),
        help="Images directory (default: images_4_test).",
    )
    parser.add_argument(
        "--truth-json",
        default=str(TRAIN_DIR / "images_4_test" / "truth_verified.json"),
        help="Truth JSON mapping image -> full FEN.",
    )
    parser.add_argument(
        "--model-path",
        default=str(TRAIN_DIR / "models" / "model_hybrid_v5_latest_best.pt"),
        help="Model checkpoint path.",
    )
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Board perspective mode used during candidate and final decode.",
    )
    parser.add_argument(
        "--output-json",
        default=str(TRAIN_DIR / "scripts" / "pipeline_v6_report.json"),
        help="Where to write JSON report.",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=20,
        help="How many failing images to print in summary.",
    )
    parser.add_argument(
        "--min-board-pass",
        type=int,
        default=None,
        help="Optional minimum board-pass threshold.",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if thresholds are not met.",
    )
    parser.add_argument(
        "--min-stage1-corner-pass",
        type=int,
        default=None,
        help="Optional minimum count where stage1 corner detection succeeded.",
    )
    parser.add_argument(
        "--min-stage3-candidate-match-pass",
        type=int,
        default=None,
        help="Optional minimum count where any stage3 candidate exactly matched truth board.",
    )
    return parser.parse_args()


def load_truth(path: str) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid truth JSON: {path}")
    return data


def safe_square_match(rec, fen_a: str, fen_b: str) -> int:
    rows_a = rec.expand_fen_board(fen_a)
    rows_b = rec.expand_fen_board(fen_b)
    if len(rows_a) != 8 or len(rows_b) != 8:
        return 0
    if any(len(r) != 8 for r in rows_a) or any(len(r) != 8 for r in rows_b):
        return 0
    return sum(1 for r in range(8) for c in range(8) if rows_a[r][c] == rows_b[r][c])


def candidate_perspective_context(rec, full_candidate_img, board_perspective: str):
    if board_perspective != "auto":
        return {
            "label_perspective_result": None,
            "labels_absent": True,
            "labels_same": False,
            "forced": board_perspective,
        }

    label_perspective_result = rec.infer_board_perspective_from_labels(full_candidate_img)
    label_details = (
        label_perspective_result["details"]
        if label_perspective_result
        else {
            "left": rec.classify_file_label_crop(rec.extract_file_label_crop(full_candidate_img, side="left")),
            "right": rec.classify_file_label_crop(rec.extract_file_label_crop(full_candidate_img, side="right")),
        }
    )
    labels_absent = label_details["left"] is None and label_details["right"] is None
    labels_same = (
        label_details["left"] is not None
        and label_details["right"] is not None
        and label_details["left"]["label"] == label_details["right"]["label"]
    )
    return {
        "label_perspective_result": label_perspective_result,
        "labels_absent": labels_absent,
        "labels_same": labels_same,
        "forced": None,
    }


def apply_perspective(rec, fen: str, ctx: dict, board_perspective: str):
    if board_perspective in {"white", "black"}:
        detected = board_perspective
        source = "override"
    else:
        if ctx["label_perspective_result"] is None:
            detected = "white"
            source = "default"
            if ctx["labels_absent"] or ctx["labels_same"]:
                fallback = rec.infer_board_perspective_from_piece_distribution(fen)
                if fallback == "black":
                    detected = "black"
                    source = "piece_distribution_fallback"
        else:
            detected = ctx["label_perspective_result"]["perspective"]
            source = ctx["label_perspective_result"]["source"]

    final_fen = rec.rotate_fen_180(fen) if detected == "black" else fen
    return final_fen, detected, source


def evaluate_image(
    rec,
    model,
    device,
    image_path: Path,
    expected_full_fen: str,
    board_perspective: str,
):
    expected_board = expected_full_fen.split()[0]
    img = Image.open(image_path).convert("RGB")
    report = {
        "image": image_path.name,
        "expected_board": expected_board,
    }

    # Stage 1: corner/quad detection
    corners = rec.find_board_corners(img)
    if corners is None:
        report["stage1"] = {"corner_found": False}
    else:
        metrics = rec.compute_quad_metrics(corners, img.size[0], img.size[1])
        report["stage1"] = {
            "corner_found": True,
            "metrics": metrics,
            "trusted": bool(rec.is_warp_geometry_trustworthy(metrics)),
            "relaxed": bool(rec.is_warp_geometry_relaxed(metrics)),
        }

    # Stage 2: candidate generation
    detector = rec.BoardDetector(debug=False)
    candidates = (
        detector.candidate_images(img)
        if rec.USE_EDGE_DETECTION
        else [("full", img, 0.0, True)]
    )
    report["stage2"] = {
        "candidate_count": len(candidates),
        "tags": [tag for tag, *_ in candidates],
        "trusted_non_full_count": sum(1 for tag, _ci, _wq, tr in candidates if tag != "full" and tr),
    }

    full_candidate_img = rec.trim_dark_edge_bars(candidates[0][1].copy())
    perspective_ctx = candidate_perspective_context(rec, full_candidate_img, board_perspective)

    # Stage 3: candidate decode
    cand_reports = []
    any_candidate_match = False
    best_candidate_match = -1
    for tag, candidate_img, warp_quality, warp_trusted in candidates:
        prepared = rec.trim_dark_edge_bars(candidate_img)
        fen_raw, conf, piece_count = rec.infer_fen_on_image_deep_topk(
            prepared,
            model,
            device,
            rec.USE_SQUARE_DETECTION,
        )
        fen_final, detected_perspective, perspective_source = apply_perspective(
            rec, fen_raw, perspective_ctx, board_perspective
        )
        match = safe_square_match(rec, fen_final, expected_board)
        matched = fen_final == expected_board
        if matched:
            any_candidate_match = True
        best_candidate_match = max(best_candidate_match, match)
        cand_reports.append(
            {
                "tag": tag,
                "warp_quality": float(warp_quality),
                "warp_trusted": bool(warp_trusted),
                "fen_raw": fen_raw,
                "fen_final": fen_final,
                "confidence": float(conf),
                "piece_count": int(piece_count),
                "detected_perspective": detected_perspective,
                "perspective_source": perspective_source,
                "square_match": int(match),
                "exact_match": bool(matched),
            }
        )

    report["stage3"] = {
        "any_candidate_match": any_candidate_match,
        "best_candidate_square_match": best_candidate_match,
        "candidates": cand_reports,
    }

    # Stage 4: final end-to-end prediction
    final_board, final_conf = rec.predict_board(
        str(image_path),
        model_path=args.model_path,
        board_perspective=board_perspective,
    )
    report["stage4"] = {
        "predicted_board": final_board,
        "confidence": float(final_conf),
        "exact_match": bool(final_board == expected_board),
        "square_match": int(safe_square_match(rec, final_board, expected_board)),
    }

    # Failure classification
    if report["stage4"]["exact_match"]:
        cause = "pass"
    elif len(candidates) <= 1:
        cause = "no_board_candidate"
    elif any_candidate_match:
        cause = "selection_error"
    else:
        cause = "candidate_decode_error"
    report["failure_cause"] = cause
    return report


def summarize(results: list[dict]):
    total = len(results)
    board_pass = sum(1 for r in results if r["stage4"]["exact_match"])
    stage1_corner_pass = sum(1 for r in results if r.get("stage1", {}).get("corner_found"))
    stage3_candidate_match_pass = sum(1 for r in results if r.get("stage3", {}).get("any_candidate_match"))
    cause_counts = {}
    for r in results:
        key = r["failure_cause"]
        cause_counts[key] = cause_counts.get(key, 0) + 1
    return {
        "images": total,
        "board_pass": board_pass,
        "board_fail": total - board_pass,
        "stage1_corner_pass": stage1_corner_pass,
        "stage1_corner_fail": total - stage1_corner_pass,
        "stage3_candidate_match_pass": stage3_candidate_match_pass,
        "stage3_candidate_match_fail": total - stage3_candidate_match_pass,
        "cause_counts": cause_counts,
    }


def main():
    global args
    args = parse_args()
    truth = load_truth(args.truth_json)
    rec = importlib.import_module(args.recognizer_module)

    rec.DEBUG_MODE = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = rec.StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    images_dir = Path(args.images_dir)
    results = []
    for image_name, expected_fen in sorted(truth.items()):
        image_path = images_dir / image_name
        if not image_path.exists():
            results.append(
                {
                    "image": image_name,
                    "expected_board": expected_fen.split()[0],
                    "failure_cause": "missing_image",
                    "stage4": {"exact_match": False, "predicted_board": None, "confidence": 0.0},
                    "stage2": {"candidate_count": 0, "tags": [], "trusted_non_full_count": 0},
                    "stage3": {"any_candidate_match": False, "best_candidate_square_match": 0, "candidates": []},
                }
            )
            continue
        try:
            results.append(
                evaluate_image(
                    rec=rec,
                    model=model,
                    device=device,
                    image_path=image_path,
                    expected_full_fen=expected_fen,
                    board_perspective=args.board_perspective,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "image": image_name,
                    "expected_board": expected_fen.split()[0],
                    "failure_cause": "runtime_error",
                    "error": str(exc),
                    "stage4": {"exact_match": False, "predicted_board": None, "confidence": 0.0},
                    "stage2": {"candidate_count": 0, "tags": [], "trusted_non_full_count": 0},
                    "stage3": {"any_candidate_match": False, "best_candidate_square_match": 0, "candidates": []},
                }
            )

    summary = summarize(results)
    report = {
        "recognizer_module": args.recognizer_module,
        "model_path": args.model_path,
        "board_perspective": args.board_perspective,
        "summary": summary,
        "results": results,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"images={summary['images']}")
    print(f"board_pass={summary['board_pass']}/{summary['images']}")
    print(f"stage1_corner_pass={summary['stage1_corner_pass']}/{summary['images']}")
    print(f"stage3_candidate_match_pass={summary['stage3_candidate_match_pass']}/{summary['images']}")
    print(f"causes={json.dumps(summary['cause_counts'], sort_keys=True)}")
    fails = [r for r in results if not r["stage4"]["exact_match"]]
    for row in fails[: args.show_failures]:
        print(
            f"FAIL {row['image']} cause={row['failure_cause']} "
            f"pred={row['stage4'].get('predicted_board')}"
        )
    print(f"report={out_path}")

    if args.fail_on_regression:
        if args.min_board_pass is not None and summary["board_pass"] < args.min_board_pass:
            raise SystemExit(
                f"Regression: board_pass {summary['board_pass']} < min_board_pass {args.min_board_pass}"
            )
        if (
            args.min_stage1_corner_pass is not None
            and summary["stage1_corner_pass"] < args.min_stage1_corner_pass
        ):
            raise SystemExit(
                "Regression: stage1_corner_pass "
                f"{summary['stage1_corner_pass']} < min_stage1_corner_pass {args.min_stage1_corner_pass}"
            )
        if (
            args.min_stage3_candidate_match_pass is not None
            and summary["stage3_candidate_match_pass"] < args.min_stage3_candidate_match_pass
        ):
            raise SystemExit(
                "Regression: stage3_candidate_match_pass "
                f"{summary['stage3_candidate_match_pass']} < "
                f"min_stage3_candidate_match_pass {args.min_stage3_candidate_match_pass}"
            )


if __name__ == "__main__":
    main()
