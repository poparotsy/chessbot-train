#!/usr/bin/env python3
"""Deep miss diagnostic for recognizer_v6.

This is the go-to tool when a hardset image misses.

Protocol:
1. Analyze the recognizer-selected candidate.
2. Analyze the same model on the `full` candidate.
3. Compare square-level top-k evidence on only the relevant squares.
4. If provided, compare against a stable baseline model.
5. Classify the failure path so we know whether to fix data/model or recognizer ranking.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
DEFAULT_RECOGNIZER_PATH = TRAIN_DIR / "recognizer_v6.py"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

_RECOGNIZER_MODULE_CACHE: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-diagnose recognizer_v6 misses.")
    parser.add_argument(
        "images",
        nargs="*",
        help="Image names from images_4_test or explicit image paths.",
    )
    parser.add_argument("--images-dir", default=str(TRAIN_DIR / "images_4_test"))
    parser.add_argument("--truth-json", default=str(TRAIN_DIR / "images_4_test" / "truth_verified.json"))
    parser.add_argument(
        "--suite-json",
        default=None,
        help="Optional JSON object of image->truth or category->images to analyze as a set.",
    )
    parser.add_argument(
        "--recognizer-path",
        default=str(DEFAULT_RECOGNIZER_PATH),
        help="Recognizer module to analyze (default: recognizer_v6.py).",
    )
    parser.add_argument("--model-path", required=True, help="Model under diagnosis.")
    parser.add_argument("--baseline-model-path", default=None, help="Optional stable comparison model.")
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Board perspective passed into recognizer_v6.",
    )
    parser.add_argument("--topk", type=int, default=5, help="How many tile alternatives to keep in the report.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON report path. When omitted, prints only the console report.",
    )
    parser.add_argument(
        "--save-crops-dir",
        default=None,
        help="Optional directory to save selected/full candidate crops for visual inspection.",
    )
    parser.add_argument(
        "--detail-level",
        choices=["human", "debug"],
        default="human",
        help="Human keeps reports compact; debug retains heavy internals.",
    )
    return parser.parse_args()


def load_truth(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"Invalid truth JSON: {path}")
    return data


def load_suite_names(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        # Accept either image->truth mapping or category->list[str].
        values = list(raw.values())
        if values and all(isinstance(v, str) for v in values):
            return sorted(str(k) for k in raw.keys())
        names: list[str] = []
        for value in values:
            if isinstance(value, list):
                names.extend(str(item) for item in value)
        if names:
            return sorted(dict.fromkeys(names))
    raise ValueError(f"Unsupported suite JSON format: {path}")


def load_recognizer_module(script_path: str | Path):
    script = str(Path(script_path).resolve())
    cached = _RECOGNIZER_MODULE_CACHE.get(script)
    if cached is not None:
        return cached
    module_name = f"recognizer_diag_{abs(hash(script))}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load recognizer module from: {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _RECOGNIZER_MODULE_CACHE[script] = module
    return module


def expand_fen(fen_board: str) -> list[list[str]]:
    rows = []
    for row in fen_board.split("/"):
        expanded: list[str] = []
        for ch in row:
            if ch.isdigit():
                expanded.extend("1" for _ in range(int(ch)))
            else:
                expanded.append(ch)
        if len(expanded) != 8:
            raise ValueError(f"Bad FEN row: {fen_board}")
        rows.append(expanded)
    if len(rows) != 8:
        raise ValueError(f"Bad FEN board: {fen_board}")
    return rows


def square_name(row: int, col: int) -> str:
    return f"{'abcdefgh'[col]}{8 - row}"


def tile_map(tile_infos: list[dict[str, Any]], perspective: str) -> dict[str, dict[str, Any]]:
    mapped: dict[str, dict[str, Any]] = {}
    for tile in tile_infos:
        row = int(tile["row"])
        col = int(tile["col"])
        if perspective == "black":
            row = 7 - row
            col = 7 - col
        mapped[square_name(row, col)] = tile
    return mapped


def board_labels(fen_board: str) -> dict[str, str]:
    rows = expand_fen(fen_board)
    return {
        square_name(r, c): rows[r][c]
        for r in range(8)
        for c in range(8)
    }


def diff_squares(expected_board: str, actual_board: str) -> list[str]:
    expected = board_labels(expected_board)
    actual = board_labels(actual_board)
    return sorted([sq for sq in expected if expected[sq] != actual.get(sq, "?")])


def classify_failure_path(chosen_ok: bool, full_ok: bool, full_available: bool) -> str:
    if chosen_ok:
        return "no_board_failure"
    if full_available and full_ok:
        return "recognizer_selection_failure"
    if full_available and not full_ok:
        return "model_or_full_crop_failure"
    return "no_full_candidate_available"


def load_model(model_path: str, device: torch.device):
    rec = load_recognizer_module(DEFAULT_RECOGNIZER_PATH)
    model = rec.StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def decode_candidate_with_details(
    candidate: tuple,
    rec,
    model,
    device: torch.device,
    board_perspective: str,
    orientation_context: dict[str, Any],
    topk: int,
) -> dict[str, Any]:
    if hasattr(rec, "diagnostic_decode_candidate_with_details"):
        return rec.diagnostic_decode_candidate_with_details(
            candidate,
            model=model,
            device=device,
            board_perspective=board_perspective,
            orientation_context=orientation_context,
            topk=topk,
        )
    tag, candidate_img, warp_quality, warp_trusted = candidate
    candidate_img = rec.trim_dark_edge_bars(candidate_img)
    variant_inputs: list[tuple[str, Image.Image]] = [("base", candidate_img)]
    if str(tag).startswith("contour_"):
        inner = rec.find_inner_board_window(candidate_img)
        if inner is not None:
            variant_inputs.append(("inner_board", inner))

    variant_rows = []
    for variant_name, variant_img in variant_inputs:
        fen, conf, piece_count, details = rec.infer_fen_on_image_clean(
            variant_img,
            model,
            device,
            rec.USE_SQUARE_DETECTION,
            return_details=True,
            topk_k=max(3, int(topk)),
        )
        sat_stats = rec.image_saturation_stats(variant_img)
        rescored_applied = False
        if sat_stats["sat_mean"] <= rec.LOW_SAT_SPARSE_SAT_MEAN_MAX and piece_count <= rec.LOW_SAT_SPARSE_PIECE_MAX:
            rescored = rec.rescore_low_saturation_sparse_from_topk(
                details.get("tile_infos", []),
                base_fen=fen,
                base_conf=conf,
            )
            if rescored is not None:
                fen, conf, piece_count = rescored
                rescored_applied = True
        variant_rows.append(
            {
                "variant": variant_name,
                "img": variant_img,
                "fen_before_orientation": fen,
                "confidence": float(conf),
                "piece_count": int(piece_count),
                "plausibility": float(rec.board_plausibility_score(fen)),
                "king_health": int(rec.king_health(fen)),
                "tile_infos": details.get("tile_infos", []),
                "rescored_low_sat_sparse": bool(rescored_applied),
                "sat_stats": sat_stats,
            }
        )

    best_variant = max(
        variant_rows,
        key=lambda row: (
            row["plausibility"],
            row["king_health"],
            row["piece_count"] if row["piece_count"] <= rec.LOW_SAT_SPARSE_PIECE_MAX else 0,
            row["confidence"],
        ),
    )
    perspective, perspective_source = rec.resolve_candidate_orientation(
        best_variant["fen_before_orientation"],
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    board_fen = (
        rec.rotate_fen_180(best_variant["fen_before_orientation"])
        if perspective == "black"
        else best_variant["fen_before_orientation"]
    )
    tile_infos_by_square = tile_map(best_variant["tile_infos"], perspective)
    return {
        "tag": tag,
        "variant": best_variant["variant"],
        "board_fen": board_fen,
        "fen_before_orientation": best_variant["fen_before_orientation"],
        "confidence": best_variant["confidence"],
        "piece_count": best_variant["piece_count"],
        "plausibility": best_variant["plausibility"],
        "king_health": best_variant["king_health"],
        "detected_perspective": perspective,
        "perspective_source": perspective_source,
        "warp_quality": float(warp_quality),
        "warp_trusted": bool(warp_trusted),
        "tile_infos_by_square": tile_infos_by_square,
        "rescored_low_sat_sparse": best_variant["rescored_low_sat_sparse"],
        "sat_stats": best_variant["sat_stats"],
        "candidate_img": best_variant["img"],
    }


def enrich_selection_rows(rec, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_raw_conf = max(float(row["confidence"]) for row in rows) if rows else 0.0
    enriched = []
    for row in rows:
        stm, stm_source = rec.infer_side_to_move_from_checks(row["board_fen"])
        conf_adj = float(row["confidence"])
        if "gradient_projection" in str(row["tag"]):
            conf_adj -= 0.010
        if "panel_split" in str(row["tag"]):
            conf_adj -= 0.030
        stats = row["sat_stats"]
        low_sat_sparse = (
            stats["sat_mean"] <= rec.LOW_SAT_SPARSE_SAT_MEAN_MAX
            and row["piece_count"] <= rec.LOW_SAT_SPARSE_PIECE_MAX
        )
        sparse_piece_bonus = row["piece_count"] if low_sat_sparse and row["confidence"] >= (best_raw_conf - 0.025) else 0
        enriched.append(
            {
                **row,
                "side_to_move": stm,
                "side_to_move_source": stm_source,
                "conf_adj": float(conf_adj),
                "sparse_piece_bonus": int(sparse_piece_bonus),
                "selection_key": (
                    row["plausibility"],
                    row["king_health"],
                    sparse_piece_bonus,
                    -int(stm_source == "default_double_check_conflict"),
                    conf_adj,
                    row["warp_quality"],
                ),
            }
        )
    return sorted(enriched, key=lambda row: row["selection_key"], reverse=True)


def topk_to_text(topk_rows: list[tuple[str, float]]) -> str:
    return ", ".join(f"{label}:{prob:.3f}" for label, prob in topk_rows)


def compare_square(
    square: str,
    expected_labels: dict[str, str],
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "square": square,
        "truth": expected_labels.get(square),
        "current": {},
    }
    for candidate_key in ("selected", "full"):
        candidate = current.get(candidate_key)
        if candidate is None:
            row["current"][candidate_key] = None
            continue
        tile = candidate["tile_infos_by_square"].get(square)
        label = board_labels(candidate["board_fen"]).get(square)
        row["current"][candidate_key] = {
            "label": label,
            "topk": tile.get("topk", []) if tile else [],
            "empty_prob": float(tile.get("empty_prob", 0.0)) if tile else None,
            "best_piece_alt_prob": float(tile.get("best_piece_alt_prob", 0.0)) if tile else None,
        }
    if baseline is not None:
        row["baseline"] = {}
        for candidate_key in ("selected", "full"):
            candidate = baseline.get(candidate_key)
            if candidate is None:
                row["baseline"][candidate_key] = None
                continue
            tile = candidate["tile_infos_by_square"].get(square)
            label = board_labels(candidate["board_fen"]).get(square)
            row["baseline"][candidate_key] = {
                "label": label,
                "topk": tile.get("topk", []) if tile else [],
                "empty_prob": float(tile.get("empty_prob", 0.0)) if tile else None,
                "best_piece_alt_prob": float(tile.get("best_piece_alt_prob", 0.0)) if tile else None,
            }
    return row


def summarize_candidate_row(row: dict[str, Any] | None, detail_level: str) -> dict[str, Any] | None:
    if row is None:
        return None
    summary = {
        "tag": row["tag"],
        "variant": row["variant"],
        "board_fen": row["board_fen"],
        "fen_before_orientation": row["fen_before_orientation"],
        "confidence": row["confidence"],
        "piece_count": row["piece_count"],
        "plausibility": row["plausibility"],
        "king_health": row["king_health"],
        "detected_perspective": row["detected_perspective"],
        "perspective_source": row["perspective_source"],
        "warp_quality": row["warp_quality"],
        "warp_trusted": row["warp_trusted"],
        "rescored_low_sat_sparse": row["rescored_low_sat_sparse"],
    }
    if detail_level == "debug":
        summary["sat_stats"] = row["sat_stats"]
        summary["side_to_move_source"] = row.get("side_to_move_source")
        summary["conf_adj"] = row.get("conf_adj")
        summary["sparse_piece_bonus"] = row.get("sparse_piece_bonus")
        summary["grid_score"] = row.get("grid_score")
        summary["outer_ring_occupancy"] = row.get("outer_ring_occupancy")
        summary["trim_box"] = row.get("trim_box")
        summary["refinement_box"] = row.get("refinement_box")
        summary["tile_infos_by_square"] = row.get("tile_infos_by_square")
    return summary


def resolve_image_path(item: str, images_dir: Path) -> Path:
    candidate = Path(item)
    if candidate.exists():
        return candidate.resolve()
    return (images_dir / item).resolve()


def analyze_model_on_image(
    image_path: Path,
    recognizer_path: str,
    model_path: str,
    truth_board: str | None,
    board_perspective: str,
    topk: int,
    detail_level: str,
    save_crops_dir: Path | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rec = load_recognizer_module(recognizer_path)
    model = rec.StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    img = Image.open(image_path).convert("RGB")
    candidates = rec.build_detector_candidates(img)
    orientation_context = rec.collect_orientation_context(candidates, board_perspective=board_perspective)
    decoded = [
        decode_candidate_with_details(
            candidate,
            rec=rec,
            model=model,
            device=device,
            board_perspective=board_perspective,
            orientation_context=orientation_context,
            topk=topk,
        )
        for candidate in candidates
    ]
    ranked = enrich_selection_rows(rec, decoded)
    scored_tuples = [
        (
            row["tag"],
            row["candidate_img"],
            row["board_fen"],
            row["confidence"],
            row["piece_count"],
            row["detected_perspective"],
            row["perspective_source"],
            row["warp_quality"],
            row["warp_trusted"],
        )
        for row in decoded
    ]
    selected_tuple = rec.select_best_candidate(scored_tuples) if scored_tuples else None

    def match_row(target_tuple):
        if target_tuple is None:
            return None
        for row in decoded:
            if (
                row["tag"] == target_tuple[0]
                and row["board_fen"] == target_tuple[2]
                and abs(float(row["confidence"]) - float(target_tuple[3])) <= 1e-9
                and int(row["piece_count"]) == int(target_tuple[4])
                and row["detected_perspective"] == target_tuple[5]
                and row["perspective_source"] == target_tuple[6]
            ):
                return row
        return None

    selected = match_row(selected_tuple)
    full = next((row for row in decoded if row["tag"] == "full"), None)
    expected_labels = board_labels(truth_board) if truth_board else None
    chosen_ok = bool(selected and truth_board and selected["board_fen"] == truth_board)
    full_ok = bool(full and truth_board and full["board_fen"] == truth_board)
    verdict = classify_failure_path(chosen_ok, full_ok, full is not None)

    if save_crops_dir is not None:
        save_crops_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem
        if selected is not None:
            selected["candidate_img"].save(save_crops_dir / f"{stem}_{Path(model_path).stem}_selected.png")
        if full is not None:
            full["candidate_img"].save(save_crops_dir / f"{stem}_{Path(model_path).stem}_full.png")

    result = {
        "model_path": model_path,
        "recognizer_path": str(Path(recognizer_path).resolve()),
        "selected": summarize_candidate_row(selected, detail_level=detail_level),
        "full": summarize_candidate_row(full, detail_level=detail_level),
        "_selected_raw": selected,
        "_full_raw": full,
        "candidate_ranking": [
            {
                "rank": idx,
                "tag": row["tag"],
                "variant": row["variant"],
                "board_fen": row["board_fen"],
                "confidence": round(float(row["confidence"]), 6),
                "conf_adj": round(float(row["conf_adj"]), 6),
                "piece_count": int(row["piece_count"]),
                "plausibility": round(float(row["plausibility"]), 6),
                "king_health": int(row["king_health"]),
                "side_to_move_source": row["side_to_move_source"],
                "warp_quality": round(float(row["warp_quality"]), 6),
                "detected_perspective": row["detected_perspective"],
                "perspective_source": row["perspective_source"],
                "selected": bool(
                    selected is not None
                    and row["tag"] == selected["tag"]
                    and row["board_fen"] == selected["board_fen"]
                    and abs(float(row["confidence"]) - float(selected["confidence"])) <= 1e-9
                ),
                "is_full": bool(row["tag"] == "full"),
                "board_ok": bool(truth_board is not None and row["board_fen"] == truth_board),
            }
            for idx, row in enumerate(ranked, start=1)
            if detail_level == "debug" or idx <= 5
        ],
        "verdict": verdict,
        "chosen_board_ok": bool(chosen_ok),
        "full_board_ok": bool(full_ok),
        "full_available": bool(full is not None),
        "candidate_tags": [row["tag"] for row in ranked],
    }
    return result


def build_focus_squares(
    truth_board: str | None,
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> list[str]:
    focus: set[str] = set()
    if truth_board and current.get("_selected_raw") is not None:
        focus.update(diff_squares(truth_board, current["_selected_raw"]["board_fen"]))
    if truth_board and current.get("_full_raw") is not None:
        focus.update(diff_squares(truth_board, current["_full_raw"]["board_fen"]))
    if baseline:
        if truth_board and baseline.get("_selected_raw") is not None:
            focus.update(diff_squares(truth_board, baseline["_selected_raw"]["board_fen"]))
        if truth_board and baseline.get("_full_raw") is not None:
            focus.update(diff_squares(truth_board, baseline["_full_raw"]["board_fen"]))
    return sorted(focus)


def analyze_image(
    image_path: Path,
    truth_full: str | None,
    recognizer_path: str,
    model_path: str,
    baseline_model_path: str | None,
    board_perspective: str,
    topk: int,
    detail_level: str,
    save_crops_dir: Path | None,
) -> dict[str, Any]:
    truth_board = truth_full.split()[0] if truth_full else None
    current = analyze_model_on_image(
        image_path=image_path,
        recognizer_path=recognizer_path,
        model_path=model_path,
        truth_board=truth_board,
        board_perspective=board_perspective,
        topk=topk,
        detail_level=detail_level,
        save_crops_dir=save_crops_dir,
    )
    baseline = None
    if baseline_model_path:
        baseline = analyze_model_on_image(
            image_path=image_path,
            recognizer_path=recognizer_path,
            model_path=baseline_model_path,
            truth_board=truth_board,
            board_perspective=board_perspective,
            topk=topk,
            detail_level=detail_level,
            save_crops_dir=save_crops_dir,
        )
    expected_labels = board_labels(truth_board) if truth_board else {}
    focus_squares = build_focus_squares(truth_board, current, baseline)
    square_rows = [
        compare_square(
            square,
            expected_labels,
            current={
                "selected": current.get("_selected_raw"),
                "full": current.get("_full_raw"),
            },
            baseline=(
                None
                if baseline is None
                else {
                    "selected": baseline.get("_selected_raw"),
                    "full": baseline.get("_full_raw"),
                }
            ),
        )
        for square in focus_squares
    ]
    current.pop("_selected_raw", None)
    current.pop("_full_raw", None)
    if baseline is not None:
        baseline.pop("_selected_raw", None)
        baseline.pop("_full_raw", None)
    return {
        "image": str(image_path),
        "truth_full": truth_full,
        "truth_board": truth_board,
        "current_model": current,
        "baseline_model": baseline,
        "focus_squares": square_rows,
    }


def write_console_report(report: dict[str, Any]) -> None:
    for image_row in report["images"]:
        print("")
        print(f"=== {Path(image_row['image']).name} ===")
        if image_row["truth_board"]:
            print(f"truth_board: {image_row['truth_board']}")

        current = image_row["current_model"]
        chosen = current["selected"]
        full = current["full"]
        print(f"model: {current['model_path']}")
        print(
            f"chosen: tag={chosen['tag']} variant={chosen['variant']} "
            f"fen={chosen['board_fen']} conf={chosen['confidence']:.4f}"
        )
        if full is not None:
            print(
                f"full:   tag={full['tag']} variant={full['variant']} "
                f"fen={full['board_fen']} conf={full['confidence']:.4f}"
            )
        print(f"verdict: {current['verdict']}")

        baseline = image_row.get("baseline_model")
        if baseline:
            base_sel = baseline["selected"]
            print(
                f"baseline: tag={base_sel['tag']} variant={base_sel['variant']} "
                f"fen={base_sel['board_fen']} conf={base_sel['confidence']:.4f}"
            )

        print("top candidates:")
        for row in current["candidate_ranking"][:5]:
            marker = "*" if row["selected"] else " "
            full_marker = " full" if row["is_full"] else ""
            board_ok = " PASS" if row["board_ok"] else ""
            print(
                f"  {marker}#{row['rank']} {row['tag']:<20} conf={row['confidence']:.4f} "
                f"adj={row['conf_adj']:.4f} plaus={row['plausibility']:.2f} "
                f"pieces={row['piece_count']:>2}{full_marker}{board_ok}"
            )

        if not image_row["focus_squares"]:
            print("focus_squares: none")
            continue
        print("focus_squares:")
        for square_row in image_row["focus_squares"]:
            square = square_row["square"]
            truth = square_row["truth"]
            cur_sel = square_row["current"]["selected"]
            cur_full = square_row["current"]["full"]
            print(f"  - {square} truth={truth}")
            if cur_sel is not None:
                print(
                    f"    current:selected label={cur_sel['label']} "
                    f"[{topk_to_text(cur_sel['topk'])}]"
                )
            if cur_full is not None:
                print(
                    f"    current:full     label={cur_full['label']} "
                    f"[{topk_to_text(cur_full['topk'])}]"
                )
            if baseline:
                base_sel = square_row["baseline"]["selected"]
                base_full = square_row["baseline"]["full"]
                if base_sel is not None:
                    print(
                        f"    baseline:selected label={base_sel['label']} "
                        f"[{topk_to_text(base_sel['topk'])}]"
                    )
                if base_full is not None:
                    print(
                        f"    baseline:full     label={base_full['label']} "
                        f"[{topk_to_text(base_full['topk'])}]"
                    )


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir)
    truth = load_truth(Path(args.truth_json))
    save_crops_dir = Path(args.save_crops_dir) if args.save_crops_dir else None

    image_items = list(args.images)
    if args.suite_json:
        image_items.extend(load_suite_names(Path(args.suite_json)))
    if not image_items:
        raise SystemExit("Provide image paths or --suite-json.")
    image_items = list(dict.fromkeys(image_items))

    rows = []
    for item in image_items:
        image_path = resolve_image_path(item, images_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {item}")
        truth_full = truth.get(image_path.name)
        rows.append(
            analyze_image(
                image_path=image_path,
                truth_full=truth_full,
                recognizer_path=args.recognizer_path,
                model_path=args.model_path,
                baseline_model_path=args.baseline_model_path,
                board_perspective=args.board_perspective,
                topk=args.topk,
                detail_level=args.detail_level,
                save_crops_dir=save_crops_dir,
            )
        )

    report = {
        "tool": "deep_diagnostic_v6",
        "recognizer_path": str(Path(args.recognizer_path).resolve()),
        "suite_json": args.suite_json,
        "detail_level": args.detail_level,
        "board_perspective": args.board_perspective,
        "model_path": args.model_path,
        "baseline_model_path": args.baseline_model_path,
        "images": rows,
    }
    write_console_report(report)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("")
        print(f"report_json: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
