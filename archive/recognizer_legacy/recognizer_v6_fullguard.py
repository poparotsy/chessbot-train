#!/usr/bin/env python3
"""Recognizer v6 variant with a full-board edge-preservation guard.

Purpose:
- keep the original recognizer easy to roll back to
- make `full` harder to beat when partial candidates delete edge occupancy
  that a plausible `full` candidate preserves
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
from PIL import Image

import recognizer_v6 as base
from recognizer_v6 import *  # noqa: F401,F403

_model_cache = base._model_cache


FULL_GUARD_MAX_CONF_GAP = 0.085
FULL_GUARD_MIN_PLAUSIBILITY = 10.0
FULL_GUARD_MIN_KING_HEALTH = 2
FULL_GUARD_MIN_PIECE_MARGIN = 1
FULL_GUARD_MIN_EDGE_DROP = 2
FULL_GUARD_MIN_EDGE_MISMATCH = 2
FULL_GUARD_FILE_DROP_WEIGHT = 0.50
FULL_GUARD_RANK_DROP_WEIGHT = 0.20
FULL_GUARD_EDGE_MISMATCH_WEIGHT = 0.35
FULL_GUARD_SAME_FEN_MAX_CONF_GAP = 0.070
FULL_GUARD_SAME_FEN_PENALTY = 0.75
FULL_GUARD_FULL_LEFT_TRIMS = (0, 8, 12, 16)
FULL_GUARD_FULL_TOP_TRIMS = (0, 12, 20)
FULL_GUARD_FULL_MIN_SIDE_RATIO = 0.70
FULL_GUARD_FULL_EDGE_OCC_CONF_GAP = 0.030
FULL_GUARD_FULL_EDGE_OCC_GRID_GAP = 0.060
FULL_GUARD_FULL_MIN_TOP_TRIM = 40
FULL_GUARD_FULL_MIN_MULTI_EDGE_SUM = 180


def _edge_drop_metrics(full_fen: str, candidate_fen: str) -> tuple[int, int, int, int]:
    full_rows = base.parse_fen_board_rows(full_fen)
    cand_rows = base.parse_fen_board_rows(candidate_fen)
    if full_rows is None or cand_rows is None:
        return 0, 0, 0, 0

    edge_drop = 0
    file_drop = 0
    rank_drop = 0
    edge_mismatch = 0
    for row in range(8):
        for col in range(8):
            if row not in {0, 7} and col not in {0, 7}:
                continue
            full_occ = full_rows[row][col] != "1"
            cand_occ = cand_rows[row][col] != "1"
            if full_rows[row][col] != cand_rows[row][col]:
                edge_mismatch += 1
            if full_occ and not cand_occ:
                edge_drop += 1
                if col in {0, 7}:
                    file_drop += 1
                if row in {0, 7}:
                    rank_drop += 1
    return edge_drop, file_drop, rank_drop, edge_mismatch


def compute_dark_edge_trim_box(candidate_img):
    arr = base.np.array(candidate_img)
    gray = base.cv2.cvtColor(arr, base.cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    h, w = gray.shape
    if h < 64 or w < 64:
        return {
            "x0": 0,
            "y0": 0,
            "x1": w,
            "y1": h,
            "left": 0,
            "top": 0,
            "right": 0,
            "bottom": 0,
            "trimmed": False,
        }
    mid = gray[int(h * 0.2): int(h * 0.8), int(w * 0.2): int(w * 0.8)]
    if mid.size == 0:
        return {
            "x0": 0,
            "y0": 0,
            "x1": w,
            "y1": h,
            "left": 0,
            "top": 0,
            "right": 0,
            "bottom": 0,
            "trimmed": False,
        }
    ref = float(base.np.median(mid))
    dark_thr = ref * 0.70
    max_trim_y = int(h * 0.18)
    max_trim_x = int(w * 0.12)
    row_mean = gray.mean(axis=1)
    col_mean = gray.mean(axis=0)
    top = 0
    while top < max_trim_y and row_mean[top] < dark_thr:
        top += 1
    bottom = 0
    while bottom < max_trim_y and row_mean[h - 1 - bottom] < dark_thr:
        bottom += 1
    left = 0
    while left < max_trim_x and col_mean[left] < dark_thr:
        left += 1
    right = 0
    while right < max_trim_x and col_mean[w - 1 - right] < dark_thr:
        right += 1
    if top < int(h * 0.02):
        top = 0
    if bottom < int(h * 0.02):
        bottom = 0
    if left < int(w * 0.015):
        left = 0
    if right < int(w * 0.015):
        right = 0
    x0, y0 = left, top
    x1, y1 = w - right, h - bottom
    if x1 - x0 < int(w * 0.65) or y1 - y0 < int(h * 0.65):
        x0, y0, x1, y1 = 0, 0, w, h
        left = top = right = bottom = 0
    return {
        "x0": int(x0),
        "y0": int(y0),
        "x1": int(x1),
        "y1": int(y1),
        "left": int(left),
        "top": int(top),
        "right": int(right),
        "bottom": int(bottom),
        "trimmed": bool(left or top or right or bottom),
    }


def apply_trim_box(candidate_img, trim_box):
    w, h = candidate_img.size
    if not trim_box.get("trimmed"):
        return candidate_img
    cropped = candidate_img.crop((trim_box["x0"], trim_box["y0"], trim_box["x1"], trim_box["y1"]))
    return cropped.resize((w, h), Image.LANCZOS)


def should_refine_full_trim(trim_box: dict[str, Any]) -> bool:
    if not trim_box.get("trimmed"):
        return False
    top = int(trim_box.get("top", 0))
    left = int(trim_box.get("left", 0))
    right = int(trim_box.get("right", 0))
    bottom = int(trim_box.get("bottom", 0))
    if top >= FULL_GUARD_FULL_MIN_TOP_TRIM:
        return True
    nonzero_edges = sum(1 for value in (left, top, right, bottom) if value > 0)
    edge_sum = left + top + right + bottom
    return nonzero_edges >= 3 and edge_sum >= FULL_GUARD_FULL_MIN_MULTI_EDGE_SUM


def _outer_ring_occupancy(fen_board: str) -> int:
    rows = base.parse_fen_board_rows(fen_board)
    if rows is None:
        return 0
    count = 0
    for row in range(8):
        for col in range(8):
            if row in {0, 7} or col in {0, 7}:
                count += int(rows[row][col] != "1")
    return count


def _tile_map(tile_infos, perspective: str):
    mapped = {}
    for tile in tile_infos:
        row = int(tile["row"])
        col = int(tile["col"])
        if perspective == "black":
            row = 7 - row
            col = 7 - col
        sq = f"{'abcdefgh'[col]}{8 - row}"
        mapped[sq] = tile
    return mapped


def _grid_score_for_full_variant(variant_img) -> float:
    detector = base.BoardDetector(debug=False)
    w, h = variant_img.size
    corners = base.np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=base.np.float32)
    return float(detector._warp_grid_score(variant_img, corners))


def _build_full_refinement_variants(trimmed_img):
    variants = []
    w, h = trimmed_img.size
    seen = set()
    trim_specs = [(0, 0, 0, 0)]
    trim_specs.extend((left, 0, 0, 0) for left in FULL_GUARD_FULL_LEFT_TRIMS if left)
    trim_specs.extend((0, top, 0, 0) for top in FULL_GUARD_FULL_TOP_TRIMS if top)
    trim_specs.extend(
        (left, top, 0, 0)
        for left in FULL_GUARD_FULL_LEFT_TRIMS if left
        for top in FULL_GUARD_FULL_TOP_TRIMS if top
    )

    for left, top, right, bottom in trim_specs:
        x0 = left
        y0 = top
        x1 = w - right
        y1 = h - bottom
        if x1 - x0 < int(w * FULL_GUARD_FULL_MIN_SIDE_RATIO):
            continue
        if y1 - y0 < int(h * FULL_GUARD_FULL_MIN_SIDE_RATIO):
            continue
        key = (x0, y0, x1, y1)
        if key in seen:
            continue
        seen.add(key)
        if key == (0, 0, w, h):
            variant_name = "base"
            variant_img = trimmed_img
        else:
            variant_name = "full_refined"
            variant_img = trimmed_img.crop((x0, y0, x1, y1)).resize((w, h), Image.LANCZOS)
        variants.append(
            {
                "variant": variant_name,
                "image": variant_img,
                "refinement_box": {
                    "x0": int(x0),
                    "y0": int(y0),
                    "x1": int(x1),
                    "y1": int(y1),
                    "left_extra": int(left),
                    "top_extra": int(top),
                    "right_extra": int(right),
                    "bottom_extra": int(bottom),
                },
            }
        )
    return variants


def _decode_variant_row(
    variant_name,
    variant_img,
    model,
    device,
    topk_k: int,
    trim_box: dict[str, Any] | None,
    refinement_box: dict[str, Any] | None,
):
    fen, conf, piece_count, details = base.infer_fen_on_image_clean(
        variant_img,
        model,
        device,
        base.USE_SQUARE_DETECTION,
        return_details=True,
        topk_k=topk_k,
    )
    sat_stats = base.image_saturation_stats(variant_img)
    rescored_applied = False
    if sat_stats["sat_mean"] <= base.LOW_SAT_SPARSE_SAT_MEAN_MAX and piece_count <= base.LOW_SAT_SPARSE_PIECE_MAX:
        rescored = base.rescore_low_saturation_sparse_from_topk(details.get("tile_infos", []), base_fen=fen, base_conf=conf)
        if rescored is not None:
            fen, conf, piece_count = rescored
            rescored_applied = True
    return {
        "variant": variant_name,
        "img": variant_img,
        "fen_before_orientation": fen,
        "confidence": float(conf),
        "piece_count": int(piece_count),
        "plausibility": float(base.board_plausibility_score(fen)),
        "king_health": int(base.king_health(fen)),
        "tile_infos": details.get("tile_infos", []),
        "rescored_low_sat_sparse": bool(rescored_applied),
        "sat_stats": sat_stats,
        "grid_score": float(_grid_score_for_full_variant(variant_img)),
        "outer_ring_occupancy": int(_outer_ring_occupancy(fen)),
        "trim_box": trim_box,
        "refinement_box": refinement_box,
    }


def _select_best_full_variant(rows):
    best = max(
        rows,
        key=lambda row: (
            row["plausibility"],
            row["king_health"],
            row["piece_count"] if row["piece_count"] <= base.LOW_SAT_SPARSE_PIECE_MAX else 0,
            row["grid_score"],
            row["confidence"],
        ),
    )
    eligible = [
        row
        for row in rows
        if row["plausibility"] == best["plausibility"]
        and row["king_health"] == best["king_health"]
        and row["confidence"] >= (best["confidence"] - FULL_GUARD_FULL_EDGE_OCC_CONF_GAP)
        and row["grid_score"] >= (best["grid_score"] - FULL_GUARD_FULL_EDGE_OCC_GRID_GAP)
    ]
    return max(
        eligible,
        key=lambda row: (
            row["outer_ring_occupancy"],
            row["grid_score"],
            row["confidence"],
            row["piece_count"] if row["piece_count"] <= base.LOW_SAT_SPARSE_PIECE_MAX else 0,
        ),
    )


def _decode_candidate_rows(
    candidate,
    model,
    device,
    board_perspective,
    orientation_context,
    topk_k: int,
):
    tag, candidate_img, warp_quality, warp_trusted = candidate
    trim_box = compute_dark_edge_trim_box(candidate_img)
    trimmed_img = apply_trim_box(candidate_img, trim_box)
    variant_specs = [{"variant": "base", "image": trimmed_img, "refinement_box": None}]
    if str(tag).startswith("contour_"):
        inner_board = base.find_inner_board_window(trimmed_img)
        if inner_board is not None:
            variant_specs.append({"variant": "inner_board", "image": inner_board, "refinement_box": None})
    if str(tag) == "full" and should_refine_full_trim(trim_box):
        variant_specs = _build_full_refinement_variants(trimmed_img)

    decoded_rows = [
        _decode_variant_row(
            spec["variant"],
            spec["image"],
            model=model,
            device=device,
            topk_k=topk_k,
            trim_box=trim_box,
            refinement_box=spec["refinement_box"],
        )
        for spec in variant_specs
    ]
    if str(tag) == "full":
        best_variant = _select_best_full_variant(decoded_rows)
    else:
        best_variant = max(
            decoded_rows,
            key=lambda row: (
                row["plausibility"],
                row["king_health"],
                row["piece_count"] if row["piece_count"] <= base.LOW_SAT_SPARSE_PIECE_MAX else 0,
                row["confidence"],
            ),
        )
    detected_perspective, perspective_source = base.resolve_candidate_orientation(
        best_variant["fen_before_orientation"],
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    board_fen = (
        base.rotate_fen_180(best_variant["fen_before_orientation"])
        if detected_perspective == "black"
        else best_variant["fen_before_orientation"]
    )
    return best_variant, {
        "tag": tag,
        "variant": best_variant["variant"],
        "board_fen": board_fen,
        "fen_before_orientation": best_variant["fen_before_orientation"],
        "confidence": best_variant["confidence"],
        "piece_count": best_variant["piece_count"],
        "plausibility": best_variant["plausibility"],
        "king_health": best_variant["king_health"],
        "detected_perspective": detected_perspective,
        "perspective_source": perspective_source,
        "warp_quality": float(warp_quality),
        "warp_trusted": bool(warp_trusted),
        "tile_infos_by_square": _tile_map(best_variant["tile_infos"], detected_perspective),
        "rescored_low_sat_sparse": best_variant["rescored_low_sat_sparse"],
        "sat_stats": best_variant["sat_stats"],
        "candidate_img": best_variant["img"],
        "grid_score": best_variant["grid_score"],
        "outer_ring_occupancy": best_variant["outer_ring_occupancy"],
        "trim_box": best_variant["trim_box"],
        "refinement_box": best_variant["refinement_box"],
    }


def diagnostic_decode_candidate_with_details(
    candidate,
    model,
    device,
    board_perspective,
    orientation_context,
    topk: int,
):
    _best_variant, row = _decode_candidate_rows(
        candidate,
        model=model,
        device=device,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
        topk_k=max(3, int(topk)),
    )
    return row


def decode_candidate(candidate, model, device, board_perspective, orientation_context):
    _best_variant, row = _decode_candidate_rows(
        candidate,
        model=model,
        device=device,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
        topk_k=5,
    )
    return (
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


def select_best_candidate(scored):
    best_raw_conf = max(float(item[3]) for item in scored) if scored else 0.0
    full_item = next((item for item in scored if item[0] == "full"), None)
    full_plausibility = base.board_plausibility_score(full_item[2]) if full_item is not None else -1e9
    full_king_health = base.king_health(full_item[2]) if full_item is not None else 0
    full_conf = float(full_item[3]) if full_item is not None else 0.0
    full_piece_count = int(full_item[4]) if full_item is not None else 0

    enriched = []
    for item in scored:
        fen_board = item[2]
        plausibility = base.board_plausibility_score(fen_board)
        k_health = base.king_health(fen_board)
        piece_count = int(item[4])
        geom_q = float(item[7])
        _, stm_src = base.infer_side_to_move_from_checks(fen_board)
        has_double_check_conflict = stm_src == "default_double_check_conflict"
        conf_adj = float(item[3])
        stats = base.image_saturation_stats(item[1])
        low_sat_sparse = stats["sat_mean"] <= base.LOW_SAT_SPARSE_SAT_MEAN_MAX and piece_count <= base.LOW_SAT_SPARSE_PIECE_MAX
        if "gradient_projection" in str(item[0]):
            conf_adj -= 0.010
        if "panel_split" in str(item[0]):
            conf_adj -= 0.030
        sparse_piece_bonus = piece_count if low_sat_sparse and float(item[3]) >= (best_raw_conf - 0.025) else 0

        full_guard_bonus = 0.0
        if (
            full_item is not None
            and item[0] != "full"
            and full_plausibility >= FULL_GUARD_MIN_PLAUSIBILITY
            and full_king_health >= FULL_GUARD_MIN_KING_HEALTH
        ):
            if (
                fen_board == full_item[2]
                and full_conf >= (float(item[3]) - FULL_GUARD_SAME_FEN_MAX_CONF_GAP)
            ):
                full_guard_bonus = -FULL_GUARD_SAME_FEN_PENALTY
            elif (
                full_piece_count >= (piece_count + FULL_GUARD_MIN_PIECE_MARGIN)
                and full_conf >= (float(item[3]) - FULL_GUARD_MAX_CONF_GAP)
            ):
                edge_drop, file_drop, rank_drop, edge_mismatch = _edge_drop_metrics(full_item[2], fen_board)
                if edge_drop >= FULL_GUARD_MIN_EDGE_DROP or edge_mismatch >= FULL_GUARD_MIN_EDGE_MISMATCH:
                    full_guard_bonus = -(
                        float(edge_drop)
                        + FULL_GUARD_FILE_DROP_WEIGHT * float(file_drop)
                        + FULL_GUARD_RANK_DROP_WEIGHT * float(rank_drop)
                        + FULL_GUARD_EDGE_MISMATCH_WEIGHT * float(edge_mismatch)
                    )
        enriched.append(
            (
                item,
                plausibility,
                k_health,
                sparse_piece_bonus,
                -int(has_double_check_conflict),
                full_guard_bonus,
                conf_adj,
                geom_q,
            )
        )
    return max(enriched, key=lambda pair: (pair[1], pair[2], pair[3], pair[4], pair[5], pair[6], pair[7]))[0]


def predict_chess_position(image_path, model_path=None, board_perspective="auto", side_to_move_override=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_path = model_path or base.MODEL_PATH
    if not resolved_model_path or not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            "Model checkpoint not found. "
            f"Tried: {resolved_model_path}. "
            "Set CHESSBOT_MODEL_PATH or pass --model-path explicitly."
        )
    global _model_cache
    if resolved_model_path in _model_cache:
        model = _model_cache[resolved_model_path]
    else:
        model = StandaloneBeastClassifier(num_classes=13).to(device)
        model.load_state_dict(torch.load(resolved_model_path, map_location=device))
        model.eval()
        _model_cache[resolved_model_path] = model
    img = Image.open(image_path).convert("RGB")
    candidates = build_detector_candidates(img)
    orientation_context = collect_orientation_context(candidates, board_perspective=board_perspective)
    scored = [
        decode_candidate(
            candidate,
            model=model,
            device=device,
            board_perspective=board_perspective,
            orientation_context=orientation_context,
        )
        for candidate in candidates
    ]
    if not scored:
        raise RuntimeError("No valid candidates found")
    best = select_best_candidate(scored)
    if side_to_move_override is not None:
        side_to_move = parse_side_to_move_override(side_to_move_override)
        side_source = "override_cli"
    else:
        side_to_move, side_source = infer_side_to_move_from_checks(best[2])
    return {
        "board_fen": best[2],
        "fen": f"{best[2]} {side_to_move} - - 0 1",
        "confidence": float(best[3]),
        "side_to_move": side_to_move,
        "side_to_move_source": side_source,
        "detected_perspective": best[5],
        "perspective_source": best[6],
        "best_tag": best[0],
    }


def predict_board(image_path, model_path=None, board_perspective="auto"):
    result = predict_chess_position(image_path, model_path, board_perspective)
    return result["board_fen"], result["confidence"]


def predict_position(image_path, model_path=None, board_perspective="auto", side_to_move_override=None):
    return predict_chess_position(image_path, model_path, board_perspective, side_to_move_override)


def main():
    parser = argparse.ArgumentParser(description="Recognize chess position from image (v6 fullguard)")
    parser.add_argument("image", help="Path to chess board image")
    parser.add_argument("--model-path", default=None, help="Override model path")
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Infer perspective automatically, or force White/Black at the bottom",
    )
    parser.add_argument("--no-edge-detection", action="store_true", help="Disable edge detection")
    parser.add_argument("--no-square-detection", action="store_true", help="Disable square grid detection")
    parser.add_argument(
        "--side-to-move",
        default=None,
        help="Override side to move (aliases: wtm|btm|w|b|white|black)",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose debugging")
    args = parser.parse_args()
    global USE_EDGE_DETECTION, USE_SQUARE_DETECTION, DEBUG_MODE
    USE_EDGE_DETECTION = not args.no_edge_detection
    USE_SQUARE_DETECTION = not args.no_square_detection
    DEBUG_MODE = args.debug
    try:
        result = predict_position(
            args.image,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
            side_to_move_override=args.side_to_move,
        )
        print(
            json.dumps(
                {
                    "success": True,
                    "fen": result["fen"],
                    "confidence": round(result["confidence"], 4),
                    "side_to_move": result["side_to_move"],
                    "side_to_move_source": result["side_to_move_source"],
                    "detected_perspective": result["detected_perspective"],
                    "perspective_source": result["perspective_source"],
                }
            )
        )
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}))


if __name__ == "__main__":
    main()
