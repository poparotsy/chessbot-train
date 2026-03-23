#!/usr/bin/env python3
import argparse
import json
import os
import sys

import torch
from PIL import Image

import recognizer_v6_candidate_core as core
from edge_grid_detector import crop_warp_to_detected_grid, detect_board_grid, score_full_frame_board


MODEL_PATH = core.MODEL_PATH
StandaloneBeastClassifier = core.StandaloneBeastClassifier
collect_orientation_context = core.collect_orientation_context
infer_side_to_move_from_checks = core.infer_side_to_move_from_checks
resolve_candidate_orientation = core.resolve_candidate_orientation
rotate_fen_180 = core.rotate_fen_180
board_plausibility_score = core.board_plausibility_score
king_health = core.king_health
image_saturation_stats = core.image_saturation_stats
parse_side_to_move_override = core.parse_side_to_move_override
infer_fen_on_image_clean = core.infer_fen_on_image_clean
LOW_SAT_SPARSE_SAT_MEAN_MAX = core.LOW_SAT_SPARSE_SAT_MEAN_MAX
LOW_SAT_SPARSE_PIECE_MAX = core.LOW_SAT_SPARSE_PIECE_MAX

USE_EDGE_DETECTION = True
USE_SQUARE_DETECTION = True
DEBUG_MODE = False


def trim_dark_edge_bars(img):
    return img


def find_inner_board_window(img):
    return None


def _debug_event(kind, **payload):
    if not DEBUG_MODE:
        return
    row = {"kind": kind}
    row.update(payload)
    print(f"DEBUG_JSON {json.dumps(row, sort_keys=True)}", file=sys.stderr)


def _candidate_tuple(tag, img, warp_quality, warp_trusted, detector_score, detector_support):
    return (tag, img, float(warp_quality), bool(warp_trusted), float(detector_score), float(detector_support))


def _candidate_sort_key(candidate):
    return (
        float(candidate[4]),
        float(candidate[5]),
        float(candidate[2]),
    )


def _material_score_order(candidates, *, epsilon=0.02):
    ordered = []
    for candidate in candidates:
        inserted = False
        score = float(candidate[4])
        for idx, current in enumerate(ordered):
            current_score = float(current[4])
            if score > (current_score + epsilon):
                ordered.insert(idx, candidate)
                inserted = True
                break
        if not inserted:
            ordered.append(candidate)
    return ordered


def _full_candidate_should_lead(img, full_meta, detector_candidates):
    ratio = img.size[0] / float(max(1, img.size[1]))
    best_detector_score = max((float(candidate[4]) for candidate in detector_candidates), default=-1.0)
    return bool(
        0.92 <= ratio <= 1.08
        and float(full_meta.get("coverage", 0.0)) >= 0.93
        and full_meta["evidence"] >= 0.34
        and full_meta["support_ratio"] >= 0.46
        and float(full_meta["score"]) >= best_detector_score
    )


def build_detector_candidates(img):
    full_meta = score_full_frame_board(img)
    full_candidate = _candidate_tuple(
        "full",
        img,
        full_meta["score"],
        full_meta["trusted"],
        full_meta["score"],
        full_meta["support_ratio"],
    )
    if not USE_EDGE_DETECTION:
        return [full_candidate]

    detector_candidates = []
    frame_ratio = img.size[0] / float(max(1, img.size[1]))
    for row in detect_board_grid(img, max_hypotheses=6):
        corners = row["corners"]
        roi = core.perspective_transform(img, corners)
        refined = crop_warp_to_detected_grid(roi)
        if refined is not None:
            roi = refined
        detector_candidates.append(
            _candidate_tuple(
                str(row["tag"]),
                roi,
                row["score"],
                row["trusted"],
                row["score"],
                row["support_ratio"],
            )
        )

    candidates = _material_score_order(detector_candidates)[:4]
    if not candidates:
        return [full_candidate]

    if _full_candidate_should_lead(img, full_meta, candidates):
        candidates = [full_candidate] + candidates
    else:
        candidates.append(full_candidate)

    _debug_event(
        "v6_candidate_pool",
        original_count=len(detector_candidates) + 1,
        capped_count=len(candidates),
        tags=[row[0] for row in candidates],
        scores=[round(float(row[4]), 6) for row in candidates],
    )
    return candidates


def decode_candidate(candidate, model, device, board_perspective, orientation_context):
    tag, candidate_img, warp_quality, warp_trusted, detector_score, detector_support = candidate
    fen, conf, piece_count = core.infer_fen_on_image_clean(
        candidate_img,
        model,
        device,
        USE_SQUARE_DETECTION,
    )
    detected_perspective, perspective_source = core.resolve_candidate_orientation(
        fen,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    final_fen = core.rotate_fen_180(fen) if detected_perspective == "black" else fen
    _debug_event(
        "v6_candidate_decode",
        tag=tag,
        conf=float(conf),
        plausibility=float(core.board_plausibility_score(final_fen)),
        king_health=int(core.king_health(final_fen)),
        piece_count=int(piece_count),
        warp_quality=float(warp_quality),
        detector_score=float(detector_score),
        detector_support=float(detector_support),
        perspective=detected_perspective,
        perspective_source=perspective_source,
    )
    return (
        tag,
        candidate_img,
        final_fen,
        conf,
        piece_count,
        detected_perspective,
        perspective_source,
        warp_quality,
        warp_trusted,
    )


def diagnostic_decode_candidate_with_details(
    candidate,
    *,
    model,
    device,
    board_perspective,
    orientation_context,
    topk=5,
):
    tag, candidate_img, warp_quality, warp_trusted, detector_score, detector_support = candidate
    fen, conf, piece_count, details = core.infer_fen_on_image_clean(
        candidate_img,
        model,
        device,
        USE_SQUARE_DETECTION,
        return_details=True,
        topk_k=max(3, int(topk)),
    )
    detected_perspective, perspective_source = core.resolve_candidate_orientation(
        fen,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    board_fen = core.rotate_fen_180(fen) if detected_perspective == "black" else fen
    tile_infos_by_square = {}
    for tile in details.get("tile_infos", []):
        row = int(tile["row"])
        col = int(tile["col"])
        if detected_perspective == "black":
            row = 7 - row
            col = 7 - col
        sq = f"{'abcdefgh'[col]}{8 - row}"
        tile_infos_by_square[sq] = tile
    return {
        "tag": tag,
        "variant": "base",
        "board_fen": board_fen,
        "fen_before_orientation": fen,
        "confidence": float(conf),
        "piece_count": int(piece_count),
        "plausibility": float(core.board_plausibility_score(board_fen)),
        "king_health": int(core.king_health(board_fen)),
        "detected_perspective": detected_perspective,
        "perspective_source": perspective_source,
        "warp_quality": float(warp_quality),
        "warp_trusted": bool(warp_trusted),
        "tile_infos_by_square": tile_infos_by_square,
        "rescored_low_sat_sparse": False,
        "sat_stats": core.image_saturation_stats(candidate_img),
        "candidate_img": candidate_img,
        "grid_score": float(detector_score),
        "detector_support": float(detector_support),
    }


def select_best_candidate(scored):
    if not scored:
        raise RuntimeError("No scored candidates")
    return scored[0]


def predict_chess_position(image_path, model_path=None, board_perspective="auto", side_to_move_override=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = core.load_model(model_path=model_path, device=device)
    img = Image.open(image_path).convert("RGB")
    candidates = build_detector_candidates(img)
    orientation_context = core.collect_orientation_context(candidates, board_perspective=board_perspective)
    best = decode_candidate(
        candidates[0],
        model=model,
        device=device,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    if side_to_move_override is not None:
        side_to_move = core.parse_side_to_move_override(side_to_move_override)
        side_source = "override_cli"
    else:
        side_to_move, side_source = core.infer_side_to_move_from_checks(best[2])
    _debug_event(
        "v6_selected_candidate",
        tag=best[0],
        confidence=float(best[3]),
        perspective_source=best[6],
        side_to_move_source=side_source,
    )
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
    parser = argparse.ArgumentParser(description="Recognizer candidate: clean detector-first path")
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
    parser.add_argument("--debug", action="store_true", help="Emit DEBUG_JSON telemetry")
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
