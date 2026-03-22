#!/usr/bin/env python3
import argparse
import json
import os
import sys

import torch
from PIL import Image

from edge_grid_detector import crop_warp_to_detected_grid, detect_board_grid, score_full_frame_board


MODEL_PATH = base.MODEL_PATH
StandaloneBeastClassifier = base.StandaloneBeastClassifier
collect_orientation_context = base.collect_orientation_context
infer_side_to_move_from_checks = base.infer_side_to_move_from_checks
resolve_candidate_orientation = base.resolve_candidate_orientation
rotate_fen_180 = base.rotate_fen_180
board_plausibility_score = base.board_plausibility_score
king_health = base.king_health
LOW_SAT_SPARSE_SAT_MEAN_MAX = base.LOW_SAT_SPARSE_SAT_MEAN_MAX
LOW_SAT_SPARSE_PIECE_MAX = base.LOW_SAT_SPARSE_PIECE_MAX
USE_EDGE_DETECTION = True
USE_SQUARE_DETECTION = True
DEBUG_MODE = False


def _debug_event(kind, **payload):
    if not DEBUG_MODE:
        return
    row = {"kind": kind}
    row.update(payload)
    print(f"DEBUG_JSON {json.dumps(row, sort_keys=True)}", file=sys.stderr)


def _candidate_sort_key(candidate):
    return (
        float(candidate[4]),
        float(candidate[5]),
        float(candidate[2]),
    )


def _full_candidate_should_compete(img, full_meta):
    ratio = img.size[0] / float(max(1, img.size[1]))
    if not (0.92 <= ratio <= 1.08):
        return False
    return (
        float(full_meta["evidence"]) >= 0.35
        and float(full_meta["support_ratio"]) >= 0.48
    )


def build_detector_candidates(img):
    full_meta = score_full_frame_board(img)
    full_candidate = (
        "full",
        img,
        1.0,
        bool(full_meta["trusted"]),
        float(full_meta["score"]),
        float(full_meta["support_ratio"]),
    )
    if not USE_EDGE_DETECTION:
        return [full_candidate]

    detector_candidates = []
    for row in detect_board_grid(img, max_hypotheses=6):
        corners = row["corners"]
        x0 = float(corners[:, 0].min())
        x1 = float(corners[:, 0].max())
        y0 = float(corners[:, 1].min())
        y1 = float(corners[:, 1].max())
        covers_full_frame = (
            x0 <= 2.0
            and y0 <= 2.0
            and x1 >= img.size[0] - 3.0
            and y1 >= img.size[1] - 3.0
        )
        frame_ratio = img.size[0] / float(max(1, img.size[1]))
        if covers_full_frame and 0.92 <= frame_ratio <= 1.08:
            roi = img
        else:
            roi = base.perspective_transform(img, corners)
            refined = crop_warp_to_detected_grid(roi)
            if refined is not None:
                roi = refined
        detector_candidates.append(
            (
                str(row["tag"]),
                roi,
                float(max(row["geometry"], row["evidence"])),
                bool(row["trusted"]),
                float(row["score"]),
                float(row["support_ratio"]),
            )
        )
    detector_candidates.sort(key=_candidate_sort_key, reverse=True)
    candidates = detector_candidates[:4]
    if not candidates:
        return [full_candidate]

    full_competes = _full_candidate_should_compete(img, full_meta)
    if full_competes:
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
    fen, conf, piece_count = base.infer_fen_on_image_clean(
        candidate_img,
        model,
        device,
        True,
    )
    detected_perspective, perspective_source = base.resolve_candidate_orientation(
        fen,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    final_fen = base.rotate_fen_180(fen) if detected_perspective == "black" else fen
    _debug_event(
        "v6_candidate_decode",
        tag=tag,
        conf=float(conf),
        piece_count=int(piece_count),
        plausibility=float(base.board_plausibility_score(final_fen)),
        king_health=int(base.king_health(final_fen)),
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
        detector_score,
        detector_support,
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
    fen, conf, piece_count, details = base.infer_fen_on_image_clean(
        candidate_img,
        model,
        device,
        USE_SQUARE_DETECTION,
        return_details=True,
        topk_k=max(3, int(topk)),
    )
    detected_perspective, perspective_source = base.resolve_candidate_orientation(
        fen,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    board_fen = base.rotate_fen_180(fen) if detected_perspective == "black" else fen
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
        "plausibility": float(base.board_plausibility_score(board_fen)),
        "king_health": int(base.king_health(board_fen)),
        "detected_perspective": detected_perspective,
        "perspective_source": perspective_source,
        "warp_quality": float(warp_quality),
        "warp_trusted": bool(warp_trusted),
        "tile_infos_by_square": tile_infos_by_square,
        "rescored_low_sat_sparse": False,
        "sat_stats": base.image_saturation_stats(candidate_img),
        "candidate_img": candidate_img,
        "grid_score": float(detector_score),
        "detector_support": float(detector_support),
    }


def select_best_candidate(scored):
    return max(
        scored,
        key=lambda item: (
            float(item[9]) if len(item) > 9 else 0.0,
            float(item[10]) if len(item) > 10 else 0.0,
            float(item[7]) if len(item) > 7 else 0.0,
            float(item[3]),
        ),
    )


def predict_chess_position(image_path, model_path=None, board_perspective="auto", side_to_move_override=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_path = model_path or MODEL_PATH
    if not resolved_model_path or not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            "Model checkpoint not found. "
            f"Tried: {resolved_model_path}. "
            "Set CHESSBOT_MODEL_PATH or pass --model-path explicitly."
        )
    if resolved_model_path in base._model_cache:
        model = base._model_cache[resolved_model_path]
    else:
        model = base.StandaloneBeastClassifier(num_classes=13).to(device)
        model.load_state_dict(torch.load(resolved_model_path, map_location=device))
        model.eval()
        base._model_cache[resolved_model_path] = model

    img = Image.open(image_path).convert("RGB")
    candidates = build_detector_candidates(img)
    orientation_context = base.collect_orientation_context(candidates, board_perspective=board_perspective)
    best = decode_candidate(
        candidates[0],
        model=model,
        device=device,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    if side_to_move_override is not None:
        side_to_move = base.parse_side_to_move_override(side_to_move_override)
        side_source = "override_cli"
    else:
        side_to_move, side_source = base.infer_side_to_move_from_checks(best[2])
    _debug_event(
        "v6_selected_candidate",
        tag=best[0],
        confidence=float(best[3]),
        piece_count=int(best[4]),
        warp_quality=float(best[7]),
        perspective=best[5],
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


def predict_position(image_path, model_path=None, board_perspective="auto"):
    return predict_chess_position(
        image_path,
        model_path=model_path,
        board_perspective=board_perspective,
        side_to_move_override=None,
    )


def predict_board(image_path, model_path=None, board_perspective="auto"):
    result = predict_position(
        image_path,
        model_path=model_path,
        board_perspective=board_perspective,
    )
    return result["board_fen"], float(result["confidence"])


def main():
    parser = argparse.ArgumentParser(description="Detector-first recognizer candidate")
    parser.add_argument("image", help="Path to chess board image")
    parser.add_argument("--model-path", default=None, help="Override model path")
    parser.add_argument(
        "--board-perspective",
        choices=["auto", "white", "black"],
        default="auto",
        help="Infer perspective automatically, or force White/Black at the bottom",
    )
    parser.add_argument("--no-edge-detection", action="store_true", help="Disable edge detection")
    parser.add_argument(
        "--side-to-move",
        default=None,
        help="Override side to move (aliases: wtm|btm|w|b|white|black)",
    )
    parser.add_argument("--debug", action="store_true", help="Verbose debugging")
    args = parser.parse_args()
    global USE_EDGE_DETECTION, DEBUG_MODE
    USE_EDGE_DETECTION = not args.no_edge_detection
    DEBUG_MODE = args.debug
    try:
        result = predict_chess_position(
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
