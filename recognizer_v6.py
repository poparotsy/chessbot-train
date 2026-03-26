#!/usr/bin/env python3
import argparse
import json
import sys

import cv2
import numpy as np
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

MATERIAL_SCORE_EPSILON = 0.02
FULL_CANDIDATE_RATIO_MIN = 0.92
FULL_CANDIDATE_RATIO_MAX = 1.08
FULL_CANDIDATE_COVERAGE_MIN = 0.93
FULL_CANDIDATE_EVIDENCE_MIN = 0.34
FULL_CANDIDATE_SUPPORT_MIN = 0.46
FULL_CANDIDATE_STRONG_COVERAGE_MIN = 0.97
FULL_CANDIDATE_STRONG_EVIDENCE_MIN = 0.50
FULL_CANDIDATE_STRONG_SUPPORT_MIN = 0.95
VALUE_CASE_ALT_MIN = 0.05
VALUE_CASE_CONF_MIN = 0.55


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


def _material_score_order(candidates, *, epsilon=MATERIAL_SCORE_EPSILON):
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
    strong_full = bool(
        FULL_CANDIDATE_RATIO_MIN <= ratio <= FULL_CANDIDATE_RATIO_MAX
        and float(full_meta.get("coverage", 0.0)) >= FULL_CANDIDATE_STRONG_COVERAGE_MIN
        and float(full_meta.get("evidence", 0.0)) >= FULL_CANDIDATE_STRONG_EVIDENCE_MIN
        and float(full_meta.get("support_ratio", 0.0)) >= FULL_CANDIDATE_STRONG_SUPPORT_MIN
    )
    return bool(
        strong_full
        or (
            FULL_CANDIDATE_RATIO_MIN <= ratio <= FULL_CANDIDATE_RATIO_MAX
            and float(full_meta.get("coverage", 0.0)) >= FULL_CANDIDATE_COVERAGE_MIN
            and float(full_meta.get("evidence", 0.0)) >= FULL_CANDIDATE_EVIDENCE_MIN
            and float(full_meta.get("support_ratio", 0.0)) >= FULL_CANDIDATE_SUPPORT_MIN
            and float(full_meta["score"]) >= best_detector_score
        )
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


def _value_only_image(img):
    arr = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    value = hsv[:, :, 2]
    return Image.fromarray(np.repeat(value[:, :, None], 3, axis=2))


def _topk_prob(tile_info, label):
    for alt_label, alt_prob in tile_info.get("topk", []):
        if alt_label == label:
            return float(alt_prob)
    return 0.0


def _fuse_value_case_labels(rgb_details, value_details):
    labels = ["1"] * 64
    fused = False
    for rgb_tile, value_tile in zip(rgb_details.get("tile_infos", []), value_details.get("tile_infos", [])):
        row = int(rgb_tile["row"])
        col = int(rgb_tile["col"])
        label = str(rgb_tile["label"])
        value_label = str(value_tile["label"])
        if (
            label != "1"
            and value_label != "1"
            and label.lower() == value_label.lower()
            and label != value_label
            and _topk_prob(rgb_tile, value_label) >= VALUE_CASE_ALT_MIN
            and _topk_prob(value_tile, value_label) >= VALUE_CASE_CONF_MIN
        ):
            label = value_label
            fused = True
        labels[(row * 8) + col] = label
    rows = ["".join(labels[r * 8 : (r + 1) * 8]) for r in range(8)]
    return core.compress_fen_board(rows), sum(1 for label in labels if label != "1"), fused


def _decode_board_with_details(candidate_img, model, device, board_perspective):
    return core.infer_fen_on_image_clean(
        candidate_img,
        model,
        device,
        USE_SQUARE_DETECTION,
        board_perspective=board_perspective,
        return_details=True,
    )


def decode_candidate(candidate, model, device, board_perspective, orientation_context):
    tag, candidate_img, warp_quality, warp_trusted, detector_score, detector_support = candidate
    value_case_fused = False
    rgb_details = None

    if str(tag) == "full":
        rgb_fen, conf, piece_count, rgb_details = _decode_board_with_details(
            candidate_img,
            model,
            device,
            "white",
        )
        value_fen, _value_conf, _value_piece_count, value_details = _decode_board_with_details(
            _value_only_image(candidate_img),
            model,
            device,
            "white",
        )
        fused_fen, fused_piece_count, value_case_fused = _fuse_value_case_labels(rgb_details, value_details)
        fen = fused_fen if value_case_fused else rgb_fen
        piece_count = fused_piece_count if value_case_fused else piece_count
    else:
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
        value_case_fused=bool(value_case_fused),
    )
    return {
        "tag": tag,
        "candidate_img": candidate_img,
        "board_fen": final_fen,
        "confidence": float(conf),
        "piece_count": int(piece_count),
        "detected_perspective": detected_perspective,
        "perspective_source": perspective_source,
        "warp_quality": float(warp_quality),
        "warp_trusted": bool(warp_trusted),
        "detector_score": float(detector_score),
        "detector_support": float(detector_support),
        "value_case_fused": bool(value_case_fused),
        "rgb_details": rgb_details,
    }


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
    value_case_fused = False
    if str(tag) == "full":
        value_fen, _value_conf, _value_piece_count, value_details = core.infer_fen_on_image_clean(
            _value_only_image(candidate_img),
            model,
            device,
            USE_SQUARE_DETECTION,
            return_details=True,
            topk_k=max(3, int(topk)),
        )
        fused_fen, fused_piece_count, value_case_fused = _fuse_value_case_labels(details, value_details)
        if value_case_fused:
            fen = fused_fen
            piece_count = fused_piece_count
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
        "value_case_fused": bool(value_case_fused),
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
        side_to_move, side_source = core.infer_side_to_move_from_checks(best["board_fen"])
    _debug_event(
        "v6_selected_candidate",
        tag=best["tag"],
        confidence=float(best["confidence"]),
        perspective_source=best["perspective_source"],
        side_to_move_source=side_source,
    )
    return {
        "board_fen": best["board_fen"],
        "fen": f"{best['board_fen']} {side_to_move} - - 0 1",
        "confidence": float(best["confidence"]),
        "side_to_move": side_to_move,
        "side_to_move_source": side_source,
        "detected_perspective": best["detected_perspective"],
        "perspective_source": best["perspective_source"],
        "best_tag": best["tag"],
        "detector_score": float(best["detector_score"]),
        "detector_support": float(best["detector_support"]),
        "value_case_fused": bool(best["value_case_fused"]),
    }


def predict_board(image_path, model_path=None, board_perspective="auto"):
    result = predict_chess_position(image_path, model_path, board_perspective)
    return result["board_fen"], result["confidence"]


def predict_position(image_path, model_path=None, board_perspective="auto", side_to_move_override=None):
    return predict_chess_position(image_path, model_path, board_perspective, side_to_move_override)


def main():
    parser = argparse.ArgumentParser(description="Recognizer v6: modular detector-first path")
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
