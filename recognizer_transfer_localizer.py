#!/usr/bin/env python3
"""Candidate recognizer using pluggable transfer localizers plus the v6 decoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
TRANSFER_DIR = SCRIPT_DIR / "scripts" / "transfer_localizer"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(TRANSFER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFER_DIR))

import recognizer_v6 as rec  # noqa: E402
from common import square_crop_from_bbox  # noqa: E402
from localizers import (  # noqa: E402
    build_edge_localizer_candidates,
    build_zero_shot_localizer_candidates,
    choose_localizer_candidate,
)


LOCALIZER_LOW_CONFIDENCE = 0.72


def _roi_from_candidate(image: Image.Image, candidate: dict) -> Image.Image:
    if candidate["kind"] == "corners" and candidate.get("corners_px") is not None:
        roi = rec.perspective_transform(image, np.asarray(candidate["corners_px"], dtype=np.float32))
        refined = rec.crop_warp_to_detected_grid(roi)
        return refined if refined is not None else roi
    bbox = candidate.get("bbox_xyxy")
    if bbox is None:
        raise ValueError("candidate missing bbox_xyxy")
    crop_box, _ = square_crop_from_bbox(bbox, image.width, image.height, pad_ratio=0.12)
    return image.crop(tuple(crop_box))


def _decode_roi(
    image: Image.Image,
    *,
    candidate: dict,
    model,
    device,
    board_perspective: str,
) -> dict:
    roi = _roi_from_candidate(image, candidate)
    tuple_candidate = (
        str(candidate["source"]),
        roi,
        float(candidate.get("score", 0.0)),
        bool(candidate.get("accepted", False)),
        float(candidate.get("score", 0.0)),
        float(candidate.get("debug", {}).get("support_ratio", 0.0) or 0.0),
    )
    orientation_context = rec.collect_orientation_context([tuple_candidate], board_perspective=board_perspective)
    decoded = rec.decode_candidate(
        tuple_candidate,
        model=model,
        device=device,
        board_perspective=board_perspective,
        orientation_context=orientation_context,
    )
    board_fen = decoded[2]
    side_to_move, side_source = rec.infer_side_to_move_from_checks(board_fen)
    en_passant = rec.infer_unique_en_passant_square(board_fen, side_to_move)
    structural_valid = rec.board_is_structurally_valid(board_fen)
    return {
        "board_fen": board_fen,
        "fen": f"{board_fen} {side_to_move} - {en_passant} 0 1",
        "confidence": float(decoded[3]),
        "side_to_move": side_to_move,
        "side_to_move_source": side_source,
        "detected_perspective": decoded[5],
        "perspective_source": decoded[6],
        "best_tag": decoded[0],
        "detector_score": float(decoded[9]),
        "detector_support": float(decoded[10]),
        "value_case_fused": False,
        "structural_valid": bool(structural_valid),
        "localizer_candidate": candidate,
    }


def _candidate_key(result: dict) -> tuple:
    candidate = result["localizer_candidate"]
    return (
        1 if result["structural_valid"] else 0,
        1 if bool(candidate.get("accepted", False)) else 0,
        float(result["confidence"]),
        float(candidate.get("score", 0.0)),
    )


def _need_fallback(result: dict) -> bool:
    return (not result["structural_valid"]) or float(result["confidence"]) < LOCALIZER_LOW_CONFIDENCE


def _primary_localizer_candidates(
    image: Image.Image,
    *,
    localizer_source: str,
    localizer_model_id: str | None,
    device: str,
) -> list[dict]:
    if localizer_source == "edge":
        return build_edge_localizer_candidates(image, rec)
    if localizer_source == "zero-shot":
        if not localizer_model_id:
            raise ValueError("--localizer-model-id is required for zero-shot localizer")
        return build_zero_shot_localizer_candidates(image, model_id=localizer_model_id, device=device)
    if localizer_source == "hybrid":
        if not localizer_model_id:
            raise ValueError("--localizer-model-id is required for hybrid localizer")
        primary = build_zero_shot_localizer_candidates(image, model_id=localizer_model_id, device=device)
        return primary + build_edge_localizer_candidates(image, rec)
    raise ValueError(f"unsupported localizer_source: {localizer_source}")


def predict_position(
    image_path: str,
    *,
    model_path: str | None = None,
    board_perspective: str = "auto",
    localizer_source: str | None = None,
    localizer_model_id: str | None = None,
    localizer_device: str | None = None,
    edge_fallback: bool = True,
    include_debug: bool = False,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    localizer_source = localizer_source or os.environ.get("CHESSBOT_LOCALIZER_SOURCE", "hybrid")
    localizer_model_id = localizer_model_id or os.environ.get("CHESSBOT_LOCALIZER_MODEL_ID", "IDEA-Research/grounding-dino-tiny")
    localizer_device = localizer_device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = rec.load_model(model_path=model_path, device=device)

    decoded_results = []
    primary_candidates = _primary_localizer_candidates(
        image,
        localizer_source=localizer_source,
        localizer_model_id=localizer_model_id,
        device=localizer_device,
    )
    selected_primary = choose_localizer_candidate(primary_candidates)
    if selected_primary is None:
        raise RuntimeError("localizer returned no candidates")
    decoded_results.append(
        _decode_roi(
            image,
            candidate=selected_primary,
            model=model,
            device=device,
            board_perspective=board_perspective,
        )
    )

    if edge_fallback and localizer_source != "edge" and _need_fallback(decoded_results[0]):
        fallback_candidates = build_edge_localizer_candidates(image, rec)
        selected_fallback = choose_localizer_candidate(fallback_candidates)
        if selected_fallback is not None:
            decoded_results.append(
                _decode_roi(
                    image,
                    candidate=selected_fallback,
                    model=model,
                    device=device,
                    board_perspective=board_perspective,
                )
            )

    best = max(decoded_results, key=_candidate_key)
    best["localizer_source"] = best["localizer_candidate"]["source"]
    best["localizer_score"] = float(best["localizer_candidate"].get("score", 0.0))
    if include_debug:
        best["debug"] = {"evaluated_candidates": decoded_results}
    return best


def predict_board(image_path: str, model_path: str | None = None, board_perspective: str = "auto"):
    result = predict_position(image_path, model_path=model_path, board_perspective=board_perspective)
    return result["board_fen"], result["confidence"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer-localizer recognizer candidate")
    parser.add_argument("image")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--board-perspective", choices=["auto", "white", "black"], default="auto")
    parser.add_argument("--localizer-source", choices=["edge", "zero-shot", "hybrid"], default=None)
    parser.add_argument("--localizer-model-id", default=None)
    parser.add_argument("--localizer-device", default=None)
    parser.add_argument("--no-edge-fallback", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    try:
        result = predict_position(
            args.image,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
            localizer_source=args.localizer_source,
            localizer_model_id=args.localizer_model_id,
            localizer_device=args.localizer_device,
            edge_fallback=not args.no_edge_fallback,
            include_debug=args.debug,
        )
        print(json.dumps({"success": True, **result}, default=str))
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}))


if __name__ == "__main__":
    main()
