#!/usr/bin/env python3
"""Shared localizer backends for benchmark and runtime integration."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from PIL import Image
import torch

from common import choose_best_candidate, corners_to_bbox, order_corners, score_candidate_box

MODEL_RUN_NAMES = {
    "google/owlv2-base-patch16-ensemble": "owlv2-base-ens",
    "google/owlv2-base-patch16": "owlv2-base",
    "google/owlv2-large-patch14-ensemble": "owlv2-large-ens",
    "google/owlvit-base-patch32": "owlvit-base32",
    "IDEA-Research/grounding-dino-tiny": "grounding-dino-tiny",
    "IDEA-Research/grounding-dino-base": "grounding-dino-base",
    "iSEE-Laboratory/llmdet_base": "llmdet-base",
    "iSEE-Laboratory/llmdet_large": "llmdet-large",
}

DEFAULT_ZERO_SHOT_MODELS = [
    "IDEA-Research/grounding-dino-tiny",
    "IDEA-Research/grounding-dino-base",
    "google/owlv2-base-patch16-ensemble",
    "google/owlv2-base-patch16",
    "google/owlvit-base-patch32",
]


def queries_for_model(model_id: str) -> list[str]:
    if "owlv2" in model_id.lower():
        return ["a photo of a chessboard", "a photo of a chess board"]
    return ["a chessboard", "a chess board"]


def run_name_for_model(model_id: str) -> str:
    if model_id in MODEL_RUN_NAMES:
        return MODEL_RUN_NAMES[model_id]
    return model_id.split("/")[-1].replace("_", "-")


@lru_cache(maxsize=8)
def _load_zero_shot_detector(model_id: str, device: str):
    from huggingface_hub.utils import disable_progress_bars
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Owlv2ForObjectDetection, Owlv2Processor
    from transformers.utils import logging as transformers_logging

    disable_progress_bars()
    transformers_logging.set_verbosity_error()

    model_id_l = model_id.lower()
    if "owlv2" in model_id_l:
        processor = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id)
    elif "owlvit" in model_id_l:
        processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    else:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    return processor, model.to(device)


def detect_zero_shot_boxes(
    image: Image.Image,
    *,
    model_id: str,
    device: str,
    threshold: float,
    text_threshold: float,
) -> list[dict[str, Any]]:
    processor, model = _load_zero_shot_detector(model_id, device)
    query_labels = queries_for_model(model_id)
    text_labels = [query_labels]
    model_id_l = model_id.lower()
    if "owlv2" in model_id_l:
        inputs = processor(text=text_labels, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=torch.tensor([(image.height, image.width)], device=device),
            threshold=threshold,
            text_labels=text_labels,
        )[0]
        return [
            {
                "box": [float(x) for x in box.tolist()],
                "score": float(score.item()),
                "label": str(label),
            }
            for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"])
        ]

    if "owlvit" in model_id_l:
        inputs = processor(text=text_labels, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=torch.tensor([(image.height, image.width)], device=device),
        )[0]
        rows: list[dict[str, Any]] = []
        for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
            idx = int(label_idx.item()) if hasattr(label_idx, "item") else int(label_idx)
            label = query_labels[idx] if 0 <= idx < len(query_labels) else str(idx)
            rows.append(
                {
                    "box": [float(x) for x in box.tolist()],
                    "score": float(score.item()),
                    "label": str(label),
                }
            )
        return rows

    inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    label_values = results.get("text_labels", results.get("labels", []))
    rows = []
    for box, score, label in zip(results["boxes"], results["scores"], label_values):
        if isinstance(label, list):
            label = ", ".join(label)
        rows.append(
            {
                "box": [float(x) for x in box.tolist()],
                "score": float(score.item()),
                "label": str(label),
            }
        )
    return rows


def normalize_bbox_candidate(
    box: list[float],
    *,
    score: float,
    source: str,
    label: str | None = None,
    image_size: tuple[int, int] | None = None,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    width, height = image_size or (1, 1)
    accepted, metrics = score_candidate_box(box, width=width, height=height)
    return {
        "kind": "bbox",
        "corners_px": None,
        "bbox_xyxy": [float(v) for v in box],
        "score": float(score),
        "source": str(source),
        "label": None if label is None else str(label),
        "accepted": bool(accepted),
        "debug": {
            **metrics,
            **(debug or {}),
        },
    }


def normalize_corner_candidate(
    corners: Any,
    *,
    score: float,
    source: str,
    label: str | None = None,
    trusted: bool | None = None,
    support_ratio: float | None = None,
    debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ordered = order_corners(corners)
    bbox = corners_to_bbox(ordered)
    return {
        "kind": "corners",
        "corners_px": ordered,
        "bbox_xyxy": bbox,
        "score": float(score),
        "source": str(source),
        "label": None if label is None else str(label),
        "accepted": True if trusted is None else bool(trusted),
        "debug": {
            "support_ratio": None if support_ratio is None else float(support_ratio),
            **(debug or {}),
        },
    }


def build_edge_localizer_candidates(image: Image.Image, recognizer_module, *, max_hypotheses: int = 4) -> list[dict[str, Any]]:
    rows = recognizer_module.detect_board_grid(image, max_hypotheses=max_hypotheses)
    candidates = []
    for row in rows:
        candidates.append(
            normalize_corner_candidate(
                row["corners"],
                score=float(row.get("score", 0.0)),
                source=f"edge:{row.get('tag', 'grid')}",
                label=str(row.get("tag", "grid")),
                trusted=bool(row.get("trusted", False)),
                support_ratio=float(row.get("support_ratio", 0.0)),
                debug={
                    "tag": row.get("tag"),
                    "geometry": float(row.get("geometry", 0.0)),
                    "evidence": float(row.get("evidence", 0.0)),
                },
            )
        )
    return candidates


def build_zero_shot_localizer_candidates(
    image: Image.Image,
    *,
    model_id: str,
    device: str,
    threshold: float = 0.15,
    text_threshold: float = 0.15,
) -> list[dict[str, Any]]:
    boxes = detect_zero_shot_boxes(
        image,
        model_id=model_id,
        device=device,
        threshold=threshold,
        text_threshold=text_threshold,
    )
    normalized = [
        normalize_bbox_candidate(
            row["box"],
            score=float(row.get("score", 0.0)),
            source=f"zero-shot:{run_name_for_model(model_id)}",
            label=str(row.get("label", "chessboard")),
            image_size=(image.width, image.height),
            debug={"model_id": model_id},
        )
        for row in boxes
    ]
    accepted = [row for row in normalized if row["accepted"]]
    return accepted or normalized


def choose_localizer_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    bbox_candidates = []
    for row in candidates:
        bbox = row.get("bbox_xyxy")
        if bbox is None:
            continue
        bbox_candidates.append({"box": bbox, "score": float(row.get("score", 0.0)), "row": row})
    best = choose_best_candidate(bbox_candidates, width=1, height=1) if bbox_candidates else None
    if best is not None and "row" in best:
        return best["row"]
    return sorted(candidates, key=lambda row: (bool(row.get("accepted")), float(row.get("score", 0.0))), reverse=True)[0]
