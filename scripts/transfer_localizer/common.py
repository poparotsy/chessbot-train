#!/usr/bin/env python3
"""Shared helpers for the Kaggle-first localizer benchmark."""

from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent.parent


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def order_corners(corners) -> list[list[float]]:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    tl = pts[int(np.argmin(sums))]
    br = pts[int(np.argmax(sums))]
    tr = pts[int(np.argmin(diffs))]
    bl = pts[int(np.argmax(diffs))]
    return [[float(v) for v in row] for row in np.vstack([tl, tr, br, bl])]


def corners_to_bbox(corners) -> list[float]:
    pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    x0 = float(np.min(pts[:, 0]))
    y0 = float(np.min(pts[:, 1]))
    x1 = float(np.max(pts[:, 0]))
    y1 = float(np.max(pts[:, 1]))
    return [x0, y0, x1, y1]


def bbox_area(box: list[float]) -> float:
    x0, y0, x1, y1 = box
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = bbox_area([ix0, iy0, ix1, iy1])
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def square_crop_from_bbox(box: list[float], width: int, height: int, pad_ratio: float = 0.12) -> tuple[list[int], list[list[float]]]:
    x0, y0, x1, y1 = [float(v) for v in box]
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    side = max(bw, bh) * (1.0 + (2.0 * float(pad_ratio)))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    crop_x0 = int(round(cx - (side / 2.0)))
    crop_y0 = int(round(cy - (side / 2.0)))
    crop_x1 = int(round(cx + (side / 2.0)))
    crop_y1 = int(round(cy + (side / 2.0)))
    if crop_x0 < 0:
        crop_x1 -= crop_x0
        crop_x0 = 0
    if crop_y0 < 0:
        crop_y1 -= crop_y0
        crop_y0 = 0
    if crop_x1 > width:
        delta = crop_x1 - width
        crop_x0 -= delta
        crop_x1 = width
    if crop_y1 > height:
        delta = crop_y1 - height
        crop_y0 -= delta
        crop_y1 = height
    crop_x0 = max(0, crop_x0)
    crop_y0 = max(0, crop_y0)
    crop_x1 = min(width, crop_x1)
    crop_y1 = min(height, crop_y1)
    crop_box = [int(crop_x0), int(crop_y0), int(crop_x1), int(crop_y1)]
    crop_corners = [
        [float(crop_box[0]), float(crop_box[1])],
        [float(crop_box[2]), float(crop_box[1])],
        [float(crop_box[2]), float(crop_box[3])],
        [float(crop_box[0]), float(crop_box[3])],
    ]
    return crop_box, crop_corners


def score_candidate_box(box: list[float], width: int, height: int) -> tuple[bool, dict]:
    x0, y0, x1, y1 = [float(v) for v in box]
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    area_ratio = (bw * bh) / float(max(1, width * height))
    aspect = bw / max(1.0, bh)
    keep = not (area_ratio < 0.10 or area_ratio > 0.98 or aspect < 0.45 or aspect > 2.2)
    return keep, {
        "area_ratio": float(area_ratio),
        "aspect_ratio": float(aspect),
    }


def choose_best_candidate(candidates: list[dict], width: int, height: int) -> dict | None:
    if not candidates:
        return None
    accepted = []
    for row in candidates:
        keep, metrics = score_candidate_box(row["box"], width, height)
        enriched = dict(row)
        enriched.update(metrics)
        enriched["accepted"] = bool(keep)
        if keep:
            accepted.append(enriched)
    pool = accepted if accepted else [dict(row, accepted=False, **score_candidate_box(row["box"], width, height)[1]) for row in candidates]
    return sorted(pool, key=lambda item: (float(item["score"]), bbox_area(item["box"])), reverse=True)[0]


def overlay_boxes(image_path: Path, out_path: Path, truth_box=None, pred_box=None, crop_box=None, title: str | None = None) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    if truth_box is not None:
        draw.rectangle(truth_box, outline=(0, 255, 0), width=4)
    if pred_box is not None:
        draw.rectangle(pred_box, outline=(255, 0, 0), width=4)
    if crop_box is not None:
        draw.rectangle(crop_box, outline=(255, 200, 0), width=3)
    if title:
        draw.text((12, 12), title, fill=(255, 255, 255))
    ensure_dir(out_path.parent)
    image.save(out_path)


def mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median_or_zero(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

