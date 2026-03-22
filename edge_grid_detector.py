#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


def _to_rgb_array(img: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    arr = np.asarray(img)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    return arr.astype(np.uint8)


def _to_gray(img: Image.Image | np.ndarray) -> np.ndarray:
    return cv2.cvtColor(_to_rgb_array(img), cv2.COLOR_RGB2GRAY)


def _normalize_line(rho: float, theta: float) -> tuple[float, float]:
    rho = float(rho)
    theta = float(theta)
    if rho < 0.0:
        rho = -rho
        theta -= math.pi
    return rho, theta % math.pi


def _order_corners(corners: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    pts = corners.astype(np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _quad_area(corners: np.ndarray) -> float:
    pts = corners.astype(np.float32)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _quad_metrics(corners: np.ndarray, width: int, height: int) -> dict[str, float]:
    pts = _order_corners(corners)
    top = np.linalg.norm(pts[1] - pts[0])
    right = np.linalg.norm(pts[2] - pts[1])
    bottom = np.linalg.norm(pts[3] - pts[2])
    left = np.linalg.norm(pts[0] - pts[3])

    def safe_ratio(a: float, b: float) -> float:
        if a <= 1e-6 or b <= 1e-6:
            return 0.0
        return float(min(a / b, b / a))

    def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        v1 = a - b
        v2 = c - b
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 <= 1e-6 or n2 <= 1e-6:
            return 0.0
        cos_t = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_t)))

    angles = [
        angle_deg(pts[3], pts[0], pts[1]),
        angle_deg(pts[0], pts[1], pts[2]),
        angle_deg(pts[1], pts[2], pts[3]),
        angle_deg(pts[2], pts[3], pts[0]),
    ]
    return {
        "area_ratio": _quad_area(pts) / float(max(1, width * height)),
        "opposite_similarity": safe_ratio(top, bottom) * safe_ratio(left, right),
        "aspect_similarity": safe_ratio(top, left) * safe_ratio(right, bottom),
        "min_angle": float(min(angles)),
        "max_angle": float(max(angles)),
    }


def _warp_geometry_quality(metrics: dict[str, float]) -> float:
    area = max(0.0, min(1.0, metrics["area_ratio"] / 0.20))
    opp = max(0.0, min(1.0, metrics["opposite_similarity"] / 0.72))
    asp = max(0.0, min(1.0, metrics["aspect_similarity"] / 0.68))
    angle_span = max(0.0, min(1.0, (metrics["max_angle"] - metrics["min_angle"]) / 180.0))
    angle = 1.0 - angle_span
    return float(0.35 * area + 0.30 * opp + 0.25 * asp + 0.10 * angle)


def _edge_map(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharp = cv2.Canny(blur, 40, 120, apertureSize=3)
    soft = cv2.Canny(blur, 18, 72, apertureSize=3)
    adap = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        9,
    )
    edges = cv2.bitwise_or(sharp, soft)
    edges = cv2.bitwise_or(edges, adap)
    return cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)


def _extract_hough_lines(gray: np.ndarray) -> list[tuple[float, float]]:
    h, w = gray.shape[:2]
    min_dim = min(h, w)
    raw = cv2.HoughLines(_edge_map(gray), 1, np.pi / 360, threshold=max(50, int(min_dim * 0.16)))
    if raw is None:
        return []
    return [_normalize_line(float(line[0][0]), float(line[0][1])) for line in raw]


def _contour_maps(gray: np.ndarray) -> list[np.ndarray]:
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    adap = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    canny = cv2.Canny(blur, 35, 120, apertureSize=3)
    return [
        cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1),
        cv2.dilate(grad, np.ones((3, 3), np.uint8), iterations=1),
        cv2.morphologyEx(adap, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1),
        cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1),
    ]


def _cluster_positions(items: Iterable[tuple[float, float]], threshold: float) -> list[dict[str, float]]:
    entries = sorted((float(pos), float(weight)) for pos, weight in items)
    if not entries:
        return []
    clusters: list[list[tuple[float, float]]] = [[entries[0]]]
    for pos, weight in entries[1:]:
        center = float(np.mean([p for p, _ in clusters[-1]]))
        if abs(pos - center) <= threshold:
            clusters[-1].append((pos, weight))
        else:
            clusters.append([(pos, weight)])
    merged = []
    for cluster in clusters:
        positions = np.array([pos for pos, _ in cluster], dtype=np.float32)
        weights = np.array([weight for _, weight in cluster], dtype=np.float32)
        merged.append(
            {
                "pos": float(np.average(positions, weights=np.maximum(weights, 1e-3))),
                "weight": float(np.sum(weights)),
            }
        )
    return merged


def _fit_nine_line_lattice(
    clustered: list[dict[str, float]],
    span: float,
    *,
    min_lines: int = 6,
) -> dict[str, object] | None:
    if len(clustered) < min_lines:
        return None
    positions = np.array([item["pos"] for item in clustered], dtype=np.float32)
    weights = np.array([item["weight"] for item in clustered], dtype=np.float32)
    total_weight = float(np.sum(weights)) + 1e-6
    best = None

    diffs = np.diff(np.sort(positions))
    step_candidates: list[float] = [float(span / 8.0), float(span / 7.5), float(span / 9.0)]
    if diffs.size:
        dense = diffs[(diffs > 1.0) & np.isfinite(diffs)]
        for diff in dense:
            for divisor in (1, 2, 3, 4, 5):
                step_candidates.append(float(diff / divisor))
    min_spacing = max(6.0, span / 20.0)
    max_spacing = max(min_spacing + 1.0, span / 3.0)
    step_candidates = sorted(
        {round(step, 2) for step in step_candidates if min_spacing <= step <= max_spacing}
    )
    if not step_candidates:
        return None

    for spacing in step_candidates:
        tolerance = max(4.0, spacing * 0.22)
        for anchor_pos in positions:
            for anchor_idx in range(9):
                origin = float(anchor_pos - anchor_idx * spacing)
                lattice = origin + spacing * np.arange(9, dtype=np.float32)
                nearest = np.min(np.abs(positions[:, None] - lattice[None, :]), axis=1)
                matched = nearest <= tolerance
                support_count = int(np.sum(matched))
                if support_count < min_lines:
                    continue
                support_ratio = float(np.sum(weights[matched]) / total_weight)
                residual = float(
                    np.sum(weights * np.minimum(nearest, tolerance))
                    / (total_weight * max(spacing, 1e-6))
                )
                visible = float(
                    np.mean(np.logical_and(lattice >= -0.1 * span, lattice <= 1.1 * span).astype(np.float32))
                )
                if visible < 0.55:
                    continue
                score = support_ratio * 5.5 + visible * 1.0 - residual * 2.0
                candidate = {
                    "positions": lattice.tolist(),
                    "spacing": float(spacing),
                    "support_ratio": float(support_ratio),
                    "regularity": 1.0,
                    "coverage": float(visible),
                    "residual": float(residual),
                    "score": float(score),
                }
                if best is None or float(candidate["score"]) > float(best["score"]):
                    best = candidate
    return best


def _warp_grid_evidence_score(img: Image.Image | np.ndarray, corners: np.ndarray, size: int = 256) -> float:
    arr = _to_rgb_array(img)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    mat = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    warped = cv2.warpPerspective(arr, mat, (size, size))
    gray = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3))
    x_proj = gx.mean(axis=0)
    y_proj = gy.mean(axis=1)
    tile = max(6, size // 8)

    def acf_score(signal: np.ndarray) -> float:
        centered = signal - float(np.mean(signal))
        denom = float(np.dot(centered, centered))
        if denom <= 1e-6:
            return 0.0
        acf = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
        peaks = []
        for k in range(1, 5):
            center = int(round(k * tile))
            if center + 3 >= len(acf):
                break
            lo = max(1, center - 3)
            hi = min(len(acf), center + 4)
            peaks.append(float(np.max(acf[lo:hi])) / denom)
        return float(np.mean(peaks)) if peaks else 0.0

    return float(max(0.0, min(1.0, 0.5 * (acf_score(x_proj) + acf_score(y_proj)))))


def detect_aligned_square_grid(
    img: Image.Image | np.ndarray,
    *,
    min_lines: int = 6,
) -> dict[str, object] | None:
    gray = _to_gray(img)
    h, w = gray.shape[:2]
    min_dim = min(h, w)
    if min_dim < 120:
        return None
    lines = _extract_hough_lines(gray)
    if len(lines) < min_lines * 2:
        return None

    x_family = []
    y_family = []
    for rho, theta in lines:
        theta_deg = theta * 180.0 / math.pi
        if min(theta_deg, abs(theta_deg - 180.0)) <= 18.0:
            x_family.append((rho, 1.0))
        elif abs(theta_deg - 90.0) <= 18.0:
            y_family.append((rho, 1.0))
    if len(x_family) < min_lines or len(y_family) < min_lines:
        return None

    x_fit = _fit_nine_line_lattice(_cluster_positions(x_family, threshold=max(4.0, w / 70.0)), float(w), min_lines=min_lines)
    y_fit = _fit_nine_line_lattice(_cluster_positions(y_family, threshold=max(4.0, h / 70.0)), float(h), min_lines=min_lines)
    if x_fit is None or y_fit is None:
        return None

    x_positions = np.clip(np.array(x_fit["positions"], dtype=np.float32), 0, w - 1)
    y_positions = np.clip(np.array(y_fit["positions"], dtype=np.float32), 0, h - 1)
    support = 0.5 * (float(x_fit["support_ratio"]) + float(y_fit["support_ratio"]))
    regularity = 0.5 * (float(x_fit["regularity"]) + float(y_fit["regularity"]))
    evidence = _warp_grid_evidence_score(
        img,
        np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32),
    )
    score = 4.0 * support + 2.5 * regularity + 2.5 * evidence
    return {
        "x_edges": [int(round(x)) for x in x_positions],
        "y_edges": [int(round(y)) for y in y_positions],
        "support_ratio": float(support),
        "regularity": float(regularity),
        "evidence": float(evidence),
        "score": float(score),
        "trusted": bool(support >= 0.72 and regularity >= 0.72 and evidence >= 0.22),
    }


def crop_warp_to_detected_grid(
    img: Image.Image | np.ndarray,
    *,
    min_support: float = 0.55,
    min_evidence: float = 0.12,
) -> Image.Image | None:
    arr = _to_rgb_array(img)
    aligned = detect_aligned_square_grid(arr)
    if aligned is None:
        return None
    if float(aligned["support_ratio"]) < min_support or float(aligned["evidence"]) < min_evidence:
        return None
    x_edges = np.array(aligned["x_edges"], dtype=np.int32)
    y_edges = np.array(aligned["y_edges"], dtype=np.int32)
    x0 = max(0, int(x_edges[0]))
    x1 = min(arr.shape[1] - 1, int(x_edges[-1]))
    y0 = max(0, int(y_edges[0]))
    y1 = min(arr.shape[0] - 1, int(y_edges[-1]))
    if x1 - x0 < 80 or y1 - y0 < 80:
        return None
    cropped = arr[y0 : y1 + 1, x0 : x1 + 1]
    return Image.fromarray(cropped)


def detect_quad_board_boxes(
    img: Image.Image | np.ndarray,
    *,
    max_candidates: int = 6,
) -> list[dict[str, object]]:
    arr = _to_rgb_array(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    candidates = []

    for contour_map in _contour_maps(gray):
        for retrieval in (cv2.RETR_EXTERNAL, cv2.RETR_LIST):
            contours, _ = cv2.findContours(contour_map, retrieval, cv2.CHAIN_APPROX_SIMPLE)
            for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
                area = float(cv2.contourArea(contour))
                if area <= float(h * w) * 0.04:
                    continue
                peri = cv2.arcLength(contour, True)
                proposals: list[np.ndarray] = []
                for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
                    approx = cv2.approxPolyDP(contour, eps * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        proposals.append(approx.reshape(4, 2).astype(np.float32))
                        break
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect).astype(np.float32)
                proposals.append(box)
                for corners in proposals:
                    ordered = _order_corners(corners)
                    metrics = _quad_metrics(ordered, w, h)
                    if metrics["area_ratio"] < 0.08:
                        continue
                    if metrics["min_angle"] < 40.0 or metrics["max_angle"] > 140.0:
                        continue
                    evidence = _warp_grid_evidence_score(arr, ordered)
                    geometry = _warp_geometry_quality(metrics)
                    score = 3.8 * evidence + 3.2 * geometry
                    candidates.append(
                        {
                            "tag": "quad_box",
                            "corners": ordered,
                            "score": float(score),
                            "support_ratio": float(evidence),
                            "regularity": float(geometry),
                            "evidence": float(evidence),
                            "geometry": float(geometry),
                            "trusted": bool(evidence >= 0.18 and geometry >= 0.65),
                        }
                    )

    try:
        import recognizer_v6 as base_recognizer

        for tag, detector in (
            ("robust_contour", base_recognizer.find_board_corners),
            ("legacy_contour", base_recognizer.find_board_corners_legacy),
        ):
            corners = detector(Image.fromarray(arr))
            if corners is None:
                continue
            ordered = _order_corners(np.asarray(corners, dtype=np.float32))
            metrics = _quad_metrics(ordered, w, h)
            evidence = _warp_grid_evidence_score(arr, ordered)
            geometry = _warp_geometry_quality(metrics)
            score = 3.8 * evidence + 3.2 * geometry
            candidates.append(
                {
                    "tag": tag,
                    "corners": ordered,
                    "score": float(score),
                    "support_ratio": float(evidence),
                    "regularity": float(geometry),
                    "evidence": float(evidence),
                    "geometry": float(geometry),
                    "trusted": bool(evidence >= 0.18 and geometry >= 0.65),
                }
            )
    except Exception:
        pass

    candidates.sort(
        key=lambda row: (
            float(row["score"]),
            float(row["evidence"]),
            float(row["geometry"]),
        ),
        reverse=True,
    )
    deduped = []
    seen = set()
    for row in candidates:
        rounded = tuple(int(round(v / 16.0)) for v in row["corners"].reshape(-1))
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append(row)
        if len(deduped) >= max_candidates:
            break
    return deduped


def score_full_frame_board(img: Image.Image | np.ndarray) -> dict[str, object]:
    arr = _to_rgb_array(img)
    h, w = arr.shape[:2]
    aligned = detect_aligned_square_grid(arr)
    evidence = _warp_grid_evidence_score(
        arr,
        np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32),
    )
    support = float(aligned["support_ratio"]) if aligned is not None else 0.0
    regularity = float(aligned["regularity"]) if aligned is not None else 0.0
    score = 4.0 * support + 2.5 * regularity + 2.5 * evidence
    return {
        "tag": "full",
        "score": float(score),
        "support_ratio": float(support),
        "regularity": float(regularity),
        "evidence": float(evidence),
        "trusted": bool(support >= 0.72 and regularity >= 0.72 and evidence >= 0.22),
    }


def detect_axis_aligned_board_boxes(
    img: Image.Image | np.ndarray,
    *,
    max_candidates: int = 4,
) -> list[dict[str, object]]:
    arr = _to_rgb_array(img)
    h, w = arr.shape[:2]
    min_dim = min(h, w)
    if min_dim < 140:
        return []

    side_fracs = (0.44, 0.52, 0.60, 0.68, 0.76, 0.84, 0.92)
    candidates = []
    for frac in side_fracs:
        side = int(round(min_dim * frac))
        if side < 120 or side > max(h, w):
            continue
        x_max = max(0, w - side)
        y_max = max(0, h - side)
        x_step = max(24, side // 6)
        y_step = max(24, side // 6)
        x_positions = list(range(0, x_max + 1, x_step))
        y_positions = list(range(0, y_max + 1, y_step))
        if not x_positions or x_positions[-1] != x_max:
            x_positions.append(x_max)
        if not y_positions or y_positions[-1] != y_max:
            y_positions.append(y_max)
        for x0 in x_positions:
            for y0 in y_positions:
                roi = arr[y0 : y0 + side, x0 : x0 + side]
                aligned = detect_aligned_square_grid(roi)
                if aligned is None:
                    continue
                x_edges = np.array(aligned["x_edges"], dtype=np.float32)
                y_edges = np.array(aligned["y_edges"], dtype=np.float32)
                corners = np.array(
                    [
                        [x0 + x_edges[0], y0 + y_edges[0]],
                        [x0 + x_edges[-1], y0 + y_edges[0]],
                        [x0 + x_edges[-1], y0 + y_edges[-1]],
                        [x0 + x_edges[0], y0 + y_edges[-1]],
                    ],
                    dtype=np.float32,
                )
                width = max(1.0, float(corners[1, 0] - corners[0, 0]))
                height = max(1.0, float(corners[2, 1] - corners[1, 1]))
                area_ratio = (width * height) / float(max(1, w * h))
                geometry = max(0.0, min(1.0, 1.0 - abs(1.0 - width / height)))
                score = float(aligned["score"] + 1.2 * geometry + 0.6 * min(1.0, area_ratio / 0.30))
                candidates.append(
                    {
                        "tag": "aligned_box",
                        "corners": corners,
                        "score": score,
                        "support_ratio": float(aligned["support_ratio"]),
                        "regularity": float(aligned["regularity"]),
                        "evidence": float(aligned["evidence"]),
                        "geometry": float(geometry),
                        "trusted": bool(aligned["trusted"] and geometry >= 0.80),
                    }
                )

    candidates.sort(
        key=lambda row: (
            float(row["score"]),
            float(row["support_ratio"]),
            float(row["evidence"]),
            float(row["regularity"]),
        ),
        reverse=True,
    )
    deduped = []
    seen = set()
    for row in candidates:
        rounded = tuple(int(round(v / 12.0)) for v in row["corners"].reshape(-1))
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append(row)
        if len(deduped) >= max_candidates:
            break
    return deduped


def detect_board_grid(
    img: Image.Image | np.ndarray,
    *,
    max_hypotheses: int = 4,
) -> list[dict[str, object]]:
    hypotheses = detect_quad_board_boxes(img, max_candidates=max_hypotheses)
    if len(hypotheses) >= max_hypotheses:
        return hypotheses[:max_hypotheses]
    axis = detect_axis_aligned_board_boxes(img, max_candidates=max_hypotheses)
    merged = hypotheses[:]
    for row in axis:
        merged.append(row)
    merged.sort(
        key=lambda row: (
            float(row["score"]),
            float(row["evidence"]),
            float(row["geometry"]),
            float(row["support_ratio"]),
        ),
        reverse=True,
    )
    deduped = []
    seen = set()
    for row in merged:
        rounded = tuple(int(round(v / 16.0)) for v in row["corners"].reshape(-1))
        if rounded in seen:
            continue
        seen.add(rounded)
        deduped.append(row)
        if len(deduped) >= max_hypotheses:
            break
    return deduped
