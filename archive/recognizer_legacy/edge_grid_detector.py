#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image

from recognizer_v6_candidate_core import (
    compute_quad_metrics,
    is_warp_geometry_relaxed,
    is_warp_geometry_trustworthy,
    order_corners,
    perspective_transform,
    warp_geometry_quality,
)


def _gray_rgb(img):
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def _smooth_1d(signal, kernel):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size < kernel or kernel <= 1:
        return signal
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(signal.reshape(-1, 1), (1, kernel), 0).reshape(-1)


def _normalize_signal(signal):
    signal = np.asarray(signal, dtype=np.float32)
    signal -= float(signal.min())
    vmax = float(signal.max())
    if vmax <= 1e-6:
        return signal
    return signal / vmax


def _sample_signal(signal, center, radius):
    lo = max(0, int(round(center - radius)))
    hi = min(len(signal), int(round(center + radius + 1)))
    if hi <= lo:
        return 0.0
    return float(np.mean(signal[lo:hi]))


def _search_axis_grid(signal):
    n = int(len(signal))
    if n < 64:
        return None
    smooth = _smooth_1d(signal, max(9, (n // 48) * 2 + 1))
    smooth = _normalize_signal(smooth)
    low_step = max(8, int(round(n * 0.085)))
    high_step = min(int(round(n * 0.18)), n // 2)
    best = None
    radius = max(1.0, n / 256.0)
    for step in range(low_step, max(low_step + 1, high_step + 1)):
        max_start = n - 1 - (8 * step)
        if max_start < 0:
            continue
        start_count = min(32, max(1, max_start + 1))
        starts = np.linspace(0, max_start, start_count)
        for start in starts:
            positions = [start + (i * step) for i in range(9)]
            values = np.array([_sample_signal(smooth, pos, radius) for pos in positions], dtype=np.float32)
            line_score = float(np.mean(values))
            support_ratio = float(np.mean(values >= max(0.25, float(np.mean(smooth)))))
            border_strength = float((values[0] + values[-1]) * 0.5)
            regularity = float(1.0 - min(1.0, np.std(np.diff(positions)) / max(1.0, step)))
            score = (0.55 * line_score) + (0.25 * support_ratio) + (0.15 * border_strength) + (0.05 * regularity)
            row = {
                "start": float(start),
                "step": float(step),
                "positions": positions,
                "line_score": line_score,
                "support_ratio": support_ratio,
                "border_strength": border_strength,
                "regularity": regularity,
                "score": score,
            }
            if best is None or row["score"] > best["score"]:
                best = row
    return best


def evaluate_warped_grid(img):
    _, gray = _gray_rgb(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)
    gx = np.abs(cv2.Sobel(norm, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(norm, cv2.CV_32F, 0, 1, ksize=3))
    col_signal = _normalize_signal(np.mean(gx, axis=0))
    row_signal = _normalize_signal(np.mean(gy, axis=1))
    x_grid = _search_axis_grid(col_signal)
    y_grid = _search_axis_grid(row_signal)
    if x_grid is None or y_grid is None:
        return {
            "evidence": 0.0,
            "support_ratio": 0.0,
            "score": 0.0,
            "grid_box": None,
            "x_grid": x_grid,
            "y_grid": y_grid,
        }
    x0 = x_grid["start"]
    x1 = x_grid["start"] + (8.0 * x_grid["step"])
    y0 = y_grid["start"]
    y1 = y_grid["start"] + (8.0 * y_grid["step"])
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    ratio = min(width / height, height / width)
    coverage = min(1.0, (width * height) / float(img.size[0] * img.size[1]))
    evidence = float(
        (0.35 * x_grid["line_score"])
        + (0.35 * y_grid["line_score"])
        + (0.15 * x_grid["border_strength"])
        + (0.15 * y_grid["border_strength"])
    )
    support_ratio = float(0.5 * (x_grid["support_ratio"] + y_grid["support_ratio"]))
    score = float((0.55 * evidence) + (0.20 * support_ratio) + (0.15 * ratio) + (0.10 * coverage))
    return {
        "evidence": evidence,
        "support_ratio": support_ratio,
        "score": score,
        "coverage": coverage,
        "ratio": ratio,
        "grid_box": (float(x0), float(y0), float(x1), float(y1)),
        "x_grid": x_grid,
        "y_grid": y_grid,
    }


def score_full_frame_board(img):
    meta = evaluate_warped_grid(img)
    ratio = img.size[0] / float(max(1, img.size[1]))
    ratio_score = min(ratio, 1.0 / max(ratio, 1e-6))
    score = float((0.70 * meta["score"]) + (0.30 * ratio_score))
    trusted = bool(
        0.88 <= ratio <= 1.12
        and meta["evidence"] >= 0.34
        and meta["support_ratio"] >= 0.46
    )
    return {
        **meta,
        "score": score,
        "trusted": trusted,
        "ratio_score": ratio_score,
    }


def crop_warp_to_detected_grid(img):
    meta = evaluate_warped_grid(img)
    box = meta["grid_box"]
    if box is None:
        return None
    x0, y0, x1, y1 = box
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    coverage = (width * height) / float(img.size[0] * img.size[1])
    ratio = width / float(max(1.0, height))
    if coverage < 0.55 or coverage > 0.995:
        return None
    if not (0.88 <= ratio <= 1.12):
        return None
    px0 = max(0, int(round(x0)))
    py0 = max(0, int(round(y0)))
    px1 = min(img.size[0], int(round(x1)))
    py1 = min(img.size[1], int(round(y1)))
    if px1 - px0 < img.size[0] * 0.55 or py1 - py0 < img.size[1] * 0.55:
        return None
    cropped = img.crop((px0, py0, px1, py1))
    if cropped.size == img.size:
        return None
    return cropped.resize(img.size, Image.LANCZOS)


def _grid_margin_score(meta, img_size, min_span_ratio=0.0):
    box = meta.get("grid_box")
    if box is None:
        return 0.0
    width, height = img_size
    x0, y0, x1, y1 = box
    margins = np.array(
        [x0, y0, max(0.0, width - x1), max(0.0, height - y1)],
        dtype=np.float32,
    )
    avg_margin = float(np.mean(margins) / max(1.0, min(width, height)))
    inset_score = float(max(0.0, 1.0 - (avg_margin / 0.18)))
    tight_score = 0.0
    if min_span_ratio >= 0.85:
        tight_score = float(max(0.0, 1.0 - (avg_margin / 0.025)))
    return max(inset_score, tight_score)


def _checker_texture_score(img):
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    cell = 64
    means = np.zeros((8, 8), dtype=np.float32)
    stds = np.zeros((8, 8), dtype=np.float32)
    for row in range(8):
        for col in range(8):
            patch = gray[row * cell:(row + 1) * cell, col * cell:(col + 1) * cell]
            inner = patch[12:-12, 12:-12]
            means[row, col] = float(np.mean(inner))
            stds[row, col] = float(np.std(inner))

    parity = (np.indices((8, 8)).sum(axis=0) % 2).astype(bool)
    even = means[~parity]
    odd = means[parity]
    sep = float(abs(float(np.mean(even)) - float(np.mean(odd))))
    within = float(0.5 * (float(np.std(even)) + float(np.std(odd))))
    contrast = sep / max(1e-4, within)
    row_alt = float(np.mean(np.abs(np.diff(means, axis=1))))
    col_alt = float(np.mean(np.abs(np.diff(means, axis=0))))
    alt = 0.5 * (row_alt + col_alt)
    texture = float(np.mean(stds))

    contrast_score = min(1.0, contrast / 1.2)
    sep_score = min(1.0, sep / 0.12)
    alt_score = min(1.0, alt / 0.18)
    texture_target = max(0.0, 1.0 - (abs(texture - 0.12) / 0.12))
    return float(
        (0.35 * contrast_score)
        + (0.25 * sep_score)
        + (0.25 * alt_score)
        + (0.15 * texture_target)
    )


def _proposal_dedupe_key(corners, width, height):
    corners = np.asarray(corners, dtype=np.float32)
    center = corners.mean(axis=0)
    metrics = compute_quad_metrics(corners, width, height)
    return (
        round(float(center[0]) / max(1.0, width), 2),
        round(float(center[1]) / max(1.0, height), 2),
        round(metrics["area_ratio"], 2),
    )


def _collect_contour_proposals(img):
    _, gray = _gray_rgb(img)
    height, width = gray.shape
    proposals = []
    seen = set()

    variants = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)
    variants.append(cv2.Canny(norm, 30, 100))
    variants.append(cv2.Canny(norm, 50, 150))
    variants.append(cv2.Canny(norm, 70, 200))
    adaptive = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    variants.append(adaptive)

    for variant in variants:
        contours, _ = cv2.findContours(variant, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:120]:
            area = cv2.contourArea(contour)
            if area < (width * height * 0.12):
                continue
            peri = cv2.arcLength(contour, True)
            candidate_quads = []
            for eps in (0.01, 0.02, 0.03, 0.05):
                approx = cv2.approxPolyDP(contour, eps * peri, True)
                if len(approx) == 4:
                    candidate_quads.append(("contour_quad", order_corners(approx.reshape(4, 2))))
                    break
            rect = cv2.minAreaRect(contour)
            rect_box = cv2.boxPoints(rect)
            candidate_quads.append(("min_area_rect", order_corners(rect_box)))
            for tag, corners in candidate_quads:
                metrics = compute_quad_metrics(corners, width, height)
                if not is_warp_geometry_relaxed(metrics):
                    continue
                dedupe = (tag,) + _proposal_dedupe_key(corners, width, height)
                if dedupe in seen:
                    continue
                seen.add(dedupe)
                proposals.append(
                    {
                        "tag": tag,
                        "corners": corners.astype(np.float32),
                        "metrics": metrics,
                    }
                )
    return proposals


def _collect_bright_board_proposals(img):
    rgb = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    height, width = rgb.shape[:2]
    mask = (((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 150)).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    proposals = []
    seen = set()
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = (w * h) / float(max(1, width * height))
        aspect_similarity = min(w / float(max(h, 1)), h / float(max(w, 1)))
        if area_ratio < 0.10 or aspect_similarity < 0.45:
            continue
        # Bright framed boards in screenshot layouts often sit inside a wider bright
        # component; search a few square windows inside that component.
        side_candidates = sorted(
            {
                int(round(h * 1.02)),
                int(round(h * 1.06)),
                int(round(h * 1.10)),
            }
        )
        x_offsets = [0.08, 0.12, 0.16]
        y_offsets = [-0.02, 0.0, 0.02]
        for side in side_candidates:
            if side <= 0:
                continue
            side = min(side, width, height)
            for x_frac in x_offsets:
                for y_frac in y_offsets:
                    px = int(round(x + (w * x_frac)))
                    py = int(round(y + (h * y_frac)))
                    px = max(0, min(px, width - side))
                    py = max(0, min(py, height - side))
                    corners = np.array(
                        [[px, py], [px + side, py], [px + side, py + side], [px, py + side]],
                        dtype=np.float32,
                    )
                    dedupe = ("bright_board_square",) + _proposal_dedupe_key(corners, width, height)
                    if dedupe in seen:
                        continue
                    seen.add(dedupe)
                    proposals.append(
                        {
                            "tag": "bright_board_square",
                            "corners": corners,
                            "metrics": compute_quad_metrics(corners, width, height),
                        }
                    )
    return proposals


def _score_proposal(img, proposal):
    corners = proposal["corners"]
    metrics = proposal["metrics"]
    span_x = float(corners[:, 0].max() - corners[:, 0].min()) / float(max(1, img.size[0]))
    span_y = float(corners[:, 1].max() - corners[:, 1].min()) / float(max(1, img.size[1]))
    min_span_ratio = float(min(span_x, span_y))
    warped = perspective_transform(img, corners)
    refined = crop_warp_to_detected_grid(warped)
    scoring_img = refined if refined is not None else warped
    warped_eval = evaluate_warped_grid(scoring_img)
    geometry = warp_geometry_quality(metrics)
    evidence = float(warped_eval["evidence"])
    support_ratio = float(warped_eval["support_ratio"])
    grid_ratio_score = float(warped_eval.get("ratio", 0.0))
    x_line = float(warped_eval["x_grid"]["line_score"]) if warped_eval.get("x_grid") else 0.0
    y_line = float(warped_eval["y_grid"]["line_score"]) if warped_eval.get("y_grid") else 0.0
    axis_balance = float(min(x_line, y_line) / max(x_line, y_line, 1e-6))
    margin_score = _grid_margin_score(
        warped_eval,
        (scoring_img.size[0], scoring_img.size[1]),
        min_span_ratio=min_span_ratio,
    )
    checker_score = _checker_texture_score(scoring_img)
    score = float(
        (0.14 * geometry)
        + (0.12 * evidence)
        + (0.10 * support_ratio)
        + (0.04 * margin_score)
        + (0.34 * checker_score)
        + (0.12 * grid_ratio_score)
        + (0.14 * axis_balance)
    )
    if refined is not None and checker_score >= 0.65:
        score += 0.03
    trusted = bool(is_warp_geometry_trustworthy(metrics) and evidence >= 0.34 and support_ratio >= 0.46)
    return {
        "tag": proposal["tag"],
        "corners": corners,
        "geometry": float(geometry),
        "evidence": evidence,
        "score": score,
        "support_ratio": support_ratio,
        "grid_ratio_score": float(grid_ratio_score),
        "axis_balance": float(axis_balance),
        "margin_score": float(margin_score),
        "min_span_ratio": float(min_span_ratio),
        "checker_score": float(checker_score),
        "trusted": trusted,
    }


def detect_board_grid(img, max_hypotheses=6):
    hypotheses = []
    full_meta = score_full_frame_board(img)
    contour_proposals = _collect_contour_proposals(img)
    for proposal in contour_proposals:
        hypotheses.append(_score_proposal(img, proposal))
    for proposal in _collect_bright_board_proposals(img):
        hypotheses.append(_score_proposal(img, proposal))

    if full_meta["grid_box"] is not None:
        x0, y0, x1, y1 = full_meta["grid_box"]
        width = max(1.0, x1 - x0)
        height = max(1.0, y1 - y0)
        ratio = width / height
        coverage = (width * height) / float(img.size[0] * img.size[1])
        if 0.88 <= ratio <= 1.12 and 0.35 <= coverage <= 0.98:
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            metrics = compute_quad_metrics(corners, img.size[0], img.size[1])
            hypotheses.append(
                _score_proposal(
                    img,
                    {
                        "tag": "aligned_box",
                        "corners": corners,
                        "metrics": metrics,
                    },
                )
            )

    deduped = []
    seen = set()
    for row in sorted(hypotheses, key=lambda item: (item["score"], item["evidence"], item["geometry"]), reverse=True):
        key = _proposal_dedupe_key(row["corners"], img.size[0], img.size[1])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= max_hypotheses:
            break
    if len(deduped) >= 2:
        best_score = float(deduped[0]["score"])
        ambiguous = [row for row in deduped if best_score - float(row["score"]) <= 0.05]
        if len(ambiguous) >= 2:
            ambiguous.sort(
                key=lambda row: (
                    (0.60 * float(row.get("checker_score", 0.0))) + (0.40 * float(row.get("evidence", 0.0))),
                    float(row.get("support_ratio", 0.0)),
                    float(row.get("score", 0.0)),
                ),
                reverse=True,
            )
            remainder = [row for row in deduped if best_score - float(row["score"]) > 0.05]
            deduped = ambiguous + remainder
    return deduped
