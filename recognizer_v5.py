import argparse
import json
import os
import re
import sys

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import recognizer_v4 as v4
except ImportError:
    import train.recognizer_v4 as v4


DEBUG_MODE = False
USE_SQUARE_DETECTION = True
USE_EDGE_DETECTION = True
PIECE_LOG_PRIOR = -0.20


def parse_fen_board_rows(fen_board):
    rows = []
    for row in fen_board.split("/"):
        expanded = []
        for ch in row:
            if ch.isdigit():
                expanded.extend("1" * int(ch))
            else:
                expanded.append(ch)
        rows.append(expanded)
    if len(rows) != 8 or any(len(row) != 8 for row in rows):
        return None
    return rows


def is_square_attacked(rows, target_r, target_c, by_white):
    if by_white:
        pawn_attackers = [("P", target_r + 1, target_c - 1), ("P", target_r + 1, target_c + 1)]
    else:
        pawn_attackers = [("p", target_r - 1, target_c - 1), ("p", target_r - 1, target_c + 1)]

    for piece, r, c in pawn_attackers:
        if 0 <= r < 8 and 0 <= c < 8 and rows[r][c] == piece:
            return True

    knight = "N" if by_white else "n"
    for dr, dc in ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)):
        r, c = target_r + dr, target_c + dc
        if 0 <= r < 8 and 0 <= c < 8 and rows[r][c] == knight:
            return True

    king = "K" if by_white else "k"
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            r, c = target_r + dr, target_c + dc
            if 0 <= r < 8 and 0 <= c < 8 and rows[r][c] == king:
                return True

    rook_queen = {"R", "Q"} if by_white else {"r", "q"}
    bishop_queen = {"B", "Q"} if by_white else {"b", "q"}

    for dr, dc, attackers in (
        (-1, 0, rook_queen),
        (1, 0, rook_queen),
        (0, -1, rook_queen),
        (0, 1, rook_queen),
        (-1, -1, bishop_queen),
        (-1, 1, bishop_queen),
        (1, -1, bishop_queen),
        (1, 1, bishop_queen),
    ):
        r, c = target_r + dr, target_c + dc
        while 0 <= r < 8 and 0 <= c < 8:
            sq = rows[r][c]
            if sq != "1":
                if sq in attackers:
                    return True
                break
            r += dr
            c += dc

    return False


def infer_side_to_move_from_checks(fen_board):
    rows = parse_fen_board_rows(fen_board)
    if rows is None:
        return "w", "default_invalid_board"

    white_king = None
    black_king = None
    for r in range(8):
        for c in range(8):
            if rows[r][c] == "K":
                white_king = (r, c)
            elif rows[r][c] == "k":
                black_king = (r, c)

    if white_king is None or black_king is None:
        return "w", "default_missing_king"

    white_in_check = is_square_attacked(rows, white_king[0], white_king[1], by_white=False)
    black_in_check = is_square_attacked(rows, black_king[0], black_king[1], by_white=True)

    if white_in_check and not black_in_check:
        return "w", "check_inference"
    if black_in_check and not white_in_check:
        return "b", "check_inference"
    if white_in_check and black_in_check:
        return "w", "default_double_check_conflict"
    return "w", "default_no_check_signal"


def board_plausibility_score(fen_board):
    """Heuristic structural score: prioritize board sanity over raw model confidence."""
    rows = v4.expand_fen_board(fen_board)
    if len(rows) != 8 or any(len(row) != 8 for row in rows):
        return -1e9

    white_king = fen_board.count("K")
    black_king = fen_board.count("k")
    white_pawns = fen_board.count("P")
    black_pawns = fen_board.count("p")
    white_pieces = sum(1 for ch in fen_board if ch.isupper())
    black_pieces = sum(1 for ch in fen_board if ch.islower())
    total_pieces = white_pieces + black_pieces
    pawns_on_back_rank = sum(1 for ch in (rows[0] + rows[7]) if ch in {"P", "p"})

    score = 0.0

    if white_king == 1:
        score += 6.0
    else:
        score -= 20.0 * abs(white_king - 1)

    if black_king == 1:
        score += 6.0
    else:
        score -= 20.0 * abs(black_king - 1)

    if white_pawns <= 8:
        score += 1.5
    else:
        score -= 5.0 * (white_pawns - 8)

    if black_pawns <= 8:
        score += 1.5
    else:
        score -= 5.0 * (black_pawns - 8)

    if white_pieces <= 16:
        score += 1.0
    else:
        score -= 2.0 * (white_pieces - 16)

    if black_pieces <= 16:
        score += 1.0
    else:
        score -= 2.0 * (black_pieces - 16)

    if total_pieces <= 32:
        score += 1.0
    else:
        score -= 2.0 * (total_pieces - 32)

    score -= 2.0 * pawns_on_back_rank
    return score


def infer_fen_on_image_deep_topk(
    img,
    model,
    device,
    use_square_detection,
    board_perspective="white",
    topk_k=8,
):
    """v5 tile inference with deeper per-tile top-k for cleaner structural repair."""
    if board_perspective not in {"white", "black"}:
        raise ValueError("board_perspective must be 'white' or 'black'")

    w, h = img.size
    if use_square_detection:
        grid = v4.detect_grid_lines(img)
        if grid is not None:
            xe, ye = grid
        else:
            xe = np.linspace(0, w, 9).astype(int)
            ye = np.linspace(0, h, 9).astype(int)
    else:
        xe = np.linspace(0, w, 9).astype(int)
        ye = np.linspace(0, h, 9).astype(int)

    fen_rows = []
    confs = []
    tile_infos = []
    piece_count = 0
    empty_idx = v4.FEN_CHARS.index("1")
    with torch.no_grad():
        for r in range(8):
            row = ""
            for c in range(8):
                image_r = r
                image_c = c
                if board_perspective == "black":
                    image_r = 7 - r
                    image_c = 7 - c

                x0 = max(0, int(xe[image_c] - v4.TILE_CONTEXT_PAD))
                y0 = max(0, int(ye[image_r] - v4.TILE_CONTEXT_PAD))
                x1 = min(w, int(xe[image_c + 1] + v4.TILE_CONTEXT_PAD))
                y1 = min(h, int(ye[image_r + 1] + v4.TILE_CONTEXT_PAD))
                tile = img.crop((x0, y0, x1, y1)).resize((v4.IMG_SIZE, v4.IMG_SIZE), Image.LANCZOS)
                img_np = np.array(tile).transpose(2, 0, 1)
                tensor = torch.from_numpy(img_np).float().to(device)
                tensor = (tensor / 127.5) - 1.0
                tensor = tensor.unsqueeze(0)

                out = torch.softmax(model(tensor), dim=1)[0]
                topk_probs, topk_pred = torch.topk(out, k=min(topk_k, len(v4.FEN_CHARS)))
                topk = [
                    (v4.FEN_CHARS[int(idx.item())], float(prob.item()))
                    for prob, idx in zip(topk_probs, topk_pred)
                ]

                # Global sparsity prior: empty squares are slightly favored unless
                # a piece probability is clearly stronger.
                adjusted_scores = torch.log(out + 1e-12) + PIECE_LOG_PRIOR
                adjusted_scores[empty_idx] = torch.log(out[empty_idx] + 1e-12)
                pred_idx = int(torch.argmax(adjusted_scores).item())
                pred_prob = float(out[pred_idx].item())

                if pred_prob < 0.35:
                    label = "1"
                else:
                    label = v4.FEN_CHARS[pred_idx]
                row += label
                if label != "1":
                    piece_count += 1
                max_prob = float(torch.max(out).item())
                confs.append(max_prob)
                tile_infos.append(
                    {
                        "row": r,
                        "col": c,
                        "label": label,
                        "prob": max_prob,
                        "topk": topk,
                    }
                )
            fen_rows.append(row)

    result = "/".join(fen_rows)
    result = "/".join(
        [re.sub(r"1+", lambda m: str(len(m.group())), row) for row in result.split("/")]
    )
    repaired = v4.repair_duplicate_kings(tile_infos)
    if repaired is not None:
        repaired_fen, repaired_conf, repaired_piece_count = repaired
        return repaired_fen, repaired_conf, repaired_piece_count
    return result, float(np.mean(confs)), piece_count


class BoardDetector:
    """Modular board detector that combines contour and lattice hypotheses."""

    def __init__(self, debug=False):
        self.debug = debug

    def _log(self, msg):
        if self.debug:
            print(f"DEBUG: {msg}", file=sys.stderr)

    def _gray(self, img):
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    def _angular_distance_deg(self, a, b):
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    def _dominant_orthogonal_orientations(self, angles, weights):
        if not angles:
            return None
        hist = np.zeros(180, dtype=np.float32)
        for angle, weight in zip(angles, weights):
            hist[int(round(angle)) % 180] += max(float(weight), 1e-3)

        primary = int(np.argmax(hist))
        secondary_target = (primary + 90) % 180
        search_radius = 18
        best_secondary = None
        best_secondary_val = -1.0
        for delta in range(-search_radius, search_radius + 1):
            idx = (secondary_target + delta) % 180
            if hist[idx] > best_secondary_val:
                best_secondary_val = float(hist[idx])
                best_secondary = idx
        if best_secondary is None or best_secondary_val <= 0.0:
            return None
        return float(primary), float(best_secondary)

    def _line_intersection(self, n1, rho1, n2, rho2):
        a = np.array([[n1[0], n1[1]], [n2[0], n2[1]]], dtype=np.float32)
        b = np.array([rho1, rho2], dtype=np.float32)
        det = float(np.linalg.det(a))
        if abs(det) < 1e-6:
            return None
        pt = np.linalg.solve(a, b)
        return float(pt[0]), float(pt[1])

    def _cluster_axis(self, items, threshold=10):
        if not items:
            return []
        items = sorted(items, key=lambda item: item[0])
        clusters = [[items[0]]]
        for item in items[1:]:
            if item[0] - clusters[-1][-1][0] <= threshold:
                clusters[-1].append(item)
            else:
                clusters.append([item])

        merged = []
        for cluster in clusters:
            positions = np.array([item[0] for item in cluster], dtype=np.float32)
            weights = np.array([item[1] for item in cluster], dtype=np.float32)
            merged.append(
                {
                    "pos": float(np.average(positions, weights=np.maximum(weights, 1e-3))),
                    "weight": float(weights.sum()),
                    "count": len(cluster),
                }
            )
        return merged

    def _best_line_window(self, clustered, span):
        if len(clustered) < span:
            return None

        best = None
        for start in range(0, len(clustered) - span + 1):
            window = clustered[start : start + span]
            positions = np.array([line["pos"] for line in window], dtype=np.float32)
            diffs = np.diff(positions)
            if np.any(diffs <= 1):
                continue
            spacing_mean = float(np.mean(diffs))
            spacing_std = float(np.std(diffs))
            regularity = 1.0 / (1.0 + (spacing_std / max(spacing_mean, 1e-6)))
            strength = float(sum(line["weight"] for line in window))
            score = regularity * 1000.0 + strength
            candidate = {
                "positions": positions,
                "regularity": regularity,
                "strength": strength,
                "spacing_mean": spacing_mean,
                "score": score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
        return best

    def detect_lattice(self, img):
        gray = self._gray(img)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120, apertureSize=3)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=60,
            minLineLength=min(img.size) // 5,
            maxLineGap=25,
        )
        if lines is None:
            return None

        segments = []
        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [int(v) for v in line]
            dx = x2 - x1
            dy = y2 - y1
            length = float(np.hypot(dx, dy))
            if length < min(img.size) * 0.18:
                continue
            angle = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
            mx = (x1 + x2) / 2.0
            my = (y1 + y2) / 2.0
            segments.append((angle, length, mx, my))
        if len(segments) < 18:
            return None

        orientations = self._dominant_orthogonal_orientations(
            [seg[0] for seg in segments],
            [seg[1] for seg in segments],
        )
        if orientations is None:
            return None
        theta_a, theta_b = orientations

        family_a = []
        family_b = []
        for angle, length, mx, my in segments:
            da = self._angular_distance_deg(angle, theta_a)
            db = self._angular_distance_deg(angle, theta_b)
            if da <= 18 and da <= db:
                na = np.deg2rad(theta_a + 90.0)
                pos = mx * np.cos(na) + my * np.sin(na)
                family_a.append((pos, length))
            elif db <= 18:
                nb = np.deg2rad(theta_b + 90.0)
                pos = mx * np.cos(nb) + my * np.sin(nb)
                family_b.append((pos, length))

        a_clusters = self._cluster_axis(family_a, threshold=max(8, img.size[0] // 80))
        b_clusters = self._cluster_axis(family_b, threshold=max(8, img.size[1] // 80))
        a_best = self._best_line_window(a_clusters, 9)
        b_best = self._best_line_window(b_clusters, 9)
        if a_best is None or b_best is None:
            return None

        n_a = np.array(
            [np.cos(np.deg2rad(theta_a + 90.0)), np.sin(np.deg2rad(theta_a + 90.0))],
            dtype=np.float32,
        )
        n_b = np.array(
            [np.cos(np.deg2rad(theta_b + 90.0)), np.sin(np.deg2rad(theta_b + 90.0))],
            dtype=np.float32,
        )

        grid_points = []
        for rho_a in a_best["positions"]:
            row_pts = []
            for rho_b in b_best["positions"]:
                pt = self._line_intersection(n_a, float(rho_a), n_b, float(rho_b))
                if pt is None:
                    return None
                row_pts.append(pt)
            grid_points.append(row_pts)

        corners = np.array(
            [
                grid_points[0][0],
                grid_points[0][-1],
                grid_points[-1][-1],
                grid_points[-1][0],
            ],
            dtype=np.float32,
        )

        score = a_best["score"] + b_best["score"]
        self._log(
            "lattice "
            f"a_reg={a_best['regularity']:.3f} b_reg={b_best['regularity']:.3f} "
            f"a_strength={a_best['strength']:.1f} b_strength={b_best['strength']:.1f} "
            f"theta=({theta_a:.1f},{theta_b:.1f})"
        )
        return {
            "tag": "lattice",
            "corners": v4.order_corners(corners),
            "score": float(score),
            "grid": (a_best["positions"], b_best["positions"]),
        }

    def detect_contour(self, img):
        corners = v4.find_board_corners(img)
        if corners is None:
            return None
        return {"tag": "robust", "corners": corners}

    def candidate_images(self, img):
        candidates = [("full", img)]
        ih, iw = img.size[1], img.size[0]

        for detector in (self.detect_lattice, self.detect_contour):
            result = detector(img)
            if result is None:
                continue
            corners = result["corners"]
            metrics = v4.compute_quad_metrics(corners, iw, ih)
            if not v4.is_warp_geometry_trustworthy(metrics):
                self._log(
                    f"reject {result['tag']} area={metrics['area_ratio']:.3f} "
                    f"opp={metrics['opposite_similarity']:.3f} asp={metrics['aspect_similarity']:.3f}"
                )
                continue
            warped = v4.perspective_transform(img, corners)
            candidates.append((result["tag"], warped))
            candidates.append((f"{result['tag']}_inset2", v4.inset_board(warped, 2)))
        return candidates


def predict_board(image_path, model_path=None, board_perspective="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_model_path = model_path or v4.MODEL_PATH
    if not resolved_model_path:
        raise FileNotFoundError("No default model found. Use --model-path explicitly.")

    model = v4.StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(resolved_model_path, map_location=device))
    model.eval()

    img = Image.open(image_path).convert("RGB")
    detector = BoardDetector(debug=DEBUG_MODE)
    candidates = detector.candidate_images(img) if USE_EDGE_DETECTION else [("full", img)]

    full_candidate_img = v4.trim_dark_edge_bars(candidates[0][1].copy())
    label_perspective_result = None
    label_details = {"left": None, "right": None}
    labels_absent = True
    labels_same = False
    if board_perspective == "auto":
        label_perspective_result = v4.infer_board_perspective_from_labels(full_candidate_img)
        label_details = label_perspective_result["details"] if label_perspective_result else {
            "left": v4.classify_file_label_crop(v4.extract_file_label_crop(full_candidate_img, side="left")),
            "right": v4.classify_file_label_crop(v4.extract_file_label_crop(full_candidate_img, side="right")),
        }
        labels_absent = label_details["left"] is None and label_details["right"] is None
        labels_same = (
            label_details["left"] is not None
            and label_details["right"] is not None
            and label_details["left"]["label"] == label_details["right"]["label"]
        )

    scored = []
    for tag, candidate_img in candidates:
        candidate_img = v4.trim_dark_edge_bars(candidate_img)
        fen, conf, piece_count = infer_fen_on_image_deep_topk(
            candidate_img,
            model,
            device,
            USE_SQUARE_DETECTION,
        )
        if board_perspective == "auto":
            if label_perspective_result is None:
                detected_perspective = "white"
                perspective_source = "default"
                if labels_absent or labels_same:
                    fallback = v4.infer_board_perspective_from_piece_distribution(fen)
                    if fallback == "black":
                        detected_perspective = "black"
                        perspective_source = "piece_distribution_fallback"
            else:
                detected_perspective = label_perspective_result["perspective"]
                perspective_source = label_perspective_result["source"]
        elif board_perspective not in {"white", "black"}:
            raise ValueError("board_perspective must be 'auto', 'white', or 'black'")
        else:
            detected_perspective = board_perspective
            perspective_source = "override"

        final_fen = v4.rotate_fen_180(fen) if detected_perspective == "black" else fen
        scored.append((tag, candidate_img, final_fen, conf, piece_count, detected_perspective, perspective_source))
        if DEBUG_MODE:
            print(
                f"DEBUG: V5 Candidate={tag} conf={conf:.4f} pieces={piece_count} "
                f"perspective={detected_perspective} source={perspective_source}",
                file=sys.stderr,
            )

    full_piece_count = next((item[4] for item in scored if item[0] == "full"), 0)
    full_conf = next((item[3] for item in scored if item[0] == "full"), 0.0)
    filtered = []
    for item in scored:
        tag, _, _, _, piece_count, _, _ = item
        if (
            tag != "full"
            and full_conf >= v4.FULL_CONF_FOR_COVERAGE_GUARD
            and full_piece_count >= v4.WARP_PIECE_COVERAGE_MIN_FULL
            and piece_count <= int(full_piece_count * v4.WARP_PIECE_COVERAGE_RATIO)
        ):
            continue
        filtered.append(item)

    if not filtered:
        filtered = scored

    enriched = []
    for item in filtered:
        plausibility = board_plausibility_score(item[2])
        enriched.append((item, plausibility))
        if DEBUG_MODE:
            print(
                f"DEBUG: V5 Candidate={item[0]} plausibility={plausibility:.2f} conf={item[3]:.4f}",
                file=sys.stderr,
            )

    (best_tag, _, best_fen, best_conf, _, best_perspective, best_perspective_source), _ = max(
        enriched, key=lambda pair: (pair[1], pair[0][3])
    )
    if DEBUG_MODE:
        print(
            f"DEBUG: V5 Selected candidate={best_tag} confidence={best_conf:.4f} "
            f"perspective={best_perspective} source={best_perspective_source} "
            f"model={resolved_model_path}",
            file=sys.stderr,
        )
    return best_fen, best_conf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize chess position from image (v5 experimental)")
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
    parser.add_argument("--debug", action="store_true", help="Verbose debugging")
    args = parser.parse_args()

    USE_EDGE_DETECTION = not args.no_edge_detection
    USE_SQUARE_DETECTION = not args.no_square_detection
    DEBUG_MODE = args.debug

    try:
        fen, conf = predict_board(
            args.image,
            model_path=args.model_path,
            board_perspective=args.board_perspective,
        )
        side_to_move, side_source = infer_side_to_move_from_checks(fen)
        print(
            json.dumps(
                {
                    "success": True,
                    "fen": f"{fen} {side_to_move} - - 0 1",
                    "confidence": round(conf, 4),
                    "side_to_move": side_to_move,
                    "side_to_move_source": side_source,
                }
            )
        )
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}))
