import os, io, random, json
from functools import lru_cache

try:
    import chess
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'python-chess'. Run: python3 -m pip install -r requirements.txt"
    ) from exc

import torch, numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageFont

# THE GLOBAL LABEL LAW (Aligned with audit_dataset.py)
FEN_CHARS = "1PNBRQKpnbrqk" 
# 0:1, 1:P, 2:N, 3:B, 4:R, 5:Q, 6:K, 7:p, 8:n, 9:b, 10:r, 11:q, 12:k

IMG_SIZE = 64
PRINT_DIAGRAM_PROFILES = {
    "mono_scan",
    "mono_print_sparse_edge",
    "mono_print_sparse_light",
    "mono_print_edge_rook",
    "book_page_sparse",
    "shirt_print_sparse",
    "edge_rook_page",
    "book_page_reference",
    "shirt_print_reference",
    "diagtransfer_hatched",
}
PRINT_DIAGRAM_BASE_SET = "cburnett"


def env_float(name, default):
    raw = os.getenv(name)
    return float(raw) if raw is not None else default


def env_int(name, default):
    raw = os.getenv(name)
    return int(raw) if raw is not None else default


def env_str(name, default):
    raw = os.getenv(name)
    return raw if raw is not None else default


BOARDS_PER_CHUNK = env_int("BOARDS_PER_CHUNK", 1000)
CHUNKS_TRAIN = env_int("CHUNKS_TRAIN", 10)
CHUNKS_VAL = env_int("CHUNKS_VAL", 2)
SEED = env_int("SEED", 1337)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOARD_THEMES_DIR = os.path.join(BASE_DIR, "board_themes")
PIECE_SETS_DIR = os.path.join(BASE_DIR, "piece_sets")
OUTPUT_DIR = os.path.join(BASE_DIR, env_str("OUTPUT_DIR", "tensors_v6_targeted_recovery_v13"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Keep broad/default profiles on the established, general-purpose piece pool.
# New imported sets should only affect training when a profile names them explicitly.
DEFAULT_GENERAL_PIECE_SET_NAMES = [
    "alpha",
    "cardinal",
    "cburnett",
    "celtic",
    "chessmonk",
    "chessnut",
    "governor",
    "icpieces",
    "maestro",
    "merida",
    "staunty",
]

BASE_CONFIG = {
    "ENABLE_WATERMARK_AUG": os.getenv("ENABLE_WATERMARK_AUG", "1") != "0",
    "WATERMARK_BOARD_PROB": env_float("WATERMARK_BOARD_PROB", 0.3),
    "WATERMARK_MIN_PER_BOARD": env_int("WATERMARK_MIN_PER_BOARD", 1),
    "WATERMARK_MAX_PER_BOARD": env_int("WATERMARK_MAX_PER_BOARD", 2),
    "WATERMARK_SCALE_MIN": env_float("WATERMARK_SCALE_MIN", 1.0),
    "WATERMARK_SCALE_MAX": env_float("WATERMARK_SCALE_MAX", 1.5),
    "WATERMARK_FULL_KING_WORDMARK_PROB": env_float("WATERMARK_FULL_KING_WORDMARK_PROB", 0.55),
    "LABELS_PROB": env_float("LABELS_PROB", 0.60),
    "TRIM_CAPTURE_PROB": env_float("TRIM_CAPTURE_PROB", 0.35),
    "ARTIFACT_EMPTY_TILE_PROB": env_float("ARTIFACT_EMPTY_TILE_PROB", 0.25),
    "HIGHLIGHT_BOARD_PROB": env_float("HIGHLIGHT_BOARD_PROB", 0.70),
    "ARROW_BOARD_PROB": env_float("ARROW_BOARD_PROB", 0.65),
    "TACTICAL_MARKER_PROB": env_float("TACTICAL_MARKER_PROB", 0.80),
    "HARD_EDGE_ROOK_PROB": env_float("HARD_EDGE_ROOK_PROB", 0.30),
    "HARD_FILE_EDGE_ROOK_PROB": env_float("HARD_FILE_EDGE_ROOK_PROB", 0.20),
    "SPARSE_BOARD_PROB": env_float("SPARSE_BOARD_PROB", 0.18),
    "SCREENSHOT_CLUTTER_PROB": env_float("SCREENSHOT_CLUTTER_PROB", 0.15),
    "DETECTOR_BANNER_PROB": env_float("DETECTOR_BANNER_PROB", 0.08),
    "DETECTOR_PARTIAL_BOARD_PROB": env_float("DETECTOR_PARTIAL_BOARD_PROB", 0.10),
    "DETECTOR_MONO_LOW_CONTRAST_PROB": env_float("DETECTOR_MONO_LOW_CONTRAST_PROB", 0.10),
    "DETECTOR_HEAVY_TRIM_PROB": env_float("DETECTOR_HEAVY_TRIM_PROB", 0.08),
    "MONO_STRUCTURAL_DAMAGE_PROB": env_float("MONO_STRUCTURAL_DAMAGE_PROB", 0.0),
    "MONO_EDGE_PIECE_FADE_PROB": env_float("MONO_EDGE_PIECE_FADE_PROB", 0.0),
    "PIECE_OCCLUSION_PROB": env_float("PIECE_OCCLUSION_PROB", 0.20),
    "LOCAL_PIECE_TILT_PROB": env_float("LOCAL_PIECE_TILT_PROB", 0.08),
    "LOCAL_PIECE_TILT_MAX_DEG": env_float("LOCAL_PIECE_TILT_MAX_DEG", 18.0),
    "KING_TILT_PRIORITY_PROB": env_float("KING_TILT_PRIORITY_PROB", 0.0),
    "AUG_ROTATE_PROB": env_float("AUG_ROTATE_PROB", 0.25),
    "AUG_ROTATE_MAX_DEG": env_float("AUG_ROTATE_MAX_DEG", 2.5),
    "AUG_PERSPECTIVE_PROB": env_float("AUG_PERSPECTIVE_PROB", 0.20),
    "AUG_PERSPECTIVE_SCALE": env_float("AUG_PERSPECTIVE_SCALE", 0.02),
    "MIN_PLIES": env_int("MIN_PLIES", 5),
    "MAX_PLIES": env_int("MAX_PLIES", 65),
}

PROFILE_OVERRIDES = {
    # v6_targeted_recovery_v14 recipe — uses default profile overrides
    "v6_targeted_recovery_v14": {},
    "clean": {
        "LABELS_PROB": 0.75,
        "TRIM_CAPTURE_PROB": 0.18,
        "ARTIFACT_EMPTY_TILE_PROB": 0.10,
        "HIGHLIGHT_BOARD_PROB": 0.45,
        "ARROW_BOARD_PROB": 0.40,
        "TACTICAL_MARKER_PROB": 0.50,
        "WATERMARK_BOARD_PROB": 0.22,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.28,
        "HARD_EDGE_ROOK_PROB": 0.18,
        "HARD_FILE_EDGE_ROOK_PROB": 0.12,
        "SPARSE_BOARD_PROB": 0.20,
        "SCREENSHOT_CLUTTER_PROB": 0.12,
        "DETECTOR_BANNER_PROB": 0.05,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.06,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.10,
        "DETECTOR_HEAVY_TRIM_PROB": 0.06,
        "MIN_PLIES": 8,
        "MAX_PLIES": 70,
    },
    "mono_scan": {
        "BOARD_THEME_NAMES": ["mono_paper_scan_light.png", "mono_paper_scan_mid.png", "mono_heather_print.png"],
        "PIECE_SET_NAMES": ["mono_print_scan", "mono_print_faded"],
        "LABELS_PROB": 0.48,
        "TRIM_CAPTURE_PROB": 0.24,
        "ARTIFACT_EMPTY_TILE_PROB": 0.22,
        "HIGHLIGHT_BOARD_PROB": 0.25,
        "ARROW_BOARD_PROB": 0.20,
        "TACTICAL_MARKER_PROB": 0.28,
        "WATERMARK_BOARD_PROB": 0.20,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.22,
        "HARD_EDGE_ROOK_PROB": 0.36,
        "HARD_FILE_EDGE_ROOK_PROB": 0.24,
        "SPARSE_BOARD_PROB": 0.56,
        "SCREENSHOT_CLUTTER_PROB": 0.20,
        "DETECTOR_BANNER_PROB": 0.06,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.08,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.88,
        "DETECTOR_HEAVY_TRIM_PROB": 0.10,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.35,
        "MONO_EDGE_PIECE_FADE_PROB": 0.18,
        "PIECE_OCCLUSION_PROB": 0.18,
        "LOCAL_PIECE_TILT_PROB": 0.10,
        "AUG_ROTATE_PROB": 0.22,
        "AUG_ROTATE_MAX_DEG": 2.5,
        "AUG_PERSPECTIVE_PROB": 0.16,
        "AUG_PERSPECTIVE_SCALE": 0.018,
        "MIN_PLIES": 0,
        "MAX_PLIES": 18,
    },
    "edge_frame": {
        "LABELS_PROB": 0.42,
        "TRIM_CAPTURE_PROB": 0.74,
        "ARTIFACT_EMPTY_TILE_PROB": 0.28,
        "HIGHLIGHT_BOARD_PROB": 0.32,
        "ARROW_BOARD_PROB": 0.22,
        "TACTICAL_MARKER_PROB": 0.30,
        "WATERMARK_BOARD_PROB": 0.48,
        "WATERMARK_MIN_PER_BOARD": 1,
        "WATERMARK_MAX_PER_BOARD": 3,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.78,
        "HARD_EDGE_ROOK_PROB": 0.78,
        "HARD_FILE_EDGE_ROOK_PROB": 0.70,
        "SPARSE_BOARD_PROB": 0.72,
        "SCREENSHOT_CLUTTER_PROB": 0.82,
        "DETECTOR_BANNER_PROB": 0.72,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.82,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.52,
        "DETECTOR_HEAVY_TRIM_PROB": 0.54,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.35,
        "MONO_EDGE_PIECE_FADE_PROB": 0.30,
        "PIECE_OCCLUSION_PROB": 0.36,
        "LOCAL_PIECE_TILT_PROB": 0.22,
        "LOCAL_PIECE_TILT_MAX_DEG": 22.0,
        "AUG_ROTATE_PROB": 0.40,
        "AUG_ROTATE_MAX_DEG": 5.0,
        "AUG_PERSPECTIVE_PROB": 0.38,
        "AUG_PERSPECTIVE_SCALE": 0.035,
        "MIN_PLIES": 0,
        "MAX_PLIES": 20,
    },
    "dark_anchor": {
        "BOARD_THEME_NAMES": ["grey.jpg", "olive.jpg", "wood4.jpg", "metal.jpg", "blue3.jpg"],
        "PIECE_SET_NAMES": ["cburnett", "merida", "maestro", "governor"],
        "LABELS_PROB": 0.62,
        "TRIM_CAPTURE_PROB": 0.14,
        "ARTIFACT_EMPTY_TILE_PROB": 0.10,
        "HIGHLIGHT_BOARD_PROB": 0.20,
        "ARROW_BOARD_PROB": 0.12,
        "TACTICAL_MARKER_PROB": 0.18,
        "WATERMARK_BOARD_PROB": 0.12,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.18,
        "HARD_EDGE_ROOK_PROB": 0.24,
        "HARD_FILE_EDGE_ROOK_PROB": 0.18,
        "SPARSE_BOARD_PROB": 0.28,
        "SCREENSHOT_CLUTTER_PROB": 0.14,
        "DETECTOR_BANNER_PROB": 0.04,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.05,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.16,
        "DETECTOR_HEAVY_TRIM_PROB": 0.04,
        "PIECE_OCCLUSION_PROB": 0.12,
        "LOCAL_PIECE_TILT_PROB": 0.05,
        "AUG_ROTATE_PROB": 0.14,
        "AUG_ROTATE_MAX_DEG": 2.0,
        "AUG_PERSPECTIVE_PROB": 0.10,
        "AUG_PERSPECTIVE_SCALE": 0.015,
        "MIN_PLIES": 6,
        "MAX_PLIES": 55,
    },
    "dark_anchor_clean": {
        "BOARD_THEME_NAMES": [
            "wood4.jpg",
            "dash.png",
            "glass.png",
            "stone.png",
            "walnut.png",
            "dark_wood.png",
        ],
        "PIECE_SET_NAMES": ["cburnett", "merida", "maestro", "governor"],
        "LABELS_PROB": 0.58,
        "TRIM_CAPTURE_PROB": 0.06,
        "ARTIFACT_EMPTY_TILE_PROB": 0.04,
        "HIGHLIGHT_BOARD_PROB": 0.04,
        "ARROW_BOARD_PROB": 0.02,
        "TACTICAL_MARKER_PROB": 0.06,
        "WATERMARK_BOARD_PROB": 0.01,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.01,
        "HARD_EDGE_ROOK_PROB": 0.08,
        "HARD_FILE_EDGE_ROOK_PROB": 0.06,
        "SPARSE_BOARD_PROB": 0.06,
        "SCREENSHOT_CLUTTER_PROB": 0.02,
        "DETECTOR_BANNER_PROB": 0.01,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.02,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.04,
        "DETECTOR_HEAVY_TRIM_PROB": 0.01,
        "PIECE_OCCLUSION_PROB": 0.05,
        "LOCAL_PIECE_TILT_PROB": 0.01,
        "AUG_ROTATE_PROB": 0.08,
        "AUG_ROTATE_MAX_DEG": 1.5,
        "AUG_PERSPECTIVE_PROB": 0.06,
        "AUG_PERSPECTIVE_SCALE": 0.010,
        "MIN_PLIES": 8,
        "MAX_PLIES": 70,
    },
    "wood_3d_arrow_clean": {
        "BOARD_THEME_NAMES": [
            "burled_wood.png",
            "walnut.png",
            "dark_wood.png",
            "wood3.jpg",
            "wood4.jpg",
            "maple2.jpg",
        ],
        "PIECE_SET_NAMES": [
            "vintage",
            "tournament",
            "bases",
            "club",
            "maestro",
            "governor",
            "merida",
        ],
        "LABELS_PROB": 0.78,
        "TRIM_CAPTURE_PROB": 0.05,
        "ARTIFACT_EMPTY_TILE_PROB": 0.02,
        "HIGHLIGHT_BOARD_PROB": 0.28,
        "ARROW_BOARD_PROB": 0.72,
        "TACTICAL_MARKER_PROB": 0.20,
        "WATERMARK_BOARD_PROB": 0.02,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.01,
        "HARD_EDGE_ROOK_PROB": 0.08,
        "HARD_FILE_EDGE_ROOK_PROB": 0.06,
        "SPARSE_BOARD_PROB": 0.18,
        "SCREENSHOT_CLUTTER_PROB": 0.02,
        "DETECTOR_BANNER_PROB": 0.00,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.01,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.00,
        "DETECTOR_HEAVY_TRIM_PROB": 0.00,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.00,
        "MONO_EDGE_PIECE_FADE_PROB": 0.00,
        "PIECE_OCCLUSION_PROB": 0.00,
        "LOCAL_PIECE_TILT_PROB": 0.05,
        "LOCAL_PIECE_TILT_MAX_DEG": 8.0,
        "AUG_ROTATE_PROB": 0.04,
        "AUG_ROTATE_MAX_DEG": 0.8,
        "AUG_PERSPECTIVE_PROB": 0.02,
        "AUG_PERSPECTIVE_SCALE": 0.006,
        "MIN_PLIES": 6,
        "MAX_PLIES": 50,
    },
    "dark_anchor_rook": {
        "BOARD_THEME_NAMES": ["grey.jpg", "olive.jpg", "wood4.jpg", "metal.jpg", "blue3.jpg"],
        "PIECE_SET_NAMES": ["cburnett", "merida", "maestro", "governor"],
        "LABELS_PROB": 0.72,
        "TRIM_CAPTURE_PROB": 0.06,
        "ARTIFACT_EMPTY_TILE_PROB": 0.04,
        "HIGHLIGHT_BOARD_PROB": 0.12,
        "ARROW_BOARD_PROB": 0.08,
        "TACTICAL_MARKER_PROB": 0.10,
        "WATERMARK_BOARD_PROB": 0.02,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.02,
        "HARD_EDGE_ROOK_PROB": 0.42,
        "HARD_FILE_EDGE_ROOK_PROB": 0.14,
        "SPARSE_BOARD_PROB": 0.14,
        "SCREENSHOT_CLUTTER_PROB": 0.06,
        "DETECTOR_BANNER_PROB": 0.01,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.02,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.04,
        "DETECTOR_HEAVY_TRIM_PROB": 0.01,
        "PIECE_OCCLUSION_PROB": 0.05,
        "LOCAL_PIECE_TILT_PROB": 0.02,
        "AUG_ROTATE_PROB": 0.08,
        "AUG_ROTATE_MAX_DEG": 1.5,
        "AUG_PERSPECTIVE_PROB": 0.06,
        "AUG_PERSPECTIVE_SCALE": 0.010,
        "MIN_PLIES": 6,
        "MAX_PLIES": 55,
    },
    "clutter": {
        "LABELS_PROB": 1.0,
        "TRIM_CAPTURE_PROB": 0.10,
        "ARTIFACT_EMPTY_TILE_PROB": 0.10,
        "HIGHLIGHT_BOARD_PROB": 0.08,
        "ARROW_BOARD_PROB": 0.04,
        "TACTICAL_MARKER_PROB": 0.06,
        "WATERMARK_BOARD_PROB": 1.0,
        "WATERMARK_MIN_PER_BOARD": 1,
        "WATERMARK_MAX_PER_BOARD": 4,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.92,
        "HARD_EDGE_ROOK_PROB": 0.15,
        "HARD_FILE_EDGE_ROOK_PROB": 0.25,
        "SPARSE_BOARD_PROB": 0.90,
        "SCREENSHOT_CLUTTER_PROB": 1.0,
        "DETECTOR_BANNER_PROB": 0.22,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.25,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.20,
        "DETECTOR_HEAVY_TRIM_PROB": 0.18,
        "MIN_PLIES": 0,
        "MAX_PLIES": 10,
    },
    "hard_combo": {
        "LABELS_PROB": 0.90,
        "TRIM_CAPTURE_PROB": 0.45,
        "ARTIFACT_EMPTY_TILE_PROB": 0.45,
        "HIGHLIGHT_BOARD_PROB": 0.45,
        "ARROW_BOARD_PROB": 0.35,
        "TACTICAL_MARKER_PROB": 0.40,
        "WATERMARK_BOARD_PROB": 0.80,
        "WATERMARK_MIN_PER_BOARD": 1,
        "WATERMARK_MAX_PER_BOARD": 3,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.85,
        "HARD_EDGE_ROOK_PROB": 0.75,
        "HARD_FILE_EDGE_ROOK_PROB": 0.55,
        "SPARSE_BOARD_PROB": 0.60,
        "SCREENSHOT_CLUTTER_PROB": 0.60,
        "DETECTOR_BANNER_PROB": 0.30,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.34,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.38,
        "DETECTOR_HEAVY_TRIM_PROB": 0.30,
        "MIN_PLIES": 0,
        "MAX_PLIES": 20,
    },
    "mono_print_sparse_edge": {
        "BOARD_THEME_NAMES": ["mono_paper_scan_light.png", "mono_paper_scan_mid.png", "mono_heather_print.png"],
        "PIECE_SET_NAMES": ["mono_print_scan", "mono_print_faded"],
        "LABELS_PROB": 0.35,
        "TRIM_CAPTURE_PROB": 0.18,
        "ARTIFACT_EMPTY_TILE_PROB": 0.16,
        "HIGHLIGHT_BOARD_PROB": 0.08,
        "ARROW_BOARD_PROB": 0.04,
        "TACTICAL_MARKER_PROB": 0.10,
        "WATERMARK_BOARD_PROB": 0.10,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.06,
        "HARD_EDGE_ROOK_PROB": 0.88,
        "HARD_FILE_EDGE_ROOK_PROB": 0.76,
        "SPARSE_BOARD_PROB": 0.76,
        "SCREENSHOT_CLUTTER_PROB": 0.15,
        "DETECTOR_BANNER_PROB": 0.02,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.06,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.98,
        "DETECTOR_HEAVY_TRIM_PROB": 0.04,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.72,
        "MONO_EDGE_PIECE_FADE_PROB": 0.55,
        "PIECE_OCCLUSION_PROB": 0.05,
        "LOCAL_PIECE_TILT_PROB": 0.10,
        "AUG_ROTATE_PROB": 0.14,
        "AUG_ROTATE_MAX_DEG": 1.2,
        "AUG_PERSPECTIVE_PROB": 0.10,
        "AUG_PERSPECTIVE_SCALE": 0.012,
        "MIN_PLIES": 0,
        "MAX_PLIES": 14,
    },
    "mono_print_sparse_light": {
        "BOARD_THEME_NAMES": ["mono_paper_scan_light.png", "mono_paper_scan_mid.png", "mono_heather_print.png"],
        "PIECE_SET_NAMES": ["mono_print_scan", "mono_print_faded"],
        "PRINT_STYLE_CHOICES": [("book_photo", 0.52), ("flat_book", 0.20), ("hatched_book", 0.18), ("shirt_photo", 0.10)],
        "LABELS_PROB": 0.42,
        "TRIM_CAPTURE_PROB": 0.10,
        "ARTIFACT_EMPTY_TILE_PROB": 0.10,
        "HIGHLIGHT_BOARD_PROB": 0.05,
        "ARROW_BOARD_PROB": 0.02,
        "TACTICAL_MARKER_PROB": 0.05,
        "WATERMARK_BOARD_PROB": 0.03,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.02,
        "HARD_EDGE_ROOK_PROB": 0.55,
        "HARD_FILE_EDGE_ROOK_PROB": 0.42,
        "SPARSE_BOARD_PROB": 0.62,
        "SCREENSHOT_CLUTTER_PROB": 0.08,
        "DETECTOR_BANNER_PROB": 0.01,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.03,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.92,
        "DETECTOR_HEAVY_TRIM_PROB": 0.01,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.32,
        "MONO_EDGE_PIECE_FADE_PROB": 0.18,
        "PIECE_OCCLUSION_PROB": 0.03,
        "LOCAL_PIECE_TILT_PROB": 0.04,
        "AUG_ROTATE_PROB": 0.08,
        "AUG_ROTATE_MAX_DEG": 0.8,
        "AUG_PERSPECTIVE_PROB": 0.05,
        "AUG_PERSPECTIVE_SCALE": 0.008,
        "MIN_PLIES": 0,
        "MAX_PLIES": 14,
    },
    "mono_print_edge_rook": {
        "BOARD_THEME_NAMES": ["mono_paper_scan_light.png", "mono_paper_scan_mid.png", "mono_heather_print.png"],
        "PIECE_SET_NAMES": ["mono_print_scan", "mono_print_faded"],
        "PRINT_STYLE_CHOICES": [("book_photo", 0.50), ("shirt_photo", 0.25), ("flat_book", 0.15), ("hatched_book", 0.10)],
        "LABELS_PROB": 0.38,
        "TRIM_CAPTURE_PROB": 0.10,
        "ARTIFACT_EMPTY_TILE_PROB": 0.10,
        "HIGHLIGHT_BOARD_PROB": 0.04,
        "ARROW_BOARD_PROB": 0.02,
        "TACTICAL_MARKER_PROB": 0.04,
        "WATERMARK_BOARD_PROB": 0.02,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.01,
        "HARD_EDGE_ROOK_PROB": 0.92,
        "HARD_FILE_EDGE_ROOK_PROB": 0.84,
        "SPARSE_BOARD_PROB": 0.70,
        "SCREENSHOT_CLUTTER_PROB": 0.06,
        "DETECTOR_BANNER_PROB": 0.00,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.02,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.95,
        "DETECTOR_HEAVY_TRIM_PROB": 0.01,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.38,
        "MONO_EDGE_PIECE_FADE_PROB": 0.24,
        "PIECE_OCCLUSION_PROB": 0.02,
        "LOCAL_PIECE_TILT_PROB": 0.03,
        "AUG_ROTATE_PROB": 0.07,
        "AUG_ROTATE_MAX_DEG": 0.8,
        "AUG_PERSPECTIVE_PROB": 0.04,
        "AUG_PERSPECTIVE_SCALE": 0.008,
        "MIN_PLIES": 0,
        "MAX_PLIES": 12,
    },
    "book_page_sparse": {
        "BOARD_THEME_NAMES": ["mono_paper_scan_light.png", "mono_paper_scan_mid.png"],
        "PIECE_SET_NAMES": ["mono_print_scan", "mono_print_faded"],
        "PRINT_STYLE_CHOICES": [("book_photo", 1.0)],
        "LABELS_PROB": 0.48,
        "TRIM_CAPTURE_PROB": 0.06,
        "ARTIFACT_EMPTY_TILE_PROB": 0.06,
        "HIGHLIGHT_BOARD_PROB": 0.03,
        "ARROW_BOARD_PROB": 0.01,
        "TACTICAL_MARKER_PROB": 0.03,
        "WATERMARK_BOARD_PROB": 0.01,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.01,
        "HARD_EDGE_ROOK_PROB": 0.34,
        "HARD_FILE_EDGE_ROOK_PROB": 0.22,
        "SPARSE_BOARD_PROB": 0.72,
        "SCREENSHOT_CLUTTER_PROB": 0.02,
        "DETECTOR_BANNER_PROB": 0.00,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.01,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.98,
        "DETECTOR_HEAVY_TRIM_PROB": 0.00,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.16,
        "MONO_EDGE_PIECE_FADE_PROB": 0.08,
        "PIECE_OCCLUSION_PROB": 0.01,
        "LOCAL_PIECE_TILT_PROB": 0.02,
        "AUG_ROTATE_PROB": 0.04,
        "AUG_ROTATE_MAX_DEG": 0.5,
        "AUG_PERSPECTIVE_PROB": 0.02,
        "AUG_PERSPECTIVE_SCALE": 0.005,
        "MIN_PLIES": 0,
        "MAX_PLIES": 14,
    },
    "shirt_print_sparse": {
        "BOARD_THEME_NAMES": ["mono_heather_print.png"],
        "PIECE_SET_NAMES": ["mono_print_scan", "mono_print_faded"],
        "PRINT_STYLE_CHOICES": [("shirt_photo", 1.0)],
        "LABELS_PROB": 0.44,
        "TRIM_CAPTURE_PROB": 0.06,
        "ARTIFACT_EMPTY_TILE_PROB": 0.06,
        "HIGHLIGHT_BOARD_PROB": 0.03,
        "ARROW_BOARD_PROB": 0.01,
        "TACTICAL_MARKER_PROB": 0.03,
        "WATERMARK_BOARD_PROB": 0.01,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.01,
        "HARD_EDGE_ROOK_PROB": 0.30,
        "HARD_FILE_EDGE_ROOK_PROB": 0.18,
        "SPARSE_BOARD_PROB": 0.68,
        "SCREENSHOT_CLUTTER_PROB": 0.02,
        "DETECTOR_BANNER_PROB": 0.00,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.01,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.98,
        "DETECTOR_HEAVY_TRIM_PROB": 0.00,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.18,
        "MONO_EDGE_PIECE_FADE_PROB": 0.10,
        "PIECE_OCCLUSION_PROB": 0.01,
        "LOCAL_PIECE_TILT_PROB": 0.03,
        "AUG_ROTATE_PROB": 0.04,
        "AUG_ROTATE_MAX_DEG": 0.6,
        "AUG_PERSPECTIVE_PROB": 0.02,
        "AUG_PERSPECTIVE_SCALE": 0.005,
        "MIN_PLIES": 0,
        "MAX_PLIES": 14,
    },
    "edge_rook_page": {
        "BOARD_THEME_NAMES": ["mono_paper_scan_light.png", "mono_paper_scan_mid.png"],
        "PIECE_SET_NAMES": ["mono_print_scan", "mono_print_faded"],
        "PRINT_STYLE_CHOICES": [("book_photo", 0.72), ("flat_book", 0.28)],
        "LABELS_PROB": 0.44,
        "TRIM_CAPTURE_PROB": 0.06,
        "ARTIFACT_EMPTY_TILE_PROB": 0.06,
        "HIGHLIGHT_BOARD_PROB": 0.02,
        "ARROW_BOARD_PROB": 0.01,
        "TACTICAL_MARKER_PROB": 0.03,
        "WATERMARK_BOARD_PROB": 0.01,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.01,
        "HARD_EDGE_ROOK_PROB": 0.94,
        "HARD_FILE_EDGE_ROOK_PROB": 0.84,
        "SPARSE_BOARD_PROB": 0.78,
        "SCREENSHOT_CLUTTER_PROB": 0.01,
        "DETECTOR_BANNER_PROB": 0.00,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.01,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.98,
        "DETECTOR_HEAVY_TRIM_PROB": 0.00,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.14,
        "MONO_EDGE_PIECE_FADE_PROB": 0.10,
        "PIECE_OCCLUSION_PROB": 0.01,
        "LOCAL_PIECE_TILT_PROB": 0.02,
        "AUG_ROTATE_PROB": 0.04,
        "AUG_ROTATE_MAX_DEG": 0.5,
        "AUG_PERSPECTIVE_PROB": 0.02,
        "AUG_PERSPECTIVE_SCALE": 0.005,
        "MIN_PLIES": 0,
        "MAX_PLIES": 12,
    },
    "book_page_reference": {
        "BOARD_THEME_NAMES": [
            "rag_paper_clean.png",
            "without_texture_clean.png",
            "watercolor_paper_clean.png",
            "chalk_old.png",
        ],
        "PIECE_SET_NAMES": ["mono_print_scan"],
        "PRINT_STYLE_CHOICES": [("book_reference", 0.70), ("book_photo", 0.30)],
        "LABELS_PROB": 0.00,
        "TRIM_CAPTURE_PROB": 0.00,
        "ARTIFACT_EMPTY_TILE_PROB": 0.02,
        "HIGHLIGHT_BOARD_PROB": 0.00,
        "ARROW_BOARD_PROB": 0.00,
        "TACTICAL_MARKER_PROB": 0.00,
        "WATERMARK_BOARD_PROB": 0.00,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.00,
        "HARD_EDGE_ROOK_PROB": 0.62,
        "HARD_FILE_EDGE_ROOK_PROB": 0.52,
        "SPARSE_BOARD_PROB": 0.88,
        "SCREENSHOT_CLUTTER_PROB": 0.00,
        "DETECTOR_BANNER_PROB": 0.00,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.00,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.00,
        "DETECTOR_HEAVY_TRIM_PROB": 0.00,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.00,
        "MONO_EDGE_PIECE_FADE_PROB": 0.03,
        "PIECE_OCCLUSION_PROB": 0.00,
        "LOCAL_PIECE_TILT_PROB": 0.02,
        "AUG_ROTATE_PROB": 0.03,
        "AUG_ROTATE_MAX_DEG": 0.4,
        "AUG_PERSPECTIVE_PROB": 0.01,
        "AUG_PERSPECTIVE_SCALE": 0.004,
        "MIN_PLIES": 0,
        "MAX_PLIES": 14,
    },
    "shirt_print_reference": {
        "BOARD_THEME_NAMES": [
            "rag_paper_framed.png",
            "watercolor_paper_framed.png",
            "watercolor_paper_framed_0.png",
            "rag_paper_framed.png",
            "watercolor_paper_framed.png",
            "watercolor_paper_framed_0.png",
            "watercolor_paper_clean.png",
            "watercolor_paper_clean_0.png",
            "without_texture_clean.png",
        ],
        "PIECE_SET_NAMES": ["mono_print_scan"],
        "PRINT_STYLE_CHOICES": [("shirt_reference", 0.78), ("shirt_photo", 0.22)],
        "LABELS_PROB": 0.00,
        "TRIM_CAPTURE_PROB": 0.00,
        "ARTIFACT_EMPTY_TILE_PROB": 0.02,
        "HIGHLIGHT_BOARD_PROB": 0.00,
        "ARROW_BOARD_PROB": 0.00,
        "TACTICAL_MARKER_PROB": 0.00,
        "WATERMARK_BOARD_PROB": 0.00,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.00,
        "HARD_EDGE_ROOK_PROB": 0.50,
        "HARD_FILE_EDGE_ROOK_PROB": 0.40,
        "SPARSE_BOARD_PROB": 0.90,
        "SCREENSHOT_CLUTTER_PROB": 0.00,
        "DETECTOR_BANNER_PROB": 0.00,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.00,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.00,
        "DETECTOR_HEAVY_TRIM_PROB": 0.00,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.00,
        "MONO_EDGE_PIECE_FADE_PROB": 0.06,
        "PIECE_OCCLUSION_PROB": 0.00,
        "LOCAL_PIECE_TILT_PROB": 0.08,
        "LOCAL_PIECE_TILT_MAX_DEG": 10.0,
        "AUG_ROTATE_PROB": 0.03,
        "AUG_ROTATE_MAX_DEG": 0.5,
        "AUG_PERSPECTIVE_PROB": 0.01,
        "AUG_PERSPECTIVE_SCALE": 0.004,
        "MIN_PLIES": 0,
        "MAX_PLIES": 14,
    },
    "logo_overlay": {
        "BOARD_THEME_NAMES": ["mono_paper_scan_light.png", "mono_paper_scan_mid.png", "mono_heather_print.png", "grey.jpg"],
        "LABELS_PROB": 0.55,
        "TRIM_CAPTURE_PROB": 0.16,
        "ARTIFACT_EMPTY_TILE_PROB": 0.45,
        "HIGHLIGHT_BOARD_PROB": 0.10,
        "ARROW_BOARD_PROB": 0.06,
        "TACTICAL_MARKER_PROB": 0.10,
        "WATERMARK_BOARD_PROB": 1.00,
        "WATERMARK_MIN_PER_BOARD": 1,
        "WATERMARK_MAX_PER_BOARD": 3,
        "WATERMARK_SCALE_MIN": 0.90,
        "WATERMARK_SCALE_MAX": 1.55,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.94,
        "HARD_EDGE_ROOK_PROB": 0.22,
        "HARD_FILE_EDGE_ROOK_PROB": 0.18,
        "SPARSE_BOARD_PROB": 0.55,
        "SCREENSHOT_CLUTTER_PROB": 0.10,
        "DETECTOR_BANNER_PROB": 0.08,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.12,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.34,
        "DETECTOR_HEAVY_TRIM_PROB": 0.10,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.20,
        "MONO_EDGE_PIECE_FADE_PROB": 0.12,
        "PIECE_OCCLUSION_PROB": 0.05,
        "LOCAL_PIECE_TILT_PROB": 0.20,
        "LOCAL_PIECE_TILT_MAX_DEG": 22.0,
        "AUG_ROTATE_PROB": 0.20,
        "AUG_ROTATE_MAX_DEG": 2.8,
        "AUG_PERSPECTIVE_PROB": 0.18,
        "AUG_PERSPECTIVE_SCALE": 0.018,
        "MIN_PLIES": 2,
        "MAX_PLIES": 26,
    },
    "logo_overlay_light": {
        "BOARD_THEME_NAMES": ["grey.jpg", "mono_paper_scan_light.png"],
        "LABELS_PROB": 0.70,
        "TRIM_CAPTURE_PROB": 0.05,
        "ARTIFACT_EMPTY_TILE_PROB": 0.10,
        "HIGHLIGHT_BOARD_PROB": 0.04,
        "ARROW_BOARD_PROB": 0.03,
        "TACTICAL_MARKER_PROB": 0.04,
        "WATERMARK_BOARD_PROB": 0.55,
        "WATERMARK_MIN_PER_BOARD": 1,
        "WATERMARK_MAX_PER_BOARD": 1,
        "WATERMARK_SCALE_MIN": 0.82,
        "WATERMARK_SCALE_MAX": 1.22,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.96,
        "HARD_EDGE_ROOK_PROB": 0.10,
        "HARD_FILE_EDGE_ROOK_PROB": 0.08,
        "SPARSE_BOARD_PROB": 0.34,
        "SCREENSHOT_CLUTTER_PROB": 0.04,
        "DETECTOR_BANNER_PROB": 0.02,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.02,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.18,
        "DETECTOR_HEAVY_TRIM_PROB": 0.02,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.06,
        "MONO_EDGE_PIECE_FADE_PROB": 0.02,
        "PIECE_OCCLUSION_PROB": 0.02,
        "LOCAL_PIECE_TILT_PROB": 0.06,
        "LOCAL_PIECE_TILT_MAX_DEG": 14.0,
        "AUG_ROTATE_PROB": 0.08,
        "AUG_ROTATE_MAX_DEG": 1.2,
        "AUG_PERSPECTIVE_PROB": 0.04,
        "AUG_PERSPECTIVE_SCALE": 0.008,
        "MIN_PLIES": 4,
        "MAX_PLIES": 30,
    },
    "digital_overlay_clean": {
        "BOARD_THEME_NAMES": ["green.png", "green-plastic.png", "olive.jpg"],
        "PIECE_SET_NAMES": ["cburnett", "merida", "maestro"],
        "LABELS_PROB": 0.92,
        "TRIM_CAPTURE_PROB": 0.00,
        "ARTIFACT_EMPTY_TILE_PROB": 0.00,
        "HIGHLIGHT_BOARD_PROB": 0.00,
        "ARROW_BOARD_PROB": 0.00,
        "TACTICAL_MARKER_PROB": 0.00,
        "WATERMARK_BOARD_PROB": 0.00,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.00,
        "HARD_EDGE_ROOK_PROB": 0.18,
        "HARD_FILE_EDGE_ROOK_PROB": 0.10,
        "SPARSE_BOARD_PROB": 0.42,
        "SCREENSHOT_CLUTTER_PROB": 0.00,
        "DETECTOR_BANNER_PROB": 0.00,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.00,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.00,
        "DETECTOR_HEAVY_TRIM_PROB": 0.00,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.00,
        "MONO_EDGE_PIECE_FADE_PROB": 0.00,
        "PIECE_OCCLUSION_PROB": 0.00,
        "LOCAL_PIECE_TILT_PROB": 0.06,
        "LOCAL_PIECE_TILT_MAX_DEG": 10.0,
        "AUG_ROTATE_PROB": 0.02,
        "AUG_ROTATE_MAX_DEG": 0.5,
        "AUG_PERSPECTIVE_PROB": 0.01,
        "AUG_PERSPECTIVE_SCALE": 0.004,
        "MIN_PLIES": 2,
        "MAX_PLIES": 28,
        "FORCE_LABEL_POV": "white",
    },
    "tilt_anchor": {
        "BOARD_THEME_NAMES": ["grey.jpg", "olive.jpg", "blue2.jpg", "dash.png", "glass.png", "stone.png", "walnut.png"],
        "PIECE_SET_NAMES": ["cburnett", "merida", "maestro"],
        "LABELS_PROB": 0.52,
        "TRIM_CAPTURE_PROB": 0.10,
        "ARTIFACT_EMPTY_TILE_PROB": 0.12,
        "HIGHLIGHT_BOARD_PROB": 0.12,
        "ARROW_BOARD_PROB": 0.08,
        "TACTICAL_MARKER_PROB": 0.12,
        "WATERMARK_BOARD_PROB": 0.10,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.08,
        "HARD_EDGE_ROOK_PROB": 0.26,
        "HARD_FILE_EDGE_ROOK_PROB": 0.18,
        "SPARSE_BOARD_PROB": 0.34,
        "SCREENSHOT_CLUTTER_PROB": 0.10,
        "DETECTOR_BANNER_PROB": 0.03,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.04,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.08,
        "DETECTOR_HEAVY_TRIM_PROB": 0.03,
        "PIECE_OCCLUSION_PROB": 0.06,
        "LOCAL_PIECE_TILT_PROB": 0.48,
        "LOCAL_PIECE_TILT_MAX_DEG": 26.0,
        "KING_TILT_PRIORITY_PROB": 0.72,
        "AUG_ROTATE_PROB": 0.18,
        "AUG_ROTATE_MAX_DEG": 2.8,
        "AUG_PERSPECTIVE_PROB": 0.12,
        "AUG_PERSPECTIVE_SCALE": 0.016,
        "MIN_PLIES": 4,
        "MAX_PLIES": 42,
    },
    "broadcast_dark_sparse": {
        "BOARD_THEME_NAMES": ["dash.png", "glass.png", "marble2.png", "stone.png", "walnut.png", "wood4.jpg"],
        "PIECE_SET_NAMES": ["cburnett", "maestro", "merida"],
        "LABELS_PROB": 0.00,
        "TRIM_CAPTURE_PROB": 0.00,
        "ARTIFACT_EMPTY_TILE_PROB": 0.00,
        "HIGHLIGHT_BOARD_PROB": 0.42,
        "ARROW_BOARD_PROB": 0.00,
        "TACTICAL_MARKER_PROB": 0.00,
        "WATERMARK_BOARD_PROB": 0.00,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.00,
        "HARD_EDGE_ROOK_PROB": 0.28,
        "HARD_FILE_EDGE_ROOK_PROB": 0.20,
        "SPARSE_BOARD_PROB": 0.92,
        "SCREENSHOT_CLUTTER_PROB": 0.16,
        "DETECTOR_BANNER_PROB": 0.10,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.00,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.00,
        "DETECTOR_HEAVY_TRIM_PROB": 0.00,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.00,
        "MONO_EDGE_PIECE_FADE_PROB": 0.02,
        "PIECE_OCCLUSION_PROB": 0.00,
        "LOCAL_PIECE_TILT_PROB": 0.68,
        "LOCAL_PIECE_TILT_MAX_DEG": 24.0,
        "KING_TILT_PRIORITY_PROB": 0.92,
        "AUG_ROTATE_PROB": 0.06,
        "AUG_ROTATE_MAX_DEG": 0.8,
        "AUG_PERSPECTIVE_PROB": 0.02,
        "AUG_PERSPECTIVE_SCALE": 0.006,
        "MIN_PLIES": 0,
        "MAX_PLIES": 12,
    },
    "diagtransfer_hatched": {
        "BOARD_THEME_NAMES": [
            "rag_paper_clean.png",
            "without_texture_clean.png",
            "watercolor_paper_clean.png",
            "chalk_old.png",
        ],
        "PIECE_SET_NAMES": ["mono_print_scan", "mono_print_faded"],
        "PRINT_STYLE_CHOICES": [("hatched_book", 0.76), ("book_photo", 0.18), ("flat_book", 0.06)],
        "PRINT_PIECE_SCALE": 0.78,
        "LABELS_PROB": 0.00,
        "TRIM_CAPTURE_PROB": 0.18,
        "ARTIFACT_EMPTY_TILE_PROB": 0.12,
        "HIGHLIGHT_BOARD_PROB": 0.12,
        "ARROW_BOARD_PROB": 0.08,
        "TACTICAL_MARKER_PROB": 0.15,
        "WATERMARK_BOARD_PROB": 0.12,
        "WATERMARK_FULL_KING_WORDMARK_PROB": 0.18,
        "HARD_EDGE_ROOK_PROB": 0.50,
        "HARD_FILE_EDGE_ROOK_PROB": 0.40,
        "SPARSE_BOARD_PROB": 0.40,
        "SCREENSHOT_CLUTTER_PROB": 0.06,
        "DETECTOR_BANNER_PROB": 0.04,
        "DETECTOR_PARTIAL_BOARD_PROB": 0.05,
        "DETECTOR_MONO_LOW_CONTRAST_PROB": 0.88,
        "DETECTOR_HEAVY_TRIM_PROB": 0.06,
        "MONO_STRUCTURAL_DAMAGE_PROB": 0.12,
        "MONO_EDGE_PIECE_FADE_PROB": 0.18,
        "PIECE_OCCLUSION_PROB": 0.10,
        "LOCAL_PIECE_TILT_PROB": 0.06,
        "AUG_ROTATE_PROB": 0.10,
        "AUG_ROTATE_MAX_DEG": 1.2,
        "AUG_PERSPECTIVE_PROB": 0.06,
        "AUG_PERSPECTIVE_SCALE": 0.008,
        "MIN_PLIES": 0,
        "MAX_PLIES": 22,
    },
}

# Deterministic data recipe (not ad-hoc random drift):
# fixed per-chunk quotas that are auditable and repeatable.
RECIPE_NAME = os.getenv("RECIPE_NAME", "v6_targeted_recovery_v13")
PROFILE_RECIPES = {
    "v6_targeted_v1": [
        ("clean", 0.30),
        ("mono_scan", 0.30),
        ("edge_frame", 0.20),
        ("clutter", 0.15),
        ("hard_combo", 0.05),
    ],
    "v6_balanced_v1": [
        ("clean", 0.45),
        ("mono_scan", 0.22),
        ("edge_frame", 0.16),
        ("clutter", 0.12),
        ("hard_combo", 0.05),
    ],
    "v6_00028_recovery_v2": [
        ("clean", 0.20),
        ("mono_scan", 0.20),
        ("mono_print_sparse_edge", 0.45),
        ("edge_frame", 0.10),
        ("hard_combo", 0.05),
    ],
    "v6_mono_logo_recovery_v1": [
        ("mono_print_sparse_edge", 0.42),
        ("mono_scan", 0.24),
        ("logo_overlay", 0.14),
        ("edge_frame", 0.10),
        ("clean", 0.07),
        ("hard_combo", 0.03),
    ],
    "v6_mono_logo_recovery_v2": [
        ("clean", 0.24),
        ("edge_frame", 0.18),
        ("mono_print_sparse_edge", 0.28),
        ("mono_scan", 0.16),
        ("logo_overlay", 0.10),
        ("hard_combo", 0.04),
    ],
    "v6_mono_logo_recovery_v3": [
        ("clean", 0.24),
        ("dark_anchor", 0.20),
        ("mono_print_sparse_edge", 0.18),
        ("mono_scan", 0.12),
        ("edge_frame", 0.12),
        ("logo_overlay", 0.07),
        ("tilt_anchor", 0.05),
        ("hard_combo", 0.02),
    ],
    "v6_mono_logo_recovery_v4": [
        ("clean", 0.30),
        ("dark_anchor_clean", 0.30),
        ("mono_print_sparse_light", 0.18),
        ("mono_print_edge_rook", 0.12),
        ("mono_scan", 0.06),
        ("tilt_anchor", 0.03),
        ("logo_overlay", 0.01),
    ],
    "v6_mono_logo_recovery_v5": [
        ("clean", 0.24),
        ("dark_anchor_clean", 0.22),
        ("dark_anchor_rook", 0.14),
        ("mono_print_sparse_light", 0.20),
        ("mono_print_edge_rook", 0.16),
        ("mono_scan", 0.03),
        ("tilt_anchor", 0.01),
    ],
    "v6_mono_logo_recovery_v6": [
        ("clean", 0.30),
        ("dark_anchor_clean", 0.24),
        ("book_page_sparse", 0.16),
        ("shirt_print_sparse", 0.10),
        ("edge_rook_page", 0.08),
        ("tilt_anchor", 0.06),
        ("logo_overlay_light", 0.04),
        ("dark_anchor_rook", 0.02),
    ],
    "v6_mono_logo_recovery_v7": [
        ("clean", 0.18),
        ("dark_anchor_clean", 0.18),
        ("book_page_reference", 0.18),
        ("shirt_print_reference", 0.14),
        ("digital_overlay_clean", 0.12),
        ("broadcast_dark_sparse", 0.12),
        ("edge_rook_page", 0.05),
        ("tilt_anchor", 0.03),
    ],
    "v6_targeted_recovery_v8": [
        ("diagtransfer_hatched", 0.24),
        ("book_page_reference", 0.18),
        ("shirt_print_reference", 0.18),
        ("broadcast_dark_sparse", 0.16),
        ("dark_anchor_clean", 0.08),
        ("clean", 0.08),
        ("digital_overlay_clean", 0.04),
        ("edge_rook_page", 0.02),
        ("tilt_anchor", 0.02),
    ],
    "v6_targeted_recovery_v9": [
        ("diagtransfer_hatched", 0.18),
        ("book_page_reference", 0.12),
        ("shirt_print_reference", 0.18),
        ("broadcast_dark_sparse", 0.16),
        ("dark_anchor_clean", 0.12),
        ("clean", 0.14),
        ("digital_overlay_clean", 0.06),
        ("edge_rook_page", 0.02),
        ("tilt_anchor", 0.02),
    ],
    "v6_targeted_recovery_v10": [
        ("wood_3d_arrow_clean", 0.18),
        ("shirt_print_reference", 0.20),
        ("broadcast_dark_sparse", 0.18),
        ("clean", 0.14),
        ("dark_anchor_clean", 0.12),
        ("diagtransfer_hatched", 0.08),
        ("book_page_reference", 0.04),
        ("digital_overlay_clean", 0.02),
        ("tilt_anchor", 0.04),
    ],
    "v6_targeted_recovery_v11": [
        ("wood_3d_arrow_clean", 0.08),
        ("shirt_print_reference", 0.18),
        ("broadcast_dark_sparse", 0.16),
        ("clean", 0.20),
        ("dark_anchor_clean", 0.20),
        ("diagtransfer_hatched", 0.06),
        ("book_page_reference", 0.04),
        ("digital_overlay_clean", 0.04),
        ("tilt_anchor", 0.04),
    ],
    "v6_targeted_recovery_v12": [
        ("wood_3d_arrow_clean", 0.04),
        ("shirt_print_reference", 0.20),
        ("broadcast_dark_sparse", 0.08),
        ("clean", 0.20),
        ("dark_anchor_clean", 0.32),
        ("diagtransfer_hatched", 0.04),
        ("book_page_reference", 0.02),
        ("digital_overlay_clean", 0.04),
        ("tilt_anchor", 0.06),
    ],
    "v6_targeted_recovery_v13": [
        ("dark_anchor_clean", 0.50),
        ("broadcast_dark_sparse", 0.14),
        ("tilt_anchor", 0.10),
        ("shirt_print_reference", 0.12),
        ("clean", 0.06),
        ("wood_3d_arrow_clean", 0.04),
        ("diagtransfer_hatched", 0.02),
        ("digital_overlay_clean", 0.02),
    ],
    "v6_targeted_recovery_v14": [
        # Balanced render profiles — covers ALL stress suite failure conditions
        ("clean", 0.22),              # UP from 0.06 — 25% fail rate on clean profiles
        ("dark_anchor_clean", 0.20),  # DOWN from 0.50 — was over-represented
        ("shirt_print_reference", 0.16),
        ("broadcast_dark_sparse", 0.12),
        ("wood_3d_arrow_clean", 0.10), # UP from 0.04 — 22% fail rate
        ("digital_overlay_clean", 0.06),
        ("book_page_reference", 0.06),
        ("diagtransfer_hatched", 0.04),
        ("tilt_anchor", 0.04),
    ],
}
DEFAULT_PROFILE_WEIGHTS = PROFILE_RECIPES.get(RECIPE_NAME, PROFILE_RECIPES["v6_mono_logo_recovery_v6"])


def get_profile_config(profile):
    if profile not in PROFILE_OVERRIDES:
        raise ValueError(f"Unknown generation profile: {profile}")
    cfg = dict(BASE_CONFIG)
    cfg.update(PROFILE_OVERRIDES[profile])
    return cfg


def choose_profile(profile=None):
    if profile:
        return profile
    names = [name for name, _ in DEFAULT_PROFILE_WEIGHTS]
    weights = [weight for _, weight in DEFAULT_PROFILE_WEIGHTS]
    return random.choices(names, weights=weights, k=1)[0]


def build_profile_plan(total_boards, profile_weights):
    """
    Build deterministic per-chunk profile plan from recipe weights.
    Uses largest-remainder rounding then shuffles plan order.
    """
    raw_counts = [(name, weight * total_boards) for name, weight in profile_weights]
    floored = [(name, int(val)) for name, val in raw_counts]
    used = sum(v for _, v in floored)
    remaining = total_boards - used

    remainders = sorted(
        ((name, raw - int(raw)) for name, raw in raw_counts),
        key=lambda x: x[1],
        reverse=True,
    )
    counts = {name: count for name, count in floored}
    for idx in range(max(0, remaining)):
        counts[remainders[idx % len(remainders)][0]] += 1

    plan = []
    for name, _weight in profile_weights:
        plan.extend([name] * counts[name])
    random.shuffle(plan)
    return plan, counts


def atomic_torch_save(payload, path):
    tmp_path = f"{path}.tmp"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def add_detector_banner_overlay(img, cfg):
    """Overlay social/banner UI blocks to stress board localization."""
    if random.random() >= cfg["DETECTOR_BANNER_PROB"]:
        return img

    draw = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    side = random.choice(("left", "right", "top", "bottom"))
    fill = random.choice([(18, 32, 82, 230), (28, 28, 28, 220), (36, 56, 74, 220)])
    line = random.choice([(255, 255, 255, 30), (190, 220, 255, 40)])

    if side in ("left", "right"):
        bw = int(w * random.uniform(0.18, 0.42))
        x0 = 0 if side == "left" else w - bw
        x1 = x0 + bw
        draw.rectangle([x0, 0, x1, h], fill=fill)
        for _ in range(18):
            y = random.randint(0, h)
            draw.line([x0, y, x1, y], fill=line, width=1)
        tx, ty = x0 + 14, random.randint(22, max(24, h // 3))
    else:
        bh = int(h * random.uniform(0.12, 0.30))
        y0 = 0 if side == "top" else h - bh
        y1 = y0 + bh
        draw.rectangle([0, y0, w, y1], fill=fill)
        for _ in range(12):
            x = random.randint(0, w)
            draw.line([x, y0, x, y1], fill=line, width=1)
        tx, ty = random.randint(18, max(20, w // 3)), y0 + 10

    try:
        font_big = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", random.randint(24, 54))
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", random.randint(12, 20))
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((tx, ty), random.choice(("Chess", "Puzzle", "Opening")), fill=(245, 245, 245, 235), font=font_big)
    draw.text((tx, ty + random.randint(42, 68)), random.choice(("Strategy", "Trainer", "Tactics")), fill=(220, 220, 220, 210), font=font_small)
    return img


def compose_partial_board_scene(img, cfg):
    """Place the board in a subregion (50-80%) to emulate banner/screenshot layouts."""
    if random.random() >= cfg["DETECTOR_PARTIAL_BOARD_PROB"]:
        return img

    w, h = img.size
    canvas = Image.new("RGB", (w, h), random.choice([(22, 26, 32), (30, 44, 61), (245, 245, 245)]))
    cdraw = ImageDraw.Draw(canvas, "RGBA")

    if random.random() < 0.72:
        board_w = int(w * random.uniform(0.52, 0.78))
        board = img.resize((board_w, h), Image.LANCZOS)
        x = w - board_w if random.random() < 0.70 else 0
        canvas.paste(board, (x, 0))
    else:
        board_h = int(h * random.uniform(0.52, 0.78))
        board = img.resize((w, board_h), Image.LANCZOS)
        y = h - board_h if random.random() < 0.70 else 0
        canvas.paste(board, (0, y))

    for _ in range(26):
        y = random.randint(0, h)
        cdraw.line([0, y, w, y], fill=(255, 255, 255, random.randint(8, 24)), width=1)

    return canvas


def build_soft_noise_map(width, height, low_res_side=24, low=0.78, high=1.18):
    small_h = max(4, height // low_res_side)
    small_w = max(4, width // low_res_side)
    noise = np.random.uniform(low, high, (small_h, small_w)).astype(np.float32)
    noise_img = Image.fromarray(np.clip(noise * 255.0, 0, 255).astype(np.uint8)).convert("L")
    noise_img = noise_img.resize((width, height), Image.BICUBIC)
    noise_img = noise_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.6, 1.4)))
    return np.asarray(noise_img, dtype=np.float32) / 255.0


def estimate_tile_border_level(tile_arr):
    border = np.concatenate(
        [
            tile_arr[0, :],
            tile_arr[-1, :],
            tile_arr[1:-1, 0],
            tile_arr[1:-1, -1],
        ]
    )
    return float(np.median(border))


def build_piece_damage_map(tile_size, edge_square=False):
    low_res = max(4, tile_size // 10)
    noise = np.random.uniform(0.55, 1.0, (low_res, low_res)).astype(np.float32)
    holes = random.randint(1, 4 if edge_square else 2)
    for _ in range(holes):
        rr = random.randrange(low_res)
        cc = random.randrange(low_res)
        r1 = max(0, rr - random.randint(0, 1))
        r2 = min(low_res, rr + random.randint(1, 2))
        c1 = max(0, cc - random.randint(0, 1))
        c2 = min(low_res, cc + random.randint(1, 2))
        noise[r1:r2, c1:c2] *= random.uniform(0.05, 0.45 if edge_square else 0.65)
    noise_img = Image.fromarray(np.clip(noise * 255.0, 0, 255).astype(np.uint8)).convert("L")
    noise_img = noise_img.resize((tile_size, tile_size), Image.BICUBIC)
    return np.asarray(noise_img, dtype=np.float32) / 255.0


def build_edge_piece_fade_mask(tile_size, row_idx, col_idx):
    xs = np.linspace(0.0, 1.0, tile_size, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, tile_size, dtype=np.float32)
    mask = np.ones((tile_size, tile_size), dtype=np.float32)

    if col_idx == 0:
        mask *= np.tile(np.clip(xs * 1.4, 0.20, 1.0), (tile_size, 1))
    elif col_idx == 7:
        mask *= np.tile(np.clip(xs[::-1] * 1.4, 0.20, 1.0), (tile_size, 1))

    if row_idx == 0:
        mask *= np.tile(np.clip(ys[:, None] * 1.4, 0.25, 1.0), (1, tile_size))
    elif row_idx == 7:
        mask *= np.tile(np.clip(ys[::-1, None] * 1.4, 0.25, 1.0), (1, tile_size))

    return mask


def apply_mono_structural_damage(src_img, mono_img, grid, cfg):
    src = np.asarray(src_img.convert("L"), dtype=np.float32)
    base = np.asarray(mono_img.convert("L"), dtype=np.float32)
    h, w = base.shape
    tile_size = w // 8

    paper = build_soft_noise_map(w, h, low_res_side=18, low=0.86, high=1.14)
    fibers_x = build_soft_noise_map(w, h, low_res_side=64, low=0.95, high=1.05)
    fibers_y = build_soft_noise_map(h, w, low_res_side=64, low=0.95, high=1.05).T

    out = np.clip((base - 128.0) * random.uniform(0.82, 0.92) + 128.0, 0, 255)
    out = np.clip(out * paper + (fibers_x - 1.0) * 12.0 + (fibers_y - 1.0) * 10.0, 0, 255)

    for row_idx in range(8):
        for col_idx in range(8):
            if not grid[row_idx][col_idx]:
                continue

            y0 = row_idx * tile_size
            y1 = y0 + tile_size
            x0 = col_idx * tile_size
            x1 = x0 + tile_size

            src_tile = src[y0:y1, x0:x1]
            out_tile = out[y0:y1, x0:x1]
            bg_level = estimate_tile_border_level(src_tile)
            delta = src_tile - bg_level
            if float(np.max(np.abs(delta))) < 10.0:
                continue

            edge_square = row_idx in (0, 7) or col_idx in (0, 7)
            contrast_scale = random.uniform(0.32, 0.58)
            if edge_square:
                contrast_scale *= random.uniform(0.55, 0.80)
            delta = delta * contrast_scale

            damage = build_piece_damage_map(tile_size, edge_square=edge_square)
            delta *= damage

            if edge_square and random.random() < cfg["MONO_EDGE_PIECE_FADE_PROB"]:
                delta *= build_edge_piece_fade_mask(tile_size, row_idx, col_idx)

            if random.random() < 0.65:
                cutoff = random.uniform(6.0, 16.0)
                delta = np.where(np.abs(delta) < cutoff, 0.0, delta)

            out[y0:y1, x0:x1] = np.clip(out_tile + delta, 0, 255)

    out_img = Image.fromarray(out.astype(np.uint8)).convert("L")
    if random.random() < 0.55:
        out_img = out_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.25, 0.9)))
    if random.random() < 0.40:
        out_img = ImageEnhance.Contrast(out_img).enhance(random.uniform(0.86, 0.96))
    return out_img.convert("RGB")


def apply_mono_book_style(img, grid, cfg):
    """Convert to low-saturation, low-contrast style seen in scans/books."""
    if random.random() >= cfg["DETECTOR_MONO_LOW_CONTRAST_PROB"]:
        return img

    gray = img.convert("L")
    gray = ImageEnhance.Contrast(gray).enhance(random.uniform(0.62, 0.92))
    gray = ImageEnhance.Brightness(gray).enhance(random.uniform(0.85, 1.10))
    out = gray.convert("RGB")
    if random.random() < cfg["MONO_STRUCTURAL_DAMAGE_PROB"]:
        out = apply_mono_structural_damage(img, out, grid, cfg)
    if random.random() < 0.45:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 1.0)))
    return out


def augment_image(img, cfg):
    """Realistic augmentation mix for robust tile classification."""
    import cv2

    # 1. Optional grayscale conversion (keep low so color cues remain primary).
    if random.random() < 0.15:
        img = img.convert('L').convert('RGB')

    # 2. Micro-rotations.
    if random.random() < cfg["AUG_ROTATE_PROB"]:
        angle = random.uniform(-cfg["AUG_ROTATE_MAX_DEG"], cfg["AUG_ROTATE_MAX_DEG"])
        img = img.rotate(angle, fillcolor=(128, 128, 128))

    # 3. Mild perspective jitter to simulate camera skew.
    if random.random() < cfg["AUG_PERSPECTIVE_PROB"]:
        arr = np.array(img)
        h, w = arr.shape[:2]
        max_j = max(2, int(min(h, w) * cfg["AUG_PERSPECTIVE_SCALE"]))
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = src + np.float32([
            [random.randint(-max_j, max_j), random.randint(-max_j, max_j)],
            [random.randint(-max_j, max_j), random.randint(-max_j, max_j)],
            [random.randint(-max_j, max_j), random.randint(-max_j, max_j)],
            [random.randint(-max_j, max_j), random.randint(-max_j, max_j)],
        ])
        mat = cv2.getPerspectiveTransform(src, dst)
        arr = cv2.warpPerspective(arr, mat, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        img = Image.fromarray(arr)

    # 4. CLAHE (randomized settings), not always-on.
    if random.random() < 0.50:
        img_np = np.array(img)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clip = random.uniform(1.6, 2.4)
        grid = random.choice([(6, 6), (8, 8), (10, 10)])
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        img = Image.fromarray(img_np)

    # 5. Moderate brightness/contrast/saturation jitter.
    if random.random() < 0.55:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.78, 1.22))
    if random.random() < 0.55:
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.80, 1.25))
    if random.random() < 0.25:
        img = ImageEnhance.Color(img).enhance(random.uniform(0.90, 1.10))

    # 6. Mild sharpening, occasional.
    if random.random() < 0.20:
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))

    # 7. Slight Gaussian noise.
    if random.random() < 0.15:
        arr = np.array(img)
        noise = np.random.normal(0, random.uniform(3.0, 10.0), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # 8. Slight blur for compression/scan artifacts.
    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.9)))

    return img


def apply_piece_occlusion_overlay(img, grid, cfg):
    """Apply strong square/arrow overlays on occupied squares (0118-like stress)."""
    if random.random() >= cfg["PIECE_OCCLUSION_PROB"]:
        return img

    draw = ImageDraw.Draw(img, "RGBA")
    ts = img.size[0] // 8
    occupied = [(r, c) for r in range(8) for c in range(8) if grid[r][c]]
    if not occupied:
        return img

    target_count = random.randint(1, 2)
    random.shuffle(occupied)
    for r, c in occupied[:target_count]:
        x0, y0 = c * ts, r * ts
        x1, y1 = x0 + ts, y0 + ts
        fill = random.choice([(228, 36, 36, 150), (221, 58, 48, 140), (36, 170, 80, 130)])
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(250, 250, 250, 80), width=1)

        if random.random() < 0.7:
            sr, sc = random.randint(0, 7), random.randint(0, 7)
            sx, sy = sc * ts + ts // 2, sr * ts + ts // 2
            tx, ty = c * ts + ts // 2, r * ts + ts // 2
            draw_arrow(draw, (sr, sc), (r, c), (20, 210, 20, 170), ts)

    return img


def apply_local_piece_tilt(img, grid, cfg):
    """Locally rotate piece tiles to mimic tilted/warped glyph rendering."""
    if random.random() >= cfg["LOCAL_PIECE_TILT_PROB"]:
        return img

    ts = img.size[0] // 8
    occupied = [(r, c) for r in range(8) for c in range(8) if grid[r][c]]
    if not occupied:
        return img

    out = img.copy()
    chosen = []
    king_priority = float(cfg.get("KING_TILT_PRIORITY_PROB", 0.0))
    king_tiles = [(r, c) for r, c in occupied if grid[r][c] in {"K", "k"}]
    if king_tiles and random.random() < king_priority:
        chosen.append(random.choice(king_tiles))

    remaining = [sq for sq in occupied if sq not in chosen]
    sample_count = min(len(remaining), random.randint(0 if chosen else 1, 1 if chosen else 2))
    if sample_count > 0:
        chosen.extend(random.sample(remaining, k=sample_count))

    for r, c in chosen:
        x0, y0 = c * ts, r * ts
        tile = out.crop((x0, y0, x0 + ts, y0 + ts))
        arr = np.array(tile)
        fill = tuple(int(v) for v in np.median(arr.reshape(-1, 3), axis=0))
        max_deg = max(1.0, float(cfg.get("LOCAL_PIECE_TILT_MAX_DEG", 18.0)))
        angle = random.uniform(-max_deg, max_deg)
        tilted = tile.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=fill)
        out.paste(tilted, (x0, y0))
    return out


def apply_digital_overlay_clean(background, grid):
    draw = ImageDraw.Draw(background, "RGBA")
    ts = background.size[0] // 8
    occupied = [(r, c) for r in range(8) for c in range(8) if grid[r][c]]
    empty = [(r, c) for r in range(8) for c in range(8) if not grid[r][c]]

    if random.random() < 0.95:
        for _ in range(random.randint(1, 2)):
            r, c = random.randint(0, 7), random.randint(0, 7)
            color = random.choice([(255, 212, 74, 92), (79, 191, 255, 88), (120, 232, 145, 80)])
            draw.rectangle([c * ts, r * ts, (c + 1) * ts, (r + 1) * ts], fill=color)

    if occupied and random.random() < 0.92:
        start_r, start_c = random.choice(occupied)
        end_r = max(0, min(7, start_r + random.randint(-3, 3)))
        end_c = max(0, min(7, start_c + random.randint(-3, 3)))
        if (start_r, start_c) != (end_r, end_c):
            color = random.choice([(39, 92, 255, 150), (28, 170, 94, 150), (36, 36, 36, 125)])
            draw_arrow(draw, (start_r, start_c), (end_r, end_c), color, ts)

    if empty:
        badge_r, badge_c = random.choice([sq for sq in empty if sq[0] >= 6] or empty)
        x0 = badge_c * ts + int(ts * 0.08)
        y0 = badge_r * ts + int(ts * 0.60)
        x1 = x0 + int(ts * 0.40)
        y1 = y0 + int(ts * 0.22)
        draw.rounded_rectangle([x0, y0, x1, y1], radius=5, fill=(240, 240, 240, 210), outline=(60, 60, 60, 90), width=1)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except Exception:
            font = ImageFont.load_default()
        draw.text((x0 + 4, y0 + 1), random.choice(("SCC", "ICC", "CLC")), fill=(34, 34, 34, 220), font=font)

    return background


def apply_broadcast_dark_style(background):
    import cv2

    arr = np.asarray(background, dtype=np.float32)
    hsv = cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(0.18, 0.42)
    hsv[:, :, 2] *= random.uniform(0.78, 0.92)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    out_img = Image.fromarray(out)
    out_img = ImageEnhance.Contrast(out_img).enhance(random.uniform(1.02, 1.12))
    if random.random() < 0.65:
        out_img = out_img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=115, threshold=2))
    return out_img


def apply_dark_anchor_grade(background):
    import cv2

    arr = np.asarray(background, dtype=np.float32)
    hsv = cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

    # Two real-anchor moods:
    # - cooler blue-grey dark boards like 00005
    # - warmer brown dark boards like 00017
    mode = random.choice(("cool_00005", "warm_00017"))
    hsv[:, :, 1] *= random.uniform(0.40, 0.70)
    hsv[:, :, 2] *= random.uniform(0.76, 0.88)
    rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)

    if mode == "cool_00005":
        rgb[:, :, 2] *= random.uniform(1.05, 1.12)
        rgb[:, :, 1] *= random.uniform(0.98, 1.04)
        rgb[:, :, 0] *= random.uniform(0.92, 0.98)
    else:
        rgb[:, :, 0] *= random.uniform(1.04, 1.12)
        rgb[:, :, 1] *= random.uniform(0.95, 1.00)
        rgb[:, :, 2] *= random.uniform(0.84, 0.92)

    out_img = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    out_img = ImageEnhance.Contrast(out_img).enhance(random.uniform(1.00, 1.08))
    out_img = ImageEnhance.Brightness(out_img).enhance(random.uniform(0.94, 0.99))
    if random.random() < 0.55:
        out_img = out_img.filter(ImageFilter.UnsharpMask(radius=0.6, percent=105, threshold=2))
    return out_img

def draw_arrow(draw, start_square, end_square, color, ts):
    """Draw a proper chess arrow with arrowhead"""
    # Calculate center points
    start_x = start_square[1] * ts + ts // 2
    start_y = start_square[0] * ts + ts // 2
    end_x = end_square[1] * ts + ts // 2
    end_y = end_square[0] * ts + ts // 2
    
    # Draw thick line
    draw.line([start_x, start_y, end_x, end_y], fill=color, width=random.randint(8, 15))
    
    # Draw arrowhead (simple triangle)
    import math
    angle = math.atan2(end_y - start_y, end_x - start_x)
    arrow_size = 20
    
    # Three points of triangle
    p1 = (end_x, end_y)
    p2 = (end_x - arrow_size * math.cos(angle - math.pi/6),
          end_y - arrow_size * math.sin(angle - math.pi/6))
    p3 = (end_x - arrow_size * math.cos(angle + math.pi/6),
          end_y - arrow_size * math.sin(angle + math.pi/6))
    
    draw.polygon([p1, p2, p3], fill=color)


def draw_annotation_marker(draw, square_rc, ts):
    """Draw tactical annotation marker: small ring, !, or !! near tile corners."""
    r, c = square_rc
    x0, y0 = c * ts, r * ts

    # Corner-biased placement, often overlapping piece/tile boundary like puzzle UIs.
    corner = random.choice(("tr", "tl", "br", "bl"))
    offset = max(6, ts // 8)
    jitter = 4
    if corner == "tr":
        center_x = x0 + ts - offset + random.randint(-jitter, jitter)
        center_y = y0 + offset + random.randint(-jitter, jitter)
    elif corner == "tl":
        center_x = x0 + offset + random.randint(-jitter, jitter)
        center_y = y0 + offset + random.randint(-jitter, jitter)
    elif corner == "br":
        center_x = x0 + ts - offset + random.randint(-jitter, jitter)
        center_y = y0 + ts - offset + random.randint(-jitter, jitter)
    else:
        center_x = x0 + offset + random.randint(-jitter, jitter)
        center_y = y0 + ts - offset + random.randint(-jitter, jitter)

    radius = random.randint(9, 12)
    marker = random.choices(["ring", "!", "!!"], weights=[0.20, 0.45, 0.35], k=1)[0]

    if marker == "ring":
        draw.ellipse(
            [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            outline=(255, 0, 0, random.randint(140, 200)),
            width=3)
        return

    # Chess.com-style brilliant marker: turquoise round badge + thick white exclamation bars.
    badge = random.choice([(37, 176, 176, 230), (29, 166, 167, 230), (42, 183, 184, 225)])
    draw.ellipse(
        [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
        fill=badge,
        outline=(18, 120, 120, 170),
        width=1)

    # Subtle inner highlight for glossy look.
    inner_r = max(3, radius - 3)
    draw.ellipse(
        [center_x - inner_r, center_y - inner_r, center_x + inner_r, center_y + inner_r],
        outline=(180, 245, 245, 60),
        width=1)

    count = 2 if marker == "!!" else 1
    bar_h = max(6, int(radius * 0.95))
    bar_w = max(2, int(radius * 0.32))
    dot_h = max(2, int(radius * 0.28))
    gap = max(2, int(bar_w * 0.6))
    total_w = count * bar_w + (count - 1) * gap
    start_x = center_x - total_w // 2
    top_y = center_y - int(radius * 0.52)

    for idx in range(count):
        x0 = int(start_x + idx * (bar_w + gap))
        x1 = int(x0 + bar_w)
        y0 = int(top_y)
        y1 = int(y0 + bar_h)
        # Tiny shadow
        draw.rounded_rectangle([x0 + 1, y0 + 1, x1 + 1, y1 + 1], radius=2, fill=(0, 0, 0, 45))
        # Main white stroke
        draw.rounded_rectangle([x0, y0, x1, y1], radius=2, fill=(244, 244, 244, 255))

        dot_y0 = int(y1 + max(1, radius * 0.08))
        dot_y1 = int(dot_y0 + dot_h)
        draw.rounded_rectangle([x0 + 1, dot_y0 + 1, x1 + 1, dot_y1 + 1], radius=1, fill=(0, 0, 0, 45))
        draw.rounded_rectangle([x0, dot_y0, x1, dot_y1], radius=1, fill=(244, 244, 244, 255))


def draw_empty_artifact(draw, square_rc, ts):
    """Draw artifact fragments on an empty square without adding piece-like silhouettes."""
    r, c = square_rc
    x0, y0 = c * ts, r * ts

    mode = random.choice(("marker", "short_arrow", "small_highlight"))
    if mode == "marker":
        draw_annotation_marker(draw, square_rc, ts)
        return

    if mode == "small_highlight":
        pad = random.randint(10, 18)
        color = random.choice([(255, 230, 0, 70), (0, 190, 255, 65), (120, 255, 120, 65)])
        draw.rectangle([x0 + pad, y0 + pad, x0 + ts - pad, y0 + ts - pad], fill=color)
        return

    # Short local arrow-like stroke confined inside one tile.
    cx, cy = x0 + ts // 2, y0 + ts // 2
    dx, dy = random.randint(-12, 12), random.randint(-12, 12)
    sx, sy = cx - dx, cy - dy
    ex, ey = cx + dx, cy + dy
    color = random.choice([(255, 165, 0, 110), (0, 255, 0, 105), (0, 150, 255, 105)])
    draw.line([sx, sy, ex, ey], fill=color, width=random.randint(4, 7))


def draw_watermark_overlay(img, draw, square_rc, ts, cfg):
    """Draw puzzle-site style watermark: piece-like silhouette + letter mark."""
    r, c = square_rc
    x0, y0 = c * ts, r * ts

    def _safe_text_size(text, font, size_hint):
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])
        except Exception:
            # Fallback for PIL/font edge cases.
            width = max(1, int(len(text) * size_hint * 0.62))
            height = max(1, int(size_hint * 1.05))
            return width, height

    def _safe_draw_text(xy, text, fill, font):
        try:
            draw.text(xy, text, fill=fill, font=font)
        except Exception:
            fallback_font = ImageFont.load_default()
            draw.text(xy, text, fill=fill, font=fallback_font)

    def _draw_wordmark_text(x, y, size, alpha=205, style="mono"):
        wordmark = random.choice(("SLC", "S.C.C", "LCC", "CHS", "ICC", "C.L.C"))
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except Exception:
            font = ImageFont.load_default()
        if style != "crest":
            tw, th = _safe_text_size(wordmark, font, size)
            tx = x - tw / 2 + random.randint(-1, 1)
            ty = y - th / 2 + random.randint(-1, 1)
            _safe_draw_text((tx + 1, ty + 1), wordmark, (30, 30, 30, 90), font)
            _safe_draw_text((tx, ty), wordmark, (248, 248, 248, alpha), font)
            return

        # Crest style: 3 caps with larger center glyph (similar to puzzle logos).
        letters = random.choice((("S", "L", "C"), ("S", "C", "C"), ("C", "L", "C"), ("I", "C", "C")))
        side_size = max(8, int(size * 0.88))
        mid_size = max(10, int(size * 1.30))
        try:
            side_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", side_size)
            mid_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", mid_size)
        except Exception:
            side_font = ImageFont.load_default()
            mid_font = ImageFont.load_default()

        side_w, _ = _safe_text_size(letters[0], side_font, side_size)
        mid_w, _ = _safe_text_size(letters[1], mid_font, mid_size)
        side2_w, _ = _safe_text_size(letters[2], side_font, side_size)
        total_w = side_w + mid_w + side2_w + max(2, int(size * 0.10)) * 2
        tx = x - total_w / 2
        baseline_y = y - side_size / 2

        gap = max(2, int(size * 0.10))
        # Shadow layer
        _safe_draw_text((tx + 1, baseline_y + 1), letters[0], (25, 25, 25, 110), side_font)
        _safe_draw_text((tx + side_w + gap + 1, baseline_y - int(size * 0.20) + 1), letters[1], (25, 25, 25, 110), mid_font)
        _safe_draw_text((tx + side_w + gap + mid_w + gap + 1, baseline_y + 1), letters[2], (25, 25, 25, 110), side_font)
        # Main layer
        fg = (18, 18, 18, alpha)
        _safe_draw_text((tx, baseline_y), letters[0], fg, side_font)
        _safe_draw_text((tx + side_w + gap, baseline_y - int(size * 0.20)), letters[1], fg, mid_font)
        _safe_draw_text((tx + side_w + gap + mid_w + gap, baseline_y), letters[2], fg, side_font)

    # High-fidelity "logo tile" style: full king + 3-cap wordmark.
    if random.random() < cfg["WATERMARK_FULL_KING_WORDMARK_PROB"]:
        # Build a badge-like overlay inside tile (closer to puzzle-site logos).
        bw = int(ts * random.uniform(0.80, 1.03))
        bh = int(ts * random.uniform(0.74, 0.98))
        bx0 = x0 + random.randint(-int(ts * 0.18), int(ts * 0.06))
        by0 = y0 + random.randint(-int(ts * 0.04), int(ts * 0.18))
        bx0 = max(0, min(img.size[0] - bw, bx0))
        by0 = max(0, min(img.size[1] - bh, by0))
        bx1, by1 = bx0 + bw, by0 + bh
        cx = (bx0 + bx1) // 2

        badge_fill = random.choice([(236, 233, 216, 210), (232, 228, 208, 220), (238, 236, 221, 200)])
        badge_edge = random.choice([(142, 132, 110, 160), (120, 112, 95, 150)])
        draw.rounded_rectangle([bx0, by0, bx1, by1], radius=max(4, int(ts * 0.08)), fill=badge_fill, outline=badge_edge, width=1)

        # Top arc-ish text line (generic, not copied).
        top_line = random.choice(("STRATEGIC CHESS", "TACTICAL CHESS", "CLASSIC CHESS", "CHESS SOCIETY"))
        try:
            top_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(7, int(ts * 0.11)))
        except Exception:
            top_font = ImageFont.load_default()
        tw, _ = _safe_text_size(top_line, top_font, max(7, int(ts * 0.11)))
        tx = cx - tw / 2 + random.randint(-2, 2)
        ty = by0 + int(bh * 0.08)
        _safe_draw_text((tx, ty), top_line, (25, 25, 25, random.randint(120, 190)), top_font)

        # Central king silhouette (dark) to trigger king-like confusion robustly.
        king_col = (22, 22, 22, random.randint(160, 230))
        kcx = cx + random.randint(-2, 2)
        kcy = by0 + int(bh * 0.56)
        kw = int(bw * 0.34)
        kh = int(bh * 0.36)
        draw.rounded_rectangle([kcx - kw // 2, kcy - int(kh * 0.14), kcx + kw // 2, kcy + kh // 2], radius=3, fill=king_col)
        draw.ellipse([kcx - int(kw * 0.18), kcy - int(kh * 0.58), kcx + int(kw * 0.18), kcy - int(kh * 0.20)], fill=king_col)
        cw = max(1, kw // 12)
        ch = max(2, kh // 6)
        draw.rectangle([kcx - cw, kcy - int(kh * 0.70), kcx + cw, kcy - int(kh * 0.45)], fill=king_col)
        draw.rectangle([kcx - int(cw * 2.6), kcy - int(kh * 0.58), kcx + int(cw * 2.6), kcy - int(kh * 0.50)], fill=king_col)

        # Centered 3-cap crest letters.
        _draw_wordmark_text(cx, by0 + int(bh * 0.60), max(10, int(ts * 0.21)), alpha=random.randint(170, 235), style="crest")

        # Bottom tiny tag.
        bottom_line = random.choice(("CLUB", "CHESS CLUB", "SOCIETY"))
        try:
            bottom_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", max(6, int(ts * 0.10)))
        except Exception:
            bottom_font = ImageFont.load_default()
        btw, _ = _safe_text_size(bottom_line, bottom_font, max(6, int(ts * 0.10)))
        _safe_draw_text((cx - btw / 2, by1 - int(bh * 0.18)), bottom_line, (30, 30, 30, random.randint(120, 195)), bottom_font)
        return

    # Bottom-corner bias matches common puzzle-site logo placement.
    anchor = random.choice(("bl", "br", "bc"))
    if anchor == "bl":
        cx = x0 + random.randint(10, 18)
        cy = y0 + ts - random.randint(10, 18)
    elif anchor == "br":
        cx = x0 + ts - random.randint(10, 18)
        cy = y0 + ts - random.randint(10, 18)
    else:
        cx = x0 + ts // 2 + random.randint(-6, 6)
        cy = y0 + ts - random.randint(10, 16)

    mark_size = int(ts * random.uniform(cfg["WATERMARK_SCALE_MIN"], cfg["WATERMARK_SCALE_MAX"]))
    half = max(8, mark_size // 2)

    # Subtle blob base.
    base = random.choice([(126, 126, 126, 88), (102, 102, 102, 100), (142, 142, 142, 76)])
    draw.ellipse([cx - half, cy - half, cx + half, cy + half], fill=base)

    # Piece-like silhouette (rook/king) inside blob.
    sil = random.choice(("rook", "king"))
    sil_col = (236, 236, 236, random.randint(105, 150))
    bx = cx - int(half * 0.42)
    by = cy - int(half * 0.48)
    bw = int(half * 0.84)
    bh = int(half * 0.98)

    if sil == "rook":
        # Rook body
        draw.rounded_rectangle([bx, by + int(bh * 0.28), bx + bw, by + bh], radius=2, fill=sil_col)
        # Rook top crenels
        tw = max(2, bw // 4)
        gap = max(1, tw // 3)
        top_y0 = by + int(bh * 0.08)
        top_y1 = by + int(bh * 0.32)
        for i in range(3):
            x0t = bx + i * (tw + gap)
            x1t = x0t + tw
            draw.rectangle([x0t, top_y0, x1t, top_y1], fill=sil_col)
    else:
        # King body
        draw.rounded_rectangle([bx, by + int(bh * 0.23), bx + bw, by + bh], radius=2, fill=sil_col)
        # Crown/head
        draw.ellipse([cx - int(bw * 0.22), by + int(bh * 0.02), cx + int(bw * 0.22), by + int(bh * 0.3)], fill=sil_col)
        # Tiny cross hint
        cxw = max(1, bw // 10)
        cyh = max(2, bh // 9)
        draw.rectangle([cx - cxw, by - cyh, cx + cxw, by + cyh], fill=sil_col)
        draw.rectangle([cx - int(cxw * 2.4), by, cx + int(cxw * 2.4), by + cyh], fill=sil_col)

    # Wordmark over the piece (club/site style).
    _draw_wordmark_text(cx, cy, max(10, int(half * 0.72)), alpha=random.randint(145, 210))


def draw_board_labels(draw, ts, pov):
    if pov == "black":
        files = "hgfedcba"
        ranks = [str(i) for i in range(1, 9)]
    else:
        files = "abcdefgh"
        ranks = [str(8 - i) for i in range(8)]

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = None

    for i, letter in enumerate(files):
        x = i * ts + ts - 12
        y = 512 - 16
        draw.text((x, y), letter, fill=(128, 128, 128, 180), font=font)

    x = 4 if random.random() < 0.5 else 512 - 14
    for i, rank in enumerate(ranks):
        y = i * ts + 4
        draw.text((x, y), rank, fill=(128, 128, 128, 180), font=font)


def vandalize(img, grid, cfg):
    """Add arrows/highlights like chess sites with artifact-on-empty oversampling."""
    draw = ImageDraw.Draw(img, "RGBA")
    w = img.size[0]
    ts = w // 8
    empty_squares = [(rr, cc) for rr in range(8) for cc in range(8) if not grid[rr][cc]]

    # Square highlights are common but not universal.
    if random.random() < cfg["HIGHLIGHT_BOARD_PROB"]:
        for _ in range(random.randint(1, 2)):
            r, c = random.randint(0, 7), random.randint(0, 7)
            color = random.choice([(0, 255, 0, 65), (255, 255, 0, 65), (255, 0, 0, 65), (0, 150, 255, 65)])
            draw.rectangle([c * ts, r * ts, (c + 1) * ts, (r + 1) * ts], fill=color)

    # Arrows appear often, but not on every board.
    if random.random() < cfg["ARROW_BOARD_PROB"]:
        for _ in range(random.randint(1, 2)):
            start_r, start_c = random.randint(0, 7), random.randint(0, 7)
            end_r = max(0, min(7, start_r + random.randint(-3, 3)))
            end_c = max(0, min(7, start_c + random.randint(-3, 3)))
            if (start_r, start_c) != (end_r, end_c):
                color = random.choice([(0, 255, 0, 110), (255, 165, 0, 110), (255, 0, 0, 110), (0, 150, 255, 110)])
                draw_arrow(draw, (start_r, start_c), (end_r, end_c), color, ts)

    # Tactical markers are frequent for robustness against ! and !! overlays.
    if random.random() < cfg["TACTICAL_MARKER_PROB"]:
        for _ in range(random.choices([1, 2, 3], weights=[0.60, 0.30, 0.10], k=1)[0]):
            r, c = random.randint(0, 7), random.randint(0, 7)
            draw_annotation_marker(draw, (r, c), ts)

    # Targeted hardening: artifact-only empty tiles to reduce empty->piece hallucination.
    if empty_squares and random.random() < cfg["ARTIFACT_EMPTY_TILE_PROB"]:
        random.shuffle(empty_squares)
        for sq in empty_squares[:random.randint(2, 4)]:
            draw_empty_artifact(draw, sq, ts)

    # Watermark-like overlays on empty squares (controlled via top-level vars).
    if cfg["ENABLE_WATERMARK_AUG"] and empty_squares and random.random() < cfg["WATERMARK_BOARD_PROB"]:
        # Bias watermark injection toward bottom ranks where logos commonly sit.
        bottom_pref = [sq for sq in empty_squares if sq[0] >= 6]
        pool = bottom_pref if bottom_pref else empty_squares
        random.shuffle(pool)
        n = random.randint(cfg["WATERMARK_MIN_PER_BOARD"], cfg["WATERMARK_MAX_PER_BOARD"])
        for sq in pool[:n]:
            draw_watermark_overlay(img, draw, sq, ts, cfg)

    return img


def simulate_trimmed_capture(img, cfg):
    """Simulate imperfect screenshots where board edges/labels are partially clipped."""
    if random.random() >= cfg["TRIM_CAPTURE_PROB"]:
        return img

    w, h = img.size
    heavy = random.random() < cfg["DETECTOR_HEAVY_TRIM_PROB"]
    if heavy:
        max_crop_x = max(6, int(w * 0.16))
        max_crop_y = max(6, int(h * 0.16))
        min_keep_ratio = 0.65
    else:
        max_crop_x = max(4, int(w * 0.05))
        max_crop_y = max(4, int(h * 0.05))
        min_keep_ratio = 0.80

    # Asymmetric side trimming is common in mobile/browser screenshots.
    left = random.randint(0, max_crop_x)
    right = random.randint(0, max_crop_x)
    top = random.randint(0, max_crop_y)
    bottom = random.randint(0, max_crop_y)

    # Keep at least min_keep_ratio of each dimension before resizing back.
    max_total_crop_x = int(w * (1.0 - min_keep_ratio))
    max_total_crop_y = int(h * (1.0 - min_keep_ratio))
    if (left + right) > max_total_crop_x:
        right = max(0, max_total_crop_x - left)
    if (top + bottom) > max_total_crop_y:
        bottom = max(0, max_total_crop_y - top)

    cropped = img.crop((left, top, w - right, h - bottom))
    return cropped.resize((w, h), Image.LANCZOS)


@lru_cache(maxsize=32)
def load_print_piece_alpha(piece_name):
    path = os.path.join(PIECE_SETS_DIR, PRINT_DIAGRAM_BASE_SET, piece_name)
    img = Image.open(path).convert("RGBA").resize((64, 64), Image.LANCZOS)
    return img.getchannel("A")


@lru_cache(maxsize=64)
def load_board_theme_base(board_theme_name, size):
    path = os.path.join(BOARD_THEMES_DIR, board_theme_name)
    return Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)


@lru_cache(maxsize=1)
def available_board_theme_names():
    return tuple(
        sorted(
            name
            for name in os.listdir(BOARD_THEMES_DIR)
            if os.path.isfile(os.path.join(BOARD_THEMES_DIR, name))
        )
    )


@lru_cache(maxsize=1)
def available_piece_set_names():
    return tuple(
        sorted(
            name
            for name in os.listdir(PIECE_SETS_DIR)
            if os.path.isdir(os.path.join(PIECE_SETS_DIR, name))
        )
    )


def choose_print_diagram_style(profile, cfg):
    forced_styles = cfg.get("PRINT_STYLE_CHOICES")
    if forced_styles:
        names = [name for name, _ in forced_styles]
        weights = [weight for _, weight in forced_styles]
        return random.choices(names, weights=weights, k=1)[0]

    if profile == "mono_print_sparse_edge":
        styles = [("hatched_book", 0.45), ("shirt_print", 0.35), ("flat_book", 0.20)]
    elif profile == "mono_print_edge_rook":
        styles = [("flat_book", 0.55), ("hatched_book", 0.35), ("shirt_print", 0.10)]
    elif profile in {"mono_print_sparse_light", "mono_scan"}:
        styles = [("flat_book", 0.58), ("hatched_book", 0.37), ("shirt_print", 0.05)]
    elif profile == "book_page_reference":
        styles = [("book_reference", 1.0)]
    elif profile == "shirt_print_reference":
        styles = [("shirt_reference", 1.0)]
    else:
        styles = [("hatched_book", 0.35), ("flat_book", 0.40), ("shirt_print", 0.25)]
    names = [name for name, _ in styles]
    weights = [weight for _, weight in styles]
    return random.choices(names, weights=weights, k=1)[0]


def build_print_grid_edges(size, tile_size, max_jitter):
    if max_jitter <= 0:
        return [i * tile_size for i in range(9)]

    edges = [0]
    min_width = tile_size - max_jitter
    for idx in range(1, 8):
        base = idx * tile_size + random.randint(-max_jitter, max_jitter)
        min_edge = edges[-1] + min_width
        max_edge = size - (8 - idx) * min_width
        edges.append(int(max(min_edge, min(base, max_edge))))
    edges.append(size)
    return edges


def render_print_board_base(style, tile_size=64, board_theme_name=None):
    size = tile_size * 8
    if board_theme_name:
        board = load_board_theme_base(board_theme_name, size).copy()
        arr = np.asarray(board, dtype=np.float32)
        if style in {"book_reference", "hatched_book", "flat_book", "book_photo"}:
            paper = build_soft_noise_map(size, size, low_res_side=22, low=0.97, high=1.03)[:, :, None]
            arr *= paper
            if style == "book_photo":
                page_x = np.linspace(1.03, 0.98, size, dtype=np.float32)[None, :, None]
                page_y = np.linspace(0.99, 1.02, size, dtype=np.float32)[:, None, None]
                arr *= page_x * page_y
        elif style in {"shirt_reference", "shirt_print", "shirt_photo"}:
            cloth_x = build_soft_noise_map(size, size, low_res_side=54, low=0.985, high=1.015)[:, :, None]
            cloth_y = build_soft_noise_map(size, size, low_res_side=70, low=0.99, high=1.01)[:, :, None]
            arr *= cloth_x * cloth_y
            if style == "shirt_photo":
                arr *= np.linspace(0.99, 1.03, size, dtype=np.float32)[:, None, None]
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGB"), [i * tile_size for i in range(9)], [i * tile_size for i in range(9)], board_theme_name
    if style in {"book_reference", "shirt_reference"}:
        board, x_edges, y_edges = render_reference_board_base(style, tile_size=tile_size)
        return board, x_edges, y_edges, None

    canvas = Image.new("L", (size, size), 226)
    draw = ImageDraw.Draw(canvas)

    if style == "hatched_book":
        light_fill, dark_fill, hatch_fill = 222, 200, 150
        x_edges = build_print_grid_edges(size, tile_size, 0)
        y_edges = build_print_grid_edges(size, tile_size, 0)
    elif style == "shirt_print":
        light_fill, dark_fill, hatch_fill = 230, 206, 164
        x_edges = build_print_grid_edges(size, tile_size, 0)
        y_edges = build_print_grid_edges(size, tile_size, 0)
    elif style == "book_photo":
        light_fill, dark_fill, hatch_fill = 235, 190, 150
        x_edges = build_print_grid_edges(size, tile_size, 4)
        y_edges = build_print_grid_edges(size, tile_size, 4)
    elif style == "shirt_photo":
        light_fill, dark_fill, hatch_fill = 218, 180, 148
        x_edges = build_print_grid_edges(size, tile_size, 5)
        y_edges = build_print_grid_edges(size, tile_size, 5)
    else:
        light_fill, dark_fill, hatch_fill = 224, 186, 166
        x_edges = build_print_grid_edges(size, tile_size, 0)
        y_edges = build_print_grid_edges(size, tile_size, 0)

    for row_idx in range(8):
        for col_idx in range(8):
            x0 = x_edges[col_idx]
            y0 = y_edges[row_idx]
            x1 = x_edges[col_idx + 1]
            y1 = y_edges[row_idx + 1]
            is_dark = (row_idx + col_idx) % 2 == 1
            fill = dark_fill if is_dark else light_fill
            draw.rectangle([x0, y0, x1, y1], fill=fill)
            if is_dark and style in {"hatched_book", "shirt_print"}:
                step = 5 if style == "hatched_book" else 6
                for offset in range(-tile_size, tile_size * 2, step):
                    draw.line(
                        [(x0 + offset, y0), (x0 + offset + (x1 - x0), y1)],
                        fill=hatch_fill,
                        width=1,
                    )

    if style in {"book_photo", "shirt_photo"}:
        line_fill = 132 if style == "book_photo" else 138
        for edge in x_edges[1:-1]:
            draw.line([edge, 0, edge, size], fill=line_fill, width=random.randint(1, 2))
        for edge in y_edges[1:-1]:
            draw.line([0, edge, size, edge], fill=line_fill, width=random.randint(1, 2))

    arr = np.asarray(canvas, dtype=np.float32)
    paper = build_soft_noise_map(size, size, low_res_side=18, low=0.90, high=1.08)
    arr *= paper

    if style in {"shirt_print", "shirt_photo"}:
        cloth = build_soft_noise_map(size, size, low_res_side=52, low=0.94, high=1.06)
        arr = arr * cloth + (cloth - 1.0) * 24.0
        if style == "shirt_photo":
            weave_x = build_soft_noise_map(size, size, low_res_side=96, low=0.985, high=1.015)
            weave_y = build_soft_noise_map(size, size, low_res_side=96, low=0.985, high=1.015).T
            weave_y = weave_y[:size, :size]
            arr = arr * weave_x * weave_y
    elif style == "book_photo":
        page_slope = np.linspace(1.03, 0.97, size, dtype=np.float32)
        arr *= np.tile(page_slope, (size, 1))
        spine_shadow = np.linspace(0.92, 1.02, size, dtype=np.float32)
        arr *= np.tile(spine_shadow[:, None], (1, size))
        margin_shadow = np.linspace(0.97, 1.03, size, dtype=np.float32)
        arr *= np.tile(margin_shadow, (size, 1))
    else:
        fibers_x = build_soft_noise_map(size, size, low_res_side=68, low=0.97, high=1.03)
        fibers_y = build_soft_noise_map(size, size, low_res_side=68, low=0.97, high=1.03).T
        fibers_y = fibers_y[:size, :size]
        arr = arr + (fibers_x - 1.0) * 14.0 + (fibers_y - 1.0) * 10.0

    board = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("L").convert("RGB")
    return board, x_edges, y_edges, None


def render_reference_board_base(style, tile_size=64):
    size = tile_size * 8
    if style == "book_reference":
        light_fill = np.array([233.0, 233.0, 231.0], dtype=np.float32)
        dark_fill = np.array([184.0, 184.0, 181.0], dtype=np.float32)
        grid_fill = np.array([104.0, 104.0, 102.0], dtype=np.float32)
        x_edges = build_print_grid_edges(size, tile_size, 1)
        y_edges = build_print_grid_edges(size, tile_size, 1)
        paper_low = (0.965, 1.025)
    else:
        light_fill = np.array([223.0, 221.0, 232.0], dtype=np.float32)
        dark_fill = np.array([188.0, 184.0, 201.0], dtype=np.float32)
        grid_fill = np.array([132.0, 128.0, 142.0], dtype=np.float32)
        x_edges = build_print_grid_edges(size, tile_size, 1)
        y_edges = build_print_grid_edges(size, tile_size, 1)
        paper_low = (0.955, 1.035)

    board = np.zeros((size, size, 3), dtype=np.float32)
    for row_idx in range(8):
        for col_idx in range(8):
            x0 = x_edges[col_idx]
            y0 = y_edges[row_idx]
            x1 = x_edges[col_idx + 1]
            y1 = y_edges[row_idx + 1]
            fill = dark_fill if (row_idx + col_idx) % 2 else light_fill
            board[y0:y1, x0:x1] = fill

    for edge in x_edges[1:-1]:
        board[:, max(0, edge - 1) : min(size, edge + 1)] = grid_fill
    for edge in y_edges[1:-1]:
        board[max(0, edge - 1) : min(size, edge + 1), :] = grid_fill

    outer = grid_fill * (0.88 if style == "book_reference" else 0.94)
    border = 4 if style == "book_reference" else 3
    board[:border, :] = outer
    board[-border:, :] = outer
    board[:, :border] = outer
    board[:, -border:] = outer

    paper = build_soft_noise_map(size, size, low_res_side=26, low=paper_low[0], high=paper_low[1])[:, :, None]
    board *= paper
    if style == "book_reference":
        page_x = np.linspace(1.03, 0.98, size, dtype=np.float32)[None, :, None]
        page_y = np.linspace(0.99, 1.02, size, dtype=np.float32)[:, None, None]
        board *= page_x * page_y
    else:
        cloth_x = build_soft_noise_map(size, size, low_res_side=58, low=0.98, high=1.02)[:, :, None]
        cloth_y = build_soft_noise_map(size, size, low_res_side=74, low=0.985, high=1.015)[:, :, None]
        board *= cloth_x * cloth_y

    return Image.fromarray(np.clip(board, 0, 255).astype(np.uint8)).convert("RGB"), x_edges, y_edges


def render_print_piece(char, style, row_idx, col_idx):
    if style in {"book_reference", "shirt_reference"}:
        return render_reference_piece(char, style, row_idx, col_idx)

    piece_name = f"{'w' if char.isupper() else 'b'}{char.upper()}.png"
    alpha_img = load_print_piece_alpha(piece_name).copy()
    edge_square = row_idx in (0, 7) or col_idx in (0, 7)

    if style in {"shirt_print", "shirt_photo"}:
        alpha_img = alpha_img.resize((58, 58), Image.BICUBIC).resize((64, 64), Image.BICUBIC)
    elif style in {"hatched_book", "book_photo"}:
        alpha_img = alpha_img.resize((60, 60), Image.BICUBIC).resize((64, 64), Image.BICUBIC)

    arr = np.asarray(alpha_img, dtype=np.float32) / 255.0
    damage = build_piece_damage_map(64, edge_square=edge_square)
    if style == "flat_book":
        damage_floor = 0.74
    elif style == "hatched_book":
        damage_floor = 0.62
    elif style == "book_photo":
        damage_floor = 0.88
    elif style == "shirt_photo":
        damage_floor = 0.76
    else:
        damage_floor = 0.30
    arr *= np.clip(damage_floor + damage * (1.0 - damage_floor), 0.0, 1.0)
    if edge_square and (style in {"shirt_print", "shirt_photo"} or random.random() < 0.45):
        arr *= build_edge_piece_fade_mask(64, row_idx, col_idx)
    if random.random() < (0.20 if style not in {"shirt_print", "shirt_photo"} else 0.60):
        cutoff = random.uniform(0.10, 0.20 if edge_square else 0.15)
        arr = np.where(arr >= cutoff, arr, 0.0)

    alpha = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).convert("L")
    if style in {"shirt_print", "shirt_photo"} and random.random() < 0.80:
        radius = random.uniform(0.35, 0.95)
        if style == "shirt_photo":
            radius = random.uniform(0.18, 0.55)
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=radius))
    elif style != "shirt_print" and random.random() < 0.25:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.10, 0.35)))

    outline = alpha.filter(ImageFilter.MaxFilter(3))
    inner = alpha.filter(ImageFilter.MinFilter(3))
    outline_arr = np.clip(
        np.asarray(outline, dtype=np.float32) / 255.0 - np.asarray(inner, dtype=np.float32) / 255.0,
        0.0,
        1.0,
    )
    inner_arr = np.asarray(inner, dtype=np.float32) / 255.0

    if char.isupper():
        fill_val = 244 if style == "flat_book" else (236 if style in {"hatched_book", "book_photo"} else 226)
        line_val = 92 if style == "flat_book" else (110 if style in {"hatched_book", "book_photo"} else 132)
        rgb = inner_arr * fill_val + outline_arr * line_val
        alpha_arr = np.maximum(inner_arr * 0.96, outline_arr) * 255.0
    else:
        fill_val = 26 if style == "flat_book" else (42 if style in {"hatched_book", "book_photo"} else 72)
        line_val = 72 if style == "flat_book" else (88 if style in {"hatched_book", "book_photo"} else 114)
        rgb = inner_arr * fill_val + outline_arr * line_val * 0.35
        alpha_arr = np.maximum(inner_arr, outline_arr * 0.8) * 255.0

    canvas = np.zeros((64, 64, 4), dtype=np.uint8)
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    canvas[:, :, 0] = rgb_u8
    canvas[:, :, 1] = rgb_u8
    canvas[:, :, 2] = rgb_u8
    canvas[:, :, 3] = np.clip(alpha_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(canvas).convert("RGBA")


def render_reference_piece(char, style, row_idx, col_idx):
    piece_name = f"{'w' if char.isupper() else 'b'}{char.upper()}.png"
    alpha_img = load_print_piece_alpha(piece_name).copy()
    edge_square = row_idx in (0, 7) or col_idx in (0, 7)

    arr = np.asarray(alpha_img, dtype=np.float32) / 255.0
    if style == "book_reference":
        floor = 0.90
    else:
        floor = 0.84
    damage = build_piece_damage_map(64, edge_square=edge_square)
    arr *= np.clip(floor + damage * (1.0 - floor), 0.0, 1.0)
    if edge_square and random.random() < (0.08 if style == "book_reference" else 0.12):
        arr *= build_edge_piece_fade_mask(64, row_idx, col_idx)
    alpha = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8)).convert("L")
    if style == "shirt_reference" and random.random() < 0.35:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.10, 0.22)))
    elif style == "book_reference" and random.random() < 0.12:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.05, 0.15)))

    outline = alpha.filter(ImageFilter.MaxFilter(3))
    inner = alpha.filter(ImageFilter.MinFilter(3))
    outline_arr = np.clip(
        np.asarray(outline, dtype=np.float32) / 255.0 - np.asarray(inner, dtype=np.float32) / 255.0,
        0.0,
        1.0,
    )
    inner_arr = np.asarray(inner, dtype=np.float32) / 255.0

    if style == "book_reference":
        white_fill = np.array([246.0, 246.0, 244.0], dtype=np.float32)
        white_line = np.array([84.0, 84.0, 82.0], dtype=np.float32)
        black_fill = np.array([28.0, 28.0, 28.0], dtype=np.float32)
        black_line = np.array([92.0, 92.0, 90.0], dtype=np.float32)
    else:
        white_fill = np.array([240.0, 239.0, 243.0], dtype=np.float32)
        white_line = np.array([116.0, 112.0, 126.0], dtype=np.float32)
        black_fill = np.array([52.0, 50.0, 62.0], dtype=np.float32)
        black_line = np.array([112.0, 106.0, 126.0], dtype=np.float32)

    canvas = np.zeros((64, 64, 4), dtype=np.uint8)
    if char.isupper():
        rgb = inner_arr[:, :, None] * white_fill + outline_arr[:, :, None] * white_line
        alpha_arr = np.maximum(inner_arr * 0.97, outline_arr) * 255.0
    else:
        rgb = inner_arr[:, :, None] * black_fill + outline_arr[:, :, None] * black_line * 0.40
        alpha_arr = np.maximum(inner_arr, outline_arr * 0.82) * 255.0
    canvas[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    canvas[:, :, 3] = np.clip(alpha_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(canvas).convert("RGBA")


def apply_print_capture_noise(img, style):
    out = img
    if style == "book_reference":
        if random.random() < 0.65:
            down = random.randint(488, 506)
            out = out.resize((down, down), Image.BILINEAR).resize((512, 512), Image.BILINEAR)
        if random.random() < 0.20:
            out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.06, 0.18)))
        if random.random() < 0.32:
            out = ImageEnhance.Contrast(out).enhance(random.uniform(0.96, 1.03))
        return out
    if style == "shirt_reference":
        if random.random() < 0.72:
            down = random.randint(470, 500)
            out = out.resize((down, down), Image.BILINEAR).resize((512, 512), Image.BILINEAR)
        if random.random() < 0.42:
            out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.10, 0.26)))
        if random.random() < 0.36:
            out = ImageEnhance.Contrast(out).enhance(random.uniform(0.94, 1.02))
        return out
    if random.random() < 0.75:
        if style in {"shirt_print", "shirt_photo"}:
            down = random.randint(392, 460)
        elif style == "book_photo":
            down = random.randint(460, 498)
        else:
            down = random.randint(470, 502)
        out = out.resize((down, down), Image.BILINEAR).resize((512, 512), Image.BILINEAR)
    if style in {"shirt_print", "shirt_photo"} and random.random() < 0.75:
        radius = random.uniform(0.30, 0.90)
        if style == "shirt_photo":
            radius = random.uniform(0.16, 0.45)
        out = out.filter(ImageFilter.GaussianBlur(radius=radius))
    elif style == "book_photo" and random.random() < 0.55:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.12, 0.40)))
    elif style != "shirt_print" and random.random() < 0.35:
        out = out.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.08, 0.28)))
    if random.random() < 0.45:
        low, high = (0.88, 0.97) if style in {"shirt_print", "shirt_photo"} else (0.95, 1.02)
        out = ImageEnhance.Contrast(out).enhance(random.uniform(low, high))
    if random.random() < 0.40:
        arr = np.asarray(out, dtype=np.float32)
        noise = np.random.normal(0.0, random.uniform(1.5, 5.0), arr.shape)
        out = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    return out


def render_print_diagram_board(fen, profile, cfg):
    style = choose_print_diagram_style(profile, cfg)
    tile_size = 64
    board_theme_choices = cfg.get("BOARD_THEME_NAMES") or []
    board_theme_name = random.choice(board_theme_choices) if board_theme_choices else None
    board, x_edges, y_edges, board_theme_used = render_print_board_base(
        style,
        tile_size=tile_size,
        board_theme_name=board_theme_name,
    )

    # Profile-specific piece scale override (default per-style scales below)
    profile_piece_scale = cfg.get("PRINT_PIECE_SCALE")

    grid = [[None] * 8 for _ in range(8)]
    for row_idx, row_str in enumerate(fen.split("/")):
        col_idx = 0
        for ch in row_str:
            if ch.isdigit():
                col_idx += int(ch)
            else:
                grid[row_idx][col_idx] = ch
                col_idx += 1

    for row_idx in range(8):
        for col_idx in range(8):
            char = grid[row_idx][col_idx]
            if not char:
                continue
            piece_img = render_print_piece(char, style, row_idx, col_idx)
            square_w = x_edges[col_idx + 1] - x_edges[col_idx]
            square_h = y_edges[row_idx + 1] - y_edges[row_idx]
            # Use profile override if set, otherwise use style-specific default
            if profile_piece_scale is not None:
                piece_scale = profile_piece_scale
            elif style == "shirt_photo":
                piece_scale = 0.76
            elif style == "book_photo":
                piece_scale = 0.80
            elif style == "shirt_reference":
                piece_scale = 0.78
            elif style == "book_reference":
                piece_scale = 0.82
            else:
                piece_scale = 0.84
            if row_idx in (0, 7) or col_idx in (0, 7):
                piece_scale *= 0.96
            piece_size = max(42, int(min(square_w, square_h) * piece_scale))
            placed = piece_img.resize((piece_size, piece_size), Image.BICUBIC)
            jitter_x = random.randint(-2, 2) if style in {"book_photo", "shirt_photo", "book_reference", "shirt_reference"} else 0
            jitter_y = random.randint(-2, 2) if style in {"book_photo", "shirt_photo", "book_reference", "shirt_reference"} else 0
            paste_x = x_edges[col_idx] + (square_w - piece_size) // 2 + jitter_x
            paste_y = y_edges[row_idx] + (square_h - piece_size) // 2 + jitter_y
            board.paste(placed, (paste_x, paste_y), placed)

    board = apply_local_piece_tilt(board, grid, cfg)
    board = apply_print_capture_noise(board, style)
    return board, grid, board_theme_used or f"print_{style}", f"print_{style}", None


def render_board(fen, return_meta=False, profile=None):
    profile = choose_profile(profile)
    cfg = get_profile_config(profile)
    if profile in PRINT_DIAGRAM_PROFILES:
        background, grid, board_file, p_set, label_pov = render_print_diagram_board(fen, profile, cfg)
        ts = 64
        tiles, labels = [], []
        for r in range(8):
            for c in range(8):
                tile = background.crop((c * ts, r * ts, (c + 1) * ts, (r + 1) * ts))
                tiles.append(np.array(tile, dtype=np.uint8).transpose(2, 0, 1))
                labels.append(FEN_CHARS.index(grid[r][c]) if grid[r][c] else 0)
        if return_meta:
            return tiles, labels, {
                "board_theme": board_file,
                "piece_set": p_set,
                "label_pov": label_pov,
                "profile": profile,
            }
        return tiles, labels

    board_choices = cfg.get("BOARD_THEME_NAMES") or available_board_theme_names()
    piece_choices = cfg.get("PIECE_SET_NAMES") or [
        name for name in DEFAULT_GENERAL_PIECE_SET_NAMES if os.path.isdir(os.path.join(PIECE_SETS_DIR, name))
    ]
    board_file = random.choice(board_choices)
    background = Image.open(os.path.join(BOARD_THEMES_DIR, board_file)).convert("RGB").resize((512, 512))
    p_set = random.choice(piece_choices)
    ts = 64
    
    grid = [[None]*8 for _ in range(8)]
    for r, row_str in enumerate(fen.split('/')):
        c = 0
        for char in row_str:
            if char.isdigit(): c += int(char)
            else: grid[r][c] = char; c += 1

    for r in range(8):
        for c in range(8):
            char = grid[r][c]
            if char:
                # Standard Lichess filename mapping: wP, bK, etc.
                p_name = f"{'w' if char.isupper() else 'b'}{char.upper()}.png"
                p_img = Image.open(os.path.join(PIECE_SETS_DIR, p_set, p_name)).convert("RGBA").resize((ts, ts))
                background.paste(p_img, (c*ts, r*ts), p_img)

    # Add coordinate labels - randomize position to match real-world variety
    label_choice = random.random()
    label_pov = None
    forced_label_pov = cfg.get("FORCE_LABEL_POV")
    if label_choice < cfg["LABELS_PROB"]:
        draw = ImageDraw.Draw(background)
        if forced_label_pov in {"white", "black"}:
            label_pov = forced_label_pov
        else:
            label_pov = random.choice(("white", "black")) if profile != "default" else "white"
        draw_board_labels(draw, ts, label_pov)
    background = vandalize(background, grid, cfg)
    if profile == "digital_overlay_clean":
        background = apply_digital_overlay_clean(background, grid)
    background = apply_piece_occlusion_overlay(background, grid, cfg)
    background = apply_local_piece_tilt(background, grid, cfg)
    if profile == "dark_anchor_clean":
        background = apply_dark_anchor_grade(background)
    if profile == "broadcast_dark_sparse":
        background = apply_broadcast_dark_style(background)

    # Detector-hard scene augmentation: partial board, banners, and low-contrast scans.
    background = compose_partial_board_scene(background, cfg)
    background = add_detector_banner_overlay(background, cfg)
    background = apply_mono_book_style(background, grid, cfg)
    
    # Apply augmentation BEFORE compression
    background = augment_image(background, cfg)
    
    # JPEG compression artifacts are common in user screenshots.
    if random.random() < 0.60:
        buf = io.BytesIO()
        background.save(buf, "JPEG", quality=random.randint(45, 92))
        background = Image.open(buf).copy()

    # Simulate trimmed screenshot captures before tile slicing.
    background = simulate_trimmed_capture(background, cfg)

    tiles, labels = [], []
    for r in range(8):
        for c in range(8):
            tile = background.crop((c*ts, r*ts, (c+1)*ts, (r+1)*ts))
            tiles.append(np.array(tile, dtype=np.uint8).transpose(2,0,1))
            labels.append(FEN_CHARS.index(grid[r][c]) if grid[r][c] else 0)
    if return_meta:
        return tiles, labels, {
            "board_theme": board_file,
            "piece_set": p_set,
            "label_pov": label_pov,
            "profile": profile,
        }
    return tiles, labels


def board_has_edge_rook(board):
    edge_squares = []
    for sq in chess.SQUARES:
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        if file_idx in (0, 7) or rank_idx in (0, 7):
            edge_squares.append(sq)
    for sq in edge_squares:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.ROOK:
            return True
    return False


def board_has_file_edge_rook(board):
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.ROOK and chess.square_file(sq) in (0, 7):
            return True
    return False


def is_sparse_board(board):
    return len(board.piece_map()) <= 10


def _random_non_adjacent_square(taken, forbidden_neighbors=()):
    candidates = []
    forbidden = set(forbidden_neighbors)
    for sq in chess.SQUARES:
        if sq in taken or sq in forbidden:
            continue
        candidates.append(sq)
    return random.choice(candidates) if candidates else None


def _king_neighborhood(square):
    file_idx = chess.square_file(square)
    rank_idx = chess.square_rank(square)
    blocked = set()
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            nf, nr = file_idx + df, rank_idx + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                blocked.add(chess.square(nf, nr))
    return blocked


def build_screenshot_clutter_board(force_edge_rook=False, force_file_edge_rook=False):
    """Build sparse screenshot-style boards with lots of empty squares."""
    board = chess.Board(None)
    taken = set()

    white_king = random.choice([
        chess.C1, chess.G1, chess.B2, chess.F2, chess.C2, chess.G2
    ])
    taken.add(white_king)
    black_king = _random_non_adjacent_square(taken, _king_neighborhood(white_king))
    if black_king is None:
        black_king = chess.E8
    taken.add(black_king)

    board.set_piece_at(white_king, chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(black_king, chess.Piece(chess.KING, chess.BLACK))

    optional_pieces = [
        (chess.QUEEN, chess.WHITE), (chess.QUEEN, chess.BLACK),
        (chess.ROOK, chess.WHITE), (chess.ROOK, chess.BLACK),
        (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.BLACK),
        (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.BLACK),
        (chess.PAWN, chess.WHITE), (chess.PAWN, chess.BLACK),
        (chess.PAWN, chess.WHITE), (chess.PAWN, chess.BLACK),
    ]
    random.shuffle(optional_pieces)

    if force_file_edge_rook or force_edge_rook:
        if force_file_edge_rook:
            edge_targets = [chess.A8, chess.H8, chess.A7, chess.H7, chess.A4, chess.H5, chess.A2, chess.H2, chess.A1, chess.H1]
        else:
            edge_targets = [chess.A7, chess.H7, chess.A2, chess.H2, chess.A4, chess.H5]
        candidates = [cand for cand in edge_targets if cand not in taken]
        if candidates:
            sq = random.choice(candidates)
            color = random.choice([chess.WHITE, chess.BLACK])
            board.set_piece_at(sq, chess.Piece(chess.ROOK, color))
            taken.add(sq)

    extra_count = random.randint(1, 5)
    for piece_type, color in optional_pieces:
        if extra_count <= 0:
            break
        candidates = []
        for sq in chess.SQUARES:
            if sq in taken:
                continue
            if piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7):
                continue
            candidates.append(sq)
        if not candidates:
            continue
        sq = random.choice(candidates)
        board.set_piece_at(sq, chess.Piece(piece_type, color))
        taken.add(sq)
        extra_count -= 1

    return board


def random_training_board(profile=None):
    profile = choose_profile(profile)
    cfg = get_profile_config(profile)
    if random.random() < cfg["SCREENSHOT_CLUTTER_PROB"]:
        force_edge_rook = random.random() < cfg["HARD_EDGE_ROOK_PROB"]
        force_file_edge_rook = random.random() < cfg["HARD_FILE_EDGE_ROOK_PROB"]
        return build_screenshot_clutter_board(
            force_edge_rook=force_edge_rook,
            force_file_edge_rook=force_file_edge_rook,
        )

    for _ in range(200):
        board = chess.Board()
        target_sparse = random.random() < cfg["SPARSE_BOARD_PROB"]
        target_edge_rook = random.random() < cfg["HARD_EDGE_ROOK_PROB"]
        target_file_edge_rook = random.random() < cfg["HARD_FILE_EDGE_ROOK_PROB"]
        plies = random.randint(cfg["MIN_PLIES"], cfg["MAX_PLIES"])
        for _ in range(plies):
            if board.is_game_over():
                break
            board.push(random.choice(list(board.legal_moves)))

        if target_sparse and not is_sparse_board(board):
            continue
        if target_edge_rook and not board_has_edge_rook(board):
            continue
        if target_file_edge_rook and not board_has_file_edge_rook(board):
            continue
        return board

    board = chess.Board()
    for _ in range(random.randint(cfg["MIN_PLIES"], cfg["MAX_PLIES"])):
        if board.is_game_over():
            break
        board.push(random.choice(list(board.legal_moves)))
    return board

if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"📦 Generation recipe: {RECIPE_NAME}")
    print(f"   Weights: {DEFAULT_PROFILE_WEIGHTS}")
    print(f"   Seed: {SEED}")

    chunk_mix_counts = {}
    chunk_order = [f"val_{i}" for i in range(CHUNKS_VAL)] + [f"train_{i}" for i in range(CHUNKS_TRAIN)]
    for name in chunk_order:
        all_x, all_y = [], []
        profile_plan, profile_counts = build_profile_plan(BOARDS_PER_CHUNK, DEFAULT_PROFILE_WEIGHTS)
        for profile in profile_plan:
            b = random_training_board(profile=profile)
            t, l = render_board(b.fen().split()[0], profile=profile)
            all_x.extend(t)
            all_y.extend(l)
        chunk_path = os.path.join(OUTPUT_DIR, f"{name}.pt")
        atomic_torch_save(
            {"x": torch.from_numpy(np.stack(all_x)), "y": torch.tensor(all_y)},
            chunk_path,
        )
        chunk_mix_counts[name] = profile_counts
        mix = ", ".join(f"{k}:{profile_counts.get(k, 0)}" for k, _ in DEFAULT_PROFILE_WEIGHTS)
        print(f"✅ Created {name} | mix[{mix}]")

    manifest = {
        "generator": "generate_hybrid_v6",
        "recipe_name": RECIPE_NAME,
        "profile_weights": {name: weight for name, weight in DEFAULT_PROFILE_WEIGHTS},
        "seed": SEED,
        "chunks_train": CHUNKS_TRAIN,
        "chunks_val": CHUNKS_VAL,
        "boards_per_chunk": BOARDS_PER_CHUNK,
        "expected_samples_per_chunk": BOARDS_PER_CHUNK * 64,
        "output_dir": OUTPUT_DIR,
        "chunk_mix_counts": chunk_mix_counts,
    }
    manifest_path = os.path.join(OUTPUT_DIR, "generation_manifest_v6.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"🧾 Wrote manifest: {manifest_path}")
