import unittest
from pathlib import Path
from unittest.mock import patch

import sys

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent.parent
SCRIPTS_DIR = TRAIN_DIR / "scripts"
for path in (TRAIN_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_v6_hardset as eval_v6
import recognizer_v6 as v6


class TestSideToMoveAliases(unittest.TestCase):
    def test_side_to_move_aliases(self):
        for alias in ("w", "white", "wtm", "white_to_move"):
            self.assertEqual(v6.parse_side_to_move_override(alias), "w")
        for alias in ("b", "black", "blak", "btm", "black_to_move"):
            self.assertEqual(v6.parse_side_to_move_override(alias), "b")
        with self.assertRaises(ValueError):
            v6.parse_side_to_move_override("invalid")


class TestFenTransforms(unittest.TestCase):
    def test_rotate_fen_180_is_involution(self):
        cases = [
            "8/8/8/8/8/8/8/8",
            "r2q2rk/bpp2p1p/p4PbQ/6N1/8/3P4/PP4PP/4RR1K",
            "3r4/3kn2p/1p4pP/1p1B1pP1/1P1K1P2/P1P5/8/4R3",
        ]
        for fen_board in cases:
            self.assertEqual(v6.rotate_fen_180(v6.rotate_fen_180(fen_board)), fen_board)


class TestBatchingInvariant(unittest.TestCase):
    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_shapes = []

        def forward(self, x):
            self.forward_shapes.append(tuple(x.shape))
            return torch.zeros((x.shape[0], 13), dtype=torch.float32, device=x.device)

    def test_infer_fen_batches_all_tiles_single_forward(self):
        model = self.CountingModel()
        device = torch.device("cpu")
        img = Image.new("RGB", (512, 512), (127, 127, 127))
        v6.infer_fen_on_image_clean(
            img,
            model,
            device,
            use_square_detection=False,
            return_details=False,
        )
        self.assertEqual(len(model.forward_shapes), 1)
        self.assertEqual(model.forward_shapes[0][0], 64)


class TestDetectorHelpers(unittest.TestCase):
    def test_build_detector_candidates_keeps_full_and_trusted_warp(self):
        base_img = Image.new("RGB", (512, 512), (120, 120, 120))
        lens = [{
            "tag": "contour_robust",
            "corners": np.array([[0, 0], [511, 0], [511, 511], [0, 511]], dtype=np.float32),
            "warp_quality": 0.8,
            "trusted": True,
            "relaxed": False,
            "metrics": {"area_ratio": 1.0},
            "lens_confidence": 1.0,
        }]
        metrics = {
            "area_ratio": 0.9,
            "opposite_similarity": 0.95,
            "aspect_similarity": 0.95,
            "min_angle": 89.0,
            "max_angle": 91.0,
        }
        with (
            patch.object(v6.BoardDetector, "lens_hypotheses", return_value=lens),
            patch.object(v6.BoardDetector, "_refine_corners_grid_fit", return_value=(lens[0]["corners"], 0.0)),
            patch.object(v6, "compute_quad_metrics", return_value=metrics),
            patch.object(v6, "perspective_transform", return_value=base_img),
        ):
            out = v6.build_detector_candidates(base_img)
        tags = [item[0] for item in out]
        self.assertIn("full", tags)
        self.assertIn("contour_robust", tags)


class TestOrientationHelpers(unittest.TestCase):
    def test_collect_orientation_context_non_auto_defaults(self):
        base_img = Image.new("RGB", (512, 512), (120, 120, 120))
        context = v6.collect_orientation_context([("full", base_img, 0.0, True)], board_perspective="white")
        self.assertIsNone(context["label_perspective_result"])
        self.assertTrue(context["labels_absent"])
        self.assertFalse(context["labels_same"])

    def test_resolve_candidate_orientation_override(self):
        context = {
            "label_perspective_result": None,
            "label_details": {"left": None, "right": None},
            "labels_absent": True,
            "labels_same": False,
        }
        perspective, source = v6.resolve_candidate_orientation(
            "8/8/8/8/8/8/8/8",
            board_perspective="black",
            orientation_context=context,
        )
        self.assertEqual(perspective, "black")
        self.assertEqual(source, "override")


class TestDecodeHelpers(unittest.TestCase):
    def test_decode_candidate_applies_allowed_rescore_and_orientation(self):
        base_img = Image.new("RGB", (512, 512), (120, 120, 120))
        candidate = ("full", base_img, 0.0, True)
        orientation_context = {
            "label_perspective_result": None,
            "label_details": {"left": None, "right": None},
            "labels_absent": True,
            "labels_same": False,
        }
        with (
            patch.object(
                v6,
                "infer_fen_on_image_clean",
                return_value=("8/8/8/8/8/8/8/K6k", 0.8, 2, {"tile_infos": []}),
            ),
            patch.object(v6, "image_saturation_stats", return_value={"sat_mean": 20.0, "sat_std": 5.0}),
            patch.object(v6, "rescore_low_saturation_sparse_from_topk", return_value=("8/8/8/8/8/8/8/K5k1", 0.81, 2)),
            patch.object(v6, "resolve_candidate_orientation", return_value=("black", "label")),
        ):
            decoded = v6.decode_candidate(
                candidate,
                model=object(),
                device=torch.device("cpu"),
                board_perspective="auto",
                orientation_context=orientation_context,
            )
        self.assertEqual(decoded[0], "full")
        self.assertEqual(decoded[5], "black")
        self.assertEqual(decoded[6], "label")
        self.assertEqual(decoded[2], v6.rotate_fen_180("8/8/8/8/8/8/8/K5k1"))


class TestSelectionHelpers(unittest.TestCase):
    def test_select_best_candidate_prefers_plausible_sparse_board(self):
        base_img = Image.new("RGB", (512, 512), (120, 120, 120))
        scored = [
            ("full", base_img, "8/8/8/8/8/8/8/8", 0.60, 0, "white", "default", 0.0, True),
            ("warp_a", base_img, "8/8/8/8/8/8/8/K6k", 0.70, 8, "white", "default", 0.7, True),
            ("warp_b", base_img, "8/8/8/8/8/8/8/K5k1", 0.55, 2, "white", "default", 0.6, True),
        ]
        best = v6.select_best_candidate(scored)
        self.assertEqual(best[0], "warp_a")


class TestTimeoutSafety(unittest.TestCase):
    def test_worker_timeout_returns_error_payload(self):
        image_path = TRAIN_DIR / "images_4_test" / "puzzle-00050.jpeg"
        payload, elapsed = eval_v6.run_recognizer_once(
            image_path=image_path,
            model_path=str(TRAIN_DIR / "models" / "model_hybrid_v5_latest_best.pt"),
            board_perspective="auto",
            timeout_sec=0.001,
            debug=False,
        )
        self.assertFalse(payload.get("success", False))
        self.assertIn("timeout", str(payload.get("error", "")).lower())
        self.assertLess(elapsed, 5.0)


if __name__ == "__main__":
    unittest.main()
