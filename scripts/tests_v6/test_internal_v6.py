import unittest
import json
from pathlib import Path
from unittest.mock import patch

import sys

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent.parent
SCRIPTS_DIR = TRAIN_DIR / "scripts"
ARCHIVE_DIR = TRAIN_DIR / "archive" / "recognizer_legacy"
for path in (TRAIN_DIR, SCRIPTS_DIR, ARCHIVE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_v6_hardset as eval_v6
import compare_deep_diagnostic_reports_v6 as deep_diff
import deep_diagnostic_v6 as deep_diag
import recognizer_v6 as v6
import recognizer_v6_fullguard as v6_fullguard


class TestDomainSuiteData(unittest.TestCase):
    def test_domain_suite_cases_exist_in_truth_and_are_non_empty(self):
        truth = eval_v6.load_truth(TRAIN_DIR / "images_4_test" / "truth_verified.json")
        suite_path = SCRIPTS_DIR / "testdata" / "v6_domain_cases.json"
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        self.assertTrue(suite)
        for category, names in suite.items():
            self.assertTrue(category)
            self.assertTrue(names)
            for name in names:
                self.assertIn(name, truth)

    def test_fullguard_suite_cases_exist_in_truth_and_are_non_empty(self):
        truth = eval_v6.load_truth(TRAIN_DIR / "images_4_test" / "truth_verified.json")
        suite_path = SCRIPTS_DIR / "testdata" / "v6_fullguard_cases.json"
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        self.assertTrue(suite)
        for category, names in suite.items():
            self.assertTrue(category)
            self.assertTrue(names)
            for name in names:
                self.assertIn(name, truth)


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
        with (
            patch.object(
                v6,
                "score_full_frame_board",
                return_value={"score": 0.95, "trusted": True, "support_ratio": 1.0, "coverage": 0.99, "evidence": 0.70},
            ),
            patch.object(
                v6,
                "detect_board_grid",
                return_value=[{
                    "tag": "contour_robust",
                    "corners": np.array([[0, 0], [511, 0], [511, 511], [0, 511]], dtype=np.float32),
                    "score": 0.80,
                    "trusted": True,
                    "support_ratio": 0.88,
                }],
            ),
            patch.object(v6, "perspective_transform", return_value=base_img),
            patch.object(v6, "crop_warp_to_detected_grid", return_value=None),
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
        self.assertEqual(context["partial_label_scores"], {"white": 0.0, "black": 0.0})

    def test_resolve_candidate_orientation_override(self):
        context = {
            "label_perspective_result": None,
            "label_details": {"left": None, "right": None},
            "labels_absent": True,
            "labels_same": False,
            "partial_label_scores": {"white": 0.0, "black": 0.0},
        }
        perspective, source = v6.resolve_candidate_orientation(
            "8/8/8/8/8/8/8/8",
            board_perspective="black",
            orientation_context=context,
        )
        self.assertEqual(perspective, "black")
        self.assertEqual(source, "override")

    def test_resolve_candidate_orientation_best_effort_flips_black_poster_case(self):
        raw_fen = "K1B5/P1P3P1/2P1r2P/2n5/1qpp2p1/p7/kp5p/2R2Q2"
        context = {
            "label_perspective_result": None,
            "label_details": {
                "left": None,
                "right": {"label": "h", "confidence": 0.6296536796536797},
            },
            "labels_absent": False,
            "labels_same": False,
            "partial_label_scores": {"white": 0.5981709956709956, "black": 0.0},
        }
        perspective, source = v6.resolve_candidate_orientation(
            raw_fen,
            board_perspective="auto",
            orientation_context=context,
        )
        self.assertEqual(perspective, "black")
        self.assertEqual(source, "best_effort_orientation")

    def test_resolve_candidate_orientation_best_effort_keeps_white_when_labels_support_it(self):
        """Test that default is returned when labels are below confidence threshold."""
        raw_fen = "K4BB1/1Q6/5p2/8/2R2r1r/N2N2q1/kp1p1p1p/b7"
        context = {
            "label_perspective_result": None,
            "label_details": {
                "left": {"label": "a", "confidence": 0.7618785578747628},
                "right": {"label": "h", "confidence": 0.6657495256166983},
            },
            "labels_absent": False,
            "labels_same": False,
            "partial_label_scores": {"white": 1.356246679316888, "black": 0.0},
        }
        perspective, source = v6.resolve_candidate_orientation(
            raw_fen,
            board_perspective="auto",
            orientation_context=context,
        )
        # Right label confidence (0.665) < threshold (0.70), so no weak_label_fallback
        # Returns default since no other fallback path triggers
        self.assertEqual(perspective, "white")
        self.assertEqual(source, "default")


class TestDecodeHelpers(unittest.TestCase):
    def test_decode_candidate_applies_orientation_to_full_candidate(self):
        base_img = Image.new("RGB", (512, 512), (120, 120, 120))
        candidate = ("full", base_img, 0.0, True, 0.9, 1.0)
        orientation_context = {
            "label_perspective_result": None,
            "label_details": {"left": None, "right": None},
            "labels_absent": True,
            "labels_same": False,
            "partial_label_scores": {"white": 0.0, "black": 0.0},
        }
        with (
            patch.object(
                v6,
                "infer_fen_on_image_clean",
                side_effect=[
                    ("8/8/8/8/8/8/8/K6k", 0.8, 2),
                    ("8/8/8/8/8/8/8/K6k", 0.8, 2, {"tile_infos": []}),
                    ("8/8/8/8/8/8/8/K5k1", 0.81, 2, {"tile_infos": []}),
                ],
            ),
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
        self.assertEqual(decoded[2], v6.rotate_fen_180("8/8/8/8/8/8/8/K6k"))


class TestSelectionHelpers(unittest.TestCase):
    def test_select_best_candidate_returns_first_scored_entry(self):
        base_img = Image.new("RGB", (512, 512), (120, 120, 120))
        scored = [
            ("full", base_img, "8/8/8/8/8/8/8/8", 0.60, 0, "white", "default", 0.0, True),
            ("warp_a", base_img, "8/8/8/8/8/8/8/K6k", 0.70, 8, "white", "default", 0.7, True),
            ("warp_b", base_img, "8/8/8/8/8/8/8/K5k1", 0.55, 2, "white", "default", 0.6, True),
        ]
        best = v6.select_best_candidate(scored)
        self.assertEqual(best[0], "full")


class TestDeepDiagnosticHelpers(unittest.TestCase):
    def test_classify_failure_path(self):
        self.assertEqual(
            deep_diag.classify_failure_path(chosen_ok=False, full_ok=True, full_available=True),
            "recognizer_selection_failure",
        )
        self.assertEqual(
            deep_diag.classify_failure_path(chosen_ok=False, full_ok=False, full_available=True),
            "model_or_full_crop_failure",
        )
        self.assertEqual(
            deep_diag.classify_failure_path(chosen_ok=True, full_ok=True, full_available=True),
            "no_board_failure",
        )

    def test_diff_squares(self):
        truth = "r7/8/8/8/8/8/P7/7R"
        actual = "8/8/8/8/8/8/8/7R"
        self.assertEqual(deep_diag.diff_squares(truth, actual), ["a2", "a8"])


class TestDeepDiagnosticDiffHelpers(unittest.TestCase):
    def test_status_rank(self):
        chosen_ok = {"current_model": {"chosen_board_ok": True, "full_board_ok": True}}
        full_only = {"current_model": {"chosen_board_ok": False, "full_board_ok": True}}
        full_fail = {"current_model": {"chosen_board_ok": False, "full_board_ok": False}}
        self.assertEqual(deep_diff.status_rank(chosen_ok), 2)
        self.assertEqual(deep_diff.status_rank(full_only), 1)
        self.assertEqual(deep_diff.status_rank(full_fail), 0)


class TestFullGuardHelpers(unittest.TestCase):
    def test_edge_drop_metrics_counts_deleted_edge_pieces(self):
        full = "r6r/p6p/8/8/8/8/P6P/R6R"
        partial = "8/8/8/8/8/8/8/8"
        edge_drop, file_drop, rank_drop, edge_mismatch = v6_fullguard._edge_drop_metrics(full, partial)
        self.assertEqual(edge_drop, 8)
        self.assertEqual(file_drop, 8)
        self.assertEqual(rank_drop, 4)
        self.assertEqual(edge_mismatch, 8)

    def test_select_best_candidate_prefers_full_on_same_fen_near_tie(self):
        base_img = Image.new("RGB", (512, 512), (120, 120, 120))
        same_fen = "k4r2/p7/P6p/2p3p1/5q2/4pB2/1R6/1R2K3"
        scored = [
            ("full", base_img, same_fen, 0.7828, 13, "white", "default", 0.0, True),
            ("contour_robust", base_img, same_fen, 0.8410, 13, "white", "default", 0.8, True),
        ]
        best = v6_fullguard.select_best_candidate(scored)
        self.assertEqual(best[0], "full")

    def test_compute_dark_edge_trim_box_identity_on_bright_image(self):
        img = Image.new("RGB", (256, 256), (180, 180, 180))
        trim_box = v6_fullguard.compute_dark_edge_trim_box(img)
        self.assertFalse(trim_box["trimmed"])
        self.assertEqual((trim_box["x0"], trim_box["y0"], trim_box["x1"], trim_box["y1"]), (0, 0, 256, 256))

    def test_full_refinement_variants_include_base(self):
        img = Image.new("RGB", (256, 256), (120, 120, 120))
        variants = v6_fullguard._build_full_refinement_variants(img)
        names = [item["variant"] for item in variants]
        self.assertIn("base", names)

    def test_should_refine_full_trim_requires_strong_trim_pattern(self):
        self.assertFalse(
            v6_fullguard.should_refine_full_trim(
                {"trimmed": True, "left": 0, "top": 0, "right": 0, "bottom": 18}
            )
        )
        self.assertFalse(
            v6_fullguard.should_refine_full_trim(
                {"trimmed": True, "left": 142, "top": 0, "right": 0, "bottom": 0}
            )
        )
        self.assertTrue(
            v6_fullguard.should_refine_full_trim(
                {"trimmed": True, "left": 59, "top": 264, "right": 61, "bottom": 185}
            )
        )


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
