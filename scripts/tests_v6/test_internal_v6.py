import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent.parent
SCRIPTS_DIR = TRAIN_DIR / "scripts"
for path in (TRAIN_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import recognizer_v6 as v6
import evaluate_v6_hardset as eval_v6


class TestSideToMoveAliases(unittest.TestCase):
    def test_side_to_move_aliases(self):
        white_aliases = ["w", "white", "wtm", "white_to_move"]
        black_aliases = ["b", "black", "blak", "btm", "black_to_move"]
        for alias in white_aliases:
            self.assertEqual(v6.parse_side_to_move_override(alias), "w")
        for alias in black_aliases:
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
            self.assertEqual(v6.v4.rotate_fen_180(v6.v4.rotate_fen_180(fen_board)), fen_board)


class TestBatchingInvariant(unittest.TestCase):
    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.forward_shapes = []

        def forward(self, x):
            self.forward_shapes.append(tuple(x.shape))
            return torch.zeros((x.shape[0], 13), dtype=torch.float32, device=x.device)

    def test_deep_topk_batches_all_tiles_single_forward(self):
        model = self.CountingModel()
        device = torch.device("cpu")
        img = Image.new("RGB", (512, 512), (127, 127, 127))
        v6.infer_fen_on_image_deep_topk(
            img,
            model,
            device,
            use_square_detection=False,
            return_details=False,
        )
        self.assertEqual(len(model.forward_shapes), 1)
        self.assertEqual(model.forward_shapes[0][0], 64)


class TestCandidateCap(unittest.TestCase):
    def test_predict_board_caps_decoded_candidates(self):
        decode_calls = {"n": 0}

        def fake_infer(*_args, **_kwargs):
            decode_calls["n"] += 1
            return "8/8/8/8/8/8/8/8", 0.5, 0, {"tile_infos": [], "confidence_summary": {}}

        class DummyModel:
            def to(self, _device):
                return self

            def load_state_dict(self, _state):
                return self

            def eval(self):
                return self

        base_img = Image.new("RGB", (512, 512), (120, 120, 120))
        candidates = [("full", base_img, 0.0, True)] + [
            (f"cand{i}", base_img, 0.5, True) for i in range(30)
        ]

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            base_img.save(tmp.name)
            with (
                patch.object(v6, "MAX_DECODE_CANDIDATES", 7),
                patch.object(v6.BoardDetector, "candidate_images", return_value=candidates),
                patch.object(v6, "infer_fen_on_image_deep_topk", side_effect=fake_infer),
                patch.object(v6.v4, "MODEL_PATH", "dummy.pt"),
                patch.object(v6.v4, "StandaloneBeastClassifier", return_value=DummyModel()),
                patch.object(v6.torch, "load", return_value={}),
                patch.object(v6.v4, "infer_board_perspective_from_labels", return_value=None),
                patch.object(v6.v4, "infer_board_perspective_from_piece_distribution", return_value="white"),
            ):
                v6.predict_board(tmp.name, model_path="dummy.pt", board_perspective="auto")
        self.assertLessEqual(decode_calls["n"], 7)


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


class TestOrientationBestGuess(unittest.TestCase):
    def test_low_signal_position_returns_none(self):
        fen_board = "8/8/8/8/K7/8/pp1Q4/k7"
        guess, details = v6.orientation_best_guess(fen_board)
        self.assertIsNone(guess)
        self.assertIn("reason", details)

    def test_high_signal_position_produces_guess(self):
        fen_board = "3r4/3kn2p/1p4pP/1p1B1pP1/1P1K1P2/P1P5/8/4R3"
        guess, details = v6.orientation_best_guess(fen_board, min_margin=0.0)
        self.assertIn(guess, {"white", "black"})
        self.assertIn("margin", details)


if __name__ == "__main__":
    unittest.main()
