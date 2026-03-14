#!/usr/bin/env python3
"""Fast side-by-side board parity check for recognizer_v6 variants.

Loads the model once per recognizer module and evaluates board FEN only
against images_4_test/truth_verified.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare board parity between two recognizer scripts.")
    parser.add_argument("--images-dir", default="images_4_test")
    parser.add_argument("--truth-json", default="images_4_test/truth_verified.json")
    parser.add_argument("--model-path", default="models/model_hybrid_v5_latest_best.pt")
    parser.add_argument("--script-a", default="recognizer_v6.py")
    parser.add_argument("--script-b", default="recognizer_v6_clean.py")
    parser.add_argument("--output-json", default="reports/v6_clean_parity_fast.json")
    return parser.parse_args()


def load_module(script_path: Path):
    module_name = f"recognizer_fast_{script_path.stem}_{abs(hash(str(script_path.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, str(script_path.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load recognizer module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_script(
    script_path: Path,
    model_path: Path,
    images_dir: Path,
    truth: dict[str, str],
) -> dict:
    mod = load_module(script_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mod.StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.eval()

    board_pass = 0
    failures = []
    for image_name in sorted(truth):
        expected_board = truth[image_name].split()[0]
        img = Image.open(images_dir / image_name).convert("RGB")
        candidates = mod.build_detector_candidates(img)
        orientation_context = mod.collect_orientation_context(candidates, board_perspective="auto")
        scored = [
            mod.decode_candidate(
                candidate,
                model=model,
                device=device,
                board_perspective="auto",
                orientation_context=orientation_context,
            )
            for candidate in candidates
        ]
        selected = mod.select_best_candidate(scored)
        predicted_board = selected[2]
        confidence = float(selected[3])
        if predicted_board == expected_board:
            board_pass += 1
        else:
            failures.append(
                {
                    "image": image_name,
                    "predicted_board": predicted_board,
                    "expected_board": expected_board,
                    "confidence": round(confidence, 4),
                }
            )

    total = len(truth)
    return {
        "script": str(script_path),
        "board_pass": board_pass,
        "images": total,
        "failures": failures,
    }


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir)
    truth_json = Path(args.truth_json)
    model_path = Path(args.model_path)
    script_a = Path(args.script_a)
    script_b = Path(args.script_b)
    output_json = Path(args.output_json)

    truth = json.loads(truth_json.read_text(encoding="utf-8"))
    if not isinstance(truth, dict) or not truth:
        raise ValueError(f"truth JSON must be non-empty object: {truth_json}")

    result_a = evaluate_script(script_a, model_path, images_dir, truth)
    result_b = evaluate_script(script_b, model_path, images_dir, truth)
    payload = {
        "truth_json": str(truth_json),
        "images_dir": str(images_dir),
        "model_path": str(model_path),
        "results": [result_a, result_b],
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{result_a['script']} -> {result_a['board_pass']}/{result_a['images']}")
    print(f"{result_b['script']} -> {result_b['board_pass']}/{result_b['images']}")
    print(f"report={output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
