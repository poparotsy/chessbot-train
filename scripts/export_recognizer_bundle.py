#!/usr/bin/env python3
"""Export the current deployable recognizer bundle to the parent chessbot repo."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
CHESSBOT_ROOT = TRAIN_DIR.parent
DEFAULT_OUTPUT_DIR = CHESSBOT_ROOT / "recognizer"
DEFAULT_MODEL_PATH = TRAIN_DIR / "models" / "model_hybrid_v6_targeted_recovery_v14_latest_best_2.pt"
DEFAULT_EXPORTED_MODEL_NAME = "model_hybrid_v6_prod_v14r2_50h_108s.pt"
DEFAULT_REQUIREMENTS = TRAIN_DIR / "recognizer" / "requirements.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export canonical recognizer bundle for deployment.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recognizer-path", type=Path, default=TRAIN_DIR / "recognizer_v6.py")
    parser.add_argument("--requirements-path", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--exported-model-name", default=DEFAULT_EXPORTED_MODEL_NAME)
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def copy_file(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prune_stale_bundle_models(out_dir: Path, keep_model_name: str):
    for model_path in out_dir.glob("*.pt"):
        if model_path.name != keep_model_name:
            model_path.unlink()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    recognizer_dst = out_dir / "recognizer_v6.py"
    requirements_dst = out_dir / "requirements.txt"
    model_dst = out_dir / args.exported_model_name

    prune_stale_bundle_models(out_dir, model_dst.name)
    copy_file(args.recognizer_path.resolve(), recognizer_dst)
    copy_file(args.requirements_path.resolve(), requirements_dst)
    copy_file(args.model_path.resolve(), model_dst)

    manifest = {
        "source_train_dir": str(TRAIN_DIR),
        "recognizer_path": str(args.recognizer_path.resolve()),
        "requirements_path": str(args.requirements_path.resolve()),
        "model_path": str(args.model_path.resolve()),
        "exported_model_name": model_dst.name,
        "output_dir": str(out_dir),
        "exported_files": [
            recognizer_dst.name,
            requirements_dst.name,
            model_dst.name,
        ],
    }

    if args.write_manifest:
        (out_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"output_dir={out_dir}")
    print(f"recognizer={recognizer_dst}")
    print(f"requirements={requirements_dst}")
    print(f"model={model_dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
