#!/usr/bin/env python3
"""Export selected model checkpoints from chessbot-train into the parent chessbot repo."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
CHESSBOT_ROOT = TRAIN_DIR.parent
DEFAULT_SOURCE_DIR = TRAIN_DIR / "models"
DEFAULT_OUTPUT_DIR = CHESSBOT_ROOT / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model checkpoints for local deployment/storage.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        default=[],
        help="Checkpoint filename to export. Repeat to export a subset. Default: export all .pt files.",
    )
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def list_models(source_dir: Path, requested: list[str]) -> list[Path]:
    if requested:
        models = [source_dir / name for name in requested]
    else:
        models = sorted(source_dir.glob("*.pt"))
    missing = [str(path) for path in models if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))
    return models


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list_models(source_dir, list(args.models))
    exported = []
    for model_path in models:
        dst = output_dir / model_path.name
        copy_file(model_path, dst)
        exported.append(dst.name)

    manifest = {
        "source_models_dir": str(source_dir),
        "output_dir": str(output_dir),
        "exported_models": exported,
    }
    if args.write_manifest:
        (output_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"output_dir={output_dir}")
    print(f"models={len(exported)}")
    for name in exported:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
