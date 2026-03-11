#!/usr/bin/env python3
"""Rank model files on hardset using recognizer_v7 defaults."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
BASE_SCRIPT = SCRIPT_DIR / "rank_models_hardset.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v7 wrapper for rank_models_hardset.py")
    parser.add_argument("--models-glob", default="models/model_hybrid_v7*.pt")
    parser.add_argument("--images-dir", default=str(ROOT_DIR / "images_4_test"))
    parser.add_argument("--truth-json", default=str(ROOT_DIR / "images_4_test" / "truth_verified.json"))
    parser.add_argument("--with-debug", action="store_true")
    parser.add_argument("--compare-full-fen", action="store_true")
    parser.add_argument(
        "--recognizer-module",
        default="recognizer_v7",
        help="Override recognizer module (default recognizer_v7).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        str(BASE_SCRIPT),
        "--models-glob",
        args.models_glob,
        "--images-dir",
        args.images_dir,
        "--truth-json",
        args.truth_json,
        "--recognizer-module",
        args.recognizer_module,
    ]
    if args.with_debug:
        cmd.append("--with-debug")
    if args.compare_full_fen:
        cmd.append("--compare-full-fen")
    proc = subprocess.run(cmd, cwd=str(ROOT_DIR))
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
