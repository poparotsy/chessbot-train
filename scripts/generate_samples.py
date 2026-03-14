#!/usr/bin/env python3
"""Generate visual sample boards using the selected training distribution."""

import argparse
import importlib
import random
from pathlib import Path

import numpy as np
from PIL import Image

import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


GENERATOR_MODULES = {
    "v5": "generate_hybrid_v5",
    "v6": "generate_hybrid_v6",
}


def load_generator(version):
    module_name = GENERATOR_MODULES[version]
    return importlib.import_module(module_name)


def normalize_profile(profile):
    if profile in (None, "", "default", "mixed-default", "mixed_default"):
        return None
    return profile.replace("-", "_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate sample board images with the selected training distribution.")
    parser.add_argument(
        "--version",
        choices=tuple(GENERATOR_MODULES.keys()),
        default="v6",
        help="Generator version to preview.")
    parser.add_argument("--count", type=int, default=12, help="Number of samples to generate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save sample images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--profile",
        default="default",
        help="Profile to preview. Use 'default' for the generator's mixed default recipe.")
    parser.add_argument(
        "--force-piece-tilt",
        action="store_true",
        help="Force local piece tilt to appear on all samples (preview-only override).")
    parser.add_argument(
        "--force-piece-occlusion",
        action="store_true",
        help="Force piece-square occlusion overlays on all samples (preview-only override).")
    parser.add_argument(
        "--tilt-max-deg",
        type=float,
        default=None,
        help="Override local piece tilt max degrees (preview-only).")
    return parser.parse_args()


def validate_profile(gen, profile):
    if profile is None:
        return
    valid = set(getattr(gen, "PROFILE_OVERRIDES", {}).keys())
    if profile not in valid:
        names = ", ".join(sorted(valid))
        raise SystemExit(f"Unknown profile '{profile}'. Valid profiles: default, {names}")


def render_board_image(gen, fen_board, profile=None):
    tiles, _labels, meta = gen.render_board(fen_board, return_meta=True, profile=profile)
    rows = []
    for row_idx in range(8):
        row_tiles = []
        for col_idx in range(8):
            tile = tiles[row_idx * 8 + col_idx].transpose(1, 2, 0)
            row_tiles.append(tile)
        rows.append(np.concatenate(row_tiles, axis=1))
    background = Image.fromarray(np.concatenate(rows, axis=0))
    return background, meta["board_theme"], meta["piece_set"], meta.get("label_pov")


def apply_preview_overrides(gen, args, profile):
    # Preview-only knobs so users can visually inspect augmentation behavior.
    targets = [profile] if profile is not None else list(gen.PROFILE_OVERRIDES.keys())

    if args.force_piece_tilt:
        gen.BASE_CONFIG["LOCAL_PIECE_TILT_PROB"] = 1.0
        for name in targets:
            gen.PROFILE_OVERRIDES.setdefault(name, {})
            gen.PROFILE_OVERRIDES[name]["LOCAL_PIECE_TILT_PROB"] = 1.0

    if args.force_piece_occlusion:
        gen.BASE_CONFIG["PIECE_OCCLUSION_PROB"] = 1.0
        for name in targets:
            gen.PROFILE_OVERRIDES.setdefault(name, {})
            gen.PROFILE_OVERRIDES[name]["PIECE_OCCLUSION_PROB"] = 1.0

    if args.tilt_max_deg is not None:
        gen.BASE_CONFIG["LOCAL_PIECE_TILT_MAX_DEG"] = args.tilt_max_deg
        for name in targets:
            gen.PROFILE_OVERRIDES.setdefault(name, {})
            gen.PROFILE_OVERRIDES[name]["LOCAL_PIECE_TILT_MAX_DEG"] = args.tilt_max_deg


def main():
    args = parse_args()
    random.seed(args.seed)
    gen = load_generator(args.version)
    profile = normalize_profile(args.profile)
    validate_profile(gen, profile)
    apply_preview_overrides(gen, args, profile)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = ROOT_DIR / f"sample_boards_{args.version}"

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    recipe_name = getattr(gen, "RECIPE_NAME", "(none)")
    default_weights = getattr(gen, "DEFAULT_PROFILE_WEIGHTS", [])

    print(f"Generating {args.count} {args.version} sample boards into: {output_dir}")
    print(
        "Profile="
        f"{args.profile} "
        f"(mode={'mixed-default' if profile is None else profile})"
    )
    if profile is None and default_weights:
        print(f"Recipe={recipe_name}")
        print(f"Weights={default_weights}")
    if args.force_piece_tilt or args.force_piece_occlusion or args.tilt_max_deg is not None:
        print(
            "Preview overrides: "
            f"force_piece_tilt={args.force_piece_tilt}, "
            f"force_piece_occlusion={args.force_piece_occlusion}, "
            f"tilt_max_deg={args.tilt_max_deg if args.tilt_max_deg is not None else 'profile-default'}"
        )
    for i in range(args.count):
        fen_board = gen.random_training_board(profile=profile).board_fen()
        image, board_theme, piece_set, label_pov = render_board_image(gen, fen_board, profile=profile)
        out_path = output_dir / f"sample_{args.version}_{i + 1:03d}.png"
        image.save(out_path)
        manifest.append(
            {
                "file": out_path.name,
                "fen_board": fen_board,
                "board_theme": board_theme,
                "piece_set": piece_set,
                "version": args.version,
                "profile": args.profile,
                "label_pov": label_pov,
            })
        print(
            f"  ✅ {out_path.name} | theme={board_theme} | pieces={piece_set} | "
            f"labels={label_pov or 'none'}"
        )

    manifest_path = output_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(
                f"{item['file']} | fen={item['fen_board']} | theme={item['board_theme']} | "
                f"pieces={item['piece_set']} | version={item['version']} | "
                f"profile={item['profile']} | labels={item['label_pov']}\n")

    print(f"\nSaved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
