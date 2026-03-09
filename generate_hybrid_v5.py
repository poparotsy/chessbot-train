#!/usr/bin/env python3
"""v5 dataset generator wrapper with targeted defaults and v5 output directory."""

import os

import numpy as np
import torch

import generate_hybrid_v4 as base


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tensors_v5")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    for name in [f"val_{i}" for i in range(base.CHUNKS_VAL)] + [f"train_{i}" for i in range(base.CHUNKS_TRAIN)]:
        all_x, all_y = [], []
        profile_counts = {k: 0 for k in base.PROFILE_OVERRIDES.keys()}
        for _ in range(base.BOARDS_PER_CHUNK):
            profile = base.choose_profile()
            profile_counts[profile] += 1
            board = base.random_training_board(profile=profile)
            tiles, labels = base.render_board(board.fen().split()[0], profile=profile)
            all_x.extend(tiles)
            all_y.extend(labels)
        payload = {
            "x": torch.from_numpy(np.stack(all_x)),
            "y": torch.tensor(all_y),
        }
        torch.save(payload, os.path.join(OUTPUT_DIR, f"{name}.pt"))
        mix = ", ".join(f"{k}:{v}" for k, v in profile_counts.items())
        print(f"✅ Created {name} | mix[{mix}]")


if __name__ == "__main__":
    main()
