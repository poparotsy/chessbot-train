#!/usr/bin/env python3
"""v5 training wrapper that defaults to tensors_v5 and v5 checkpoint names."""

import os


def _set_default_env(name, value):
    if os.getenv(name) is None:
        os.environ[name] = value


_set_default_env("DATA_DIR", "tensors_v5")
_set_default_env("MODEL_SAVE_PATH", "models/model_hybrid_v5_latest_best.pt")
_set_default_env("FINAL_MODEL_SAVE_PATH", "models/model_hybrid_v5_final.pt")
_set_default_env("BASE_MODEL_PATH", "models/model_hybrid_v4_300e_last_best.pt")
_set_default_env("CHECKPOINT_DIR", "models/checkpoints_v5")

import train_hybrid_v4 as base


if __name__ == "__main__":
    base.train()
