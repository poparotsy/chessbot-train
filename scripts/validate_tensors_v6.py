#!/usr/bin/env python3
"""Validate generated v6 tensor chunks for shape/range/integrity issues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch


REQUIRED_KEYS = {"x", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate tensors_v6 dataset chunks.")
    parser.add_argument("--data-dir", default="tensors_v6", help="Directory containing train_*.pt / val_*.pt")
    parser.add_argument(
        "--manifest-json",
        default="",
        help="Optional manifest path (defaults to <data-dir>/generation_manifest_v6.json when present)",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on warnings, not only hard errors.")
    return parser.parse_args()


def validate_file(path: Path) -> Tuple[List[str], List[str], Dict[str, int]]:
    errors: List[str] = []
    warnings: List[str] = []
    stats = {
        "samples": 0,
        "empty": 0,
        "nonempty": 0,
    }
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return [f"{path.name}: torch.load failed: {exc}"], warnings, stats

    if not isinstance(payload, dict):
        return [f"{path.name}: payload is not a dict"], warnings, stats
    missing = REQUIRED_KEYS - set(payload.keys())
    if missing:
        return [f"{path.name}: missing keys: {sorted(missing)}"], warnings, stats

    x = payload["x"]
    y = payload["y"]
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        return [f"{path.name}: x/y must both be torch tensors"], warnings, stats

    if x.ndim != 4 or tuple(x.shape[1:]) != (3, 64, 64):
        errors.append(f"{path.name}: x shape expected [N,3,64,64], got {tuple(x.shape)}")
    if y.ndim != 1:
        errors.append(f"{path.name}: y shape expected [N], got {tuple(y.shape)}")
    if x.ndim >= 1 and y.ndim >= 1 and x.shape[0] != y.shape[0]:
        errors.append(f"{path.name}: batch mismatch x={x.shape[0]} y={y.shape[0]}")
    if errors:
        return errors, warnings, stats

    if x.dtype != torch.uint8:
        errors.append(f"{path.name}: x dtype expected uint8, got {x.dtype}")
    if y.dtype != torch.int64:
        errors.append(f"{path.name}: y dtype expected int64, got {y.dtype}")

    if x.is_floating_point() and not torch.isfinite(x).all().item():
        errors.append(f"{path.name}: x contains NaN/Inf")
    if y.is_floating_point() and not torch.isfinite(y).all().item():
        errors.append(f"{path.name}: y contains NaN/Inf")

    x_min = int(x.min().item())
    x_max = int(x.max().item())
    if x_min < 0 or x_max > 255:
        errors.append(f"{path.name}: x values out of uint8 range: min={x_min} max={x_max}")

    y_min = int(y.min().item())
    y_max = int(y.max().item())
    if y_min < 0 or y_max > 12:
        errors.append(f"{path.name}: y labels out of range [0,12]: min={y_min} max={y_max}")

    empty = int((y == 0).sum().item())
    nonempty = int(y.numel() - empty)
    stats["samples"] = int(y.numel())
    stats["empty"] = empty
    stats["nonempty"] = nonempty

    if nonempty == 0:
        warnings.append(f"{path.name}: all labels are empty")

    return errors, warnings, stats


def load_manifest(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data dir not found: {data_dir}")
        return 2

    train_files = sorted(data_dir.glob("train_*.pt"))
    val_files = sorted(data_dir.glob("val_*.pt"))
    all_files = train_files + val_files
    if not all_files:
        print(f"ERROR: no train_*.pt or val_*.pt found in {data_dir}")
        return 2

    manifest_path = Path(args.manifest_json) if args.manifest_json else (data_dir / "generation_manifest_v6.json")
    manifest = load_manifest(manifest_path)

    hard_errors: List[str] = []
    warnings: List[str] = []
    agg = {"samples": 0, "empty": 0, "nonempty": 0}

    per_file_samples: Dict[str, int] = {}
    for fp in all_files:
        e, w, s = validate_file(fp)
        hard_errors.extend(e)
        warnings.extend(w)
        for key in agg:
            agg[key] += int(s[key])
        per_file_samples[fp.name] = int(s["samples"])

    if manifest is not None:
        exp_train = manifest.get("chunks_train")
        exp_val = manifest.get("chunks_val")
        if isinstance(exp_train, int) and len(train_files) != exp_train:
            hard_errors.append(
                f"manifest mismatch: expected {exp_train} train chunks, found {len(train_files)}"
            )
        if isinstance(exp_val, int) and len(val_files) != exp_val:
            hard_errors.append(
                f"manifest mismatch: expected {exp_val} val chunks, found {len(val_files)}"
            )
        expected_samples = manifest.get("expected_samples_per_chunk")
        if isinstance(expected_samples, int) and expected_samples > 0:
            for file_name, got in per_file_samples.items():
                if got != expected_samples:
                    hard_errors.append(
                        f"{file_name}: expected {expected_samples} samples (manifest), got {got}"
                    )

    density = float(agg["nonempty"]) / float(max(1, agg["samples"]))
    print("=== V6 TENSOR VALIDATION ===")
    print(f"data_dir={data_dir}")
    print(f"train_chunks={len(train_files)} val_chunks={len(val_files)} total_chunks={len(all_files)}")
    print(f"samples={agg['samples']} empty={agg['empty']} nonempty={agg['nonempty']}")
    print(f"nonempty_density={agg['nonempty']}/{agg['samples']} ({density:.4f})")
    if manifest is not None:
        recipe = manifest.get("recipe_name", "?")
        mix = manifest.get("profile_weights", {})
        print(f"manifest={manifest_path} recipe={recipe} profile_weights={mix}")
    else:
        print(f"manifest=missing_or_invalid ({manifest_path})")

    if warnings:
        print("\nWARNINGS:")
        for msg in warnings[:40]:
            print(f"- {msg}")
        if len(warnings) > 40:
            print(f"- ... {len(warnings) - 40} more")

    if hard_errors:
        print("\nERRORS:")
        for msg in hard_errors[:60]:
            print(f"- {msg}")
        if len(hard_errors) > 60:
            print(f"- ... {len(hard_errors) - 60} more")
        return 2

    if args.strict and warnings:
        print("\nSTRICT MODE: warnings treated as failure.")
        return 3

    print("\nOK: tensors_v6 passed integrity validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
