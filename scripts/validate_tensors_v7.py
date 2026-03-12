#!/usr/bin/env python3
"""Validate generated v7 tensor chunks for shape/range/integrity issues."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch


REQUIRED_KEYS = {
    "x",
    "piece",
    "piece_mask",
    "corners",
    "board_present",
    "pov",
    "stm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate tensors_v7 dataset chunks.")
    parser.add_argument("--data-dir", default="tensors_v7", help="Directory containing train_*.pt / val_*.pt")
    parser.add_argument(
        "--manifest-json",
        default="",
        help="Optional manifest path (defaults to <data-dir>/generation_manifest_v7.json when present)",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on warnings, not only hard errors.")
    return parser.parse_args()


def is_finite_tensor(t: torch.Tensor) -> bool:
    if t.is_floating_point():
        return bool(torch.isfinite(t).all().item())
    return True


def shape_ok(batch: int, t: torch.Tensor, expected_tail: Tuple[int, ...]) -> bool:
    return t.ndim == (1 + len(expected_tail)) and t.shape[0] == batch and tuple(t.shape[1:]) == expected_tail


def summarize_bool(mask: torch.Tensor) -> Dict[str, int]:
    mask = mask.to(torch.int64)
    return {"zero": int((mask == 0).sum().item()), "one": int((mask == 1).sum().item())}


def validate_file(path: Path) -> Tuple[List[str], List[str], Dict[str, float]]:
    errors: List[str] = []
    warnings: List[str] = []
    stats: Dict[str, float] = {
        "samples": 0,
        "board_present_1": 0,
        "board_present_0": 0,
        "pov_unknown": 0,
        "stm_unknown": 0,
        "piece_nonempty_tiles": 0,
        "piece_total_tiles": 0,
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
    piece = payload["piece"]
    piece_mask = payload["piece_mask"]
    corners = payload["corners"]
    board_present = payload["board_present"]
    pov = payload["pov"]
    stm = payload["stm"]

    tensors = {
        "x": x,
        "piece": piece,
        "piece_mask": piece_mask,
        "corners": corners,
        "board_present": board_present,
        "pov": pov,
        "stm": stm,
    }
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            errors.append(f"{path.name}: {name} is not a tensor")
            return errors, warnings, stats
        if not is_finite_tensor(t):
            errors.append(f"{path.name}: {name} contains NaN/Inf")

    batch = int(x.shape[0]) if x.ndim >= 1 else -1
    if batch <= 0:
        errors.append(f"{path.name}: invalid batch size in x: {tuple(x.shape)}")
        return errors, warnings, stats

    if not shape_ok(batch, x, (3, 320, 320)):
        errors.append(f"{path.name}: x shape expected [N,3,320,320], got {tuple(x.shape)}")
    if not shape_ok(batch, piece, (64,)):
        errors.append(f"{path.name}: piece shape expected [N,64], got {tuple(piece.shape)}")
    if not shape_ok(batch, piece_mask, ()):
        errors.append(f"{path.name}: piece_mask shape expected [N], got {tuple(piece_mask.shape)}")
    if not shape_ok(batch, corners, (8,)):
        errors.append(f"{path.name}: corners shape expected [N,8], got {tuple(corners.shape)}")
    if not shape_ok(batch, board_present, ()):
        errors.append(f"{path.name}: board_present shape expected [N], got {tuple(board_present.shape)}")
    if not shape_ok(batch, pov, ()):
        errors.append(f"{path.name}: pov shape expected [N], got {tuple(pov.shape)}")
    if not shape_ok(batch, stm, ()):
        errors.append(f"{path.name}: stm shape expected [N], got {tuple(stm.shape)}")

    if errors:
        return errors, warnings, stats

    if x.dtype != torch.uint8:
        errors.append(f"{path.name}: x dtype expected uint8, got {x.dtype}")
    if piece.dtype != torch.int64:
        errors.append(f"{path.name}: piece dtype expected int64, got {piece.dtype}")
    if piece_mask.dtype not in (torch.float32, torch.float64):
        errors.append(f"{path.name}: piece_mask dtype expected float, got {piece_mask.dtype}")
    if corners.dtype not in (torch.float32, torch.float64):
        errors.append(f"{path.name}: corners dtype expected float, got {corners.dtype}")
    if board_present.dtype not in (torch.float32, torch.float64):
        errors.append(f"{path.name}: board_present dtype expected float, got {board_present.dtype}")
    if pov.dtype != torch.int64:
        errors.append(f"{path.name}: pov dtype expected int64, got {pov.dtype}")
    if stm.dtype != torch.int64:
        errors.append(f"{path.name}: stm dtype expected int64, got {stm.dtype}")

    x_min = int(x.min().item())
    x_max = int(x.max().item())
    if x_min < 0 or x_max > 255:
        errors.append(f"{path.name}: x values out of uint8 range: min={x_min} max={x_max}")

    p_min = int(piece.min().item())
    p_max = int(piece.max().item())
    if p_min < 0 or p_max > 12:
        errors.append(f"{path.name}: piece label out of range [0,12]: min={p_min} max={p_max}")

    if float(corners.min().item()) < 0.0 - 1e-6 or float(corners.max().item()) > 1.0 + 1e-6:
        errors.append(
            f"{path.name}: corners expected normalized [0,1], got min={float(corners.min().item()):.4f} "
            f"max={float(corners.max().item()):.4f}"
        )

    bp_i = board_present.round().to(torch.int64)
    pm_i = piece_mask.round().to(torch.int64)
    if not torch.equal(bp_i, pm_i):
        mismatch = int((bp_i != pm_i).sum().item())
        errors.append(f"{path.name}: piece_mask != board_present in {mismatch} samples")

    invalid_bp = int(((bp_i != 0) & (bp_i != 1)).sum().item())
    if invalid_bp > 0:
        errors.append(f"{path.name}: board_present has non-binary values in {invalid_bp} samples")

    invalid_pov = int(((pov < 0) | (pov > 2)).sum().item())
    invalid_stm = int(((stm < 0) | (stm > 2)).sum().item())
    if invalid_pov > 0:
        errors.append(f"{path.name}: pov contains invalid classes in {invalid_pov} samples")
    if invalid_stm > 0:
        errors.append(f"{path.name}: stm contains invalid classes in {invalid_stm} samples")

    absent = bp_i == 0
    present = bp_i == 1
    if int(absent.sum().item()) > 0:
        bad_abs_piece = int((piece[absent] != 0).any(dim=1).sum().item())
        bad_abs_corners = int((corners[absent].abs().sum(dim=1) > 1e-6).sum().item())
        bad_abs_pov = int((pov[absent] != 2).sum().item())
        bad_abs_stm = int((stm[absent] != 2).sum().item())
        if bad_abs_piece > 0:
            errors.append(f"{path.name}: absent samples with non-empty piece labels: {bad_abs_piece}")
        if bad_abs_corners > 0:
            errors.append(f"{path.name}: absent samples with non-zero corners: {bad_abs_corners}")
        if bad_abs_pov > 0:
            errors.append(f"{path.name}: absent samples with pov != unknown(2): {bad_abs_pov}")
        if bad_abs_stm > 0:
            errors.append(f"{path.name}: absent samples with stm != unknown(2): {bad_abs_stm}")

    if int(present.sum().item()) > 0:
        bad_pres_pov = int(((pov[present] != 0) & (pov[present] != 1)).sum().item())
        bad_pres_stm = int(((stm[present] != 0) & (stm[present] != 1)).sum().item())
        if bad_pres_pov > 0:
            errors.append(f"{path.name}: present samples with pov not in {{0,1}}: {bad_pres_pov}")
        if bad_pres_stm > 0:
            errors.append(f"{path.name}: present samples with stm not in {{0,1}}: {bad_pres_stm}")
        zero_piece_on_present = int((piece[present].sum(dim=1) == 0).sum().item())
        if zero_piece_on_present > 0:
            warnings.append(f"{path.name}: present samples with all-empty piece labels: {zero_piece_on_present}")

    stats["samples"] = batch
    stats["board_present_1"] = int((bp_i == 1).sum().item())
    stats["board_present_0"] = int((bp_i == 0).sum().item())
    stats["pov_unknown"] = int((pov == 2).sum().item())
    stats["stm_unknown"] = int((stm == 2).sum().item())
    stats["piece_nonempty_tiles"] = int((piece != 0).sum().item())
    stats["piece_total_tiles"] = int(piece.numel())
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

    manifest_path = Path(args.manifest_json) if args.manifest_json else (data_dir / "generation_manifest_v7.json")
    manifest = load_manifest(manifest_path)

    hard_errors: List[str] = []
    warnings: List[str] = []
    agg = {
        "samples": 0,
        "board_present_1": 0,
        "board_present_0": 0,
        "pov_unknown": 0,
        "stm_unknown": 0,
        "piece_nonempty_tiles": 0,
        "piece_total_tiles": 0,
    }

    for fp in all_files:
        e, w, s = validate_file(fp)
        hard_errors.extend(e)
        warnings.extend(w)
        for k in agg:
            agg[k] += int(s[k])

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

    print("=== V7 TENSOR VALIDATION ===")
    print(f"data_dir={data_dir}")
    print(f"train_chunks={len(train_files)} val_chunks={len(val_files)} total_chunks={len(all_files)}")
    print(f"samples={agg['samples']} present={agg['board_present_1']} absent={agg['board_present_0']}")
    print(
        "unknown_labels: "
        f"pov={agg['pov_unknown']} stm={agg['stm_unknown']}"
    )
    density = (
        float(agg["piece_nonempty_tiles"]) / float(max(1, agg["piece_total_tiles"]))
        if agg["piece_total_tiles"] > 0
        else 0.0
    )
    print(
        "piece_density: "
        f"{agg['piece_nonempty_tiles']}/{agg['piece_total_tiles']} ({density:.4f})"
    )
    if manifest is not None:
        preset = manifest.get("dataset_preset", "?")
        recipe = manifest.get("recipe_name", "?")
        print(f"manifest={manifest_path} preset={preset} recipe={recipe}")
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

    print("\nOK: tensors_v7 passed integrity validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
