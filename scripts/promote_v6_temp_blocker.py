#!/usr/bin/env python3
"""Promote a temp inbox image into the canonical blocker manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = SCRIPT_DIR.parent
DEFAULT_MANIFEST = TRAIN_DIR / "scripts" / "testdata" / "v6_temp_canaries.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a temp inbox image into the blocker manifest.")
    parser.add_argument("--source-path", required=True, help="Workspace-relative image path, typically temp/<file>.")
    parser.add_argument("--truth-fen", default=None, help="Full truth FEN. Required for --status locked.")
    parser.add_argument("--note", default="", help="Short note for the blocker entry.")
    parser.add_argument("--status", choices=["locked", "pending"], default="pending")
    parser.add_argument("--blocker-id", default=None, help="Optional explicit blocker ID.")
    parser.add_argument("--manifest-json", default=str(DEFAULT_MANIFEST))
    return parser.parse_args()


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {"note": "", "locked_cases": [], "pending_cases": []}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"manifest JSON must be an object: {path}")
    raw.setdefault("note", "")
    raw.setdefault("locked_cases", [])
    raw.setdefault("pending_cases", [])
    return raw


def _all_ids(manifest: dict) -> set[str]:
    out = set()
    for bucket in ("locked_cases", "pending_cases"):
        for row in manifest.get(bucket, []):
            out.add(str(row["blocker_id"]))
    return out


def _next_blocker_id(manifest: dict) -> str:
    nums = []
    for blocker_id in _all_ids(manifest):
        if blocker_id.startswith("real-blocker-"):
            suffix = blocker_id.split("-")[-1]
            if suffix.isdigit():
                nums.append(int(suffix))
    next_num = (max(nums) + 1) if nums else 1
    return f"real-blocker-{next_num:04d}"


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest_json)
    manifest = _load_manifest(manifest_path)
    source_path = str(args.source_path)
    image_path = TRAIN_DIR / source_path
    if not image_path.exists():
        raise FileNotFoundError(f"source image not found: {image_path}")
    if args.status == "locked" and not args.truth_fen:
        raise ValueError("--truth-fen is required for --status locked")

    blocker_id = args.blocker_id or _next_blocker_id(manifest)
    locked_cases = [row for row in manifest.get("locked_cases", []) if row.get("source_path") != source_path and row.get("blocker_id") != blocker_id]
    pending_cases = [row for row in manifest.get("pending_cases", []) if row.get("source_path") != source_path and row.get("blocker_id") != blocker_id]

    entry = {
        "blocker_id": blocker_id,
        "source_path": source_path,
        "note": args.note,
    }
    if args.status == "locked":
        entry["truth_fen"] = str(args.truth_fen)
        locked_cases.append(entry)
    else:
        pending_cases.append(entry)

    manifest["locked_cases"] = sorted(locked_cases, key=lambda row: str(row["blocker_id"]))
    manifest["pending_cases"] = sorted(pending_cases, key=lambda row: str(row["blocker_id"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"blocker_id={blocker_id}")
    print(f"status={args.status}")
    print(f"source_path={source_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
