#!/usr/bin/env python3
"""Normalize SAM2 propagation exports into the localizer manifest schema."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import normalize_localizer_manifest_row, write_jsonl, load_json, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize SAM2 rows into localizer manifest JSONL.")
    parser.add_argument("--input", required=True, help="Input JSON or JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--default-split", default="sam2_real")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "rows" in payload and isinstance(payload["rows"], list):
        return payload["rows"]
    raise ValueError(f"Unsupported SAM2 payload: {path}")


def main() -> int:
    args = parse_args()
    rows = []
    for index, row in enumerate(_load_rows(Path(args.input)), start=1):
        normalized = normalize_localizer_manifest_row(row)
        normalized.setdefault("id", f"sam2-{index:05d}")
        normalized.setdefault("split", args.default_split)
        normalized.setdefault("source_type", "sam2_real")
        normalized.setdefault("source_id", normalized["id"])
        normalized.setdefault("truth_fen", None)
        normalized.setdefault("label_status", "labeled")
        normalized.setdefault("domain_tags", ["sam2"])
        rows.append(normalized)
    write_jsonl(Path(args.output), rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
