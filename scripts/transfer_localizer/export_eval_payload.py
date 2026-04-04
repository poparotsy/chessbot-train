#!/usr/bin/env python3
"""Export a Kaggle-friendly payload for zero-training board-localizer benchmarking."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from common import TRAIN_DIR, ensure_dir, load_json, now_iso, write_json, write_jsonl


DEFAULT_OUTPUT_DIR = TRAIN_DIR / "generated" / "transfer_localizer_v1"
DEFAULT_STRESS_DIR = TRAIN_DIR / "generated" / "v6_stress_suite"
DEFAULT_BLOCKER_MANIFEST = TRAIN_DIR / "scripts" / "testdata" / "v6_temp_canaries.json"


def _log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Kaggle-ready eval payload for zero-shot localizer benchmarking.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stress-dir", default=str(DEFAULT_STRESS_DIR))
    parser.add_argument("--blocker-manifest", default=str(DEFAULT_BLOCKER_MANIFEST))
    return parser.parse_args()


def _copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def _ensure_stress_suite(stress_dir: Path) -> None:
    manifest_path = stress_dir / "manifest.json"
    truth_path = stress_dir / "truth.json"
    images_dir = stress_dir / "images"
    if manifest_path.exists() and truth_path.exists() and images_dir.exists():
        _log(f"stress_suite={stress_dir}")
        return
    _log("=== BUILDING V6 STRESS SUITE ===")
    script_path = TRAIN_DIR / "scripts" / "generate_v6_stress_suite.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--count",
        "120",
        "--seed",
        "1337",
        "--output-dir",
        str(stress_dir),
    ]
    subprocess.run(cmd, check=True, cwd=str(TRAIN_DIR))
    _log(f"stress_suite={stress_dir}")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    _log("=== EXPORTING LOCALIZER PAYLOAD ===")
    _log(f"output_dir={output_dir}")
    images_dir = ensure_dir(output_dir / "images")
    manifests_dir = ensure_dir(output_dir / "manifests")

    stress_dir = Path(args.stress_dir)
    _ensure_stress_suite(stress_dir)
    stress_manifest = load_json(stress_dir / "manifest.json")
    stress_truth = load_json(stress_dir / "truth.json")
    synthetic_rows = []
    synthetic_items = stress_manifest["items"]
    _log(f"synthetic_images={len(synthetic_items)}")
    for idx, item in enumerate(synthetic_items, start=1):
        file_name = str(item["file"])
        src = stress_dir / "images" / file_name
        dst = images_dir / file_name
        _copy_file(src, dst)
        synthetic_rows.append(
            {
                "id": f"synthetic-{file_name.rsplit('.', 1)[0]}",
                "image_path": f"images/{file_name}",
                "split": "synthetic",
                "source_type": "synthetic",
                "source_id": file_name,
                "truth_fen": str(stress_truth[file_name]),
                "synthetic_template": str(item["source_template"]),
                "synthetic_template_corners_px": None,
                "synthetic_template_bbox_xyxy": None,
                "synthetic_source_size": item.get("source_size"),
                "synthetic_detector_tag": item.get("detector_tag"),
                "synthetic_detector_score": item.get("detector_score"),
                "synthetic_detector_support_ratio": item.get("detector_support_ratio"),
                "blocker_id": None,
                "original_filename": file_name,
            }
        )
        if idx == 1 or idx % 20 == 0 or idx == len(synthetic_items):
            _log(f"synthetic_progress={idx}/{len(synthetic_items)}")

    blocker_manifest = load_json(Path(args.blocker_manifest))
    blocker_rows = []
    locked_cases = blocker_manifest.get("locked_cases", [])
    _log(f"locked_blockers={len(locked_cases)}")
    for idx, item in enumerate(locked_cases, start=1):
        src = TRAIN_DIR / item["managed_path"]
        suffix = src.suffix.lower()
        file_name = f"{item['blocker_id']}{suffix}"
        dst = images_dir / file_name
        _copy_file(src, dst)
        blocker_rows.append(
            {
                "id": str(item["blocker_id"]),
                "image_path": f"images/{file_name}",
                "split": "real_blocker",
                "source_type": "real_blocker",
                "source_id": str(item["blocker_id"]),
                "truth_fen": str(item["truth_fen"]),
                "synthetic_template": None,
                "synthetic_template_corners_px": None,
                "synthetic_template_bbox_xyxy": None,
                "blocker_id": str(item["blocker_id"]),
                "original_filename": str(item["original_filename"]),
            }
        )
        _log(f"blocker_progress={idx}/{len(locked_cases)} {item['blocker_id']}")

    all_rows = synthetic_rows + blocker_rows
    _log("=== WRITING MANIFESTS ===")
    write_jsonl(manifests_dir / "synthetic.jsonl", synthetic_rows)
    write_jsonl(manifests_dir / "real_blockers.jsonl", blocker_rows)
    write_jsonl(manifests_dir / "all.jsonl", all_rows)
    write_json(
        manifests_dir / "summary.json",
        {
            "generated_at": now_iso(),
            "synthetic_count": len(synthetic_rows),
            "real_blocker_count": len(blocker_rows),
            "total_count": len(all_rows),
        },
    )

    _log(f"synthetic_count={len(synthetic_rows)}")
    _log(f"real_blocker_count={len(blocker_rows)}")
    _log(f"all_manifest={manifests_dir / 'all.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
