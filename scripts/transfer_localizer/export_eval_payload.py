#!/usr/bin/env python3
"""Export a benchmark and training payload for transfer-localizer work."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

from common import TRAIN_DIR, corners_to_bbox, ensure_dir, load_json, load_jsonl, normalize_localizer_manifest_row, now_iso, order_corners, write_json, write_jsonl


DEFAULT_OUTPUT_DIR = TRAIN_DIR / "generated" / "transfer_localizer_v1"
DEFAULT_STRESS_DIR = TRAIN_DIR / "generated" / "v6_stress_suite"
DEFAULT_BLOCKER_MANIFEST = TRAIN_DIR / "scripts" / "testdata" / "v6_temp_canaries.json"
DEFAULT_TEMPLATE_DIR = TRAIN_DIR / "images_4_test"


def _log(message: str) -> None:
    print(message, flush=True)


def _write_status(status_root: Path, **fields: object) -> None:
    write_json(
        status_root / "status.json",
        {
            "updated_at": now_iso(),
            **fields,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export localizer payloads for zero-shot benchmarking and fine-tune manifests.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stress-dir", default=str(DEFAULT_STRESS_DIR))
    parser.add_argument("--blocker-manifest", default=str(DEFAULT_BLOCKER_MANIFEST))
    parser.add_argument("--template-dir", default=str(DEFAULT_TEMPLATE_DIR))
    parser.add_argument("--sam2-jsonl", action="append", default=None, help="Optional SAM2 propagated manifest JSONL(s).")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Synthetic validation fraction for fine-tune manifests.")
    return parser.parse_args()


def _copy_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def _load_recognizer_module():
    script = TRAIN_DIR / "recognizer_v6.py"
    module_name = f"transfer_localizer_export_recognizer_{abs(hash(str(script.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load recognizer from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _build_template_truth_map(template_dir: Path, required_names: set[str]) -> dict[str, dict]:
    rec = _load_recognizer_module()
    truth_map: dict[str, dict] = {}
    usable_paths = [template_dir / name for name in sorted(required_names)]
    _log(f"template_truth_candidates={len(usable_paths)}")
    for idx, path in enumerate(usable_paths, start=1):
        if not path.exists():
            raise RuntimeError(f"missing template file {path.name}")
        img = Image.open(path).convert("RGB")
        proposals = rec.detect_board_grid(img, max_hypotheses=1)
        if not proposals:
            raise RuntimeError(f"no board proposal for template {path.name}")
        ordered = order_corners(proposals[0]["corners"])
        truth_map[path.name] = {
            "corners_px": ordered,
            "bbox_xyxy": corners_to_bbox(ordered),
        }
        if idx == 1 or idx % 10 == 0 or idx == len(usable_paths):
            _log(f"template_truth_progress={idx}/{len(usable_paths)}")
    return truth_map


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


def _synthetic_split(index: int, total: int, val_ratio: float) -> str:
    val_every = max(2, int(round(1.0 / max(1e-6, val_ratio))))
    if total <= 1:
        return "train"
    return "val" if (index % val_every) == 0 else "train"


def _load_sam2_rows(paths: list[str] | None) -> list[dict]:
    if not paths:
        return []
    rows: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"SAM2 manifest not found: {path}")
        for row in load_jsonl(path):
            normalized = normalize_localizer_manifest_row(row)
            normalized.setdefault("split", "sam2_real")
            normalized.setdefault("source_type", "sam2_real")
            normalized.setdefault("source_id", normalized.get("id"))
            normalized.setdefault("truth_fen", None)
            normalized.setdefault("label_status", "labeled")
            normalized.setdefault("domain_tags", ["sam2"])
            rows.append(normalized)
    return rows


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    _log("=== EXPORTING LOCALIZER PAYLOAD ===")
    _log(f"output_dir={output_dir}")
    ensure_dir(output_dir)
    images_dir = ensure_dir(output_dir / "images")
    manifests_dir = ensure_dir(output_dir / "manifests")
    _write_status(output_dir, stage="starting", output_dir=str(output_dir))

    stress_dir = Path(args.stress_dir)
    _write_status(output_dir, stage="ensuring_stress_suite", stress_dir=str(stress_dir))
    _ensure_stress_suite(stress_dir)
    stress_manifest = load_json(stress_dir / "manifest.json")
    stress_truth = load_json(stress_dir / "truth.json")
    required_template_names = {str(item["source_template"]) for item in stress_manifest["items"]}
    template_truth_map = _build_template_truth_map(Path(args.template_dir), required_template_names)
    synthetic_rows = []
    localizer_train_rows = []
    localizer_val_rows = []
    synthetic_items = stress_manifest["items"]
    _log(f"synthetic_images={len(synthetic_items)}")
    _write_status(output_dir, stage="copying_synthetic", synthetic_total=len(synthetic_items), synthetic_done=0)
    for idx, item in enumerate(synthetic_items, start=1):
        file_name = str(item["file"])
        src = stress_dir / "images" / file_name
        dst = images_dir / file_name
        _copy_file(src, dst)
        template_name = str(item["source_template"])
        template_truth = template_truth_map[template_name]
        row = {
            "id": f"synthetic-{file_name.rsplit('.', 1)[0]}",
            "image_path": f"images/{file_name}",
            "split": "synthetic",
            "source_type": "synthetic",
            "source_id": file_name,
            "truth_fen": str(stress_truth[file_name]),
            "corners_px": template_truth["corners_px"],
            "bbox_xyxy": template_truth["bbox_xyxy"],
            "synthetic_template": template_name,
            "synthetic_template_corners_px": template_truth["corners_px"],
            "synthetic_template_bbox_xyxy": template_truth["bbox_xyxy"],
            "synthetic_source_size": item.get("source_size"),
            "synthetic_detector_tag": item.get("detector_tag"),
            "synthetic_detector_score": item.get("detector_score"),
            "synthetic_detector_support_ratio": item.get("detector_support_ratio"),
            "blocker_id": None,
            "original_filename": file_name,
            "label_status": "labeled",
            "domain_tags": ["synthetic", "stress_suite"],
        }
        synthetic_rows.append(row)
        fine_tune_row = dict(row)
        fine_tune_row["split"] = _synthetic_split(idx, len(synthetic_items), float(args.val_ratio))
        if fine_tune_row["split"] == "train":
            localizer_train_rows.append(fine_tune_row)
        else:
            localizer_val_rows.append(fine_tune_row)
        if idx == 1 or idx % 20 == 0 or idx == len(synthetic_items):
            _log(f"synthetic_progress={idx}/{len(synthetic_items)}")
            _write_status(output_dir, stage="copying_synthetic", synthetic_total=len(synthetic_items), synthetic_done=idx)

    blocker_manifest = load_json(Path(args.blocker_manifest))
    blocker_rows = []
    locked_cases = blocker_manifest.get("locked_cases", [])
    _log(f"locked_blockers={len(locked_cases)}")
    _write_status(
        output_dir,
        stage="copying_blockers",
        synthetic_total=len(synthetic_items),
        synthetic_done=len(synthetic_items),
        blocker_total=len(locked_cases),
        blocker_done=0,
    )
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
                "corners_px": None,
                "bbox_xyxy": None,
                "synthetic_template": None,
                "synthetic_template_corners_px": None,
                "synthetic_template_bbox_xyxy": None,
                "blocker_id": str(item["blocker_id"]),
                "original_filename": str(item["original_filename"]),
                "label_status": "needs_annotation",
                "domain_tags": ["real", "blocker"],
            }
        )
        _log(f"blocker_progress={idx}/{len(locked_cases)} {item['blocker_id']}")
        _write_status(
            output_dir,
            stage="copying_blockers",
            synthetic_total=len(synthetic_items),
            synthetic_done=len(synthetic_items),
            blocker_total=len(locked_cases),
            blocker_done=idx,
        )

    sam2_rows = _load_sam2_rows(args.sam2_jsonl)
    all_rows = synthetic_rows + blocker_rows + sam2_rows
    localizer_train_rows.extend([row for row in sam2_rows if str(row.get("split")) not in {"val", "eval"}])
    localizer_val_rows.extend([row for row in sam2_rows if str(row.get("split")) in {"val", "eval"}])
    _log("=== WRITING MANIFESTS ===")
    _write_status(output_dir, stage="writing_manifests", total_rows=len(all_rows))
    write_jsonl(manifests_dir / "synthetic.jsonl", synthetic_rows)
    write_jsonl(manifests_dir / "real_blockers.jsonl", blocker_rows)
    write_jsonl(manifests_dir / "localizer_train.jsonl", localizer_train_rows)
    write_jsonl(manifests_dir / "localizer_val.jsonl", localizer_val_rows)
    write_jsonl(manifests_dir / "localizer_all.jsonl", localizer_train_rows + localizer_val_rows + blocker_rows)
    write_jsonl(manifests_dir / "all.jsonl", all_rows)
    write_json(
        manifests_dir / "summary.json",
        {
            "generated_at": now_iso(),
            "synthetic_count": len(synthetic_rows),
            "real_blocker_count": len(blocker_rows),
            "sam2_count": len(sam2_rows),
            "localizer_train_count": len(localizer_train_rows),
            "localizer_val_count": len(localizer_val_rows),
            "total_count": len(all_rows),
        },
    )

    _log(f"synthetic_count={len(synthetic_rows)}")
    _log(f"real_blocker_count={len(blocker_rows)}")
    _log(f"sam2_count={len(sam2_rows)}")
    _log(f"localizer_train_manifest={manifests_dir / 'localizer_train.jsonl'}")
    _log(f"localizer_val_manifest={manifests_dir / 'localizer_val.jsonl'}")
    _log(f"all_manifest={manifests_dir / 'all.jsonl'}")
    _write_status(
        output_dir,
        stage="complete",
        synthetic_count=len(synthetic_rows),
        real_blocker_count=len(blocker_rows),
        sam2_count=len(sam2_rows),
        total_count=len(all_rows),
        all_manifest=str(manifests_dir / "all.jsonl"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
