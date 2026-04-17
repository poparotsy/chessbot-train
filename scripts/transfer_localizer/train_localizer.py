#!/usr/bin/env python3
"""Prepare and optionally launch fine-tuning runs for localizer backends."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from common import TRAIN_DIR, ensure_dir, load_jsonl, normalize_localizer_manifest_row, write_json


DEFAULT_DATA_ROOT = TRAIN_DIR / "generated" / "transfer_localizer_v1"
DEFAULT_EXPORT_ROOT = TRAIN_DIR / "generated" / "transfer_localizer_training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare localizer fine-tune datasets for RF-DETR or YOLO backends.")
    parser.add_argument("--backend", choices=["rfdetr", "yolo"], required=True)
    parser.add_argument("--train-manifest", default=str(DEFAULT_DATA_ROOT / "manifests" / "localizer_train.jsonl"))
    parser.add_argument("--val-manifest", default=str(DEFAULT_DATA_ROOT / "manifests" / "localizer_val.jsonl"))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_EXPORT_ROOT))
    parser.add_argument("--run-name", default="rf-detr-nano")
    parser.add_argument("--weights", default=None, help="Optional pretrained weights/model id override.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--launch", action="store_true", help="Run backend training if dependencies are installed.")
    return parser.parse_args()


def _resolved_image_path(data_root: Path, row: dict) -> Path:
    image_path = Path(str(row["image_path"]))
    if image_path.is_absolute():
        return image_path
    return data_root / image_path


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for row in load_jsonl(path):
        normalized = normalize_localizer_manifest_row(row)
        if normalized.get("bbox_xyxy") is None:
            continue
        rows.append(normalized)
    return rows


def _coco_image(image_id: int, image_path: Path) -> dict:
    from PIL import Image

    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size
    return {
        "id": image_id,
        "file_name": image_path.name,
        "width": width,
        "height": height,
    }


def _coco_annotation(annotation_id: int, image_id: int, bbox: list[float]) -> dict:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [x0, y0, width, height],
        "area": width * height,
        "iscrowd": 0,
    }


def _export_coco(rows: list[dict], *, data_root: Path, images_out: Path, annotation_out: Path) -> None:
    ensure_dir(images_out)
    images = []
    annotations = []
    for image_id, row in enumerate(rows, start=1):
        src = _resolved_image_path(data_root, row)
        dst = images_out / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        images.append(_coco_image(image_id, dst))
        annotations.append(_coco_annotation(image_id, image_id, row["bbox_xyxy"]))
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "chessboard"}],
    }
    write_json(annotation_out, payload)


def _export_yolo(rows: list[dict], *, data_root: Path, images_out: Path, labels_out: Path) -> None:
    from PIL import Image

    ensure_dir(images_out)
    ensure_dir(labels_out)
    for row in rows:
        src = _resolved_image_path(data_root, row)
        dst = images_out / src.name
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        with Image.open(dst).convert("RGB") as image:
            width, height = image.size
        x0, y0, x1, y1 = [float(v) for v in row["bbox_xyxy"]]
        x_center = ((x0 + x1) / 2.0) / width
        y_center = ((y0 + y1) / 2.0) / height
        box_width = (x1 - x0) / width
        box_height = (y1 - y0) / height
        label_path = labels_out / f"{dst.stem}.txt"
        label_path.write_text(f"0 {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}\n", encoding="utf-8")


def _write_yolo_dataset_yaml(root: Path) -> Path:
    path = root / "dataset.yaml"
    path.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: train/images",
                "val: val/images",
                "names:",
                "  0: chessboard",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _launch_rfdetr(args: argparse.Namespace, run_root: Path) -> int:
    train_json = run_root / "annotations" / "train.json"
    val_json = run_root / "annotations" / "val.json"
    cmd = [
        sys.executable,
        "-m",
        "rfdetr.train",
        "--train-coco",
        str(train_json),
        "--val-coco",
        str(val_json),
        "--train-images",
        str(run_root / "train" / "images"),
        "--val-images",
        str(run_root / "val" / "images"),
        "--model",
        args.weights or "rf-detr-nano",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch),
        "--workers",
        str(args.workers),
    ]
    return subprocess.run(cmd, cwd=str(TRAIN_DIR)).returncode


def _launch_yolo(args: argparse.Namespace, run_root: Path) -> int:
    dataset_yaml = _write_yolo_dataset_yaml(run_root)
    cmd = [
        sys.executable,
        "-m",
        "ultralytics",
        "train",
        "detect",
        "model=" + str(args.weights or "yolo11n.pt"),
        "data=" + str(dataset_yaml),
        "epochs=" + str(args.epochs),
        "imgsz=" + str(args.imgsz),
        "batch=" + str(args.batch),
        "workers=" + str(args.workers),
        "name=" + str(args.run_name),
    ]
    return subprocess.run(cmd, cwd=str(TRAIN_DIR)).returncode


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    run_root = ensure_dir(Path(args.output_dir) / args.run_name / args.backend)
    train_rows = _load_rows(Path(args.train_manifest))
    val_rows = _load_rows(Path(args.val_manifest))
    if not train_rows:
        raise SystemExit("No labeled training rows found in train manifest.")
    if not val_rows:
        raise SystemExit("No labeled validation rows found in val manifest.")

    summary = {
        "backend": args.backend,
        "run_name": args.run_name,
        "weights": args.weights,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "run_root": str(run_root),
    }

    if args.backend == "rfdetr":
        _export_coco(train_rows, data_root=data_root, images_out=run_root / "train" / "images", annotation_out=run_root / "annotations" / "train.json")
        _export_coco(val_rows, data_root=data_root, images_out=run_root / "val" / "images", annotation_out=run_root / "annotations" / "val.json")
    else:
        _export_yolo(train_rows, data_root=data_root, images_out=run_root / "train" / "images", labels_out=run_root / "train" / "labels")
        _export_yolo(val_rows, data_root=data_root, images_out=run_root / "val" / "images", labels_out=run_root / "val" / "labels")
        summary["dataset_yaml"] = str(_write_yolo_dataset_yaml(run_root))

    write_json(run_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))

    if not args.launch:
        return 0
    if args.backend == "rfdetr":
        return _launch_rfdetr(args, run_root)
    return _launch_yolo(args, run_root)


if __name__ == "__main__":
    raise SystemExit(main())
