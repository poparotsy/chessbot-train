#!/usr/bin/env python3
"""Benchmark zero-shot board detectors on the exported Kaggle payload."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import signal
import subprocess
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from PIL import Image
from huggingface_hub.utils import disable_progress_bars
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Owlv2ForObjectDetection, Owlv2Processor
from transformers.utils import logging as transformers_logging

from common import TRAIN_DIR, bbox_iou, choose_best_candidate, ensure_dir, mean_or_zero, median_or_zero, now_iso, overlay_boxes, square_crop_from_bbox, write_json


DEFAULT_DATA_ROOT = TRAIN_DIR / "generated" / "transfer_localizer_v1"
DEFAULT_REPORTS_DIR = TRAIN_DIR / "reports" / "transfer_localizer_v1"
DEFAULT_VIZ_DIR = TRAIN_DIR / "generated" / "transfer_localizer_v1" / "viz"
DEFAULT_QUICK_IMAGES_DIR = TRAIN_DIR / "images_4_test"
DEFAULT_QUICK_SUITE_JSON = TRAIN_DIR / "scripts" / "testdata" / "v6_quick_cases.json"
MODEL_RUN_NAMES = {
    "google/owlv2-base-patch16-ensemble": "owlv2-base-ens",
    "google/owlv2-base-patch16": "owlv2-base",
    "google/owlv2-large-patch14-ensemble": "owlv2-large-ens",
    "google/owlvit-base-patch32": "owlvit-base32",
    "IDEA-Research/grounding-dino-tiny": "grounding-dino-tiny",
    "IDEA-Research/grounding-dino-base": "grounding-dino-base",
    "iSEE-Laboratory/llmdet_base": "llmdet-base",
    "iSEE-Laboratory/llmdet_large": "llmdet-large",
}
DEFAULT_MODELS = [
    "IDEA-Research/grounding-dino-tiny",
    "IDEA-Research/grounding-dino-base",
    "google/owlv2-base-patch16-ensemble",
    "google/owlv2-base-patch16",
    "google/owlvit-base-patch32",
    "iSEE-Laboratory/llmdet_base",
    "google/owlv2-large-patch14-ensemble",
]

disable_progress_bars()
transformers_logging.set_verbosity_error()
logging.getLogger("huggingface_hub.utils._headers").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r".*Use `text_labels` instead to retrieve string object names.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*loaded as a fast processor by default.*",
    category=UserWarning,
)


def _log(message: str, quiet: bool = False, prefix: str | None = None) -> None:
    if not quiet:
        if prefix:
            print(f"[{prefix}] {message}", flush=True)
        else:
            print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark zero-shot board localizers against the Kaggle payload.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--viz-dir", default=str(DEFAULT_VIZ_DIR))
    parser.add_argument("--quick-images-dir", default=str(DEFAULT_QUICK_IMAGES_DIR))
    parser.add_argument("--quick-suite-json", default=str(DEFAULT_QUICK_SUITE_JSON))
    parser.add_argument("--model-id", action="append", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--compare-json", default=None, help="Optional comparison JSON output path.")
    parser.add_argument("--compare-md", default=None, help="Optional comparison Markdown output path.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--board-perspective", choices=["auto", "white", "black"], default="auto")
    parser.add_argument("--write-viz", action="store_true")
    parser.add_argument("--write-crops", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--serial-models", action="store_true", help="Run models one after another even if multiple GPUs are available.")
    parser.add_argument("--quick-only", action="store_true", help="Run only the quick images_4_test sanity set.")
    parser.add_argument("--detector-only", action="store_true", help="Score raw detector boxes/crops only and skip recognizer refinement.")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_quick_rows(images_dir: Path, truth_json: Path) -> list[dict]:
    truth = json.loads(truth_json.read_text(encoding="utf-8"))
    rows = []
    for name, fen in truth.items():
        rows.append(
            {
                "id": f"quick-{name.rsplit('.', 1)[0]}",
                "image_path": str((images_dir / name).resolve()),
                "source_type": "quick_suite",
                "source_id": name,
                "truth_fen": fen,
                "blocker_id": None,
                "original_filename": name,
                "synthetic_template_bbox_xyxy": None,
            }
        )
    return rows


def _load_recognizer_module():
    script = TRAIN_DIR / "recognizer_v6.py"
    module_name = f"transfer_localizer_recognizer_{abs(hash(str(script.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load recognizer from {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _queries_for_model(model_id: str) -> list[str]:
    if "owlv2" in model_id.lower():
        return ["a photo of a chessboard", "a photo of a chess board"]
    return ["a chessboard", "a chess board"]


def _run_name_for_model(model_id: str) -> str:
    if model_id in MODEL_RUN_NAMES:
        return MODEL_RUN_NAMES[model_id]
    return model_id.split("/")[-1].replace("_", "-")


def _load_model_and_processor(model_id: str, device: str, quiet: bool = False, prefix: str | None = None):
    model_id_l = model_id.lower()
    _log("loading_processor", quiet, prefix)
    if "owlv2" in model_id_l:
        processor = Owlv2Processor.from_pretrained(model_id)
    elif "owlvit" in model_id_l:
        processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
        _log("loading_model", quiet, prefix)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    else:
        processor = AutoProcessor.from_pretrained(model_id)
        _log("loading_model", quiet, prefix)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    return processor, model.to(device)


def _unsupported_model_report(model_id: str, run_name: str, reason: str) -> dict:
    return {
        "generated_at": now_iso(),
        "model_id": model_id,
        "run_name": run_name,
        "status": "unsupported_model",
        "reason": reason,
        "image_count": 0,
        "detection_count": 0,
        "no_detection_count": 0,
        "rejected_detection_count": 0,
        "quick_pass_count": 0,
        "quick_pass_rate": 0.0,
        "quick_detected_count": 0,
        "quick_detected_rate": 0.0,
        "quick_retire": True,
        "synthetic_box_iou_mean": 0.0,
        "synthetic_box_iou_pass_count": 0,
        "synthetic_box_iou_pass_rate": 0.0,
        "synthetic_downstream_warp_success_count": 0,
        "synthetic_downstream_warp_success_rate": 0.0,
        "synthetic_oracle_warp_success_count": 0,
        "synthetic_oracle_warp_success_rate": 0.0,
        "blocker_detected_count": 0,
        "blocker_pass_count": 0,
        "blocker_pass_rate": 0.0,
        "median_inference_seconds": 0.0,
        "failed_ids": [],
        "results": [],
    }


def _print_run_summary(report: dict, quick_only: bool) -> None:
    if quick_only:
        print(
            f"{report['run_name']}: status={report.get('status', 'ok')} "
            f"quick_pass={report['quick_pass_count']}/{report.get('quick_count', 0)} "
            f"quick_retire={str(report.get('quick_retire', False)).lower()} "
            f"median_sec={report['median_inference_seconds']:.4f}",
            flush=True,
        )
        return
    print(
        f"{report['run_name']}: status={report.get('status', 'ok')} "
        f"quick_pass={report['quick_pass_count']}/{report.get('quick_count', 0)} "
        f"blocker_detected={report['blocker_detected_count']} "
        f"blocker_pass={report['blocker_pass_count']} "
        f"synthetic_box_iou={report['synthetic_box_iou_mean']:.4f} "
        f"synthetic_warp_rate={report['synthetic_downstream_warp_success_rate']:.4f} "
        f"median_sec={report['median_inference_seconds']:.4f}",
        flush=True,
    )


def _compare_sort_key(row: dict, quick_only: bool):
    if quick_only:
        return (
            int(row.get("quick_pass_count", 0)),
            float(row.get("quick_pass_rate", 0.0)),
            -float(row.get("median_inference_seconds", 0.0)),
        )
    return (
        int(row.get("blocker_pass_count", 0)),
        float(row.get("synthetic_downstream_warp_success_rate", 0.0)),
        int(row.get("quick_pass_count", 0)),
        float(row.get("synthetic_box_iou_mean", 0.0)),
        -float(row.get("median_inference_seconds", 0.0)),
    )


def _build_compare_payload(runs: list[dict], quick_only: bool) -> dict:
    ranked = sorted(runs, key=lambda row: _compare_sort_key(row, quick_only), reverse=True)
    winner = ranked[0] if ranked else None
    return {
        "generated_at": now_iso(),
        "quick_only": bool(quick_only),
        "winner": None if winner is None else {
            "run_name": winner["run_name"],
            "model_id": winner["model_id"],
            "status": winner.get("status", "ok"),
        },
        "runs": ranked,
    }


def _compare_markdown(compare_payload: dict) -> str:
    lines = [
        "# Transfer Localizer Model Comparison",
        "",
        f"- generated_at: `{compare_payload['generated_at']}`",
        f"- quick_only: `{str(compare_payload['quick_only']).lower()}`",
    ]
    winner = compare_payload.get("winner")
    if winner:
        lines.append(f"- winner: `{winner['run_name']}` (`{winner['model_id']}`)")
    lines.extend(
        [
            "",
            "| rank | run | model | status | quick | blocker_detected | blocker_pass | synth_iou | synth_warp | median_sec |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for idx, row in enumerate(compare_payload.get("runs", []), start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("run_name")),
                    str(row.get("model_id")),
                    str(row.get("status", "ok")),
                    f"{int(row.get('quick_pass_count', 0))}/{int(row.get('quick_count', 0))}",
                    str(int(row.get("blocker_detected_count", 0))),
                    str(int(row.get("blocker_pass_count", 0))),
                    f"{float(row.get('synthetic_box_iou_mean', 0.0)):.4f}",
                    f"{float(row.get('synthetic_downstream_warp_success_rate', 0.0)):.4f}",
                    f"{float(row.get('median_inference_seconds', 0.0)):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _detect_boxes(image: Image.Image, model_id: str, processor, model, device: str, threshold: float, text_threshold: float) -> list[dict]:
    query_labels = _queries_for_model(model_id)
    text_labels = [query_labels]
    start = time.perf_counter()
    if "owlv2" in model_id.lower():
        inputs = processor(text=text_labels, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=torch.tensor([(image.height, image.width)], device=device),
            threshold=threshold,
            text_labels=text_labels,
        )[0]
        elapsed = time.perf_counter() - start
        rows = []
        for box, score, label in zip(results["boxes"], results["scores"], results["text_labels"]):
            rows.append(
                {
                    "box": [float(x) for x in box.tolist()],
                    "score": float(score.item()),
                    "label": str(label),
                    "elapsed_sec": float(elapsed),
                }
            )
        return rows

    if "owlvit" in model_id.lower():
        inputs = processor(text=text_labels, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=torch.tensor([(image.height, image.width)], device=device),
        )[0]
        elapsed = time.perf_counter() - start
        rows = []
        for box, score, label_idx in zip(results["boxes"], results["scores"], results["labels"]):
            idx = int(label_idx.item()) if hasattr(label_idx, "item") else int(label_idx)
            label = query_labels[idx] if 0 <= idx < len(query_labels) else str(idx)
            rows.append(
                {
                    "box": [float(x) for x in box.tolist()],
                    "score": float(score.item()),
                    "label": str(label),
                    "elapsed_sec": float(elapsed),
                }
            )
        return rows

    inputs = processor(images=image, text=text_labels, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]
    elapsed = time.perf_counter() - start
    label_values = results.get("text_labels", results.get("labels", []))
    rows = []
    for box, score, label in zip(results["boxes"], results["scores"], label_values):
        if isinstance(label, list):
            label = ", ".join(label)
        rows.append(
            {
                "box": [float(x) for x in box.tolist()],
                "score": float(score.item()),
                "label": str(label),
                "elapsed_sec": float(elapsed),
            }
        )
    return rows


def _run_deterministic_refinement(image_path: Path, crop_box: list[int], board_perspective: str) -> dict:
    rec = _load_recognizer_module()
    image = Image.open(image_path).convert("RGB")
    cropped = image.crop(tuple(crop_box))
    tmp_dir = Path(tempfile.mkdtemp(prefix="chessbot_localizer_crop_"))
    tmp_image = tmp_dir / image_path.name
    try:
        cropped.save(tmp_image)
        result = rec.predict_position(str(tmp_image), board_perspective=board_perspective)
        return {
            "success": True,
            "fen": str(result.get("fen") or ""),
            "best_tag": result.get("best_tag"),
            "detector_score": result.get("detector_score"),
            "detector_support": result.get("detector_support"),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _truth_box_for_row(row: dict) -> list[float] | None:
    truth_box = row.get("synthetic_template_bbox_xyxy")
    if truth_box is None:
        return None
    return [float(v) for v in truth_box]


def benchmark_model(args: argparse.Namespace, model_id: str, run_name: str) -> dict:
    data_root = Path(args.data_root)
    reports_dir = ensure_dir(Path(args.reports_dir))
    viz_dir = ensure_dir(Path(args.viz_dir) / run_name)
    quick_rows = _load_quick_rows(Path(args.quick_images_dir), Path(args.quick_suite_json))
    rows = quick_rows if args.quick_only else quick_rows + _load_rows(data_root / "manifests" / "all.jsonl")
    log_prefix = run_name
    _log("=== ZERO-SHOT LOCALIZER BENCHMARK ===", args.quiet, log_prefix)
    _log(f"run_name={run_name}", args.quiet, log_prefix)
    _log(f"model_id={model_id}", args.quiet, log_prefix)
    _log(f"device={args.device}", args.quiet, log_prefix)
    try:
        processor, model = _load_model_and_processor(model_id, args.device, args.quiet, log_prefix)
    except Exception as exc:
        report = _unsupported_model_report(model_id, run_name, str(exc))
        write_json(reports_dir / f"{run_name}.json", report)
        _log("=== RUN SKIPPED ===", args.quiet, log_prefix)
        _log(f"reason=unsupported_model", args.quiet, log_prefix)
        _log(f"report_json={reports_dir / f'{run_name}.json'}", args.quiet, log_prefix)
        return report
    _log(f"images={len(rows)}", args.quiet, log_prefix)

    inference_times = []
    detection_count = 0
    no_detection_count = 0
    rejected_detection_count = 0
    synthetic_box_ious = []
    synthetic_box_iou_pass_count = 0
    synthetic_warp_success = 0
    synthetic_oracle_warp_success = 0
    quick_pass = 0
    quick_detected_count = 0
    blocker_pass = 0
    blocker_detected_count = 0
    failed_ids = []
    results = []

    for idx, row in enumerate(rows, start=1):
        image_path = data_root / row["image_path"]
        if row["source_type"] == "quick_suite":
            image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        detections = _detect_boxes(
            image=image,
            model_id=model_id,
            processor=processor,
            model=model,
            device=args.device,
            threshold=float(args.threshold),
            text_threshold=float(args.text_threshold),
        )
        if detections:
            inference_times.append(float(detections[0]["elapsed_sec"]))
            detection_count += 1
        best = choose_best_candidate(detections, image.width, image.height)
        if best is None:
            failed_ids.append(row["id"])
            no_detection_count += 1
            results.append({"id": row["id"], "success": False, "error": "no_detections", "failure_stage": "no_box"})
            _log(f"[{idx:03d}/{len(rows)}] {row['id']} no_box", args.quiet, log_prefix)
            continue

        pred_box = [float(v) for v in best["box"]]
        if not bool(best.get("accepted", False)):
            rejected_detection_count += 1
        crop_box, _ = square_crop_from_bbox(pred_box, image.width, image.height, pad_ratio=0.12)
        quick_detector_stage = row["source_type"] == "quick_suite"
        run_refine = not args.detector_only and not quick_detector_stage
        refine = {"success": False, "skipped": True} if not run_refine else _run_deterministic_refinement(image_path, crop_box, args.board_perspective)
        truth_box = _truth_box_for_row(row)
        box_iou = None
        oracle_refine = None
        if truth_box is not None:
            box_iou = float(bbox_iou(pred_box, truth_box))
            synthetic_box_ious.append(box_iou)
            if box_iou >= 0.5:
                synthetic_box_iou_pass_count += 1
            oracle_crop_box, _ = square_crop_from_bbox(truth_box, image.width, image.height, pad_ratio=0.12)
            oracle_refine = _run_deterministic_refinement(image_path, oracle_crop_box, args.board_perspective)
            if oracle_refine.get("success"):
                synthetic_oracle_warp_success += 1

        record = {
            "id": row["id"],
            "source_type": row["source_type"],
            "box": pred_box,
            "crop_box": crop_box,
            "truth_box": truth_box,
            "box_iou": box_iou,
            "score": float(best["score"]),
            "label": best["label"],
            "accepted": bool(best.get("accepted", False)),
            "refine_success": bool(refine.get("success", False)),
            "detector_only": bool(args.detector_only),
        }

        if row["source_type"] == "synthetic":
            if not args.detector_only and refine.get("success"):
                synthetic_warp_success += 1
            record["oracle_refine_success"] = bool(oracle_refine and oracle_refine.get("success", False)) if not args.detector_only else None
            if args.detector_only:
                if box_iou is not None and box_iou >= 0.5:
                    record["failure_stage"] = None
                elif box_iou is not None:
                    record["failure_stage"] = "bad_box"
                else:
                    record["failure_stage"] = "no_box_truth"
            elif refine.get("success"):
                record["failure_stage"] = None
            elif oracle_refine and oracle_refine.get("success", False):
                record["failure_stage"] = "predicted_crop_downstream_fail"
            elif box_iou is not None and box_iou < 0.5:
                record["failure_stage"] = "bad_box"
            else:
                record["failure_stage"] = "downstream_fail"
        else:
            if row["source_type"] == "real_blocker":
                blocker_detected_count += 1
            record["truth_fen"] = row["truth_fen"]
            if args.detector_only or quick_detector_stage:
                if row["source_type"] == "quick_suite":
                    quick_pass += 1 if bool(best.get("accepted", False)) else 0
                    record["quick_pass"] = bool(best.get("accepted", False))
                else:
                    blocker_pass += 1 if bool(best.get("accepted", False)) else 0
                    record["blocker_pass"] = bool(best.get("accepted", False))
                record["failure_stage"] = None if bool(best.get("accepted", False)) else "bad_box"
            elif refine.get("success") and str(refine.get("fen") or "") == str(row["truth_fen"]):
                if row["source_type"] == "quick_suite":
                    quick_pass += 1
                    record["quick_pass"] = True
                else:
                    blocker_pass += 1
                    record["blocker_pass"] = True
                record["failure_stage"] = None
            else:
                if row["source_type"] == "quick_suite":
                    record["quick_pass"] = False
                else:
                    record["blocker_pass"] = False
                record["failure_stage"] = "downstream_fail"
            if row["source_type"] == "quick_suite":
                quick_detected_count += 1

        if args.detector_only or quick_detector_stage:
            if not bool(best.get("accepted", False)):
                failed_ids.append(row["id"])
        elif not refine.get("success", False):
            failed_ids.append(row["id"])
            record["error"] = refine.get("error")
        elif row["source_type"] == "real_blocker" and not record["blocker_pass"]:
            failed_ids.append(row["id"])
        elif row["source_type"] == "quick_suite" and not record["quick_pass"]:
            failed_ids.append(row["id"])

        if args.write_viz:
            overlay_boxes(
                image_path=image_path,
                out_path=viz_dir / f"{row['id']}.png",
                truth_box=truth_box,
                pred_box=pred_box,
                crop_box=crop_box,
                title=f"{run_name} {row['id']}",
            )
        if args.write_crops:
            crop_dir = ensure_dir(viz_dir / "crops")
            image.crop(tuple(crop_box)).save(crop_dir / f"{row['id']}.png")

        results.append(record)
        if row["source_type"] == "quick_suite":
            status = "PASS" if record.get("quick_pass") else "FAIL"
            _log(
                f"[{idx:03d}/{len(rows)}] {row['id']} quick_{status.lower()} "
                f"stage={record['failure_stage'] or 'pass'} score={record['score']:.4f}",
                args.quiet,
                log_prefix,
            )
        elif row["source_type"] == "real_blocker":
            status = "PASS" if record.get("blocker_pass") else "FAIL"
            _log(
                f"[{idx:03d}/{len(rows)}] {row['id']} blocker_{status.lower()} "
                f"stage={record['failure_stage'] or 'pass'} score={record['score']:.4f}",
                args.quiet,
                log_prefix,
            )
        elif idx == 1 or idx % 10 == 0 or idx == len(rows):
            status = "OK" if refine.get("success") else "FAIL"
            _log(
                f"[{idx:03d}/{len(rows)}] {row['id']} synthetic_{status.lower()} "
                f"stage={record['failure_stage'] or 'pass'} "
                f"iou={record['box_iou']:.3f} score={record['score']:.4f}" if record['box_iou'] is not None
                else f"[{idx:03d}/{len(rows)}] {row['id']} synthetic_{status.lower()} stage={record['failure_stage'] or 'pass'} score={record['score']:.4f}",
                args.quiet,
                log_prefix,
            )

    quick_count = sum(1 for row in rows if row["source_type"] == "quick_suite")
    synthetic_count = sum(1 for row in rows if row["source_type"] == "synthetic")
    blocker_count = sum(1 for row in rows if row["source_type"] == "real_blocker")
    report = {
        "generated_at": now_iso(),
        "model_id": model_id,
        "run_name": run_name,
        "image_count": len(rows),
        "quick_count": quick_count,
        "detection_count": detection_count,
        "no_detection_count": no_detection_count,
        "rejected_detection_count": rejected_detection_count,
        "quick_pass_count": quick_pass,
        "quick_pass_rate": (float(quick_pass) / float(quick_count)) if quick_count else 0.0,
        "quick_detected_count": quick_detected_count,
        "quick_detected_rate": (float(quick_detected_count) / float(quick_count)) if quick_count else 0.0,
        "quick_retire": bool(quick_count and quick_pass < quick_count),
        "detector_only": bool(args.detector_only),
        "synthetic_box_iou_mean": mean_or_zero(synthetic_box_ious),
        "synthetic_box_iou_pass_count": synthetic_box_iou_pass_count,
        "synthetic_box_iou_pass_rate": (float(synthetic_box_iou_pass_count) / float(synthetic_count)) if synthetic_count else 0.0,
        "synthetic_downstream_warp_success_count": synthetic_warp_success,
        "synthetic_downstream_warp_success_rate": (float(synthetic_warp_success) / float(synthetic_count)) if synthetic_count else 0.0,
        "synthetic_oracle_warp_success_count": synthetic_oracle_warp_success,
        "synthetic_oracle_warp_success_rate": (float(synthetic_oracle_warp_success) / float(synthetic_count)) if synthetic_count else 0.0,
        "blocker_detected_count": blocker_detected_count,
        "blocker_pass_count": blocker_pass,
        "blocker_pass_rate": (float(blocker_pass) / float(blocker_count)) if blocker_count else 0.0,
        "median_inference_seconds": median_or_zero(inference_times),
        "failed_ids": failed_ids,
        "results": results,
    }
    write_json(reports_dir / f"{run_name}.json", report)
    _log("=== RUN COMPLETE ===", args.quiet, log_prefix)
    _log(f"report_json={reports_dir / f'{run_name}.json'}", args.quiet, log_prefix)
    _log(f"quick_pass={report['quick_pass_count']}/{quick_count}", args.quiet, log_prefix)
    _log(f"quick_retire={str(report['quick_retire']).lower()}", args.quiet, log_prefix)
    if not args.quick_only:
        _log(f"blocker_detected={report['blocker_detected_count']}", args.quiet, log_prefix)
        _log(f"synthetic_box_iou_mean={report['synthetic_box_iou_mean']:.4f}", args.quiet, log_prefix)
        _log(f"synthetic_oracle_warp_rate={report['synthetic_oracle_warp_success_rate']:.4f}", args.quiet, log_prefix)
        _log(f"blocker_pass={report['blocker_pass_count']}", args.quiet, log_prefix)
        _log(f"synthetic_warp_rate={report['synthetic_downstream_warp_success_rate']:.4f}", args.quiet, log_prefix)
    _log(f"median_sec={report['median_inference_seconds']:.4f}", args.quiet, log_prefix)
    return report


def _supports_parallel_gpu_runs(args: argparse.Namespace, runs: list[tuple[str, str]]) -> bool:
    return (
        not args.serial_models
        and len(runs) > 1
        and str(args.device).strip().lower() == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    )


def _terminate_pending_processes(pending: list[tuple[subprocess.Popen[str], int, str, str, Path]], quiet: bool) -> None:
    for proc, _gpu_id, _model_id, run_name, _report_path in pending:
        if proc.poll() is not None:
            continue
        _log(f"interrupting run_name={run_name}", quiet)
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
    deadline = time.time() + 5.0
    for proc, _gpu_id, _model_id, run_name, _report_path in pending:
        if proc.poll() is not None:
            continue
        timeout = max(0.0, deadline - time.time())
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _log(f"killing run_name={run_name}", quiet)
            proc.kill()


def _launch_parallel_model_runs(args: argparse.Namespace, runs: list[tuple[str, str]]) -> list[dict]:
    reports_dir = ensure_dir(Path(args.reports_dir))
    available_gpu_ids = list(range(torch.cuda.device_count()))
    pending: list[tuple[subprocess.Popen[str], int, str, str, Path]] = []
    queued = list(runs)
    summaries: list[dict] = []

    _log(f"parallel_gpu_slots={len(available_gpu_ids)}", args.quiet)

    try:
        while queued or pending:
            while available_gpu_ids and queued:
                gpu_id = available_gpu_ids.pop(0)
                model_id, run_name = queued.pop(0)
                report_path = reports_dir / f"{run_name}.json"
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--data-root",
                    str(args.data_root),
                    "--reports-dir",
                    str(args.reports_dir),
                    "--viz-dir",
                    str(args.viz_dir),
                    "--quick-images-dir",
                    str(args.quick_images_dir),
                    "--quick-suite-json",
                    str(args.quick_suite_json),
                    "--model-id",
                    model_id,
                    "--run-name",
                    run_name,
                    "--device",
                    "cuda",
                    "--threshold",
                    str(args.threshold),
                    "--text-threshold",
                    str(args.text_threshold),
                    "--board-perspective",
                    str(args.board_perspective),
                ]
                if args.write_viz:
                    cmd.append("--write-viz")
                if args.write_crops:
                    cmd.append("--write-crops")
                if args.quiet:
                    cmd.append("--quiet")
                if args.quick_only:
                    cmd.append("--quick-only")
                if args.detector_only:
                    cmd.append("--detector-only")

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                _log(f"launch run_name={run_name} gpu={gpu_id}", args.quiet)
                proc = subprocess.Popen(cmd, cwd=str(TRAIN_DIR), env=env, text=True)
                pending.append((proc, gpu_id, model_id, run_name, report_path))

            still_pending: list[tuple[subprocess.Popen[str], int, str, str, Path]] = []
            for proc, gpu_id, model_id, run_name, report_path in pending:
                exit_code = proc.poll()
                if exit_code is None:
                    still_pending.append((proc, gpu_id, model_id, run_name, report_path))
                    continue
                available_gpu_ids.append(gpu_id)
                available_gpu_ids.sort()
                if exit_code != 0:
                    raise SystemExit(exit_code)
                report = json.loads(report_path.read_text(encoding="utf-8"))
                summaries.append(
                    {
                        "run_name": run_name,
                        "model_id": model_id,
                        "status": report.get("status", "ok"),
                        "quick_count": report.get("quick_count", 0),
                        "quick_pass_count": report.get("quick_pass_count", 0),
                        "quick_pass_rate": report.get("quick_pass_rate", 0.0),
                        "blocker_detected_count": report.get("blocker_detected_count", 0),
                        "blocker_pass_count": report["blocker_pass_count"],
                        "synthetic_box_iou_mean": report.get("synthetic_box_iou_mean", 0.0),
                        "synthetic_downstream_warp_success_rate": report["synthetic_downstream_warp_success_rate"],
                        "median_inference_seconds": report["median_inference_seconds"],
                        "report_json": str(report_path),
                    }
                )
                _print_run_summary(report, args.quick_only)
            pending = still_pending
            if pending:
                time.sleep(1.0)
    except KeyboardInterrupt:
        _terminate_pending_processes(pending, args.quiet)
        raise SystemExit(130)

    return summaries


def main() -> int:
    args = parse_args()
    runs = []
    if args.all_models:
        for model_id in DEFAULT_MODELS:
            runs.append((model_id, _run_name_for_model(model_id)))
    else:
        if not args.model_id:
            raise SystemExit("--model-id or --all-models is required")
        if len(args.model_id) == 1:
            model_id = args.model_id[0]
            runs.append((model_id, args.run_name or _run_name_for_model(model_id)))
        else:
            for model_id in args.model_id:
                runs.append((model_id, _run_name_for_model(model_id)))

    try:
        if _supports_parallel_gpu_runs(args, runs):
            summaries = _launch_parallel_model_runs(args, runs)
        else:
            summaries = []
            for model_id, run_name in runs:
                report = benchmark_model(args, model_id=model_id, run_name=run_name)
                summaries.append(
                    {
                        "run_name": run_name,
                        "model_id": model_id,
                        "status": report.get("status", "ok"),
                        "quick_count": report.get("quick_count", 0),
                        "quick_pass_count": report.get("quick_pass_count", 0),
                        "quick_pass_rate": report.get("quick_pass_rate", 0.0),
                        "blocker_detected_count": report.get("blocker_detected_count", 0),
                        "blocker_pass_count": report["blocker_pass_count"],
                        "synthetic_box_iou_mean": report.get("synthetic_box_iou_mean", 0.0),
                        "synthetic_downstream_warp_success_rate": report["synthetic_downstream_warp_success_rate"],
                        "median_inference_seconds": report["median_inference_seconds"],
                        "report_json": str(Path(args.reports_dir) / f"{run_name}.json"),
                    }
                )
                _print_run_summary(report, args.quick_only)
    except KeyboardInterrupt:
        _log("interrupted", args.quiet)
        raise SystemExit(130)

    compare_payload = _build_compare_payload(summaries, args.quick_only)
    compare_json_path = Path(args.compare_json) if args.compare_json else Path(args.reports_dir) / "summary.json"
    compare_md_path = Path(args.compare_md) if args.compare_md else Path(args.reports_dir) / "summary.md"
    write_json(compare_json_path, compare_payload)
    compare_md_path.parent.mkdir(parents=True, exist_ok=True)
    compare_md_path.write_text(_compare_markdown(compare_payload), encoding="utf-8")
    _log(f"summary_json={compare_json_path}", args.quiet)
    _log(f"summary_md={compare_md_path}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
