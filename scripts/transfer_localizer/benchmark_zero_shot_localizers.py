#!/usr/bin/env python3
"""Benchmark zero-shot board detectors on the exported Kaggle payload."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Owlv2ForObjectDetection, Owlv2Processor

from common import TRAIN_DIR, choose_best_candidate, ensure_dir, mean_or_zero, median_or_zero, now_iso, overlay_boxes, square_crop_from_bbox, write_json


DEFAULT_DATA_ROOT = TRAIN_DIR / "generated" / "transfer_localizer_v1"
DEFAULT_REPORTS_DIR = TRAIN_DIR / "reports" / "transfer_localizer_v1"
DEFAULT_VIZ_DIR = TRAIN_DIR / "generated" / "transfer_localizer_v1" / "viz"
DEFAULT_MODELS = [
    "google/owlv2-base-patch16-ensemble",
    "IDEA-Research/grounding-dino-tiny",
    "IDEA-Research/grounding-dino-base",
]


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
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--text-threshold", type=float, default=0.15)
    parser.add_argument("--board-perspective", choices=["auto", "white", "black"], default="auto")
    parser.add_argument("--write-viz", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--serial-models", action="store_true", help="Run models one after another even if multiple GPUs are available.")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
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


def _load_model_and_processor(model_id: str, device: str):
    model_id_l = model_id.lower()
    if "owlv2" in model_id_l:
        processor = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id)
    else:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
    return processor, model.to(device)


def _detect_boxes(image: Image.Image, model_id: str, processor, model, device: str, threshold: float, text_threshold: float) -> list[dict]:
    text_labels = [_queries_for_model(model_id)]
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


def benchmark_model(args: argparse.Namespace, model_id: str, run_name: str) -> dict:
    data_root = Path(args.data_root)
    reports_dir = ensure_dir(Path(args.reports_dir))
    viz_dir = ensure_dir(Path(args.viz_dir) / run_name)
    rows = _load_rows(data_root / "manifests" / "all.jsonl")
    log_prefix = run_name
    _log("=== ZERO-SHOT LOCALIZER BENCHMARK ===", args.quiet, log_prefix)
    _log(f"run_name={run_name}", args.quiet, log_prefix)
    _log(f"model_id={model_id}", args.quiet, log_prefix)
    _log(f"device={args.device}", args.quiet, log_prefix)
    processor, model = _load_model_and_processor(model_id, args.device)
    _log(f"images={len(rows)}", args.quiet, log_prefix)

    inference_times = []
    detection_count = 0
    synthetic_warp_success = 0
    blocker_pass = 0
    failed_ids = []
    results = []

    for idx, row in enumerate(rows, start=1):
        image_path = data_root / row["image_path"]
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
            results.append({"id": row["id"], "success": False, "error": "no_detections"})
            _log(f"[{idx:03d}/{len(rows)}] {row['id']} NO_DETECTIONS", args.quiet, log_prefix)
            continue

        pred_box = [float(v) for v in best["box"]]
        crop_box, _ = square_crop_from_bbox(pred_box, image.width, image.height, pad_ratio=0.12)
        refine = _run_deterministic_refinement(image_path, crop_box, args.board_perspective)

        record = {
            "id": row["id"],
            "source_type": row["source_type"],
            "box": pred_box,
            "crop_box": crop_box,
            "score": float(best["score"]),
            "label": best["label"],
            "accepted": bool(best.get("accepted", False)),
            "refine_success": bool(refine.get("success", False)),
        }

        if row["source_type"] == "synthetic":
            if refine.get("success"):
                synthetic_warp_success += 1
        else:
            record["truth_fen"] = row["truth_fen"]
            if refine.get("success") and str(refine.get("fen") or "") == str(row["truth_fen"]):
                blocker_pass += 1
                record["blocker_pass"] = True
            else:
                record["blocker_pass"] = False

        if not refine.get("success", False):
            failed_ids.append(row["id"])
            record["error"] = refine.get("error")
        elif row["source_type"] == "real_blocker" and not record["blocker_pass"]:
            failed_ids.append(row["id"])

        if args.write_viz:
            overlay_boxes(
                image_path=image_path,
                out_path=viz_dir / f"{row['id']}.png",
                truth_box=None,
                pred_box=pred_box,
                crop_box=crop_box,
                title=f"{run_name} {row['id']}",
            )

        results.append(record)
        if row["source_type"] == "real_blocker":
            status = "PASS" if record.get("blocker_pass") else "FAIL"
            _log(
                f"[{idx:03d}/{len(rows)}] {row['id']} blocker_{status.lower()} "
                f"score={record['score']:.4f}",
                args.quiet,
                log_prefix,
            )
        elif idx == 1 or idx % 10 == 0 or idx == len(rows):
            status = "OK" if refine.get("success") else "FAIL"
            _log(
                f"[{idx:03d}/{len(rows)}] {row['id']} synthetic_{status.lower()} "
                f"score={record['score']:.4f}",
                args.quiet,
                log_prefix,
            )

    synthetic_count = sum(1 for row in rows if row["source_type"] == "synthetic")
    blocker_count = sum(1 for row in rows if row["source_type"] == "real_blocker")
    report = {
        "generated_at": now_iso(),
        "model_id": model_id,
        "run_name": run_name,
        "image_count": len(rows),
        "detection_count": detection_count,
        "synthetic_box_iou_mean": None,
        "synthetic_downstream_warp_success_count": synthetic_warp_success,
        "synthetic_downstream_warp_success_rate": (float(synthetic_warp_success) / float(synthetic_count)) if synthetic_count else 0.0,
        "blocker_pass_count": blocker_pass,
        "blocker_pass_rate": (float(blocker_pass) / float(blocker_count)) if blocker_count else 0.0,
        "median_inference_seconds": median_or_zero(inference_times),
        "failed_ids": failed_ids,
        "results": results,
    }
    write_json(reports_dir / f"{run_name}.json", report)
    _log("=== RUN COMPLETE ===", args.quiet, log_prefix)
    _log(f"report_json={reports_dir / f'{run_name}.json'}", args.quiet, log_prefix)
    _log(f"blocker_pass={report['blocker_pass_count']}", args.quiet, log_prefix)
    _log(f"synthetic_warp_rate={report['synthetic_downstream_warp_success_rate']:.4f}", args.quiet, log_prefix)
    _log(f"median_sec={report['median_inference_seconds']:.4f}", args.quiet, log_prefix)
    return report


def _supports_parallel_gpu_runs(args: argparse.Namespace, runs: list[tuple[str, str]]) -> bool:
    return (
        args.all_models
        and not args.serial_models
        and len(runs) > 1
        and str(args.device).strip().lower() == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.device_count() > 1
    )


def _launch_parallel_model_runs(args: argparse.Namespace, runs: list[tuple[str, str]]) -> list[dict]:
    reports_dir = ensure_dir(Path(args.reports_dir))
    available_gpu_ids = list(range(torch.cuda.device_count()))
    pending: list[tuple[subprocess.Popen[str], int, str, str, Path]] = []
    queued = list(runs)
    summaries: list[dict] = []

    _log(f"parallel_gpu_slots={len(available_gpu_ids)}", args.quiet)

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
            if args.quiet:
                cmd.append("--quiet")

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
                    "blocker_pass_count": report["blocker_pass_count"],
                    "synthetic_downstream_warp_success_rate": report["synthetic_downstream_warp_success_rate"],
                    "median_inference_seconds": report["median_inference_seconds"],
                    "report_json": str(report_path),
                }
            )
            print(
                f"{run_name}: blocker_pass={report['blocker_pass_count']} "
                f"synthetic_warp_rate={report['synthetic_downstream_warp_success_rate']:.4f} "
                f"median_sec={report['median_inference_seconds']:.4f}",
                flush=True,
            )
        pending = still_pending
        if pending:
            time.sleep(1.0)

    return summaries


def main() -> int:
    args = parse_args()
    runs = []
    if args.all_models:
        for model_id in DEFAULT_MODELS:
            name = model_id.split("/")[-1].replace("-patch16-ensemble", "")
            runs.append((model_id, name))
    else:
        if not args.model_id:
            raise SystemExit("--model-id or --all-models is required")
        runs.append((args.model_id, args.run_name or args.model_id.split("/")[-1]))

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
                    "blocker_pass_count": report["blocker_pass_count"],
                    "synthetic_downstream_warp_success_rate": report["synthetic_downstream_warp_success_rate"],
                    "median_inference_seconds": report["median_inference_seconds"],
                    "report_json": str(Path(args.reports_dir) / f"{run_name}.json"),
                }
            )
            print(
                f"{run_name}: blocker_pass={report['blocker_pass_count']} "
                f"synthetic_warp_rate={report['synthetic_downstream_warp_success_rate']:.4f} "
                f"median_sec={report['median_inference_seconds']:.4f}",
                flush=True,
            )

    write_json(Path(args.reports_dir) / "summary.json", {"runs": summaries})
    _log(f"summary_json={Path(args.reports_dir) / 'summary.json'}", args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
