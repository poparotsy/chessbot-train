# Transfer Localizer: Current State And Next Steps

## Short Answer

Yes, a new recognizer candidate exists now.

It is:

- `recognizer_transfer_localizer.py`

It is **not** the production recognizer yet.

It is a candidate runtime that:

- accepts a pluggable localizer
- supports `edge`, `zero-shot`, and `hybrid` localizer modes
- reuses the existing `recognizer_v6` decoder path
- keeps production `recognizer_v6.py` untouched

## What Was Built

### 1. Candidate recognizer runtime

- `recognizer_transfer_localizer.py`

Purpose:

- run the new two-stage recognizer design
- localizer first
- existing v6 decode second
- optional edge fallback if the learned/zero-shot localizer is weak

### 2. Shared localizer backend layer

- `scripts/transfer_localizer/localizers.py`

Purpose:

- normalize localizer outputs into one schema
- support zero-shot detector models
- support edge-localizer baseline
- provide common candidate selection

### 3. Manifest and dataset expansion

- `scripts/transfer_localizer/export_eval_payload.py`
- `scripts/transfer_localizer/common.py`
- `scripts/transfer_localizer/import_sam2_manifest.py`

Purpose:

- keep the old benchmark export flow
- also generate localizer training manifests
- accept SAM 2 propagated real-data manifests

Generated manifests now include:

- `manifests/localizer_train.jsonl`
- `manifests/localizer_val.jsonl`
- `manifests/localizer_all.jsonl`

### 4. Training-prep scripts

- `scripts/transfer_localizer/train_localizer.py`

Purpose:

- prepare fine-tune datasets for:
  - `RF-DETR`
  - `YOLO`
- optionally launch training if dependencies are installed

### 5. Evaluation and promotion scripts

- `scripts/transfer_localizer/benchmark_zero_shot_localizers.py`
- `scripts/transfer_localizer/compare_recognizers.py`

Purpose:

- benchmark zero-shot localizers
- generate `summary.json` and `summary.md`
- compare current `recognizer_v6.py` vs `recognizer_transfer_localizer.py`

### 6. Tests

- `scripts/tests_v6/test_transfer_localizer.py`

Purpose:

- smoke coverage for manifest normalization and comparison logic

## What Is Not Done Yet

These are still pending:

- actual fine-tuned `RF-DETR Nano` model training
- actual fine-tuned `YOLO-nano / YOLO11-nano` model training
- end-to-end benchmark proof that the candidate beats current v6 on blocker cases
- promotion of the candidate path into production `recognizer_v6.py`

So:

- the new recognizer framework is built
- the new trained production localizer is **not** built yet

## What To Run Next

### Path A: Validate the current candidate runtime

Run a quick edge-backed candidate smoke check:

```bash
python3 recognizer_transfer_localizer.py images_4_test/puzzle-00001.jpeg --localizer-source edge
```

Try hybrid mode if zero-shot deps are installed:

```bash
python3 recognizer_transfer_localizer.py images_4_test/puzzle-00001.jpeg \
  --localizer-source hybrid \
  --localizer-model-id IDEA-Research/grounding-dino-tiny
```

### Path B: Regenerate localizer payloads

```bash
python3 scripts/transfer_localizer/export_eval_payload.py
```

If you have SAM 2 propagated manifests:

```bash
python3 scripts/transfer_localizer/export_eval_payload.py \
  --sam2-jsonl path/to/sam2_real.jsonl
```

### Path C: Benchmark zero-shot models

```bash
python3 scripts/transfer_localizer/benchmark_zero_shot_localizers.py --all-models --write-viz
```

Expected outputs:

- `reports/transfer_localizer_v1/*.json`
- `reports/transfer_localizer_v1/summary.json`
- `reports/transfer_localizer_v1/summary.md`

### Path D: Prepare fine-tune data for RF-DETR

```bash
python3 scripts/transfer_localizer/train_localizer.py \
  --backend rfdetr \
  --run-name rf-detr-nano
```

### Path E: Prepare fine-tune data for YOLO

```bash
python3 scripts/transfer_localizer/train_localizer.py \
  --backend yolo \
  --run-name yolo11n-board
```

### Path F: Compare current recognizer vs candidate recognizer

```bash
python3 scripts/transfer_localizer/compare_recognizers.py
```

This is the main decision gate before promotion.

## Recommended Order

1. Run `export_eval_payload.py`
2. Run `benchmark_zero_shot_localizers.py`
3. If zero-shot is not enough, prepare `RF-DETR` fine-tune data
4. Train the localizer externally or in a proper ML environment
5. Point the candidate recognizer at the trained model backend
6. Run `compare_recognizers.py`
7. Promote only if blocker gains are real and hard/domain/stress do not regress

## Practical Decision

If you want the fastest serious next move, do this:

1. run the payload export
2. run the zero-shot benchmark
3. use the result to decide whether to invest directly in `RF-DETR Nano` fine-tuning

That is the shortest path from the current code to an actual bulletproof localizer decision.
