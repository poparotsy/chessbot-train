# Transfer Localizer Kaggle Run

This is a Kaggle-only benchmark.

No training.
No fine-tuning.
No local heavy run.

## 1. Clone The Repo In Kaggle

```python
!git clone https://github.com/poparotsy/chessbot-train.git /kaggle/working/chessbot-train
%cd /kaggle/working/chessbot-train
```

## 2. Install Deps And Export The Payload

```python
!python3 -m pip install -q torch torchvision pillow numpy opencv-python-headless python-chess "transformers==4.50.0" timm
!python3 scripts/transfer_localizer/export_eval_payload.py
```

This creates:

- `generated/transfer_localizer_v1/images/`
- `generated/transfer_localizer_v1/manifests/synthetic.jsonl`
- `generated/transfer_localizer_v1/manifests/real_blockers.jsonl`
- `generated/transfer_localizer_v1/manifests/all.jsonl`

## 3. Run All Detector Benchmarks

```python
!python3 scripts/transfer_localizer/benchmark_zero_shot_localizers.py --all-models --write-viz
```

This runs:

- `google/owlv2-base-patch16-ensemble`
- `IDEA-Research/grounding-dino-tiny`
- `IDEA-Research/grounding-dino-base`

## Output

Reports:

- `reports/transfer_localizer_v1/owlv2-base.json`
- `reports/transfer_localizer_v1/grounding-dino-tiny.json`
- `reports/transfer_localizer_v1/grounding-dino-base.json`
- `reports/transfer_localizer_v1/summary.json`

Overlays:

- `generated/transfer_localizer_v1/viz/`

## Winner Rule

Pick the winner by:

1. highest `blocker_pass_count`
2. then highest `synthetic_downstream_warp_success_rate`
3. then lowest `median_inference_seconds`
