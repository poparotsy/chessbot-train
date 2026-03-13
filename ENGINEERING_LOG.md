# Engineering Log

This file is the mandatory change log for this repo.
Every commit must be documented with:

- `commit`
- `objective`
- `files`
- `behavior_change`
- `validation`
- `result`

---

## Entry

- commit: `c629882`
- objective: Add strict hardset guardrails and cleaner runtime visibility for v7 training.
- files:
  - `train_hybrid_v7.py`
  - `generate_hybrid_v7.py`
- behavior_change:
  - Added revision banners to both scripts.
  - Added hardset guard paths (at that time): reject zero-baseline, candidate gating logic, cleaner interrupt handling.
- validation:
  - `python3 -m py_compile train_hybrid_v7.py generate_hybrid_v7.py`
  - smoke training/ranking checks on temporary tensor data.
- result:
  - Commit pushed.
  - Later superseded by follow-up training policy changes.

## Entry

- commit: `1033116`
- objective: Set safer default v7 trainer paths and defaults for no-option runs.
- files:
  - `train_hybrid_v7.py`
- behavior_change:
  - Default model/checkpoint paths moved to non-`scratch` naming.
  - Resume default enabled.
  - Relaxed early hardset gating defaults at that time.
- validation:
  - `python3 -m py_compile train_hybrid_v7.py`
  - runtime startup log check for revised defaults.
- result:
  - Commit pushed.
  - Partially superseded by later simplification.

## Entry

- commit: `5cccf57`
- objective: Remove ranking/gating from v7 trainer loop and add dataset integrity validator.
- files:
  - `train_hybrid_v7.py`
  - `scripts/validate_tensors_v7.py`
- behavior_change:
  - Training selection now saves best model by validation score only.
  - Removed in-loop hardset/rank dependency from `train_hybrid_v7.py`.
  - Added strict tensor integrity validation script for `tensors_v7`.
- validation:
  - `python3 -m py_compile train_hybrid_v7.py scripts/validate_tensors_v7.py`
  - `python3 scripts/validate_tensors_v7.py --data-dir tensors_v7` (on Kaggle: passed with `OK`).
- result:
  - Commit pushed.
  - Current base for v7 trainer + data integrity checks.

## Entry

- commit: `pending`
- objective: Implement v6 recovery lane (standalone eval/generate/train/rank/validate) and add global low-saturation sparse decode re-score in `recognizer_v6`.
- files:
  - `recognizer_v6.py`
  - `generate_hybrid_v6.py`
  - `train_hybrid_v6.py`
  - `scripts/evaluate_v6_hardset.py`
  - `scripts/rank_models_v6.py`
  - `scripts/validate_tensors_v6.py`
- behavior_change:
  - Added structured debug JSON events in `recognizer_v6` for geometry candidates, candidate decode stats, and selected candidate (`--debug` only).
  - Added one global low-saturation sparse-board top-k re-score path in `recognizer_v6` (no image-ID hardcoding, deterministic fallback to existing path).
  - Added deterministic hardset evaluator for v6 with per-image timing and output reports:
    - `reports/v6_eval_latest.json`
    - `reports/v6_failures_latest.json`
  - Added external v6-only model ranker (`scripts/rank_models_v6.py`) with compatibility checks.
  - Added standalone v6 generator (`generate_hybrid_v6.py`) with targeted profile mix and manifest output (`generation_manifest_v6.json`).
  - Added standalone v6 trainer (`train_hybrid_v6.py`) with validation-only best-checkpoint selection, deterministic checkpoint path, resume support, and clean interrupt checkpoint save.
  - Added `scripts/validate_tensors_v6.py` for v6 tensor integrity + manifest consistency checks.
- validation:
  - `python3 -m py_compile recognizer_v6.py generate_hybrid_v6.py train_hybrid_v6.py scripts/evaluate_v6_hardset.py scripts/rank_models_v6.py scripts/validate_tensors_v6.py`
  - `BOARDS_PER_CHUNK=4 CHUNKS_TRAIN=1 CHUNKS_VAL=1 OUTPUT_DIR=/tmp/tensors_v6_smoke python3 generate_hybrid_v6.py`
  - `python3 scripts/validate_tensors_v6.py --data-dir /tmp/tensors_v6_smoke`
  - `python3 - <<'PY' ... import train_hybrid_v6 as t; t.DATA_DIR='/tmp/tensors_v6_smoke'; t.EPOCHS=1; t.train() ... PY`
  - `python3 scripts/rank_models_v6.py --models-glob "models/model_hybrid_v5_latest_best.pt" --truth-json /tmp/truth_v6_smoke.json --images-dir images_4_test`
  - `python3 recognizer_v6.py images_4_test/puzzle-00028.jpeg --model-path models/model_hybrid_v5_latest_best.pt --debug`
  - `python3 - <<'PY' ... read reports/v6_eval_latest.json for 00024/00028/00041/00046/00049 ... PY`
- result:
  - Baseline lock preserved at board `48/49` on `truth_verified.json` (remaining board miss: `puzzle-00028.jpeg`).
  - Target single-image checks from report:
    - `00024` pass
    - `00028` fail
    - `00041` board pass (full-fen side-to-move mismatch)
    - `00046` pass
    - `00049` pass
  - v6 data/training lane is now standalone and ready for targeted runs.

## Entry

- commit: `pending`
- objective: Add a deterministic post-change v6 test system (quick gate + internal invariants + relative performance baseline).
- files:
  - `scripts/test_after_change_v6.py`
  - `scripts/benchmark_v6.py`
  - `scripts/tests_v6/__init__.py`
  - `scripts/tests_v6/test_internal_v6.py`
  - `scripts/testdata/v6_quick_cases.json`
  - `scripts/testdata/v6_orientation_cases.json`
- behavior_change:
  - Added a single default post-change command for v6:
    - `python3 scripts/test_after_change_v6.py --profile quick`
  - Added orientation regression checks with forced `white`/`black` consistency and `auto` validation.
  - Added quick hardset baseline gating (`no drop` vs stored baseline pass count).
  - Added internal invariant tests (`unittest`) for:
    - side-to-move alias parsing,
    - FEN 180-rotation involution,
    - batched deep-topk tile inference call count,
    - decode-candidate cap enforcement,
    - timeout-safe evaluator behavior.
  - Added performance benchmark + relative baseline check keyed by machine fingerprint:
    - median latency ratio gate,
    - p95 latency ratio gate,
    - timeout-free requirement.
- validation:
  - `python3 -m py_compile scripts/benchmark_v6.py scripts/test_after_change_v6.py scripts/tests_v6/test_internal_v6.py`
  - `python3 -m unittest discover -s scripts/tests_v6 -p 'test_*.py' -v`
  - `python3 scripts/benchmark_v6.py --update-baseline --model-path models/model_hybrid_v5_latest_best.pt`
  - `python3 scripts/test_after_change_v6.py --profile quick --update-quick-baseline --update-perf-baseline --model-path models/model_hybrid_v5_latest_best.pt`
  - `python3 scripts/test_after_change_v6.py --profile quick --model-path models/model_hybrid_v5_latest_best.pt`
- result:
  - Quick gate is operational and repeatable.
  - Current quick baseline initialized at `11/12` (remaining miss: `puzzle-00028.jpeg`).
  - Performance baseline check passes on this machine fingerprint after bootstrap.

## Entry

- commit: `pending`
- objective: Add `orientation_best_guess` as a last-resort auto-POV heuristic in `recognizer_v6`, following a global pros-style fallback stack.
- files:
  - `recognizer_v6.py`
  - `scripts/tests_v6/test_internal_v6.py`
- behavior_change:
  - Auto orientation in `recognizer_v6` now uses this fallback order:
    1. board labels (`a`/`h`) if high-confidence,
    2. strong piece-distribution margin,
    3. `orientation_best_guess` (global pawn/king/check structural scoring),
    4. default.
  - Added strict low-signal guards so best-guess does not trigger on sparse/ambiguous boards.
  - Added internal tests to lock low-signal suppression and high-signal guess behavior.
- validation:
  - `python3 -m py_compile recognizer_v6.py`
  - `python3 -m unittest discover -s scripts/tests_v6 -p 'test_*.py' -v`
  - `python3 scripts/test_after_change_v6.py --profile quick --model-path models/model_hybrid_v5_latest_best.pt`
- result:
  - Quick gate remains green at `11/12` (only `00028` still failing in quick set).
  - Recovered from intermediate regression on sparse boards (`00021`/`00049`) by tightening best-guess activation thresholds.
