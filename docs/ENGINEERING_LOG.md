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

- commit: `this commit`
- objective: Record persistent guardrail that v7 remains experimental/failing unless it beats locked hardset baseline.
- files:
  - `MIND.md`
  - `ENGINEERING_LOG.md`
- behavior_change:
  - Added a permanent project-memory rule:
    - v7 is experimental/failing by default,
    - no runtime default switch to v7,
    - any v7 claim must show hardset proof against `images_4_test/truth_verified.json`.
- validation:
  - `sed -n '1,120p' MIND.md` (manual verify guardrail exists)
- result:
  - Guardrail added to persistent memory list; future work must explicitly prove v7 before promotion.

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

## Entry

- commit: `pending`
- objective: Stabilize `recognizer_v6` auto-orientation after fallback regressions and remove the recent hang-prone behavior path from default runtime decisions.
- files:
  - `recognizer_v6.py`
- behavior_change:
  - Disabled `orientation_best_guess` in active runtime path by default (`ORIENTATION_BEST_GUESS_ENABLED = False`).
  - Tightened piece-distribution fallback usage in auto mode:
    - only eligible when label signal is absent or ambiguous (`labels_absent || labels_same`),
    - skipped when opposite-side weak labels exist.
  - Added weak-label fallback for `a/h` detection with confidence threshold:
    - `ORIENTATION_WEAK_LABEL_MIN_CONF = 0.70`,
    - if left/right classify as `h/a` or `a/h` above threshold, orientation is set before piece-distribution fallback.
- validation:
  - `python3 -m py_compile recognizer_v6.py`
  - `python3 recognizer_v6.py images_4_test/puzzle-00014.jpeg --model-path models/model_hybrid_v5_latest_best.pt`
  - `python3 recognizer_v6.py images_4_test/puzzle-00019.jpeg --model-path models/model_hybrid_v5_latest_best.pt`
  - `python3 recognizer_v6.py images_4_test/puzzle-00030.jpeg --model-path models/model_hybrid_v5_latest_best.pt`
  - `python3 recognizer_v6.py images_4_test/puzzle-00050.jpeg --model-path models/model_hybrid_v5_latest_best.pt`
  - `python3 scripts/rank_models_v6.py --models-glob "models/model_hybrid_v5_latest_best.pt" --truth-json images_4_test/truth_verified.json`
- result:
  - Recovered orientation regressions:
    - `00014` PASS (was auto-flipped),
    - `00019` PASS (was auto-flipped),
    - `00050` PASS (was regressed during fallback tightening),
    - `00030` remains PASS.
  - Intermediate hardset after this patch: `48/50` (`00028`, `00044` pending).

## Entry

- commit: `pending`
- objective: Fix residual geometry-selection miss on `00044` using a global panel-alignment rule and verify impact against neighboring hard cases (`00046`, `00047`).
- files:
  - `recognizer_v6.py`
- behavior_change:
  - Added directional trim-then-resize candidate for `panel_split_*` detections:
    - `PANEL_DIRECTIONAL_TRIM_FRAC = 0.08`
    - emits candidate tags like `panel_split_top_trim8`, `panel_split_left_trim8`.
  - Added `directional_trim_resize(...)` helper (global, non-image-specific).
  - Added candidate-selection penalty against `default_double_check_conflict` boards as a tie-breaker only (does not override plausibility-first ranking).
- validation:
  - `python3 -m py_compile recognizer_v6.py`
  - `python3 recognizer_v6.py images_4_test/puzzle-00044.jpeg --model-path models/model_hybrid_v5_latest_best.pt --debug`
  - `python3 recognizer_v6.py images_4_test/puzzle-00046.jpeg --model-path models/model_hybrid_v5_latest_best.pt --debug`
  - `python3 recognizer_v6.py images_4_test/puzzle-00047.jpeg --model-path models/model_hybrid_v5_latest_best.pt --debug`
  - `python3 scripts/rank_models_v6.py --models-glob "models/model_hybrid_v5_latest_best.pt" --truth-json images_4_test/truth_verified.json`
- result:
  - `00044` now PASS via `panel_split_top_trim8` candidate.
  - `00046` and `00047` pass via contour-based candidates (not dependent on the new `panel_split_*_trim8` path).
  - Current full hardset score with v6 recognizer + v5 model: `49/50`.
  - Remaining miss: `00028`.

## Entry

- commit: `pending`
- objective: Hard cleanup of `recognizer_v6` with zero behavior change: remove legacy shadow entrypoint, centralize config, split active runtime into detector/decode/orientation/selection helpers, and document architecture.
- files:
  - `recognizer_v6.py`
  - `scripts/tests_v6/test_internal_v6.py`
  - `V6_RECOGNIZER_ARCHITECTURE.md`
- behavior_change:
  - Removed duplicate legacy `predict_board` implementation (shadowed entrypoint).
  - Centralized runtime config in the top-level constants block (removed mid-file redefinition block).
  - Refactored active runtime path into helper functions:
    - `build_detector_candidates`
    - `collect_orientation_context`
    - `resolve_candidate_orientation`
    - `decode_candidate`
    - `select_best_candidate`
    - `king_health`
  - Kept CLI/API unchanged.
  - Added helper-level unit tests for detector/orientation/selection modules.
  - Added helper-level unit test for decode module (`decode_candidate`) covering low-saturation rescore + orientation application.
  - Added architecture documentation describing stage boundaries and test mapping.
- validation:
  - `python3 -m py_compile recognizer_v6.py scripts/tests_v6/test_internal_v6.py scripts/test_after_change_v6.py scripts/evaluate_v6_hardset.py`
  - `python3 -m unittest -q scripts.tests_v6.test_internal_v6`
  - `python3 scripts/test_after_change_v6.py --profile quick --model-path models/model_hybrid_v5_latest_best.pt`
  - `python3 scripts/evaluate_v6_hardset.py --truth-json images_4_test/truth_verified.json --model-path models/model_hybrid_v5_latest_best.pt`
- result:
  - Internal tests: `12/12` PASS.
  - Quick gate: PASS (`11/12`, same expected miss `00028`).
  - Full hardset: unchanged at `49/50` board, `47/50` full-fen.
  - No cross-version delegation added; v6 runtime stays standalone.

## Entry

- commit: `pending`
- objective: Remove ambiguous model fallback behavior and make v6 default model selection deterministic.
- files:
  - `recognizer_v6.py`
  - `scripts/tests_v6/test_internal_v6.py`
  - `V6_RECOGNIZER_ARCHITECTURE.md`
- behavior_change:
  - Replaced multi-directory/multi-file model auto-search block with deterministic default path:
    - `models/model_hybrid_v6_latest_best.pt`
  - Added explicit overrides only via:
    - `--model-path`
    - `CHESSBOT_MODEL_PATH`
  - Added explicit checkpoint existence check in `predict_board` with clear error text.
  - Kept runtime behavior unchanged on the validated default setup.
  - Updated candidate-cap unit test for new path existence guard.
  - Documented deterministic model resolution in v6 architecture doc.
- validation:
  - `python3 -m py_compile recognizer_v6.py scripts/tests_v6/test_internal_v6.py`
  - `python3 -m unittest -q scripts.tests_v6.test_internal_v6`
  - `python3 scripts/test_after_change_v6.py --profile quick --model-path models/model_hybrid_v5_latest_best.pt`
- result:
  - Internal tests: `12/12` PASS.
  - Quick gate: PASS (`11/12`, unchanged baseline).
  - Perf gate: PASS.

## Entry

- commit: `pending`
- objective: Add a global low-saturation contrast enhancement helper for scan/book-like boards and evaluate impact on v6 hard cases.
- files:
  - `recognizer_v6.py`
  - `scripts/tests_v6/test_internal_v6.py`
  - `V6_RECOGNIZER_ARCHITECTURE.md`
- behavior_change:
  - Added `enhance_low_saturation_image(...)` (CLAHE + mild unsharp) with profile guards:
    - `LOW_SAT_ENHANCE_SAT_MAX`
    - `LOW_SAT_ENHANCE_VAL_STD_MAX`
  - Integrated enhancement into detector candidate generation:
    - adds enhanced-source candidate set (`*_enhsrc`) for low-saturation inputs.
    - keeps original `full` candidate first to preserve deterministic orientation context.
  - Added unit tests for:
    - enhanced-source candidate injection
    - colorful-input skip behavior.
- validation:
  - `python3 -m py_compile recognizer_v6.py scripts/tests_v6/test_internal_v6.py`
  - `python3 -m unittest -q scripts.tests_v6.test_internal_v6`
  - `python3 recognizer_v6.py images_4_test/puzzle-00028.jpeg --model-path models/model_hybrid_v5_latest_best.pt --debug`
  - `python3 scripts/test_after_change_v6.py --profile quick --model-path models/model_hybrid_v5_latest_best.pt`
- result:
  - Internal tests: `14/14` PASS.
  - Quick gate: PASS (`11/12`, unchanged miss `00028`).
  - Enhancement is active on `00028` (debug confirms), but final board prediction did not improve yet.

## Entry

- commit: `pending`
- objective: Diagnose `00028` root cause and add a dedicated v6 data profile for mono print edge-rook boards.
- files:
  - `generate_hybrid_v6.py`
  - `ENGINEERING_LOG.md`
- behavior_change:
  - Added new profile: `mono_rook_scan`.
  - Added new recipe: `v6_00028_recovery_v2`.
  - Recipe mix:
    - `clean=0.20`
    - `mono_scan=0.20`
    - `mono_rook_scan=0.45`
    - `edge_frame=0.10`
    - `hard_combo=0.05`
- validation:
  - `python3 -m py_compile generate_hybrid_v6.py`
  - `RECIPE_NAME=v6_00028_recovery_v2 BOARDS_PER_CHUNK=20 CHUNKS_TRAIN=1 CHUNKS_VAL=1 OUTPUT_DIR=/tmp/tensors_v6_00028_smoke python3 generate_hybrid_v6.py`
  - `python3 scripts/validate_tensors_v6.py --data-dir /tmp/tensors_v6_00028_smoke`
- result:
  - Generation smoke passed and manifest includes `mono_rook_scan`.
  - Data validation passed.
  - Diagnostic conclusion on `00028`: geometry candidates are present, but rook logits remain weak on critical squares; this points to model/domain gap more than edge-quad failure.

## Entry

- commit: `pending`
- objective: Add projection-based board detector path (Sobel gradient projections) and stabilize candidate selection for mono/print hard cases.
- files:
  - `recognizer_v6.py`
  - `scripts/tests_v6/test_internal_v6.py`
  - `V6_RECOGNIZER_ARCHITECTURE.md`
- behavior_change:
  - Added `BoardDetector.detect_gradient_projection(...)` candidate.
  - Wired `gradient_projection` into `lens_hypotheses`.
  - Added detector unit test for gradient-projection path (`puzzle-00028` fixture).
  - Added small confidence prior penalty for `gradient_projection*` tags during final selection to avoid over-selection vs contour/gfit on non-target cases.
- validation:
  - `python3 -m py_compile recognizer_v6.py scripts/tests_v6/test_internal_v6.py`
  - `python3 -m unittest -q scripts.tests_v6.test_internal_v6`
  - `python3 recognizer_v6.py images_4_test/puzzle-00028.jpeg --model-path models/model_hybrid_v5_latest_best.pt`
  - `python3 recognizer_v6.py images_4_test/puzzle-00049.jpeg --model-path models/model_hybrid_v5_latest_best.pt`
- result:
  - `00028` improved from missing rooks to near-correct geometry-aligned decode.
  - `00049` remained correct after selection prior adjustment.

## Entry

- commit: `pending`
- objective: Strengthen low-saturation sparse rescoring to consider multiple top-k alternatives and recover missing edge rooks globally.
- files:
  - `recognizer_v6.py`
  - `V6_RECOGNIZER_ARCHITECTURE.md`
- behavior_change:
  - `rescore_low_saturation_sparse_from_topk` now evaluates up to `LOW_SAT_SPARSE_ALT_OPTIONS` alternatives per empty square (beam options), not only the single top non-empty class.
  - Added global edge-rook objective term (`LOW_SAT_EDGE_ROOK_OBJECTIVE_BONUS`) in sparse rescoring.
  - Added guarded missing-edge-rook promotion inside sparse rescoring when one color has zero rooks and opposite color still has rooks.
  - Added queen-count sanity penalties in `board_plausibility_score` to de-prioritize implausible multi-queen hallucinations.
- validation:
  - `python3 -m py_compile recognizer_v6.py`
  - `python3 -m unittest -q scripts.tests_v6.test_internal_v6`
  - `python3 scripts/test_after_change_v6.py --profile quick --model-path models/model_hybrid_v5_latest_best.pt`
  - `python3 scripts/evaluate_v6_hardset.py --truth-json images_4_test/truth_verified.json --model-path models/model_hybrid_v5_latest_best.pt`
- result:
  - Quick gate: `12/12` board pass.
  - Full hardset: `50/50` board pass, `48/50` full-fen (side-to-move heuristic remains separate).
  - `puzzle-00028` now matches expected board FEN.

## Entry

- commit: `pending`
- objective: Build a deterministic image-by-image path analysis for `recognizer_v6` and capture exact selected pipeline branch per puzzle.
- files:
  - `recognizer_v6.py`
  - `scripts/analyze_v6_paths.py`
  - `reports/v6_path_analysis_latest.json`
  - `reports/v6_path_analysis_latest.md`
- behavior_change:
  - No inference behavior change.
  - Added debug-only telemetry events:
    - `v6_candidate_pool` (candidate tags/count after cap logic),
    - richer `v6_candidate_decode` payload (`fen_board`, `plausibility`, `king_health`, `stm_source`, `conf_adj`).
  - Added `scripts/analyze_v6_paths.py`:
    - runs `recognizer_v6.py --debug` per image,
    - parses `DEBUG_JSON` events,
    - records full per-image function path, selected candidate family/tag, selection top-3, and risk flags.
- validation:
  - `python3 -m py_compile recognizer_v6.py scripts/analyze_v6_paths.py`
  - `python3 scripts/analyze_v6_paths.py --model-path models/model_hybrid_v5_latest_best.pt --truth-json images_4_test/truth_verified.json --images-dir images_4_test --output-json reports/v6_path_analysis_latest.json --output-md reports/v6_path_analysis_latest.md`
- result:
  - `board_pass=50/50`
  - `full_pass=48/50`
  - Selected family distribution:
    - `full=25`, `contour=13`, `lattice=7`, `panel_split=3`, `axis_grid=1`, `gradient_projection=1`.
  - High-leverage fragile paths are now explicit in report (not hidden): `00028`, `00031`, `00044`, `00046`, `00049`.

## Entry

- commit: `pending`
- objective: Configure the next standalone `v6` training lane around the mono/print/edge-rook family, with a smaller logo bucket and isolated artifacts.
- files:
  - `generate_hybrid_v6.py`
  - `train_hybrid_v6.py`
- behavior_change:
  - Generator defaults now target a new isolated output directory:
    - `OUTPUT_DIR=tensors_v6_mono_logo`
  - Added `logo_overlay` generation profile for `00024`-style square/logo interference, with a little local piece tilt.
  - Switched default recipe to `v6_mono_logo_recovery_v1`:
    - `mono_print_sparse_edge=0.42`
    - `mono_scan=0.24`
    - `logo_overlay=0.14`
    - `edge_frame=0.10`
    - `clean=0.07`
    - `hard_combo=0.03`
  - Trainer defaults now use an isolated run lane:
    - `DATA_DIR=tensors_v6_mono_logo`
    - `MODEL_SAVE_PATH=models/model_hybrid_v6_mono_logo_latest_best.pt`
    - `FINAL_MODEL_SAVE_PATH=models/model_hybrid_v6_mono_logo_final.pt`
    - `CHECKPOINT_DIR=models/checkpoints_v6_mono_logo`
  - Trainer warm-start remains on the current reference base model:
    - `BASE_MODEL_PATH=models/model_hybrid_v5_latest_best.pt`
  - Trainer now starts a fresh run by default:
    - `RESUME_FROM_CHECKPOINT=False`
    - `EPOCHS=120`
    - `LEARNING_RATE=2e-6`
- validation:
  - `python3 -m py_compile generate_hybrid_v6.py train_hybrid_v6.py`
  - `python3 - <<'PY' ... import generate_hybrid_v6 ... print(OUTPUT_DIR, RECIPE_NAME, DEFAULT_PROFILE_WEIGHTS) ... PY`
  - `python3 - <<'PY' ... import train_hybrid_v6 ... print(DATA_DIR, MODEL_SAVE_PATH, CHECKPOINT_DIR, BASE_MODEL_PATH, EPOCHS, LEARNING_RATE, RESUME_FROM_CHECKPOINT) ... PY`
- result:
  - next Kaggle run is isolated from older v6 tensors/models/checkpoints
  - default recipe is now weighted toward the `00003/00028/00031` mono-print family while still including a smaller `00024` logo bucket
  - trainer will warm-start from `model_hybrid_v5_latest_best.pt` unless you override it

## Entry

- commit: `pending`
- objective: Rename the recovery profile so the generator config describes the actual image family instead of implying piece-class targeting.
- files:
  - `generate_hybrid_v6.py`
- behavior_change:
  - Renamed profile key:
    - `mono_rook_scan` -> `mono_print_sparse_edge`
  - Updated all recipe references to the new name.
- validation:
  - `python3 -m py_compile generate_hybrid_v6.py`
  - `python3 - <<'PY' ... import generate_hybrid_v6 ... print(RECIPE_NAME, DEFAULT_PROFILE_WEIGHTS, profiles_ok) ... PY`
- result:
  - default recovery recipe now reads as:
    - `mono_print_sparse_edge=0.42`
    - `mono_scan=0.24`
    - `logo_overlay=0.14`
    - `edge_frame=0.10`
    - `clean=0.07`
    - `hard_combo=0.03`

## Entry

- commit: `pending`
- objective: Replace the incorrect gray-filter mono lane with a tile-safe printed-board renderer that actually matches the `00003/00028/00031` family.
- files:
  - `generate_hybrid_v6.py`
  - `scripts/generate_samples.py`
  - `scripts/build_v6_mono_assets.py`
  - `board_themes/mono_paper_scan_light.png`
  - `board_themes/mono_paper_scan_mid.png`
  - `board_themes/mono_heather_print.png`
  - `piece_sets/mono_print_scan/*`
  - `piece_sets/mono_print_faded/*`
- behavior_change:
  - `mono_print_sparse_edge` and `mono_scan` no longer rely on the old scene-augmentation path for their core printed-board samples.
  - Added a tile-safe procedural print renderer with three diagram styles:
    - `print_hatched_book`
    - `print_shirt_print`
    - `print_flat_book`
  - Printed-board styles keep the 8x8 board aligned before slicing, avoiding the label noise caused by rotating/perspective-warping/trimming the full board before tile extraction.
  - Added dedicated mono/print board themes and piece sets for local preview and fallback asset-based use.
  - Updated `scripts/generate_samples.py` to support `--version v5|v6` and to preview the active v6 generator path instead of being hard-wired to v5.
- validation:
  - `python3 -m py_compile generate_hybrid_v6.py scripts/generate_samples.py scripts/build_v6_mono_assets.py`
  - `python3 scripts/build_v6_mono_assets.py`
  - `python3 scripts/generate_samples.py --version v6 --profile mono-print-sparse-edge --count 12`
  - visual inspection of generated samples against `images_4_test/puzzle-00003.jpeg`, `images_4_test/puzzle-00028.jpeg`, and `images_4_test/puzzle-00031.jpeg`
- result:
  - current `mono_print_sparse_edge` samples are now in the printed mono family instead of “clean synthetic board but gray”
  - generator is ready for regenerating the next v6 mono/logo dataset
  - existing tensors generated before this change should be treated as stale for the mono/logo lane

## Entry

- commit: `pending`
- objective: Move legacy v5 scripts under `scripts/v5/` without breaking direct execution from the new location.
- files:
  - `generate_hybrid_v5.py` -> `scripts/v5/generate_hybrid_v5.py`
  - `recognizer_v5.py` -> `scripts/v5/recognizer_v5.py`
  - `train_hybrid_v5.py` -> `scripts/v5/train_hybrid_v5.py`
- behavior_change:
  - Relocated legacy v5 generator, recognizer, and trainer into `scripts/v5/`.
  - Updated the moved scripts to resolve board themes, piece sets, models, tensors, and rank/eval script paths from the project root instead of the script directory.
  - No intended model/runtime behavior change beyond preserving the old entrypoints from the new folder.
- validation:
  - `python3 -m py_compile scripts/v5/generate_hybrid_v5.py scripts/v5/recognizer_v5.py scripts/v5/train_hybrid_v5.py`
  - `python3 scripts/v5/recognizer_v5.py --help`
  - `python3 - <<'PY' ... import moved v5 generator/trainer and print resolved root paths ... PY`
- result:
  - direct v5 CLI usage from `scripts/v5/` now resolves root assets/models correctly
  - top-level v5 files can be removed as part of the move commit

## Entry

- commit: `pending`
- objective: Save the current working v6 runtime and external evaluation helpers, while leaving the stale internal quick-gate script out of the commit.
- files:
  - `recognizer_v6.py`
  - `scripts/benchmark_v6.py`
  - `scripts/evaluate_v6_hardset.py`
  - `scripts/rank_models_v6.py`
- behavior_change:
  - Saved the current parity-style `recognizer_v6.py` runtime path.
  - Added `--script` override support to the external v6 helper tools so they can evaluate alternate recognizer modules without hard-wiring `recognizer_v6.py`.
  - Left `scripts/test_after_change_v6.py` and the internal unittest suite out of this save point because those tests no longer match the current recognizer API.
- validation:
  - `python3 -m py_compile recognizer_v6.py scripts/benchmark_v6.py scripts/evaluate_v6_hardset.py scripts/rank_models_v6.py`
  - `python3 scripts/evaluate_v6_hardset.py --truth-json images_4_test/truth_verified.json --model-path models/model_hybrid_v5_latest_best.pt --timeout-sec 45`
  - `python3 scripts/rank_models_v6.py --models-glob "models/model_hybrid_v5_latest_best.pt" --truth-json images_4_test/truth_verified.json`
  - `python3 scripts/benchmark_v6.py --model-path models/model_hybrid_v5_latest_best.pt --timeout-sec 45`
- result:
  - `evaluate_v6_hardset.py`: `board_pass=50/50`, `full_pass=48/50`
  - `rank_models_v6.py`: `models/model_hybrid_v5_latest_best.pt -> 50/50 (avg_conf=0.9976)`
  - `benchmark_v6.py`: `median=3.018s`, `p95=4.143s`, `timeouts=0`, baseline check passed

## Entry

- commit: `pending`
- objective: Remove Pillow 13 deprecation warnings from the v6 generator before the next Kaggle run.
- files:
  - `generate_hybrid_v6.py`
- behavior_change:
  - Replaced `Image.fromarray(..., "L")` and `Image.fromarray(..., "RGBA")` calls with `Image.fromarray(...).convert(...)`.
  - No intended rendering behavior change; this is compatibility cleanup only.
- validation:
  - `python3 -m py_compile generate_hybrid_v6.py`
  - `python3 scripts/generate_samples.py --version v6 --profile mono-print-sparse-edge --count 1 --output-dir /tmp/v6_warn_smoke`
- result:
  - v6 generator remains runnable
  - smoke sample generated at `/tmp/v6_warn_smoke/sample_v6_001.png`
  - deprecated `mode` argument usage removed from `generate_hybrid_v6.py`

## Entry

- commit: `pending`
- objective: Prevent late v6 training crashes on corrupted tensor chunks and make the failure actionable.
- files:
  - `generate_hybrid_v6.py`
  - `train_hybrid_v6.py`
- behavior_change:
  - Added atomic chunk saving to the v6 generator via temp file + `os.replace`, so interrupted generation cannot leave a truncated final `.pt` archive behind.
  - Added dataset preflight to the v6 trainer so every `train_*.pt` and `val_*.pt` chunk is loaded and shape-checked before training starts.
  - Wrapped runtime chunk loads in a shared helper that raises a direct file-specific error and points to `scripts/validate_tensors_v6.py`.
- validation:
  - `python3 -m py_compile train_hybrid_v6.py generate_hybrid_v6.py`
  - `python3 - <<'PY' ... import train_hybrid_v6, create corrupt temp .pt, call load_tensor_chunk(...) ... PY`
- result:
  - corrupted chunks now fail early with an explicit file path and validator command
  - future v6 generation runs write chunk files atomically instead of directly to the final archive path

## Entry

- commit: `pending`
- objective: Prepare a safer next v6 mono/logo training lane and make trainer ranking/resume behavior explicit.
- files:
  - `generate_hybrid_v6.py`
  - `train_hybrid_v6.py`
  - `scripts/rank_models_v6.py`
- behavior_change:
  - Added `v6_mono_logo_recovery_v2` with more anchor coverage and less destructive mono corruption, and moved the default v6 generator output to `tensors_v6_mono_logo_v2`.
  - Moved the default v6 trainer outputs/checkpoints to the matching `*_v2` lane while keeping the warm-start base model on `models/model_hybrid_v5_latest_best.pt`.
  - Enabled resume-by-default in the v6 trainer through `RESUME_FROM_CHECKPOINT`, added an explicit startup line showing resume mode, and added optional `RUN_RANK_AFTER_BEST` support to run `scripts/rank_models_v6.py` after a new best save.
  - Updated `scripts/rank_models_v6.py` to print model miss lists immediately after each score line and include the misses in the final JSON summary.
- validation:
  - `python3 -m py_compile generate_hybrid_v6.py train_hybrid_v6.py scripts/rank_models_v6.py`
  - `python3 - <<'PY' ... import generate_hybrid_v6/train_hybrid_v6 and print recipe/output/model/checkpoint defaults ... PY`
  - `python3 scripts/generate_samples.py --version v6 --profile mono-print-sparse-edge --count 4 --output-dir /tmp/v6_recipe_probe`
- result:
  - next v6 run is isolated to `tensors_v6_mono_logo_v2` / `models/model_hybrid_v6_mono_logo_v2_*`
  - trainer can resume from `latest.pt` by default and optionally auto-rank on each best save
  - rank output now prints the miss list without needing to scroll through the per-image progress log

## Entry

- commit: `pending`
- objective: Save a temporary snapshot of the standalone `recognizer_v6_clean.py` experiment and its parity/eval artifacts without mixing it into the active v6 recipe work.
- files:
  - `recognizer_v6_clean.py`
  - `scripts/compare_v6_parity_fast.py`
  - `reports/v6_benchmark_latest.json`
  - `reports/v6_clean_parity_fast.json`
  - `reports/v6_clean_parity_fast.log`
  - `reports/v6_eval_baseline_old.log`
  - `reports/v6_eval_latest.json`
  - `reports/v6_failures_latest.json`
  - `reports/v6_perf_baselines.json`
  - `reports/v6_quick_baseline.json`
- behavior_change:
  - No runtime promotion. This commit only preserves the temporary clean-rewrite candidate, the parity helper, and the evaluation/benchmark artifacts associated with that experiment.
- validation:
  - `python3 -m py_compile recognizer_v6_clean.py scripts/compare_v6_parity_fast.py`
- result:
  - temp v6 clean/parity work is preserved in git without touching the active `recognizer_v6.py` path

## Entry

- commit: `pending`
- objective: Prepare a more conservative `v3` mono/logo recovery lane after the `v2` run plateaued at `46/50` with drift between miss sets.
- files:
  - `generate_hybrid_v6.py`
  - `train_hybrid_v6.py`
- behavior_change:
  - Added two new v6 data buckets: `dark_anchor` for ordinary darker real-board anchors and `tilt_anchor` for explicit tilted-piece robustness.
  - Switched the default generator recipe to `v6_mono_logo_recovery_v3` and the default tensor output to `tensors_v6_mono_logo_v3`.
  - Switched the default trainer lane to `model_hybrid_v6_mono_logo_v3_*` and `checkpoints_v6_mono_logo_v3`, keeping the warm-start base model on `models/model_hybrid_v5_latest_best.pt`.
  - New `v3` recipe mix:
    - `clean 0.24`
    - `dark_anchor 0.20`
    - `mono_print_sparse_edge 0.18`
    - `mono_scan 0.12`
    - `edge_frame 0.12`
    - `logo_overlay 0.07`
    - `tilt_anchor 0.05`
    - `hard_combo 0.02`
- validation:
  - `python3 -m py_compile generate_hybrid_v6.py train_hybrid_v6.py`
  - `python3 - <<'PY' ... import generate_hybrid_v6/train_hybrid_v6 and print recipe/output/model/checkpoint defaults ... PY`
  - `python3 scripts/generate_samples.py --version v6 --profile dark-anchor --count 3 --output-dir /tmp/v6_dark_anchor_probe`
  - `python3 scripts/generate_samples.py --version v6 --profile tilt-anchor --count 3 --output-dir /tmp/v6_tilt_anchor_probe`
- result:
  - next v6 experiment is isolated to the `*_v3` lane and is anchored more heavily toward ordinary dark boards, moderate mono print, and explicit tilt robustness

## Entry

- commit: `pending`
- objective: Prepare a narrower `v4` recovery lane after direct checkpoint analysis showed `00005/00017` were anti-regressions and `00028/00031` were still dropping sparse edge pieces as empty.
- files:
  - `generate_hybrid_v6.py`
  - `train_hybrid_v6.py`
- behavior_change:
  - Added `dark_anchor_clean` as a true ordinary-board anchor bucket with near-zero watermark, trim, sparse bias, mono degradation, and tilt so dark real boards stop being “unlearned”.
  - Split the mono target into two lighter buckets:
    - `mono_print_sparse_light` for the general `00003/00028/00031` family with much lower structural damage and edge fade
    - `mono_print_edge_rook` for narrower `00028`-style faint edge-rook boards without the heavy board-emptying damage from earlier lanes
  - Switched the default generator recipe to `v6_mono_logo_recovery_v4` and the default tensor output to `tensors_v6_mono_logo_v4`.
  - Switched the default trainer lane to `model_hybrid_v6_mono_logo_v4_*` and `checkpoints_v6_mono_logo_v4`, keeping the warm-start base model on `models/model_hybrid_v5_latest_best.pt`.
  - New `v4` recipe mix:
    - `clean 0.30`
    - `dark_anchor_clean 0.30`
    - `mono_print_sparse_light 0.18`
    - `mono_print_edge_rook 0.12`
    - `mono_scan 0.06`
    - `tilt_anchor 0.03`
    - `logo_overlay 0.01`
- validation:
  - `python3 -m py_compile generate_hybrid_v6.py train_hybrid_v6.py`
  - `python3 - <<'PY' ... import generate_hybrid_v6/train_hybrid_v6 and print recipe/output/model/checkpoint defaults ... PY`
  - `python3 scripts/generate_samples.py --version v6 --profile dark-anchor-clean --count 3 --output-dir /tmp/v6_dark_anchor_clean_probe`
  - `python3 scripts/generate_samples.py --version v6 --profile mono-print-sparse-light --count 3 --output-dir /tmp/v6_mono_sparse_light_probe`
  - `python3 scripts/generate_samples.py --version v6 --profile mono-print-edge-rook --count 3 --output-dir /tmp/v6_mono_edge_rook_probe`
- result:
  - next v6 experiment is isolated to the `*_v4` lane
  - the next run heavily favors true dark anchors and lighter mono-print targets instead of mixing ordinary boards with synthetic damage-heavy “anchors”

## Entry

- commit: `pending`
- objective: Tidy the generator/trainer back down to one canonical targeted recovery lane, wire the new book/t-shirt board themes into the print path, and start the next training run from the current v7 checkpoint.
- files:
  - `generate_hybrid_v6.py`
  - `train_hybrid_v6.py`
  - `docs/ENGINEERING_LOG.md`
- behavior_change:
  - Removed the extra temporary targeted recipe/docs/scripts that had accumulated around the miss-recovery experiments and set one canonical generator default: `v6_targeted_recovery_v8`.
  - Kept `diagtransfer_hatched` as the book/t-shirt style bucket name, but treated it as a normal profile name only; no special architecture or separate toolchain was introduced.
  - Wired print-diagram rendering to use the new cropped board textures under `board_themes/` for the book/t-shirt class instead of relying only on the older procedural mono boards.
  - Adjusted the active targeted mix to focus on the remaining misses:
    - `diagtransfer_hatched 0.24`
    - `book_page_reference 0.18`
    - `shirt_print_reference 0.18`
    - `broadcast_dark_sparse 0.16`
    - `dark_anchor_clean 0.08`
    - `clean 0.08`
    - `digital_overlay_clean 0.04`
    - `edge_rook_page 0.02`
    - `tilt_anchor 0.02`
  - Hardened non-print board-theme discovery so reference folders like `board_themes/new/` are ignored during random theme selection.
  - Switched the trainer defaults to the new `targeted_recovery_v8` lane and changed the warm-start base checkpoint from `model_hybrid_v5_latest_best.pt` to `model_hybrid_v6_mono_logo_v7_latest_best.pt`.
- validation:
  - `python3 -m py_compile generate_hybrid_v6.py train_hybrid_v6.py`
  - `python3 -m unittest -q scripts.tests_v6.test_internal_v6`
  - `python3 scripts/generate_samples.py --version v6 --count 3 --profile diagtransfer_hatched --output-dir /tmp/chessbot_targeted_smoke/diagtransfer`
  - `python3 scripts/generate_samples.py --version v6 --count 3 --profile book_page_reference --output-dir /tmp/chessbot_targeted_smoke/book`
  - `python3 scripts/generate_samples.py --version v6 --count 3 --profile shirt_print_reference --output-dir /tmp/chessbot_targeted_smoke/shirt`
  - `python3 scripts/generate_samples.py --version v6 --count 3 --profile broadcast_dark_sparse --output-dir /tmp/chessbot_targeted_smoke/broadcast`
  - `BOARDS_PER_CHUNK=32 CHUNKS_TRAIN=1 CHUNKS_VAL=1 OUTPUT_DIR=tensors_v6_targeted_recovery_v8_smoke python3 generate_hybrid_v6.py`
  - `python3 scripts/validate_tensors_v6.py --data-dir tensors_v6_targeted_recovery_v8_smoke`
  - `python3 generate_hybrid_v6.py`
  - `python3 scripts/validate_tensors_v6.py --data-dir tensors_v6_targeted_recovery_v8`
  - `python3 train_hybrid_v6.py`
- result:
  - the repo now has one clean targeted recovery lane instead of several overlapping experimental defaults
  - the full targeted dataset was generated successfully into `tensors_v6_targeted_recovery_v8`
  - training was started in the `models/model_hybrid_v6_targeted_recovery_v8_*` lane from the current `v7` checkpoint

## Entry

- commit: `pending`
- objective: Record the `v8` and `v9` targeted recovery outcomes, fix the `v9` output-dir mismatch, and prepare a narrower `v10` lane for the last three hardset misses.
- files:
  - `.gitignore`
  - `docs/ENGINEERING_LOG.md`
  - `generate_hybrid_v6.py`
  - `train_hybrid_v6.py`
- behavior_change:
  - Confirmed `model_hybrid_v6_targeted_recovery_v8_latest_best.pt` as the active targeted recovery winner at `47/50`, with only `00026`, `00031`, and `00049` remaining.
  - Confirmed `v9` tied `v8` at `47/50` with the same remaining misses, so the frontier is now domain coverage for the final three cases rather than longer optimization on the same recipe.
  - Fixed the generator default output-dir mismatch that was producing `v9` recipe tensors under `tensors_v6_targeted_recovery_v8`.
  - Added `wood_3d_arrow_clean` to restore the clean wood-board + slightly 3D piece + arrow lane needed for `00026`.
  - Increased shirt-print pressure in `shirt_print_reference` for `00031`.
  - Increased king-biased tilt pressure in `broadcast_dark_sparse` and `tilt_anchor` for `00049`.
  - Reduced book-weight now that `00028` is stably solved and shifted the active generator default to `v6_targeted_recovery_v10`.
  - Switched the trainer defaults to the `targeted_recovery_v10` lane and set the warm-start base checkpoint to `models/model_hybrid_v6_targeted_recovery_v9_latest_best.pt`.
  - Ignored root-level `tensors_v*` outputs so generated datasets stop showing up as untracked noise in the main repo.
- validation:
  - `models/model_hybrid_v6_targeted_recovery_v8_latest_best.pt -> 47/50`
  - `models/model_hybrid_v6_targeted_recovery_v9_latest_best.pt -> 47/50`
  - `python3 -m py_compile generate_hybrid_v6.py train_hybrid_v6.py`
  - `python3 - <<'PY' ... import generate_hybrid_v6/train_hybrid_v6 and print recipe/output/model/base defaults ... PY`
- result:
  - `v8` remains the practical targeted-recovery reference point
  - `v9` confirmed the same hardset frontier instead of extending it
  - the next run is isolated to a narrower `v10` recipe focused on `00026`, `00031`, and `00049`
