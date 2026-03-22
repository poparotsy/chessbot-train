# Recognizer Agent Guide

## Purpose

This document is the working handoff for AI agents operating on the chess recognizer in `chessbot-train`.

The goal is to fix the recognizer cleanly, not to hide recognizer failures behind hacks or endless retraining.

This guide covers:

- the current recognizer state
- the current problem we are solving
- non-negotiable coding rules
- the script inventory under `./scripts`
- the standard debugging and validation workflow
- recommended skills and best practices for agents

Current date context for this snapshot: `2026-03-22`.

---

## What We Are Trying To Do

We are rebuilding the recognizer so that:

- board localization is detector-first and geometry-native
- the candidate path is honest and diagnosable
- `images_4_test` is treated as a recognizer/localizer benchmark, not a model benchmark
- misses do not trigger fake “model failure” conclusions unless proven by diagnostics

The active work is isolated to:

- `recognizer_v6_candidate.py`
- `edge_grid_detector.py`

The main production recognizer:

- `recognizer_v6.py`

must stay untouched until the candidate is genuinely better and stable.

---

## Current Issues To Solve

### Main structural issue

`recognizer_v6.py` is a large mixed-responsibility file. It combines:

- board detection
- warping
- square slicing
- piece decoding
- orientation
- candidate ranking
- CLI/runtime behavior

That coupling is why recognizer cleanup has repeatedly turned into selector hacks and false model blame.

### Current recognizer mission

Fix the candidate recognizer so it surfaces the right board hypothesis without:

- chess-plausibility rescue
- crop rescue
- trim rescue
- random confidence heuristics
- “special case this image” logic

### Current active status

Current hardset status for `recognizer_v6_candidate.py` with the repo-default model path:

- `board_pass=40/50`
- `full_pass=38/50`
- current failing images:
  - `00024`
  - `00027`
  - `00028`
  - `00029`
  - `00032`
  - `00033`
  - `00035`
  - `00037`
  - `00039`
  - `00043`

Current known preserved wins:

- `00001`
- `00002`
- `00003`
- `00005`
- `00026`
- `00031`
- `00049`

Current interpretation:

- the candidate path is script-compatible and materially cleaner than the legacy recognizer
- the recognizer cleanup is not finished
- the remaining misses are still recognizer/localizer-path failures unless diagnostics prove otherwise

Important context:

- the current default model in `recognizer_v6.py` is `models/model_hybrid_v6_champion_48of50.pt`
- many disagreements seen earlier came from forcing different checkpoints during testing

---

## Non-Negotiable Rules

These rules are not optional.

### Core coding rules

- No hacks.
- No selector rescue logic.
- No chess-plausibility ranking to override detector failure.
- No trim/inset/image-specific rescue paths.
- No “if this case looks weird, do X” logic.
- No retraining to compensate for recognizer mistakes on `images_4_test`.

### Scope rules

- Keep all experimental work in `recognizer_v6_candidate.py` and `edge_grid_detector.py`.
- Do not edit `recognizer_v6.py` until the candidate clears the real gate.
- Use scripts and truth files as the source of truth, not memory or guesswork.

### Interpretation rules

- `images_4_test` is a recognizer/localizer benchmark.
- If an image in `images_4_test` misses, assume recognizer failure first.
- Only call something a model failure after script-based evidence proves the crop/path is already correct.

### Acceptable detector work

These are valid and encouraged:

- `HoughLines`
- `HoughLinesP`
- contour-based quad finding
- homography / perspective refinement
- line clustering
- geometry scoring
- grid/lattice scoring
- edge-supported corner refinement

These are not acceptable:

- ranking hacks
- image-specific trims
- candidate-specific bonuses/penalties that are not general geometry
- “confidence rescue” based on downstream decode output

---

## Current Recognizer State

### Main recognizer

File:

- `recognizer_v6.py`

Status:

- baseline
- large and coupled
- not to be edited during cleanup phase

Current default model:

- `models/model_hybrid_v6_champion_48of50.pt`

### Candidate recognizer

File:

- `recognizer_v6_candidate.py`

Purpose:

- thin recognizer wrapper over the clean detector/localizer path
- script-friendly diagnostic/testing surface
- one primary detector-selected path for live prediction

Current responsibilities:

- build candidate pool
- keep `full` available for diagnostics
- choose `full` only when full-frame grid evidence is strong enough
- expose `predict_board`, `predict_position`, and `diagnostic_decode_candidate_with_details`
- emit minimal `DEBUG_JSON` events for path analysis

### Detector module

File:

- `edge_grid_detector.py`

Purpose:

- general geometry-first board localization

Current detector families:

- aligned grid via Hough-based lattice scoring
- contour/min-area-rect quads
- robust contour proposals sourced from the legacy contour detector
- full-frame grid evidence scoring
- warped-grid refinement

### Tooling cleanup already done

- `scripts/tile_audit_compare.py` now runs from repo root without manual `PYTHONPATH`

---

## Standard Workflow

Use this workflow every time you change the candidate.

### 1. Compile sanity

```bash
python3 -m py_compile edge_grid_detector.py recognizer_v6_candidate.py
```

### 2. Spot checks

Always run the key cases first:

```bash
python3 recognizer_v6_candidate.py images_4_test/puzzle-00001.jpeg
python3 recognizer_v6_candidate.py images_4_test/puzzle-00002.png
python3 recognizer_v6_candidate.py images_4_test/puzzle-00003.jpeg
python3 recognizer_v6_candidate.py images_4_test/puzzle-00005.jpeg
python3 recognizer_v6_candidate.py images_4_test/puzzle-00026.jpeg
python3 recognizer_v6_candidate.py images_4_test/puzzle-00031.jpeg
python3 recognizer_v6_candidate.py images_4_test/puzzle-00049.jpeg
```

### 3. Miss diagnosis

If any image misses, use:

```bash
python3 scripts/deep_diagnostic_v6.py puzzle-XXXXX.ext \
  --images-dir images_4_test \
  --truth-json images_4_test/truth_verified.json \
  --recognizer-path recognizer_v6_candidate.py \
  --model-path models/model_hybrid_v6_champion_48of50.pt \
  --detail-level human
```

Use it to answer:

- did the candidate choose the right board family?
- was `full` available?
- is the miss still localizer-side or already downstream decode?

### 4. Square-level evidence

When you need to prove model vs crop:

```bash
python3 scripts/tile_audit_compare.py images_4_test/puzzle-XXXXX.ext \
  --models current=models/model_hybrid_v6_champion_48of50.pt \
  --json
```

Use it to answer:

- what board candidate the old path would choose
- whether a different crop fixes the issue with the same model
- whether the model really fails on the correct crop

### 5. Real gate

Run the real evaluator:

```bash
python3 scripts/evaluate_v6_hardset.py \
  --script recognizer_v6_candidate.py \
  --images-dir images_4_test \
  --truth-json images_4_test/truth_verified.json
```

Do not claim success until this gate agrees.

---

## Script Inventory

This section covers the top-level scripts in `./scripts`.

## Core working scripts

### `scripts/evaluate_v6_hardset.py`

What it does:

- deterministic hardset evaluator for a recognizer script
- supports `--script` so you can evaluate `recognizer_v6_candidate.py`

When to use:

- primary correctness gate

Use:

```bash
python3 scripts/evaluate_v6_hardset.py --script recognizer_v6_candidate.py
```

Typical full form:

```bash
python3 scripts/evaluate_v6_hardset.py \
  --script recognizer_v6_candidate.py \
  --images-dir images_4_test \
  --truth-json images_4_test/truth_verified.json
```

### `scripts/deep_diagnostic_v6.py`

What it does:

- deep per-image miss analysis
- compares selected candidate vs full candidate
- shows focus squares and failure classification

When to use:

- immediately after a miss

Use:

```bash
python3 scripts/deep_diagnostic_v6.py puzzle-00026.jpeg \
  --images-dir images_4_test \
  --truth-json images_4_test/truth_verified.json \
  --recognizer-path recognizer_v6_candidate.py \
  --model-path models/model_hybrid_v6_champion_48of50.pt
```

### `scripts/tile_audit_compare.py`

What it does:

- compares per-tile model predictions on a chosen candidate board
- useful for proving “model vs crop”

When to use:

- when an agent claims “model failure”
- when you need square-level evidence

Use:

```bash
python3 scripts/tile_audit_compare.py images_4_test/puzzle-00003.jpeg \
  --models current=models/model_hybrid_v6_champion_48of50.pt \
  --json
```

### `scripts/analyze_v6_paths.py`

What it does:

- analyzes execution path image-by-image using `DEBUG_JSON` telemetry

When to use:

- when you need to understand which detector family and candidate path are being selected

Current note:

- the candidate now emits minimal debug events; this script is useful, but still lighter than the original recognizer path analysis

Typical use:

```bash
python3 scripts/analyze_v6_paths.py \
  --recognizer-path recognizer_v6_candidate.py \
  --images-dir images_4_test \
  --truth-json images_4_test/truth_verified.json
```

### `scripts/compare_deep_diagnostic_reports_v6.py`

What it does:

- compares two deep-diagnostic JSON reports and flags regressions

When to use:

- before/after a detector change
- when validating that a fix did not quietly break another image

### `scripts/evaluate_v6_domain_suite.py`

What it does:

- evaluates domain-focused regression slices

When to use:

- after hardset stability improves
- for small targeted suites

### `scripts/benchmark_v6.py`

What it does:

- benchmark + relative baseline gate for quick-set latency

When to use:

- only after correctness is stable

## Geometry/boundary/data scripts

### `scripts/generate_boundary_suite_v6.py`

What it does:

- generates geometry/boundary stress images

Current status:

- currently broken in this checkout because it imports `generate_hybrid_v5` incorrectly

Use later, after fixing its import path.

### `scripts/evaluate_boundary_suite_v6.py`

What it does:

- evaluates a recognizer against the generated boundary suite

When to use:

- after the boundary suite is available and working

### `scripts/validate_tensors_v6.py`

What it does:

- validates generated tensor chunks for shape/range/integrity

When to use:

- only when working on data generation or training inputs

### `scripts/generate_samples.py`

What it does:

- previews generated training/sample boards

When to use:

- to inspect generator behavior visually
- not part of the recognizer hot path

### `scripts/build_v6_mono_assets.py`

What it does:

- generates mono/print board and piece assets

When to use:

- when building print/mono asset packs
- not part of recognizer debugging

### `scripts/png_rgba.py`

What it does:

- normalizes PNG palette assets to RGBA

When to use:

- asset hygiene only

### `scripts/setup_assets.py`

What it does:

- asset setup helper

Current status:

- blocked here because the runtime is missing the native `cairo` library

## Model ranking / comparison scripts

### `scripts/rank_models_v6.py`

What it does:

- ranks v6-compatible checkpoints on hardset using a chosen recognizer script

When to use:

- only after recognizer behavior is stable enough to compare checkpoints fairly

### `scripts/rank_models_hardset.py`

What it does:

- ranks generic checkpoint files on a hard puzzle set

When to use:

- checkpoint comparison work

### `scripts/evaluate_v5.py`

What it does:

- evaluates legacy `recognizer_v5`

Current status:

- currently broken as checked in because the import path for `recognizer_v5` is wrong

## Legacy / side scripts

### `scripts/evaluate_fullguard_stress_v6.py`

What it does:

- stress-checks `recognizer_v6_fullguard`

When to use:

- only as a legacy comparison/oracle
- do not use it as the target architecture

### `scripts/model_diagnostic.py`

What it does:

- old model checkpoint inspection helper

Current status:

- points at a missing v4 checkpoint in this checkout

### `scripts/visulaizer.py`

What it does:

- old visualization helper

Current status:

- not part of the active recognizer workflow
- file name is misspelled and should be treated as legacy utility

---

## Debugging Scripts: What, When, How

These are the scripts agents should use most.

### `deep_diagnostic_v6.py`

Use when:

- one image misses

Answers:

- did the selected candidate fail?
- did `full` do better?
- which squares are wrong?
- is it localizer-side or already decoder-side?

### `tile_audit_compare.py`

Use when:

- a miss might be blamed on the model

Answers:

- with the same model, does a different board candidate already solve it?
- which squares differ?

### `analyze_v6_paths.py`

Use when:

- you need path-level traceability across many images

Answers:

- which candidate family was chosen
- which debug events fired
- whether path pressure/candidate capping is happening

### `evaluate_v6_hardset.py`

Use when:

- you need the real score

Answers:

- pass/fail image-by-image
- board accuracy
- full FEN accuracy

---

## Skills To Use

Recommended skills for agents working on this repo:

- `computer-vision-opencv`
  - primary CV skill for detector/localizer work
- `opencv`
  - direct OpenCV reference and image-processing guidance
- `deep-learning-python`
  - useful for code organization and PyTorch-side inspection
- `deep-learning-pytorch`
  - useful when comparing model behavior on fixed crops
- `optimizing-deep-learning-models`
  - use later, only after recognizer behavior is stable
- `find-skills`
  - use when you need a missing capability instead of improvising

Recommended order:

1. `computer-vision-opencv`
2. `opencv`
3. `deep-learning-python`
4. `deep-learning-pytorch`

Do not jump to optimization/training skills before ruling out recognizer failure.

---

## Best Practices For Agents

- Make one geometry change at a time.
- Run the standard workflow after every meaningful change.
- Do not call something a model failure unless `tile_audit_compare.py` or `deep_diagnostic_v6.py` proves the crop/path is already correct.
- Prefer adding general proposal families over adding ranking exceptions.
- Preserve detector wins while widening full-frame validity only when the full-frame evidence is strong.
- Keep the candidate path script-compatible at all times.
- Do not merge candidate changes into `recognizer_v6.py` until the full hardset gate is good enough.

---

## Immediate Working Task

The active task for agents is:

- continue the candidate-only cleanup
- preserve the current wins on `00001`, `00002`, `00003`, `00005`, `00026`, `00031`, and `00049`
- work down the current fail set: `00024`, `00027`, `00028`, `00029`, `00032`, `00033`, `00035`, `00037`, `00039`, `00043`
- use the hardset plus diagnostics as the source of truth for every claim
- fix only the detector/localizer side unless a script proves otherwise

If you start work on this recognizer, start here:

1. read this document
2. run `python3 -m py_compile edge_grid_detector.py recognizer_v6_candidate.py`
3. run the spot-check images
4. run `evaluate_v6_hardset.py --script recognizer_v6_candidate.py`
5. use `deep_diagnostic_v6.py` on the first miss
