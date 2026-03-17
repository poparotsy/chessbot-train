# Chessbot Train - Project Context

## Project Overview

This is the **training repository** for the Chessbot chessboard image recognition system. The project trains deep learning models to convert chessboard images into FEN (Forsyth-Edwards Notation) strings. The system uses a hybrid approach combining:

- **CNN-based tile classification** (64 individual square predictions)
- **Computer vision heuristics** (board detection, perspective warping, orientation)
- **Multi-candidate decoding** with plausibility scoring

**Current State:** v6 recognizer with v5 model achieving **49-50/50 board accuracy** on the hardset test suite (`images_4_test/truth_verified.json`).

---

## Directory Structure

```
chessbot-train/
├── recognizer_v6.py          # Main inference pipeline (detector → decode → select)
├── generate_hybrid_v6.py     # Training data generator with augmentation profiles
├── train_hybrid_v6.py        # Model trainer with validation-based best selection
├── requirements.txt          # Dependencies: torch, numpy, pillow, opencv-python-headless, python-chess
│
├── models/                   # Trained checkpoints and model artifacts
├── tensors_v6_*/             # Generated training tensors (train.pt, val.pt)
├── images_4_test/            # Hardset test images (50 puzzles) + truth_verified.json
├── board_themes/             # Board texture assets for synthetic data generation
├── piece_sets/               # Piece image sets (cburnett, mono_print_scan, etc.)
│
├── scripts/
│   ├── evaluate_v6_hardset.py    # Full hardset evaluation
│   ├── test_after_change_v6.py   # Quick regression gate (12 images)
│   ├── benchmark_v6.py           # Performance benchmarking
│   ├── rank_models_v6.py         # Model ranking against hardset
│   ├── validate_tensors_v6.py    # Tensor integrity validation
│   ├── analyze_v6_paths.py       # Per-image pipeline path analysis
│   └── tests_v6/                 # Unit tests for recognizer components
│
└── docs/
    ├── V6_RECOGNIZER_ARCHITECTURE.md  # Runtime stage documentation
    ├── MULTI_HEAD_V6_PLAN.md          # Multi-head model roadmap
    ├── CURRICULUM_LEARNING_GUIDE.md   # Training phase documentation
    ├── MIND.md                        # Persistent project memory/guardrails
    └── ENGINEERING_LOG.md             # Mandatory change log (every commit)
```

---

## Key Concepts

### Recognition Pipeline (v6)

The recognizer follows a 5-stage pipeline:

1. **Model Resolution**: Deterministic path to `models/model_hybrid_v5_latest_best.pt` (or env override via `CHESSBOT_MODEL_PATH`)

2. **Detector Stage**: Generates board candidates via:
   - Contour-based quad detection
   - Gradient projection (Sobel x/y for mono/print boards)
   - Panel-split detection with directional trim
   - Low-saturation enhancement source candidates

3. **Orientation Context**: Computes file-label evidence (`a`/`h` detection) once for `auto` mode

4. **Decode Stage**: Runs tile classifier with:
   - Low-saturation sparse re-scoring (multi-option top-k beam)
   - Edge-rook objective bonus
   - Orientation resolution

5. **Selection Stage**: Applies plausibility-first filtering, sparse-board overrides, deterministic tie-breaking

### Data Generation Profiles

The generator uses **profile mixes** to create targeted training data:

- `clean`: Standard digital boards with mild augmentation
- `mono_scan`: Monochrome scan-like boards
- `mono_print_sparse_edge`: Printed diagram style (targets puzzles 00003/00028/00031)
- `logo_overlay`: Boards with square/logo interference (targets puzzle 00024)
- `edge_frame`: Edge rook emphasis
- `hard_combo`: Extreme augmentation combinations

### Training Philosophy

**Curriculum Learning** is critical:
- Phase 1 (1-50e): Clean boards, basic shapes
- Phase 2 (51-100e): Lighting variation, UI clutter
- Phase 3 (101-150e): CLAHE, extreme augmentation matching inference preprocessing

**Never train with extreme augmentations from scratch** - the model will fail to converge.

---

## Building and Running

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start Commands

**Evaluate current model:**
```bash
python3 scripts/evaluate_v6_hardset.py --truth-json images_4_test/truth_verified.json
```

**Quick regression gate (12 images):**
```bash
python3 scripts/test_after_change_v6.py --profile quick
```

**Run inference on single image:**
```bash
python3 recognizer_v6.py images_4_test/puzzle-00028.jpeg --debug
```

**Generate training data:**
```bash
OUTPUT_DIR=tensors_v6_mono_logo_v7 \
RECIPE_NAME=v6_mono_logo_recovery_v1 \
BOARDS_PER_CHUNK=1000 \
CHUNKS_TRAIN=10 \
CHUNKS_VAL=2 \
python3 generate_hybrid_v6.py
```

**Train model:**
```bash
DATA_DIR=tensors_v6_mono_logo_v7 \
EPOCHS=120 \
LEARNING_RATE=2e-6 \
RESUME_FROM_CHECKPOINT=true \
python3 train_hybrid_v6.py
```

**Rank models against hardset:**
```bash
python3 scripts/rank_models_v6.py \
  --models-glob "models/model_hybrid_*.pt" \
  --truth-json images_4_test/truth_verified.json
```

---

## Development Conventions

### Mandatory Rules (from MIND.md)

1. **Single dependency file**: `requirements.txt` only
2. **Profile-mix logging enabled**: Generator outputs `mix[...]` in creation logs
3. **Standalone scripts**: v5/v6 generators and trainers are standalone (no cross-version imports)
4. **Documentation mandatory**: Every commit must update `ENGINEERING_LOG.md` with:
   - Commit hash
   - Objective/rationale
   - Files changed
   - Behavior changes
   - Validation commands and outcome
5. **v7 guardrail**: Treat v7 as experimental/failing until it beats locked v6/v5 hardset baseline

### Testing Practices

- **Internal tests**: `python3 -m unittest discover -s scripts/tests_v6 -v`
- **Quick gate**: `scripts/test_after_change_v6.py --profile quick` (12 images, ~11/12 pass baseline)
- **Full hardset**: `scripts/evaluate_v6_hardset.py` (50 images, 49/50 board pass baseline)
- **Performance gate**: `scripts/benchmark_v6.py` (median/p95 latency checks)

### Change Workflow

1. Make code change
2. Run `python3 -m py_compile <modified_file>`
3. Run unit tests: `python3 -m unittest -q scripts/tests_v6`
4. Run quick gate: `python3 scripts/test_after_change_v6.py --profile quick`
5. Update `ENGINEERING_LOG.md` with entry
6. Commit

---

## Known Issues and Baselines

### Current Performance (v6 + v5 model)

| Metric | Score |
|--------|-------|
| Quick gate | 11-12/12 |
| Full hardset (board) | 49-50/50 |
| Full hardset (full-FEN) | 47-48/50 |

### Remaining Failure Case

**puzzle-00028.jpeg**: The only consistent miss. Root cause analysis shows:
- Geometry candidates are present (gradient projection path)
- Rook logits remain weak on critical squares
- Model/data gap rather than edge-quad failure
- Targeted via `mono_print_sparse_edge` profile in training data

### Side-to-Move Heuristic

Full-FEN misses often stem from side-to-move inference, not board position. This is a separate heuristic layer outside the core model.

---

## Architecture Decisions

### Deterministic Model Path

No implicit fallback. Model path resolution:
1. CLI `--model-path` argument
2. Environment `CHESSBOT_MODEL_PATH`
3. Default: `models/model_hybrid_v5_latest_best.pt`

### Candidate Cap Enforcement

`MAX_DECODE_CANDIDATES` cap applied deterministically after candidate generation to prevent runaway computation.

### Low-Saturation Rescoring

Global sparse-board re-score path with:
- Multi-option top-k beam (up to N alternatives per empty square)
- Edge-rook objective bonus
- Missing-edge-rook promotion when one color has zero rooks

### Orientation Fallback Order (auto mode)

1. Board labels (`a`/`h`) if high-confidence
2. Strong piece-distribution margin
3. Default (orientation_best_guess disabled after regression)

---

## Multi-Head Roadmap (v6+)

Future work may extend to multi-head architecture:

- `head_pieces`: 8x8x13 logits (core FEN)
- `head_board_geom`: 4 corners + board-presence score
- `head_pov`: white/black orientation
- `head_stm`: side-to-move (w/b/unknown)
- `head_quality`: confidence scalar

This would reduce reliance on post-hoc heuristics by making the model directly responsible for geometry, orientation, and turn prediction.

---

## File Reference

| File | Purpose |
|------|---------|
| `recognizer_v6.py` | Main inference: `predict_board(image_path)` returns `(fen, confidence)` |
| `generate_hybrid_v6.py` | Data generator with 20+ augmentation profiles |
| `train_hybrid_v6.py` | Trainer with focal loss, validation-based best selection |
| `scripts/evaluate_v6_hardset.py` | Full 50-image evaluation with timing |
| `scripts/test_after_change_v6.py` | Quick 12-image regression gate |
| `V6_RECOGNIZER_ARCHITECTURE.md` | Detailed runtime stage documentation |
| `ENGINEERING_LOG.md` | Mandatory commit log (read before any change) |
| `MIND.md` | Persistent project guardrails and memory |
