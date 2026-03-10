# Multi-Head Model Plan (v6)

## 1) What “multi-head” means

A multi-head model uses:
- one **shared backbone** (common visual features)
- multiple **task heads** (separate outputs for different targets)

So it is **not** “one head per data source type.”  
It is “one head per prediction task.”

---

## 2) Why this fits this project

Current v5 mixes:
- tile classifier confidence
- board detection heuristics
- post-fixes (duplicate king repair, plausibility checks, side-to-move checks)

That works, but some failures are structural (board crop/orientation/metadata), not tile-classification-only.

Multi-head gives the model explicit responsibility for:
- board localization quality
- board orientation / POV
- piece grid prediction
- side-to-move (optional early)

This reduces reliance on patches for decisions the model can learn directly.

---

## 3) Proposed v6 heads

Use a single full-board input (`512x512`), shared CNN/ViT backbone, then:

1. `head_pieces` (required)
- Output: `8x8x13` logits (`1PNBRQKpnbrqk`)
- Purpose: core FEN board prediction

2. `head_board_geom` (required)
- Output: 4 board corners (8 values) + board-presence score
- Purpose: robust board assembly even with banners/crops/partial layouts

3. `head_pov` (required)
- Output: `white_bottom` vs `black_bottom`
- Purpose: replace fragile orientation inference

4. `head_stm` (optional in v6.1, recommended in v6.2)
- Output: `w / b / unknown`
- Purpose: reduce check-based side-to-move heuristics

5. `head_quality` (optional)
- Output: confidence/quality scalar
- Purpose: gate bad detections and decide retry/fallback

---

## 4) Labels needed

Already available or easy to generate:

- Piece grid labels: already from FEN
- POV label: already in generator metadata (`label_pov`)
- Side-to-move: from generated board state (`board.turn`)
- Geometry labels: exact synthetic board corners from render/warp pipeline
- Quality label: synthetic corruption severity (or binary pass/fail)

For real images (`images_4_test` / `temp`):
- keep piece labels (truth FEN)
- add optional manual POV + STM where known
- geometry labels can be skipped initially (train geom mostly on synthetic)

---

## 5) Loss design

Total loss:

`L = w_piece*L_piece + w_geom*L_geom + w_pov*L_pov + w_stm*L_stm + w_q*L_quality`

Suggested start:
- `w_piece = 1.0`
- `w_geom = 0.35`
- `w_pov = 0.25`
- `w_stm = 0.15` (if enabled)
- `w_q = 0.10` (if enabled)

Important:
- Keep `L_piece` dominant.
- Increase `w_geom/w_pov` only if localization/orientation remains unstable.

---

## 6) Inference flow (clean pipeline)

1. Predict geometry + board presence.
2. Warp/crop from predicted corners.
3. Predict POV.
4. Predict piece grid.
5. Predict STM (or fallback to heuristic if low confidence).
6. Run legality/plausibility check as final validation, not primary decoder.

This makes heuristics a safety net, not the main driver.

---

## 7) How to start (minimal disruption)

### Phase A (fast)
- Keep current v5 model for production.
- Add `train_hybrid_v6.py` + `recognizer_v6.py` only.
- Reuse generator; extend it to emit:
  - corner labels
  - pov labels
  - stm labels

### Phase B (first usable v6)
- Train with `head_pieces + head_pov + head_geom`.
- Keep stm from current check heuristic.

### Phase C (full v6)
- Add `head_stm`.
- Add confidence/quality head if needed.
- Compare v6 vs v5 on:
  - `images_4_test`
  - `temp`
  - hardset ranking script

---

## 8) Acceptance criteria

Ship v6 only if:

1. Hardset score >= current best (`29/30`) consistently
2. No regression on known sensitive puzzles (`00021`, `00024`, `00028`, `00030`)
3. Orientation errors reduced vs v5
4. Fewer post-repair interventions (duplicate-king repair trigger rate goes down)

---

## 9) Direct answer to your question

“Is it a head for each type of data (board dimensions, direction, etc.)?”

Close, but better phrasing:
- Use a head for each **target task**:
  - geometry (board dimensions/corners)
  - direction (POV)
  - pieces
  - side-to-move

That is the pro-level multi-head setup for this project.

