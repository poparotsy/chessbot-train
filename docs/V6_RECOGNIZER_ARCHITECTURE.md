# Recognizer V6 Architecture

This document describes the active runtime path in `recognizer_v6.py` after the cleanup refactor.

## Goals

- Keep behavior stable while reducing regression risk.
- Remove shadow entrypoints and centralize runtime config.
- Make the pipeline testable by stage.

## Runtime Stages

0. Model resolution (deterministic)
- Default checkpoint path is fixed to `models/model_hybrid_v6_latest_best.pt`.
- Override path only via:
  - CLI: `--model-path`
  - env: `CHESSBOT_MODEL_PATH`
- No implicit fallback to older model families.

1. Detector stage: `build_detector_candidates(img)`
- Produces board candidates from `BoardDetector`.
- Optionally adds enhanced-source candidates for low-saturation scan/book images using CLAHE + mild unsharp (`enhance_low_saturation_image`).
- Includes gradient-projection candidate (`detect_gradient_projection`) based on Sobel x/y projection peaks.
- Adds directional trim variants for `panel_split_*` candidates.
- Applies `MAX_DECODE_CANDIDATES` cap deterministically.

2. Orientation-context stage: `collect_orientation_context(candidates, board_perspective)`
- Computes file-label evidence once for `auto` mode.
- Returns cached label context for all candidate decodes.

3. Decode stage: `decode_candidate(candidate, model, device, board_perspective, orientation_context)`
- Runs `infer_fen_on_image_deep_topk`.
- Applies low-saturation sparse re-score rule with multi-option top-k beam and edge-rook objective bonus.
- Resolves orientation via `resolve_candidate_orientation`.
- Emits structured debug payload for the decoded candidate.

4. Selection stage: `select_best_candidate(scored)`
- Applies plausibility-first filtering.
- Applies sparse-board override rule for trusted warps.
- Ranks remaining candidates with deterministic tie-break order.

5. Entrypoint: `predict_board(image_path, model_path=None, board_perspective="auto")`
- Wires stages in fixed order.
- Returns `(board_fen, confidence)`.

## Test Coverage Mapping

- Detector helper: trim augmentation and candidate cap behavior.
- Orientation helper: non-auto defaults and forced override behavior.
- Decode helper: batched tile forward invariant (single model pass for 64 tiles).
- Selection helper: sparse override path behavior.
- Existing safety tests: timeout isolation and FEN transforms.
