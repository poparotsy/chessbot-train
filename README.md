## chessbot-train

Training, evaluation, and recognizer source-of-truth repo.

## Main Files

- [`recognizer_v6.py`](/Users/guru/workspace/current/chessbot/chessbot-train/recognizer_v6.py)
  - canonical standalone recognizer
- [`generate_hybrid_v6.py`](/Users/guru/workspace/current/chessbot/chessbot-train/generate_hybrid_v6.py)
  - synthetic board/data generation
- [`train_hybrid_v6.py`](/Users/guru/workspace/current/chessbot/chessbot-train/train_hybrid_v6.py)
  - model training entrypoint

## Key Directories

- [`models`](/Users/guru/workspace/current/chessbot/chessbot-train/models)
  - local source checkpoints
- [`scripts`](/Users/guru/workspace/current/chessbot/chessbot-train/scripts)
  - evaluation, diagnostics, exports, utilities
- [`archive/recognizer_legacy`](/Users/guru/workspace/current/chessbot/chessbot-train/archive/recognizer_legacy)
  - legacy recognizer variants and archived helper files
- [`images_4_test`](/Users/guru/workspace/current/chessbot/chessbot-train/images_4_test)
  - primary hardset benchmark

## Exports

Export the deployable recognizer bundle to the parent workspace:

```bash
python3 /Users/guru/workspace/current/chessbot/chessbot-train/scripts/export_recognizer_bundle.py --write-manifest
```

Export selected or all checkpoints to the parent workspace:

```bash
python3 /Users/guru/workspace/current/chessbot/chessbot-train/scripts/export_models_bundle.py --write-manifest
```

## Rule

Keep this repo as the source of truth.
Do not treat exported copies in the parent workspace as editable source.
