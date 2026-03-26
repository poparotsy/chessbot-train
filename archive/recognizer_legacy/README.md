Legacy and experimental recognizer files moved out of the train root.

These are kept only for rollback, diagnostics, or historical comparison:
- `recognizer_v6_candidate.py`
- `recognizer_v6_candidate_core.py`
- `edge_grid_detector.py`
- `recognizer_v6_fullguard.py`
- `recognizer_v6_legacy.py`

Production entrypoints stay in the train root:
- `recognizer_v6.py`
- `generate_hybrid_v6.py`
- `train_hybrid_v6.py`
