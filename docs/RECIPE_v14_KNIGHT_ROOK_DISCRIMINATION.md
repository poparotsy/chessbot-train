# V14 Knight/Rook Discrimination Recovery Recipe

## Problem
The v13 model confuses knights (n/N) with rooks (r/R) in 23 tiles across 12 stress suite images.
Root causes:
1. **Missing piece sets in training**: club, tournament, print_shirt_photo — used in stress suite but NOT in v13 training
2. **Render profile imbalance**: v13 is 50% dark_anchor_clean, only 6% clean — model fails on clean render profiles (25% fail rate)
3. **Knight/rook visual similarity**: Under wood_3d_arrow, digital_overlay, and dark_anchor render profiles, knights lose distinctive features

## Training Config

### Environment Variables
```bash
export RECIPE_NAME="v6_targeted_recovery_v14"
export BOARDS_PER_CHUNK=1000
export CHUNKS_TRAIN=12
export CHUNKS_VAL=3
export SEED=1337
export EPOCHS=120
export LEARNING_RATE=2e-6
export BATCH_SIZE_PER_GPU=256
export RESUME_FROM_CHECKPOINT=1
export BASE_MODEL_PATH=models/model_hybrid_v6_targeted_recovery_v11_latest_best.pt
```

### Recipe: v6_targeted_recovery_v14

Add to `PROFILE_RECIPES` in `generate_hybrid_v6.py`:

```python
"v6_targeted_recovery_v14": [
    # Render profile weights — balanced to cover ALL failure conditions
    ("clean", 0.22),              # UP from 0.06 — fixes clean render failures (25% fail rate)
    ("dark_anchor_clean", 0.20),  # DOWN from 0.50 — was over-represented
    ("shirt_print_reference", 0.16),
    ("broadcast_dark_sparse", 0.12),
    ("wood_3d_arrow_clean", 0.10), # UP from 0.04 — 22% fail rate, needs more
    ("digital_overlay_clean", 0.06),
    ("book_page_reference", 0.06),
    ("diagtransfer_hatched", 0.04),
    ("tilt_anchor", 0.04),
],
```

### Piece Set Expansion

The key fix: add missing piece sets that appear in stress failures.

In the `"clean"` profile (line ~112), add:
```python
"clean": {
    "PIECE_SET_NAMES": [
        "alpha", "cardinal", "cburnett", "celtic", "chessmonk",
        "chessnut", "governor", "icpieces", "maestro", "merida", "staunty",
        # NEW — missing piece sets from stress suite failures:
        "club", "tournament",
    ],
    ...existing config...
},
```

In the `"dark_anchor"` profile (line ~193), add:
```python
"dark_anchor": {
    "BOARD_THEME_NAMES": ["grey.jpg", "olive.jpg", "wood4.jpg", "metal.jpg", "blue3.jpg"],
    "PIECE_SET_NAMES": [
        "cburnett", "merida", "maestro", "governor",
        # NEW:
        "club", "celtic", "chessmonk",
    ],
    ...existing config...
},
```

In the `"wood_3d_arrow_clean"` profile, ensure it has:
```python
"wood_3d_arrow_clean": {
    "PIECE_SET_NAMES": [
        "cburnett", "merida", "maestro", "governor",
        "club", "celtic", "tournament",
    ],
    ...existing config...
},
```

### Kaggle Notebook Cells

```python
# Cell 1: Clone repo
!git clone https://github.com/poparotsy/chessbot-train.git /kaggle/working/chessbot-train
%cd /kaggle/working/chessbot-train

# Cell 2: Install dependencies
!python3 -m pip install -q torch torchvision pillow numpy opencv-python-headless python-chess

# Cell 3: Apply recipe changes (generate_hybrid_v6.py already has v14 recipe + piece set changes)
# No manual edits needed if generate_hybrid_v6.py already includes:
# - v6_targeted_recovery_v14 in PROFILE_RECIPES
# - club, tournament in clean profile PIECE_SET_NAMES
# - club, celtic, chessmonk, tournament in dark_anchor profile PIECE_SET_NAMES

# Cell 4: Generate training data
import os
os.environ["RECIPE_NAME"] = "v6_targeted_recovery_v14"
os.environ["OUTPUT_DIR"] = "tensors_v6_targeted_recovery_v14"
os.environ["BOARDS_PER_CHUNK"] = "1000"
os.environ["CHUNKS_TRAIN"] = "12"   # 12K training boards
os.environ["CHUNKS_VAL"] = "3"      # 3K validation boards
os.environ["SEED"] = "1337"
!python3 generate_hybrid_v6.py

# Cell 5: Train (defaults are already set to v14 in train_hybrid_v6.py)
!python3 train_hybrid_v6.py

# Cell 6: Evaluate
!python3 scripts/evaluate_v6_hardset.py --truth-json images_4_test/truth_verified.json --model-path models/model_hybrid_v6_targeted_recovery_v14_latest_best.pt
!python3 scripts/evaluate_v6_stress_suite.py --model-path models/model_hybrid_v6_targeted_recovery_v14_latest_best.pt
!python3 scripts/evaluate_v6_domain_suite.py --model-path models/model_hybrid_v6_targeted_recovery_v14_latest_best.pt
```

### Export to recognizer + models repos
```bash
python3 scripts/export_recognizer_bundle.py --write-manifest
python3 scripts/export_models_bundle.py --write-manifest
```

### Expected Improvements
- Club/tournament piece sets: fixes n↔r confusion on these specific piece styles
- More clean render data: fixes 25% fail rate on clean profiles
- Balanced dark_anchor: reduces overfitting to dark boards
- Target: 110+/120 stress suite, maintain 50/50 hardset board
