# 📚 Curriculum Learning - The Path to 99.99% Accuracy

## Overview

This guide documents the proven progression path for training the Chess Image Recognizer. To achieve **99.99% accuracy**, we must train the model progressively. Throwing extreme augmentations (like CLAHE, noise, blurs) at a fresh model causes it to fail. Instead, we teach it the basics, and gradually raise the difficulty.

We are currently at **Phase 2 (100 Epochs)**. The next step is to generate `tensors_v4` using CLAHE to master real-world lighting.

---

## The Training Plan

### ✅ Phase 1: Baseline Foundation (Epochs 1-50)

_Status: Completed_
**Goal:** Teach the model basic piece shapes and board coordinates on clean, standardized boards.
**Augmentation Recipe:**

- Mild brightness/contrast variation (±20%)
- Occasional grayscale (15%)
- Minimal board vandalization (1-2 arrows, 15% chance of red highlights)
  **Results:** Val Acc ~95%, capable of recognizing standard digital boards.

### ✅ Phase 2: Intermediate Variations (Epochs 51-100)

_Status: Completed_
**Goal:** Introduce wider lighting conditions, heavier JPEG compression, and heavier UI element clutter.
**Augmentation Recipe:**

- Wider brightness/contrast variation (±40%)
- Occasional noise (Gaussian 5-15)
- Moderately heavy vandalization (25% chance)
- Common grayscale (25%)
  **Results:** Val Acc ~98%. Model performs well on clean real-world photos, but struggles on edge cases with poor lighting (e.g. `puzzle-00003.jpeg` and `puzzle-00002.png` edge detection misalignments).

### 🚀 Phase 3: The 99.99% Mastery (Epochs 101-150)

_Status: Pending Next Step_
**Goal:** Achieve near-perfect recognition on _any_ image by injecting the exact same extreme preprocessing techniques the Recognizer uses (CLAHE, Denoising, Sharpening) directly into the training data.
**Augmentation Recipe (`generate_hybrid_v4.py`):**

- **[NEW] Adaptive Histogram Equalization (CLAHE):** Applied to 50% of the training boards. This forces the CNN to understand the hyper-contrasted textures of empty squares that CLAHE produces.
- **[NEW] Unsharp Masking & Denoising:** Simulating blurry or noisy camera artifacts.
- **[NEW] Minor Rotation (±2 degrees):** To handle slightly askew camera angles.
- **Extreme Vandalization:** Heavy arrows and circles covering pieces.

---

## Action Plan: Executing Phase 3

To implement Phase 3 and push the model to 99.99% accuracy, follow these steps:

### 1. Generating the Hard Dataset (v4)

We will create a new generator script, `generate_hybrid_v4.py`, that incorporates CLAHE and Unsharp masking into the image synthesis.

```bash
mkdir tensors_v4
python3 scripts/generate_hybrid_v4.py
```

### 2. Continued Training (Beast Mode)

We will resume the 100-epoch checkpoint on the new dataset.

```bash
python3 scripts/train_hybrid_v4.py # Trains to 150e, loading model_hybrid_100e.pt
```

### 3. Inference Update

We will update the recognizer (`recognizer_v4.py`) to _permanently_ apply CLAHE and edge detection to all incoming images, knowing that the new CNN has been specifically trained to understand that preprocessed output perfectly.

---

## Verification & Deployment

Use the test scripts to validate the FEN improvements:

```bash
python3 scripts/verify_lichess.py images_4_test
```

This script loops through all images, generates the FEN, and gives a clickable `https://lichess.org/analysis/fromPosition/...` link so you can visually verify the exact prediction.
