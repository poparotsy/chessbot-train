# 📚 Curriculum Learning - Progressive Training Guide

## Overview

Train the model progressively with increasingly difficult data. This approach:
- ✅ Faster convergence (model learns basics first)
- ✅ Better generalization (gradual difficulty increase)
- ✅ Prevents catastrophic forgetting
- ✅ Higher final accuracy

---

## Training Strategy

### Phase 1: Baseline (Epochs 1-50)
**Goal:** Learn basic piece recognition on clean, standard boards

#### Settings in `generate_hybrid_v3.py`:
```python
# Brightness - moderate range
if random.random() > 0.5:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))

# Contrast - moderate range
if random.random() > 0.5:
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.1))

# Grayscale - occasional
if random.random() > 0.85:  # 15% chance
    img = img.convert('L').convert('RGB')

# Arrows - standard
num_arrows = random.randint(1, 2)

# Highlights - standard
num_highlights = random.randint(1, 2)

# Red circles - occasional
if random.random() < 0.15:  # 15% chance
    draw_red_circle(...)

# Vandalization - light
if random.random() < 0.15:  # 15% chance
    vandalize_board(...)

# JPEG compression - moderate
quality = random.randint(50, 90)
```

#### Commands:
```bash
cd /path/to/chessimg2pos/train
python3 generate_hybrid_v3.py
python3 train_hybrid_v3.py  # Trains to epoch 50
```

**Expected:** Val Acc ~0.95+, Test images ~60-70% correct

---

### Phase 2: Intermediate (Epochs 51-100)
**Goal:** Handle brightness variations, more augmentation

#### Settings to Change in `generate_hybrid_v3.py`:
```python
# Brightness - WIDER range (handles washed out images)
if random.random() > 0.5:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.5))

# Contrast - wider range
if random.random() > 0.5:
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))

# Grayscale - more common
if random.random() > 0.75:  # 25% chance
    img = img.convert('L').convert('RGB')

# Arrows - more
num_arrows = random.randint(1, 3)

# Highlights - more
num_highlights = random.randint(1, 3)

# Red circles - more common
if random.random() < 0.25:  # 25% chance
    draw_red_circle(...)

# Vandalization - moderate
if random.random() < 0.25:  # 25% chance
    vandalize_board(...)

# JPEG compression - wider range
quality = random.randint(40, 90)

# Noise - slightly stronger
noise = np.random.normal(0, random.randint(5, 15), arr.shape)
```

#### Commands:
```bash
# Regenerate dataset with new settings
rm -rf tensors_v3
python3 generate_hybrid_v3.py

# Continue training from epoch 50
python3 train_hybrid_v3.py  # Automatically loads latest.pt, trains 51→100
```

**Expected:** Val Acc ~0.97+, Test images ~70-80% correct

---

### Phase 3: Advanced (Epochs 101-150) - Optional
**Goal:** Handle extreme cases, very challenging images

#### Settings to Change in `generate_hybrid_v3.py`:
```python
# Brightness - EXTREME range
if random.random() > 0.5:
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.7))

# Contrast - extreme range
if random.random() > 0.5:
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))

# Grayscale - very common
if random.random() > 0.6:  # 40% chance
    img = img.convert('L').convert('RGB')

# Rotation - add slight rotation (NEW!)
if random.random() > 0.7:  # 30% chance
    img = img.rotate(random.uniform(-5, 5), fillcolor=(128, 128, 128))

# JPEG compression - extreme
quality = random.randint(30, 90)

# Noise - stronger
noise = np.random.normal(0, random.randint(8, 20), arr.shape)

# Vandalization - heavy
if random.random() < 0.35:  # 35% chance
    vandalize_board(...)

# Multiple red circles
if random.random() < 0.3:
    for _ in range(random.randint(1, 3)):
        draw_red_circle(...)
```

#### Commands:
```bash
# Regenerate dataset with extreme settings
rm -rf tensors_v3
python3 generate_hybrid_v3.py

# Continue training from epoch 100
python3 train_hybrid_v3.py  # Automatically loads latest.pt, trains 101→150
```

**Expected:** Val Acc ~0.98+, Test images ~80-90% correct

---

## Quick Reference: What to Edit

### Location: `generate_hybrid_v3.py`

Find the `augment_image()` function (around line 15-45) and modify:

1. **Brightness range:** `random.uniform(min, max)`
2. **Contrast range:** `random.uniform(min, max)`
3. **Grayscale probability:** `if random.random() > threshold`
4. **Noise strength:** `random.randint(min, max)`
5. **JPEG quality:** `random.randint(min, max)`

Find the `render_board()` function (around line 100+) and modify:

6. **Arrows:** `num_arrows = random.randint(min, max)`
7. **Highlights:** `num_highlights = random.randint(min, max)`
8. **Red circles probability:** `if random.random() < threshold`
9. **Vandalization probability:** `if random.random() < threshold`

---

## Training Commands Summary

```bash
# Phase 1: Baseline
python3 generate_hybrid_v3.py
python3 train_hybrid_v3.py

# Phase 2: Intermediate (after editing generate_hybrid_v3.py)
rm -rf tensors_v3
python3 generate_hybrid_v3.py
python3 train_hybrid_v3.py

# Phase 3: Advanced (after editing generate_hybrid_v3.py again)
rm -rf tensors_v3
python3 generate_hybrid_v3.py
python3 train_hybrid_v3.py
```

---

## Important Notes

### ✅ Checkpoint System Works Automatically
- `latest.pt` is saved every epoch
- Training auto-resumes from last checkpoint
- No need to manually specify start epoch

### ✅ Best Model Tracking
- `model_hybrid_beast.pt` saves only when validation accuracy improves
- This is your final model for inference

### ✅ Data Regeneration Required
- Must delete `tensors_v3/` before regenerating
- Each phase needs fresh data with new augmentation settings

### ⚠️ Don't Skip Phases
- Start with Phase 1 (easy data)
- Model needs solid foundation before hard examples
- Jumping to Phase 3 directly will fail

---

## Alternative: Mixed Data Approach

Instead of pure curriculum, you can mix difficulties:

### Phase 1: 100% Easy
### Phase 2: 70% Easy + 30% Hard
### Phase 3: 50% Easy + 50% Hard

This requires modifying `generate_hybrid_v3.py` to randomly choose augmentation strength per board.

---

## Monitoring Progress

### During Training:
```
✅ EPOCH 50 | Avg Loss: 0.0001 | Val Acc: 0.9500 | Time: 220s
```

### After Each Phase:
```bash
# Test on real images
python3 scripts/recognizer_v3.py --debug images_4_test/puzzle-00001.jpeg
```

### Expected Progression:
- Phase 1: ~60-70% test accuracy
- Phase 2: ~70-80% test accuracy
- Phase 3: ~80-90% test accuracy

---

## Files Modified Per Phase

### Phase 1 → Phase 2:
- ✏️ Edit `generate_hybrid_v3.py` (augmentation settings)
- 🗑️ Delete `tensors_v3/`
- ▶️ Run `generate_hybrid_v3.py`
- ▶️ Run `train_hybrid_v3.py`

### Phase 2 → Phase 3:
- ✏️ Edit `generate_hybrid_v3.py` (augmentation settings)
- 🗑️ Delete `tensors_v3/`
- ▶️ Run `generate_hybrid_v3.py`
- ▶️ Run `train_hybrid_v3.py`

---

## Success Metrics

### Phase 1 Success:
- ✅ Val Acc > 0.95
- ✅ Loss < 0.01
- ✅ puzzle-00001.jpeg recognized correctly

### Phase 2 Success:
- ✅ Val Acc > 0.97
- ✅ Loss < 0.005
- ✅ 70%+ of test images correct

### Phase 3 Success:
- ✅ Val Acc > 0.98
- ✅ Loss < 0.003
- ✅ 80%+ of test images correct
- ✅ Confidence scores > 0.70 on correct predictions

---

## Troubleshooting

### If accuracy drops after new phase:
- Augmentation too aggressive
- Reduce difficulty slightly
- Train for more epochs (e.g., 75 instead of 50)

### If accuracy plateaus:
- Augmentation not challenging enough
- Increase difficulty
- Add new augmentation types

### If loss increases:
- Learning rate too high for fine-tuning
- Consider reducing LR in later phases
- Or use more epochs to stabilize

---

## Final Checklist

Before starting:
- ✅ `train_hybrid_v3.py` has Focal Loss enabled
- ✅ `train_hybrid_v3.py` has 6-layer classifier
- ✅ `scripts/recognizer_v3.py` has edge detection
- ✅ `models/checkpoints/` directory exists
- ✅ Backup current `model_hybrid_beast.pt` if you have one

After Phase 3:
- ✅ Test on all puzzle images
- ✅ Check confidence scores
- ✅ Save final model
- ✅ Document results

---

## Good Luck! 🚀

This progressive approach should give you significantly better results than training on hard data from the start!
