import torch
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the first chunk
data_path = "tensors_v4/val_0.pt"
data = torch.load(data_path, map_location='cpu')
x = data['x']  # The images
y = data['y']  # The labels

print(f"--- Dataset Report ---")
print(f"Shape: {x.shape}")
print(f"Dtype: {x.dtype}")
print(f"Min Value: {x.min().item():.4f}")
print(f"Max Value: {x.max().item():.4f}")

# Label Map for reference
labels = ["Empty", "W_Pawn", "W_Knight", "W_Bishop", "W_Rook", "W_Queen", "W_King",
          "B_Pawn", "B_Knight", "B_Bishop", "B_Rook", "B_Queen", "B_King"]

# 2. Set up visualization
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
axes = axes.flatten()

for i in range(16):
    img = x[i]
    
    # Handle permutations (if it's C,H,W or H,W,C)
    if img.shape[0] == 3:
        img = img.permute(1, 2, 0)
    
    img_np = img.numpy()
    
    # If it's 0-1 range, matplotlib handles it fine. 
    # If it's -1 to 1, we need to shift it back to 0-1 for display
    if img_np.min() < 0:
        img_np = (img_np + 1.0) / 2.0
    
    axes[i].imshow(img_np)
    axes[i].set_title(f"Label: {labels[y[i].item()]}")
    axes[i].axis('off')

plt.tight_layout()
plt.savefig("debug_tiles.png")
print("--- Check debug_tiles.png to see the images! ---")

