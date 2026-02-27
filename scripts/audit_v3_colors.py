import torch, glob, os, random
import matplotlib.pyplot as plt
import numpy as np

def audit_colors():
    files = sorted(glob.glob("tensors_v3/*.pt"))
    if not files: return
    
    label_names = ["Empty", "W_P", "W_N", "W_B", "W_R", "W_Q", "W_K", 
                   "B_P", "B_N", "B_B", "B_R", "B_Q", "B_K"]
    
    # Load one chunk
    data = torch.load(files[0], map_location='cpu')
    x, y = data['x'], data['y']
    
    sample_pieces = []
    sample_labels = []

    # Specifically hunt for 8 White and 8 Black samples
    white_indices = ( (y >= 1) & (y <= 6) ).nonzero(as_tuple=True)[0].tolist()
    black_indices = ( (y >= 7) & (y <= 12) ).nonzero(as_tuple=True)[0].tolist()
    
    random.shuffle(white_indices)
    random.shuffle(black_indices)
    
    # Combine 8 of each
    indices = white_indices[:8] + black_indices[:8]

    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(indices):
        img = x[idx].permute(1, 2, 0).numpy()
        plt.subplot(4, 4, i+1)
        plt.imshow(img)
        plt.title(label_names[y[idx].item()])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("color_verification.png")
    print("🖼️ Verification saved to 'color_verification.png'. Check for both colors!")

if __name__ == "__main__":
    audit_colors()

