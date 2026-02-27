import torch
import glob
import os
import matplotlib.pyplot as plt
import numpy as np

def audit():
    files = sorted(glob.glob("tensors_v3/*.pt"))
    if not files:
        print("❌ No .pt files found in tensors_v3/")
        return

    label_names = ["Empty", "W_P", "W_N", "W_B", "W_R", "W_Q", "W_K", 
                   "B_P", "B_N", "B_B", "B_R", "B_Q", "B_K"]
    
    total_stats = torch.zeros(13)
    sample_pieces = []
    sample_labels = []

    print(f"🧐 Auditing {len(files)} files...")

    for f in files:
        data = torch.load(f, map_location='cpu')
        x, y = data['x'], data['y']
        
        # 1. Check Normalization
        v_min, v_max = x.min().item(), x.max().item()
        
        # 2. Count Labels
        unique, counts = torch.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            total_stats[u] += c
            
        # 3. Grab samples of PIECES (Label > 0)
                    ################
        piece_indices = (y > 0).nonzero(as_tuple=True)[0]
        if len(piece_indices) > 0 and len(sample_pieces) < 16:
           selected_indices = np.random.choice(piece_indices.numpy(), size=min(4, len(piece_indices)), replace=False)
           for idx in selected_indices:
               if len(sample_pieces) < 16:
                  img = x[idx]
                  if img.shape[0] == 3: 
                     img = img.permute(1, 2, 0)
                  sample_pieces.append(img.numpy())
                  sample_labels.append(label_names[y[idx].item()])
                    ################
        print(f"File: {os.path.basename(f)} | Range: [{v_min:.2f}, {v_max:.2f}] | Pieces: {len(piece_indices)}")

    # --- FINAL REPORT ---
    print("\n" + "="*30)
    print("FINAL DATASET REPORT")
    print("="*30)
    for i, name in enumerate(label_names):
        print(f"{name.ljust(8)}: {int(total_stats[i].item()):,}")
    
    if total_stats[1:].sum() == 0:
        print("\n❌ FATAL ERROR: Zero pieces found in the entire dataset!")
    else:
        # Save a grid of PIECES to see if they are actually visible
        plt.figure(figsize=(12, 12))
        for i in range(len(sample_pieces)):
            plt.subplot(4, 4, i+1)
            img = sample_pieces[i]
            # Normalize for display if it's -1 to 1
            if img.min() < 0: img = (img + 1) / 2
            plt.imshow(img)
            plt.title(sample_labels[i])
            plt.axis('off')
        plt.tight_layout()
        plt.savefig("audit_pieces.png")
        print("\n🖼️ Check 'audit_pieces.png' to see if the pieces are actually drawn!")

if __name__ == "__main__":
    audit()

