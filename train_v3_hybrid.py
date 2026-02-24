import torch
import torch.nn as nn
import torch.optim as optim
import glob
import os
import gc
import time
from torch.utils.data import DataLoader, TensorDataset

# ============ PRODUCTION CONFIG ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_COUNT = torch.cuda.device_count()
# Scaled batch size for Dual GPU (adjust if you hit memory limits)
BATCH_SIZE = 512 * max(1, GPU_COUNT) 
LEARNING_RATE = 1e-4
EPOCHS = 50
DATA_DIR = "tensors_v3"
MODEL_SAVE_PATH = "model_hybrid_beast.pt"

# ============ MODEL ARCHITECTURE ============
class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        # Input: (3, 64, 64)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), # 32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), # 16
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), # 8
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2)  # 4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 13)
        )

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

def train():
    print(f"\n🚀 STARTING PRODUCTION TRAINING")
    print(f"💻 Hardware: {DEVICE} ({GPU_COUNT} GPUs)")
    print(f"📊 Batch Size: {BATCH_SIZE} | Learning Rate: {LEARNING_RATE}")
    
    # 1. Initialize Model & Multi-GPU support
    model = ChessCNN()
    if GPU_COUNT > 1:
        print(f"💡 Parallelizing model across {GPU_COUNT} GPUs")
        model = nn.DataParallel(model)
    model.to(DEVICE)

    # 2. Weighted Loss (Crucial: prioritize pieces over empty squares)
    weights = torch.tensor([0.7] + [1.3]*12).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # 3. Find Data
    train_files = sorted(glob.glob(f"{DATA_DIR}/train_*.pt"))
    val_files = sorted(glob.glob(f"{DATA_DIR}/val_*.pt"))
    
    if not train_files:
        print(f"❌ ERROR: No tensors found in {DATA_DIR}")
        return

    # 4. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        processed_chunks = 0

        for f in train_files:
            # Load chunk into CPU RAM first
            data = torch.load(f, map_location='cpu')
            
            # NORMALIZATION LAW: uint8 -> float32 [-1, 1]
            x = (data['x'].float() / 127.5) - 1.0
            y = data['y'].long()
            
            loader = DataLoader(TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
            
            chunk_loss = 0
            for bx, by in loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                chunk_loss += loss.item()
            
            total_loss += (chunk_loss / len(loader))
            processed_chunks += 1
            
            # Print intermediate progress
            if processed_chunks % 2 == 0:
                print(f"   Epoch {epoch+1} | Progress: {processed_chunks}/{len(train_files)} chunks | Current Loss: {chunk_loss/len(loader):.4f}")

            # Strict Memory Cleanup
            del data, x, y, loader
            gc.collect()
            torch.cuda.empty_cache()

        # 5. Validation Phase
        model.eval()
        v_acc = 0
        with torch.no_grad():
            v_data = torch.load(val_files[0], map_location='cpu')
            vx = (v_data['x'].float().to(DEVICE) / 127.5) - 1.0
            vy = v_data['y'].to(DEVICE)
            v_out = model(vx)
            v_acc = (torch.argmax(v_out, 1) == vy).float().mean().item()
            del v_data, vx, vy
            gc.collect()

        # 6. Epoch Summary
        duration = time.time() - epoch_start
        print(f"✅ EPOCH {epoch+1:02d} | Loss: {total_loss/len(train_files):.4f} | Val Acc: {v_acc:.4f} | Time: {duration:.1f}s")
        
        # Save Model (handling DataParallel wrapper)
        save_obj = model.module.state_dict() if GPU_COUNT > 1 else model.state_dict()
        torch.save(save_obj, MODEL_SAVE_PATH)

if __name__ == "__main__":
    train()
