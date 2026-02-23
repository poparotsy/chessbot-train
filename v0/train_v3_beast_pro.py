"""
🚀 V3 BEAST TRAINER - BRAVERY EDITION
Objectives: 64x64 RGB, Weighted Loss (0.5 vs 1.0), and Translation Jitter.
This version fixes the 'Empty Board' and 'Piece Confusion' errors.
"""

import glob
import os
import time

# pylint: disable=import-error
import torch
from torch import nn, amp
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

# ============ CONFIG ============
BATCH_SIZE = 1024
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "models/checkpoint_beast_v3.pt"
BEST_MODEL_PATH = "models/model_beast_v3.pt"


# pylint: disable=too-few-public-methods
class StandaloneBeastClassifier(nn.Module):
    """Standalone 64x64 RGB Architecture."""

    def __init__(self, num_classes=13):
        super().__init__()
        self.conv1 = self._make_block(3, 64)
        self.conv2 = self._make_block(64, 128)
        self.conv3 = self._make_block(128, 256)
        self.conv4 = self._make_block(256, 512)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    @staticmethod
    def _make_block(in_c, out_c):
        """Conv Block."""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    # pylint: disable=arguments-differ
    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.classifier(x)


def load_v3_data():
    """Loads byte-tensors and ensures [N, C, H, W] shape."""
    files = sorted(glob.glob("tensors_v3/train_chunk_*.pt"))
    if not files:
        raise FileNotFoundError("No chunks found in tensors_v3/")

    x_data = torch.cat([torch.load(f, map_location='cpu')['x'] for f in files])
    y_data = torch.cat([torch.load(f, map_location='cpu')['y']
                       for f in files]).long()

    if x_data.shape[-1] == 3:
        x_data = x_data.permute(0, 3, 1, 2)
    return TensorDataset(x_data, y_data)


def save_v3_checkpoint(model, optimizer, scaler, stats):
    """Saves checkpoint and best model (clean of module prefix)."""
    os.makedirs("models", exist_ok=True)
    raw = model.module if hasattr(model, 'module') else model
    torch.save({
        'epoch': stats['epoch'],
        'model_state_dict': raw.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_loss': min(stats['best_loss'], stats['avg_loss'])
    }, CHECKPOINT_PATH)
    if stats['avg_loss'] < stats['best_loss']:
        torch.save(raw.state_dict(), BEST_MODEL_PATH)
        print(f"✨ New Best Model Saved (Loss: {stats['avg_loss']:.4f})")
        return stats['avg_loss']
    return stats['best_loss']


# pylint: disable=too-many-locals
def train():
    """Main training loop with Jitter Augmentation."""
    print(f"🚀 V3 BRAVERY STARTING ON {DEVICE}")

    # 1. Load Data
    dataset = load_v3_data()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True)

    # 2. Setup Model & Bravery Weights
    model = StandaloneBeastClassifier(num_classes=13).to(DEVICE)
    weights = torch.ones(13).to(DEVICE)
    weights[0], weights[1:] = 0.5, 5.0  # BRAVERY FIX

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scaler = amp.GradScaler('cuda')

    # 3. SPATIAL AUGMENTER (The 'Piece Detection' Fix)
    augmenter = nn.Sequential(
        transforms.RandomAffine(
            degrees=0, translate=(
                0.7, 0.7), scale=(
                0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ).to(DEVICE)

    start_epoch, best_loss = 0, float('inf')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE * 2,
        steps_per_epoch=len(loader), epochs=EPOCHS
    )

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_start, running_loss = time.time(), 0
        for _, (imgs, lbls) in enumerate(loader):
            imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE)

            # Normalization + JITTER
            imgs = augmenter((imgs.float() / 127.5) - 1.0)

            with amp.autocast('cuda'):
                loss = criterion(model(imgs), lbls)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(
            f"✅ Epoch {
                epoch +
                1} | Loss: {
                avg_loss:.4f} | Time: {
                time.time() -
                epoch_start:.1f}s")
        best_loss = save_v3_checkpoint(model, optimizer, scaler,
                                       {'epoch': epoch, 'avg_loss': avg_loss, 'best_loss': best_loss})


if __name__ == "__main__":
    train()
