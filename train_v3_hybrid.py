import glob
import os
import time
import torch
from torch import nn, amp
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

BATCH_SIZE, EPOCHS, LR = 1024, 30, 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT, BEST_MODEL = "models/checkpoint_v3.pt", "models/model_v3_beast.pt"


class StandaloneBeastClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super().__init__()
        self.conv1 = self._blk(3, 64)
        self.conv2 = self._blk(64, 128)
        self.conv3 = self._blk(128, 256)
        self.conv4 = self._blk(256, 512)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(512 * 4 * 4, 1024), nn.ReLU(True),
                                        nn.Dropout(0.4), nn.Linear(1024, 512), nn.ReLU(True), nn.Linear(512, num_classes))

    @staticmethod
    def _blk(ic, oc):
        return nn.Sequential(nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(
            oc), nn.ReLU(True), nn.MaxPool2d(2))

    def forward(self, x):
        return self.classifier(self.conv4(
            self.conv3(self.conv2(self.conv1(x)))))


def train():
    print(f"🚀 HYBRID BEAST STARTING ON {DEVICE}")
    files = sorted(glob.glob("tensors_v3/train_chunk_*.pt"))
    x = torch.cat([torch.load(f, map_location='cpu')['x'] for f in files])
    y = torch.cat([torch.load(f, map_location='cpu')['y']
                  for f in files]).long()
    if x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)

    loader = DataLoader(
        TensorDataset(
            x,
            y),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True)
    model = StandaloneBeastClassifier(num_classes=13).to(DEVICE)

    # 0.8/1.2 BALANCED WEIGHTS
    weights = torch.ones(13).to(DEVICE)
    weights[0], weights[1:] = 0.8, 1.2
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scaler, start_epoch, best_loss = amp.GradScaler('cuda'), 0, float('inf')

    # RESUME & DATA PARALLEL
    if os.path.exists(CHECKPOINT):
        ckpt = torch.load(CHECKPOINT, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        best_loss = ckpt.get('best_loss', 1e6)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 2,
        steps_per_epoch=len(loader),
        epochs=max(
            EPOCHS - start_epoch,
            1))
    augmenter = nn.Sequential(
        transforms.RandomAffine(
            0, translate=(
                0.07, 0.07), scale=(
                0.95, 1.05))).to(DEVICE)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t_start, r_loss = time.time(), 0
        for i, (imgs, lbls) in enumerate(loader):
            imgs, lbls = imgs.to(DEVICE, non_blocking=True), lbls.to(DEVICE)
            imgs = augmenter((imgs.float() / 127.5) - 1.0)
            with amp.autocast('cuda'):
                loss = criterion(model(imgs), lbls)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            r_loss += loss.item()
            if i % 100 == 0:
                sps = (i * BATCH_SIZE) / (time.time() - t_start + 1e-6)
                print(
                    f"   Epoch {epoch + 1} | {i}/{len(loader)} | {sps:.0f} img/s | Loss: {loss.item():.4f}")

        avg_l = r_loss / len(loader)
        print(f"✅ Epoch {epoch + 1} | Avg Loss: {avg_l:.4f}")
        os.makedirs("models", exist_ok=True)
        raw = model.module if hasattr(model, 'module') else model
        torch.save({'epoch': epoch,
                    'model_state_dict': raw.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_loss': min(best_loss,
                                     avg_l)},
                   CHECKPOINT)
        if avg_l < best_loss:
            best_loss = avg_l
            torch.save(raw.state_dict(), BEST_MODEL)
            print("✨ Saved Best Model")


if __name__ == "__main__":
    train()
