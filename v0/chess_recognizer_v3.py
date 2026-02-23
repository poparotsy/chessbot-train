#!/usr/bin/env python3
"""
🚀 V3 BEAST RECOGNIZER - FINAL PRODUCTION VERSION
Features: Exact Linspace Slicing, Synced Normalization, and Hallucination Filtering.
"""

import os
import sys
import json
import re
import torch
import numpy as np
from torch import nn
from PIL import Image

# ============ CONFIG ============
IMG_SIZE = 64
FEN_CHARS = "1RNBQKPrnbqkp"
MODEL_PATH = "./models/model_beast_v3.pt"
CONF_THRESHOLD = 0.65  # Filter out 'Bravery' noise


class StandaloneBeastClassifier(nn.Module):
    """The brain - 64x64 RGB Architecture."""

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
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.classifier(x)


def compress_fen(board_fen):
    """Converts 1111 -> 4 notation."""
    rows = board_fen.split('/')
    return '/'.join([re.sub(r'1+', lambda m: str(len(m.group())), r)
                    for r in rows])


def get_inference_tools():
    """Loads model and returns (device, model)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StandaloneBeastClassifier(num_classes=13).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return device, model


def process_tiles(full_img, model, device):
    """Slices image and predicts pieces with hallucination filtering."""
    width, height = full_img.size
    x_edges = np.linspace(0, width, 9).astype(int)
    y_edges = np.linspace(0, height, 9).astype(int)
    fen_rows, total_conf = [], 0

    with torch.no_grad():
        for r in range(8):
            row_str = ""
            for c in range(8):
                tile = full_img.crop(
                    (x_edges[c], y_edges[r], x_edges[c + 1], y_edges[r + 1]))
                tile = tile.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

                # Normalization Synced to 'Bravery' Trainer
                img_np = np.array(tile).transpose(2, 0, 1)
                img_t = torch.from_numpy(
                    img_np).float().unsqueeze(0).to(device)
                img_t = (img_t / 127.5) - 1.0

                outputs = model(img_t)
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)

                # FILTER: If not confident, or if it's a weak guess, mark as
                # Empty
                if conf.item() < CONF_THRESHOLD:
                    row_str += "1"
                else:
                    row_str += FEN_CHARS[pred.item()]

                total_conf += conf.item()
            fen_rows.append(row_str)

    return "/".join(fen_rows), (total_conf / 64)


def main():
    if len(sys.argv) < 2:
        return
    try:
        device, model = get_inference_tools()
        img = Image.open(sys.argv[1]).convert("RGB")
        raw_fen, confidence = process_tiles(img, model, device)
        compressed = compress_fen(raw_fen)
        print(json.dumps({
            "success": True,
            "fen": f"{compressed} w - - 0 1",
            "confidence": round(confidence, 4)
        }))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))


if __name__ == "__main__":
    main()
