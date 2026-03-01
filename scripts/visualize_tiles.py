import cv2
import numpy as np
import sys
import torch
from PIL import Image
import os
from recognizer_v3 import StandaloneBeastClassifier, find_board_corners, perspective_transform

os.makedirs('images_4_test/debug/tiles', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = StandaloneBeastClassifier().to(device)
m.load_state_dict(torch.load('models/model_hybrid_100e.pt', map_location=device))
m.eval()

img = Image.open('images_4_test/puzzle-00002.png').convert('RGB')
corners = find_board_corners(img)
if corners is not None:
    img = perspective_transform(img, corners)

w, h = img.size
xe, ye = np.linspace(0, w, 9).astype(int), np.linspace(0, h, 9).astype(int)

FEN_CHARS = "1PNBRQKpnbrqk"

for r in range(8):
    for c in range(8):
        tile = img.crop((xe[c], ye[r], xe[c + 1], ye[r + 1])).resize((64, 64), Image.LANCZOS)
        tile.save(f'images_4_test/debug/tiles/tile_{r}_{c}.png')
        
        img_np = np.array(tile).transpose(2,0,1)
        it = torch.from_numpy(img_np).float().to(device)
        it = (it / 127.5) - 1.0
        it = it.unsqueeze(0)
        out = torch.softmax(m(it), dim=1)
        prob, pred = torch.max(out, 1)
        
        # print non-empty predictions
        if pred.item() != 0:
            print(f'({r}, {c}) -> {FEN_CHARS[pred.item()]} [{prob.item():.4f}]')
