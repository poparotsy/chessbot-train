import sys
import json
import re
import torch
import numpy as np
from torch import nn
from PIL import Image

IMG_SIZE, FEN_CHARS, MODEL_PATH = 64, "1PNBRQKpnbrqk", "./models/model_hybrid_beast.pt"


############
class StandaloneBeastClassifier(nn.Module):
    def __init__(self, num_classes=13):
        super(StandaloneBeastClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2)
        )
        
        # This exact order ensures index 1 and 4 are the Linear layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),               # index 0
            nn.Linear(512 * 4 * 4, 1024),  # index 1 (MATCHES WEIGHTS)
            nn.ReLU(),                     # index 2
            nn.Dropout(0.3),               # index 3
            nn.Linear(1024, num_classes)   # index 4 (MATCHES WEIGHTS)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten here so it doesn't mess up the classifier indexing
        return self.classifier(x)

############
# 1. Update the Mapping Law

def predict_board(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # 2. Ensure RGB processing (Trainer used PIL, so stay in RGB)
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    xe, ye = np.linspace(0, w, 9).astype(int), np.linspace(0, h, 9).astype(int)
    
    fen, confs = [], []
    with torch.no_grad():
        for r in range(8):
            row = ""
            for c in range(8):
                # Crop and resize
                tile = img.crop((xe[c], ye[r], xe[c + 1], ye[r + 1])).resize((64, 64), Image.LANCZOS)
                
                # 3. Apply "THE LAW" Normalization exactly as in Trainer
                # Transpose to [C, H, W] then scale
                img_np = np.array(tile).transpose(2, 0, 1)
                it = torch.from_numpy(img_np).float().to(device)
                it = (it / 127.5) - 1.0 
                it = it.unsqueeze(0)
                
                # Inference
                out = torch.softmax(model(it), dim=1)
                prob, pred = torch.max(out, 1)
                
                row += FEN_CHARS[pred.item()]
                confs.append(prob.item())
            fen.append(row)
    # ... rest of your FEN compression logic ...
    res = "/".join(fen)
    res = "/".join([re.sub(r'1+', lambda m: str(len(m.group())), r)
                   for r in res.split('/')])
    return res, np.mean(confs)


if __name__ == "__main__":
    try:
        f, c = predict_board(sys.argv[1])
        print(json.dumps(
            {"success": True, "fen": f"{f} w - - 0 1", "confidence": round(c, 4)}))
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
