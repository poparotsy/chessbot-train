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
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.7),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 13)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten here so it doesn't mess up the classifier indexing
        return self.classifier(x)

############
# 1. Update the Mapping Law

def preprocess_image(img):
    """Apply Chess.com-style preprocessing to improve recognition"""
    from PIL import ImageEnhance, ImageFilter
    
    # 1. Slight sharpening to enhance edges
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    
    # 2. Contrast enhancement
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.1)
    
    # 3. Brightness normalization
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.05)
    
    return img

def predict_board(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    img = preprocess_image(img)  # Apply preprocessing
    
    w, h = img.size
    xe, ye = np.linspace(0, w, 9).astype(int), np.linspace(0, h, 9).astype(int)
    
    fen, confs = [], []
    with torch.no_grad():
        for r in range(8):
            row = ""
            for c in range(8):
                # Crop and resize with preprocessing
                tile = img.crop((xe[c], ye[r], xe[c + 1], ye[r + 1])).resize((64, 64), Image.LANCZOS)
                
                # Apply "THE LAW" Normalization exactly as in Trainer
                img_np = np.array(tile).transpose(2, 0, 1)
                it = torch.from_numpy(img_np).float().to(device)
                it = (it / 127.5) - 1.0 
                it = it.unsqueeze(0)
                
                # Inference
                out = torch.softmax(model(it), dim=1)
                prob, pred = torch.max(out, 1)
                
                # Empty square confidence threshold (prevent hallucination)
                if prob.item() < 0.35:
                    row += '1'  # Low confidence = empty square
                else:
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
