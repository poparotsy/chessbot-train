import sys
import json
import re
import torch
import numpy as np
from torch import nn
from PIL import Image

IMG_SIZE, FEN_CHARS, MODEL_PATH = 64, "1RNBQKPrnbqkp", "./models/model_v3_beast.pt"


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
    def _blk(
        ic,
        oc): return nn.Sequential(
        nn.Conv2d(
            ic,
            oc,
            3,
            padding=1),
        nn.BatchNorm2d(oc),
        nn.ReLU(True),
        nn.MaxPool2d(2))

    def forward(
        self, x): return self.classifier(
        self.conv4(
            self.conv3(
                self.conv2(
                    self.conv1(x)))))


def predict_board(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    xe, ye = np.linspace(0, w, 9).astype(int), np.linspace(0, h, 9).astype(int)
    fen, confs = [], []
    with torch.no_grad():
        for r in range(8):
            row = ""
            for c in range(8):
                tile = img.crop((xe[c], ye[r], xe[c + 1], ye[r + 1])
                                ).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
                it = (
                    torch.from_numpy(
                        np.transpose(
                            np.array(tile),
                            (2,
                             0,
                             1))).float().unsqueeze(0).to(device) / 127.5) - 1.0
                out = torch.softmax(model(it), dim=1)
                prob, pred = torch.max(out, 1)
                row += FEN_CHARS[pred.item()]
                confs.append(prob.item())
            fen.append(row)
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
