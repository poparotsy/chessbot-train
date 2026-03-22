#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.dirname(SCRIPT_DIR)
V5_DIR = os.path.join(SCRIPT_DIR, "v5")
V4_DIR = os.path.join(SCRIPT_DIR, "v4")
for path in (TRAIN_DIR, V5_DIR, V4_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import recognizer_v5 as r5

r4 = r5.v4
args_no_edge_detection = False


@dataclass
class ModelSpec:
    name: str
    path: str


def parse_model_specs(items):
    specs = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = os.path.splitext(os.path.basename(path))[0]
        specs.append(ModelSpec(name=name.strip(), path=path.strip()))
    return specs


def square_name(row, col):
    return f"{'abcdefgh'[col]}{8 - row}"


def build_candidates(original_img):
    if args_no_edge_detection:
        return [("full", original_img)]
    return r5.BoardDetector(debug=False).candidate_images(original_img)


def classify_board(model, device, board_img, topk):
    w, h = board_img.size
    xe = np.linspace(0, w, 9).astype(int)
    ye = np.linspace(0, h, 9).astype(int)

    grid = []
    mean_conf = 0.0
    piece_count = 0

    with torch.no_grad():
        for rr in range(8):
            row = []
            for cc in range(8):
                tile = board_img.crop((xe[cc], ye[rr], xe[cc + 1], ye[rr + 1])).resize(
                    (r4.IMG_SIZE, r4.IMG_SIZE), Image.LANCZOS
                )
                arr = np.array(tile).transpose(2, 0, 1)
                t = torch.from_numpy(arr).float().to(device)
                t = (t / 127.5) - 1.0
                t = t.unsqueeze(0)

                out = torch.softmax(model(t), dim=1)[0]
                vals, idx = torch.topk(out, topk)
                top1_prob = float(vals[0].item())
                top1_label = r4.FEN_CHARS[int(idx[0].item())]
                label = "1" if top1_prob < 0.35 else top1_label
                if label != "1":
                    piece_count += 1

                row.append(
                    {
                        "label": label,
                        "top1_prob": top1_prob,
                        "topk": [
                            (r4.FEN_CHARS[int(i.item())], float(v.item()))
                            for v, i in zip(vals, idx)
                        ],
                    }
                )
                mean_conf += top1_prob
            grid.append(row)

    mean_conf /= 64.0
    return grid, mean_conf, piece_count


def fen_from_grid(grid):
    import re

    rows = []
    for rr in range(8):
        s = "".join(grid[rr][cc]["label"] for cc in range(8))
        rows.append(re.sub(r"1+", lambda m: str(len(m.group())), s))
    return "/".join(rows)


def select_candidate(scored):
    full_conf = next((it["mean_conf"] for it in scored if it["tag"] == "full"), 0.0)
    full_pieces = next((it["piece_count"] for it in scored if it["tag"] == "full"), 0)

    filtered = []
    for item in scored:
        if (
            item["tag"] != "full"
            and full_conf >= r4.FULL_CONF_FOR_COVERAGE_GUARD
            and full_pieces >= r4.WARP_PIECE_COVERAGE_MIN_FULL
            and item["piece_count"] <= int(full_pieces * r4.WARP_PIECE_COVERAGE_RATIO)
        ):
            continue
        filtered.append(item)
    if not filtered:
        filtered = scored

    return max(filtered, key=lambda it: (r5.board_plausibility_score(it["fen"]), it["mean_conf"]))


def load_model(path, device):
    model = r4.StandaloneBeastClassifier(num_classes=13).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Compare per-tile predictions across models.")
    parser.add_argument("image", help="Path to puzzle image")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model specs: name=path or path (space separated)",
    )
    parser.add_argument("--topk", type=int, default=3, help="Top-k classes per tile")
    parser.add_argument("--only-diff", action="store_true", help="Print only differing squares")
    parser.add_argument("--json", action="store_true", help="Emit JSON payload")
    parser.add_argument("--no-edge-detection", action="store_true", help="Disable edge candidates")
    args = parser.parse_args()

    global args_no_edge_detection
    args_no_edge_detection = args.no_edge_detection
    specs = parse_model_specs(args.models)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original = Image.open(args.image).convert("RGB")
    candidates = build_candidates(original)

    results = {}
    for spec in specs:
        model = load_model(spec.path, device)
        scored = []
        for tag, board_img in candidates:
            grid, mean_conf, piece_count = classify_board(model, device, board_img, args.topk)
            scored.append(
                {
                    "tag": tag,
                    "grid": grid,
                    "mean_conf": mean_conf,
                    "piece_count": piece_count,
                    "fen": fen_from_grid(grid),
                }
            )
        best = select_candidate(scored)
        results[spec.name] = {
            "model_path": spec.path,
            "candidate": best["tag"],
            "mean_conf": best["mean_conf"],
            "piece_count": best["piece_count"],
            "fen": fen_from_grid(best["grid"]),
            "grid": best["grid"],
        }

    model_names = list(results.keys())
    base = model_names[0]
    diffs = []
    for rr in range(8):
        for cc in range(8):
            labels = {name: results[name]["grid"][rr][cc]["label"] for name in model_names}
            if len(set(labels.values())) > 1:
                diffs.append((rr, cc, labels))

    payload = {
        "image": args.image,
        "models": {
            name: {
                "model_path": results[name]["model_path"],
                "candidate": results[name]["candidate"],
                "mean_conf": round(results[name]["mean_conf"], 4),
                "piece_count": results[name]["piece_count"],
                "fen": results[name]["fen"],
            }
            for name in model_names
        },
        "diff_count": len(diffs),
        "diff_squares": [
            {
                "square": square_name(rr, cc),
                "labels": labels,
                "topk": {
                    name: results[name]["grid"][rr][cc]["topk"] for name in model_names
                },
            }
            for rr, cc, labels in diffs
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Image: {args.image}")
    for name in model_names:
        meta = payload["models"][name]
        print(
            f"[{name}] candidate={meta['candidate']} conf={meta['mean_conf']:.4f} "
            f"pieces={meta['piece_count']} fen={meta['fen']}"
        )
    print(f"Differing squares: {payload['diff_count']}")

    for item in payload["diff_squares"]:
        if args.only_diff:
            print(f"- {item['square']} labels={item['labels']}")
            continue
        print(f"- {item['square']}")
        for name in model_names:
            topk_txt = ", ".join([f"{lab}:{prob:.3f}" for lab, prob in item["topk"][name]])
            print(f"  {name}: {item['labels'][name]} [{topk_txt}]")


if __name__ == "__main__":
    main()
