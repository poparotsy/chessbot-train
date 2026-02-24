#!/usr/bin/env python3
"""
🧐 BATCH BEAST RECOGNIZER
Compatible with V3 Synced Recognizer.
"""

import os
import glob
import json
import sys
from PIL import Image

# Correct Import (only 2 values)
from chess_recognizer_v3 import get_inference_tools, process_tiles, compress_fen


def run_batch(folder_path):
    """Processes every image in the folder and prints a table."""
    try:
        # Fixed: Unpacks 2 values to match recognizer script
        device, model = get_inference_tools()
        print(f"🚀 Model loaded on {device}. Thresholding noise...\n")
    except Exception as err:
        print(f"❌ Failed to load model: {err}")
        return

    images = sorted(glob.glob(os.path.join(folder_path, "*.[jp][pn]g")))
    if not images:
        print(f"❓ No images found in {folder_path}")
        return

    print(f"{'Image Name':<25} | {'Conf':<6} | {'FEN Result'}")
    print("-" * 95)

    for img_path in images:
        name = os.path.basename(img_path)
        try:
            img = Image.open(img_path).convert("RGB")
            # Fixed: Only passes img, model, device
            raw_fen, confidence = process_tiles(img, model, device)
            print(f"{name:<25} | {confidence:.4f} | {compress_fen(raw_fen)}")
        except Exception as err:
            print(f"{name:<25} | ERROR  | {str(err)}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "images_4_test"
    run_batch(target)
