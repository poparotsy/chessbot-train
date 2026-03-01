#!/usr/bin/env python3
"""
🧐 BATCH BEAST RECOGNIZER
Compatible with V3 Synced Recognizer.
"""

import os
import glob
import sys

# Fixed import for v3
from recognizer_v3 import predict_board


def run_batch(folder_path):
    """Processes every image in the folder and prints a table."""
    print(f"🚀 Running batch inference with recognizer_v3...\n")

    patterns = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.PNG")
    images = []
    for pattern in patterns:
        images.extend(glob.glob(os.path.join(folder_path, pattern)))
    images = sorted(set(images))
    if not images:
        print(f"❓ No images found in {folder_path}")
        return

    print(f"{'Image Name':<25} | {'Conf':<6} | {'FEN Result'}")
    print("-" * 95)

    for img_path in images:
        name = os.path.basename(img_path)
        try:
            raw_fen, confidence = predict_board(img_path)
            # Add implicit turn since original FEN doesn't have it
            fen_with_turn = f"{raw_fen} w - - 0 1"
            print(f"{name:<25} | {confidence:.4f} | {fen_with_turn}")
        except Exception as err:
            print(f"{name:<25} | ERROR  | {str(err)}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "images_4_test"
    run_batch(target)
