#!/usr/bin/env python3
"""
Regression checker for recognizer_v3 against manual_recognize_v3.txt.
Fails fast if any image prediction drifts.
"""

import json
import os
import re
import sys
from typing import Dict, List, Tuple

from recognizer_v3 import predict_board


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGES_DIR = os.path.join(ROOT_DIR, "images_4_test")
MANUAL_FILE = os.path.join(ROOT_DIR, "manual_recognize_v3.txt")

LINE_PATTERN = re.compile(r"^(puzzle-\d+\.(?:png|jpe?g))\s*:\s*(.*)$")


def load_expected(path: str) -> Dict[str, str]:
    expected: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            match = LINE_PATTERN.match(line)
            if not match:
                continue
            name, rest = match.groups()

            fen_match = re.search(r'"fen"\s*:\s*"([^"]+)"', rest)
            if fen_match:
                expected[name] = fen_match.group(1)
                continue

            quoted = re.findall(r'"([^"]+)"', rest)
            if quoted:
                expected[name] = quoted[0]
    return expected


def list_images(path: str) -> List[str]:
    out: List[str] = []
    for entry in sorted(os.listdir(path)):
        if entry.lower().endswith((".png", ".jpg", ".jpeg")):
            out.append(entry)
    return out


def main() -> int:
    expected = load_expected(MANUAL_FILE)
    images = list_images(IMAGES_DIR)

    mismatches: List[Tuple[str, str, str]] = []
    missing_expected: List[str] = []

    for name in images:
        if name not in expected:
            missing_expected.append(name)
            continue
        image_path = os.path.join(IMAGES_DIR, name)
        raw_fen, _ = predict_board(image_path)
        predicted = f"{raw_fen} w - - 0 1"
        if predicted != expected[name]:
            mismatches.append((name, expected[name], predicted))

    if missing_expected:
        print("Missing expected labels for:")
        for name in missing_expected:
            print(f"  - {name}")
        return 2

    if mismatches:
        print(f"FAIL: {len(mismatches)} mismatches")
        for name, exp, got in mismatches:
            print(f"\n{name}")
            print(f"  expected: {exp}")
            print(f"  got     : {got}")
        return 1

    print(f"PASS: {len(images)} / {len(images)} images match manual_recognize_v3.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
