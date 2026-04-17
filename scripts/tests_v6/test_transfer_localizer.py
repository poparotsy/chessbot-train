from __future__ import annotations

import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent.parent
TRANSFER_DIR = SCRIPT_DIR / "transfer_localizer"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(TRANSFER_DIR) not in sys.path:
    sys.path.insert(0, str(TRANSFER_DIR))

from common import normalize_localizer_manifest_row
from compare_recognizers import _compare
from export_eval_payload import _load_sam2_rows
from train_localizer import _load_rows


def test_normalize_localizer_manifest_row_orders_corners_and_bbox():
    row = normalize_localizer_manifest_row(
        {
            "id": "sample",
            "corners_px": [[20, 20], [5, 5], [20, 5], [5, 20]],
        }
    )
    assert row["corners_px"] == [[5.0, 5.0], [20.0, 5.0], [20.0, 20.0], [5.0, 20.0]]
    assert row["bbox_xyxy"] == [5.0, 5.0, 20.0, 20.0]


def test_load_sam2_rows_applies_defaults(tmp_path: Path):
    input_path = tmp_path / "sam2.jsonl"
    input_path.write_text(json.dumps({"id": "sam2-1", "image_path": "images/foo.png", "corners_px": [[0, 0], [10, 0], [10, 10], [0, 10]]}) + "\n", encoding="utf-8")
    rows = _load_sam2_rows([str(input_path)])
    assert len(rows) == 1
    assert rows[0]["source_type"] == "sam2_real"
    assert rows[0]["label_status"] == "labeled"
    assert rows[0]["bbox_xyxy"] == [0.0, 0.0, 10.0, 10.0]


def test_train_localizer_load_rows_filters_unlabeled(tmp_path: Path):
    manifest = tmp_path / "localizer_train.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps({"id": "labeled", "image_path": "images/a.png", "bbox_xyxy": [0, 0, 10, 10]}),
                json.dumps({"id": "unlabeled", "image_path": "images/b.png", "bbox_xyxy": None}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = _load_rows(manifest)
    assert [row["id"] for row in rows] == ["labeled"]


def test_compare_recognizers_reports_preserved_wins_and_regressions():
    baseline = {
        "hard": {
            "results": [
                {"image": "a.png", "board_ok": True},
                {"image": "b.png", "board_ok": False},
                {"image": "c.png", "board_ok": True},
            ]
        },
        "blocker_failures": ["b.png"],
    }
    candidate = {
        "hard": {
            "results": [
                {"image": "a.png", "board_ok": True},
                {"image": "b.png", "board_ok": True},
                {"image": "c.png", "board_ok": False},
            ]
        },
        "blocker_failures": [],
    }
    diff = _compare(baseline, candidate)
    assert diff["preserved_wins"] == ["a.png"]
    assert diff["new_wins"] == ["b.png"]
    assert diff["regressions"] == ["c.png"]
