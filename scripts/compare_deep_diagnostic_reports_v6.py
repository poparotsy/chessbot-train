#!/usr/bin/env python3
"""Compare before/after deep diagnostic snapshots and flag regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare before/after deep diagnostic reports.")
    parser.add_argument("--before-json", required=True)
    parser.add_argument("--after-json", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any image regresses without an offsetting positive deviation on that image.",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "images" not in data:
        raise ValueError(f"Invalid deep diagnostic report: {path}")
    return data


def status_rank(image_row: dict) -> int:
    current = image_row["current_model"]
    if current["chosen_board_ok"]:
        return 2
    if current["full_board_ok"]:
        return 1
    return 0


def status_label(image_row: dict) -> str:
    current = image_row["current_model"]
    if current["chosen_board_ok"]:
        return "chosen_ok"
    if current["full_board_ok"]:
        return "full_only"
    return "full_fail"


def index_by_name(report: dict) -> dict[str, dict]:
    return {Path(row["image"]).name: row for row in report["images"]}


def main() -> int:
    args = parse_args()
    before = load_report(Path(args.before_json))
    after = load_report(Path(args.after_json))
    before_rows = index_by_name(before)
    after_rows = index_by_name(after)
    names = sorted(set(before_rows) | set(after_rows))

    diffs = []
    regressions = []
    improvements = []
    unchanged = []

    for name in names:
        b = before_rows.get(name)
        a = after_rows.get(name)
        if b is None or a is None:
            diffs.append(
                {
                    "image": name,
                    "change": "added_or_removed",
                    "before_present": b is not None,
                    "after_present": a is not None,
                }
            )
            continue

        before_rank = status_rank(b)
        after_rank = status_rank(a)
        before_status = status_label(b)
        after_status = status_label(a)
        before_tag = b["current_model"]["selected"]["tag"] if b["current_model"]["selected"] else None
        after_tag = a["current_model"]["selected"]["tag"] if a["current_model"]["selected"] else None
        before_fen = b["current_model"]["selected"]["board_fen"] if b["current_model"]["selected"] else None
        after_fen = a["current_model"]["selected"]["board_fen"] if a["current_model"]["selected"] else None

        if after_rank > before_rank:
            change = "improved"
            improvements.append(name)
        elif after_rank < before_rank:
            change = "regressed"
            regressions.append(name)
        elif before_tag != after_tag or before_fen != after_fen:
            change = "deviated_same_grade"
            regressions.append(name)
        else:
            change = "unchanged"
            unchanged.append(name)

        diffs.append(
            {
                "image": name,
                "change": change,
                "before_status": before_status,
                "after_status": after_status,
                "before_selected_tag": before_tag,
                "after_selected_tag": after_tag,
                "before_selected_fen": before_fen,
                "after_selected_fen": after_fen,
                "before_verdict": b["current_model"]["verdict"],
                "after_verdict": a["current_model"]["verdict"],
            }
        )

    summary = {
        "tool": "compare_deep_diagnostic_reports_v6",
        "before_json": args.before_json,
        "after_json": args.after_json,
        "images": len(names),
        "improvements": improvements,
        "regressions": regressions,
        "unchanged": unchanged,
        "has_regression": bool(regressions),
        "diffs": diffs,
    }

    print("=== DEEP DIAGNOSTIC DIFF ===")
    print(f"images={summary['images']}")
    print(f"improvements={len(improvements)}")
    print(f"regressions={len(regressions)}")
    if improvements:
        print(f"  improved: {', '.join(improvements)}")
    if regressions:
        print(f"  regressed: {', '.join(regressions)}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.fail_on_regression and regressions:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
