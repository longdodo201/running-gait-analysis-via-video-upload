"""Calculate frame-level 2D knee angles from Phase 1 landmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gait_metrics import knee_angle_rows  # noqa: E402
from src.report import write_knee_angles_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate per-frame 2D knee angles from landmark JSON.")
    parser.add_argument("input_json", type=Path, help="Phase 1 landmark JSON path.")
    parser.add_argument("output_csv", type=Path, help="Destination CSV path.")
    parser.add_argument("--visibility-threshold", type=float, default=0.5)
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    args = parser.parse_args()

    try:
        with args.input_json.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError("Input JSON root must be an object.")
        rows = knee_angle_rows(document, args.visibility_threshold, args.presence_threshold)
        write_knee_angles_csv(rows, args.output_csv)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote knee angles for {len(rows)} frames to {args.output_csv}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
