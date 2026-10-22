"""Create a blank manual-review CSV and candidate-frame contact sheet."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.manual_review import (  # noqa: E402
    build_review_rows,
    write_candidate_contact_sheet,
    write_review_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare manual review artifacts from Phase 2.4 candidates.")
    parser.add_argument("events_csv", type=Path, help="Existing Phase 2.4 candidate-event CSV.")
    parser.add_argument("video", type=Path, help="Original source MP4 used to create the landmarks.")
    parser.add_argument("review_csv", type=Path, help="New blank-label review CSV.")
    parser.add_argument("contact_sheet", type=Path, help="New labeled candidate-frame contact sheet (JPG/PNG).")
    args = parser.parse_args()
    try:
        input_paths = {args.events_csv.resolve(), args.video.resolve()}
        output_paths = {args.review_csv.resolve(), args.contact_sheet.resolve()}
        if input_paths & output_paths or args.review_csv.resolve() == args.contact_sheet.resolve():
            raise ValueError("Review outputs must use new paths distinct from inputs and each other.")
        if any(path.exists() for path in (args.review_csv, args.contact_sheet)):
            raise ValueError("Refusing to overwrite an existing review artifact.")
        with args.events_csv.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"side", "frame_index", "timestamp_seconds", "signal_name", "signal_value_px"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError("Input is missing required Phase 2.4 event columns.")
            events = list(reader)
        rows = build_review_rows(events)
        write_candidate_contact_sheet(events, args.video, args.contact_sheet)
        write_review_csv(rows, args.review_csv)
    except (OSError, csv.Error, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Prepared {len(rows)} review rows with blank labels: {args.review_csv}")
    print(f"Prepared {len(events)} candidate frames for review: {args.contact_sheet}")
    print("Review labels manually; this command does not run pose inference or assign labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
