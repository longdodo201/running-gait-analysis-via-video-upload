"""Create before/candidate/after contact sheet for uncertain/low-confidence rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.manual_review import (  # noqa: E402
    select_review_targets,
    write_review_sequence_contact_sheet,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract contextual video frames for uncertain or low-confidence manual-review events."
    )
    parser.add_argument("review_csv", type=Path, help="Current Phase 2.5 review CSV; it is read-only here.")
    parser.add_argument("events_csv", type=Path, help="Phase 2.4 candidate CSV with landmark confidence.")
    parser.add_argument("video", type=Path, help="Original source MP4.")
    parser.add_argument("output_contact_sheet", type=Path, help="New output image path; existing files are not overwritten.")
    parser.add_argument("--offset-seconds", type=float, default=0.2)
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Below-threshold visibility or presence selects an event for review (default: detector cutoff 0.5).")
    args = parser.parse_args()
    try:
        if args.output_contact_sheet.resolve() in {
            args.review_csv.resolve(), args.events_csv.resolve(), args.video.resolve()
        }:
            raise ValueError("Contact sheet output must be a new path distinct from inputs.")
        if args.output_contact_sheet.exists():
            raise ValueError("Refusing to overwrite an existing contact sheet.")
        with args.review_csv.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not {"event_id", "side", "frame_index", "timestamp_seconds", "manual_label"}.issubset(reader.fieldnames or ()):
                raise ValueError("Review CSV is missing required Phase 2.5 columns.")
            review_rows = list(reader)
        with args.events_csv.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"side", "frame_index", "landmark_visibility", "landmark_presence"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError("Phase 2.4 CSV is missing candidate confidence columns.")
            candidate_rows = list(reader)
        targets = select_review_targets(review_rows, candidate_rows, args.confidence_threshold)
        if not targets:
            print("No uncertain or low-confidence events found; no frames were extracted and no output was created.")
            print(f"Low-confidence threshold: visibility or presence below {args.confidence_threshold:g}.")
            print("Manual labels and source CSV files were left unchanged.")
            return 0
        output = write_review_sequence_contact_sheet(
            targets, args.video, args.output_contact_sheet, args.offset_seconds
        )
    except (OSError, csv.Error, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Created context sheet for {len(targets)} events at {output}.")
    for target in targets:
        print(f"  {target['event_id']} {target['side']} frame {target['frame_index']} "
              f"t={target['timestamp_seconds']} reason={target['review_reason']}")
    print("Each row contains frames at approximately -offset, candidate frame, and +offset; no labels were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
