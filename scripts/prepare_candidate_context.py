"""Create a before/candidate/after contact sheet for every saved candidate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.manual_review import (  # noqa: E402
    REVIEW_COLUMNS,
    align_candidate_context_targets,
    write_review_sequence_contact_sheet,
)


def _paths_overlap(output_dir: Path, input_path: Path) -> bool:
    output = output_dir.resolve()
    source = input_path.resolve()
    return output == source or output in source.parents or source in output.parents


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a three-frame context sheet for each existing Phase 2.4 candidate."
    )
    parser.add_argument("events_csv", type=Path, help="Existing Phase 2.4 candidate CSV.")
    parser.add_argument("review_csv", type=Path, help="Existing Phase 3.2 review CSV with stable event IDs.")
    parser.add_argument("video", type=Path, help="Original source MP4; decoded for display only.")
    parser.add_argument("output_dir", type=Path, help="New output directory; an existing directory is rejected.")
    parser.add_argument("--offset-frames", type=int, default=3,
                        help="Frames before and after the candidate (default: 3).")
    args = parser.parse_args()

    try:
        if args.offset_frames <= 0:
            raise ValueError("--offset-frames must be positive.")
        input_paths = (args.events_csv, args.review_csv, args.video)
        if any(not path.is_file() for path in input_paths):
            missing = next(path for path in input_paths if not path.is_file())
            raise FileNotFoundError(f"Required input does not exist: {missing}")
        if args.output_dir.exists():
            raise FileExistsError(f"Refusing to use existing output directory: {args.output_dir}")
        if any(_paths_overlap(args.output_dir, path) for path in input_paths):
            raise ValueError("Output directory must be separate from every input path.")

        with args.events_csv.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"side", "frame_index", "timestamp_seconds", "signal_name", "signal_value_px"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError("Candidate CSV is missing required Phase 2.4 columns.")
            candidate_rows = list(reader)
        with args.review_csv.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(REVIEW_COLUMNS).issubset(reader.fieldnames or ()):
                raise ValueError("Review CSV is missing required Phase 2.5 columns.")
            review_rows = list(reader)

        targets = align_candidate_context_targets(review_rows, candidate_rows)
        output = args.output_dir / "candidate_context_triptych_phase32.jpg"
        write_review_sequence_contact_sheet(
            targets,
            args.video,
            output,
            offset_frames=args.offset_frames,
        )
    except (OSError, csv.Error, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Created context sheet for {len(targets)} candidates: {output}")
    print(f"Frames are offset by {args.offset_frames} source frames on each side.")
    print("The candidate events remain exploratory and are not confirmed foot-strikes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
