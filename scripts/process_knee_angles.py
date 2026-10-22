"""Smooth and inspect an existing Phase 2.2 knee-angle CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quality import diagnose_knee_angles  # noqa: E402
from src.report import KNEE_ANGLE_COLUMNS, write_smoothed_knee_angles_csv  # noqa: E402
from src.signal_processing import add_smoothed_knee_angles  # noqa: E402
from src.visualization import write_knee_angle_plot_svg  # noqa: E402


def _print_diagnostics(rows: list[dict[str, str]]) -> None:
    print("Adjacent-change flags are heuristic review cues, not proof of erroneous measurements.")
    for side in ("left", "right"):
        result = diagnose_knee_angles(rows, side)
        cutoff = (
            f"{result.tukey_cutoff_deg:.3f} deg"
            if result.tukey_cutoff_deg is not None
            else "unavailable"
        )
        print(
            f"{side.title()}: missing angles={result.missing_angle_count}; "
            f"usable={result.usable_angle_count}; statuses={result.status_counts}; "
            f"adjacent comparisons={result.adjacent_change_count}; "
            f"Tukey cutoff={cutoff}; flagged={len(result.large_change_flags)}"
        )
        for flag in result.large_change_flags:
            print(
                f"  heuristic flag: frames {flag.from_frame_index}->{flag.to_frame_index}, "
                f"absolute change={flag.absolute_change_deg:.3f} deg"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add segment-aware smoothed angles and diagnostics to a Phase 2.2 CSV."
    )
    parser.add_argument("input_csv", type=Path, help="Existing Phase 2.2 knee-angle CSV.")
    parser.add_argument("output_csv", type=Path, help="New Phase 2.3 CSV; the input is not overwritten.")
    parser.add_argument("--plot", type=Path, help="SVG plot path (default: output CSV path with .svg suffix).")
    args = parser.parse_args()

    try:
        plot_path = args.plot or args.output_csv.with_suffix(".svg")
        if args.input_csv.resolve() == args.output_csv.resolve():
            raise ValueError("Phase 2.3 output CSV must differ from the Phase 2.2 input CSV.")
        if plot_path.resolve() in {args.input_csv.resolve(), args.output_csv.resolve()}:
            raise ValueError("Plot output must differ from the input and output CSV paths.")
        with args.input_csv.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            original_columns = reader.fieldnames
            if original_columns is None or not set(KNEE_ANGLE_COLUMNS).issubset(original_columns):
                raise ValueError("Input must contain all Phase 2.2 knee-angle columns.")
            rows = list(reader)

        diagnostics_rows = [dict(row) for row in rows]
        _print_diagnostics(diagnostics_rows)
        smoothed_rows = add_smoothed_knee_angles(rows, window_size=5)
        write_smoothed_knee_angles_csv(smoothed_rows, original_columns, args.output_csv)
        write_knee_angle_plot_svg(smoothed_rows, plot_path)
    except (OSError, csv.Error, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Saved {len(rows)} rows to {args.output_csv}.")
    print(f"Saved raw/smoothed time-series plot to {plot_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
