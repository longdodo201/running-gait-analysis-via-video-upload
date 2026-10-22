"""Run the exploratory Phase 2.4 heel-trajectory candidate detector."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.gait_events import (  # noqa: E402
    EventDetectionParameters,
    detect_leg_cycle_candidates,
    validate_phase23_alignment,
)
from src.report import write_candidate_events_csv  # noqa: E402
from src.visualization import write_gait_event_diagnostic_svg  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find unvalidated heel vertical low-point candidates from saved landmarks."
    )
    parser.add_argument("landmarks_json", type=Path, help="Saved Phase 1 landmark JSON.")
    parser.add_argument("phase23_csv", type=Path, help="Aligned Phase 2.3 CSV for context plotting.")
    parser.add_argument("output_csv", type=Path, help="New Phase 2.4 candidate-event CSV.")
    parser.add_argument("--plot", type=Path, help="SVG diagnostic plot (default: output CSV with .svg suffix).")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--smoothing-window-frames", type=int, default=5)
    parser.add_argument("--minimum-prominence-px", type=float, default=20.0)
    parser.add_argument("--minimum-separation-frames", type=int, default=12)
    parser.add_argument("--prominence-window-frames", type=int, default=30)
    args = parser.parse_args()

    try:
        plot_path = args.plot or args.output_csv.with_suffix(".svg")
        inputs = {args.landmarks_json.resolve(), args.phase23_csv.resolve()}
        outputs = {args.output_csv.resolve(), plot_path.resolve()}
        if inputs & outputs or args.output_csv.resolve() == plot_path.resolve():
            raise ValueError("Phase 2.4 outputs must use new paths distinct from inputs and each other.")
        if any(path.exists() for path in (args.output_csv, plot_path)):
            raise ValueError("Refusing to overwrite an existing Phase 2.4 output.")

        with args.landmarks_json.open("r", encoding="utf-8") as stream:
            document = json.load(stream)
        with args.phase23_csv.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            knee_rows = list(reader)
            columns = set(reader.fieldnames or ())
        required = {
            "frame_index", "timestamp_seconds", "left_knee_angle_deg",
            "right_knee_angle_deg", "left_knee_angle_smoothed_deg",
            "right_knee_angle_smoothed_deg",
        }
        if not required.issubset(columns):
            raise ValueError("Phase 2.3 CSV is missing required frame, time, or knee-angle columns.")

        settings = EventDetectionParameters(
            confidence_threshold=args.confidence_threshold,
            smoothing_window_frames=args.smoothing_window_frames,
            minimum_prominence_pixels=args.minimum_prominence_px,
            minimum_separation_frames=args.minimum_separation_frames,
            prominence_window_frames=args.prominence_window_frames,
        )
        result = detect_leg_cycle_candidates(document, settings)
        validate_phase23_alignment(result.samples, knee_rows)
        write_candidate_events_csv(result.events, args.output_csv)
        write_gait_event_diagnostic_svg(result.samples, result.events, knee_rows, plot_path)
    except (OSError, csv.Error, json.JSONDecodeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Event: local maximum in image-y of median-smoothed heel trajectory; candidate only, not foot-strike.")
    print(f"Settings: confidence={settings.confidence_threshold:g}; median window={settings.smoothing_window_frames} frames; "
          f"prominence>={settings.minimum_prominence_pixels:g}px; min separation={settings.minimum_separation_frames} frames; "
          f"prominence radius={settings.prominence_window_frames} frames.")
    for side in ("left", "right"):
        side_events = [event for event in result.events if event.side == side]
        status_counts: dict[str, int] = {}
        for sample in result.samples:
            status = getattr(sample, f"{side}_status")
            status_counts[status] = status_counts.get(status, 0) + 1
        print(f"{side.title()}: {len(side_events)} candidates; heel sample statuses={status_counts}.")
    print("Candidates are exploratory leg-cycle timing proxies; no foot-strike labels, cadence, step time, or clinical interpretation are produced.")
    print(f"Saved candidate events to {args.output_csv}.")
    print(f"Saved diagnostics to {plot_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
