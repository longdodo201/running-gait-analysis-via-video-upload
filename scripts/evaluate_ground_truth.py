"""Evaluate saved foot-strike landmarks and metrics without pose inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.foot_strike_analysis import (
    candidates_to_predictions,
    detect_contact_zone_candidates,
    load_ground_truth,
    validate_ground_truth_source_metadata,
    write_ground_truth_analysis,
)
from src.gait_events import detect_leg_cycle_candidates
from src.gait_metrics import knee_angle_rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("landmarks_json", type=Path)
    parser.add_argument("ground_truth_csv", type=Path)
    parser.add_argument("output_dir", type=Path, help="New directory; existing output is refused")
    parser.add_argument("--tolerance-frames", type=int, default=2)
    parser.add_argument("--source-video", type=Path, help="Optional video file for source-fingerprint verification")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        print(f"ERROR: Refusing to overwrite existing output directory: {output}", file=sys.stderr)
        return 2
    try:
        document = json.loads(args.landmarks_json.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("Landmark JSON root must be an object.")
        video = document.get("video", {})
        fps = float(video.get("fps", 0)) if isinstance(video, dict) else 0.0
        validate_ground_truth_source_metadata(
            args.ground_truth_csv, document, source_video_path=args.source_video
        )
        truth = load_ground_truth(args.ground_truth_csv, document)
        candidates = detect_leg_cycle_candidates(document)
        predictions = {
            "A_heel_local_extrema": candidates_to_predictions(candidates.events),
            "B_foot_contact_zone": detect_contact_zone_candidates(document),
        }
        artifacts = write_ground_truth_analysis(
            output,
            ground_truth=truth,
            algorithms=predictions,
            fps=fps,
            document=document,
            knee_rows=knee_angle_rows(document),
            tolerance_frames=args.tolerance_frames,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Source: {document.get('source_video')}")
    print(f"Ground Truth: {args.ground_truth_csv.resolve()} ({len(truth)} events)")
    print(f"FPS: {fps:g}; frames: {len(document.get('frames', []))}; tolerance: plus/minus {args.tolerance_frames} frames")
    print(f"Output: {output}")
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
