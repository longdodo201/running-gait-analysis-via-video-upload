"""CLI for local video pose-landmark extraction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import ExtractionError, NoPoseDetectedError, extract_video  # noqa: E402
from src.pose_estimator import PoseEstimationError  # noqa: E402
from src.video_io import VideoError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract timestamped MediaPipe pose landmarks from a local MP4.")
    parser.add_argument("video", type=Path, help="Path to the input MP4 video.")
    parser.add_argument("--model", type=Path, default=Path("models/pose_landmarker_full.task"))
    parser.add_argument("--output", type=Path, default=Path("outputs/landmarks.json"))
    parser.add_argument("--annotated-video", type=Path, help="Optional path for an MP4 skeleton overlay.")
    parser.add_argument("--max-bytes", type=int, default=100 * 1024 * 1024, help="Maximum input size; use 0 to disable.")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Maximum duration; use 0 to disable.")
    args = parser.parse_args()
    try:
        from src.video_io import VideoLimits
        limits = VideoLimits(
            max_bytes=args.max_bytes or None,
            max_duration_seconds=args.max_seconds or None,
        )
        extract_video(args.video, args.model, args.output, args.annotated_video, limits=limits)
    except (VideoError, PoseEstimationError, NoPoseDetectedError, ExtractionError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

