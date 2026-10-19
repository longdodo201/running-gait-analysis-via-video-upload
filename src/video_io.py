"""Video validation and sequential frame access using OpenCV."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
import math


class VideoError(ValueError):
    """Raised when an input video is missing, unsupported, or unreadable."""


@dataclass(frozen=True)
class VideoLimits:
    """Configurable application limits; these are not MediaPipe limits."""

    max_bytes: int | None = 100 * 1024 * 1024
    max_duration_seconds: float | None = 60.0

    def __post_init__(self) -> None:
        if self.max_bytes is not None and self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive or None.")
        if self.max_duration_seconds is not None and (
            not math.isfinite(self.max_duration_seconds) or self.max_duration_seconds <= 0
        ):
            raise ValueError("max_duration_seconds must be positive and finite or None.")


@dataclass(frozen=True)
class VideoMetadata:
    """Video properties reported by the decoder. FPS or duration may be unknown."""

    width: int
    height: int
    fps: float | None
    frame_count: int | None
    duration_seconds: float | None
    warnings: tuple[str, ...] = ()


def _valid_fps(value: float) -> float | None:
    return value if math.isfinite(value) and value > 0 else None


def validate_video(
    path: str | Path,
    limits: VideoLimits | None = None,
    capture_factory: Callable[[str], object] | None = None,
) -> tuple[Path, VideoMetadata, object]:
    """Open an MP4, validate its metadata, and return its capture for sequential reads.

    The caller owns the returned capture and must release it, preferably with
    :func:`iter_video_frames` which guarantees release on completion or error.
    ``capture_factory`` exists to make validation testable without OpenCV.
    """
    video_path = Path(path).expanduser()
    if not video_path.is_file():
        raise VideoError(f"Video file does not exist or is not a regular file: {video_path}")
    if video_path.suffix.lower() != ".mp4":
        raise VideoError(f"Unsupported video format '{video_path.suffix or '(no extension)'}'; use MP4.")

    configured_limits = limits or VideoLimits()
    size_bytes = video_path.stat().st_size
    if configured_limits.max_bytes is not None and size_bytes > configured_limits.max_bytes:
        raise VideoError(
            f"Video is {size_bytes} bytes, above the configured limit of "
            f"{configured_limits.max_bytes} bytes."
        )
    if size_bytes == 0:
        raise VideoError("Video file is empty.")

    if capture_factory is None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on runtime installation
            raise VideoError("OpenCV is unavailable. Install dependencies from requirements.txt.") from exc
        capture_factory = cv2.VideoCapture

    capture = capture_factory(str(video_path))
    try:
        if not capture.isOpened():
            raise VideoError(f"OpenCV cannot open this video; check that it is a valid MP4 with a supported codec: {video_path}")

        # OpenCV CAP_PROP_* property identifiers are stable numeric API values.
        width = int(capture.get(3))
        height = int(capture.get(4))
        fps = _valid_fps(float(capture.get(5)))
        raw_count = float(capture.get(7))
        frame_count = int(raw_count) if math.isfinite(raw_count) and raw_count > 0 else None
        if width <= 0 or height <= 0:
            raise VideoError(f"Video reports invalid frame dimensions: {width}x{height}.")

        warnings: list[str] = []
        duration = frame_count / fps if frame_count is not None and fps is not None else None
        if fps is None:
            warnings.append("Source FPS is missing or invalid; timestamps will be null.")
        if duration is None:
            warnings.append("Video duration could not be determined from decoder metadata.")
        elif configured_limits.max_duration_seconds is not None and duration > configured_limits.max_duration_seconds:
            raise VideoError(
                f"Video duration {duration:.3f}s exceeds the configured limit of "
                f"{configured_limits.max_duration_seconds:.3f}s."
            )

        first_ok, first_frame = capture.read()
        if not first_ok or first_frame is None:
            raise VideoError("Video opened, but its first frame could not be decoded.")
        if not capture.set(1, 0):  # CAP_PROP_POS_FRAMES
            raise VideoError("Decoder could not rewind after the readability check.")
        metadata = VideoMetadata(width, height, fps, frame_count, duration, tuple(warnings))
        return video_path, metadata, capture
    except Exception:
        capture.release()
        raise


def iter_video_frames(capture: object) -> Iterator[tuple[int, object]]:
    """Yield decoded (zero-based frame index, BGR frame) pairs and always release capture."""
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None:
                raise VideoError(f"Decoder returned an empty frame at index {index}.")
            yield index, frame
            index += 1
        if index == 0:
            raise VideoError("Video contains no decodable frames.")
    finally:
        capture.release()

