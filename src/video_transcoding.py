"""Optional local conversion of an existing overlay to browser-friendly H.264."""

from __future__ import annotations

import math
from pathlib import Path
import uuid


class VideoTranscodeError(RuntimeError):
    """Raised when an overlay cannot be safely converted to H.264."""


def transcode_overlay_to_h264(source_path: str | Path) -> Path:
    """Convert an existing MP4 overlay without modifying it or running pose inference.

    The H.264 file is published only after OpenCV can reopen it, reports an H.264
    codec tag, and decodes a frame. A failed conversion removes only this call's
    temporary file and leaves the source untouched.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on runtime installation
        raise VideoTranscodeError("OpenCV is required for local H.264 conversion.") from exc

    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise VideoTranscodeError(f"Overlay video does not exist: {source}")
    if source.suffix.lower() != ".mp4":
        raise VideoTranscodeError("Overlay video must use the .mp4 extension.")

    destination = source.with_name(f"{source.stem}_h264.mp4")
    if destination.exists():
        raise VideoTranscodeError(f"Refusing to overwrite existing H.264 video: {destination}")

    # Keep a plain .mp4 filename: Windows Media Foundation may reject hidden or
    # multi-suffix temporary paths even when the fallback writer supports H.264.
    temporary = source.with_name(f"h264_tmp_{uuid.uuid4().hex}.mp4")
    capture = None
    writer = None
    try:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise VideoTranscodeError(f"Could not open overlay video for conversion: {source}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise VideoTranscodeError("Overlay video has invalid FPS or frame dimensions.")

        writer = cv2.VideoWriter(
            str(temporary),
            cv2.VideoWriter_fourcc(*"avc1"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise VideoTranscodeError("OpenCV could not initialize an H.264 (avc1) encoder.")

        frames_written = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None or frame.shape[:2] != (height, width):
                raise VideoTranscodeError("Overlay contains an invalid or inconsistent video frame.")
            writer.write(frame)
            frames_written += 1

        writer.release()
        writer = None
        capture.release()
        capture = None
        if frames_written == 0:
            raise VideoTranscodeError("Overlay video contains no decodable frames.")
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise VideoTranscodeError("H.264 encoder did not produce a video file.")

        verification = cv2.VideoCapture(str(temporary))
        try:
            if not verification.isOpened():
                raise VideoTranscodeError("Could not reopen the converted H.264 video.")
            fourcc_value = int(verification.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4)).lower()
            decoded, _frame = verification.read()
            if fourcc not in {"h264", "avc1"} or not decoded:
                raise VideoTranscodeError(
                    f"Converted video failed H.264 verification (codec={fourcc or 'unknown'}, decoded={decoded})."
                )
        finally:
            verification.release()

        if destination.exists():
            raise VideoTranscodeError(f"Refusing to overwrite existing H.264 video: {destination}")
        temporary.replace(destination)
        return destination
    except VideoTranscodeError:
        raise
    except Exception as exc:
        raise VideoTranscodeError(f"Local H.264 conversion failed: {exc}") from exc
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        temporary.unlink(missing_ok=True)
