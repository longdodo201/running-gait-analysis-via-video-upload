"""Video-to-landmark extraction orchestration; independent of any web UI."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable
import math

from .landmarks import ExtractionDocument, LandmarkFrame, Pose, write_json
from .video_io import VideoError, VideoLimits, iter_video_frames, validate_video


class NoPoseDetectedError(RuntimeError):
    """Raised when all readable video frames produce no detected person."""


class ExtractionError(RuntimeError):
    """Raised for model, decoder, serialization, or artifact failures."""


def timestamp_for_frame(frame_index: int, fps: float | None) -> float | None:
    """Calculate a source timestamp only when FPS is a valid positive finite value."""
    if fps is None or not math.isfinite(fps) or fps <= 0:
        return None
    return frame_index / fps


def extract_video(
    video_path: str | Path,
    model_path: str | Path,
    output_json: str | Path,
    annotated_video: str | Path | None = None,
    limits: VideoLimits | None = None,
    progress: Callable[[str], None] = print,
    estimator_factory: Callable[[str | Path, float | None], object] | None = None,
    capture_factory: Callable[[str], object] | None = None,
    annotation_factory: Callable[[str | Path, float, int, int], object] | None = None,
) -> ExtractionDocument:
    """Validate a local MP4, extract poses frame-by-frame, and save JSON plus optional overlay."""
    configured_limits = limits or VideoLimits()
    source, metadata, capture = validate_video(video_path, limits=configured_limits, capture_factory=capture_factory)
    output_path = Path(output_json)
    if output_path.suffix.lower() != ".json":
        capture.release()
        raise VideoError("Landmark output must use the .json extension.")
    if output_path.resolve() == source.resolve():
        capture.release()
        raise VideoError("Landmark JSON output cannot overwrite the source video.")
    if annotated_video is not None and Path(annotated_video).resolve() in {source.resolve(), output_path.resolve()}:
        capture.release()
        raise VideoError("Annotated video output must differ from the source and JSON output paths.")

    if estimator_factory is None:
        from .pose_estimator import MediaPipePoseEstimator
        estimator_factory = MediaPipePoseEstimator
    if annotation_factory is None and annotated_video is not None:
        from .annotation import AnnotatedVideoWriter
        annotation_factory = AnnotatedVideoWriter

    estimator = None
    writer = None
    writer_success = False
    frames: list[LandmarkFrame] = []
    pose_count = 0
    decoded_count = 0
    warnings = list(metadata.warnings)
    writer_fps = metadata.fps or 30.0
    if metadata.fps is None and annotated_video is not None:
        warnings.append("Annotated video uses 30 FPS because source FPS is unavailable; source timestamps remain null.")
    try:
        try:
            estimator = estimator_factory(model_path, metadata.fps)
            if annotated_video is not None:
                writer = annotation_factory(annotated_video, writer_fps, metadata.width, metadata.height)
            progress(f"Processing {source.name} ({metadata.width}x{metadata.height}, FPS={metadata.fps or 'unknown'}).")
            for frame_index, frame in iter_video_frames(capture):
                decoded_count += 1
                if getattr(frame, "shape", None) is not None:
                    actual_height, actual_width = frame.shape[:2]
                    if (actual_width, actual_height) != (metadata.width, metadata.height):
                        raise VideoError(
                            f"Decoded frame {frame_index} is {actual_width}x{actual_height}, "
                            f"but metadata reports {metadata.width}x{metadata.height}."
                        )
                poses = tuple(estimator.detect(frame, frame_index))
                pose_count += len(poses)
                frames.append(LandmarkFrame(frame_index, timestamp_for_frame(frame_index, metadata.fps), poses))
                if writer is not None:
                    writer.write(frame, poses)
                progress(f"Processed frame {frame_index + 1}; detected poses: {len(poses)}.")
            actual_duration = decoded_count / metadata.fps if metadata.fps is not None else None
            if (
                configured_limits.max_duration_seconds is not None
                and actual_duration is not None
                and actual_duration > configured_limits.max_duration_seconds
            ):
                raise VideoError(
                    f"Decoded video duration {actual_duration:.3f}s exceeds the configured limit of "
                    f"{configured_limits.max_duration_seconds:.3f}s."
                )
            if pose_count == 0:
                raise NoPoseDetectedError(
                    "No person was detected in any decoded frame. Check visibility, lighting, framing, and occlusion."
                )
            document = ExtractionDocument(
                schema_version="1.0",
                source_video=source.name,
                video=asdict(metadata),
                timestamp_basis="frame_index / source_fps" if metadata.fps is not None else "unavailable",
                warnings=tuple(warnings),
                frames=tuple(frames),
            )
            write_json(document, output_path)
            writer_success = True
            progress(f"Saved {decoded_count} frames and {pose_count} detected poses to {output_path}.")
            return document
        except (VideoError, NoPoseDetectedError, ExtractionError):
            raise
        except Exception as exc:
            raise ExtractionError(f"Video processing failed after {decoded_count} decoded frames: {exc}") from exc
    finally:
        # iter_video_frames owns/release capture after iteration starts; release is idempotent in OpenCV.
        capture.release()
        if estimator is not None:
            try:
                estimator.close()
            except Exception as exc:
                if writer is not None:
                    writer.close(success=False)
                if not writer_success:
                    raise ExtractionError(f"Failed to close pose estimator cleanly: {exc}") from exc
                raise ExtractionError(f"Pose estimator cleanup failed after output creation: {exc}") from exc
        if writer is not None:
            writer.close(success=writer_success)

