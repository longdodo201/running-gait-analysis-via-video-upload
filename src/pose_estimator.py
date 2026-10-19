"""Small MediaPipe Pose Landmarker adapter for local CPU inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .landmarks import pose_from_mediapipe


class PoseEstimationError(RuntimeError):
    """Raised when the local model cannot be initialized or inference fails."""


class MediaPipePoseEstimator:
    """Own a local MediaPipe Landmarker and return named poses for each BGR frame."""

    def __init__(self, model_path: str | Path, fps: float | None, max_poses: int = 1) -> None:
        model = Path(model_path).expanduser()
        if not model.is_file():
            raise PoseEstimationError(f"Pose model asset not found: {model}. Follow docs/PHASE0_SETUP.md.")
        if max_poses < 1:
            raise ValueError("max_poses must be at least 1.")
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - depends on runtime installation
            raise PoseEstimationError("MediaPipe and OpenCV are required; install requirements.txt.") from exc

        self._cv2 = cv2
        self._mp = mp
        self._fps = fps
        self._running_mode = (
            mp.tasks.vision.RunningMode.VIDEO if fps is not None
            else mp.tasks.vision.RunningMode.IMAGE
        )
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model.resolve()),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=self._running_mode,
            num_poses=max_poses,
        )
        try:
            self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:
            raise PoseEstimationError(f"Could not initialize MediaPipe Pose Landmarker: {exc}") from exc
        self._last_timestamp_ms = -1

    def detect(self, bgr_frame: Any, frame_index: int) -> tuple[Any, ...]:
        """Run one frame through the selected MediaPipe mode."""
        try:
            rgb = self._cv2.cvtColor(bgr_frame, self._cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            if self._running_mode == self._mp.tasks.vision.RunningMode.VIDEO:
                timestamp_ms = round(frame_index * 1000.0 / self._fps)
                timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
                self._last_timestamp_ms = timestamp_ms
                result = self._landmarker.detect_for_video(image, timestamp_ms)
            else:
                result = self._landmarker.detect(image)
            return tuple(pose_from_mediapipe(pose) for pose in result.pose_landmarks)
        except Exception as exc:
            if isinstance(exc, PoseEstimationError):
                raise
            raise PoseEstimationError(f"Pose inference failed at frame {frame_index}: {exc}") from exc

    def close(self) -> None:
        """Release MediaPipe native resources."""
        self._landmarker.close()

    def __enter__(self) -> "MediaPipePoseEstimator":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

