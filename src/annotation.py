"""Optional OpenCV skeleton overlay rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .landmarks import Pose


# MediaPipe Pose topology for the principal visible-body connections.
POSE_CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31), (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


class AnnotatedVideoWriter:
    """Write a temporary MP4 overlay and atomically publish it on successful close."""

    def __init__(self, destination: str | Path, fps: float, width: int, height: int) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on runtime installation
            raise RuntimeError("OpenCV is required to write annotated video.") from exc
        self._cv2 = cv2
        self.destination = Path(destination)
        if self.destination.suffix.lower() != ".mp4":
            raise ValueError("Annotated video output must use the .mp4 extension.")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.destination.with_name(self.destination.stem + ".partial" + self.destination.suffix)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.temporary), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            self._writer.release()
            self.temporary.unlink(missing_ok=True)
            raise RuntimeError("OpenCV could not initialize the MP4 writer (mp4v codec).")
        self._finished = False

    def write(self, frame: Any, poses: tuple[Pose, ...]) -> None:
        """Draw detected pose landmarks and connections, then append the frame."""
        height, width = frame.shape[:2]
        for pose in poses:
            points = pose.landmarks
            pixels = []
            for landmark in points.values():
                x = int(round(landmark.x * width))
                y = int(round(landmark.y * height))
                pixels.append((x, y))
                self._cv2.circle(frame, (x, y), 3, (40, 220, 40), -1)
            for start, end in POSE_CONNECTIONS:
                self._cv2.line(frame, pixels[start], pixels[end], (40, 180, 255), 2)
        self._writer.write(frame)

    def close(self, success: bool) -> None:
        """Release the encoder and publish only a completed output."""
        if self._finished:
            return
        self._finished = True
        self._writer.release()
        if success and self.temporary.is_file() and self.temporary.stat().st_size > 0:
            self.temporary.replace(self.destination)
        elif success:
            self.temporary.unlink(missing_ok=True)
            raise RuntimeError("Annotated MP4 was not produced by the video encoder.")
        else:
            self.temporary.unlink(missing_ok=True)

