"""Frame-level 2D knee-angle estimates from Phase 1 landmark JSON."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .landmarks import angle_2d_degrees


@dataclass(frozen=True)
class KneeAngleRow:
    """Left and right knee-angle results for one source frame."""

    frame_index: Any
    timestamp_seconds: Any
    left_knee_angle_deg: float | None
    right_knee_angle_deg: float | None
    left_knee_status: str
    right_knee_status: str


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _knee_angle(
    pose: object,
    side: str,
    width: float | None,
    height: float | None,
    visibility_threshold: float,
    presence_threshold: float,
) -> tuple[float | None, str]:
    if pose is None:
        return None, "no_pose"
    if not isinstance(pose, dict):
        return None, "missing_landmark"
    landmarks = pose.get("landmarks")
    if not isinstance(landmarks, dict):
        return None, "missing_landmark"

    points = []
    for name in (f"{side}_hip", f"{side}_knee", f"{side}_ankle"):
        landmark = landmarks.get(name)
        if not isinstance(landmark, dict):
            return None, "missing_landmark"
        x = _number(landmark.get("x"))
        y = _number(landmark.get("y"))
        if x is None or y is None:
            return None, "invalid_coordinate"
        points.append((x, y, landmark))

    # Confidence is checked for all three landmarks; missing confidence is not
    # silently treated as a high-confidence observation.
    for _, _, landmark in points:
        if landmark.get("visibility") is None or landmark.get("presence") is None:
            return None, "missing_confidence"
        visibility = _number(landmark.get("visibility"))
        presence = _number(landmark.get("presence"))
        if visibility is None or presence is None:
            return None, "invalid_confidence"
        if visibility < visibility_threshold or presence < presence_threshold:
            return None, "low_confidence"

    if width is None or height is None or width <= 0 or height <= 0:
        return None, "invalid_coordinate"
    pixel_points = [(x * width, y * height) for x, y, _ in points]
    try:
        angle = angle_2d_degrees(pixel_points[0], pixel_points[1], pixel_points[2])
    except ValueError:
        return None, "degenerate_geometry"
    return angle, "ok"


def knee_angle_rows(
    document: dict[str, Any],
    visibility_threshold: float = 0.5,
    presence_threshold: float = 0.5,
) -> list[KneeAngleRow]:
    """Calculate both knees for pose 0, retaining one result for every frame."""
    for name, threshold in (
        ("visibility_threshold", visibility_threshold),
        ("presence_threshold", presence_threshold),
    ):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    video = document.get("video")
    video = video if isinstance(video, dict) else {}
    width = _number(video.get("width"))
    height = _number(video.get("height"))
    frames = document.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Input JSON must contain a frames list.")

    rows: list[KneeAngleRow] = []
    for frame in frames:
        frame = frame if isinstance(frame, dict) else {}
        poses = frame.get("poses")
        pose = poses[0] if isinstance(poses, list) and poses else None
        left_angle, left_status = _knee_angle(
            pose, "left", width, height, visibility_threshold, presence_threshold
        )
        right_angle, right_status = _knee_angle(
            pose, "right", width, height, visibility_threshold, presence_threshold
        )
        rows.append(KneeAngleRow(
            frame_index=frame.get("frame_index"),
            timestamp_seconds=frame.get("timestamp_seconds"),
            left_knee_angle_deg=left_angle,
            right_knee_angle_deg=right_angle,
            left_knee_status=left_status,
            right_knee_status=right_status,
        ))
    return rows
