"""Named landmark schema and JSON serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any


LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)


@dataclass(frozen=True)
class Landmark:
    """MediaPipe normalized image coordinates and optional model confidence fields."""

    x: float
    y: float
    z: float
    visibility: float | None = None
    presence: float | None = None


@dataclass(frozen=True)
class Pose:
    """One detected person's 33 named pose landmarks."""

    landmarks: dict[str, Landmark]


@dataclass(frozen=True)
class LandmarkFrame:
    """One decoded video frame and zero or more detected poses."""

    frame_index: int
    timestamp_seconds: float | None
    poses: tuple[Pose, ...]


@dataclass(frozen=True)
class ExtractionDocument:
    """Versioned JSON output containing source metadata and all decoded frames."""

    schema_version: str
    source_video: str
    video: dict[str, Any]
    timestamp_basis: str
    warnings: tuple[str, ...]
    frames: tuple[LandmarkFrame, ...] = field(default_factory=tuple)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def pose_from_mediapipe(landmarks: list[object]) -> Pose:
    """Convert MediaPipe normalized landmarks to the stable named output schema."""
    if len(landmarks) != len(LANDMARK_NAMES):
        raise ValueError(f"Expected {len(LANDMARK_NAMES)} pose landmarks; received {len(landmarks)}.")
    named: dict[str, Landmark] = {}
    for name, point in zip(LANDMARK_NAMES, landmarks):
        named[name] = Landmark(
            x=float(point.x), y=float(point.y), z=float(point.z),
            visibility=_optional_float(getattr(point, "visibility", None)),
            presence=_optional_float(getattr(point, "presence", None)),
        )
    return Pose(named)


def angle_2d_degrees(point_a: tuple[float, float], vertex: tuple[float, float], point_c: tuple[float, float]) -> float:
    """Return the 2D interior angle at ``vertex`` in degrees.

    Coordinates must already use the same scale (for normalized image points,
    convert x/y to pixels before calling). Invalid and degenerate geometry is
    rejected with ``ValueError``.
    """
    points = (point_a, vertex, point_c)
    if any(not isinstance(point, (tuple, list)) or len(point) != 2 for point in points):
        raise ValueError("Angle coordinates must be numeric pairs.")
    values = tuple(value for point in points for value in point)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("Angle coordinates must be numeric pairs.")
    coordinates = tuple(float(value) for value in values)
    if len(coordinates) != 6 or not all(math.isfinite(value) for value in coordinates):
        raise ValueError("Angle coordinates must contain six finite numbers.")

    ax, ay, vx, vy, cx, cy = coordinates
    first = (ax - vx, ay - vy)
    second = (cx - vx, cy - vy)
    first_length = math.hypot(*first)
    second_length = math.hypot(*second)
    if first_length == 0 or second_length == 0:
        raise ValueError("Angle is undefined when either segment has zero length.")

    cosine = (first[0] * second[0] + first[1] * second[1]) / (first_length * second_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def document_to_dict(document: ExtractionDocument) -> dict[str, Any]:
    """Return JSON-safe primitive values, preserving frame and landmark names."""
    return asdict(document)


def write_json(document: ExtractionDocument, destination: str | Path) -> Path:
    """Write JSON atomically so failed serialization does not leave a partial result."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(document_to_dict(document), stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return output

