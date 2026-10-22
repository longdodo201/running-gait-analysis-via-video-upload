"""Exploratory heel-trajectory candidates; these are not verified foot-strikes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from statistics import median
from typing import Any

from .signal_processing import smooth_valid_segments


@dataclass(frozen=True)
class EventDetectionParameters:
    """Configurable baseline settings; values are heuristics, not validated thresholds."""

    confidence_threshold: float = 0.5
    smoothing_window_frames: int = 5
    minimum_prominence_pixels: float = 20.0
    minimum_separation_frames: int = 12
    prominence_window_frames: int = 30


@dataclass(frozen=True)
class HeelSignalSample:
    """One frame of raw and smoothed heel vertical position for each side."""

    frame_index: int | None
    timestamp_seconds: float | None
    left_heel_y_px: float | None
    left_heel_y_smoothed_px: float | None
    left_visibility: float | None
    left_presence: float | None
    left_status: str
    right_heel_y_px: float | None
    right_heel_y_smoothed_px: float | None
    right_visibility: float | None
    right_presence: float | None
    right_status: str


@dataclass(frozen=True)
class CandidateEvent:
    """A local maximum in heel image-y, retained as an unvalidated candidate."""

    side: str
    event_type: str
    frame_index: int
    timestamp_seconds: float | None
    signal_name: str
    signal_value_px: float
    prominence_px: float
    landmark_visibility: float
    landmark_presence: float
    flags: str = "exploratory_unvalidated"


@dataclass(frozen=True)
class EventDetectionResult:
    samples: tuple[HeelSignalSample, ...]
    events: tuple[CandidateEvent, ...]
    parameters: EventDetectionParameters


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _frame_index(value: object) -> int | None:
    number = _finite_number(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _landmark_signal(
    frame: Mapping[str, Any], side: str, image_height: float, confidence_threshold: float
) -> tuple[float | None, float | None, float | None, str]:
    poses = frame.get("poses")
    pose = poses[0] if isinstance(poses, list) and poses else None
    if not isinstance(pose, dict):
        return None, None, None, "no_pose"
    landmarks = pose.get("landmarks")
    landmark = landmarks.get(f"{side}_heel") if isinstance(landmarks, dict) else None
    if not isinstance(landmark, dict):
        return None, None, None, "missing_landmark"

    visibility = _finite_number(landmark.get("visibility"))
    presence = _finite_number(landmark.get("presence"))
    if visibility is None or presence is None:
        return None, visibility, presence, "missing_confidence"
    if visibility < confidence_threshold or presence < confidence_threshold:
        return None, visibility, presence, "low_confidence"
    y = _finite_number(landmark.get("y"))
    if y is None:
        return None, visibility, presence, "invalid_coordinate"
    return y * image_height, visibility, presence, "ok"


def _local_peaks(
    values: Sequence[float | None], start: int, end: int, params: EventDetectionParameters
) -> list[tuple[int, float]]:
    """Return (index, prominence) for local maxima within one valid segment."""
    candidates: list[tuple[int, float]] = []
    index = start + 1
    while index < end - 1:
        plateau_end = index
        while plateau_end + 1 < end and values[plateau_end + 1] == values[index]:
            plateau_end += 1

        previous = values[index - 1]
        current = values[index]
        following = values[plateau_end + 1] if plateau_end + 1 < end else None
        if previous is not None and current is not None and following is not None:
            if current > previous and current > following:
                peak_index = (index + plateau_end) // 2
                window_start = max(start, peak_index - params.prominence_window_frames)
                window_end = min(end, peak_index + params.prominence_window_frames + 1)
                left_values = [v for v in values[window_start:peak_index + 1] if v is not None]
                right_values = [v for v in values[peak_index:window_end] if v is not None]
                prominence = current - max(min(left_values), min(right_values))
                if prominence >= params.minimum_prominence_pixels:
                    candidates.append((peak_index, prominence))
        index = plateau_end + 1
    return candidates


def _side_events(
    side: str,
    values: Sequence[float | None],
    frames: Sequence[Mapping[str, Any]],
    confidences: Sequence[tuple[float | None, float | None]],
    params: EventDetectionParameters,
) -> list[CandidateEvent]:
    all_peaks: list[tuple[int, float]] = []
    index = 0
    while index < len(values):
        if values[index] is None:
            index += 1
            continue
        start = index
        while index < len(values) and values[index] is not None:
            index += 1
        all_peaks.extend(_local_peaks(values, start, index, params))

    # Keep the more prominent event when candidates violate the minimum spacing.
    selected: list[tuple[int, float]] = []
    for peak in sorted(all_peaks, key=lambda item: (-item[1], item[0])):
        if all(abs(peak[0] - kept[0]) >= params.minimum_separation_frames for kept in selected):
            selected.append(peak)

    events: list[CandidateEvent] = []
    for index, prominence in sorted(selected):
        frame = frames[index]
        frame_index = _frame_index(frame.get("frame_index"))
        visibility, presence = confidences[index]
        timestamp = _finite_number(frame.get("timestamp_seconds"))
        if frame_index is None or visibility is None or presence is None:
            continue
        signal_value = values[index]
        if signal_value is None:
            continue
        events.append(CandidateEvent(
            side=side,
            event_type="heel_vertical_low_point_candidate",
            frame_index=frame_index,
            timestamp_seconds=timestamp,
            signal_name=f"{side}_heel_y_px_median{params.smoothing_window_frames}",
            signal_value_px=signal_value,
            prominence_px=prominence,
            landmark_visibility=visibility,
            landmark_presence=presence,
        ))
    return events


def detect_leg_cycle_candidates(
    document: Mapping[str, Any],
    params: EventDetectionParameters | None = None,
) -> EventDetectionResult:
    """Find heel vertical low-point candidates independently for left and right.

    Image y increases downward, so a local maximum in heel y is a vertical
    low-point candidate. It is not labeled as ground contact or foot-strike.
    """
    settings = params or EventDetectionParameters()
    if not math.isfinite(settings.confidence_threshold) or not 0 <= settings.confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    if settings.minimum_prominence_pixels < 0:
        raise ValueError("minimum_prominence_pixels must not be negative.")
    if not math.isfinite(settings.minimum_prominence_pixels):
        raise ValueError("minimum_prominence_pixels must be finite.")
    if settings.minimum_separation_frames < 1:
        raise ValueError("minimum_separation_frames must be positive.")
    if settings.prominence_window_frames < 1:
        raise ValueError("prominence_window_frames must be positive.")
    if settings.smoothing_window_frames < 1 or settings.smoothing_window_frames % 2 == 0:
        raise ValueError("smoothing_window_frames must be a positive odd integer.")

    video = document.get("video")
    height = _finite_number(video.get("height")) if isinstance(video, Mapping) else None
    frames = document.get("frames")
    if height is None or height <= 0:
        raise ValueError("Input JSON must provide a positive video height.")
    if not isinstance(frames, list):
        raise ValueError("Input JSON must contain a frames list.")
    source_frames = [frame if isinstance(frame, Mapping) else {} for frame in frames]

    side_data: dict[
        str,
        tuple[
            list[float | None],
            list[tuple[float | None, float | None]],
            list[str],
            list[float | None],
        ],
    ] = {}
    for side in ("left", "right"):
        raw: list[float | None] = []
        confidence: list[tuple[float | None, float | None]] = []
        statuses: list[str] = []
        for frame in source_frames:
            y, visibility, presence, status = _landmark_signal(
                frame, side, height, settings.confidence_threshold
            )
            raw.append(y)
            confidence.append((visibility, presence))
            statuses.append(status)
        smoothed = smooth_valid_segments(raw, settings.smoothing_window_frames)
        side_data[side] = (raw, confidence, statuses, smoothed)

    samples: list[HeelSignalSample] = []
    events: list[CandidateEvent] = []
    for index, frame in enumerate(source_frames):
        sample_values: dict[str, Any] = {"frame_index": _frame_index(frame.get("frame_index"))}
        sample_values["timestamp_seconds"] = _finite_number(frame.get("timestamp_seconds"))
        for side in ("left", "right"):
            raw, confidence, statuses, smoothed = side_data[side]
            sample_values[f"{side}_heel_y_px"] = raw[index]
            sample_values[f"{side}_heel_y_smoothed_px"] = smoothed[index]
            sample_values[f"{side}_visibility"] = confidence[index][0]
            sample_values[f"{side}_presence"] = confidence[index][1]
            sample_values[f"{side}_status"] = statuses[index]
        samples.append(HeelSignalSample(**sample_values))

    for side in ("left", "right"):
        _, confidence, _, smoothed = side_data[side]
        events.extend(_side_events(side, smoothed, source_frames, confidence, settings))

    return EventDetectionResult(tuple(samples), tuple(events), settings)


def median_candidate_interval_seconds(events: Sequence[CandidateEvent], side: str) -> float | None:
    """Return a descriptive median interval between timestamped same-side candidates."""
    times = sorted(
        event.timestamp_seconds for event in events
        if event.side == side and event.timestamp_seconds is not None
    )
    intervals = [second - first for first, second in zip(times, times[1:]) if second > first]
    return median(intervals) if intervals else None


def validate_phase23_alignment(
    samples: Sequence[HeelSignalSample], knee_rows: Sequence[Mapping[str, Any]]
) -> None:
    """Ensure the contextual Phase 2.3 rows describe the same source frames."""
    if len(samples) != len(knee_rows):
        raise ValueError("Phase 2.3 CSV row count does not match the landmark JSON frame count.")
    for index, (sample, row) in enumerate(zip(samples, knee_rows)):
        csv_frame = _frame_index(row.get("frame_index"))
        if sample.frame_index is None or csv_frame != sample.frame_index:
            raise ValueError(f"Phase 2.3 CSV frame index mismatch at row {index}.")
        csv_time = _finite_number(row.get("timestamp_seconds"))
        if (sample.timestamp_seconds is None) != (csv_time is None):
            raise ValueError(f"Phase 2.3 CSV timestamp mismatch at row {index}.")
        if sample.timestamp_seconds is not None and csv_time is not None:
            if not math.isclose(sample.timestamp_seconds, csv_time, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Phase 2.3 CSV timestamp mismatch at row {index}.")
