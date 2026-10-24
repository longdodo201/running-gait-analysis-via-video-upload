"""Experimental cadence estimates from the existing per-side event candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from .gait_events import CandidateEvent


@dataclass(frozen=True)
class CandidateCadenceParameters:
    """Controls candidate de-duplication and when a clip-wide estimate is reported."""

    minimum_event_separation_seconds: float = 0.15
    minimum_event_count: int = 4
    minimum_duration_seconds: float = 3.0
    window_seconds: float = 2.0
    window_step_seconds: float = 0.5


def estimate_candidate_cadence(
    events: Sequence[CandidateEvent],
    *,
    frames: Sequence[Mapping[str, Any]],
    fps: float | None,
    duration_seconds: float | None,
    parameters: CandidateCadenceParameters | None = None,
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Estimate steps/minute from heel candidates, preserving explicit uncertainty.

    The existing detector already suppresses nearby candidates independently for
    each side. This second configurable spacing rule removes duplicate left/right
    candidates that land almost simultaneously. These remain exploratory events,
    not verified ground contacts.
    """
    settings = parameters or CandidateCadenceParameters()
    numeric_parameters = (
        settings.minimum_event_separation_seconds,
        settings.minimum_duration_seconds,
        settings.window_seconds,
        settings.window_step_seconds,
    )
    if any(not math.isfinite(value) or value <= 0 for value in numeric_parameters):
        raise ValueError("Cadence timing parameters must be finite and positive.")
    if settings.minimum_event_count < 1:
        raise ValueError("minimum_event_count must be positive.")

    actual_fps = fps if fps is not None and math.isfinite(fps) and fps > 0 else None
    duration = (
        duration_seconds
        if duration_seconds is not None and math.isfinite(duration_seconds) and duration_seconds > 0
        else len(frames) / actual_fps if actual_fps is not None and frames
        else None
    )

    timestamps_by_frame: dict[int, float] = {}
    for index, frame in enumerate(frames):
        frame_index = frame.get("frame_index", index)
        try:
            frame_index = int(frame_index)
        except (TypeError, ValueError, OverflowError):
            continue
        timestamp = frame.get("timestamp_seconds")
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError, OverflowError):
            timestamp = frame_index / actual_fps if actual_fps is not None else None
        if timestamp is not None and math.isfinite(timestamp) and timestamp >= 0:
            timestamps_by_frame[frame_index] = timestamp

    candidates: dict[tuple[str, int], tuple[float, float]] = {}
    for event in events:
        if event.side not in {"left", "right"} or event.frame_index < 0:
            continue
        timestamp = event.timestamp_seconds
        if timestamp is None or not math.isfinite(timestamp):
            timestamp = timestamps_by_frame.get(event.frame_index)
        if timestamp is None or timestamp < 0 or (duration is not None and timestamp > duration):
            continue
        try:
            prominence = float(event.prominence_px)
        except (TypeError, ValueError, OverflowError):
            prominence = 0.0
        if not math.isfinite(prominence):
            prominence = 0.0
        key = (event.side, event.frame_index)
        if key not in candidates or prominence > candidates[key][1]:
            candidates[key] = (timestamp, prominence)

    # Prefer the clearest candidate when bilateral detections are nearly
    # simultaneous. Sorting first makes the selection deterministic.
    ordered = sorted(
        ((side, frame, timestamp, prominence)
         for (side, frame), (timestamp, prominence) in candidates.items()),
        key=lambda item: (-item[3], item[2], item[1], item[0]),
    )
    retained: list[tuple[str, int, float, float]] = []
    for candidate in ordered:
        if all(
            abs(candidate[2] - kept[2]) >= settings.minimum_event_separation_seconds
            for kept in retained
        ):
            retained.append(candidate)
    retained.sort(key=lambda item: (item[2], item[1], item[0]))

    status = "ok"
    warning = "Exploratory heel-trajectory candidates are not confirmed foot-strikes."
    if duration is None:
        status = "timing_unavailable"
        warning = "Cadence is unavailable because valid video FPS/duration metadata is missing."
    elif len(retained) < settings.minimum_event_count:
        status = "insufficient_events"
        warning = (
            f"Cadence not reported: {len(retained)} retained events; at least "
            f"{settings.minimum_event_count} are required."
        )
    elif duration < settings.minimum_duration_seconds:
        status = "insufficient_duration"
        warning = (
            f"Cadence not reported: clip duration {duration:.3f}s is below the "
            f"{settings.minimum_duration_seconds:.3f}s minimum."
        )

    cadence = len(retained) / duration * 60.0 if status == "ok" and duration else None
    series: list[dict[str, float | int]] = []
    if duration is not None and status == "ok":
        start = 0.0
        while start + settings.window_seconds <= duration + 1e-9:
            end = start + settings.window_seconds
            count = sum(start <= event[2] < end for event in retained)
            series.append({
                "window_start_seconds": start,
                "window_end_seconds": end,
                "window_center_seconds": start + settings.window_seconds / 2,
                "step_count": count,
                "cadence_steps_per_minute": count / settings.window_seconds * 60.0,
            })
            start += settings.window_step_seconds

    candidate_counts = {
        side: sum(event.side == side for event in events)
        for side in ("left", "right")
    }
    retained_counts = {
        side: sum(event[0] == side for event in retained)
        for side in ("left", "right")
    }
    summary = {
        "status": status,
        "event_source": "exploratory_heel_trajectory_candidates",
        "average_cadence_steps_per_minute": cadence,
        "candidate_event_count": len(events),
        "retained_event_count": len(retained),
        "candidate_event_count_by_side": candidate_counts,
        "retained_event_count_by_side": retained_counts,
        "duration_seconds": duration,
        "fps": actual_fps,
        "average_method": "retained candidate events / decoded video duration * 60",
        "minimum_event_separation_seconds": settings.minimum_event_separation_seconds,
        "minimum_event_count": settings.minimum_event_count,
        "minimum_duration_seconds": settings.minimum_duration_seconds,
        "window_seconds": settings.window_seconds,
        "window_step_seconds": settings.window_step_seconds,
        "series_window_count": len(series),
        "warning": warning,
        "parameters": asdict(settings),
    }
    return series, summary
