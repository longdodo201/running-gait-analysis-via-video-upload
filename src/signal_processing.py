"""Small offline signal-processing helpers for frame-level gait series."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Any


SMOOTHED_ANGLE_COLUMNS = (
    "left_knee_angle_smoothed_deg",
    "right_knee_angle_smoothed_deg",
)


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def smooth_valid_segments(
    values: Sequence[float | None], window_size: int = 5
) -> list[float | None]:
    """Apply a centered rolling median without crossing invalid samples.

    Non-finite or non-numeric values are treated as gaps and remain ``None``.
    Segments shorter than the requested window are returned unchanged because
    they do not provide enough context for the configured smoother. For longer
    segments, edge windows are shortened to the available values in that same
    segment.
    """
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")

    clean_values = [_finite_number(value) for value in values]
    result: list[float | None] = [None] * len(clean_values)
    radius = window_size // 2
    index = 0
    while index < len(clean_values):
        if clean_values[index] is None:
            index += 1
            continue

        segment_start = index
        while index < len(clean_values) and clean_values[index] is not None:
            index += 1
        segment_end = index
        segment_length = segment_end - segment_start

        if segment_length < window_size:
            result[segment_start:segment_end] = clean_values[segment_start:segment_end]
            continue

        for position in range(segment_start, segment_end):
            window_start = max(segment_start, position - radius)
            window_end = min(segment_end, position + radius + 1)
            window = [value for value in clean_values[window_start:window_end] if value is not None]
            result[position] = statistics.median(window)

    return result


def _usable_row_angle(row: Mapping[str, Any], side: str) -> float | None:
    if row.get(f"{side}_knee_status") != "ok":
        return None
    angle = _finite_number(row.get(f"{side}_knee_angle_deg"))
    return angle if angle is not None and 0 <= angle <= 180 else None


def add_smoothed_knee_angles(
    rows: Sequence[Mapping[str, Any]], window_size: int = 5
) -> list[dict[str, Any]]:
    """Copy source rows and append smoothed left/right angles, preserving raw fields."""
    left_values = [_usable_row_angle(row, "left") for row in rows]
    right_values = [_usable_row_angle(row, "right") for row in rows]
    left_smoothed = smooth_valid_segments(left_values, window_size)
    right_smoothed = smooth_valid_segments(right_values, window_size)

    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        new_row = dict(row)
        new_row[SMOOTHED_ANGLE_COLUMNS[0]] = left_smoothed[index]
        new_row[SMOOTHED_ANGLE_COLUMNS[1]] = right_smoothed[index]
        output.append(new_row)
    return output
