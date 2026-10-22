"""Heuristic quality diagnostics for frame-level knee-angle CSV data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .signal_processing import _finite_number


@dataclass(frozen=True)
class AdjacentChangeFlag:
    """An exploratory large change between two consecutive usable frames."""

    from_frame_index: int
    to_frame_index: int
    from_timestamp_seconds: float | None
    to_timestamp_seconds: float | None
    absolute_change_deg: float


@dataclass(frozen=True)
class SideQualityDiagnostics:
    """Coverage and Tukey-rule diagnostics for one knee-angle series."""

    missing_angle_count: int
    usable_angle_count: int
    status_counts: dict[str, int]
    adjacent_change_count: int
    q1_absolute_change_deg: float | None
    q3_absolute_change_deg: float | None
    tukey_cutoff_deg: float | None
    large_change_flags: tuple[AdjacentChangeFlag, ...]


def _angle(row: Mapping[str, Any], side: str) -> float | None:
    angle = _finite_number(row.get(f"{side}_knee_angle_deg"))
    return angle if angle is not None and 0 <= angle <= 180 else None


def _frame_index(row: Mapping[str, Any]) -> int | None:
    value = row.get("frame_index")
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if str(value).strip() == str(parsed) else None


def _timestamp(row: Mapping[str, Any]) -> float | None:
    value = _finite_number(row.get("timestamp_seconds"))
    return value


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Linearly interpolate a percentile from sorted values."""
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def diagnose_knee_angles(
    rows: Sequence[Mapping[str, Any]], side: str
) -> SideQualityDiagnostics:
    """Summarize coverage and flag Tukey outliers among adjacent valid frames.

    Flags are heuristic review cues only. They do not identify erroneous
    measurements and never remove or alter source observations.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'.")

    statuses = Counter(
        str(row.get(f"{side}_knee_status") or "<missing>") for row in rows
    )
    values = [_angle(row, side) for row in rows]
    missing_count = sum(value is None for value in values)
    usable_count = sum(
        value is not None and row.get(f"{side}_knee_status") == "ok"
        for row, value in zip(rows, values)
    )

    adjacent_changes: list[tuple[float, int, int, float | None, float | None]] = []
    for index in range(len(rows) - 1):
        first, second = rows[index], rows[index + 1]
        first_angle = values[index]
        second_angle = values[index + 1]
        first_frame = _frame_index(first)
        second_frame = _frame_index(second)
        if (
            first_angle is None
            or second_angle is None
            or first.get(f"{side}_knee_status") != "ok"
            or second.get(f"{side}_knee_status") != "ok"
            or first_frame is None
            or second_frame != first_frame + 1
        ):
            continue
        adjacent_changes.append((
            abs(second_angle - first_angle),
            first_frame,
            second_frame,
            _timestamp(first),
            _timestamp(second),
        ))

    sorted_changes = sorted(change[0] for change in adjacent_changes)
    if sorted_changes:
        q1 = _percentile(sorted_changes, 0.25)
        q3 = _percentile(sorted_changes, 0.75)
        cutoff = q3 + 1.5 * (q3 - q1)
        flags = tuple(
            AdjacentChangeFlag(frame_a, frame_b, time_a, time_b, change)
            for change, frame_a, frame_b, time_a, time_b in adjacent_changes
            if change > cutoff
        )
    else:
        q1 = q3 = cutoff = None
        flags = ()

    return SideQualityDiagnostics(
        missing_angle_count=missing_count,
        usable_angle_count=usable_count,
        status_counts=dict(sorted(statuses.items())),
        adjacent_change_count=len(adjacent_changes),
        q1_absolute_change_deg=q1,
        q3_absolute_change_deg=q3,
        tukey_cutoff_deg=cutoff,
        large_change_flags=flags,
    )
