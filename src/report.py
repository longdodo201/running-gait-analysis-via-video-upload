"""Writers for analysis result artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .gait_metrics import KneeAngleRow
from .gait_events import CandidateEvent
from .signal_processing import SMOOTHED_ANGLE_COLUMNS


KNEE_ANGLE_COLUMNS = (
    "frame_index",
    "timestamp_seconds",
    "left_knee_angle_deg",
    "right_knee_angle_deg",
    "left_knee_status",
    "right_knee_status",
)


def write_knee_angles_csv(rows: Iterable[KneeAngleRow], destination: str | Path) -> Path:
    """Write frame-level knee angles and statuses, leaving unavailable angles blank."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=KNEE_ANGLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: getattr(row, column) for column in KNEE_ANGLE_COLUMNS})
    return output


def write_smoothed_knee_angles_csv(
    rows: Iterable[dict[str, object]],
    original_columns: Iterable[str],
    destination: str | Path,
) -> Path:
    """Write Phase 2.3 rows with every source column plus smoothed-angle columns."""
    columns = list(original_columns)
    required = set(KNEE_ANGLE_COLUMNS)
    if not required.issubset(columns):
        missing = ", ".join(sorted(required.difference(columns)))
        raise ValueError(f"Input CSV is missing required Phase 2.2 columns: {missing}.")
    output_columns = columns + [column for column in SMOOTHED_ANGLE_COLUMNS if column not in columns]
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in output_columns})
    return output


EVENT_COLUMNS = (
    "side",
    "event_type",
    "frame_index",
    "timestamp_seconds",
    "signal_name",
    "signal_value_px",
    "prominence_px",
    "landmark_visibility",
    "landmark_presence",
    "flags",
)


def write_candidate_events_csv(rows: Iterable[CandidateEvent], destination: str | Path) -> Path:
    """Write candidate timestamps and signal features without implying validation."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: getattr(row, column) for column in EVENT_COLUMNS})
    return output
