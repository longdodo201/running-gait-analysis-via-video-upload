"""Manual review templates and contact sheets for saved event candidates."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REVIEW_COLUMNS = (
    "event_id",
    "side",
    "frame_index",
    "timestamp_seconds",
    "candidate_signal_name",
    "candidate_signal_value_px",
    "manual_label",
    "reviewer_notes",
)
ALLOWED_MANUAL_LABELS = frozenset({"confirmed", "rejected", "uncertain"})


def validate_manual_label(value: str) -> str:
    """Return a supported label, allowing blank values for unreviewed rows."""
    normalized = value.strip().lower()
    if normalized and normalized not in ALLOWED_MANUAL_LABELS:
        raise ValueError("Manual label must be blank, confirmed, rejected, or uncertain.")
    return normalized


def build_review_rows(events: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Create one blank-label manual-review row for each event in input order."""
    rows: list[dict[str, str]] = []
    for index, event in enumerate(events, start=1):
        side = event.get("side", "")
        if side not in {"left", "right"}:
            raise ValueError(f"Candidate row {index} has an invalid side.")
        try:
            frame_index = int(event["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Candidate row {index} has an invalid frame_index.") from exc
        if frame_index < 0:
            raise ValueError(f"Candidate row {index} has a negative frame_index.")
        rows.append({
            "event_id": f"E{index:03d}",
            "side": side,
            "frame_index": str(frame_index),
            "timestamp_seconds": event.get("timestamp_seconds", ""),
            "candidate_signal_name": event.get("signal_name", ""),
            "candidate_signal_value_px": event.get("signal_value_px", ""),
            "manual_label": "",
            "reviewer_notes": "",
        })
    return rows


def write_review_csv(rows: Iterable[Mapping[str, str]], destination: str | Path) -> Path:
    """Write an editable review template; labels are deliberately left empty."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            label = validate_manual_label(row.get("manual_label", ""))
            writer.writerow({column: (label if column == "manual_label" else row.get(column, "")) for column in REVIEW_COLUMNS})
    return output


def validate_review_csv(path: str | Path) -> int:
    """Validate manually entered labels and return the number of reviewed rows."""
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(REVIEW_COLUMNS).issubset(reader.fieldnames or ()):
            raise ValueError("Review CSV is missing required columns.")
        rows = list(reader)
    reviewed = 0
    for row_number, row in enumerate(rows, start=2):
        label = validate_manual_label(row.get("manual_label", ""))
        if label:
            reviewed += 1
        if not row.get("event_id") or not row.get("frame_index"):
            raise ValueError(f"Review CSV row {row_number} is missing its event ID or frame index.")
    return reviewed


def _confidence_number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def select_review_targets(
    review_rows: Sequence[Mapping[str, str]],
    candidate_events: Sequence[Mapping[str, str]],
    confidence_threshold: float = 0.5,
) -> list[dict[str, str]]:
    """Select uncertain labels or candidates below the detector confidence cutoff.

    Missing or invalid candidate confidence is considered low confidence. The
    function copies rows and never changes the source review labels.
    """
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    by_event: dict[tuple[str, str], Mapping[str, str]] = {}
    for event in candidate_events:
        key = (event.get("side", ""), event.get("frame_index", ""))
        if key in by_event:
            raise ValueError(f"Duplicate candidate event for side/frame {key}.")
        by_event[key] = event

    selected: list[dict[str, str]] = []
    for row in review_rows:
        label = validate_manual_label(row.get("manual_label", ""))
        side, frame = row.get("side", ""), row.get("frame_index", "")
        event = by_event.get((side, frame))
        if event is None:
            raise ValueError(f"No Phase 2.4 candidate matches {side} frame {frame}.")
        visibility = _confidence_number(event.get("landmark_visibility"))
        presence = _confidence_number(event.get("landmark_presence"))
        low_confidence = (
            visibility is None or presence is None
            or visibility < confidence_threshold or presence < confidence_threshold
        )
        if label == "uncertain" or low_confidence:
            target = dict(row)
            reasons = []
            if label == "uncertain":
                reasons.append("manual_uncertain")
            if low_confidence:
                reasons.append("low_landmark_confidence")
            target["review_reason"] = ";".join(reasons)
            target["landmark_visibility"] = event.get("landmark_visibility", "")
            target["landmark_presence"] = event.get("landmark_presence", "")
            selected.append(target)
    return selected


def review_frame_indices(
    frame_index: int,
    timestamp_seconds: float | None,
    fps: float,
    offset_seconds: float = 0.2,
    *,
    offset_frames: int | None = None,
) -> tuple[int, int, int]:
    """Return before, candidate, and after indices using seconds or exact frames.

    ``offset_frames`` selects an exact frame distance and takes precedence over
    ``offset_seconds``. The returned context indices are clamped to zero here;
    the caller clamps the after index to the video frame count.
    """
    if frame_index < 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("frame_index must be non-negative and fps must be positive and finite.")
    if offset_frames is not None:
        if type(offset_frames) is not int or offset_frames <= 0:
            raise ValueError("offset_frames must be a positive integer.")
        if timestamp_seconds is not None and (not math.isfinite(timestamp_seconds) or timestamp_seconds < 0):
            raise ValueError("timestamp_seconds must be non-negative and finite when provided.")
        return max(0, frame_index - offset_frames), frame_index, frame_index + offset_frames
    if not math.isfinite(offset_seconds) or offset_seconds <= 0:
        raise ValueError("offset_seconds must be positive and finite.")
    event_time = timestamp_seconds if timestamp_seconds is not None else frame_index / fps
    if not math.isfinite(event_time) or event_time < 0:
        raise ValueError("timestamp_seconds must be non-negative and finite when provided.")
    before = max(0, round((event_time - offset_seconds) * fps))
    after = round((event_time + offset_seconds) * fps)
    return before, frame_index, after


def clamp_review_frame_indices(
    indices: tuple[int, int, int], frame_count: int
) -> tuple[int, int, int]:
    """Clamp context frames to clip bounds while preserving the candidate index."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive.")
    before, candidate, after = indices
    if candidate < 0 or candidate >= frame_count:
        raise ValueError("Candidate frame_index is outside the source video.")
    return min(max(0, before), frame_count - 1), candidate, min(max(0, after), frame_count - 1)


def write_review_sequence_contact_sheet(
    targets: Sequence[Mapping[str, str]],
    video_path: str | Path,
    destination: str | Path,
    offset_seconds: float = 0.2,
    *,
    offset_frames: int | None = None,
) -> Path:
    """Create before/candidate/after tiles without running pose inference.

    Use ``offset_frames`` for an exact source-frame context window, or retain
    the existing time-based behavior with ``offset_seconds``.
    """
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Contact sheets require the project's OpenCV and NumPy dependencies.") from exc
    if not targets:
        raise ValueError("No uncertain or low-confidence events were selected.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open source video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("Source video does not report a valid FPS for time-based frame extraction.")
    raw_frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = int(round(raw_frame_count)) if math.isfinite(raw_frame_count) and raw_frame_count > 0 else None
    if frame_count is None:
        capture.release()
        raise ValueError("Source video does not report a valid frame count for bounded context extraction.")

    sequences: list[tuple[Mapping[str, str], tuple[int, int, int]]] = []
    requested: set[int] = set()
    for target in targets:
        try:
            center = int(target["frame_index"])
            timestamp_raw = target.get("timestamp_seconds", "")
            timestamp = float(timestamp_raw) if timestamp_raw else None
        except (KeyError, TypeError, ValueError) as exc:
            capture.release()
            raise ValueError("Selected review row has invalid frame or timestamp data.") from exc
        indices = clamp_review_frame_indices(
            review_frame_indices(
                center, timestamp, fps, offset_seconds, offset_frames=offset_frames
            ),
            frame_count,
        )
        sequences.append((target, indices))
        requested.update(indices)

    frames: dict[int, object] = {}
    max_index = max(requested)
    current = 0
    try:
        while current <= max_index:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if current in requested:
                frames[current] = frame
            current += 1
    finally:
        capture.release()
    missing = sorted(requested.difference(frames))
    if missing:
        raise ValueError(f"Source video ended before requested frame(s): {missing}")

    columns = 3
    tile_width, tile_height, header_height, margin = 300, 380, 54, 10
    sheet = np.full(
        (margin + len(sequences) * (tile_height + header_height + margin),
         margin + columns * (tile_width + margin), 3),
        245, dtype=np.uint8,
    )
    labels = ("BEFORE", "CANDIDATE", "AFTER")
    for row_index, (target, indices) in enumerate(sequences):
        event_id = target.get("event_id", f"E{row_index + 1:03d}")
        side = target.get("side", "?")
        event_time_raw = target.get("timestamp_seconds", "")
        try:
            event_time_label = f"{float(event_time_raw):.3f}s"
        except (TypeError, ValueError):
            event_time_label = "time unavailable"
        for col_index, (kind, source_index) in enumerate(zip(labels, indices)):
            x = margin + col_index * (tile_width + margin)
            y = margin + row_index * (tile_height + header_height + margin)
            actual_time = source_index / fps
            cv2.rectangle(sheet, (x, y), (x + tile_width, y + header_height), (40, 40, 40), -1)
            frame_label = f"{kind} (same frame as candidate)" if kind != "CANDIDATE" and source_index == indices[1] else kind
            cv2.putText(sheet, f"{event_id} {side.upper()} {frame_label} (event {event_time_label})",
                        (x + 5, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(sheet, f"frame {source_index} | t={actual_time:.3f}s",
                        (x + 5, y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            frame = frames[source_index]
            frame_height, frame_width = frame.shape[:2]
            scale = min(tile_width / frame_width, tile_height / frame_height)
            resized_width, resized_height = max(1, round(frame_width * scale)), max(1, round(frame_height * scale))
            resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
            offset_x = x + (tile_width - resized_width) // 2
            offset_y = y + header_height + (tile_height - resized_height) // 2
            sheet[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"OpenCV could not write review sequence sheet: {output}")
    return output


def align_candidate_context_targets(
    review_rows: Sequence[Mapping[str, str]],
    candidate_events: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    """Join candidate rows to saved review IDs without changing either input.

    Output order follows the Phase 2.4 candidate CSV. The match uses side and
    frame, and timestamps must agree, so edited labels remain attached to the
    same event when candidate rows are passed to the context renderer.
    """
    review_by_event: dict[tuple[str, str], Mapping[str, str]] = {}
    for row in review_rows:
        key = (row.get("side", ""), row.get("frame_index", ""))
        if not row.get("event_id") or key in review_by_event:
            raise ValueError(f"Review CSV has a missing event ID or duplicate event key: {key}.")
        review_by_event[key] = row

    targets: list[dict[str, str]] = []
    seen_candidates: set[tuple[str, str]] = set()
    for event in candidate_events:
        key = (event.get("side", ""), event.get("frame_index", ""))
        if key in seen_candidates:
            raise ValueError(f"Candidate CSV has a duplicate side/frame event: {key}.")
        seen_candidates.add(key)
        review = review_by_event.get(key)
        if review is None:
            raise ValueError(f"No review row matches candidate side/frame {key}.")
        if review.get("timestamp_seconds", "") != event.get("timestamp_seconds", ""):
            raise ValueError(f"Candidate and review timestamps differ for {key}.")
        target = dict(review)
        target["side"] = event.get("side", "")
        target["frame_index"] = event.get("frame_index", "")
        target["timestamp_seconds"] = event.get("timestamp_seconds", "")
        targets.append(target)

    if len(targets) != len(review_rows):
        raise ValueError("Candidate and review CSVs do not contain the same events.")
    return targets


def write_candidate_contact_sheet(
    events: Sequence[Mapping[str, str]], video_path: str | Path, destination: str | Path
) -> Path:
    """Decode only for visual review (no pose inference) and save labeled candidate frames."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Contact sheets require the project's OpenCV and NumPy dependencies.") from exc

    indexed_events: list[tuple[int, Mapping[str, str]]] = []
    for event in events:
        try:
            frame_index = int(event["frame_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Candidate event has an invalid frame_index.") from exc
        if frame_index < 0:
            raise ValueError("Candidate event frame_index must be non-negative.")
        indexed_events.append((frame_index, event))
    if not indexed_events:
        raise ValueError("No candidate events were provided for the contact sheet.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open source video: {video_path}")
    requested = {index for index, _ in indexed_events}
    frames: dict[int, object] = {}
    frame_index = 0
    try:
        while frame_index <= max(requested):
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_index in requested:
                frames[frame_index] = frame
            frame_index += 1
    finally:
        capture.release()
    missing = sorted(requested.difference(frames))
    if missing:
        raise ValueError(f"Source video ended before candidate frame(s): {missing}")

    columns = 4
    tile_width, tile_height, title_height, margin = 220, 280, 44, 8
    rows = (len(indexed_events) + columns - 1) // columns
    sheet = np.full(
        (margin + rows * (tile_height + title_height + margin),
         margin + columns * (tile_width + margin), 3),
        245, dtype=np.uint8,
    )
    for item_index, (source_index, event) in enumerate(indexed_events):
        row_index, column_index = divmod(item_index, columns)
        x = margin + column_index * (tile_width + margin)
        y = margin + row_index * (tile_height + title_height + margin)
        event_id = f"E{item_index + 1:03d}"
        timestamp = event.get("timestamp_seconds", "")
        side = event.get("side", "?")
        cv2.rectangle(sheet, (x, y), (x + tile_width, y + title_height), (40, 40, 40), -1)
        try:
            time_label = f"t = {float(timestamp):.3f} s"
        except (TypeError, ValueError):
            time_label = "timestamp unavailable"
        cv2.putText(sheet, f"{event_id} {side.upper()} frame {source_index}", (x + 5, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, time_label, (x + 5, y + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)

        frame = frames[source_index]
        frame_height, frame_width = frame.shape[:2]
        scale = min(tile_width / frame_width, tile_height / frame_height)
        resized_width = max(1, round(frame_width * scale))
        resized_height = max(1, round(frame_height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        offset_x = x + (tile_width - resized_width) // 2
        offset_y = y + title_height + (tile_height - resized_height) // 2
        sheet[offset_y:offset_y + resized_height, offset_x:offset_x + resized_width] = resized

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"OpenCV could not write contact sheet: {output}")
    return output
