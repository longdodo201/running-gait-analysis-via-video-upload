"""Ground-truth evaluation and event-level gait summaries.

The image-contact detector in this module is deliberately distinct from the
heel-y local-extrema detector: it looks for a foot landmark entering a fixed
near-belt zone while its vertical motion is limited. It remains exploratory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .gait_events import CandidateEvent
from .gait_metrics import KneeAngleRow


GROUND_TRUTH_COLUMNS = (
    "event_id", "frame_index", "timestamp_seconds", "foot_side", "event_type",
    "label_confidence", "notes",
)
EVALUATION_COLUMNS = (
    "algorithm", "side", "tp", "fp", "fn", "precision", "recall", "f1",
    "mean_signed_error_frames", "mean_absolute_error_frames",
    "mean_signed_error_ms", "mean_absolute_error_ms", "matched_count",
)
MATCH_COLUMNS = (
    "algorithm", "side", "event_id", "ground_truth_frame", "prediction_frame",
    "signed_error_frames", "absolute_error_frames", "signed_error_ms", "absolute_error_ms",
)
KNEE_EVENT_COLUMNS = (
    "event_id", "frame_index", "timestamp_seconds", "foot_side", "knee_angle_deg",
    "hip_visibility", "knee_visibility", "ankle_visibility", "min_visibility",
    "hip_presence", "knee_presence", "ankle_presence", "min_presence", "status", "valid",
)
CADENCE_COLUMNS = (
    "window_start_seconds", "window_end_seconds", "window_center_seconds",
    "step_count", "cadence_steps_per_minute",
)


@dataclass(frozen=True)
class GroundTruthEvent:
    event_id: str
    frame_index: int
    timestamp_seconds: float
    side: str
    event_type: str = "foot_strike"
    label_confidence: str = ""
    notes: str = ""


@dataclass(frozen=True)
class PredictionEvent:
    side: str
    frame_index: int
    timestamp_seconds: float | None


@dataclass(frozen=True)
class ContactZoneParameters:
    """Dataset-level heuristic settings; not fitted to the Ground Truth labels."""

    ground_line_ratio: float = 0.92
    contact_band_pixels: float = 18.0
    max_vertical_speed_pixels_per_second: float = 600.0
    minimum_separation_frames: int = 8
    confidence_threshold: float = 0.5


def _finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # accepts CSV numeric strings
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def load_ground_truth(path: str | Path, document: Mapping[str, Any]) -> tuple[GroundTruthEvent, ...]:
    """Read and validate event labels against the saved source-frame timeline."""
    source = Path(path)
    video = document.get("video")
    frames = document.get("frames")
    if not isinstance(video, Mapping) or not isinstance(frames, list):
        raise ValueError("Landmark JSON must contain video metadata and a frames list.")
    fps = _finite(video.get("fps"))
    if fps is None or fps <= 0:
        raise ValueError("Landmark JSON must provide a positive source FPS.")

    with source.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        if not set(GROUND_TRUTH_COLUMNS).issubset(columns):
            missing = sorted(set(GROUND_TRUTH_COLUMNS) - columns)
            raise ValueError(f"Ground Truth CSV is missing required columns: {missing}.")
        rows = list(reader)

    events: list[GroundTruthEvent] = []
    seen: set[tuple[str, int]] = set()
    for line_number, row in enumerate(rows, start=2):
        try:
            frame = int(row["frame_index"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Ground Truth row {line_number} has an invalid frame_index.") from exc
        side = str(row["foot_side"]).strip().lower()
        event_type = str(row["event_type"]).strip().lower()
        event_id = str(row["event_id"]).strip()
        timestamp = _finite(row.get("timestamp_seconds"))
        if not event_id:
            raise ValueError(f"Ground Truth row {line_number} has an empty event_id.")
        if side not in {"left", "right"}:
            raise ValueError(f"Ground Truth row {line_number} foot_side must be left or right.")
        if event_type != "foot_strike":
            raise ValueError(f"Ground Truth row {line_number} event_type must be foot_strike.")
        if frame < 0 or frame >= len(frames):
            raise ValueError(f"Ground Truth row {line_number} frame_index is outside the landmark timeline.")
        if timestamp is None or timestamp < 0:
            raise ValueError(f"Ground Truth row {line_number} has an invalid timestamp_seconds.")
        source_frame = frames[frame]
        if not isinstance(source_frame, Mapping) or int(source_frame.get("frame_index", -1)) != frame:
            raise ValueError(f"Ground Truth row {line_number} does not align with the landmark frame index.")
        source_time = _finite(source_frame.get("timestamp_seconds"))
        # The GT was made from OpenCV decoder PTS while the saved landmarks use
        # source FPS. Half a frame permits representation/rounding differences.
        if source_time is None or abs(timestamp - source_time) > (0.5 / fps + 1e-6):
            raise ValueError(f"Ground Truth row {line_number} timestamp does not align with its source frame.")
        key = (side, frame)
        if key in seen:
            raise ValueError(f"Ground Truth contains a duplicate {side} event at frame {frame}.")
        seen.add(key)
        confidence = str(row.get("label_confidence", "")).strip().lower()
        if confidence and confidence not in {"high", "medium", "low"}:
            raise ValueError(f"Ground Truth row {line_number} has unsupported label_confidence {confidence!r}.")
        events.append(GroundTruthEvent(
            event_id=event_id,
            frame_index=frame,
            timestamp_seconds=timestamp,
            side=side,
            event_type=event_type,
            label_confidence=confidence,
            notes=str(row.get("notes", "")).strip(),
        ))
    return tuple(sorted(events, key=lambda event: (event.frame_index, event.side, event.event_id)))


def validate_ground_truth_source_metadata(
    ground_truth_path: str | Path,
    document: Mapping[str, Any],
    source_video_path: str | Path | None = None,
) -> None:
    """Validate an optional dataset sidecar against landmarks and source bytes."""
    metadata_path = Path(ground_truth_path).with_name("foot_strike_ground_truth_metadata.json")
    if not metadata_path.is_file():
        return
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Ground Truth source metadata cannot be read: {exc}") from exc
    if not isinstance(metadata, Mapping):
        raise ValueError("Ground Truth source metadata must be a JSON object.")
    video = document.get("video")
    if not isinstance(video, Mapping):
        raise ValueError("Landmark JSON is missing video metadata.")
    source_name = Path(str(document.get("source_video", ""))).name
    if source_name.lower() != str(metadata.get("source_video", "")).lower():
        raise ValueError("Ground Truth source metadata does not match the landmark source filename.")
    expected_count = metadata.get("frame_count")
    actual_count = len(document.get("frames", []))
    if expected_count is not None and int(expected_count) != actual_count:
        raise ValueError("Ground Truth source metadata frame count does not match landmarks.")
    for key in ("fps", "width", "height"):
        expected = _finite(metadata.get(key))
        actual = _finite(video.get(key))
        if expected is not None and (actual is None or not math.isclose(expected, actual, rel_tol=0, abs_tol=1e-6)):
            raise ValueError(f"Ground Truth source metadata {key} does not match landmarks.")
    if source_video_path is not None:
        expected_hash = str(metadata.get("source_sha256", "")).strip().lower()
        if not expected_hash:
            raise ValueError("Ground Truth source metadata has no source_sha256 fingerprint.")
        digest = hashlib.sha256()
        with Path(source_video_path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_hash:
            raise ValueError("Video bytes do not match the source fingerprint saved with Ground Truth.")


def detect_contact_zone_candidates(
    document: Mapping[str, Any],
    params: ContactZoneParameters | None = None,
) -> tuple[PredictionEvent, ...]:
    """Detect foot-index/heel near-belt zone entries with limited vertical speed.

    This uses foot geometry and a belt-height zone rather than heel-y peaks.
    The default belt height is 92% of image height and should be treated as a
    heuristic calibration for side-view clips, not a universal treadmill line.
    """
    settings = params or ContactZoneParameters()
    if not 0 < settings.ground_line_ratio <= 1:
        raise ValueError("ground_line_ratio must be in (0, 1].")
    if (not math.isfinite(settings.contact_band_pixels)
            or not math.isfinite(settings.max_vertical_speed_pixels_per_second)
            or settings.contact_band_pixels < 0
            or settings.max_vertical_speed_pixels_per_second < 0):
        raise ValueError("Contact band and vertical-speed threshold must not be negative.")
    if settings.minimum_separation_frames < 1:
        raise ValueError("minimum_separation_frames must be positive.")
    if not 0 <= settings.confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1.")
    video = document.get("video")
    frames = document.get("frames")
    if not isinstance(video, Mapping) or not isinstance(frames, list):
        raise ValueError("Input JSON must contain video metadata and frames.")
    height = _finite(video.get("height"))
    fps = _finite(video.get("fps"))
    if height is None or height <= 0 or fps is None or fps <= 0:
        raise ValueError("Input JSON must provide positive video height and FPS.")
    belt_y = height * settings.ground_line_ratio

    side_signals: dict[str, list[tuple[float | None, float | None, float | None]]] = {}
    for side in ("left", "right"):
        values: list[tuple[float | None, float | None, float | None]] = []
        for frame in frames:
            poses = frame.get("poses") if isinstance(frame, Mapping) else None
            pose = poses[0] if isinstance(poses, list) and poses else None
            landmarks = pose.get("landmarks") if isinstance(pose, Mapping) else None
            if not isinstance(landmarks, Mapping):
                values.append((None, None, None))
                continue
            points = [landmarks.get(f"{side}_{part}") for part in ("heel", "foot_index")]
            parsed: list[tuple[float, float, float]] = []
            for point in points:
                if not isinstance(point, Mapping):
                    continue
                y = _finite(point.get("y"))
                visibility = _finite(point.get("visibility"))
                presence = _finite(point.get("presence"))
                if (y is not None and visibility is not None and presence is not None
                        and visibility >= settings.confidence_threshold
                        and presence >= settings.confidence_threshold):
                    parsed.append((y * height, visibility, presence))
            if not parsed:
                values.append((None, None, None))
            else:
                # Image y grows downward; the larger y coordinate is the lower
                # of the heel and toe estimates for this frame.
                values.append((max(point[0] for point in parsed),
                               min(point[1] for point in parsed),
                               min(point[2] for point in parsed)))
        side_signals[side] = values

    candidates: list[PredictionEvent] = []
    for side, values in side_signals.items():
        initial_y = values[0][0] if values else None
        previous_in_zone = initial_y is not None and initial_y >= belt_y - settings.contact_band_pixels
        last_event_frame = -settings.minimum_separation_frames
        for index, (foot_y, visibility, presence) in enumerate(values[1:], start=1):
            in_zone = foot_y is not None and foot_y >= belt_y - settings.contact_band_pixels
            if foot_y is None:
                previous_in_zone = False
                continue
            previous = values[index - 1] if index > 0 else (None, None, None)
            previous_y = previous[0]
            if index == 0:
                vertical_speed = 0.0
            else:
                previous_frame = frames[index - 1]
                current_frame = frames[index]
                previous_time = _finite(previous_frame.get("timestamp_seconds")) if isinstance(previous_frame, Mapping) else None
                current_time = _finite(current_frame.get("timestamp_seconds")) if isinstance(current_frame, Mapping) else None
                delta_time = current_time - previous_time if current_time is not None and previous_time is not None else 0.0
                if previous_y is None or delta_time <= 0:
                    vertical_speed = math.inf
                else:
                    vertical_speed = abs(foot_y - previous_y) / delta_time
            if (in_zone and not previous_in_zone
                    and vertical_speed <= settings.max_vertical_speed_pixels_per_second
                    and index - last_event_frame >= settings.minimum_separation_frames):
                frame_index = int(frames[index].get("frame_index", index))
                timestamp = _finite(frames[index].get("timestamp_seconds"))
                candidates.append(PredictionEvent(side, frame_index, timestamp))
                last_event_frame = index
            previous_in_zone = bool(in_zone)
    return tuple(sorted(candidates, key=lambda event: (event.frame_index, event.side)))


def candidates_to_predictions(events: Sequence[CandidateEvent]) -> tuple[PredictionEvent, ...]:
    return tuple(PredictionEvent(event.side, event.frame_index, event.timestamp_seconds) for event in events)


def _optimal_pairs(
    ground_truth: Sequence[GroundTruthEvent], predictions: Sequence[PredictionEvent], tolerance_frames: int,
) -> tuple[tuple[int, int], ...]:
    """Maximize match count, then minimize total absolute timing error."""
    gt = tuple(sorted(enumerate(ground_truth), key=lambda item: (item[1].frame_index, item[0])))
    pred = tuple(sorted(enumerate(predictions), key=lambda item: (item[1].frame_index, item[0])))

    @lru_cache(maxsize=None)
    def solve(i: int, j: int) -> tuple[int, int, tuple[tuple[int, int], ...]]:
        if i >= len(gt) or j >= len(pred):
            return 0, 0, ()
        options = [solve(i + 1, j), solve(i, j + 1)]
        gi, g_event = gt[i]
        pi, p_event = pred[j]
        error = p_event.frame_index - g_event.frame_index
        if abs(error) <= tolerance_frames:
            count, cost, pairs = solve(i + 1, j + 1)
            options.append((count + 1, cost + abs(error), ((gi, pi),) + pairs))
        # Deterministic tie-break: most matches, least error, then earliest pairs.
        return min(options, key=lambda result: (-result[0], result[1], result[2]))

    return solve(0, 0)[2]


def evaluate_predictions(
    ground_truth: Sequence[GroundTruthEvent],
    algorithms: Mapping[str, Sequence[PredictionEvent]],
    *,
    fps: float,
    tolerance_frames: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return per-algorithm/per-side metrics and event-level one-to-one matches."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number.")
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must not be negative.")
    summary: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for algorithm, all_predictions in algorithms.items():
        for side in ("left", "right"):
            truth = [event for event in ground_truth if event.side == side]
            predictions = [event for event in all_predictions if event.side == side]
            pairs = _optimal_pairs(truth, predictions, tolerance_frames)
            errors = [predictions[p].frame_index - truth[g].frame_index for g, p in pairs]
            signed_ms: list[float] = []
            absolute_ms: list[float] = []
            for g_index, p_index in pairs:
                gt_event = truth[g_index]
                prediction = predictions[p_index]
                if prediction.timestamp_seconds is not None:
                    error_ms = (prediction.timestamp_seconds - gt_event.timestamp_seconds) * 1000.0
                else:
                    error_ms = (prediction.frame_index - gt_event.frame_index) * 1000.0 / fps
                signed_ms.append(error_ms)
                match_rows.append({
                    "algorithm": algorithm,
                    "side": side,
                    "event_id": gt_event.event_id,
                    "ground_truth_frame": gt_event.frame_index,
                    "prediction_frame": prediction.frame_index,
                    "signed_error_frames": prediction.frame_index - gt_event.frame_index,
                    "absolute_error_frames": abs(prediction.frame_index - gt_event.frame_index),
                    "signed_error_ms": error_ms,
                    "absolute_error_ms": abs(error_ms),
                })
            tp = len(pairs)
            fp = len(predictions) - tp
            fn = len(truth) - tp
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = (2 * precision * recall / (precision + recall)
                  if precision is not None and recall is not None and precision + recall else 0.0
                  if precision is not None and recall is not None else None)
            summary.append({
                "algorithm": algorithm,
                "side": side,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_signed_error_frames": mean(errors) if errors else None,
                "mean_absolute_error_frames": mean(abs(error) for error in errors) if errors else None,
                "mean_signed_error_ms": mean(signed_ms) if signed_ms else None,
                "mean_absolute_error_ms": mean(abs(error) for error in signed_ms) if signed_ms else None,
                "matched_count": tp,
            })
    return summary, match_rows


def event_knee_angle_rows(
    ground_truth: Sequence[GroundTruthEvent],
    document: Mapping[str, Any],
    knee_rows: Sequence[KneeAngleRow],
    *,
    confidence_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach validated frame-level knee angles and confidence to each event."""
    frames = document.get("frames")
    if not isinstance(frames, list) or len(knee_rows) != len(frames):
        raise ValueError("Knee-angle rows must align one-to-one with the landmark frames.")
    output: list[dict[str, Any]] = []
    for event in ground_truth:
        frame = frames[event.frame_index]
        poses = frame.get("poses") if isinstance(frame, Mapping) else None
        pose = poses[0] if isinstance(poses, list) and poses else None
        landmarks = pose.get("landmarks") if isinstance(pose, Mapping) else None
        side = event.side
        names = (f"{side}_hip", f"{side}_knee", f"{side}_ankle")
        confidence_values: dict[str, tuple[float | None, float | None]] = {}
        for part, name in zip(("hip", "knee", "ankle"), names):
            landmark = landmarks.get(name) if isinstance(landmarks, Mapping) else None
            visibility = _finite(landmark.get("visibility")) if isinstance(landmark, Mapping) else None
            presence = _finite(landmark.get("presence")) if isinstance(landmark, Mapping) else None
            confidence_values[part] = (visibility, presence)
        visibility_numbers = [v for v, _ in confidence_values.values() if v is not None]
        presence_numbers = [p for _, p in confidence_values.values() if p is not None]
        angle_row = knee_rows[event.frame_index]
        angle = getattr(angle_row, f"{side}_knee_angle_deg")
        status = getattr(angle_row, f"{side}_knee_status")
        valid = status == "ok" and angle is not None
        min_visibility = min(visibility_numbers) if len(visibility_numbers) == 3 else None
        min_presence = min(presence_numbers) if len(presence_numbers) == 3 else None
        if valid and (min_visibility is None or min_presence is None
                      or min_visibility < confidence_threshold or min_presence < confidence_threshold):
            valid = False
            status = "low_confidence"
        output.append({
            "event_id": event.event_id,
            "frame_index": event.frame_index,
            "timestamp_seconds": event.timestamp_seconds,
            "foot_side": side,
            "knee_angle_deg": float(angle) if valid and angle is not None else None,
            "hip_visibility": confidence_values["hip"][0],
            "knee_visibility": confidence_values["knee"][0],
            "ankle_visibility": confidence_values["ankle"][0],
            "min_visibility": min_visibility,
            "hip_presence": confidence_values["hip"][1],
            "knee_presence": confidence_values["knee"][1],
            "ankle_presence": confidence_values["ankle"][1],
            "min_presence": min_presence,
            "status": status,
            "valid": valid,
        })
    sides: dict[str, Any] = {}
    for side in ("left", "right"):
        side_rows = [row for row in output if row["foot_side"] == side]
        values = [row["knee_angle_deg"] for row in side_rows if row["valid"]]
        sides[side] = {
            "event_count": len(side_rows),
            "valid_event_count": len(values),
            "mean_knee_angle_deg": mean(values) if values else None,
        }
    return output, sides


def cadence_series(
    events: Sequence[GroundTruthEvent],
    *,
    duration_seconds: float,
    window_seconds: float = 2.0,
    step_seconds: float = 0.5,
) -> tuple[list[dict[str, float | int]], dict[str, Any]]:
    """Compute clip-wide event rate and a trailing-window cadence series."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("duration_seconds must be finite and positive.")
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("Cadence window and step must be positive.")
    unique: dict[tuple[int, str], GroundTruthEvent] = {}
    for event in events:
        unique[(event.frame_index, event.side)] = event
    valid_events = sorted(unique.values(), key=lambda item: (item.timestamp_seconds, item.frame_index, item.side))
    valid_events = [event for event in valid_events if 0 <= event.timestamp_seconds <= duration_seconds]
    enough_events = len(valid_events) >= 4
    enough_duration = duration_seconds >= 3.0
    average = len(valid_events) / duration_seconds * 60.0 if enough_events and enough_duration else None
    series: list[dict[str, float | int]] = []
    start = 0.0
    while start + window_seconds <= duration_seconds + 1e-9:
        end = start + window_seconds
        count = sum(start <= event.timestamp_seconds < end for event in valid_events)
        series.append({
            "window_start_seconds": start,
            "window_end_seconds": end,
            "window_center_seconds": start + window_seconds / 2,
            "step_count": count,
            "cadence_steps_per_minute": count / window_seconds * 60.0,
        })
        start += step_seconds
    summary = {
        "step_count": len(valid_events),
        "clip_duration_seconds": duration_seconds,
        "average_cadence_steps_per_minute": average,
        "average_method": "all Ground Truth foot-strike rows divided by full clip duration, multiplied by 60",
        "window_seconds": window_seconds,
        "window_step_seconds": step_seconds,
        "series_method": "trailing half-open event-count window [start, end); count / window duration * 60",
        "series_window_count": len(series),
        "warning": "Single-clip observed event rate; partial gait cycles at clip boundaries and manual labels affect cadence."
                   if average is not None else "Insufficient data for a stable clip-average cadence; at least four events and three seconds are required.",
    }
    if not series:
        summary["warning"] += " Clip is shorter than the cadence window; no time-series values were produced."
    return series, summary


def write_ground_truth_analysis(
    output_dir: str | Path,
    *,
    ground_truth: Sequence[GroundTruthEvent],
    algorithms: Mapping[str, Sequence[PredictionEvent]],
    fps: float,
    document: Mapping[str, Any],
    knee_rows: Sequence[KneeAngleRow],
    tolerance_frames: int = 2,
    contact_parameters: ContactZoneParameters | None = None,
) -> dict[str, Path]:
    """Write evaluation, knee-angle, cadence, summary, and report artifacts."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries, matches = evaluate_predictions(
        ground_truth, algorithms, fps=fps, tolerance_frames=tolerance_frames
    )
    knee_data, knee_summary = event_knee_angle_rows(ground_truth, document, knee_rows)
    video = document.get("video", {})
    duration = _finite(video.get("duration_seconds")) if isinstance(video, Mapping) else None
    if duration is None or duration <= 0:
        frames = document.get("frames", [])
        times = [_finite(frame.get("timestamp_seconds")) for frame in frames if isinstance(frame, Mapping)]
        valid_times = [value for value in times if value is not None]
        duration = (valid_times[-1] + 1 / fps) if valid_times else 0.0
    series, cadence_summary = cadence_series(ground_truth, duration_seconds=duration)

    paths = {
        "evaluation_csv": output / "foot_strike_evaluation_summary.csv",
        "matches_csv": output / "foot_strike_evaluation_matches.csv",
        "evaluation_json": output / "foot_strike_evaluation.json",
        "knee_csv": output / "knee_angles_at_ground_truth.csv",
        "cadence_csv": output / "cadence.csv",
        "metrics_json": output / "ground_truth_metrics_summary.json",
        "report": output / "ground_truth_evaluation_report.md",
        "cadence_svg": output / "cadence.svg",
    }
    for key, columns, rows in (
        ("evaluation_csv", EVALUATION_COLUMNS, summaries),
        ("matches_csv", MATCH_COLUMNS, matches),
        ("knee_csv", KNEE_EVENT_COLUMNS, knee_data),
        ("cadence_csv", CADENCE_COLUMNS, series),
    ):
        with paths[key].open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    metrics = {
        "source_video": document.get("source_video"),
        "fps": fps,
        "frame_count": len(document.get("frames", [])),
        "tolerance_frames": tolerance_frames,
        "tolerance_ms_at_fps": tolerance_frames * 1000.0 / fps,
        "ground_truth_event_count": len(ground_truth),
        "event_source": "Ground Truth CSV",
        "algorithm_a": {
            "name": "heel_trajectory_local_extrema",
            "description": "Existing heel image-y local maxima with smoothing, prominence and spacing thresholds; exploratory.",
        },
        "algorithm_b": {
            "name": "foot_contact_zone",
            "description": "Heel/foot-index enters a near-belt image-y zone with limited vertical speed; exploratory, separate from heel extrema.",
            "parameters": asdict(contact_parameters or ContactZoneParameters()),
        },
        "evaluation": summaries,
        "matches": matches,
        "knee_angle_at_ground_truth": knee_summary,
        "cadence": cadence_summary,
        "limitations": [
            "One video is not evidence of general detector accuracy.",
            "Ground Truth is manual video annotation, not force-plate reference data.",
            "Algorithm B uses pose landmarks derived from the same video and a heuristic belt-height ratio; it is not an independent sensor or validated contact measurement.",
            "2D knee angle depends on camera perspective, occlusion and pose-estimation error.",
        ],
    }
    with paths["evaluation_json"].open("x", encoding="utf-8") as stream:
        json.dump({"evaluation": summaries, "matches": matches}, stream, indent=2, allow_nan=False)
        stream.write("\n")
    with paths["metrics_json"].open("x", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, allow_nan=False)
        stream.write("\n")

    def display(value: object) -> str:
        return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)

    report_lines = [
        "# Foot-strike and Ground Truth gait analysis",
        "",
        f"- Source video: `{document.get('source_video')}`",
        f"- Frame count / FPS: {len(document.get('frames', []))} / {fps:g}",
        f"- Ground Truth events: {len(ground_truth)}",
        f"- Matching: same side, one-to-one optimal matching, ±{tolerance_frames} frames (±{tolerance_frames * 1000 / fps:.2f} ms).",
        "- Event timing error sign: prediction frame/time minus Ground Truth frame/time.",
        "",
        "## Event evaluation",
        "",
        "| Algorithm | Side | TP | FP | FN | Precision | Recall | F1 | Signed error (frames) | Absolute error (frames) | Signed error (ms) | Absolute error (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report_lines.append("| " + " | ".join((
            str(row["algorithm"]), str(row["side"]), str(row["tp"]), str(row["fp"]), str(row["fn"]),
            display(row["precision"]), display(row["recall"]), display(row["f1"]),
            display(row["mean_signed_error_frames"]), display(row["mean_absolute_error_frames"]),
            display(row["mean_signed_error_ms"]), display(row["mean_absolute_error_ms"]),
        )) + " |")
    report_lines.extend([
        "",
        "## Knee angle at Ground Truth events",
        "",
        "| Side | Events | Valid angles | Mean angle (degrees) |",
        "|---|---:|---:|---:|",
    ])
    for side in ("left", "right"):
        item = knee_summary[side]
        report_lines.append(f"| {side} | {item['event_count']} | {item['valid_event_count']} | {display(item['mean_knee_angle_deg'])} |")
    report_lines.extend([
        "",
        "Angles use hip-knee and ankle-knee vectors; a straight leg is approximately 180 degrees. A missing/low-confidence landmark yields no angle.",
        "",
        "## Cadence",
        "",
        f"- Step count: {cadence_summary['step_count']} Ground Truth events (each left/right foot-strike counts as one step).",
        f"- Average: {display(cadence_summary['average_cadence_steps_per_minute'])} steps/min = event count / full clip duration ({display(duration)} s) × 60.",
        f"- Time series: {cadence_summary['window_seconds']:g}-second trailing windows stepped every {cadence_summary['window_step_seconds']:g} seconds; event count / window × 60.",
        f"- Limitation: {cadence_summary['warning']}",
        "",
        "## Interpretation limits",
        "",
        "The two algorithms are compared on one manually annotated clip only. Precision/recall/F1 are descriptive for this clip and do not establish general accuracy. The reference labels are visual observations, not laboratory contact measurements. Algorithm B uses the foot-index/heel coordinates and a heuristic belt-height ratio; its pose inputs can share errors with Algorithm A. The 2D knee angles are geometric estimates, not validated biomechanical measurements or clinical advice.",
        "",
    ])
    paths["report"].write_text("\n".join(report_lines), encoding="utf-8")
    _write_cadence_svg(paths["cadence_svg"], series, duration)
    return paths


def _write_cadence_svg(path: Path, series: Sequence[Mapping[str, Any]], duration: float) -> None:
    """Write a compact, accessible SVG cadence series in the dashboard palette."""
    width, height = 960, 320
    left, right, top, bottom = 64, 24, 38, 52
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [float(row["cadence_steps_per_minute"]) for row in series]
    max_y = max(240.0, max(values, default=0.0) * 1.1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<title id="title">Ground Truth cadence over time</title>',
        '<desc id="desc">Step events counted in a trailing two-second window, converted to steps per minute.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="480" y="24" text-anchor="middle" font-size="17">Ground Truth cadence (2-second windows)</text>',
    ]
    for tick in (0, 60, 120, 180, 240):
        y = top + plot_height * (1 - tick / max_y)
        parts.extend((f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#ddd"/>',
                      f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-size="11">{tick}</text>'))
    if series:
        points = []
        for row in series:
            time = float(row["window_center_seconds"])
            cadence = float(row["cadence_steps_per_minute"])
            x = left + (time / duration) * plot_width if duration > 0 else left
            y = top + plot_height * (1 - cadence / max_y)
            points.append(f"{x:.2f},{y:.2f}")
        parts.append(f'<polyline fill="none" stroke="#39d5e6" stroke-width="3" points="{" ".join(points)}"/>')
    parts.extend((f'<text x="{left + plot_width/2:.1f}" y="{height-12}" text-anchor="middle" font-size="12">Time (seconds)</text>',
                  f'<text x="16" y="{top + plot_height/2:.1f}" transform="rotate(-90 16 {top + plot_height/2:.1f})" text-anchor="middle" font-size="12">Steps / minute</text>',
                  '</svg>'))
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
