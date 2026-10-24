"""Orchestrate the existing offline gait-analysis stages for one local video.

Artifacts are built in a temporary sibling directory and published by a same-
filesystem directory rename only after all analysis and writers succeed. The
requested output directory must not already exist. The parent directory may be
created when needed; on failure only this invocation's temporary directory is
removed. This protects existing files but does not make a rename atomic across
all filesystems or protect against external processes changing the destination.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable

from .gait_events import (
    EventDetectionParameters,
    EventDetectionResult,
    detect_leg_cycle_candidates,
    validate_phase23_alignment,
)
from .gait_metrics import KneeAngleRow, knee_angle_rows
from .cadence import CandidateCadenceParameters, estimate_candidate_cadence
from .foot_strike_analysis import (
    ContactZoneParameters,
    candidates_to_predictions,
    detect_contact_zone_candidates,
    load_ground_truth,
    validate_ground_truth_source_metadata,
    write_ground_truth_analysis,
)
from .pipeline import extract_video
from .quality import SideQualityDiagnostics, diagnose_knee_angles
from .report import (
    KNEE_ANGLE_COLUMNS,
    write_candidate_events_csv,
    write_knee_angles_csv,
    write_smoothed_knee_angles_csv,
)
from .signal_processing import add_smoothed_knee_angles
from .video_io import VideoLimits, validate_video
from .visualization import write_gait_event_diagnostic_svg, write_knee_angle_plot_svg


class AnalysisPipelineError(RuntimeError):
    """A pipeline stage failed; the original exception is available as its cause."""


@dataclass(frozen=True)
class AnalysisResult:
    """Paths and compact diagnostics for one completed video analysis."""

    input_video: Path
    output_dir: Path
    annotated_video: Path
    landmarks_json: Path
    knee_angles_csv: Path
    smoothed_knee_angles_csv: Path
    knee_angles_svg: Path
    gait_events_csv: Path
    gait_events_svg: Path
    left_quality: SideQualityDiagnostics
    right_quality: SideQualityDiagnostics
    candidate_event_count: int
    left_candidate_event_count: int
    right_candidate_event_count: int
    event_interpretation: str = "Exploratory heel-trajectory candidates; not confirmed foot-strikes."
    ground_truth_path: Path | None = None
    foot_strike_evaluation_csv: Path | None = None
    foot_strike_matches_csv: Path | None = None
    foot_strike_evaluation_json: Path | None = None
    knee_angles_at_ground_truth_csv: Path | None = None
    cadence_csv: Path | None = None
    cadence_svg: Path | None = None
    ground_truth_metrics_json: Path | None = None
    ground_truth_report: Path | None = None
    candidate_cadence_csv: Path | None = None
    candidate_cadence_summary_json: Path | None = None
    diagnostics_json: Path | None = None
    cadence_summary: dict[str, Any] | None = None
    analysis_warnings: tuple[str, ...] = ()
    video_metadata: dict[str, Any] | None = None


def _is_same_or_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _check_no_path_overlap(output_dir: Path, input_path: Path, model_path: Path) -> None:
    target = output_dir.resolve()
    for label, artifact in (("input video", input_path), ("model asset", model_path)):
        resolved = artifact.resolve()
        if _is_same_or_within(target, resolved) or _is_same_or_within(resolved, target):
            raise ValueError(f"Output directory overlaps the {label}: {artifact}.")


def _run_stage(name: str, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return operation(*args, **kwargs)
    except Exception as exc:
        raise AnalysisPipelineError(f"{name} failed: {exc}") from exc


def _rename_directory_with_retry(source: Path, destination: Path, attempts: int = 5) -> None:
    """Retry transient Windows sharing/access errors while publishing a run directory."""
    for attempt in range(attempts):
        try:
            os.rename(source, destination)
            return
        except OSError as exc:
            retryable_windows_error = (
                os.name == "nt" and getattr(exc, "winerror", None) in {5, 32, 33}
            )
            if not retryable_windows_error or attempt + 1 == attempts:
                raise
            if destination.exists():
                raise FileExistsError(
                    f"Output directory appeared during analysis: {destination}"
                ) from exc
            time.sleep(0.1 * (2**attempt))


def analyze_video(
    video_path: str | Path,
    output_dir: str | Path,
    model_path: str | Path = Path("models/pose_landmarker_full.task"),
    limits: VideoLimits | None = None,
    visibility_threshold: float = 0.5,
    presence_threshold: float = 0.5,
    event_parameters: EventDetectionParameters | None = None,
    cadence_parameters: CandidateCadenceParameters | None = None,
    ground_truth_path: str | Path | None = None,
    contact_zone_parameters: ContactZoneParameters | None = None,
    matching_tolerance_frames: int = 2,
    progress: Callable[[str], None] | None = None,
) -> AnalysisResult:
    """Run the analysis stages and publish artifacts for one MP4.

    Relative video/model/output paths are interpreted from the current working
    directory, matching the existing scripts and model-path behavior. A
    matching existing landmarks file is not reused because the current JSON
    records only the source basename, which cannot identify a video reliably.
    """
    source = Path(video_path).expanduser().resolve()
    model = Path(model_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    _check_no_path_overlap(destination, source, model)
    if destination.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {destination}")
    if not model.is_file():
        raise FileNotFoundError(f"Pose model asset does not exist: {model}")

    # Validate decoder access and configured limits before model initialization.
    capture = _run_stage("Video validation", validate_video, source, limits)
    try:
        capture[2].release()
    except Exception as exc:
        raise AnalysisPipelineError(f"Video validation cleanup failed: {exc}") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    stem = source.stem
    paths = {
        "annotated_video": temporary_dir / f"{stem}_annotated.mp4",
        "landmarks_json": temporary_dir / f"{stem}_landmarks.json",
        "knee_angles_csv": temporary_dir / f"{stem}_knee_angles_phase22.csv",
        "smoothed_knee_angles_csv": temporary_dir / f"{stem}_knee_angles_phase23.csv",
        "knee_angles_svg": temporary_dir / f"{stem}_knee_angles_phase23.svg",
        "gait_events_csv": temporary_dir / f"{stem}_gait_events_phase24.csv",
        "gait_events_svg": temporary_dir / f"{stem}_gait_events_phase24.svg",
        "candidate_cadence_csv": temporary_dir / "candidate_cadence.csv",
        "candidate_cadence_summary_json": temporary_dir / "candidate_cadence_summary.json",
        "diagnostics_json": temporary_dir / "analysis_diagnostics.json",
    }
    analysis_warnings: list[str] = []
    try:
        _run_stage(
            "Landmark extraction",
            extract_video,
            source,
            model,
            paths["landmarks_json"],
            annotated_video=paths["annotated_video"],
            limits=limits,
            progress=progress if progress is not None else (lambda _message: None),
        )

        def load_landmarks() -> dict[str, Any]:
            with paths["landmarks_json"].open("r", encoding="utf-8") as stream:
                document = json.load(stream)
            if not isinstance(document, dict):
                raise ValueError("Landmark JSON root must be an object.")
            return document

        document = _run_stage("Landmark JSON loading", load_landmarks)
        event_detection_warning: str | None = None
        try:
            angle_rows = _run_stage(
                "Knee-angle calculation",
                knee_angle_rows,
                document,
                visibility_threshold,
                presence_threshold,
            )
        except Exception as exc:
            analysis_warnings.append(f"Knee-angle calculation is unavailable: {exc}")
            angle_rows = [
                KneeAngleRow(
                    frame_index=frame.get("frame_index", index) if isinstance(frame, dict) else index,
                    timestamp_seconds=frame.get("timestamp_seconds") if isinstance(frame, dict) else None,
                    left_knee_angle_deg=None,
                    right_knee_angle_deg=None,
                    left_knee_status="analysis_error",
                    right_knee_status="analysis_error",
                )
                for index, frame in enumerate(document.get("frames", []))
            ]
        angle_mappings = [asdict(row) for row in angle_rows]
        left_quality = _run_stage("Left-side quality diagnostics", diagnose_knee_angles, angle_mappings, "left")
        right_quality = _run_stage("Right-side quality diagnostics", diagnose_knee_angles, angle_mappings, "right")
        smoothed_rows = _run_stage("Knee-angle smoothing", add_smoothed_knee_angles, angle_mappings)

        try:
            event_result = _run_stage(
                "Exploratory event detection",
                detect_leg_cycle_candidates,
                document,
                event_parameters,
            )
            _run_stage("Phase 2.3/event alignment", validate_phase23_alignment, event_result.samples, smoothed_rows)
        except Exception as exc:
            event_detection_warning = str(exc)
            analysis_warnings.append(f"Exploratory event detection is unavailable: {exc}")
            event_result = EventDetectionResult((), (), event_parameters or EventDetectionParameters())

        video_metadata = document.get("video", {})
        video_metadata = video_metadata if isinstance(video_metadata, dict) else {}
        fps = video_metadata.get("fps")
        frame_count = len(document.get("frames", []))
        duration = frame_count / float(fps) if isinstance(fps, (int, float)) and fps > 0 else None
        try:
            cadence_series, cadence_summary = estimate_candidate_cadence(
                event_result.events,
                frames=document.get("frames", []),
                fps=float(fps) if isinstance(fps, (int, float)) and fps > 0 else None,
                duration_seconds=duration,
                parameters=cadence_parameters,
            )
            if event_detection_warning is not None:
                cadence_summary["status"] = "event_detection_unavailable"
                cadence_summary["average_cadence_steps_per_minute"] = None
                cadence_summary["warning"] = (
                    "Cadence is unavailable because exploratory event detection or frame alignment failed."
                )
        except Exception as exc:
            analysis_warnings.append(f"Candidate cadence is unavailable: {exc}")
            cadence_series = []
            cadence_summary = {
                "status": "calculation_error",
                "event_source": "exploratory_heel_trajectory_candidates",
                "average_cadence_steps_per_minute": None,
                "candidate_event_count": len(event_result.events),
                "retained_event_count": 0,
                "duration_seconds": duration,
                "fps": fps,
                "warning": str(exc),
            }
        _run_stage("Phase 2.2 CSV writing", write_knee_angles_csv, angle_rows, paths["knee_angles_csv"])
        _run_stage(
            "Phase 2.3 CSV writing",
            write_smoothed_knee_angles_csv,
            smoothed_rows,
            KNEE_ANGLE_COLUMNS,
            paths["smoothed_knee_angles_csv"],
        )
        _run_stage("Phase 2.3 plot writing", write_knee_angle_plot_svg, smoothed_rows, paths["knee_angles_svg"])
        _run_stage(
            "Phase 2.4 CSV writing",
            write_candidate_events_csv,
            event_result.events,
            paths["gait_events_csv"],
        )
        with paths["candidate_cadence_csv"].open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=(
                "window_start_seconds", "window_end_seconds", "window_center_seconds",
                "step_count", "cadence_steps_per_minute",
            ))
            writer.writeheader()
            writer.writerows(cadence_series)
        paths["candidate_cadence_summary_json"].write_text(
            json.dumps(cadence_summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _run_stage(
            "Phase 2.4 plot writing",
            write_gait_event_diagnostic_svg,
            event_result.samples,
            event_result.events,
            smoothed_rows,
            paths["gait_events_svg"],
        )

        ground_truth_artifacts: dict[str, Path] = {}
        resolved_ground_truth: Path | None = None
        if ground_truth_path is not None:
            resolved_ground_truth = Path(ground_truth_path).expanduser().resolve()

            def analyze_ground_truth() -> dict[str, Path]:
                source_name = Path(str(document.get("source_video", ""))).name
                if not source_name or source_name.lower() != source.name.lower():
                    raise ValueError(
                        "Ground Truth source filename does not match this analyzed video."
                    )
                validate_ground_truth_source_metadata(
                    resolved_ground_truth, document, source_video_path=source
                )
                video_metadata = document.get("video", {})
                fps = video_metadata.get("fps") if isinstance(video_metadata, dict) else None
                if not isinstance(fps, (int, float)) or fps <= 0:
                    raise ValueError("Ground Truth evaluation requires valid source FPS metadata.")
                truth = load_ground_truth(resolved_ground_truth, document)
                predictions_a = candidates_to_predictions(event_result.events)
                predictions_b = detect_contact_zone_candidates(document, contact_zone_parameters)
                algorithms = {
                    "A_heel_local_extrema": predictions_a,
                    "B_foot_contact_zone": predictions_b,
                }
                artifacts = write_ground_truth_analysis(
                    temporary_dir,
                    ground_truth=truth,
                    algorithms=algorithms,
                    fps=float(fps),
                    document=document,
                    knee_rows=angle_rows,
                    tolerance_frames=matching_tolerance_frames,
                    contact_parameters=contact_zone_parameters,
                )
                return artifacts

            try:
                ground_truth_artifacts = _run_stage(
                    "Ground Truth evaluation and metrics", analyze_ground_truth
                )
            except Exception as exc:
                analysis_warnings.append(f"Ground Truth evaluation is unavailable: {exc}")

        landmark_frames = document.get("frames", [])
        valid_landmark_frames = sum(
            bool(frame.get("poses")) for frame in landmark_frames if isinstance(frame, dict)
        )
        if not landmark_frames or valid_landmark_frames / len(landmark_frames) < 0.5:
            analysis_warnings.append("Pose tracking is low quality: fewer than half of frames contain a detected pose.")
        for side, quality in (("left", left_quality), ("right", right_quality)):
            if quality.usable_angle_count == 0:
                analysis_warnings.append(f"No valid {side} knee-angle measurements were available.")
            elif quality.valid_frame_percentage < 50.0:
                analysis_warnings.append(
                    f"Low {side} knee-angle coverage: {quality.valid_frame_percentage:.1f}% of frames were valid."
                )
        diagnostics = {
            "source_video": document.get("source_video"),
            "video": {
                "fps": video_metadata.get("fps"),
                "frame_count": frame_count,
                "duration_seconds": duration,
                "width": video_metadata.get("width"),
                "height": video_metadata.get("height"),
                "timestamp_basis": document.get("timestamp_basis"),
            },
            "valid_landmark_frames": valid_landmark_frames,
            "valid_landmark_frame_percentage": (
                valid_landmark_frames * 100.0 / len(landmark_frames) if landmark_frames else 0.0
            ),
            "knee_angles": {
                "left": asdict(left_quality),
                "right": asdict(right_quality),
            },
            "candidate_events": {
                "total": len(event_result.events),
                "by_side": {
                    side: sum(event.side == side for event in event_result.events)
                    for side in ("left", "right")
                },
                "cadence": cadence_summary,
            },
            "ground_truth_evaluation": (
                "completed" if ground_truth_artifacts
                else "unavailable" if ground_truth_path is not None
                else "not_requested"
            ),
            "warnings": analysis_warnings,
        }
        paths["diagnostics_json"].write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        def publish() -> None:
            if destination.exists():
                raise FileExistsError(f"Output directory appeared during analysis: {destination}")
            _rename_directory_with_retry(temporary_dir, destination)

        _run_stage("Output publication", publish)
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    event_counts = {
        side: sum(event.side == side for event in event_result.events)
        for side in ("left", "right")
    }
    final_paths = {name: destination / path.name for name, path in paths.items()}
    final_ground_truth_artifacts = {
        name: destination / path.name for name, path in ground_truth_artifacts.items()
    }
    return AnalysisResult(
        input_video=source,
        output_dir=destination,
        **final_paths,
        left_quality=left_quality,
        right_quality=right_quality,
        candidate_event_count=len(event_result.events),
        left_candidate_event_count=event_counts["left"],
        right_candidate_event_count=event_counts["right"],
        ground_truth_path=resolved_ground_truth,
        foot_strike_evaluation_csv=final_ground_truth_artifacts.get("evaluation_csv"),
        foot_strike_matches_csv=final_ground_truth_artifacts.get("matches_csv"),
        foot_strike_evaluation_json=final_ground_truth_artifacts.get("evaluation_json"),
        knee_angles_at_ground_truth_csv=final_ground_truth_artifacts.get("knee_csv"),
        cadence_csv=final_ground_truth_artifacts.get("cadence_csv"),
        cadence_svg=final_ground_truth_artifacts.get("cadence_svg"),
        ground_truth_metrics_json=final_ground_truth_artifacts.get("metrics_json"),
        ground_truth_report=final_ground_truth_artifacts.get("report"),
        cadence_summary=cadence_summary,
        analysis_warnings=tuple(analysis_warnings),
        video_metadata=video_metadata,
    )
