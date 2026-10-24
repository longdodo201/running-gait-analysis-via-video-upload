"""Minimal offline Streamlit interface for one-video running analysis."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Callable
import uuid

from src.analysis_pipeline import AnalysisResult, analyze_video
from src.video_io import VideoLimits
from src.video_transcoding import VideoTranscodeError, transcode_overlay_to_h264
from src.visualization import ankle_trajectory_playback_html


OUTPUT_ROOT = Path("outputs") / "streamlit_runs"
MODEL_PATH = Path("models") / "pose_landmarker_full.task"
GROUND_TRUTH_PATH = Path("outputs") / "foot_strike_ground_truth.csv"
MAX_UPLOAD_BYTES = VideoLimits().max_bytes
RESULT_STATE_KEY = "running_gait_analysis_result"
PLAYBACK_VIDEO_STATE_KEY = "running_gait_analysis_h264_video"
PLAYBACK_ERROR_STATE_KEY = "running_gait_analysis_h264_error"


DASHBOARD_STYLES = """
<style>
:root { --dash-bg: #07121a; --dash-panel: #0d1c27; --dash-line: #203744;
        --dash-text: #e8f1f4; --dash-muted: #9bb0ba; --dash-cyan: #39d5e6;
        --dash-orange: #ff9d55; }
[data-testid="stAppViewContainer"] { background: radial-gradient(ellipse at 18% 0%, #102a35 0%, var(--dash-bg) 48%); color: var(--dash-text); }
[data-testid="stHeader"] { background: rgba(7, 18, 26, .88); }
.main .block-container { max-width: 1600px; padding-top: 1.4rem; padding-bottom: 3rem; }
h1 { font-size: clamp(2.2rem, 4vw, 3.5rem) !important; letter-spacing: -.045em; }
h2, h3 { letter-spacing: -.025em; }
[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(13, 28, 39, .78); border-color: var(--dash-line); border-radius: 16px; }
[data-testid="stMetricValue"] { color: var(--dash-text); font-size: 1.55rem; }
[data-testid="stCaptionContainer"] { color: var(--dash-muted); }
[data-testid="stVideo"] video { width: 100%; border-radius: 12px; background: #02070a; }
[data-testid="stButton"] button { border-radius: 10px; }
.dashboard-eyebrow { color: var(--dash-cyan); font-size: .72rem; font-weight: 750; letter-spacing: .16em; text-transform: uppercase; margin: 0 0 .45rem; }
.dashboard-eyebrow--orange { color: var(--dash-orange); }
.dashboard-intro { color: var(--dash-muted); margin-top: -.6rem; }
.dashboard-side-label { font-size: .76rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; margin: .2rem 0 .7rem; }
.dashboard-side-label--left { color: var(--dash-cyan); }
.dashboard-side-label--right { color: var(--dash-orange); }
.future-signal-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .8rem; }
.future-signal-card { height: 100%; box-sizing: border-box; padding: 1rem; border: 1px solid var(--dash-line); border-radius: 14px; background: linear-gradient(145deg, rgba(14, 32, 44, .96), rgba(8, 20, 28, .96)); }
.future-signal-card__label { color: var(--dash-muted); font-size: .7rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
.future-signal-card__value { color: var(--dash-text); font-size: 1.05rem; font-weight: 700; margin: .55rem 0 .35rem; }
.future-signal-card__note { color: var(--dash-muted); font-size: .82rem; line-height: 1.45; }
.dashboard-knee-chart svg { display: block; width: 100%; height: auto; max-width: 100%; }
@media (max-width: 760px) {
  .main .block-container { padding-left: 1rem; padding-right: 1rem; }
  .future-signal-grid { grid-template-columns: 1fr; }
  [data-testid="stHorizontalBlock"] { gap: .7rem; }
}
</style>
"""


def new_run_output_dir(output_root: str | Path, uploaded_name: str) -> Path:
    """Return a unique, not-yet-created directory for one analysis run."""
    root = Path(output_root).expanduser().resolve()
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(uploaded_name).stem).strip("-_") or "running-video"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    while True:
        destination = root / f"{stamp}-{stem}-{uuid.uuid4().hex[:8]}"
        if not destination.exists():
            return destination


def process_uploaded_video(
    uploaded_file: Any,
    *,
    output_root: str | Path = OUTPUT_ROOT,
    model_path: str | Path = MODEL_PATH,
    analyzer: Callable[..., AnalysisResult] = analyze_video,
    progress: Callable[[str], None] | None = None,
    ground_truth_path: str | Path | None = None,
) -> AnalysisResult:
    """Copy an uploaded MP4 to a temporary file and analyze it exactly once.

    The temporary upload is always removed; generated results remain under the
    ignored output directory and each invocation receives a fresh destination.
    """
    filename = str(getattr(uploaded_file, "name", ""))
    if Path(filename).suffix.lower() != ".mp4":
        raise ValueError("Please upload an MP4 video.")

    try:
        content = uploaded_file.getbuffer()
    except (AttributeError, TypeError):
        content = memoryview(uploaded_file.getvalue())
    if not content:
        raise ValueError("The uploaded video is empty.")
    if MAX_UPLOAD_BYTES is not None and len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("The video exceeds the 100 MiB application upload limit.")

    safe_filename = Path(filename).name
    with tempfile.TemporaryDirectory(prefix="running-gait-upload-") as temporary_dir:
        temporary_path = Path(temporary_dir) / safe_filename
        temporary_path.write_bytes(content)
        output_dir = new_run_output_dir(output_root, filename)
        analyzer_kwargs: dict[str, Any] = {"model_path": model_path, "progress": progress}
        selected_ground_truth = Path(ground_truth_path) if ground_truth_path is not None else None
        if selected_ground_truth is None and Path(filename).stem.lower() == "running_gait_sideview":
            if GROUND_TRUTH_PATH.is_file():
                selected_ground_truth = GROUND_TRUTH_PATH
        if selected_ground_truth is not None:
            analyzer_kwargs["ground_truth_path"] = selected_ground_truth
        return analyzer(temporary_path, output_dir, **analyzer_kwargs)


def _clear_previous_result() -> None:
    import streamlit as st

    st.session_state.pop(RESULT_STATE_KEY, None)
    st.session_state.pop(PLAYBACK_VIDEO_STATE_KEY, None)
    st.session_state.pop(PLAYBACK_ERROR_STATE_KEY, None)


def _read_overlay_video(st: Any, path: Path) -> bytes | None:
    """Read a completed overlay for playback and download, reporting file issues."""
    try:
        if not path.is_file():
            st.warning("The skeleton overlay video is missing. You can still download the other analysis files.")
            return None
        video_bytes = path.read_bytes()
    except OSError as exc:
        st.warning(f"The skeleton overlay video could not be read: {exc}")
        return None

    if not video_bytes:
        st.warning("The skeleton overlay video is empty, so it cannot be played or downloaded.")
        return None
    return video_bytes


def _analysis_error_message(error: Exception) -> str:
    """Make known overlay encoder failures actionable while preserving other details."""
    message = str(error)
    lowered = message.lower()
    if "mp4 writer" in lowered or "annotated mp4 was not produced" in lowered:
        return (
            "The skeleton overlay could not be encoded. This computer may not have a usable "
            "MP4V encoder; check OpenCV codec support and retry. "
            f"Details: {message}"
        )
    return f"Analysis could not be completed: {message}"


def _dashboard_svg(path: Path) -> str:
    """Adapt a generated plot SVG for a responsive dark dashboard display."""
    svg = path.read_text(encoding="utf-8")
    svg = svg.replace(
        '<svg xmlns=',
        '<svg style="width:100%;height:auto;max-width:100%;display:block" xmlns=',
        1,
    )
    svg = svg.replace('fill="white"', 'fill="#0d1c27"', 1)
    svg = svg.replace("fill:#222", "fill:#e8f1f4", 1)
    svg = svg.replace("stroke:#ddd", "stroke:#263c49", 1)
    svg = svg.replace("stroke:#555", "stroke:#69808b", 1)
    svg = svg.replace('stroke="#222"', 'stroke="#d2e0e5"')
    return svg


def _dashboard_knee_svg(path: Path) -> str:
    """Keep the knee-plot rendering helper explicit at its call site."""
    return _dashboard_svg(path)


def _render_future_signal_cards(st: Any, result: AnalysisResult) -> None:
    """Show inference metrics for every video and Ground Truth details when available."""
    cadence = getattr(result, "cadence_summary", None) or {}
    cadence_value = cadence.get("average_cadence_steps_per_minute")
    cadence_status = cadence.get("status", "insufficient_events")
    cadence_text = (
        f"{cadence_value:.1f} steps/min"
        if isinstance(cadence_value, (int, float)) and cadence_status == "ok"
        else "Insufficient events" if cadence_status == "insufficient_events"
        else "Unavailable"
    )
    duration = cadence.get("duration_seconds")
    duration_text = f"{duration:.2f} s" if isinstance(duration, (int, float)) else "duration unavailable"
    cadence_note = (
        f"{cadence.get('retained_event_count', 0)} of {cadence.get('candidate_event_count', 0)} "
        f"candidates retained · {duration_text} · exploratory estimate"
    )
    side_quality = {
        "left": getattr(result, "left_quality", None),
        "right": getattr(result, "right_quality", None),
    }
    knee_cards = []
    for side, label in (("left", "Left knee angle"), ("right", "Right knee angle")):
        quality = side_quality[side]
        count = getattr(quality, "usable_angle_count", 0)
        percentage = getattr(quality, "valid_frame_percentage", 0.0)
        angle = getattr(quality, "mean_knee_angle_deg", None)
        angle_text = f"{angle:.1f}° mean" if isinstance(angle, (int, float)) else "Insufficient data"
        knee_cards.append(
            f'<div class="future-signal-card"><div class="future-signal-card__label">{label} · all frames</div>'
            f'<div class="future-signal-card__value">{angle_text}</div>'
            f'<div class="future-signal-card__note">{count} valid frames · {percentage:.1f}% coverage; '
            "2D hip-knee-ankle angle, independent of Ground Truth and detected events.</div></div>"
        )
    st.markdown(
        '<div class="future-signal-grid">'
        f'<div class="future-signal-card"><div class="future-signal-card__label">Cadence · candidate estimate</div>'
        f'<div class="future-signal-card__value">{cadence_text}</div>'
        f'<div class="future-signal-card__note">{cadence_note}</div></div>'
        + "".join(knee_cards)
        + "</div>",
        unsafe_allow_html=True,
    )
    if cadence_status != "ok":
        st.warning(cadence.get("warning", "Cadence is unavailable because too few candidate events were detected."))
    for warning in getattr(result, "analysis_warnings", ()):
        st.warning(warning)
    st.caption(
        "Cadence counts retained heel-trajectory candidates divided by decoded video duration × 60; "
        "these candidates are exploratory and are not confirmed foot-strikes. Knee-angle means use "
        "all valid hip-knee-ankle frames; straight leg ≈ 180°."
    )

    metrics_path = getattr(result, "ground_truth_metrics_json", None)
    if metrics_path is not None and Path(metrics_path).is_file():
        try:
            metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
            cadence = metrics["cadence"]
            knee = metrics["knee_angle_at_ground_truth"]
            cadence_value = cadence.get("average_cadence_steps_per_minute")
            cadence_text = (
                f"{cadence_value:.1f} steps/min"
                if isinstance(cadence_value, (int, float)) else "Insufficient data"
            )
            left_knee, right_knee = knee["left"], knee["right"]
            left_angle = left_knee.get("mean_knee_angle_deg")
            right_angle = right_knee.get("mean_knee_angle_deg")
            valid_count = left_knee["valid_event_count"] + right_knee["valid_event_count"]
            total_count = left_knee["event_count"] + right_knee["event_count"]
            knee_text = (
                f"L {left_angle:.1f}° · R {right_angle:.1f}°"
                if isinstance(left_angle, (int, float)) and isinstance(right_angle, (int, float))
                else f"{valid_count}/{total_count} valid events"
            )
            st.markdown(
                f"""
                <div class="future-signal-grid">
                  <div class="future-signal-card">
                    <div class="future-signal-card__label">Cadence · Ground Truth</div>
                    <div class="future-signal-card__value">{cadence_text}</div>
                    <div class="future-signal-card__note">{cadence['step_count']} labeled steps / {cadence['clip_duration_seconds']:.2f} s analyzed.</div>
                  </div>
                  <div class="future-signal-card">
                    <div class="future-signal-card__label">Knee angle at foot-strike · Ground Truth</div>
                    <div class="future-signal-card__value">{knee_text}</div>
                    <div class="future-signal-card__note">{valid_count}/{total_count} events have usable hip, knee and ankle landmarks; straight leg ≈ 180°.</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(
                "Cadence is labeled-event count divided by full clip duration. Knee angles are 2D geometric estimates. "
                "Detector predictions are reported separately; heel candidates are not treated as confirmed contacts."
            )
            evaluation = metrics.get("evaluation", [])
            if evaluation:
                table_rows = [
                    "| Algorithm | Side | TP | FP | FN | Precision | Recall | F1 | MAE (frames) | MAE (ms) |",
                    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
                for row in evaluation:
                    def fmt(value: object) -> str:
                        if value is None:
                            return "—"
                        return f"{value:.3f}" if isinstance(value, float) else str(value)
                    table_rows.append(
                        f"| {row['algorithm']} | {row['side']} | {row['tp']} | {row['fp']} | {row['fn']} | "
                        f"{fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['f1'])} | "
                        f"{fmt(row['mean_absolute_error_frames'])} | {fmt(row['mean_absolute_error_ms'])} |"
                    )
                st.markdown("\n".join(table_rows))
                st.caption("Single-video descriptive results; they do not establish general detector accuracy.")
            cadence_svg = getattr(result, "cadence_svg", None)
            if cadence_svg is not None and Path(cadence_svg).is_file():
                with st.expander("Cadence over time"):
                    st.markdown(_dashboard_svg(Path(cadence_svg)), unsafe_allow_html=True)
            return
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.warning(f"Ground Truth metrics could not be displayed: {exc}")


def _show_result(
    st: Any,
    result: AnalysisResult,
    *,
    playback_video_path: str | Path | None = None,
    playback_error: str | None = None,
) -> None:
    browser_video_path = Path(playback_video_path) if playback_video_path is not None else None
    browser_video_bytes = _read_overlay_video(st, browser_video_path) if browser_video_path else None
    overlay_bytes = browser_video_bytes or _read_overlay_video(st, result.annotated_video)
    video_column, analysis_column = st.columns(
        [1.15, 0.95], gap="large", vertical_alignment="top", wrap=True
    )
    with video_column:
        with st.container(border=True):
            st.markdown('<div class="dashboard-eyebrow">Video review</div>', unsafe_allow_html=True)
            if playback_error:
                st.warning(
                    "Could not create the browser-compatible H.264 overlay. The original skeleton video "
                    "is preserved for download; browser playback of its MP4V codec may not be supported. "
                    f"Details: {playback_error}"
                )
            if overlay_bytes is not None:
                try:
                    st.video(overlay_bytes, format="video/mp4")
                except Exception as exc:
                    st.warning(
                        "Streamlit could not display the skeleton overlay video. Use the download below "
                        f"and open it in a local video player. Details: {exc}"
                    )
            st.caption(
                "Pose landmarks are estimates from a single-camera 2D view. Camera angle, perspective, "
                "occlusion, frame rate, and landmark jitter can affect the overlay."
            )
            st.caption(
                "The player uses a local H.264 conversion when supported; the original MP4V overlay "
                "remains available in downloads."
                if browser_video_bytes is not None
                else "If this browser cannot play MP4V, download the overlay and open it in a local video player."
            )

    with analysis_column:
        with st.container(border=True):
            st.markdown('<div class="dashboard-eyebrow">Angle analysis</div>', unsafe_allow_html=True)
            st.subheader("Knee angle over time")
            st.markdown(_dashboard_knee_svg(result.knee_angles_svg), unsafe_allow_html=True)
            st.caption("2D angle estimates in degrees. Missing or unusable samples remain visible as gaps.")

            st.markdown('<div class="dashboard-eyebrow">Data quality</div>', unsafe_allow_html=True)
            side_columns = st.columns(2, gap="small", wrap=True)
            for column, label, side_class, quality in (
                (side_columns[0], "Left knee", "left", result.left_quality),
                (side_columns[1], "Right knee", "right", result.right_quality),
            ):
                column.markdown(
                    f'<div class="dashboard-side-label dashboard-side-label--{side_class}">{label}</div>',
                    unsafe_allow_html=True,
                )
                column.metric("Usable frames", quality.usable_angle_count)
                column.metric("Valid coverage", f"{quality.valid_frame_percentage:.1f}%")
                column.metric("Missing / unusable", quality.missing_angle_count)
                column.caption(f"Large-change flags: {len(quality.large_change_flags)} (heuristic review cue)")
            st.caption("Coverage and flags describe the video data, not biomechanical accuracy.")

    st.markdown('<div class="dashboard-eyebrow dashboard-eyebrow--orange">More running signals</div>', unsafe_allow_html=True)
    _render_future_signal_cards(st, result)
    st.markdown('<div class="dashboard-eyebrow">Ankle path</div>', unsafe_allow_html=True)
    try:
        landmark_document = json.loads(result.landmarks_json.read_text(encoding="utf-8"))
        ankle_component = ankle_trajectory_playback_html(landmark_document)
        st.iframe(ankle_component, height="content")
        st.caption(
            "2D image coordinates in source-video pixels (not real-world distance). "
            "Image y increases downward. Samples require visibility and presence ≥ 0.5; "
            "missing, low-confidence, invalid, or out-of-frame samples are omitted and break the line."
        )
    except (OSError, ValueError, TypeError) as exc:
        st.warning(f"The ankle trajectory is unavailable because the landmark data could not be read: {exc}")
    st.warning(
        "Candidate cadence is an exploratory estimate from heel-trajectory events, not confirmed contact timing. "
        "Ground Truth evaluation remains separate; general knee-angle statistics use all valid frames."
    )
    with st.expander("Review exploratory heel-event signal"):
        st.markdown(_dashboard_svg(result.gait_events_svg), unsafe_allow_html=True)

    with st.expander("Download analysis files"):
        downloads = []
        if browser_video_bytes is not None:
            downloads.extend((
                ("Skeleton overlay (H.264 MP4)", browser_video_path, "video/mp4", browser_video_bytes),
                ("Original skeleton overlay (MP4V MP4)", result.annotated_video, "video/mp4", None),
            ))
        else:
            downloads.append(("Annotated video (MP4)", result.annotated_video, "video/mp4", overlay_bytes))
        downloads.extend((
            ("Landmarks (JSON)", result.landmarks_json, "application/json", None),
            ("Knee angles (CSV)", result.knee_angles_csv, "text/csv", None),
            ("Smoothed knee angles (CSV)", result.smoothed_knee_angles_csv, "text/csv", None),
            ("Knee-angle plot (SVG)", result.knee_angles_svg, "image/svg+xml", None),
            ("Exploratory heel-event candidates (CSV)", result.gait_events_csv, "text/csv", None),
            ("Exploratory event plot (SVG)", result.gait_events_svg, "image/svg+xml", None),
        ))
        extra_downloads = (
            ("Foot-strike evaluation summary (CSV)", getattr(result, "foot_strike_evaluation_csv", None), "text/csv"),
            ("Foot-strike event matches (CSV)", getattr(result, "foot_strike_matches_csv", None), "text/csv"),
            ("Foot-strike evaluation (JSON)", getattr(result, "foot_strike_evaluation_json", None), "application/json"),
            ("Knee angles at Ground Truth events (CSV)", getattr(result, "knee_angles_at_ground_truth_csv", None), "text/csv"),
            ("Cadence series (CSV)", getattr(result, "cadence_csv", None), "text/csv"),
            ("Cadence chart (SVG)", getattr(result, "cadence_svg", None), "image/svg+xml"),
            ("Ground Truth metrics summary (JSON)", getattr(result, "ground_truth_metrics_json", None), "application/json"),
            ("Ground Truth evaluation report (Markdown)", getattr(result, "ground_truth_report", None), "text/markdown"),
        )
        downloads.extend(
            (label, Path(path), mime, None)
            for label, path, mime in extra_downloads
            if path is not None
        )
        diagnostic_downloads = (
            ("Candidate cadence windows (CSV)", getattr(result, "candidate_cadence_csv", None), "text/csv"),
            ("Candidate cadence summary (JSON)", getattr(result, "candidate_cadence_summary_json", None), "application/json"),
            ("Analysis diagnostics (JSON)", getattr(result, "diagnostics_json", None), "application/json"),
        )
        downloads.extend(
            (label, Path(path), mime, None)
            for label, path, mime in diagnostic_downloads
            if path is not None
        )
        for label, path, mime, cached_data in downloads:
            try:
                data = cached_data if cached_data is not None else path.read_bytes()
                if data:
                    st.download_button(label, data, file_name=path.name, mime=mime)
            except OSError as exc:
                st.warning(f"{label} is unavailable because its file could not be read: {exc}")


def render_app(
    st: Any,
    *,
    output_root: str | Path = OUTPUT_ROOT,
    model_path: str | Path = MODEL_PATH,
    analyzer: Callable[..., AnalysisResult] = analyze_video,
) -> None:
    """Render the app against a Streamlit-compatible object (injectable in tests)."""
    st.markdown(DASHBOARD_STYLES, unsafe_allow_html=True)
    st.markdown('<div class="dashboard-eyebrow">Offline motion review</div>', unsafe_allow_html=True)
    st.title("Running gait analysis")
    st.markdown(
        '<p class="dashboard-intro">Upload a side-view running video for a private, local post-run analysis.</p>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown('<div class="dashboard-eyebrow">Start a new analysis</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Running video (MP4, up to 100 MiB and 60 seconds)",
            type=["mp4"],
            key="video_upload",
            on_change=_clear_previous_result,
        )
        analyze_clicked = uploaded_file is not None and st.button("Analyze video", type="primary")

    if analyze_clicked:
        try:
            with st.status("Analyzing video locally…", expanded=True) as status:
                last_reported_frame = 0

                def report_progress(message: str) -> None:
                    nonlocal last_reported_frame
                    match = re.match(r"Processed frame (\d+);", message)
                    if match:
                        frame = int(match.group(1))
                        if frame == 1 or frame - last_reported_frame >= 30:
                            status.write(f"Processed {frame} frames…")
                            last_reported_frame = frame
                    else:
                        status.write(message)

                result = process_uploaded_video(
                    uploaded_file,
                    output_root=output_root,
                    model_path=model_path,
                    analyzer=analyzer,
                    progress=report_progress,
                )
                status.write("Preparing a browser-compatible H.264 overlay from the generated skeleton video…")
                try:
                    playback_video_path = transcode_overlay_to_h264(result.annotated_video)
                    playback_error = None
                    status.write("H.264 overlay is ready for playback.")
                except VideoTranscodeError as exc:
                    playback_video_path = None
                    playback_error = str(exc)
                    status.write("H.264 conversion failed; the original overlay is preserved.")
            st.session_state[RESULT_STATE_KEY] = result
            st.session_state[PLAYBACK_VIDEO_STATE_KEY] = playback_video_path
            st.session_state[PLAYBACK_ERROR_STATE_KEY] = playback_error
            st.success("Analysis complete.")
        except Exception as exc:
            st.error(_analysis_error_message(exc))

    result = st.session_state.get(RESULT_STATE_KEY)
    if result is not None:
        _show_result(
            st,
            result,
            playback_video_path=st.session_state.get(PLAYBACK_VIDEO_STATE_KEY),
            playback_error=st.session_state.get(PLAYBACK_ERROR_STATE_KEY),
        )


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Running gait analysis", layout="wide")
    render_app(st)


if __name__ == "__main__":
    main()
