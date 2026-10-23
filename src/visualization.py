"""Dependency-free SVG plots for frame-level knee-angle series."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

from .gait_events import CandidateEvent, HeelSignalSample
from .signal_processing import SMOOTHED_ANGLE_COLUMNS, _finite_number


def dashboard_svg_for_display(path: str | Path) -> str:
    """Read an SVG artifact and adapt its styling for the dark dashboard."""
    svg = Path(path).read_text(encoding="utf-8")
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


def _landmark_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _ankle_point(
    frame: Mapping[str, Any], side: str, visibility_threshold: float, presence_threshold: float
) -> tuple[float, float, int, float | None] | None:
    """Return an in-frame ankle in source pixels after confidence filtering."""
    frame_index = frame.get("frame_index")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int):
        return None
    poses = frame.get("poses")
    if not isinstance(poses, list) or not poses or not isinstance(poses[0], Mapping):
        return None
    landmarks = poses[0].get("landmarks")
    if not isinstance(landmarks, Mapping):
        return None
    landmark = landmarks.get(f"{side}_ankle")
    if not isinstance(landmark, Mapping):
        return None
    x, y = _landmark_number(landmark.get("x")), _landmark_number(landmark.get("y"))
    visibility = _landmark_number(landmark.get("visibility"))
    presence = _landmark_number(landmark.get("presence"))
    timestamp = _landmark_number(frame.get("timestamp_seconds"))
    if (
        x is None or y is None or visibility is None or presence is None
        or not 0 <= x <= 1 or not 0 <= y <= 1
        or not 0 <= visibility <= 1 or not 0 <= presence <= 1
        or visibility < visibility_threshold or presence < presence_threshold
    ):
        return None
    return x, y, frame_index, timestamp


def ankle_trajectory_svg(
    document: Mapping[str, Any],
    visibility_threshold: float = 0.5,
    presence_threshold: float = 0.5,
) -> str:
    """Render confidence-filtered ankle trajectories in source-image pixel coordinates.

    MediaPipe normalized x/y values are multiplied by the source width/height;
    no per-series or per-frame normalization is applied. The plot preserves the
    source image aspect ratio and keeps image y increasing downward. Only
    adjacent valid frames are joined, so every missing, low-confidence, or
    out-of-frame sample breaks the line. Samples require both confidence fields.
    """
    if not isinstance(document, Mapping):
        raise ValueError("Landmark document must be an object.")
    video = document.get("video")
    frames = document.get("frames")
    if not isinstance(video, Mapping) or not isinstance(frames, list):
        raise ValueError("Landmark document must contain video metadata and a frames list.")
    width = _landmark_number(video.get("width"))
    height = _landmark_number(video.get("height"))
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("Landmark document must contain positive video width and height.")
    thresholds = (_landmark_number(visibility_threshold), _landmark_number(presence_threshold))
    if any(value is None or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Confidence thresholds must be finite values from 0 to 1.")
    visibility_threshold, presence_threshold = thresholds  # type: ignore[assignment]

    view_width, view_height = 780, 900
    plot_left, plot_top, plot_width, plot_height = 90.0, 70.0, 600.0, 760.0
    colors = {"left": "#39d5e6", "right": "#ff9d55"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;max-width:100%;display:block" width="{view_width}" height="{view_height}" viewBox="0 0 {view_width} {view_height}" role="img" aria-labelledby="ankle-title ankle-desc">',
        '<title id="ankle-title">Left and right ankle trajectory in image coordinates</title>',
        '<desc id="ankle-desc">Two-dimensional ankle positions from pose landmarks. Coordinates are source-image pixels; image y increases downward. Low-confidence, missing, invalid, and out-of-frame samples are omitted and break trajectory lines.</desc>',
        '<rect width="100%" height="100%" rx="14" fill="#0d1c27"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#e8f1f4}.grid{stroke:#263c49;stroke-width:1}.axis{stroke:#69808b;stroke-width:1}.trajectory{fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}</style>',
        f'<text x="{view_width / 2}" y="30" text-anchor="middle" font-size="19" font-weight="700">2D ankle trajectory in image coordinates</text>',
    ]
    for tick in range(5):
        x_value = width * tick / 4
        x = plot_left + plot_width * tick / 4
        y_value = height * tick / 4
        y = plot_top + plot_height * tick / 4
        parts.extend((
            f'<line class="grid" x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_top + plot_height}"/>',
            f'<text x="{x:.2f}" y="{plot_top + plot_height + 22}" text-anchor="middle" font-size="12">{x_value:.0f}</text>',
            f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" y2="{y:.2f}"/>',
            f'<text x="{plot_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{y_value:.0f}</text>',
        ))
    parts.extend((
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="none" class="axis"/>',
        f'<text x="{plot_left + plot_width / 2}" y="{plot_top + plot_height + 48}" text-anchor="middle" font-size="14">Image x (pixels)</text>',
        f'<text x="24" y="{plot_top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 24 {plot_top + plot_height / 2})">Image y (pixels; increases downward)</text>',
    ))

    for side in ("left", "right"):
        valid_points = [_ankle_point(frame, side, visibility_threshold, presence_threshold) for frame in frames]
        count = sum(point is not None for point in valid_points)
        commands: list[str] = []
        previous_index: int | None = None
        circles: list[str] = []
        for point in valid_points:
            if point is None:
                previous_index = None
                continue
            x_normalized, y_normalized, frame_index, timestamp = point
            x_px, y_px = x_normalized * width, y_normalized * height
            x = plot_left + x_px / width * plot_width
            y = plot_top + y_px / height * plot_height
            command = "L" if previous_index is not None and frame_index == previous_index + 1 else "M"
            commands.append(f"{command} {x:.2f} {y:.2f}")
            previous_index = frame_index
            time_text = f", {timestamp:.3f} s" if timestamp is not None else ""
            circles.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.2" fill="{colors[side]}" opacity=".72"><title>{side.title()} ankle: frame {frame_index}{time_text}; image ({x_px:.1f}, {y_px:.1f}) px</title></circle>'
            )
        parts.append(
            f'<path class="trajectory" data-side="{side}" d="{" ".join(commands)}" stroke="{colors[side]}"/>'
        )
        parts.extend(circles)
        proportion = 100 * count / len(frames) if frames else 0.0
        legend_y = 860 if side == "left" else 882
        parts.append(
            f'<text x="{plot_left}" y="{legend_y}" font-size="13" fill="{colors[side]}"><tspan fill="{colors[side]}">● {side.title()} ankle</tspan><tspan fill="#9bb0ba"> · {count}/{len(frames)} valid in-frame samples ({proportion:.1f}%)</tspan></text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def ankle_trajectory_timeline(
    document: Mapping[str, Any],
    visibility_threshold: float = 0.5,
    presence_threshold: float = 0.5,
) -> dict[str, Any]:
    """Prepare a compact, confidence-filtered trajectory timeline for playback."""
    if not isinstance(document, Mapping):
        raise ValueError("Landmark document must be an object.")
    video = document.get("video")
    frames = document.get("frames")
    if not isinstance(video, Mapping) or not isinstance(frames, list):
        raise ValueError("Landmark document must contain video metadata and a frames list.")
    width, height = _landmark_number(video.get("width")), _landmark_number(video.get("height"))
    fps = _landmark_number(video.get("fps"))
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("Landmark document must contain positive video width and height.")
    if fps is not None and fps <= 0:
        fps = None
    thresholds = (_landmark_number(visibility_threshold), _landmark_number(presence_threshold))
    if any(value is None or not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Confidence thresholds must be finite values from 0 to 1.")
    visibility_threshold, presence_threshold = thresholds  # type: ignore[assignment]

    timeline_frames: list[dict[str, Any]] = []
    previous_time = -math.inf
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("Every landmark frame must be an object.")
        frame_index = frame.get("frame_index")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int):
            raise ValueError("Every landmark frame must have an integer frame_index.")
        timestamp = _landmark_number(frame.get("timestamp_seconds"))
        if timestamp is None and fps is not None:
            timestamp = frame_index / fps
        if timestamp is None:
            raise ValueError("Playback synchronization needs frame timestamps or a valid video FPS.")
        if timestamp < previous_time:
            raise ValueError("Landmark frame timestamps must be non-decreasing for playback synchronization.")
        previous_time = timestamp

        row: dict[str, Any] = {"frame_index": frame_index, "time_seconds": timestamp}
        for side in ("left", "right"):
            point = _ankle_point(frame, side, visibility_threshold, presence_threshold)
            row[side] = [point[0] * width, point[1] * height] if point is not None else None
        timeline_frames.append(row)
    return {"width": width, "height": height, "fps": fps, "frames": timeline_frames}


def frame_index_at_time(frames: Sequence[Mapping[str, Any]], time_seconds: float) -> int:
    """Return the last source frame at or before playback time, or -1 before start."""
    time_value = _landmark_number(time_seconds)
    if time_value is None:
        return -1
    timestamps = [_landmark_number(frame.get("time_seconds")) for frame in frames]
    if any(timestamp is None for timestamp in timestamps):
        raise ValueError("Every synchronized frame must have a finite timestamp.")
    numeric_times = [timestamp for timestamp in timestamps if timestamp is not None]
    if any(right < left for left, right in zip(numeric_times, numeric_times[1:])):
        raise ValueError("Frame timestamps must be non-decreasing.")
    position = bisect_right(numeric_times, time_value) - 1
    return int(frames[position]["frame_index"]) if position >= 0 else -1


def cumulative_trajectory_path(
    frames: Sequence[Mapping[str, Any]], side: str, through_frame_index: int,
    width: float, height: float,
) -> str:
    """Build one side's path through a frame, leaving gaps where samples are invalid."""
    if side not in {"left", "right"}:
        raise ValueError("Side must be 'left' or 'right'.")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    plot_left, plot_top, plot_width, plot_height = 90.0, 70.0, 600.0, 760.0
    commands: list[str] = []
    previous_index: int | None = None
    for frame in frames:
        frame_index = frame.get("frame_index")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index > through_frame_index:
            break
        point = frame.get(side)
        if (
            not isinstance(point, (tuple, list)) or len(point) != 2
            or _landmark_number(point[0]) is None or _landmark_number(point[1]) is None
        ):
            previous_index = None
            continue
        x_px, y_px = float(point[0]), float(point[1])
        if not 0 <= x_px <= width or not 0 <= y_px <= height:
            previous_index = None
            continue
        x = plot_left + x_px / width * plot_width
        y = plot_top + y_px / height * plot_height
        command = "L" if previous_index is not None and frame_index == previous_index + 1 else "M"
        commands.append(f"{command} {x:.2f} {y:.2f}")
        previous_index = frame_index
    return " ".join(commands)


def ankle_trajectory_playback_html(
    document: Mapping[str, Any],
    visibility_threshold: float = 0.5,
    presence_threshold: float = 0.5,
) -> str:
    """Create a small HTML chart that follows the existing Streamlit video player.

    The component listens to the parent page's native Streamlit video element.
    It only updates SVG paths in the browser; no Streamlit rerun or inference is
    triggered by playback, pause, or seeking.
    """
    timeline = ankle_trajectory_timeline(document, visibility_threshold, presence_threshold)
    data = json.dumps(timeline, separators=(",", ":"), allow_nan=False)
    width, height = 780, 940
    plot_left, plot_top, plot_width, plot_height = 90.0, 70.0, 600.0, 760.0
    source_width, source_height = timeline["width"], timeline["height"]
    parts = [
        '<!doctype html><html><head><meta charset="utf-8"><style>',
        'html,body{margin:0;background:#07121a;color:#e8f1f4;font-family:Arial,sans-serif}svg{display:block;width:100%;height:auto;max-width:900px;margin:auto;background:#0d1c27;border-radius:14px}text{fill:#e8f1f4}.grid{stroke:#263c49;stroke-width:1}.axis{stroke:#69808b;stroke-width:1}.trajectory{fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}.status{max-width:900px;margin:.45rem auto 0;color:#9bb0ba;font-size:.82rem}',
        '</style></head><body>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="ankle-title ankle-desc">',
        '<title id="ankle-title">Synchronized 2D ankle trajectory in image coordinates</title>',
        '<desc id="ankle-desc">The path grows with video playback time. Image coordinates are source-video pixels, and image y increases downward. Missing or low-confidence samples create gaps.</desc>',
        '<rect width="100%" height="100%" rx="14" fill="#0d1c27"/>',
        '<style>text{font-family:Arial,sans-serif}.trajectory{fill:none;stroke-width:3;stroke-linecap:round;stroke-linejoin:round}</style>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="19" font-weight="700">2D ankle trajectory in image coordinates</text>',
    ]
    for tick in range(5):
        x, y = plot_left + plot_width * tick / 4, plot_top + plot_height * tick / 4
        parts.extend((
            f'<line class="grid" x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_top + plot_height}"/>',
            f'<text x="{x:.2f}" y="{plot_top + plot_height + 22}" text-anchor="middle" font-size="12">{source_width * tick / 4:.0f}</text>',
            f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" y2="{y:.2f}"/>',
            f'<text x="{plot_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{source_height * tick / 4:.0f}</text>',
        ))
    parts.extend((
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_width}" height="{plot_height}" fill="none" class="axis"/>',
        f'<text x="{plot_left + plot_width / 2}" y="{plot_top + plot_height + 48}" text-anchor="middle" font-size="14">Image x (pixels)</text>',
        f'<text x="24" y="{plot_top + plot_height / 2}" text-anchor="middle" font-size="14" transform="rotate(-90 24 {plot_top + plot_height / 2})">Image y (pixels; increases downward)</text>',
        '<path id="left-ankle-path" class="trajectory" stroke="#39d5e6"/>',
        '<path id="right-ankle-path" class="trajectory" stroke="#ff9d55"/>',
    ))
    for side, color, y in (("left", "#39d5e6", 905), ("right", "#ff9d55", 927)):
        count = sum(frame[side] is not None for frame in timeline["frames"])
        total = len(timeline["frames"])
        percent = 100 * count / total if total else 0.0
        parts.append(
            f'<text x="{plot_left}" y="{y}" font-size="15"><tspan fill="{color}">● {side.title()} ankle</tspan><tspan fill="#9bb0ba"> · {count}/{total} valid in-frame samples ({percent:.1f}%)</tspan></text>'
        )
    parts.extend((
        '</svg><div id="playback-status" class="status" role="status">Waiting for video player…</div>',
        f'<script>const timeline={data};',
        '''
const times=timeline.frames.map(frame=>frame.time_seconds);
function frameAtTime(time){let low=0,high=times.length;while(low<high){const mid=(low+high)>>1;if(times[mid]<=time)low=mid+1;else high=mid;}return low-1;}
function pathFor(side,target){let d=[],previous=null;for(let i=0;i<=target;i++){const frame=timeline.frames[i],point=frame[side];if(!point){previous=null;continue;}const index=frame.frame_index;if(point[0]<0||point[0]>timeline.width||point[1]<0||point[1]>timeline.height){previous=null;continue;}const x=90+point[0]/timeline.width*600,y=70+point[1]/timeline.height*760,command=previous===index-1?'L':'M';d.push(command+' '+x.toFixed(2)+' '+y.toFixed(2));previous=index;}return d.join(' ');}
function render(time){const target=frameAtTime(time);document.getElementById('left-ankle-path').setAttribute('d',pathFor('left',target));document.getElementById('right-ankle-path').setAttribute('d',pathFor('right',target));const frame=target>=0?timeline.frames[target]:null;document.getElementById('playback-status').textContent=frame?'Showing trajectory through frame '+frame.frame_index+' at '+time.toFixed(2)+' s.':'Video is before the first analyzed frame.';}
function bindVideo(){let video;try{const parentDocument=window.parent.document;video=parentDocument.querySelector('[data-testid="stVideo"] video');if(!video){video=Array.from(parentDocument.querySelectorAll('video')).find(candidate=>candidate.getBoundingClientRect().width>0)||null;}}catch(error){document.getElementById('playback-status').textContent='Could not connect to the video player in this browser.';return false;}if(!video)return false;const update=()=>render(video.currentTime||0);['timeupdate','seeking','seeked','play','pause','loadedmetadata','durationchange'].forEach(name=>video.addEventListener(name,update));video.addEventListener('error',()=>{document.getElementById('playback-status').textContent='Video playback failed; the ankle plot is held at the last available time.';});update();document.getElementById('playback-status').textContent='Synced to video playback. Use the video controls to play, pause, or seek.';return true;}
render(0);let tries=0;const timer=setInterval(()=>{tries++;if(bindVideo()){clearInterval(timer);}else if(tries>=75){clearInterval(timer);render(times.length?times[times.length-1]:0);document.getElementById('playback-status').textContent='Video player was not available; showing the full ankle trajectory.';}},200);
''',
        '</script></body></html>',
    ))
    return "".join(parts)


def _series_value(row: Mapping[str, Any], column: str) -> float | None:
    angle = _finite_number(row.get(column))
    return angle if angle is not None and 0 <= angle <= 180 else None


def _path_data(
    rows: Sequence[Mapping[str, Any]], x_values: Sequence[float | None], column: str,
    plot_left: float, plot_top: float, plot_width: float, plot_height: float,
    time_min: float, time_max: float,
) -> str:
    commands: list[str] = []
    drawing = False
    for row, timestamp in zip(rows, x_values):
        value = _series_value(row, column) if timestamp is not None else None
        if value is None or timestamp is None:
            drawing = False
            continue
        x = plot_left + (timestamp - time_min) / (time_max - time_min) * plot_width
        y = plot_top + (180 - value) / 195 * plot_height
        commands.append(f"{'L' if drawing else 'M'} {x:.2f} {y:.2f}")
        drawing = True
    return " ".join(commands)


def write_knee_angle_plot_svg(
    rows: Sequence[Mapping[str, Any]], destination: str | Path
) -> Path:
    """Write a two-panel raw/smoothed plot; missing observations break lines."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamps = [_finite_number(row.get("timestamp_seconds")) for row in rows]
    if any(value is not None for value in timestamps):
        x_values = timestamps
        x_label = "Time (seconds)"
    else:
        x_values = [_finite_number(row.get("frame_index")) for row in rows]
        x_label = "Frame index (timestamps unavailable)"
    available_times = [value for value in x_values if value is not None]
    time_min = min(available_times) if available_times else 0.0
    time_max = max(available_times) if available_times else 1.0
    if time_min == time_max:
        time_min -= 0.5
        time_max += 0.5

    width, height = 1120, 760
    left, plot_width, plot_height = 90.0, 990.0, 230.0
    panels = (("Left knee", "left"), ("Right knee", "right"))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title description">',
        '<title id="title">Raw and smoothed knee angles over time</title>',
        '<desc id="description">Two time-series panels. Missing samples are shown as line gaps and crosses near the bottom of each panel.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#555;stroke-width:1}.raw{fill:none;stroke-width:1.5}.smooth{fill:none;stroke-width:2.4;stroke-dasharray:7 4}</style>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="19">Knee angle over time (raw and smoothed)</text>',
    ]

    for panel_index, (title, side) in enumerate(panels):
        side_color = "#39d5e6" if side == "left" else "#ff9d55"
        smooth_color = "#9cecf3" if side == "left" else "#ffd0a8"
        top = 60.0 + panel_index * 350.0
        plot_top = top + 36.0
        plot_bottom = plot_top + plot_height
        parts.append(f'<text x="{left}" y="{top + 20}" font-size="16" font-weight="bold">{title}</text>')

        for tick in (0, 45, 90, 135, 180):
            y = plot_top + (180 - tick) / 195 * plot_height
            parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
            parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12">{tick}</text>')
        parts.append(f'<line class="axis" x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_bottom}"/>')
        parts.append(f'<line class="axis" x1="{left}" y1="{plot_bottom}" x2="{left + plot_width}" y2="{plot_bottom}"/>')

        tick_count = 5
        for tick_index in range(tick_count + 1):
            value = time_min + (time_max - time_min) * tick_index / tick_count
            x = left + plot_width * tick_index / tick_count
            parts.append(f'<line class="axis" x1="{x:.2f}" y1="{plot_bottom}" x2="{x:.2f}" y2="{plot_bottom + 5}"/>')
            parts.append(f'<text x="{x:.2f}" y="{plot_bottom + 20}" text-anchor="middle" font-size="12">{value:.2f}</text>')

        raw_column = f"{side}_knee_angle_deg"
        smooth_column = SMOOTHED_ANGLE_COLUMNS[0 if side == "left" else 1]
        raw_path = _path_data(rows, x_values, raw_column, left, plot_top, plot_width, plot_height, time_min, time_max)
        smooth_path = _path_data(rows, x_values, smooth_column, left, plot_top, plot_width, plot_height, time_min, time_max)
        parts.append(f'<path class="raw" stroke="{side_color}" d="{raw_path}"/>')
        parts.append(f'<path class="smooth" stroke="{smooth_color}" d="{smooth_path}"/>')

        missing_x = [
            left + (timestamp - time_min) / (time_max - time_min) * plot_width
            for row, timestamp in zip(rows, x_values)
            if timestamp is not None and _series_value(row, raw_column) is None
        ]
        for x in missing_x:
            marker_y = plot_bottom - 4
            parts.append(f'<path d="M {x - 4:.2f} {marker_y - 4:.2f} l 8 8 m 0 -8 l -8 8" stroke="#222" stroke-width="1.5"/>')

        legend_y = top + 20
        legend_x = left + plot_width - 220
        parts.append(f'<line x1="{legend_x}" y1="{legend_y - 5}" x2="{legend_x + 25}" y2="{legend_y - 5}" stroke="{side_color}" stroke-width="2"/>')
        parts.append(f'<text x="{legend_x + 31}" y="{legend_y}" font-size="12">Raw</text>')
        parts.append(f'<line x1="{legend_x + 90}" y1="{legend_y - 5}" x2="{legend_x + 115}" y2="{legend_y - 5}" stroke="{smooth_color}" stroke-width="2.4" stroke-dasharray="7 4"/>')
        parts.append(f'<text x="{legend_x + 121}" y="{legend_y}" font-size="12">Smoothed</text>')

        if panel_index == 0:
            parts.append(f'<text transform="translate(22 {plot_top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-size="13">Knee angle (degrees)</text>')
        if panel_index == 1:
            parts.append(f'<text x="{left + plot_width / 2}" y="{plot_bottom + 42}" text-anchor="middle" font-size="13">{x_label}</text>')
            parts.append(f'<text x="{left}" y="{height - 7}" font-size="11">Crosses mark missing raw angles; gaps are not interpolated.</text>')

    parts.append('</svg>')
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return output


def _event_signal_path(
    samples: Sequence[HeelSignalSample],
    x_values: Sequence[float | None],
    attribute: str,
    plot_left: float,
    plot_top: float,
    plot_width: float,
    plot_height: float,
    time_min: float,
    time_max: float,
    value_min: float,
    value_max: float,
) -> str:
    commands: list[str] = []
    drawing = False
    for sample, timestamp in zip(samples, x_values):
        value = _finite_number(getattr(sample, attribute)) if timestamp is not None else None
        if value is None or timestamp is None:
            drawing = False
            continue
        x = plot_left + (timestamp - time_min) / (time_max - time_min) * plot_width
        # Screen y grows downward, so larger heel image-y is drawn lower.
        y = plot_top + (value_max - value) / (value_max - value_min) * plot_height
        commands.append(f"{'L' if drawing else 'M'} {x:.2f} {y:.2f}")
        drawing = True
    return " ".join(commands)


def _mapping_signal_path(
    rows: Sequence[Mapping[str, Any]],
    x_values: Sequence[float | None],
    column: str,
    plot_left: float,
    plot_top: float,
    plot_width: float,
    plot_height: float,
    time_min: float,
    time_max: float,
    value_min: float,
    value_max: float,
) -> str:
    commands: list[str] = []
    drawing = False
    for row, timestamp in zip(rows, x_values):
        value = _series_value(row, column) if timestamp is not None else None
        if value is None or timestamp is None:
            drawing = False
            continue
        x = plot_left + (timestamp - time_min) / (time_max - time_min) * plot_width
        y = plot_top + (value_max - value) / (value_max - value_min) * plot_height
        commands.append(f"{'L' if drawing else 'M'} {x:.2f} {y:.2f}")
        drawing = True
    return " ".join(commands)


def write_gait_event_diagnostic_svg(
    samples: Sequence[HeelSignalSample],
    events: Sequence[CandidateEvent],
    knee_rows: Sequence[Mapping[str, Any]],
    destination: str | Path,
) -> Path:
    """Plot heel signals and candidate marks, plus Phase 2.3 angles for context."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_times = [sample.timestamp_seconds for sample in samples]
    if any(value is not None for value in source_times):
        x_values = source_times
        x_label = "Time (seconds)"
    else:
        x_values = [float(sample.frame_index) if sample.frame_index is not None else None for sample in samples]
        x_label = "Frame index (timestamps unavailable)"
    available_times = [value for value in x_values if value is not None]
    time_min = min(available_times) if available_times else 0.0
    time_max = max(available_times) if available_times else 1.0
    if time_min == time_max:
        time_min -= 0.5
        time_max += 0.5

    width, height = 1180, 960
    plot_left, plot_width, plot_height = 100.0, 1040.0, 190.0
    panel_tops = (55.0, 320.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="event-title event-desc">',
        '<title id="event-title">Exploratory heel-trajectory event candidates</title>',
        '<desc id="event-desc">Separate left and right heel image-y signals with median-smoothed traces and detected vertical low-point candidates; a lower panel shows knee angles for context only.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd;stroke-width:1}.axis{stroke:#555;stroke-width:1}.raw{fill:none;stroke-width:1;opacity:.7}.smooth{fill:none;stroke-width:2}.context{fill:none;stroke-width:1.5}</style>',
        f'<text x="{width / 2}" y="26" text-anchor="middle" font-size="19">Exploratory heel vertical low-point candidates</text>',
    ]

    for side_index, side in enumerate(("left", "right")):
        side_raw_color = "#39d5e6" if side == "left" else "#ff9d55"
        side_smooth_color = "#9cecf3" if side == "left" else "#ffd0a8"
        top = panel_tops[side_index]
        plot_top = top + 30
        plot_bottom = plot_top + plot_height
        raw_attribute = f"{side}_heel_y_px"
        smooth_attribute = f"{side}_heel_y_smoothed_px"
        values = [
            _finite_number(getattr(sample, raw_attribute)) for sample in samples
        ]
        visible_values = [value for value in values if value is not None]
        if visible_values:
            value_min = min(visible_values)
            value_max = max(visible_values)
            padding = max(8.0, (value_max - value_min) * 0.08)
            value_min -= padding
            value_max += padding
        else:
            value_min, value_max = 0.0, 1.0
        if value_min == value_max:
            value_min -= 0.5
            value_max += 0.5

        parts.append(f'<text x="{plot_left}" y="{top + 18}" font-size="15" font-weight="bold">{side.title()} heel y (event signal; larger image-y is lower)</text>')
        for tick_index in range(5):
            value = value_min + (value_max - value_min) * tick_index / 4
            y = plot_top + plot_height * tick_index / 4
            parts.append(f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" y2="{y:.2f}"/>')
            parts.append(f'<text x="{plot_left - 9}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{value:.0f}</text>')
        parts.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>')
        parts.append(f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_left + plot_width}" y2="{plot_bottom}"/>')
        parts.append(f'<text transform="translate(22 {plot_top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-size="12">Heel image y (pixels)</text>')

        raw_path = _event_signal_path(samples, x_values, raw_attribute, plot_left, plot_top, plot_width, plot_height, time_min, time_max, value_min, value_max)
        smooth_path = _event_signal_path(samples, x_values, smooth_attribute, plot_left, plot_top, plot_width, plot_height, time_min, time_max, value_min, value_max)
        parts.append(f'<path class="raw" stroke="{side_raw_color}" d="{raw_path}"/>')
        parts.append(f'<path class="smooth" stroke="{side_smooth_color}" d="{smooth_path}"/>')

        for event in events:
            if event.side != side or event.timestamp_seconds is None:
                continue
            x = plot_left + (event.timestamp_seconds - time_min) / (time_max - time_min) * plot_width
            y = plot_top + (value_max - event.signal_value_px) / (value_max - value_min) * plot_height
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{side_raw_color}" stroke="white" stroke-width="1"><title>{side} candidate frame {event.frame_index}, {event.timestamp_seconds:.3f}s, prominence {event.prominence_px:.1f}px</title></circle>')

        legend_y = top + 18
        legend_x = plot_left + plot_width - 330
        parts.append(f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 22}" y2="{legend_y - 4}" stroke="{side_raw_color}" stroke-width="1.5"/><text x="{legend_x + 28}" y="{legend_y}" font-size="11">Raw</text>')
        parts.append(f'<line x1="{legend_x + 80}" y1="{legend_y - 4}" x2="{legend_x + 102}" y2="{legend_y - 4}" stroke="{side_smooth_color}" stroke-width="2"/><text x="{legend_x + 108}" y="{legend_y}" font-size="11">Median-smoothed</text>')
        parts.append(f'<circle cx="{legend_x + 240}" cy="{legend_y - 4}" r="4" fill="{side_raw_color}"/><text x="{legend_x + 250}" y="{legend_y}" font-size="11">Candidate</text>')

    # Plot the Phase 2.3 knee series separately as context; it is not an event feature.
    top = 585.0
    plot_top = top + 30
    plot_bottom = plot_top + 180
    parts.append(f'<text x="{plot_left}" y="{top + 18}" font-size="15" font-weight="bold">Phase 2.3 knee-angle context (not used for event detection)</text>')
    for tick in (0, 45, 90, 135, 180):
        y = plot_top + (180 - tick) / 180 * (plot_bottom - plot_top)
        parts.append(f'<line class="grid" x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" y2="{y:.2f}"/>')
        parts.append(f'<text x="{plot_left - 9}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{tick}</text>')
    parts.append(f'<line class="axis" x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}"/>')
    parts.append(f'<line class="axis" x1="{plot_left}" y1="{plot_bottom}" x2="{plot_left + plot_width}" y2="{plot_bottom}"/>')
    parts.append(f'<text transform="translate(22 {plot_top + 90}) rotate(-90)" text-anchor="middle" font-size="12">Knee angle (degrees)</text>')

    knee_series = (
        ("left_knee_angle_deg", "#39d5e6", "Raw left"),
        (SMOOTHED_ANGLE_COLUMNS[0], "#9cecf3", "Smoothed left"),
        ("right_knee_angle_deg", "#ff9d55", "Raw right"),
        (SMOOTHED_ANGLE_COLUMNS[1], "#ffd0a8", "Smoothed right"),
    )
    for column, color, label in knee_series:
        path = _mapping_signal_path(knee_rows, x_values, column, plot_left, plot_top, plot_width, plot_bottom - plot_top, time_min, time_max, 0, 180)
        dash = ' stroke-dasharray="6 3"' if column in SMOOTHED_ANGLE_COLUMNS else ""
        parts.append(f'<path class="context" stroke="{color}"{dash} d="{path}"/>')
    legend_x, legend_y = plot_left + 8, plot_bottom + 17
    for index, (column, color, label) in enumerate(knee_series):
        x = legend_x + index * 180
        dash = ' stroke-dasharray="6 3"' if column in SMOOTHED_ANGLE_COLUMNS else ""
        parts.append(f'<line x1="{x}" y1="{legend_y - 4}" x2="{x + 20}" y2="{legend_y - 4}" stroke="{color}" stroke-width="2"{dash}/><text x="{x + 26}" y="{legend_y}" font-size="11">{label}</text>')
    parts.append(f'<text x="{plot_left + plot_width / 2}" y="{plot_bottom + 39}" text-anchor="middle" font-size="12">{x_label}</text>')

    parts.append('</svg>')
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return output
