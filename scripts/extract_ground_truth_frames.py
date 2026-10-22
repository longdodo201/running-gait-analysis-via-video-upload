"""Extract every decodable video frame into a new annotation workspace.

This utility does not run pose estimation or modify the source video. The
destination must not exist, so earlier annotation material cannot be replaced.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path
import sys
from typing import Sequence

import cv2


FRAME_COLUMNS = ("frame_index", "timestamp_seconds", "timestamp_ms", "image_filename")
ANNOTATION_COLUMNS = (
    "event_id",
    "frame_index",
    "timestamp_seconds",
    "foot_side",
    "event_type",
    "label_confidence",
    "notes",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _readme_text(
    *,
    source: Path,
    backend: str,
    fps: float,
    frame_count_reported: int | None,
    width: int,
    height: int,
    frame_rows: Sequence[dict[str, object]],
    failed_indices: Sequence[int],
    timestamp_method: str,
    source_hash_before: str,
    source_hash_after: str,
) -> str:
    decoded_count = len(frame_rows) + len(failed_indices)
    if fps > 0:
        duration = decoded_count / fps
        duration_line = f"Estimated duration from decoded frame count / FPS: {duration:.6f} seconds."
    elif frame_rows:
        duration = float(frame_rows[-1]["timestamp_seconds"])
        duration_line = f"Last decoded frame timestamp: {duration:.6f} seconds; FPS unavailable."
    else:
        duration_line = "Duration unavailable because no valid FPS or frame was decoded."

    reported = "unknown" if frame_count_reported is None else str(frame_count_reported)
    if failed_indices:
        failed_text = ", ".join(str(index) for index in failed_indices)
    else:
        failed_text = "none"
    source_unchanged = "yes" if source_hash_before == source_hash_after else "NO — hash changed"
    interval_note = (
        f"At {fps:g} FPS, nominal frame spacing is approximately {1000.0 / fps:.2f} ms."
        if fps > 0
        else "Nominal frame spacing cannot be computed because FPS metadata is unavailable."
    )
    return f"""# Running gait side-view frames

## Extraction record

- Source video: `{source}`
- Decoder backend: `{backend}`
- Reported FPS: {fps:.9g}
- Reported frame count: {reported}
- Decoded frame count: {decoded_count}
- Successfully written images / index rows: {len(frame_rows)}
- Failed or unavailable frame indices: {failed_text}
- Resolution: {width} × {height} pixels
- {duration_line}
- Per-frame timestamp method: {timestamp_method}
- Source SHA-256 unchanged: {source_unchanged}

Timestamps are recorded from the decoder where available. If the decoder did
not provide a finite, non-negative timestamp for a frame, the script falls
back to `frame_index / FPS`; that fallback is not exact for variable-frame-rate
video. Decimal places in the CSV do not imply greater measurement accuracy.

## View frames in order

Open the `frames` folder and sort by filename ascending. Names are zero-based
and padded (`frame_000000.jpg`, `frame_000001.jpg`, …), so alphabetical order
matches source-frame order. Images retain the decoded frame resolution and are
not resized, cropped, or annotated.

## Find a frame timestamp

Look up `frame_index` in `frame_index.csv`. Each row gives the frame timestamp
in seconds and milliseconds and the corresponding image filename. Frame indices
are zero-based. The CSV contains one row for every successfully written image.

## Annotate foot-strikes

Add one row per observed event to `foot_strike_ground_truth.csv`:

- `event_id`: a unique ID such as `E001`.
- `frame_index`: the zero-based frame of the event.
- `timestamp_seconds`: copy the value for that frame from `frame_index.csv`.
- `foot_side`: `left` or `right`.
- `event_type`: `foot_strike`.
- `label_confidence`: `high`, `medium`, or `low`.
- `notes`: optional context.

Use the first frame where the shoe is visibly touching the treadmill belt as
the annotation frame, consistently for both feet. If the exact landing moment
cannot be determined, use a low confidence label and note that it is ambiguous
and should be checked in the source video. Do not infer events from missing or
unclear frames.

{interval_note}

The extracted frames support visual annotation only. Manual video labels are
observational references and are not equivalent to force-plate ground truth.
"""


def extract_frames(video_path: Path, output_dir: Path) -> dict[str, object]:
    """Extract frames and write the requested index, blank template and README."""
    source = video_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input video does not exist: {source}")
    if destination.exists():
        raise FileExistsError(f"Output directory already exists; refusing to overwrite: {destination}")

    source_hash_before = _sha256(source)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open the input video: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 0.0
    raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_reported = int(raw_count) if math.isfinite(raw_count) and raw_count >= 0 else None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    backend = capture.getBackendName()

    destination.mkdir(parents=True, exist_ok=False)
    frames_dir = destination / "frames"
    frames_dir.mkdir()
    rows: list[dict[str, object]] = []
    failed_indices: list[int] = []
    timestamp_methods: set[str] = set()
    frame_index = 0
    padding = max(6, len(str(max(0, (frame_count_reported or 1) - 1))))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                failed_indices.append(frame_index)
                frame_index += 1
                continue

            decoded_timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if math.isfinite(decoded_timestamp_ms) and decoded_timestamp_ms >= 0:
                timestamp_seconds = decoded_timestamp_ms / 1000.0
                timestamp_methods.add("OpenCV CAP_PROP_POS_MSEC after sequential decode")
            elif fps > 0:
                timestamp_seconds = frame_index / fps
                timestamp_methods.add("frame_index / reported FPS fallback")
            else:
                timestamp_seconds = 0.0
                timestamp_methods.add("0.0 fallback (decoder timestamp and FPS unavailable)")

            filename = f"frame_{frame_index:0{padding}d}.jpg"
            image_path = frames_dir / filename
            if not cv2.imwrite(str(image_path), frame):
                failed_indices.append(frame_index)
                frame_index += 1
                continue
            rows.append({
                "frame_index": frame_index,
                "timestamp_seconds": f"{timestamp_seconds:.9f}",
                "timestamp_ms": f"{timestamp_seconds * 1000.0:.6f}",
                "image_filename": filename,
            })
            frame_index += 1
    finally:
        capture.release()

    if frame_count_reported is not None and frame_index < frame_count_reported:
        failed_indices.extend(range(frame_index, frame_count_reported))

    _write_csv(destination / "frame_index.csv", FRAME_COLUMNS, rows)
    _write_csv(destination / "foot_strike_ground_truth.csv", ANNOTATION_COLUMNS, [])
    source_hash_after = _sha256(source)
    timestamp_method = "; ".join(sorted(timestamp_methods)) if timestamp_methods else "no frames decoded"
    readme = _readme_text(
        source=source,
        backend=backend,
        fps=fps,
        frame_count_reported=frame_count_reported,
        width=width,
        height=height,
        frame_rows=rows,
        failed_indices=sorted(set(failed_indices)),
        timestamp_method=timestamp_method,
        source_hash_before=source_hash_before,
        source_hash_after=source_hash_after,
    )
    (destination / "README.md").write_text(readme, encoding="utf-8")

    actual_read_count = frame_index
    expected_count = frame_count_reported
    sequential_indices = [int(row["frame_index"]) for row in rows]
    successful_images_match = all(
        (frames_dir / str(row["image_filename"])).is_file() for row in rows
    )
    no_duplicate_indices = len(sequential_indices) == len(set(sequential_indices))
    source_unchanged = source_hash_before == source_hash_after
    result = {
        "source": source,
        "output_dir": destination,
        "backend": backend,
        "fps": fps,
        "reported_frame_count": frame_count_reported,
        "read_frame_count": actual_read_count,
        "width": width,
        "height": height,
        "duration_seconds": actual_read_count / fps if fps > 0 else None,
        "exported_count": len(rows),
        "failed_indices": sorted(set(failed_indices)),
        "timestamp_method": timestamp_method,
        "continuous_source_indices": sequential_indices == list(range(actual_read_count)),
        "no_duplicate_indices": no_duplicate_indices,
        "every_index_row_has_image": successful_images_match,
        "source_unchanged": source_unchanged,
        "count_matches_metadata": expected_count is None or actual_read_count == expected_count,
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_video", type=Path, help="Source video file")
    parser.add_argument("output_dir", type=Path, help="New output directory; must not already exist")
    args = parser.parse_args(argv)
    try:
        result = extract_frames(args.input_video, args.output_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Input: {result['source']}")
    print(f"Output: {result['output_dir']}")
    print(f"Decoder: {result['backend']}")
    print(f"FPS: {result['fps']:.9g}")
    print(f"Reported frames: {result['reported_frame_count']}")
    print(f"Frames read: {result['read_frame_count']}")
    print(f"Resolution: {result['width']}x{result['height']}")
    print(f"Duration estimate: {result['duration_seconds']}")
    print(f"Images written: {result['exported_count']}")
    print(f"Failed frame indices: {result['failed_indices']}")
    print(f"Timestamp method: {result['timestamp_method']}")
    for key in (
        "continuous_source_indices",
        "no_duplicate_indices",
        "every_index_row_has_image",
        "source_unchanged",
        "count_matches_metadata",
    ):
        print(f"{key}: {result[key]}")
    return 0 if all(result[key] for key in (
        "continuous_source_indices",
        "no_duplicate_indices",
        "every_index_row_has_image",
        "source_unchanged",
        "count_matches_metadata",
    )) and not result["failed_indices"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
