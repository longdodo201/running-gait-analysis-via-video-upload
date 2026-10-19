# Phase 1: Local video pose extraction

## Input and constraints

The CLI accepts a local `.mp4` file. By default, the application rejects files larger than 100 MiB and videos longer than 60 seconds. These configurable application limits are not MediaPipe limits. Use `--max-bytes 0` or `--max-seconds 0` to disable the corresponding limit. OpenCV must be able to decode the MP4 codec.

The source video is never overwritten. The pose model is not downloaded automatically; obtain `models/pose_landmarker_full.task` and verify its SHA-256 using `docs/PHASE0_SETUP.md`.

## Command

From the repository root in the prepared Python environment:

```powershell
python scripts\process_video.py "data\samples\my_run.mp4" --model "models\pose_landmarker_full.task" --output "outputs\my_run_landmarks.json" --annotated-video "outputs\my_run_annotated.mp4"
```

The overlay option is optional. Omitting it avoids video encoding. The output directory is created as needed.

## Flow and failure behavior

1. Validate the file, `.mp4` suffix, size limit, decodability, dimensions, FPS, frame count, and metadata-derived duration.
2. Sequentially decode every frame and enforce the duration limit again using the actual decoded frame count when FPS is valid.
3. Run the local MediaPipe Pose Landmarker CPU delegate on each frame. VIDEO mode is used when the source FPS is valid; otherwise IMAGE mode is used and timestamps are null.
4. Preserve zero-pose frames in memory. If no person is detected in the entire video, report an actionable error and do not write a landmark JSON result.
5. Write JSON atomically. When requested, render a skeleton overlay into an MP4 temporary file and publish it only after successful processing.

Unreadable videos, invalid metadata, inference failures, and encoder initialization failures are reported; processing does not silently omit failed frames. Capture, estimator, and optional writer resources are closed on success and exceptions.

## JSON schema version 1.0

```json
{
  "schema_version": "1.0",
  "source_video": "my_run.mp4",
  "video": {
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "frame_count": 180,
    "duration_seconds": 6.0,
    "warnings": []
  },
  "timestamp_basis": "frame_index / source_fps",
  "warnings": [],
  "frames": [
    {
      "frame_index": 0,
      "timestamp_seconds": 0.0,
      "poses": [
        {
          "landmarks": {
            "nose": {
              "x": 0.5,
              "y": 0.2,
              "z": -0.1,
              "visibility": 0.99,
              "presence": 0.99
            }
          }
        }
      ]
    }
  ]
}
```

Each detected pose contains all 33 MediaPipe pose names in stable order. Coordinates are the model's normalized image `x`/`y` and relative `z`; `z` is not a metric depth measurement. `visibility` and `presence` are optional confidence-like outputs, not accuracy. Frames with no detected person have an empty `poses` array. `timestamp_seconds` is `frame_index / source_fps` for valid positive finite FPS and `null` otherwise. The output records only the input basename, not its absolute path.

## Tests and manual smoke test

Run the offline tests:

```powershell
python -m unittest discover -s tests -v
```

They exercise path/format/size/duration validation, missing or invalid FPS, timestamp calculation, named landmark conversion, JSON schema, no-pose behavior, inference failure reporting, and resource cleanup with synthetic fixtures. They do not test real MediaPipe inference or actual MP4 codec behavior.

For a manual smoke test, use a short running MP4 you are permitted to process and the verified local model asset. Run the CLI command above, inspect the JSON frame count and detections, and play the optional annotated MP4. Record the Python/dependency/model versions and any decoder warnings. A successful smoke test only demonstrates execution on that clip, not gait-event or metric validity.

## Limitations

- The model is applied per frame; no gait event detector, gait metric, quality score, or medical assessment is implemented in Phase 1.
- Results depend on camera angle, framing, lighting, clothing, occlusion, resolution, codec, FPS, and landmark jitter. A side-on full-body view is preferred.
- Source timestamps assume constant FPS when OpenCV reports a valid FPS. Variable-frame-rate timestamp mapping is not recovered from OpenCV metadata.
- Invalid FPS is handled without inventing source times: output timestamps are null and IMAGE mode is used. Optional overlay encoding uses a documented 30 FPS fallback in this case.
- The app does not resample, stabilize, or alter the decoded input frame dimensions.
- No real running video or ground-truth dataset is bundled or claimed as tested.
