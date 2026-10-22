# Phase 3: Ground Truth evaluation and event metrics

## Dataset in this repository

The current annotation file is `outputs/foot_strike_ground_truth.csv`. It has 31 event rows for `running_gait_sideview.mp4`, frames 27–318, on a 321-frame, 720 × 878 video at 30 FPS (10.7 seconds). Each row is one left/right foot-strike. `label_confidence` and `notes` are blank in the supplied labels; the analysis preserves that rather than assigning confidence.

The matching landmark JSON is `outputs/running_gait_sideview_landmarks.json`. The evaluation validates event frames and timestamps against that saved timeline. It performs no pose inference and does not modify the Ground Truth.

`outputs/foot_strike_ground_truth_metadata.json` stores the source video's SHA-256 and decoded metadata. When the original MP4 is available, the evaluation CLI checks its fingerprint so a different file with the same name cannot inherit these labels.

## Algorithms and matching

- **A — `heel_trajectory_local_extrema`:** the existing smoothed heel image-y local maxima with prominence and spacing thresholds.
- **B — `foot_contact_zone`:** a different pose-image signal. For each side, the lower of heel and foot-index y enters the near-belt zone while vertical speed is below the configured threshold. The default belt-height ratio is 0.92 of image height, contact band 18 pixels, speed limit 600 pixels/second, and minimum same-side spacing 8 frames. These are exploratory heuristics; they share MediaPipe landmarks with Algorithm A, and the belt-height ratio may not transfer to another camera or treadmill.

Both algorithms consume the same saved landmarks and compare with the same event labels. Matching is separated by side and uses a one-to-one dynamic-programming assignment that maximizes matched pairs before minimizing total absolute frame error, with tolerance ±2 frames. Error sign is prediction minus reference. Milliseconds use the actual per-frame timestamps; for this 30 FPS constant-frame-rate video, one frame is about 33.33 ms.

## Knee angle and cadence

For each Ground Truth event, knee angle uses the existing 2D angle at the knee between hip-knee and ankle-knee vectors. A straight leg is approximately 180°. The event CSV includes per-landmark visibility/presence, minimum confidence, status and validity; missing or low-confidence geometry produces a blank angle.

Cadence counts each valid left or right Ground Truth event as one step and reports event count divided by full clip duration × 60. The time series uses trailing 2-second half-open windows stepped every 0.5 seconds. Fewer than four events or a clip shorter than three seconds yields no clip-average number. A single clip with partial cycles at its boundaries may bias the average.

## Reproduce from saved data (without MediaPipe)

Choose a new output directory; existing directories are refused:

```powershell
python scripts/evaluate_ground_truth.py `
  outputs/running_gait_sideview_landmarks.json `
  outputs/foot_strike_ground_truth.csv `
  outputs/running_gait_sideview_ground_truth_eval `
  --source-video "path/to/running_gait_sideview.mp4"
```

The analysis writes event summary/matches CSV and JSON, event-level knee angles, cadence CSV/SVG, a combined metrics JSON and a Markdown report. Streamlit automatically selects the current Ground Truth only for an upload named `running_gait_sideview.mp4`; the pipeline additionally checks source filename and Ground Truth frame/timestamp alignment. Other videos do not inherit these labels.

## Limits

The reference is visual annotation, not force-plate ground truth. A single side-view clip cannot establish general event-detector performance. Algorithm B uses a heuristic belt-height zone and shares pose-model input errors with Algorithm A, even though its event rule differs. Knee angles remain 2D estimates affected by camera perspective, occlusion, keypoint error and frame timing; no clinical or normative interpretation is provided.
