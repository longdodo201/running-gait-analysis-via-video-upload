# Phase 2.4 exploratory event candidates

This experiment uses saved Phase 1 landmarks and the aligned Phase 2.3 knee-angle CSV. It does not run MediaPipe or decode the source video. The knee angles are plotted only for context; event detection uses the pose-0 heel trajectory independently for each side.

The baseline converts normalized heel `y` to image pixels, rejects samples below the visibility or presence threshold, applies the Phase 2.3 segment-aware median filter, then finds prominent local maxima. Since image `y` increases downward, these maxima are heel vertical low-point candidates. They are not verified foot-strikes, toe-offs, or ground contacts. The output therefore reports exploratory leg-cycle timing proxies only; it does not compute cadence or step time.

Run from the repository root, using distinct new output paths:

```powershell
python scripts/detect_gait_events.py `
  outputs/running_gait_sideview2_landmarks.json `
  outputs/running_gait_sideview2_knee_angles_phase23.csv `
  outputs/running_gait_sideview2_gait_events_phase24.csv
```

This writes a candidate-event CSV and an SVG diagnostic plot at the same basename. Existing Phase 2.2 and Phase 2.3 outputs are inputs only and are never overwritten. The Phase 2.3 file must have matching frame indices and timestamps.

Parameters can be changed on the command line:

- `--confidence-threshold` (default `0.5`): higher values reject more heel samples and create longer gaps.
- `--smoothing-window-frames` (default `5`, positive odd): larger windows attenuate jitter more but can flatten or shift short features.
- `--minimum-prominence-px` (default `20`): higher values retain fewer, stronger vertical excursions. Prominence is measured against minima in the local window.
- `--minimum-separation-frames` (default `12`): higher values suppress nearby candidates, but can also suppress real repeated motion.
- `--prominence-window-frames` (default `30`): sets the local context radius used to estimate peak prominence.
- `--plot PATH`: choose a different SVG output path.

The CSV records side, candidate type, source frame/time, smoothed signal name/value, prominence, heel visibility/presence, and an explicit exploratory flag. The plot shows raw and filtered heel `y`, candidate marks, and Phase 2.3 knee-angle context with gaps preserved.

## Interpretation limits

This is an unvalidated signal heuristic on one side-view clip, without manual event annotations or force/contact ground truth. Heel image position alone cannot establish contact. Occlusion, perspective, camera motion, pose jitter, running style, and frame rate can alter or obscure the extrema. Candidate intervals are not step times. Do not interpret the results clinically. A next evaluation step is frame-level manual labeling of a representative set of visible contact events, ideally compared with synchronized contact/force measurements; only then can event definitions and detection quality be assessed.
