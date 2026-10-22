# Phase 2.3: knee-angle smoothing and diagnostics

Run this from the repository root after creating a Phase 2.2 knee-angle CSV. This reads only that CSV; it does not decode a video or run MediaPipe.

```powershell
python scripts\process_knee_angles.py "outputs\running_gait_sideview2_knee_angles_phase22.csv" "outputs\running_gait_sideview2_knee_angles_phase23.csv"
```

The command writes a new CSV and an SVG plot next to it with the same base name. Use `--plot "outputs\my_knee_angles.svg"` to choose another plot path. The Phase 2.2 input file is never overwritten.

The new CSV keeps all input columns and appends `left_knee_angle_smoothed_deg` and `right_knee_angle_smoothed_deg`. Frame indices, timestamps, raw angles, and statuses are copied from the input. The centered 5-frame rolling median runs separately on contiguous rows whose angle is finite, within 0–180 degrees, and whose status is `ok`; gaps remain blank and are never interpolated across. Valid segments shorter than five frames are copied unchanged. At the ends of longer segments, the median uses the available samples within that segment.

The console reports missing-angle and status counts and applies Tukey's `Q3 + 1.5 × IQR` rule to absolute changes between adjacent, consecutive, usable frames. These flags are heuristic review cues only; they do not prove an angle is erroneous and do not remove observations. The plot shows raw and smoothed series separately, preserves gaps, and marks missing raw angles.

Smoothing is exploratory and may flatten brief real peaks. Angles remain single-camera 2D estimates affected by viewpoint, occlusion, and landmark error. This phase does not calculate gait-cycle ROM, detect gait events, or provide clinical assessment; cycle-specific ROM is deferred until Phase 4 can use defined gait-cycle segments.
