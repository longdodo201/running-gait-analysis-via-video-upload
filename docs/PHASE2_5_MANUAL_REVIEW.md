# Phase 2.5 manual review of event candidates

This phase creates a review worksheet and a contact sheet from the existing Phase 2.4 candidate CSV plus the original MP4. It decodes video frames for display only; it does not run MediaPipe, alter Phase 2.4 detection, or infer contact events automatically.

From the repository root, use the same original video that produced the saved landmarks. For this clip:

```powershell
python scripts/prepare_event_review.py `
  outputs/running_gait_sideview2_gait_events_phase24.csv `
  "C:\Users\Admin\Videos\running_gait_sideview2.mp4" `
  outputs/running_gait_sideview2_event_review_phase25.csv `
  outputs/running_gait_sideview2_event_contact_sheet_phase25.jpg
```

Open the contact sheet and the original video in a local player. Each tile shows its event ID, side, source frame index, and candidate timestamp. Check that the video is the exact source clip: the frame index is zero-based and the timestamp comes from the source FPS used by Phase 1. Seek to the timestamp, inspect the candidate frame and nearby frames (about 0.2–0.3 seconds on either side), and compare it with visible foot/ground contact. Use the contact sheet as an index; playback around the event is often clearer than a single frame.

Edit only the `manual_label` and `reviewer_notes` columns in the review CSV. Leave a label blank if not reviewed. Supported labels are:

- `confirmed`: the event definition being reviewed is visibly supported at that time.
- `rejected`: the video gives enough evidence that the candidate does not match the event definition.
- `uncertain`: evidence is ambiguous, including when the relevant foot or ground is not visible.

The candidate name is `heel_vertical_low_point_candidate`; reviewers should record whether that visual feature aligns with an observable contact moment, not silently redefine it. Validate labels after editing:

```powershell
python scripts/validate_event_review.py outputs/running_gait_sideview2_event_review_phase25.csv
```

The validator accepts blank labels or the three supported values. It reports how many rows have a label; it does not verify that a human judgment is correct.

## Review an event with nearby frames

After saving manual labels in the review CSV, create a separate contact sheet for events labeled `uncertain` plus events whose Phase 2.4 heel visibility or presence is below `0.5` (the detector's existing confidence cutoff). Missing/invalid confidence also selects an event. This does not change labels or candidate detection:

```powershell
python scripts/prepare_uncertain_event_sequences.py `
  outputs/running_gait_sideview2_event_review_phase25.csv `
  outputs/running_gait_sideview2_gait_events_phase24.csv `
  "C:\Users\Admin\Videos\running_gait_sideview2.mp4" `
  outputs/running_gait_sideview2_uncertain_context_phase25.jpg
```

For each selected event, the contact sheet shows three frames in order: approximately 0.2 seconds before the timestamp, the exact candidate frame index, and approximately 0.2 seconds after. Each tile includes event ID, side, source frame index, that frame's time, and the original candidate time. The before/after frames are the nearest decoded frames at the video's reported FPS; for a 30 FPS source this is about six frames on either side. The candidate tile always uses the original candidate frame index. Adjust the interval with `--offset-seconds`; adjust the confidence cutoff with `--confidence-threshold`.

If the review CSV has no `uncertain` rows and no candidate confidence below the cutoff, the tool reports that no events qualify and creates no file. Review the three-frame sequence, then update only the existing review CSV's `manual_label` or `reviewer_notes` yourself. The tool never assigns a label. Run `scripts/validate_event_review.py` again after editing.

Labels assigned from a single still image are preliminary manual annotations, not externally verified ground truth. Keep events labeled `uncertain` provisional until their surrounding sequence has been reviewed; even sequence review may not prove ground contact from an occluded side view.

## Limits

Visual confirmation can be limited by occlusion, low frame rate, motion blur, and the side-view camera angle. A single view may not show whether the foot has made contact, and heel vertical extrema are not equivalent to foot-strike. Preserve `uncertain` labels when evidence is weak. These labels are for evaluating the exploratory candidates; they do not establish cadence, step time, clinical metrics, or validated detector performance. Keep the original Phase 2.2–2.4 artifacts unchanged and save review results to new paths.
