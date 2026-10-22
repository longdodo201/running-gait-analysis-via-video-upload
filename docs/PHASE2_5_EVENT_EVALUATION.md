# Phase 2.5 preliminary candidate review

## Scope and source data

This report compares the saved Phase 2.4 candidate events with the reviewer labels currently stored in `outputs/running_gait_sideview2_event_review_phase25.csv`. It does not rerun pose estimation or alter the detector or source CSV files.

The source clip metadata in `outputs/running_gait_sideview2_landmarks.json` is 720 × 912 pixels, 30 FPS, 349 frames, and 11.633 seconds. The Phase 2.4 candidate CSV has 34 rows. Its columns are:

`side`, `event_type`, `frame_index`, `timestamp_seconds`, `signal_name`, `signal_value_px`, `prominence_px`, `landmark_visibility`, `landmark_presence`, `flags`.

The Phase 2.5 review CSV has 34 rows and columns:

`event_id`, `side`, `frame_index`, `timestamp_seconds`, `candidate_signal_name`, `candidate_signal_value_px`, `manual_label`, `reviewer_notes`.

There is no dedicated reviewer-confidence column. Subjective observer confidence and notes are recorded together in `reviewer_notes`; this is distinct from MediaPipe landmark visibility/presence in the Phase 2.4 CSV. The review rows were matched to candidate rows by side, frame index, and timestamp; all 34 align, with no mismatch. Candidate IDs E001–E034 are review-sheet identifiers, not fields in the Phase 2.4 event schema.

## Detector behavior verified in code

`src/gait_events.py::detect_leg_cycle_candidates` reads pose index 0 and extracts the left and right heel normalized image `y` values independently, converts them to pixels using video height, and rejects samples with missing/invalid values or visibility/presence below 0.5. The resulting valid segments use the existing 5-frame centered median. Within each segment, `_local_peaks` finds local maxima in image `y` (downward is positive), keeps peaks with at least 20 px of local prominence using a 30-frame radius, then suppresses same-side events closer than 12 frames, retaining the more prominent peak. The output type is explicitly `heel_vertical_low_point_candidate`; flags are `exploratory_unvalidated`.

These are heel-trajectory extrema. The algorithm does not observe ground contact and does not establish foot-strike, toe-off, or gait-cycle boundaries.

## Results on the current labels

| Side | Candidates | Confirmed | Rejected | Uncertain | Resolved-label precision proxy |
|---|---:|---:|---:|---:|---:|
| Left | 17 | 13 | 4 | 0 | 13 / 17 = 76.5% |
| Right | 17 | 8 | 8 | 1 | 8 / 16 = 50.0% |
| **Total** | **34** | **21** | **12** | **1** | **21 / 33 = 63.6%** |

The single uncertain candidate is E034. It is excluded from both numerator and denominator; it is not treated as correct or incorrect. Thus 33 of 34 candidates (97.1%) have a resolved manual label. The percentages are conditional on the reviewed candidates with a determinate label. They are not a validated estimate of foot-strike detector precision because the detector's event is a heel vertical low-point candidate, and the labels are preliminary visual judgments from one clip.

The counts above are computed from the saved labels, not inferred from reviewer notes. This report does not compute recall, F1, cadence, or step time.

## What “precision” means here

For the current candidate-only review, a provisional resolved-label precision proxy is:

`confirmed / (confirmed + rejected) = 21 / (21 + 12) = 63.6%`.

This excludes `uncertain` candidates and must be reported together with the excluded count and resolved-label coverage. It answers only: “Among the candidates the reviewer could classify, what fraction did the reviewer confirm under the current visual criterion?” It does not estimate missed foot-strikes, and it should not be described as general detector accuracy or as clinical validity.

For a defensible foot-strike precision/recall evaluation, first define the reference event independently of the detector (for example, first visible shoe–belt contact for each side). Then match predictions to reference events one-to-one on the same side within a predeclared time tolerance. Precision is `matched predictions / all predictions`; recall is `matched reference events / all reference events`. Report the tolerance, unmatched counts, uncertain annotations, and frame-time resolution. Do not use candidate timestamps themselves as the complete reference list.

## Manual full-clip reference procedure

1. Define the event being annotated before reviewing: for example, the first video frame where that side's shoe visibly contacts the treadmill belt. Keep left/right separate. If contact is hidden or cannot be localized, record it as uncertain rather than guessing.
2. Review the complete source clip from beginning to end in a local player with frame stepping, including intervals with no detector candidates. For a less biased reference pass, hide the candidate contact sheet and candidate CSV until the independent annotations are complete.
3. Create one row per manually observed contact in a separate reference file with `reference_id`, `side`, zero-based `frame_index`, `timestamp_seconds`, and `notes`. Include all visible contacts, not only those near E001–E034. Record uncertain/occluded intervals separately so they are not mistaken for negatives.
4. Reconcile the annotation list after the full pass and, if possible, ask a second reviewer to label the clip independently. Resolve disagreements explicitly or retain them as uncertain.
5. Only then match each prediction to at most one same-side reference event within a tolerance chosen in advance. Publish the match rule and counts of matched, unmatched predictions, unmatched reference events, and uncertain references.

At 30 FPS the nominal interval is about 33 ms per frame, so contact timing between frames is not observable. A single side-view recording also has occlusion, motion blur, camera/perspective and belt-visibility limitations. Manual review by one observer is subjective and may have annotation error; these labels are preliminary, not perfect ground truth. Synchronized force/contact measurements or additional views would provide stronger reference evidence.

## Reproducibility

Current counts can be checked with:

```powershell
python scripts/validate_event_review.py outputs/running_gait_sideview2_event_review_phase25.csv
```

The validator checks label vocabulary and rows; it does not establish event validity. Keep this report with the reviewed CSV and preserve the original Phase 2.4 candidates for any later independent-reference evaluation.
