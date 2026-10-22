from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.foot_strike_analysis import (
    GroundTruthEvent,
    PredictionEvent,
    cadence_series,
    detect_contact_zone_candidates,
    evaluate_predictions,
    event_knee_angle_rows,
    load_ground_truth,
    validate_ground_truth_source_metadata,
)
from src.gait_metrics import KneeAngleRow


def _document() -> dict:
    def landmark(y: float, confidence: float = 0.9) -> dict:
        return {"x": 0.5, "y": y, "visibility": confidence, "presence": confidence}

    frames = []
    for i, y in enumerate((0.7, 0.85, 0.85, 0.7, 0.7)):
        landmarks = {}
        for side in ("left", "right"):
            landmarks.update({
                f"{side}_heel": landmark(y),
                f"{side}_foot_index": landmark(y),
                f"{side}_hip": landmark(0.3),
                f"{side}_knee": landmark(0.5),
                f"{side}_ankle": landmark(0.7),
            })
        frames.append({"frame_index": i, "timestamp_seconds": i / 30,
                       "poses": [{"landmarks": landmarks}]})
    return {"source_video": "sample.mp4", "video": {"width": 100, "height": 100,
            "fps": 30.0, "duration_seconds": 5 / 30}, "frames": frames}


class FootStrikeAnalysisTests(unittest.TestCase):
    def test_matching_is_one_to_one_same_side_and_maximizes_valid_matches(self) -> None:
        truth = (
            GroundTruthEvent("E1", 10, 10 / 30, "left"),
            GroundTruthEvent("E2", 12, 12 / 30, "left"),
            GroundTruthEvent("E3", 10, 10 / 30, "right"),
        )
        predictions = (
            PredictionEvent("left", 11, 11 / 30),
            PredictionEvent("left", 13, 13 / 30),
            PredictionEvent("right", 20, 20 / 30),
            PredictionEvent("left", 10, 10 / 30),
        )
        summary, matches = evaluate_predictions(truth, {"A": predictions}, fps=30, tolerance_frames=1)
        left = next(row for row in summary if row["side"] == "left")
        right = next(row for row in summary if row["side"] == "right")
        self.assertEqual((left["tp"], left["fp"], left["fn"]), (2, 1, 0))
        self.assertEqual((right["tp"], right["fp"], right["fn"]), (0, 1, 1))
        self.assertEqual(len(matches), 2)
        self.assertEqual({row["event_id"] for row in matches}, {"E1", "E2"})

    def test_contact_zone_uses_foot_geometry_not_heel_peak_and_skips_initial_contact(self) -> None:
        document = _document()
        results = detect_contact_zone_candidates(document)
        self.assertFalse(any(event.frame_index == 0 for event in results))
        self.assertTrue(any(event.side == "left" and event.frame_index == 1 for event in results))

    def test_contact_zone_excludes_low_confidence_landmarks(self) -> None:
        document = _document()
        document["frames"][1]["poses"][0]["landmarks"]["left_foot_index"]["visibility"] = 0.1
        document["frames"][1]["poses"][0]["landmarks"]["left_heel"]["visibility"] = 0.1
        results = detect_contact_zone_candidates(document)
        self.assertFalse(any(event.side == "left" for event in results))

    def test_ground_truth_loader_checks_frame_and_timestamp_alignment(self) -> None:
        document = _document()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("event_id", "frame_index", "timestamp_seconds", "foot_side",
                                 "event_type", "label_confidence", "notes"))
                writer.writerow(("E001", 2, "0.066666667", "left", "foot_strike", "", ""))
            events = load_ground_truth(path, document)
            self.assertEqual((events[0].frame_index, events[0].side), (2, "left"))
            text = path.read_text(encoding="utf-8").replace("0.066666667", "0.5")
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "timestamp does not align"):
                load_ground_truth(path, document)

    def test_source_metadata_checks_video_fingerprint_and_timeline(self) -> None:
        document = _document()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "foot_strike_ground_truth.csv"
            labels.write_text("placeholder\n", encoding="utf-8")
            source = root / "sample.mp4"
            source.write_bytes(b"video-bytes")
            sidecar = {
                "source_video": "sample.mp4",
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "frame_count": 5,
                "fps": 30.0,
                "width": 100,
                "height": 100,
            }
            metadata_path = root / "foot_strike_ground_truth_metadata.json"
            metadata_path.write_text(json.dumps(sidecar), encoding="utf-8")
            validate_ground_truth_source_metadata(labels, document, source)
            source.write_bytes(b"different-video")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                validate_ground_truth_source_metadata(labels, document, source)

    def test_event_knee_angles_preserve_invalid_status_without_fake_values(self) -> None:
        document = _document()
        rows = [KneeAngleRow(i, i / 30, 170.0 if i == 1 else None, None,
                             "ok" if i == 1 else "low_confidence", "missing_landmark")
                for i in range(5)]
        truth = (GroundTruthEvent("E1", 1, 1 / 30, "left"),
                 GroundTruthEvent("E2", 2, 2 / 30, "right"))
        output, summary = event_knee_angle_rows(truth, document, rows)
        self.assertTrue(output[0]["valid"])
        self.assertAlmostEqual(output[0]["knee_angle_deg"], 170.0)
        self.assertFalse(output[1]["valid"])
        self.assertIsNone(output[1]["knee_angle_deg"])
        self.assertEqual(summary["left"]["valid_event_count"], 1)
        self.assertEqual(summary["right"]["valid_event_count"], 0)

    def test_cadence_counts_both_sides_and_handles_short_clips(self) -> None:
        events = (GroundTruthEvent("E1", 0, 0.0, "left"),
                  GroundTruthEvent("E2", 15, 0.5, "right"),
                  GroundTruthEvent("E2_DUP", 15, 0.5, "right"))
        series, summary = cadence_series(events, duration_seconds=2.0)
        self.assertEqual(summary["step_count"], 2)
        self.assertIsNone(summary["average_cadence_steps_per_minute"])
        self.assertIn("Insufficient data", summary["warning"])
        self.assertEqual(len(series), 1)
        short_series, short_summary = cadence_series(events, duration_seconds=1.0)
        self.assertEqual(short_series, [])
        self.assertIn("shorter than the cadence window", short_summary["warning"])


if __name__ == "__main__":
    unittest.main()
