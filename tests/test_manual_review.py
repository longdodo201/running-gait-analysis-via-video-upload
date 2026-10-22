from __future__ import annotations

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from scripts import prepare_candidate_context
from src.manual_review import (
    REVIEW_COLUMNS,
    align_candidate_context_targets,
    build_review_rows,
    clamp_review_frame_indices,
    review_frame_indices,
    select_review_targets,
    validate_manual_label,
    validate_review_csv,
    write_review_csv,
)


EVENTS = [
    {
        "side": "left",
        "frame_index": "11",
        "timestamp_seconds": "0.36666666666666664",
        "signal_name": "left_heel_y_px_median5",
        "signal_value_px": "784.9063882827759",
    },
    {
        "side": "right",
        "frame_index": "28",
        "timestamp_seconds": "0.9333333333333333",
        "signal_name": "right_heel_y_px_median5",
        "signal_value_px": "809.6760263442993",
    },
]


class ManualReviewTests(unittest.TestCase):
    def test_export_has_one_row_per_candidate_and_leaves_labels_blank(self) -> None:
        review_rows = build_review_rows(EVENTS)
        self.assertEqual(len(review_rows), len(EVENTS))
        self.assertEqual(review_rows[0]["event_id"], "E001")
        self.assertEqual(review_rows[0]["side"], "left")
        self.assertEqual(review_rows[0]["frame_index"], "11")
        self.assertEqual(review_rows[0]["timestamp_seconds"], EVENTS[0]["timestamp_seconds"])
        self.assertEqual(review_rows[0]["candidate_signal_value_px"], EVENTS[0]["signal_value_px"])
        self.assertEqual(review_rows[0]["manual_label"], "")

        with tempfile.TemporaryDirectory() as directory:
            path = write_review_csv(review_rows, Path(directory) / "review.csv")
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames or ()), REVIEW_COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["manual_label"] == "" for row in rows))
            self.assertEqual(validate_review_csv(path), 0)

    def test_supported_labels_validate_after_manual_entry(self) -> None:
        for label in ("confirmed", "rejected", "uncertain"):
            self.assertEqual(validate_manual_label(label), label)
        self.assertEqual(validate_manual_label("  CONFIRMED "), "confirmed")
        with self.assertRaises(ValueError):
            validate_manual_label("foot-strike")

    def test_review_validator_counts_labels_and_rejects_invalid_label(self) -> None:
        rows = build_review_rows(EVENTS)
        rows[0]["manual_label"] = "confirmed"
        rows[1]["manual_label"] = "uncertain"
        with tempfile.TemporaryDirectory() as directory:
            path = write_review_csv(rows, Path(directory) / "review.csv")
            self.assertEqual(validate_review_csv(path), 2)
            rows[1]["manual_label"] = "maybe"
            with self.assertRaises(ValueError):
                write_review_csv(rows, path)

    def test_invalid_candidate_side_or_frame_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_review_rows([dict(EVENTS[0], side="center")])
        with self.assertRaises(ValueError):
            build_review_rows([dict(EVENTS[0], frame_index="-1")])

    def test_sequence_frame_indices_keep_candidate_and_add_context(self) -> None:
        self.assertEqual(review_frame_indices(30, 1.0, 30.0, 0.2), (24, 30, 36))
        self.assertEqual(review_frame_indices(0, 0.0, 30.0, 0.2), (0, 0, 6))
        self.assertEqual(clamp_review_frame_indices((338, 344, 350), 349), (338, 344, 348))
        with self.assertRaises(ValueError):
            clamp_review_frame_indices((338, 349, 350), 349)

    def test_exact_frame_offset_and_video_boundary_clamping(self) -> None:
        self.assertEqual(review_frame_indices(30, 1.0, 30.0, offset_frames=3), (27, 30, 33))
        self.assertEqual(review_frame_indices(0, 0.0, 30.0, offset_frames=3), (0, 0, 3))
        self.assertEqual(clamp_review_frame_indices((0, 0, 3), 349), (0, 0, 3))
        self.assertEqual(clamp_review_frame_indices((344, 347, 350), 349), (344, 347, 348))
        with self.assertRaises(ValueError):
            review_frame_indices(5, 0.1, 30.0, offset_frames=0)

    def test_candidate_context_targets_keep_review_ids_and_candidate_order(self) -> None:
        review = build_review_rows(EVENTS)
        review[1]["manual_label"] = "uncertain"
        targets = align_candidate_context_targets(review, list(reversed(EVENTS)))
        self.assertEqual([row["event_id"] for row in targets], ["E002", "E001"])
        self.assertEqual([row["frame_index"] for row in targets], ["28", "11"])
        self.assertEqual(targets[0]["manual_label"], "uncertain")
        self.assertEqual(review[0]["event_id"], "E001")
        with self.assertRaisesRegex(ValueError, "timestamps differ"):
            align_candidate_context_targets(review, [dict(EVENTS[0], timestamp_seconds="0.4"), EVENTS[1]])

    def test_context_cli_refuses_an_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events_csv = root / "events.csv"
            review_csv = root / "review.csv"
            video = root / "clip.mp4"
            events_csv.write_text("placeholder", encoding="utf-8")
            review_csv.write_text("placeholder", encoding="utf-8")
            video.write_bytes(b"placeholder")
            output_dir = root / "existing-context"
            output_dir.mkdir()
            marker = output_dir / "keep.jpg"
            marker.write_bytes(b"existing")
            with mock.patch.object(sys, "argv", [
                "prepare_candidate_context.py", str(events_csv), str(review_csv), str(video), str(output_dir),
            ]), mock.patch.object(prepare_candidate_context, "write_review_sequence_contact_sheet") as writer:
                self.assertEqual(prepare_candidate_context.main(), 2)
            writer.assert_not_called()
            self.assertEqual(marker.read_bytes(), b"existing")

    def test_target_selection_uses_uncertain_label_or_low_confidence_without_relabeling(self) -> None:
        review = build_review_rows(EVENTS)
        review[0]["manual_label"] = "uncertain"
        review[1]["manual_label"] = "confirmed"
        candidates = [
            dict(EVENTS[0], landmark_visibility="0.9", landmark_presence="0.98"),
            dict(EVENTS[1], landmark_visibility="0.4", landmark_presence="0.98"),
        ]
        selected = select_review_targets(review, candidates, confidence_threshold=0.5)
        self.assertEqual([row["event_id"] for row in selected], ["E001", "E002"])
        self.assertEqual([row["review_reason"] for row in selected], ["manual_uncertain", "low_landmark_confidence"])
        self.assertEqual(review[0]["manual_label"], "uncertain")
        self.assertEqual(review[1]["manual_label"], "confirmed")

    def test_normal_confidence_non_uncertain_events_are_not_selected(self) -> None:
        review = build_review_rows(EVENTS)
        review[0]["manual_label"] = "confirmed"
        review[1]["manual_label"] = "rejected"
        candidates = [dict(row, landmark_visibility="0.9", landmark_presence="0.95") for row in EVENTS]
        self.assertEqual(select_review_targets(review, candidates), [])


if __name__ == "__main__":
    unittest.main()
