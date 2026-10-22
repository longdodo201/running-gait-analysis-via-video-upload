"""Tests for segment-aware rolling median smoothing."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from src.report import KNEE_ANGLE_COLUMNS, write_smoothed_knee_angles_csv
from src.signal_processing import add_smoothed_knee_angles, smooth_valid_segments
from src.visualization import write_knee_angle_plot_svg


class SegmentSmoothingTests(unittest.TestCase):
    def test_centered_median_reduces_an_isolated_spike(self) -> None:
        values = [100.0, 100.0, 100.0, 180.0, 100.0, 100.0, 100.0]
        smoothed = smooth_valid_segments(values)
        self.assertEqual(smoothed[3], 100.0)
        self.assertEqual(smoothed[0], 100.0)

    def test_does_not_smooth_across_missing_samples(self) -> None:
        values = [10.0] * 6 + [None] + [100.0] * 6
        smoothed = smooth_valid_segments(values)
        self.assertEqual(smoothed, values)
        self.assertIsNone(smoothed[6])

    def test_short_valid_segments_are_copied_unchanged(self) -> None:
        values = [10.0, 80.0, 20.0, None, 40.0, 50.0]
        self.assertEqual(smooth_valid_segments(values), values)

    def test_non_finite_values_become_gaps(self) -> None:
        smoothed = smooth_valid_segments([1.0, float("nan"), 2.0])
        self.assertEqual(smoothed, [1.0, None, 2.0])

    def test_out_of_range_knee_angle_breaks_the_valid_segment(self) -> None:
        rows = [
            {"left_knee_angle_deg": "100", "left_knee_status": "ok"},
            {"left_knee_angle_deg": "181", "left_knee_status": "ok"},
            {"left_knee_angle_deg": "110", "left_knee_status": "ok"},
        ]
        result = add_smoothed_knee_angles(rows)
        self.assertEqual(result[0]["left_knee_angle_smoothed_deg"], 100.0)
        self.assertIsNone(result[1]["left_knee_angle_smoothed_deg"])
        self.assertEqual(result[2]["left_knee_angle_smoothed_deg"], 110.0)

    def test_window_must_be_positive_and_odd(self) -> None:
        for window in (0, 2, -3):
            with self.subTest(window=window), self.assertRaises(ValueError):
                smooth_valid_segments([1.0], window)


class SmoothedRowsTests(unittest.TestCase):
    def test_preserves_rows_raw_fields_timestamps_and_missing_values(self) -> None:
        rows = [
            {"frame_index": "4", "timestamp_seconds": "0.133", "left_knee_angle_deg": "100", "left_knee_status": "ok", "right_knee_angle_deg": "", "right_knee_status": "low_confidence"},
            {"frame_index": "5", "timestamp_seconds": "0.167", "left_knee_angle_deg": "180", "left_knee_status": "ok", "right_knee_angle_deg": "90", "right_knee_status": "ok"},
        ]
        output = add_smoothed_knee_angles(rows)
        self.assertEqual(len(output), len(rows))
        for original, updated in zip(rows, output):
            for key, value in original.items():
                self.assertEqual(updated[key], value)
        self.assertEqual([row["timestamp_seconds"] for row in output], ["0.133", "0.167"])
        self.assertIsNone(output[0]["right_knee_angle_smoothed_deg"])
        self.assertEqual(output[1]["right_knee_angle_smoothed_deg"], 90.0)

    def test_each_side_is_smoothed_independently(self) -> None:
        values = [100.0, 100.0, 100.0, 180.0, 100.0, 100.0, 100.0]
        rows = [
            {"left_knee_angle_deg": str(value), "left_knee_status": "ok",
             "right_knee_angle_deg": str(200 - value), "right_knee_status": "ok"}
            for value in values
        ]
        output = add_smoothed_knee_angles(rows)
        self.assertEqual(output[3]["left_knee_angle_smoothed_deg"], 100.0)
        self.assertEqual(output[3]["right_knee_angle_smoothed_deg"], 100.0)

    def test_csv_adds_smoothed_fields_without_changing_phase22_values(self) -> None:
        rows = [
            {"frame_index": "10", "timestamp_seconds": "0.333", "left_knee_angle_deg": "100.0",
             "right_knee_angle_deg": "", "left_knee_status": "ok", "right_knee_status": "low_confidence"}
        ]
        output_rows = add_smoothed_knee_angles(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = write_smoothed_knee_angles_csv(output_rows, KNEE_ANGLE_COLUMNS, Path(directory) / "phase23.csv")
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                result = list(reader)
                columns = reader.fieldnames
        self.assertEqual(columns, list(KNEE_ANGLE_COLUMNS) + [
            "left_knee_angle_smoothed_deg", "right_knee_angle_smoothed_deg",
        ])
        self.assertEqual(len(result), len(rows))
        for column in KNEE_ANGLE_COLUMNS:
            self.assertEqual(result[0][column], rows[0][column])
        self.assertEqual(result[0]["left_knee_angle_smoothed_deg"], "100.0")
        self.assertEqual(result[0]["right_knee_angle_smoothed_deg"], "")

    def test_svg_plot_labels_raw_smoothed_and_missing_gaps(self) -> None:
        rows = [
            {"timestamp_seconds": "0", "left_knee_angle_deg": "100", "left_knee_angle_smoothed_deg": 100},
            {"timestamp_seconds": "0.03", "left_knee_angle_deg": "", "left_knee_angle_smoothed_deg": None},
            {"timestamp_seconds": "0.06", "left_knee_angle_deg": "110", "left_knee_angle_smoothed_deg": 110},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = write_knee_angle_plot_svg(rows, Path(directory) / "angles.svg")
            svg = path.read_text(encoding="utf-8")
        self.assertIn("Time (seconds)", svg)
        self.assertIn("Knee angle (degrees)", svg)
        self.assertIn(">Raw</text>", svg)
        self.assertIn(">Smoothed</text>", svg)
        self.assertIn("Crosses mark missing raw angles", svg)
        self.assertIn('d="M ', svg)

    def test_svg_uses_cyan_for_left_and_orange_for_right(self) -> None:
        rows = [
            {
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "left_knee_angle_deg": 90,
                "right_knee_angle_deg": 95,
                "left_knee_angle_smoothed_deg": 91,
                "right_knee_angle_smoothed_deg": 94,
            },
            {
                "frame_index": 1,
                "timestamp_seconds": 0.1,
                "left_knee_angle_deg": 92,
                "right_knee_angle_deg": 93,
                "left_knee_angle_smoothed_deg": 91,
                "right_knee_angle_smoothed_deg": 94,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            svg = write_knee_angle_plot_svg(rows, Path(directory) / "colored-angles.svg").read_text(
                encoding="utf-8"
            )
        self.assertIn('stroke="#39d5e6"', svg)
        self.assertIn('stroke="#9cecf3"', svg)
        self.assertIn('stroke="#ff9d55"', svg)
        self.assertIn('stroke="#ffd0a8"', svg)

    def test_svg_uses_frame_index_when_all_timestamps_are_null(self) -> None:
        rows = [{"frame_index": "0", "timestamp_seconds": None, "left_knee_angle_deg": "100"}]
        with tempfile.TemporaryDirectory() as directory:
            path = write_knee_angle_plot_svg(rows, Path(directory) / "no-times.svg")
            svg = path.read_text(encoding="utf-8")
        self.assertIn("Frame index (timestamps unavailable)", svg)


if __name__ == "__main__":
    unittest.main()
