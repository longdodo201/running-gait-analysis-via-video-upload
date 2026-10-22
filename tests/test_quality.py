"""Tests for descriptive knee-angle quality diagnostics."""

from __future__ import annotations

import unittest

from src.quality import diagnose_knee_angles


class KneeAngleQualityTests(unittest.TestCase):
    def test_reports_missing_statuses_and_tukey_large_change(self) -> None:
        angles = [100, 101, 100, 101, 100, 100, 100, 180, ""]
        rows = [
            {"frame_index": str(index), "timestamp_seconds": str(index / 30),
             "left_knee_angle_deg": str(angle),
             "left_knee_status": "low_confidence" if angle == "" else "ok"}
            for index, angle in enumerate(angles)
        ]
        result = diagnose_knee_angles(rows, "left")
        self.assertEqual(result.missing_angle_count, 1)
        self.assertEqual(result.usable_angle_count, 8)
        self.assertEqual(result.status_counts, {"low_confidence": 1, "ok": 8})
        self.assertEqual(result.adjacent_change_count, 7)
        self.assertEqual(len(result.large_change_flags), 1)
        flag = result.large_change_flags[0]
        self.assertEqual((flag.from_frame_index, flag.to_frame_index), (6, 7))
        self.assertAlmostEqual(flag.absolute_change_deg, 80.0)

    def test_does_not_compare_across_a_missing_row_or_frame_index_gap(self) -> None:
        rows = [
            {"frame_index": "0", "left_knee_angle_deg": "10", "left_knee_status": "ok"},
            {"frame_index": "1", "left_knee_angle_deg": "", "left_knee_status": "low_confidence"},
            {"frame_index": "3", "left_knee_angle_deg": "170", "left_knee_status": "ok"},
        ]
        result = diagnose_knee_angles(rows, "left")
        self.assertEqual(result.adjacent_change_count, 0)
        self.assertEqual(result.large_change_flags, ())
        self.assertIsNone(result.tukey_cutoff_deg)

    def test_rejects_unknown_side(self) -> None:
        with self.assertRaisesRegex(ValueError, "left.*right"):
            diagnose_knee_angles([], "center")


if __name__ == "__main__":
    unittest.main()
