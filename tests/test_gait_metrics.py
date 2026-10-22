"""Tests for frame-level knee-angle extraction and CSV output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.gait_metrics import knee_angle_rows
from src.report import KNEE_ANGLE_COLUMNS, write_knee_angles_csv


def point(x: object, y: object, visibility: object = 0.9, presence: object = 0.9) -> dict[str, object]:
    return {"x": x, "y": y, "z": 0.0, "visibility": visibility, "presence": presence}


def pose(side: str = "left", points: tuple[tuple[object, object], ...] = ((0, 1), (0, 0), (1, 0))) -> dict:
    hip, knee, ankle = points
    return {"landmarks": {
        f"{side}_hip": point(*hip),
        f"{side}_knee": point(*knee),
        f"{side}_ankle": point(*ankle),
    }}


def document(frames: list[dict], width: int = 100, height: int = 100) -> dict:
    return {"video": {"width": width, "height": height}, "frames": frames}


class KneeAngleRowsTests(unittest.TestCase):
    def test_converts_normalized_points_with_non_square_frame_dimensions(self) -> None:
        # Pixel positions are (50,20), (100,50), (140,-16.667): a right angle.
        left = pose("left", ((0.25, 0.2), (0.5, 0.5), (0.7, -1 / 6)))
        row = knee_angle_rows(document([{"frame_index": 0, "timestamp_seconds": 0, "poses": [left]}], 200, 100))[0]
        self.assertAlmostEqual(row.left_knee_angle_deg or 0, 90.0, places=6)
        self.assertEqual(row.left_knee_status, "ok")

    def test_calculates_both_sides_and_keeps_each_frame_and_null_timestamp(self) -> None:
        left = pose("left")
        right = pose("right", ((0, 0.5), (0.5, 0.5), (1, 0.5)))
        frames = [
            {"frame_index": 7, "timestamp_seconds": None, "poses": [left, pose("left", ((0, 0), (1, 1), (0, 1)))]},
            {"frame_index": 2, "timestamp_seconds": 0.2, "poses": [right]},
        ]
        rows = knee_angle_rows(document(frames))
        self.assertEqual([row.frame_index for row in rows], [7, 2])
        self.assertEqual([row.timestamp_seconds for row in rows], [None, 0.2])
        self.assertAlmostEqual(rows[0].left_knee_angle_deg or 0, 90.0)
        self.assertEqual(rows[0].right_knee_status, "missing_landmark")
        self.assertAlmostEqual(rows[1].right_knee_angle_deg or 0, 180.0)

    def test_missing_pose_is_retained_with_no_pose_status(self) -> None:
        rows = knee_angle_rows(document([{"frame_index": 3, "timestamp_seconds": None, "poses": []}]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].left_knee_status, "no_pose")
        self.assertEqual(rows[0].right_knee_status, "no_pose")
        self.assertIsNone(rows[0].left_knee_angle_deg)

    def test_missing_landmark_invalid_coordinates_and_missing_confidence(self) -> None:
        missing = {"landmarks": {"left_hip": point(0, 1), "left_knee": point(0, 0)}}
        invalid = pose("left")
        invalid["landmarks"]["left_knee"]["x"] = float("inf")
        no_confidence = pose("left")
        no_confidence["landmarks"]["left_ankle"]["visibility"] = None
        rows = knee_angle_rows(document([
            {"poses": [missing]}, {"poses": [invalid]}, {"poses": [no_confidence]},
        ]))
        self.assertEqual([row.left_knee_status for row in rows], [
            "missing_landmark", "invalid_coordinate", "missing_confidence",
        ])
        self.assertTrue(all(row.left_knee_angle_deg is None for row in rows))

    def test_visibility_and_presence_thresholds_are_enforced(self) -> None:
        low_visibility = pose("left")
        low_visibility["landmarks"]["left_hip"]["visibility"] = 0.49
        low_presence = pose("right")
        low_presence["landmarks"]["right_knee"]["presence"] = 0.49
        row = knee_angle_rows(document([{"poses": [
            {"landmarks": {**low_visibility["landmarks"], **low_presence["landmarks"]}}
        ]}]))[0]
        self.assertEqual(row.left_knee_status, "low_confidence")
        self.assertEqual(row.right_knee_status, "low_confidence")
        self.assertEqual(knee_angle_rows(document([{"poses": [low_visibility]}]), 0.4)[0].left_knee_status, "ok")

    def test_invalid_confidence_is_not_treated_as_missing_or_valid(self) -> None:
        invalid_confidence = pose("left")
        invalid_confidence["landmarks"]["left_knee"]["presence"] = float("nan")
        row = knee_angle_rows(document([{"poses": [invalid_confidence]}]))[0]
        self.assertEqual(row.left_knee_status, "invalid_confidence")

    def test_degenerate_geometry_and_bad_dimensions_are_unavailable(self) -> None:
        degenerate = pose("left", ((0.5, 0.5), (0.5, 0.5), (1, 0.5)))
        row = knee_angle_rows(document([{"poses": [degenerate]}]))[0]
        self.assertEqual(row.left_knee_status, "degenerate_geometry")
        bad_dimensions = knee_angle_rows(document([{"poses": [pose()]}], width=0))[0]
        self.assertEqual(bad_dimensions.left_knee_status, "invalid_coordinate")

    def test_rejects_thresholds_outside_unit_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            knee_angle_rows(document([]), visibility_threshold=1.1)


class KneeAngleCsvTests(unittest.TestCase):
    def test_csv_has_expected_columns_and_blank_unavailable_angle(self) -> None:
        rows = knee_angle_rows(document([
            {"frame_index": 0, "timestamp_seconds": None, "poses": []},
        ]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subdir" / "angles.csv"
            write_knee_angles_csv(rows, path)
            with path.open(encoding="utf-8", newline="") as stream:
                parsed = list(csv.DictReader(stream))
            self.assertEqual(tuple(parsed[0]), KNEE_ANGLE_COLUMNS)
            self.assertEqual(parsed[0]["frame_index"], "0")
            self.assertEqual(parsed[0]["timestamp_seconds"], "")
            self.assertEqual(parsed[0]["left_knee_angle_deg"], "")
            self.assertEqual(parsed[0]["left_knee_status"], "no_pose")


if __name__ == "__main__":
    unittest.main()
