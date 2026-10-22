"""Mathematical tests for the 2D interior-angle helper."""

from __future__ import annotations

import math
import unittest

from src.landmarks import angle_2d_degrees


class Angle2DTests(unittest.TestCase):
    def test_right_angle_is_90_degrees(self) -> None:
        self.assertAlmostEqual(angle_2d_degrees((0, 1), (0, 0), (1, 0)), 90.0)

    def test_straight_angle_is_180_degrees(self) -> None:
        self.assertAlmostEqual(angle_2d_degrees((-1, 0), (0, 0), (1, 0)), 180.0)

    def test_rejects_zero_length_segment(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero length"):
            angle_2d_degrees((0, 0), (0, 0), (1, 0))

    def test_rejects_non_finite_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            angle_2d_degrees((math.nan, 1), (0, 0), (1, 0))

    def test_rejects_non_numeric_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "numeric"):
            angle_2d_degrees(("bad", 1), (0, 0), (1, 0))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
