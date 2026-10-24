"""Tests for event-candidate cadence accounting and explicit insufficient-data states."""

from __future__ import annotations

import unittest

from src.cadence import CandidateCadenceParameters, estimate_candidate_cadence
from src.gait_events import CandidateEvent


def candidate(side: str, frame: int, time: float | None, prominence: float = 30.0) -> CandidateEvent:
    return CandidateEvent(
        side=side,
        event_type="heel_vertical_low_point_candidate",
        frame_index=frame,
        timestamp_seconds=time,
        signal_name=f"{side}_heel_y_px",
        signal_value_px=800.0,
        prominence_px=prominence,
        landmark_visibility=0.9,
        landmark_presence=0.9,
    )


class CandidateCadenceTests(unittest.TestCase):
    def test_uses_video_duration_and_suppresses_duplicates_across_sides(self) -> None:
        events = (
            candidate("left", 10, 0.333, 30),
            candidate("right", 11, 0.367, 20),  # near-simultaneous duplicate
            candidate("right", 20, 0.667, 35),
            candidate("left", 30, 1.0, 32),
            candidate("right", 40, 1.333, 28),
            candidate("left", 50, 1.667, 31),
        )
        frames = [{"frame_index": i, "timestamp_seconds": i / 30} for i in range(60)]
        series, summary = estimate_candidate_cadence(
            events, frames=frames, fps=30.0, duration_seconds=2.0,
        )
        self.assertEqual(summary["status"], "insufficient_duration")
        self.assertEqual(summary["candidate_event_count"], 6)
        self.assertEqual(summary["retained_event_count"], 5)
        self.assertIsNone(summary["average_cadence_steps_per_minute"])
        self.assertEqual(series, [])

    def test_calculates_cadence_from_actual_timestamps_without_ground_truth(self) -> None:
        events = tuple(
            candidate("left" if index % 2 == 0 else "right", index * 9, index * 0.3)
            for index in range(10)
        )
        frames = [{"frame_index": i, "timestamp_seconds": i / 30} for i in range(300)]
        series, summary = estimate_candidate_cadence(
            events, frames=frames, fps=30.0, duration_seconds=10.0,
        )
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["retained_event_count"], 10)
        self.assertAlmostEqual(summary["average_cadence_steps_per_minute"], 60.0)
        self.assertEqual(summary["duration_seconds"], 10.0)
        self.assertEqual(len(series), 17)

    def test_uses_frame_index_and_fps_when_event_timestamp_is_missing(self) -> None:
        events = tuple(candidate("left", frame, None) for frame in (0, 15, 30, 45))
        frames = [{"frame_index": i, "timestamp_seconds": None} for i in range(90)]
        _, summary = estimate_candidate_cadence(
            events, frames=frames, fps=30.0, duration_seconds=None,
        )
        self.assertEqual(summary["status"], "ok")
        self.assertAlmostEqual(summary["average_cadence_steps_per_minute"], 80.0)
        self.assertEqual(summary["duration_seconds"], 3.0)

    def test_uses_actual_fps_and_timeline_for_a_different_resolution_and_frame_rate(self) -> None:
        fps = 60.0
        frames = [{"frame_index": i, "timestamp_seconds": i / fps} for i in range(360)]
        events = tuple(
            candidate("left" if index % 2 == 0 else "right", index * 24, index * 0.4)
            for index in range(10)
        )
        _, summary = estimate_candidate_cadence(
            events, frames=frames, fps=fps, duration_seconds=len(frames) / fps,
        )
        self.assertEqual(summary["fps"], fps)
        self.assertEqual(summary["duration_seconds"], 6.0)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["average_cadence_steps_per_minute"], 100.0)

    def test_insufficient_events_and_missing_timing_are_not_reported_as_zero_cadence(self) -> None:
        frames = [{"frame_index": i, "timestamp_seconds": None} for i in range(90)]
        _, too_few = estimate_candidate_cadence(
            (candidate("left", 20, 2 / 3),), frames=frames, fps=30, duration_seconds=3,
        )
        self.assertEqual(too_few["status"], "insufficient_events")
        self.assertIsNone(too_few["average_cadence_steps_per_minute"])

        _, no_timing = estimate_candidate_cadence(
            (candidate("left", 20, None),), frames=frames, fps=None, duration_seconds=None,
        )
        self.assertEqual(no_timing["status"], "timing_unavailable")
        self.assertIsNone(no_timing["average_cadence_steps_per_minute"])

    def test_separation_is_configurable(self) -> None:
        events = (candidate("left", 10, 1.0), candidate("right", 11, 1.1))
        params = CandidateCadenceParameters(minimum_event_separation_seconds=0.05)
        _, summary = estimate_candidate_cadence(
            events, frames=[], fps=30, duration_seconds=5, parameters=params,
        )
        self.assertEqual(summary["retained_event_count"], 2)
        self.assertEqual(summary["minimum_event_separation_seconds"], 0.05)


if __name__ == "__main__":
    unittest.main()
