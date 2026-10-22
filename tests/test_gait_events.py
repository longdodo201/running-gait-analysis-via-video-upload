from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from src.gait_events import EventDetectionParameters, detect_leg_cycle_candidates
from src.visualization import write_gait_event_diagnostic_svg


def make_document(
    count: int = 80,
    peaks: tuple[int, ...] = (12, 32, 52),
    noise: tuple[float, ...] | None = None,
) -> dict[str, object]:
    frames = []
    for index in range(count):
        bump = max((max(0.0, 55.0 - 8.0 * abs(index - peak)) for peak in peaks), default=0.0)
        jitter = noise[index % len(noise)] if noise else 0.0
        y_px = 650.0 + bump + jitter
        landmarks = {
            "left_heel": {"x": 0.5, "y": y_px / 1000.0, "visibility": 0.95, "presence": 0.98},
            "right_heel": {"x": 0.5, "y": (y_px + 3.0) / 1000.0, "visibility": 0.94, "presence": 0.97},
        }
        frames.append({
            "frame_index": index,
            "timestamp_seconds": index / 30.0,
            "poses": [{"landmarks": landmarks}],
        })
    return {"video": {"height": 1000, "fps": 30}, "frames": frames}


class GaitEventTests(unittest.TestCase):
    def test_event_plot_uses_cyan_and_orange_for_left_and_right(self) -> None:
        result = detect_leg_cycle_candidates(make_document())
        with tempfile.TemporaryDirectory() as directory:
            svg_path = write_gait_event_diagnostic_svg(
                result.samples, result.events, [], Path(directory) / "events.svg"
            )
            svg = svg_path.read_text(encoding="utf-8")
        self.assertIn('stroke="#39d5e6"', svg)
        self.assertIn('stroke="#ff9d55"', svg)
        self.assertIn('fill="#39d5e6"', svg)
        self.assertIn('fill="#ff9d55"', svg)

    def test_synthetic_events_preserve_side_and_source_times(self) -> None:
        result = detect_leg_cycle_candidates(make_document())
        left = [event for event in result.events if event.side == "left"]
        right = [event for event in result.events if event.side == "right"]
        self.assertEqual([event.frame_index for event in left], [12, 32, 52])
        self.assertEqual([event.frame_index for event in right], [12, 32, 52])
        self.assertAlmostEqual(left[0].timestamp_seconds, 12 / 30)
        self.assertEqual(left[0].signal_name, "left_heel_y_px_median5")
        self.assertGreater(left[0].prominence_px, 20)
        self.assertEqual(left[0].flags, "exploratory_unvalidated")

    def test_noisy_signal_retains_known_candidates(self) -> None:
        result = detect_leg_cycle_candidates(make_document(noise=(-3, 2, 1, -2, 0, 3)))
        left_frames = [event.frame_index for event in result.events if event.side == "left"]
        self.assertEqual(len(left_frames), 3)
        for actual, expected in zip(left_frames, (12, 32, 52)):
            self.assertLessEqual(abs(actual - expected), 1)

    def test_low_confidence_sample_is_gap_and_not_event(self) -> None:
        document = make_document(peaks=(12, 32, 52))
        frames = document["frames"]
        assert isinstance(frames, list)
        landmark = frames[32]["poses"][0]["landmarks"]["left_heel"]
        landmark["visibility"] = 0.1
        result = detect_leg_cycle_candidates(document)
        sample = result.samples[32]
        self.assertEqual(sample.left_status, "low_confidence")
        self.assertIsNone(sample.left_heel_y_px)
        self.assertIsNone(sample.left_heel_y_smoothed_px)
        self.assertNotIn(32, [event.frame_index for event in result.events if event.side == "left"])

    def test_missing_samples_split_segments_and_keep_other_events(self) -> None:
        document = make_document(peaks=(12, 52))
        frames = document["frames"]
        assert isinstance(frames, list)
        del frames[32]["poses"][0]["landmarks"]["left_heel"]
        result = detect_leg_cycle_candidates(document)
        self.assertEqual(result.samples[32].left_status, "missing_landmark")
        frames_found = [event.frame_index for event in result.events if event.side == "left"]
        self.assertEqual(frames_found, [12, 52])

    def test_insufficient_data_returns_no_events(self) -> None:
        document = make_document(count=4, peaks=())
        result = detect_leg_cycle_candidates(document)
        self.assertEqual(result.events, ())

    def test_minimum_separation_suppresses_nearby_candidates(self) -> None:
        document = make_document(count=70, peaks=(20, 29, 52))
        params = EventDetectionParameters(minimum_separation_frames=12)
        result = detect_leg_cycle_candidates(document, params)
        left = [event.frame_index for event in result.events if event.side == "left"]
        self.assertTrue(all(second - first >= 12 for first, second in zip(left, left[1:])))

    def test_short_and_invalid_parameters_are_reported(self) -> None:
        with self.assertRaises(ValueError):
            detect_leg_cycle_candidates(make_document(), EventDetectionParameters(smoothing_window_frames=4))
        with self.assertRaises(ValueError):
            detect_leg_cycle_candidates(make_document(), EventDetectionParameters(confidence_threshold=1.1))


if __name__ == "__main__":
    unittest.main()
