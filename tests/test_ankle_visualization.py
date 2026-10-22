from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from src.visualization import (
    ankle_trajectory_playback_html,
    ankle_trajectory_svg,
    ankle_trajectory_timeline,
    cumulative_trajectory_path,
    frame_index_at_time,
)


def landmark(x: float, y: float, *, visibility: float = 0.9, presence: float = 0.9) -> dict:
    return {"x": x, "y": y, "visibility": visibility, "presence": presence}


def frame(index: int, left=None, right=None, timestamp: float | None = None) -> dict:
    landmarks = {}
    if left is not None:
        landmarks["left_ankle"] = left
    if right is not None:
        landmarks["right_ankle"] = right
    return {
        "frame_index": index,
        "timestamp_seconds": timestamp,
        "poses": [{"landmarks": landmarks}],
    }


def document(frames: list[dict]) -> dict:
    return {"video": {"width": 720, "height": 912, "fps": 30.0}, "frames": frames}


class AnkleTrajectorySvgTests(unittest.TestCase):
    def test_normalized_coordinates_map_to_source_pixel_positions(self) -> None:
        svg = ankle_trajectory_svg(document([
            frame(0, landmark(0.25, 0.25), landmark(0.75, 0.75), 0.0),
        ]))
        root = ET.fromstring(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        left = root.find('.//svg:circle[@cx="240.00"][@cy="260.00"]', ns)
        right = root.find('.//svg:circle[@cx="540.00"][@cy="640.00"]', ns)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        self.assertIn("Image y (pixels; increases downward)", svg)
        self.assertIn("image (180.0, 228.0) px", svg)

    def test_missing_and_low_confidence_samples_break_the_path(self) -> None:
        svg = ankle_trajectory_svg(document([
            frame(0, landmark(0.1, 0.1), landmark(0.1, 0.1)),
            frame(1, landmark(0.2, 0.2, visibility=0.2), landmark(0.2, 0.2)),
            frame(2, None, landmark(0.3, 0.3)),
            frame(3, landmark(0.4, 0.4), landmark(0.4, 0.4)),
        ]))
        root = ET.fromstring(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        left_path = root.find('.//svg:path[@data-side="left"]', ns)
        right_path = root.find('.//svg:path[@data-side="right"]', ns)
        self.assertIsNotNone(left_path)
        self.assertIsNotNone(right_path)
        self.assertEqual(left_path.attrib["d"].count("M "), 2)
        self.assertNotIn(" L ", left_path.attrib["d"])
        self.assertIn("2/4 valid in-frame samples (50.0%)", svg)
        self.assertIn("4/4 valid in-frame samples (100.0%)", svg)

    def test_left_and_right_trajectories_use_dashboard_colors(self) -> None:
        svg = ankle_trajectory_svg(document([
            frame(0, landmark(0.1, 0.1), landmark(0.9, 0.9)),
            frame(1, landmark(0.2, 0.2), landmark(0.8, 0.8)),
        ]))
        root = ET.fromstring(svg)
        ns = {"svg": "http://www.w3.org/2000/svg"}
        left = root.find('.//svg:path[@data-side="left"]', ns)
        right = root.find('.//svg:path[@data-side="right"]', ns)
        self.assertEqual(left.attrib["stroke"], "#39d5e6")
        self.assertEqual(right.attrib["stroke"], "#ff9d55")
        self.assertIn("M ", left.attrib["d"])
        self.assertIn(" L ", left.attrib["d"])

    def test_missing_pose_confidence_and_invalid_coordinates_are_excluded(self) -> None:
        invalid_frames = [
            frame(0, landmark(0.1, 0.1), landmark(0.1, 0.1)),
            {"frame_index": 1, "timestamp_seconds": 0.1, "poses": []},
            frame(2, landmark(float("nan"), 0.1), landmark(0.2, 0.2)),
            frame(3, landmark(1.2, 0.1), landmark(0.3, 0.3)),
            frame(4, {"x": 0.5, "y": 0.5, "visibility": 0.9}, landmark(0.4, 0.4)),
        ]
        svg = ankle_trajectory_svg(document(invalid_frames))
        self.assertIn("1/5 valid in-frame samples (20.0%)", svg)
        self.assertIn("4/5 valid in-frame samples (80.0%)", svg)

    def test_playback_time_maps_to_the_latest_frame_at_or_before_time(self) -> None:
        timeline = ankle_trajectory_timeline(document([
            frame(0, landmark(0.1, 0.1), landmark(0.2, 0.2), 0.0),
            frame(1, landmark(0.2, 0.2), landmark(0.3, 0.3), 0.5),
            frame(2, landmark(0.3, 0.3), landmark(0.4, 0.4), 1.0),
        ]))
        self.assertEqual(frame_index_at_time(timeline["frames"], -0.01), -1)
        self.assertEqual(frame_index_at_time(timeline["frames"], 0.49), 0)
        self.assertEqual(frame_index_at_time(timeline["frames"], 0.5), 1)
        self.assertEqual(frame_index_at_time(timeline["frames"], 8.0), 2)

    def test_cumulative_paths_grow_to_playback_frame_and_seeking_backward_truncates(self) -> None:
        timeline = ankle_trajectory_timeline(document([
            frame(0, landmark(0.1, 0.1), landmark(0.8, 0.8), 0.0),
            frame(1, landmark(0.2, 0.2), landmark(0.7, 0.7), 0.1),
            frame(2, landmark(0.3, 0.3), landmark(0.6, 0.6), 0.2),
            frame(3, landmark(0.4, 0.4), landmark(0.5, 0.5), 0.3),
        ]))
        early = cumulative_trajectory_path(timeline["frames"], "left", 1, 720, 912)
        later = cumulative_trajectory_path(timeline["frames"], "left", 3, 720, 912)
        seek_back = cumulative_trajectory_path(timeline["frames"], "left", 1, 720, 912)
        self.assertEqual(early, seek_back)
        self.assertLess(len(early), len(later))
        self.assertEqual(later.count("L "), 3)

    def test_invalid_samples_remain_gaps_in_cumulative_path(self) -> None:
        timeline = ankle_trajectory_timeline(document([
            frame(0, landmark(0.1, 0.1), landmark(0.1, 0.1), 0.0),
            frame(1, landmark(0.2, 0.2, visibility=0.1), landmark(0.2, 0.2), 0.1),
            frame(2, None, landmark(0.3, 0.3), 0.2),
            frame(3, landmark(0.4, 0.4), landmark(0.4, 0.4), 0.3),
        ]))
        self.assertIsNone(timeline["frames"][1]["left"])
        self.assertIsNone(timeline["frames"][2]["left"])
        path = cumulative_trajectory_path(timeline["frames"], "left", 3, 720, 912)
        self.assertEqual(path.count("M "), 2)
        self.assertNotIn(" L ", path)

    def test_playback_component_listens_to_native_video_without_streamlit_callbacks(self) -> None:
        html = ankle_trajectory_playback_html(document([
            frame(0, landmark(0.2, 0.4), landmark(0.8, 0.6), 0.0),
        ]))
        self.assertIn("parentDocument.querySelector", html)
        self.assertIn("timeupdate", html)
        self.assertIn("seeking", html)
        self.assertIn("seeked", html)
        self.assertIn("pause", html)
        self.assertIn("#39d5e6", html)
        self.assertIn("#ff9d55", html)


if __name__ == "__main__":
    unittest.main()
