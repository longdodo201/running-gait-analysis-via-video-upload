"""Offline tests for Phase 1 video handling and landmark serialization."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from src.landmarks import LANDMARK_NAMES, Landmark, Pose, pose_from_mediapipe
from src.pipeline import ExtractionError, NoPoseDetectedError, extract_video, timestamp_for_frame
from src.video_io import VideoError, VideoLimits, iter_video_frames, validate_video


class FakeCapture:
    def __init__(self, frames: list[object], fps: float = 25.0, opened: bool = True, reported_count: int | None = None) -> None:
        self.frames = frames
        self.fps = fps
        self.opened = opened
        self.reported_count = reported_count
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop: int) -> float:
        return {3: 640.0, 4: 480.0, 5: self.fps, 7: float(self.reported_count if self.reported_count is not None else len(self.frames))}[prop]

    def set(self, prop: int, value: float) -> bool:
        self.position = int(value)
        return True

    def read(self) -> tuple[bool, object | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class FakeEstimator:
    def __init__(self, poses: tuple[Pose, ...] = ()) -> None:
        self.poses = poses
        self.closed = False
        self.fail_on_frame: int | None = None

    def detect(self, frame: object, frame_index: int) -> tuple[Pose, ...]:
        if frame_index == self.fail_on_frame:
            raise RuntimeError("synthetic inference failure")
        return self.poses

    def close(self) -> None:
        self.closed = True


class FakeWriter:
    def __init__(self) -> None:
        self.frames: list[tuple[object, tuple[Pose, ...]]] = []
        self.closed_with: bool | None = None

    def write(self, frame: object, poses: tuple[Pose, ...]) -> None:
        self.frames.append((frame, poses))

    def close(self, success: bool) -> None:
        self.closed_with = success


def fake_pose() -> Pose:
    return Pose({name: Landmark(0.1, 0.2, 0.0, 0.9, 0.8) for name in LANDMARK_NAMES})


class VideoIoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video = Path(self.temp_dir.name) / "clip.mp4"
        self.video.write_bytes(b"fake video bytes")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_missing_and_unsupported_files_are_rejected(self) -> None:
        with self.assertRaisesRegex(VideoError, "does not exist"):
            validate_video(self.video.with_name("missing.mp4"), capture_factory=lambda _: None)
        avi = self.video.with_suffix(".avi")
        avi.write_bytes(b"x")
        with self.assertRaisesRegex(VideoError, "Unsupported video format"):
            validate_video(avi, capture_factory=lambda _: None)

    def test_size_limit_and_unreadable_video_are_rejected_and_released(self) -> None:
        with self.assertRaisesRegex(VideoError, "configured limit"):
            validate_video(self.video, VideoLimits(max_bytes=1), capture_factory=lambda _: None)
        capture = FakeCapture([], opened=False)
        with self.assertRaisesRegex(VideoError, "cannot open"):
            validate_video(self.video, capture_factory=lambda _: capture)
        self.assertTrue(capture.released)

    def test_invalid_fps_is_unknown_and_first_frame_must_decode(self) -> None:
        no_fps = FakeCapture([object()], fps=float("nan"))
        _, metadata, returned = validate_video(self.video, capture_factory=lambda _: no_fps)
        self.assertIsNone(metadata.fps)
        self.assertIsNone(metadata.duration_seconds)
        returned.release()
        with self.assertRaisesRegex(VideoError, "first frame"):
            validate_video(self.video, capture_factory=lambda _: FakeCapture([]))

    def test_frame_iterator_releases_capture_on_failure(self) -> None:
        capture = FakeCapture([object()])
        self.assertEqual(list(iter_video_frames(capture)), [(0, capture.frames[0])])
        self.assertTrue(capture.released)


class SerializationAndPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.video = self.root / "clip.mp4"
        self.video.write_bytes(b"synthetic")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def factory(self, capture: FakeCapture):
        return lambda _: capture

    def test_duration_limit_is_enforced_using_decoded_frame_count(self) -> None:
        capture = FakeCapture([object(), object()], fps=1.0, reported_count=1)
        estimator = FakeEstimator((fake_pose(),))
        with self.assertRaisesRegex(VideoError, "Decoded video duration"):
            extract_video(
                self.video, "unused.task", self.root / "too-long.json",
                limits=VideoLimits(max_bytes=None, max_duration_seconds=1.0),
                capture_factory=self.factory(capture), estimator_factory=lambda *_: estimator,
                progress=lambda _: None,
            )
        self.assertFalse((self.root / "too-long.json").exists())
        self.assertTrue(capture.released)
        self.assertTrue(estimator.closed)

    def test_timestamps_only_exist_for_valid_fps(self) -> None:
        self.assertEqual(timestamp_for_frame(15, 30.0), 0.5)
        self.assertIsNone(timestamp_for_frame(1, None))
        self.assertIsNone(timestamp_for_frame(1, 0.0))
        self.assertIsNone(timestamp_for_frame(1, float("inf")))

    def test_mediapipe_landmarks_are_named_and_optional_fields_preserved(self) -> None:
        raw = [SimpleNamespace(x=i / 33, y=0.5, z=-0.1, visibility=0.7, presence=None) for i in range(33)]
        pose = pose_from_mediapipe(raw)
        self.assertEqual(tuple(pose.landmarks), LANDMARK_NAMES)
        self.assertEqual(pose.landmarks["left_ankle"].visibility, 0.7)
        self.assertIsNone(pose.landmarks["left_ankle"].presence)
        with self.assertRaisesRegex(ValueError, "Expected 33"):
            pose_from_mediapipe(raw[:-1])

    def test_pipeline_writes_versioned_json_and_releases_resources(self) -> None:
        capture = FakeCapture([object(), object()], fps=20.0)
        estimator = FakeEstimator((fake_pose(),))
        output = self.root / "out" / "landmarks.json"
        document = extract_video(
            self.video, "unused.task", output,
            capture_factory=self.factory(capture),
            estimator_factory=lambda _model, _fps: estimator,
            progress=lambda _: None,
        )
        parsed = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(document.schema_version, "1.0")
        self.assertEqual(len(parsed["frames"]), 2)
        self.assertEqual(parsed["frames"][1]["timestamp_seconds"], 0.05)
        self.assertEqual(parsed["frames"][0]["poses"][0]["landmarks"]["left_ankle"]["x"], 0.1)
        self.assertTrue(capture.released)
        self.assertTrue(estimator.closed)

    def test_optional_annotation_receives_every_frame_and_closes_on_error(self) -> None:
        capture = FakeCapture([object(), object()])
        estimator = FakeEstimator((fake_pose(),))
        writer = FakeWriter()
        extract_video(
            self.video, "unused.task", self.root / "annotated.json", self.root / "annotated.mp4",
            capture_factory=self.factory(capture), estimator_factory=lambda *_: estimator,
            annotation_factory=lambda *_: writer, progress=lambda _: None,
        )
        self.assertEqual(len(writer.frames), 2)
        self.assertTrue(writer.closed_with)

        capture2 = FakeCapture([object()])
        estimator2 = FakeEstimator((fake_pose(),))
        estimator2.fail_on_frame = 0
        failed_writer = FakeWriter()
        with self.assertRaises(ExtractionError):
            extract_video(
                self.video, "unused.task", self.root / "annotated-failure.json", self.root / "annotated-failure.mp4",
                capture_factory=self.factory(capture2), estimator_factory=lambda *_: estimator2,
                annotation_factory=lambda *_: failed_writer, progress=lambda _: None,
            )
        self.assertFalse(failed_writer.closed_with)
        self.assertTrue(capture2.released)
        self.assertTrue(estimator2.closed)

    def test_no_pose_and_inference_failure_report_errors_and_cleanup(self) -> None:
        capture = FakeCapture([object()])
        estimator = FakeEstimator()
        with self.assertRaises(NoPoseDetectedError):
            extract_video(
                self.video, "unused.task", self.root / "none.json",
                capture_factory=self.factory(capture), estimator_factory=lambda *_: estimator,
                progress=lambda _: None,
            )
        self.assertFalse((self.root / "none.json").exists())
        self.assertTrue(capture.released)
        self.assertTrue(estimator.closed)

        capture2 = FakeCapture([object()])
        estimator2 = FakeEstimator((fake_pose(),))
        estimator2.fail_on_frame = 0
        with self.assertRaisesRegex(Exception, "inference failure"):
            extract_video(
                self.video, "unused.task", self.root / "failure.json",
                capture_factory=self.factory(capture2), estimator_factory=lambda *_: estimator2,
                progress=lambda _: None,
            )
        self.assertTrue(capture2.released)
        self.assertTrue(estimator2.closed)


if __name__ == "__main__":
    unittest.main()
