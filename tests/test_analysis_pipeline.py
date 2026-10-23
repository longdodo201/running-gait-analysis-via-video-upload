"""Integration tests for orchestration using synthetic landmarks, not inference."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import src.analysis_pipeline as analysis
from src.gait_events import EventDetectionParameters
from src.landmarks import LANDMARK_NAMES, ExtractionDocument, Landmark, LandmarkFrame, Pose, write_json
from src.video_io import VideoMetadata


class FakeCapture:
    def __init__(self) -> None:
        self.released = False

    def release(self) -> None:
        self.released = True


def _pose(heel_y: float) -> Pose:
    landmarks = {}
    for side in ("left", "right"):
        landmarks.update({
            f"{side}_hip": Landmark(0.4, 0.3, 0.0, 0.95, 0.95),
            f"{side}_knee": Landmark(0.4, 0.5, 0.0, 0.95, 0.95),
            f"{side}_ankle": Landmark(0.6, 0.5, 0.0, 0.95, 0.95),
            f"{side}_heel": Landmark(0.6, heel_y, 0.0, 0.95, 0.95),
        })
    return Pose(landmarks)


def _document(frame_count: int = 15) -> ExtractionDocument:
    pattern = (0.2, 0.3, 0.7, 0.3, 0.2)
    frames = tuple(
        LandmarkFrame(index, index / 30, (_pose(pattern[index % len(pattern)]),))
        for index in range(frame_count)
    )
    return ExtractionDocument(
        schema_version="1.0",
        source_video="runner.mp4",
        video={"width": 640, "height": 480, "fps": 30.0, "frame_count": frame_count,
               "duration_seconds": frame_count / 30, "warnings": ()},
        timestamp_basis="frame_index / source_fps",
        warnings=(),
        frames=frames,
    )


def _cacheable_document() -> ExtractionDocument:
    document = _document()
    default_landmarks = {
        name: Landmark(0.5, 0.5, 0.0, 0.95, 0.95)
        for name in LANDMARK_NAMES
    }
    frames = tuple(
        LandmarkFrame(
            frame.frame_index,
            frame.timestamp_seconds,
            (Pose({**default_landmarks, **frame.poses[0].landmarks}),),
        )
        for frame in document.frames
    )
    return ExtractionDocument(
        schema_version=document.schema_version,
        source_video=document.source_video,
        video=document.video,
        timestamp_basis=document.timestamp_basis,
        warnings=document.warnings,
        frames=frames,
    )


class AnalysisPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "runner.mp4"
        self.video.write_bytes(b"synthetic video placeholder")
        self.model = self.root / "pose.task"
        self.model.write_bytes(b"synthetic model placeholder")
        self.output = self.root / "analysis"
        self.capture = FakeCapture()
        self.metadata = VideoMetadata(640, 480, 30.0, 15, 0.5)
        self.source_document = _document()
        self.extraction_calls = 0
        self.annotated_path: Path | None = None

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _validation(self, path, limits=None):
        return Path(path), self.metadata, self.capture

    def _extraction(self, video, model, output_json, annotated_video=None, limits=None, progress=None, **kwargs):
        self.extraction_calls += 1
        self.annotated_path = Path(annotated_video) if annotated_video is not None else None
        write_json(self.source_document, output_json)
        if self.annotated_path is not None:
            self.annotated_path.write_bytes(b"synthetic annotated video")
        return self.source_document

    def _successful_patches(self):
        return (
            mock.patch.object(analysis, "validate_video", side_effect=self._validation),
            mock.patch.object(analysis, "extract_video", side_effect=self._extraction),
        )

    def test_invalid_video_fails_before_extraction(self) -> None:
        missing = self.root / "missing.mp4"
        with mock.patch.object(analysis, "extract_video") as extraction:
            with self.assertRaises(analysis.AnalysisPipelineError) as caught:
                analysis.analyze_video(missing, self.output, self.model)
        self.assertIn("Video validation failed", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, FileNotFoundError | ValueError)
        extraction.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_output_directory_is_rejected_unchanged(self) -> None:
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing", encoding="utf-8")
        with mock.patch.object(analysis, "extract_video") as extraction:
            with self.assertRaises(FileExistsError):
                analysis.analyze_video(self.video, self.output, self.model)
        extraction.assert_not_called()
        self.assertEqual(marker.read_text(encoding="utf-8"), "existing")

    def test_pipeline_writes_seven_expected_artifacts_and_returns_diagnostics(self) -> None:
        events = EventDetectionParameters(
            confidence_threshold=0.7,
            smoothing_window_frames=3,
            minimum_prominence_pixels=0,
            minimum_separation_frames=2,
            prominence_window_frames=3,
        )
        progress = mock.Mock()
        order: list[str] = []
        original_knee = analysis.knee_angle_rows
        original_quality = analysis.diagnose_knee_angles
        original_smooth = analysis.add_smoothed_knee_angles
        original_detect = analysis.detect_leg_cycle_candidates
        original_align = analysis.validate_phase23_alignment
        original_write_phase22 = analysis.write_knee_angles_csv

        def record(name, fn):
            def wrapped(*args, **kwargs):
                order.append(name)
                return fn(*args, **kwargs)
            return wrapped

        validation, extraction = self._successful_patches()
        with validation, extraction, \
             mock.patch.object(analysis, "knee_angle_rows", side_effect=record("angles", original_knee)) as angles, \
             mock.patch.object(analysis, "diagnose_knee_angles", side_effect=record("quality", original_quality)), \
             mock.patch.object(analysis, "add_smoothed_knee_angles", side_effect=record("smooth", original_smooth)), \
             mock.patch.object(analysis, "detect_leg_cycle_candidates", side_effect=record("events", original_detect)) as detect, \
             mock.patch.object(analysis, "validate_phase23_alignment", side_effect=record("align", original_align)), \
             mock.patch.object(analysis, "write_knee_angles_csv", side_effect=record("write", original_write_phase22)):
            result = analysis.analyze_video(
                self.video,
                self.output,
                self.model,
                visibility_threshold=0.7,
                presence_threshold=0.8,
                event_parameters=events,
                progress=progress,
            )

        angles.assert_called_once()
        self.assertEqual(angles.call_args.args[1:], (0.7, 0.8))
        detect.assert_called_once_with(mock.ANY, events)
        event_document = detect.call_args.args[0]
        self.assertIsInstance(event_document, dict)
        self.assertIsInstance(event_document["frames"], list)
        self.assertIsInstance(event_document["frames"][0]["poses"], list)
        self.assertEqual(order, ["angles", "quality", "quality", "smooth", "events", "align", "write"])
        self.assertTrue(self.capture.released)
        self.assertEqual(result.input_video, self.video.resolve())
        self.assertEqual(self.extraction_calls, 1)
        self.assertEqual(result.annotated_video.name, self.annotated_path.name)
        self.assertTrue(result.annotated_video.is_file())
        self.assertIn("not confirmed foot-strikes", result.event_interpretation)
        self.assertEqual(result.left_quality.usable_angle_count, 15)
        self.assertEqual(result.right_quality.usable_angle_count, 15)
        self.assertEqual(
            result.candidate_event_count,
            result.left_candidate_event_count + result.right_candidate_event_count,
        )

        expected = {
            "runner_annotated.mp4",
            "runner_landmarks.json",
            "runner_knee_angles_phase22.csv",
            "runner_knee_angles_phase23.csv",
            "runner_knee_angles_phase23.svg",
            "runner_gait_events_phase24.csv",
            "runner_gait_events_phase24.svg",
        }
        self.assertEqual({path.name for path in self.output.iterdir()}, expected)
        with result.knee_angles_csv.open(encoding="utf-8", newline="") as stream:
            phase22 = list(csv.DictReader(stream))
        with result.smoothed_knee_angles_csv.open(encoding="utf-8", newline="") as stream:
            phase23 = list(csv.DictReader(stream))
        with result.gait_events_csv.open(encoding="utf-8", newline="") as stream:
            event_rows = list(csv.DictReader(stream))
        self.assertEqual(len(phase22), 15)
        self.assertEqual(len(phase23), 15)
        self.assertEqual(len(event_rows), result.candidate_event_count)
        self.assertEqual(tuple(phase22[0]), (
            "frame_index", "timestamp_seconds", "left_knee_angle_deg", "right_knee_angle_deg",
            "left_knee_status", "right_knee_status",
        ))
        self.assertEqual(phase23[0]["frame_index"], phase22[0]["frame_index"])
        self.assertTrue(result.knee_angles_svg.is_file())
        self.assertTrue(result.gait_events_svg.is_file())

    def test_pipeline_writes_ground_truth_analysis_without_replacing_standard_artifacts(self) -> None:
        validation, extraction = self._successful_patches()
        ground_truth = self.root / "foot_strike_ground_truth.csv"
        ground_truth.write_text(
            "event_id,frame_index,timestamp_seconds,foot_side,event_type,label_confidence,notes\n"
            "E001,2,0.066666667,left,foot_strike,,\n",
            encoding="utf-8",
        )
        with validation, extraction:
            result = analysis.analyze_video(
                self.video, self.output, self.model, ground_truth_path=ground_truth
            )
        self.assertEqual(result.ground_truth_path, ground_truth.resolve())
        self.assertTrue(result.foot_strike_evaluation_csv.is_file())
        self.assertTrue(result.foot_strike_matches_csv.is_file())
        self.assertTrue(result.knee_angles_at_ground_truth_csv.is_file())
        self.assertTrue(result.cadence_svg.is_file())
        self.assertEqual(result.foot_strike_evaluation_csv.parent, self.output.resolve())
        self.assertEqual(len(list(self.output.glob("*ground_truth*"))), 3)

    def test_alignment_failure_does_not_publish_outputs_and_preserves_cause(self) -> None:
        validation, extraction = self._successful_patches()
        failure = ValueError("synthetic alignment mismatch")
        with validation, extraction, mock.patch.object(analysis, "validate_phase23_alignment", side_effect=failure):
            with self.assertRaises(analysis.AnalysisPipelineError) as caught:
                analysis.analyze_video(self.video, self.output, self.model)
        self.assertIn("Phase 2.3/event alignment failed", str(caught.exception))
        self.assertIs(caught.exception.__cause__, failure)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".analysis.tmp-*")), [])

    def test_writer_failure_cleans_only_temporary_artifacts(self) -> None:
        validation, extraction = self._successful_patches()
        failure = OSError("synthetic CSV writer failure")
        with validation, extraction, mock.patch.object(analysis, "write_candidate_events_csv", side_effect=failure):
            with self.assertRaises(analysis.AnalysisPipelineError) as caught:
                analysis.analyze_video(self.video, self.output, self.model)
        self.assertIn("Phase 2.4 CSV writing failed", str(caught.exception))
        self.assertIs(caught.exception.__cause__, failure)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".analysis.tmp-*")), [])

    def test_extraction_failure_does_not_publish_partial_directory(self) -> None:
        validation, _ = self._successful_patches()
        failure = RuntimeError("synthetic extraction failure")
        with validation, mock.patch.object(analysis, "extract_video", side_effect=failure):
            with self.assertRaises(analysis.AnalysisPipelineError) as caught:
                analysis.analyze_video(self.video, self.output, self.model)
        self.assertIn("Landmark extraction failed", str(caught.exception))
        self.assertIs(caught.exception.__cause__, failure)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".analysis.tmp-*")), [])

    def test_cache_hit_reuses_extraction_but_recomputes_outputs(self) -> None:
        self.source_document = _cacheable_document()
        cache_dir = self.root / "cache"
        first_output = self.root / "analysis-first"
        second_output = self.root / "analysis-second"
        validation, extraction = self._successful_patches()
        with validation, extraction:
            first = analysis.analyze_video(
                self.video, first_output, self.model, cache_dir=cache_dir
            )
            progress = mock.Mock()
            second = analysis.analyze_video(
                self.video, second_output, self.model, cache_dir=cache_dir, progress=progress
            )

        self.assertEqual(self.extraction_calls, 1)
        self.assertTrue(any("cache hit" in call.args[0].lower() for call in progress.call_args_list))
        first_artifacts = {path.name: path.read_bytes() for path in first_output.iterdir()}
        second_artifacts = {path.name: path.read_bytes() for path in second_output.iterdir()}
        self.assertEqual(first_artifacts, second_artifacts)
        self.assertTrue(first.landmarks_json.is_file())
        self.assertTrue(second.annotated_video.is_file())

    def test_corrupt_cache_entry_recomputes_extraction(self) -> None:
        self.source_document = _cacheable_document()
        cache_dir = self.root / "cache"
        validation, extraction = self._successful_patches()
        with validation, extraction:
            analysis.analyze_video(
                self.video, self.root / "analysis-first", self.model, cache_dir=cache_dir
            )
            cached_landmarks = next(cache_dir.glob("*.landmarks.json"))
            cached_landmarks.write_text("corrupt", encoding="utf-8")
            result = analysis.analyze_video(
                self.video, self.root / "analysis-second", self.model, cache_dir=cache_dir
            )

        self.assertEqual(self.extraction_calls, 2)
        self.assertTrue(result.landmarks_json.is_file())
        manifest_path = next(cache_dir.glob("*.manifest.json"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_cache = cache_dir / manifest["artifacts"]["landmarks_json"]
        self.assertNotEqual(current_cache, cached_landmarks)
        self.assertEqual(json.loads(current_cache.read_text(encoding="utf-8"))["schema_version"], "1.0")


if __name__ == "__main__":
    unittest.main()
