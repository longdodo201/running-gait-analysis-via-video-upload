from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
import json

import app


class FakeUpload:
    name = "test run.mp4"

    def __init__(self, content: bytes = b"synthetic mp4 bytes") -> None:
        self._content = content

    def getbuffer(self) -> memoryview:
        return memoryview(self._content)


def fake_result(output_dir: Path) -> SimpleNamespace:
    output_dir.mkdir(parents=True)
    artifact_names = (
        "runner_annotated.mp4",
        "runner_landmarks.json",
        "runner_knee_angles_phase22.csv",
        "runner_knee_angles_phase23.csv",
        "runner_knee_angles_phase23.svg",
        "runner_gait_events_phase24.csv",
        "runner_gait_events_phase24.svg",
    )
    paths = {}
    for name in artifact_names:
        path = output_dir / name
        if name.endswith("_landmarks.json"):
            path.write_text(
                json.dumps({
                    "video": {"width": 720, "height": 912},
                    "frames": [{
                        "frame_index": 0,
                        "timestamp_seconds": 0.0,
                        "poses": [{"landmarks": {
                            "left_ankle": {"x": 0.25, "y": 0.5, "visibility": 0.9, "presence": 0.9},
                            "right_ankle": {"x": 0.75, "y": 0.5, "visibility": 0.9, "presence": 0.9},
                        }}],
                    }],
                }),
                encoding="utf-8",
            )
        else:
            path.write_text("<svg/>" if path.suffix == ".svg" else "artifact", encoding="utf-8")
        paths[name] = path
    quality = SimpleNamespace(usable_angle_count=8, missing_angle_count=2, large_change_flags=())
    return SimpleNamespace(
        annotated_video=paths["runner_annotated.mp4"],
        landmarks_json=paths["runner_landmarks.json"],
        knee_angles_csv=paths["runner_knee_angles_phase22.csv"],
        smoothed_knee_angles_csv=paths["runner_knee_angles_phase23.csv"],
        knee_angles_svg=paths["runner_knee_angles_phase23.svg"],
        gait_events_csv=paths["runner_gait_events_phase24.csv"],
        gait_events_svg=paths["runner_gait_events_phase24.svg"],
        left_quality=quality,
        right_quality=quality,
        event_interpretation="Exploratory heel-trajectory candidates; not confirmed foot-strikes.",
    )


class UploadedVideoFlowTests(unittest.TestCase):
    def test_upload_is_analyzed_once_in_a_unique_output_and_temp_file_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            observed_inputs: list[Path] = []
            observed_outputs: list[Path] = []

            def analyzer(video_path, output_dir, *, model_path, progress):
                video = Path(video_path)
                output = Path(output_dir)
                self.assertEqual(video.name, "test run.mp4")
                self.assertEqual(video.read_bytes(), b"synthetic mp4 bytes")
                self.assertEqual(model_path, root / "pose.task")
                self.assertFalse(output.exists())
                observed_inputs.append(video)
                observed_outputs.append(output)
                output.mkdir(parents=True)
                (output / "keep.txt").write_text("preserved", encoding="utf-8")
                return output

            first = app.process_uploaded_video(
                FakeUpload(), output_root=root / "runs", model_path=root / "pose.task", analyzer=analyzer
            )
            second = app.process_uploaded_video(
                FakeUpload(), output_root=root / "runs", model_path=root / "pose.task", analyzer=analyzer
            )

            self.assertEqual(len(observed_inputs), 2)
            self.assertFalse(observed_inputs[0].exists())
            self.assertFalse(observed_inputs[1].exists())
            self.assertNotEqual(first, second)
            self.assertEqual((first / "keep.txt").read_text(encoding="utf-8"), "preserved")

    def test_matching_video_upload_uses_existing_ground_truth_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ground_truth = root / "foot_strike_ground_truth.csv"
            ground_truth.write_text("header\n", encoding="utf-8")
            observed = {}

            def analyzer(video_path, output_dir, *, model_path, progress, ground_truth_path):
                observed["ground_truth_path"] = Path(ground_truth_path)
                observed["video_name"] = Path(video_path).name
                return output_dir

            upload = FakeUpload()
            upload.name = "running_gait_sideview.mp4"
            with mock.patch.object(app, "GROUND_TRUTH_PATH", ground_truth):
                app.process_uploaded_video(upload, output_root=root / "runs", analyzer=analyzer)
            self.assertEqual(observed["ground_truth_path"], ground_truth)
            self.assertEqual(observed["video_name"], "running_gait_sideview.mp4")

    def test_non_mp4_and_empty_upload_are_rejected_before_analysis(self) -> None:
        analyzer = mock.Mock()
        invalid = FakeUpload()
        invalid.name = "clip.mov"
        with self.assertRaisesRegex(ValueError, "MP4"):
            app.process_uploaded_video(invalid, analyzer=analyzer)
        with self.assertRaisesRegex(ValueError, "empty"):
            app.process_uploaded_video(FakeUpload(b""), analyzer=analyzer)
        analyzer.assert_not_called()

    def test_streamlit_upload_button_renders_mocked_analysis_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            st = mock.MagicMock()
            st.session_state = {}
            st.file_uploader.return_value = FakeUpload()
            st.button.return_value = True
            st.columns.return_value = [mock.MagicMock(), mock.MagicMock()]
            st.expander.return_value.__enter__.return_value = None
            transcode = mock.patch.object(
                app,
                "transcode_overlay_to_h264",
                side_effect=app.VideoTranscodeError("synthetic encoder unavailable"),
            )

            calls = []

            def analyzer(video_path, output_dir, *, model_path, progress):
                calls.append((Path(video_path), Path(output_dir)))
                self.assertTrue(Path(video_path).is_file())
                return fake_result(Path(output_dir))

            with transcode as transcode_mock:
                app.render_app(st, output_root=root / "runs", model_path=root / "pose.task", analyzer=analyzer)

            self.assertEqual(len(calls), 1)
            self.assertFalse(calls[0][0].exists())
            transcode_mock.assert_called_once_with(calls[0][1] / "runner_annotated.mp4")
            st.video.assert_called_once_with(b"artifact", format="video/mp4")
            self.assertEqual(st.html.call_count, 0)
            self.assertTrue(
                any("<svg/>" in str(call.args[0]) for call in st.markdown.call_args_list if call.args)
            )
            self.assertEqual(st.download_button.call_count, 7)
            self.assertIn(app.RESULT_STATE_KEY, st.session_state)
            self.assertIsNone(st.session_state[app.PLAYBACK_VIDEO_STATE_KEY])
            self.assertIn("synthetic encoder unavailable", st.session_state[app.PLAYBACK_ERROR_STATE_KEY])
            st.success.assert_called_once_with("Analysis complete.")

    def test_h264_overlay_is_played_and_original_is_still_downloadable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            st = mock.MagicMock()
            st.session_state = {}
            st.file_uploader.return_value = FakeUpload()
            st.button.return_value = True
            st.columns.return_value = [mock.MagicMock(), mock.MagicMock()]
            st.expander.return_value.__enter__.return_value = None

            def analyzer(_video_path, output_dir, *, model_path, progress):
                return fake_result(Path(output_dir))

            def create_h264_overlay(source_path):
                h264 = Path(source_path).with_name("runner_annotated_h264.mp4")
                h264.write_bytes(b"h264-artifact")
                return h264

            with mock.patch.object(app, "transcode_overlay_to_h264", side_effect=create_h264_overlay):
                app.render_app(st, output_root=root / "runs", model_path=root / "pose.task", analyzer=analyzer)

            self.assertEqual(st.video.call_args.args[0], b"h264-artifact")
            self.assertIn("Skeleton overlay (H.264 MP4)", [call.args[0] for call in st.download_button.call_args_list])
            self.assertIn("Original skeleton overlay (MP4V MP4)", [call.args[0] for call in st.download_button.call_args_list])
            self.assertEqual(st.download_button.call_count, 8)

    def test_missing_empty_and_unreadable_overlay_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            st = mock.MagicMock()

            missing = app._read_overlay_video(st, root / "missing.mp4")
            self.assertIsNone(missing)
            self.assertIn("missing", st.warning.call_args.args[0].lower())

            empty = root / "empty.mp4"
            empty.write_bytes(b"")
            self.assertIsNone(app._read_overlay_video(st, empty))
            self.assertIn("empty", st.warning.call_args.args[0].lower())

            unreadable = root / "unreadable.mp4"
            unreadable.write_bytes(b"video")
            with mock.patch.object(Path, "read_bytes", side_effect=OSError("access denied")):
                self.assertIsNone(app._read_overlay_video(st, unreadable))
            self.assertIn("could not be read", st.warning.call_args.args[0].lower())

    def test_streamlit_video_render_error_keeps_download_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = fake_result(root / "result")
            st = mock.MagicMock()
            st.columns.return_value = [mock.MagicMock(), mock.MagicMock()]
            st.expander.return_value.__enter__.return_value = None
            st.video.side_effect = RuntimeError("unsupported media")

            app._show_result(st, result)

            self.assertEqual(st.download_button.call_count, 7)
            warning_messages = [call.args[0].lower() for call in st.warning.call_args_list]
            self.assertTrue(any("download below" in message for message in warning_messages))

    def test_mp4_encoder_failure_has_actionable_message(self) -> None:
        message = app._analysis_error_message(
            RuntimeError("Landmark extraction failed: OpenCV could not initialize the MP4 writer (mp4v codec).")
        )
        self.assertIn("could not be encoded", message)
        self.assertIn("MP4V encoder", message)

    def test_dashboard_keeps_unavailable_metrics_explicit_and_side_colors_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = fake_result(Path(temporary) / "result")
            st = mock.MagicMock()
            st.columns.return_value = [mock.MagicMock(), mock.MagicMock()]
            st.expander.return_value.__enter__.return_value = None

            app._show_result(st, result)

            dashboard_markup = "\n".join(
                str(call.args[0]) for call in st.markdown.call_args_list if call.args
            )
            self.assertIn("Cadence", dashboard_markup)
            self.assertIn("Knee angle at foot-strike", dashboard_markup)
            self.assertIn("Ankle path", dashboard_markup)
            ankle_component = st.iframe.call_args.args[0]
            self.assertIn("2D ankle trajectory in image coordinates", ankle_component)
            self.assertIn("timeupdate", ankle_component)
            self.assertIn("seeked", ankle_component)
            self.assertEqual(dashboard_markup.count("Not available"), 2)
            self.assertTrue(any("image coordinates in source-video pixels" in str(call.args[0]) for call in st.caption.call_args_list))
            self.assertIn("#39d5e6", app.DASHBOARD_STYLES)
            self.assertIn("#ff9d55", app.DASHBOARD_STYLES)
            self.assertIn("@media (max-width: 760px)", app.DASHBOARD_STYLES)
            self.assertTrue(
                any("experimental candidates only" in str(call.args[0]) for call in st.warning.call_args_list)
            )

    def test_dashboard_displays_ground_truth_metrics_and_algorithm_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = fake_result(root / "result")
            metrics_path = root / "metrics.json"
            metrics_path.write_text(json.dumps({
                "cadence": {"average_cadence_steps_per_minute": 173.8,
                            "step_count": 31, "clip_duration_seconds": 10.7},
                "knee_angle_at_ground_truth": {
                    "left": {"event_count": 16, "valid_event_count": 16, "mean_knee_angle_deg": 147.2},
                    "right": {"event_count": 15, "valid_event_count": 15, "mean_knee_angle_deg": 146.0},
                },
                "evaluation": [{"algorithm": "A_heel_local_extrema", "side": "left", "tp": 13,
                                "fp": 1, "fn": 3, "precision": 0.9, "recall": 0.8, "f1": 0.85,
                                "mean_absolute_error_frames": 0.8, "mean_absolute_error_ms": 27.0}],
            }), encoding="utf-8")
            result.ground_truth_metrics_json = metrics_path
            st = mock.MagicMock()
            st.columns.return_value = [mock.MagicMock(), mock.MagicMock()]
            st.expander.return_value.__enter__.return_value = None

            app._show_result(st, result)

            markup = "\n".join(str(call.args[0]) for call in st.markdown.call_args_list if call.args)
            self.assertIn("173.8 steps/min", markup)
            self.assertIn("L 147.2° · R 146.0°", markup)
            self.assertIn("A_heel_local_extrema", markup)
            self.assertNotIn("Not available", markup)

    def test_dashboard_svg_uses_responsive_dark_presentation_without_rewriting_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            svg_path = Path(temporary) / "angles.svg"
            original = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="760">'
                '<rect fill="white"/><style>text{fill:#222}.grid{stroke:#ddd}.axis{stroke:#555}'
                '</style><path stroke="#39d5e6"/></svg>'
            )
            svg_path.write_text(original, encoding="utf-8")

            rendered = app._dashboard_knee_svg(svg_path)

            self.assertIn('style="width:100%;height:auto;max-width:100%;display:block"', rendered)
            self.assertIn('fill="#0d1c27"', rendered)
            self.assertIn("fill:#e8f1f4", rendered)
            self.assertEqual(svg_path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
