from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from src.landmark_cache import LandmarkExtractionCache, landmark_cache_key
from src.landmarks import LANDMARK_NAMES, ExtractionDocument, Landmark, LandmarkFrame, Pose, write_json
from src.video_io import VideoMetadata


def _document(source_name: str, metadata: VideoMetadata) -> ExtractionDocument:
    points = {
        name: Landmark(0.5, 0.5, 0.0, 0.9, 0.9)
        for name in LANDMARK_NAMES
    }
    pose = Pose(points)
    frames = tuple(
        LandmarkFrame(index, index / metadata.fps, (pose,))
        for index in range(metadata.frame_count or 0)
    )
    return ExtractionDocument(
        schema_version="1.0",
        source_video=source_name,
        video={
            "width": metadata.width,
            "height": metadata.height,
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "duration_seconds": metadata.duration_seconds,
            "warnings": metadata.warnings,
        },
        timestamp_basis="frame_index / source_fps" if metadata.fps is not None else "unavailable",
        warnings=metadata.warnings,
        frames=frames,
    )


class LandmarkCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "runner.mp4"
        self.video.write_bytes(b"video bytes")
        self.model = self.root / "pose.task"
        self.model.write_bytes(b"model bytes")
        self.metadata = VideoMetadata(640, 480, 30.0, 3, 0.1)
        self.cache = LandmarkExtractionCache(self.root / "cache")
        self.key = landmark_cache_key(self.video, self.model, self.metadata)
        self.landmarks = self.root / "runner_landmarks.json"
        write_json(_document(self.video.name, self.metadata), self.landmarks)
        self.overlay = self.root / "runner_annotated.mp4"
        self.overlay.write_bytes(b"annotated video")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _store(self) -> None:
        self.cache.store(
            self.key,
            source_name=self.video.name,
            metadata=self.metadata,
            landmarks_source=self.landmarks,
            annotated_video_source=self.overlay,
        )

    def _restore(self, destination: Path | None = None) -> bool:
        output = destination or self.root / "restored"
        output.mkdir(exist_ok=True)
        return self.cache.restore(
            self.key,
            source_name=self.video.name,
            metadata=self.metadata,
            landmarks_destination=output / "landmarks.json",
            annotated_video_destination=output / "annotated.mp4",
        )

    def test_cache_hit_restores_matching_landmarks_and_overlay(self) -> None:
        self.assertFalse(self._restore())
        self._store()
        self.assertTrue(self._restore())
        restored = self.root / "restored"
        self.assertEqual((restored / "landmarks.json").read_bytes(), self.landmarks.read_bytes())
        self.assertEqual((restored / "annotated.mp4").read_bytes(), self.overlay.read_bytes())

    def test_changed_video_model_settings_or_metadata_changes_key(self) -> None:
        original = self.key
        self.video.write_bytes(b"changed video")
        self.assertNotEqual(landmark_cache_key(self.video, self.model, self.metadata), original)
        self.video.write_bytes(b"video bytes")
        self.model.write_bytes(b"changed model")
        self.assertNotEqual(landmark_cache_key(self.video, self.model, self.metadata), original)
        self.model.write_bytes(b"model bytes")
        self.assertNotEqual(
            landmark_cache_key(self.video, self.model, self.metadata, extraction_settings={"max_poses": 2}),
            original,
        )
        changed_metadata = VideoMetadata(1280, 480, 30.0, 3, 0.1)
        self.assertNotEqual(landmark_cache_key(self.video, self.model, changed_metadata), original)

    def test_corrupt_artifact_is_a_cache_miss(self) -> None:
        self._store()
        manifest_path = self.cache._manifest_path(self.key)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cached_json = self.cache.directory / manifest["artifacts"]["landmarks_json"]
        cached_json.write_text("not-json", encoding="utf-8")
        self.assertFalse(self._restore())

    def test_incompatible_cache_schema_is_a_cache_miss(self) -> None:
        self._store()
        manifest_path = self.cache._manifest_path(self.key)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["cache_schema_version"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertFalse(self._restore())

    def test_invalid_landmark_document_is_not_stored(self) -> None:
        self.landmarks.write_text('{"schema_version":"old"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid landmark document"):
            self._store()


if __name__ == "__main__":
    unittest.main()
