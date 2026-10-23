"""Content-addressed cache for reusable pose-extraction artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
import uuid

from .landmarks import LANDMARK_NAMES
from .video_io import VideoMetadata


LOGGER = logging.getLogger(__name__)
CACHE_SCHEMA_VERSION = 2
LANDMARK_SCHEMA_VERSION = "1.0"
EXTRACTION_IMPLEMENTATION_VERSION = 1


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version(package)
    except PackageNotFoundError:
        return None


def _json_value(value: Any) -> Any:
    """Normalize dataclass tuples and reject non-JSON metadata values."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def landmark_cache_key(
    video_path: str | Path,
    model_path: str | Path,
    metadata: VideoMetadata,
    *,
    extraction_settings: Mapping[str, Any] | None = None,
) -> str:
    """Return a key tied to both input contents and pose-extraction context."""
    fps = metadata.fps
    identity = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "landmark_schema_version": LANDMARK_SCHEMA_VERSION,
        "implementation_version": EXTRACTION_IMPLEMENTATION_VERSION,
        "source_video_name": Path(video_path).name,
        "source_video_sha256": _sha256_file(video_path),
        "model_sha256": _sha256_file(model_path),
        "video_metadata": _json_value(asdict(metadata)),
        "runtime": {
            "mediapipe": _package_version("mediapipe"),
            "opencv_contrib_python": _package_version("opencv-contrib-python"),
        },
        "pose_settings": {
            "max_poses": 1,
            "running_mode": "VIDEO" if fps is not None and math.isfinite(fps) and fps > 0 else "IMAGE",
            "timestamp_basis": "frame_index / source_fps" if fps is not None and fps > 0 else "unavailable",
            "additional": _json_value(dict(extraction_settings or {})),
        },
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class LandmarkExtractionCache:
    """Store verified landmark JSON and its corresponding annotated video.

    Cache entries contain only pose-extraction artifacts. Gait metrics and
    event evaluation are recomputed for every analysis with the current settings.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser().resolve()

    def _manifest_path(self, key: str) -> Path:
        return self.directory / f"{key}.manifest.json"

    @staticmethod
    def _validate_document(
        document: object,
        *,
        source_name: str,
        metadata: VideoMetadata,
    ) -> bool:
        if not isinstance(document, dict) or document.get("schema_version") != LANDMARK_SCHEMA_VERSION:
            return False
        if document.get("source_video") != source_name:
            return False
        expected_metadata = _json_value(asdict(metadata))
        if document.get("video") != expected_metadata:
            return False
        expected_timestamp_basis = "frame_index / source_fps" if metadata.fps is not None else "unavailable"
        if document.get("timestamp_basis") != expected_timestamp_basis:
            return False
        frames = document.get("frames")
        if not isinstance(frames, list) or not frames:
            return False
        if metadata.frame_count is not None and len(frames) != metadata.frame_count:
            return False

        for index, frame in enumerate(frames):
            frame_index = frame.get("frame_index") if isinstance(frame, dict) else None
            if isinstance(frame_index, bool) or frame_index != index:
                return False
            poses = frame.get("poses")
            if not isinstance(poses, list):
                return False
            timestamp = frame.get("timestamp_seconds")
            if metadata.fps is None:
                if timestamp is not None:
                    return False
            elif (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp)
                or not math.isclose(timestamp, index / metadata.fps, rel_tol=0, abs_tol=1e-12)
            ):
                return False
            for pose in poses:
                if not isinstance(pose, dict) or not isinstance(pose.get("landmarks"), dict):
                    return False
                landmarks = pose["landmarks"]
                if set(landmarks) != set(LANDMARK_NAMES):
                    return False
                for landmark in landmarks.values():
                    if not isinstance(landmark, dict):
                        return False
                    for coordinate in ("x", "y", "z"):
                        value = landmark.get(coordinate)
                        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                            return False
                    for confidence in ("visibility", "presence"):
                        value = landmark.get(confidence)
                        if value is not None and (
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(value)
                        ):
                            return False
        return True

    def restore(
        self,
        key: str,
        *,
        source_name: str,
        metadata: VideoMetadata,
        landmarks_destination: str | Path,
        annotated_video_destination: str | Path,
    ) -> bool:
        """Copy a complete compatible cache entry into a run's temporary output."""
        manifest_path = self._manifest_path(key)
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                not isinstance(manifest, dict)
                or manifest.get("cache_schema_version") != CACHE_SCHEMA_VERSION
                or manifest.get("cache_key") != key
                or manifest.get("source_video") != source_name
                or manifest.get("video_metadata") != _json_value(asdict(metadata))
            ):
                raise ValueError("cache manifest is incompatible")
            artifact_names = manifest.get("artifacts")
            hashes = manifest.get("sha256")
            if not isinstance(artifact_names, dict) or not isinstance(hashes, dict):
                raise ValueError("cache manifest has no artifact references")
            landmarks_name = artifact_names.get("landmarks_json")
            video_name = artifact_names.get("annotated_video")
            if (
                not isinstance(landmarks_name, str)
                or Path(landmarks_name).name != landmarks_name
                or not landmarks_name.startswith(f"{key}.")
                or not landmarks_name.endswith(".landmarks.json")
                or not isinstance(video_name, str)
                or Path(video_name).name != video_name
                or not video_name.startswith(f"{key}.")
                or not video_name.endswith(".annotated.mp4")
            ):
                raise ValueError("cache manifest contains invalid artifact paths")
            landmarks_path = self.directory / landmarks_name
            video_path = self.directory / video_name
            if _sha256_file(landmarks_path) != hashes.get("landmarks_json"):
                raise ValueError("cached landmark checksum mismatch")
            if _sha256_file(video_path) != hashes.get("annotated_video"):
                raise ValueError("cached video checksum mismatch")
            if video_path.stat().st_size == 0:
                raise ValueError("cached annotated video is empty")
            document = json.loads(landmarks_path.read_text(encoding="utf-8"))
            if not self._validate_document(document, source_name=source_name, metadata=metadata):
                raise ValueError("cached landmark document failed schema validation")
            shutil.copyfile(landmarks_path, landmarks_destination)
            shutil.copyfile(video_path, annotated_video_destination)
            return True
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring invalid or unavailable landmark cache entry %s: %s", key, exc)
            return False

    @staticmethod
    def _atomic_copy(source: Path, destination: Path) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def store(
        self,
        key: str,
        *,
        source_name: str,
        metadata: VideoMetadata,
        landmarks_source: str | Path,
        annotated_video_source: str | Path,
    ) -> None:
        """Atomically publish extracted files, with the manifest written last."""
        source_json = Path(landmarks_source)
        source_video = Path(annotated_video_source)
        document = json.loads(source_json.read_text(encoding="utf-8"))
        if not self._validate_document(document, source_name=source_name, metadata=metadata):
            raise ValueError("refusing to cache an invalid landmark document")
        if not source_video.is_file() or source_video.stat().st_size == 0:
            raise ValueError("refusing to cache a missing or empty annotated video")

        self.directory.mkdir(parents=True, exist_ok=True)
        generation = uuid.uuid4().hex
        cached_json = self.directory / f"{key}.{generation}.landmarks.json"
        cached_video = self.directory / f"{key}.{generation}.annotated.mp4"
        manifest_path = self._manifest_path(key)
        self._atomic_copy(source_json, cached_json)
        self._atomic_copy(source_video, cached_video)
        manifest = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "cache_key": key,
            "source_video": source_name,
            "video_metadata": _json_value(asdict(metadata)),
            "artifacts": {
                "landmarks_json": cached_json.name,
                "annotated_video": cached_video.name,
            },
            "sha256": {
                "landmarks_json": _sha256_file(cached_json),
                "annotated_video": _sha256_file(cached_video),
            },
        }
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{manifest_path.name}.", dir=self.directory)
        temporary_manifest = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")
            os.replace(temporary_manifest, manifest_path)
        finally:
            temporary_manifest.unlink(missing_ok=True)
