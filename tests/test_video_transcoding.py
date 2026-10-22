from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from src.video_transcoding import VideoTranscodeError, transcode_overlay_to_h264


class FakeFrame:
    shape = (2, 2, 3)


class FakeCapture:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.verification = "h264_tmp_" in self.path.name
        self.frames = [FakeFrame()] if self.verification else [FakeFrame(), FakeFrame()]
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return {
            1: 30.0,
            2: 2.0,
            3: 2.0,
            4: 1.0 if self.verification else 2.0,
            5: int.from_bytes(b"h264", "little") if self.verification else int.from_bytes(b"FMP4", "little"),
        }[property_id]

    def read(self) -> tuple[bool, FakeFrame | None]:
        if self.read_count >= len(self.frames):
            return False, None
        self.read_count += 1
        return True, self.frames[self.read_count - 1]

    def release(self) -> None:
        self.released = True


class FakeWriter:
    def __init__(self, path: str, opened: bool = True) -> None:
        self.path = Path(path)
        self.opened = opened
        self.path.write_bytes(b"")
        self.frame_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def write(self, _frame: FakeFrame) -> None:
        self.path.write_bytes(self.path.read_bytes() + b"frame")
        self.frame_count += 1

    def release(self) -> None:
        self.released = True


class FakeCv2:
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FRAME_COUNT = 4
    CAP_PROP_FOURCC = 5

    def __init__(self, writer_opened: bool = True) -> None:
        self.writer_opened = writer_opened
        self.writer_fourcc = None
        self.writers: list[FakeWriter] = []
        self.captures: list[FakeCapture] = []

    def VideoCapture(self, path: str) -> FakeCapture:
        capture = FakeCapture(path)
        self.captures.append(capture)
        return capture

    def VideoWriter_fourcc(self, *chars: str) -> int:
        self.writer_fourcc = chars
        return 0

    def VideoWriter(self, path: str, _fourcc: int, _fps: float, _size: tuple[int, int]) -> FakeWriter:
        writer = FakeWriter(path, opened=self.writer_opened)
        self.writers.append(writer)
        return writer


class VideoTranscodingTests(unittest.TestCase):
    def test_transcodes_existing_mp4_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "overlay.mp4"
            source.write_bytes(b"original mp4v bytes")
            cv2 = FakeCv2()

            with mock.patch.dict(sys.modules, {"cv2": cv2}):
                result = transcode_overlay_to_h264(source)

            self.assertEqual(result, source.with_name("overlay_h264.mp4"))
            self.assertEqual(source.read_bytes(), b"original mp4v bytes")
            self.assertTrue(result.is_file())
            self.assertEqual(cv2.writer_fourcc, tuple("avc1"))
            self.assertTrue(all(capture.released for capture in cv2.captures))
            self.assertTrue(all(writer.released for writer in cv2.writers))

    def test_encoder_failure_preserves_original_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "overlay.mp4"
            source.write_bytes(b"original mp4v bytes")
            cv2 = FakeCv2(writer_opened=False)

            with mock.patch.dict(sys.modules, {"cv2": cv2}):
                with self.assertRaisesRegex(VideoTranscodeError, "could not initialize"):
                    transcode_overlay_to_h264(source)

            self.assertEqual(source.read_bytes(), b"original mp4v bytes")
            self.assertFalse(source.with_name("overlay_h264.mp4").exists())
            self.assertEqual(list(Path(temporary).glob("h264_tmp_*.mp4")), [])

    def test_refuses_to_overwrite_existing_h264_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "overlay.mp4"
            destination = Path(temporary) / "overlay_h264.mp4"
            source.write_bytes(b"source")
            destination.write_bytes(b"existing")
            cv2 = FakeCv2()

            with mock.patch.dict(sys.modules, {"cv2": cv2}):
                with self.assertRaisesRegex(VideoTranscodeError, "Refusing to overwrite"):
                    transcode_overlay_to_h264(source)

            self.assertEqual(destination.read_bytes(), b"existing")
            self.assertEqual(cv2.captures, [])


if __name__ == "__main__":
    unittest.main()
