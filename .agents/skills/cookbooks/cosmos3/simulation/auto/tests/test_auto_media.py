"""Exercise Auto media preparation with real FFmpeg encoding and decoding."""

from fractions import Fraction
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

AUTO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUTO_ROOT))

from helpers.auto_helpers import prepare_controls, probe_video, split_cameras


class AutoMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(shutil.which(name) for name in ("ffmpeg", "ffprobe")):
            raise unittest.SkipTest("Install ffmpeg and ffprobe for media integration tests.")

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_video(self, name, levels, fps, width=64, height=48):
        """Encode a changing gray level per frame to detect dropped/repeated frames."""
        path = self.root / name
        pixels = b"".join(bytes([level]) * (width * height) for level in levels)
        subprocess.run(
            [
                "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", "gray",
                "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-crf", "0", "-threads", "1",
                "-pix_fmt", "yuv420p", "-r", str(fps), str(path),
            ],
            input=pixels, capture_output=True, check=True,
        )
        return path

    def frame_levels(self, path, width=64, height=48):
        decoded = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                "-fps_mode", "passthrough", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
            ],
            capture_output=True, check=True,
        ).stdout
        frame_size = width * height
        self.assertEqual(len(decoded) % frame_size, 0)
        return decoded[::frame_size]

    def assert_video(self, path, reference, fps, width=64, height=48):
        info = probe_video(path)
        self.assertEqual(info["frames"], len(reference))
        self.assertEqual(info["fps"], Fraction(fps))
        self.assertEqual((info["width"], info["height"]), (width, height))
        self.assertAlmostEqual(float(info["duration"]), len(reference) / fps, places=5)
        actual = self.frame_levels(path, width, height)
        self.assertEqual(len(actual), len(reference))
        # The helpers use lossy H.264; allow quantization but preserve each frame's identity.
        self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, reference, strict=True)), 2)

    def test_trim_preserves_frame_rate_count_geometry_and_sequence(self):
        for fps in (30, 24):
            with self.subTest(fps=fps):
                views, references = [], []
                for index, width in enumerate((64, 80)):
                    source = self.make_video(
                        f"control-{fps}-{index}.mp4", [20 + i % 211 for i in range(241)], fps, width,
                    )
                    views.append({"camera_key": f"camera_{index}", "control_path": source.name})
                    references.append(self.frame_levels(source, width)[:201])
                prepared = prepare_controls(views, self.root, self.root / f"prepared-{fps}", 201, fps)
                self.assertEqual(len(prepared), 2)
                for output, reference, width in zip(prepared, references, (64, 80), strict=True):
                    self.assert_video(output, reference, fps, width)

    def test_split_preserves_camera_boundaries_and_each_frames_timeline(self):
        for fps in (30, 24):
            with self.subTest(fps=fps):
                frames = 201
                source = self.make_video(
                    f"camera-major-{fps}.mp4",
                    [20 + i for i in range(frames)] + [235 - i for i in range(frames)], fps,
                )
                reference = self.frame_levels(source)
                camera_order = ("camera_front", "camera_rear")
                outputs = split_cameras(
                    source, self.root, camera_order, frames, fps, width=64, height=48,
                )
                self.assertEqual(tuple(outputs), camera_order)
                for index, camera in enumerate(camera_order):
                    self.assert_video(outputs[camera], reference[index * frames:(index + 1) * frames], fps)


if __name__ == "__main__":
    unittest.main()
