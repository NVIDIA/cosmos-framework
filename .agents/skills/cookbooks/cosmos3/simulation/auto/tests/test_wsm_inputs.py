"""Check caller-supplied raw-scene-to-inference handoff without proprietary fixtures or GPUs."""

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

AUTO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUTO_ROOT))

from helpers.auto_helpers import CAMERA_ORDER, probe_video
from helpers.daytime_sample import native_request
from helpers.wsm_helpers import caption_window, export_wsm_request, inspect_scene


def write_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as archive:
        for name, array in arrays.items():
            stream = io.BytesIO()
            np.save(stream, array)
            payload = stream.getvalue()
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))


class WsmInferenceInputTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "rds_hq"
        self.captions = self.root / "captions"
        self.cameras = [CAMERA_ORDER[1], CAMERA_ORDER[0]]
        # One f-theta vector per camera; width/height occupy entries 2 and 3.
        write_arrays(
            self.raw / "ftheta_intrinsic/scene.tar",
            {
                f"scene.ftheta_intrinsic.{camera}.npy": np.array([0, 0, 1924, 1084, 1, 0, 0, 0, 0, 0, 0], np.float32)
                for camera in CAMERA_ORDER[:7]
            },
        )
        # Each pose is a [4,4] transform on the RGB/WSM frame timeline.
        write_arrays(
            self.raw / "pose/scene.tar",
            {f"scene.{frame:06d}.pose.{camera}.npy": np.eye(4) for camera in CAMERA_ORDER[:7] for frame in range(297)},
        )
        (self.raw / "all_object_info").mkdir()
        with tarfile.open(self.raw / "all_object_info/scene.tar", "w"):
            pass

    def write_captions(self, cameras: list[str], frames: int) -> None:
        self.captions.mkdir()
        for camera in cameras:
            (self.captions / f"{camera}.json").write_text(
                json.dumps(
                    {
                        "caption_structured": {
                            f"chunk_0_{frames}": {
                                "caption": f"Daytime {camera}.",
                                "start_frame": 0,
                                "end_frame": frames,
                            },
                        }
                    }
                )
            )

    def test_export_is_a_portable_native_t2v_request(self) -> None:
        scene = inspect_scene(self.raw, "scene", self.cameras, 9)
        self.assertEqual(scene["resolutions"], dict.fromkeys(self.cameras, [1280, 720]))
        self.assertEqual(list(scene["resolutions"]), self.cameras)
        output = self.root / "rendered"
        output.mkdir()
        sampling = {"num_frames": 9, "fps": 30, "seed": 0, "guidance": 3, "shift": 10, "num_steps": 35}
        with self.assertRaises(FileNotFoundError):
            export_wsm_request(output, scene, sampling, self.captions)
        self.write_captions(self.cameras, 9)
        originals = {path: path.read_bytes() for path in self.captions.iterdir()}
        with self.assertRaises(FileNotFoundError):
            export_wsm_request(output, scene, sampling, self.captions)
        for camera in self.cameras:
            video = output / "world_scenario" / f"ftheta_{camera}" / "scene_0.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
        path = export_wsm_request(output, scene, sampling, self.captions)
        request = json.loads(path.read_text())
        self.assertEqual([view["camera_key"] for view in request["multiview"]["views"]], self.cameras)
        self.assertTrue(all(not Path(view["control_path"]).is_absolute() for view in request["multiview"]["views"]))
        resolved = native_request(path, mode="text2video")
        self.assertEqual(resolved["num_video_frames_per_chunk"], 9)
        self.assertEqual(resolved["num_conditional_frames"], 0)
        self.assertNotIn("lidar", resolved)
        self.assertTrue(all("vision_path" not in view for view in resolved["multiview"]["views"]))
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)
        with self.assertRaisesRegex(ValueError, "separate directory"):
            export_wsm_request(output, scene, sampling, output / "captions")

    def test_joint_notebook_exports_native_lidar_fields(self) -> None:
        output = self.root / "rendered"
        control = output / "lidar/controls/scene/lidar_control.safetensors"
        control.parent.mkdir(parents=True)
        save_file(
            {"frames": np.zeros((3, 3, 128, 1800), dtype=np.float32)}, str(control),
            metadata={"source": json.dumps({"clip": "scene", "camera_frame_ids": [0, 3, 6]})},
        )
        request_path = output / "request.json"
        original = {"model_mode": "text2video", "num_frames": 9}
        request_path.write_text(json.dumps(original))
        notebook = json.loads((AUTO_ROOT / "generate_wsm.ipynb").read_text())
        cell = next(c for c in notebook["cells"] if c["id"] == "render-lidar-control")
        scope = dict(
            GENERATE_LIDAR_CONTROL=True, FPS=30, NUM_FRAMES=9, OUTPUT_ROOT=output,
            WSM_PYTHON=Path(sys.executable), INPUT_ROOT=self.raw, CLIP_ID="scene",
            CAMERAS=self.cameras, REQUEST_PATH=request_path, subprocess=subprocess, json=json,
        )
        # The renderer is independently covered; exercise the actual export cell
        # against a physical control and keep this contract regression CPU-only.
        with patch("subprocess.run"), contextlib.redirect_stdout(io.StringIO()):
            exec("".join(cell["source"]), scope)
        request = json.loads(scope["joint_request_path"].read_text())
        self.assertEqual(request["lidar"], {"control_path": str(control)})
        self.assertEqual(request["model_mode"], "text2video")
        self.assertEqual(json.loads(request_path.read_text()), original)

    def test_notebook_renders_before_caption_preparation(self) -> None:
        notebook = json.loads((AUTO_ROOT / "generate_wsm.ipynb").read_text())
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
        output = self.root / "rendered"
        env = {
            "COSMOS_COOKBOOK_ROOT": str(AUTO_ROOT),
            "AUTO_RDS_HQ_ROOT": str(self.raw),
            "AUTO_RDS_HQ_CLIP": "scene",
            "WSM_PYTHON": sys.executable,
            "AUTO_WSM_OUTPUT_ROOT": str(output),
        }
        with patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
            os.environ.pop("AUTO_RDS_HQ_CAPTIONS", None)
            scope = {}
            exec(cells["configure"], scope)
            self.assertFalse(scope["CAPTION_ROOT"].exists())
            with patch("subprocess.run") as render:
                exec(cells["render"], scope)
            command = render.call_args.args[0]
            self.assertIn("input_fps_spec.input_pose_fps=30", command)
            self.assertIn("output_frame_spec.target_render_fps=30", command)
            self.assertIn("output_frame_spec.target_chunk_frame=297", command)
            self.assertIn("resolution_spec=resize_1280x720", command)
            self.assertIn("s3_spec=no_upload", command)
            for camera in scope["CAMERAS"]:
                video = output / "world_scenario" / f"ftheta_{camera}" / "scene_0.mp4"
                video.parent.mkdir(parents=True)
                video.write_bytes(b"video")
            scope.update(
                probe_video=lambda path: {"frames": 297, "fps": 30, "width": 1280, "height": 720},
                Video=lambda **kwargs: kwargs["filename"],
                display=lambda video: None,
            )
            exec(cells["check-renders"], scope)
            with self.assertRaises(FileNotFoundError):
                exec(cells["export-request"], scope)
            self.assertFalse((output / "request.json").exists())
            self.captions = scope["CAPTION_ROOT"]
            self.write_captions(scope["CAMERAS"], 297)
            exec(cells["export-request"], scope)
            request = native_request(scope["REQUEST_PATH"], mode="text2video")
            self.assertEqual([view["camera_key"] for view in request["multiview"]["views"]], scope["CAMERAS"])
            self.assertEqual((request["num_frames"], request["fps"]), (297, 30))

    def test_reused_resized_video_and_source_profile_are_distinguished(self) -> None:
        if not all(shutil.which(name) for name in ("ffmpeg", "ffprobe")):
            self.skipTest("Install ffmpeg and ffprobe for the rendered-video handoff check.")
        camera = CAMERA_ORDER[0]
        output = self.root / "rendered"
        video = output / "world_scenario" / f"ftheta_{camera}" / "scene_0.mp4"
        video.parent.mkdir(parents=True)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=1280x720:r=30",
                "-frames:v",
                "297",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-threads",
                "1",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            check=True,
            capture_output=True,
        )
        notebook = json.loads((AUTO_ROOT / "generate_wsm.ipynb").read_text())
        check = next("".join(cell["source"]) for cell in notebook["cells"] if cell["id"] == "check-renders")
        scope = {
            "CAMERAS": [camera],
            "OUTPUT_ROOT": output,
            "CLIP_ID": "scene",
            "NUM_FRAMES": 297,
            "FPS": 30,
            "CAPTION_ROOT": self.captions,
            "probe_video": probe_video,
            "Video": lambda **kwargs: kwargs["filename"],
            "display": lambda video: None,
            "scene": inspect_scene(self.raw, "scene", [camera], 297, resolution_spec="resize_1280x720"),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec(check, scope)
        source = inspect_scene(self.raw, "scene", [camera], 297, resolution_spec="source")
        self.assertEqual(source["resolutions"][camera], [1924, 1084])
        scope["scene"] = source
        with self.assertRaisesRegex(ValueError, "Unexpected rendered video"):
            exec(check, scope)

        # Follow the actual WSM-export -> custom-request path in both native notebooks.
        self.write_captions([camera], 297)
        request_path = export_wsm_request(
            output,
            inspect_scene(self.raw, "scene", [camera], 297),
            {"num_frames": 297, "fps": 30, "seed": 0, "guidance": 3, "num_steps": 35, "shift": 10},
            self.captions,
        )
        for name in ("run_auto_with_cosmos_framework.ipynb", "run_sim-auto_with_cosmos_framework.ipynb"):
            notebook = json.loads((AUTO_ROOT / name).read_text())
            cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
            with (
                self.subTest(notebook=name),
                patch.dict(os.environ, {"COSMOS_COOKBOOK_ROOT": str(AUTO_ROOT)}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                native = {}
                exec(cells["configure"], native)
                native.update(
                    CUSTOM_REQUEST=request_path,
                    REQUEST_PATH=request_path,
                    MODE="text2video",
                    OUTPUT_ROOT=self.root / name,
                    CHECKPOINT_DIR=self.root / "separate-model-checkpoint",
                )
                exec(cells["prepare"], native)
                request = json.loads(native["prepared_request"].read_text())
                self.assertEqual(request["model_mode"], "text2video")
                self.assertNotIn("lidar", request)
                self.assertEqual(request["multiview"]["views"][0]["control_path"], str(video))
                self.assertEqual(request["guidance"], 1 if "sim-auto" in name else 3)
                with patch("subprocess.run") as launch:
                    exec(cells["generate"], native)
                self.assertIn(str(native["prepared_request"]), launch.call_args.args[0])

    def test_missing_calibration_and_poses_fail_without_fallback(self) -> None:
        with self.assertRaisesRegex(ValueError, "calibration for every"):
            inspect_scene(self.raw, "scene", self.cameras + [CAMERA_ORDER[7]], 9)
        with self.assertRaisesRegex(ValueError, "Missing requested pose"):
            inspect_scene(self.raw, "scene", self.cameras, 301)
        for frames, fps in ((8, 30), (9, 10)):
            with self.assertRaises(ValueError):
                inspect_scene(self.raw, "scene", self.cameras, frames, fps)
        with self.assertRaisesRegex(ValueError, "resolution_spec"):
            inspect_scene(self.raw, "scene", self.cameras, 9, resolution_spec="unknown")
        with self.assertRaisesRegex(ValueError, "camera mapping"):
            inspect_scene(self.raw, "scene", ["front_camera"], 9)
        write_arrays(
            self.raw / "pinhole_intrinsic/pinhole_scene.tar",
            {"pinhole_scene.pinhole_intrinsic.camera_front_wide_120fov.npy": np.array([800, 800, 640, 360, 1280, 720])},
        )
        with self.assertRaisesRegex(ValueError, "camera_type=pinhole"):
            inspect_scene(self.raw, "pinhole_scene", [CAMERA_ORDER[0]], 9)

    def test_caption_window_clips_timestamps_without_requesting_later_events(self) -> None:
        document = {
            "caption_structured": {
                "chunk_0_300": {
                    "start_frame": 0,
                    "end_frame": 300,
                    "caption": "A rainy night in a coastal town; the road is wet.\n"
                    "[0.0s-4.5s]: Driving.\n[4.5s-8.0s]: Passing trees.\n[8.0s-10.0s]: Later turn.",
                }
            }
        }
        self.assertEqual(
            caption_window(document, 201, 30),
            "A rainy night in a coastal town; the road is wet.\n[0s-4.5s]: Driving.\n[4.5s-6.7s]: Passing trees.",
        )
        with self.assertRaisesRegex(ValueError, "covering"):
            caption_window(document, 301, 30)
        chunk = document["caption_structured"]["chunk_0_300"]
        for text in (
            "[8s-2s]: Reversed.",
            "[0s-9.9s]: ",
            "[0s-5s]: First.\n[4s-9.9s]: Overlap.",
            "[-1s-3s]: Negative.",
            "[0s-infs]: Infinite.",
            "[nans-3s]: Invalid.",
            "[0s-laters]: Invalid.",
            "Setting.\n[8s-10s]: Outside selected interval.",
        ):
            with self.subTest(caption=text):
                chunk["caption"] = text
                with self.assertRaises(ValueError):
                    caption_window(document, 201, 30)
        chunk["caption"] = "A rainy night in a coastal town; the road is wet."
        self.assertEqual(caption_window(document, 201, 30), chunk["caption"])


if __name__ == "__main__":
    unittest.main()
