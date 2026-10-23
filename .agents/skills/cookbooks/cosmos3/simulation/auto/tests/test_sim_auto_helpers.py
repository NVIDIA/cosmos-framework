"""CPU regressions for the seven-camera Sim-Auto cookbook requests and HTTP client."""

import copy
import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")

import httpx
from PIL import Image as PILImage

AUTO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUTO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers.sim_auto_vllm_omni_helpers import (
    CAMERA_ORDER, MODEL_ID, load_request, prepare_request, require_inputs,
)
from helpers.sim_auto_sglang_helpers import prepare_request as prepare_sglang_request
from test_daytime_sample import make_sample, mock_convert

ASSET_DIR = AUTO_ROOT / "assets" / "daytime_scene"


class SimAutoFixture:
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.sample_root = make_sample(self.root / "sample")
        converter = patch("helpers.daytime_sample.convert_control", side_effect=mock_convert)
        converter.start()
        self.addCleanup(converter.stop)
        requests = self.root / "requests"
        requests.mkdir()
        for name in ("sim_auto.json", "sim_auto_joint.json"):
            shutil.copy(ASSET_DIR / name, requests / name)
        (requests / "lidar_control.safetensors").write_bytes(b"lidar placeholder")
        self.request_path = requests / "sim_auto.json"
        self.joint_path = requests / "sim_auto_joint.json"
        self.manifest = json.loads(self.request_path.read_text())


class SimAutoRequestTests(SimAutoFixture, unittest.TestCase):
    def test_requests_use_the_daytime_sample_cameras(self):
        self.assertEqual(self.manifest["name"], "daytime_scene")
        self.assertEqual((self.manifest["num_frames"], self.manifest["fps"], self.manifest["seed"]), (297, 30, 0))
        views = self.manifest["multiview"]["views"]
        self.assertEqual(tuple(view["camera_key"] for view in views), CAMERA_ORDER)
        for view in views:
            self.assertEqual(Path(view["control_path"]).name, f"{view['camera_key']}.mp4")
            self.assertEqual(Path(view["vision_path"]).name, f"{view['camera_key']}.mp4")
        joint = json.loads(self.joint_path.read_text())
        self.assertEqual(joint.pop("lidar"), {"control_path": "daytime_scene/lidar/control.tar", "return_output": True})
        self.assertEqual(joint, self.manifest)

    def test_load_resolves_against_the_sample_and_control_override(self):
        manifest = load_request(self.request_path, self.sample_root)
        self.assertEqual(manifest["prompt"], f"Daytime caption for {CAMERA_ORDER[0]}.")
        for source, resolved in zip(self.manifest["multiview"]["views"], manifest["multiview"]["views"], strict=True):
            self.assertEqual(Path(resolved["vision_path"]), self.sample_root / source["vision_path"])
            self.assertEqual(Path(resolved["control_path"]), self.sample_root / source["control_path"])
            self.assertEqual(resolved["prompt"], f"Daytime caption for {source['camera_key']}.")
            self.assertNotIn("caption_path", resolved)
        override = self.root / "alternate"
        remapped = load_request(self.request_path, self.sample_root, control_dir=override)
        for view in remapped["multiview"]["views"]:
            self.assertEqual(Path(view["control_path"]), override / f"{view['camera_key']}.mp4")
            self.assertEqual(Path(view["vision_path"]).parent, self.sample_root / "daytime_scene" / "rgb")
        joint = load_request(self.joint_path, self.sample_root, lidar_output=self.root / "lidar_control.safetensors")
        self.assertEqual(Path(joint["lidar"]["control_path"]), self.root / "lidar_control.safetensors")

    def test_upload_indexes_sampling_and_fresh_sessions(self):
        manifest = load_request(self.request_path, self.sample_root)
        original = copy.deepcopy(manifest)
        form, paths = prepare_request(manifest)
        extra = json.loads(form["extra_params"])
        self.assertEqual(manifest, original)
        self.assertEqual(len(paths), 14)
        self.assertEqual(form["model"], MODEL_ID)
        self.assertEqual((form["guidance_scale"], form["num_inference_steps"], form["seed"]), ("1.0", "4", "0"))
        self.assertEqual((form["num_frames"], float(form["fps"]), form["size"]), ("297", 30, "832x480"))
        self.assertEqual(form["enable_frame_interpolation"], "false")
        self.assertNotIn("negative_prompt", form)
        self.assertNotIn("flow_shift", form)
        self.assertNotIn("lidar", extra)
        self.assertEqual(extra["wsm"], {})
        self.assertTrue(extra["reset"] and extra["close_session"])
        self.assertTrue(extra["multiview"]["condition_video_as_image"])
        for index, (source, mapped) in enumerate(zip(manifest["multiview"]["views"], extra["multiview"]["views"], strict=True)):
            self.assertEqual(mapped["prompt"], source["prompt"])
            self.assertEqual(mapped["camera_key"], source["camera_key"])
            for role, offset in (("control", 0), ("vision", 1)):
                self.assertNotIn(f"{role}_path", mapped)
                self.assertEqual(mapped[f"{role}_reference_index"], 2 * index + offset)
                self.assertEqual(paths[2 * index + offset], Path(source[f"{role}_path"]))
        second, _ = prepare_request(manifest)
        self.assertNotEqual(extra["session_id"], json.loads(second["extra_params"])["session_id"])

    def test_joint_request_uploads_the_lidar_control_last(self):
        manifest = load_request(self.joint_path, self.sample_root, lidar_output=self.root / "lidar_control.safetensors")
        form, paths = prepare_request(manifest)
        extra = json.loads(form["extra_params"])
        self.assertEqual(len(paths), 15)
        self.assertEqual(paths[-1], self.root / "lidar_control.safetensors")
        self.assertEqual(extra["lidar"], {"control_reference_index": 14, "return_output": True})
        self.assertEqual([view["vision_reference_index"] for view in extra["multiview"]["views"]], list(range(1, 14, 2)))
        self.assertTrue(extra["reset"] and extra["close_session"] and extra["session_id"].startswith("sim-auto-"))
        self.assertEqual((form["guidance_scale"], form["num_inference_steps"], form["fps"]), ("1.0", "4", "30"))

    def test_invalid_sampling_and_frame_counts_reject_before_submission(self):
        for path, field, value in (
            (self.request_path, "guidance_scale", 6), (self.request_path, "num_inference_steps", 35),
            (self.request_path, "flow_shift", 10), (self.request_path, "negative_prompt", "blurry"),
            (self.request_path, "lidar", {}), (self.request_path, "num_frames", 296),
            (self.request_path, "fps", 0), (self.request_path, "fps", float("nan")),
            (self.request_path, "seed", 2.5), (self.request_path, "resolution", "720"),
            (self.joint_path, "fps", 24),
            (self.joint_path, "lidar", {"control_path": "lidar.safetensors", "return_output": False}),
            (self.joint_path, "lidar", {"control_path": "lidar.safetensors", "return_output": True, "condition_path": "x"}),
            (self.joint_path, "lidar", {"control_path": "lidar.tar", "return_output": True}),
        ):
            with self.subTest(request=path.name, field=field, value=value):
                manifest = load_request(path, self.sample_root, lidar_output=self.root / "lidar_control.safetensors")
                manifest[field] = value
                with self.assertRaises(ValueError):
                    prepare_request(manifest)

    def test_misordered_cameras_and_partial_rgb_conditions_reject(self):
        manifest = load_request(self.request_path, self.sample_root)
        manifest["multiview"]["views"].reverse()
        with self.assertRaisesRegex(ValueError, "canonical"):
            prepare_request(manifest)
        manifest = load_request(self.request_path, self.sample_root)
        manifest["multiview"]["views"][0].pop("vision_path")
        with self.assertRaisesRegex(ValueError, "vision_path"):
            prepare_request(manifest)

    def test_missing_inputs_are_reported_together_and_lfs_pointers_reject(self):
        manifest = load_request(self.joint_path, self.sample_root, lidar_output=self.root / "lidar_control.safetensors")
        for view in manifest["multiview"]["views"]:
            Path(view["control_path"]).unlink()
        Path(manifest["lidar"]["control_path"]).unlink()
        with self.assertRaises(ValueError) as raised:
            require_inputs(manifest)
        self.assertIn("CONTROL_DIR", str(raised.exception))
        self.assertIn("LiDAR control", str(raised.exception))
        for camera in CAMERA_ORDER:
            self.assertIn(camera, str(raised.exception))
        for view in manifest["multiview"]["views"]:
            Path(view["control_path"]).write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        Path(manifest["lidar"]["control_path"]).write_bytes(b"lidar placeholder")
        with self.assertRaisesRegex(ValueError, "Unhydrated Git LFS pointer"):
            prepare_request(manifest)

    def test_rgb_cannot_be_used_as_its_own_wsm_control(self):
        manifest = load_request(self.request_path, self.sample_root)
        manifest["multiview"]["views"][0]["control_path"] = manifest["multiview"]["views"][0]["vision_path"]
        with self.assertRaisesRegex(ValueError, "separate inputs"):
            prepare_request(manifest)


class SimAutoNotebookTests(SimAutoFixture, unittest.TestCase):
    # Cells that decode real safetensors LiDAR; the fixture's LiDAR files are placeholders.
    SKIPPED = {"sim-auto-joint-preview", "sim-auto-joint-visualize", "sim-auto-joint-video"}

    def test_both_examples_run_in_order_and_failed_runs_cannot_reuse_prior_videos(self):
        notebook = json.loads((AUTO_ROOT / "run_sim_auto_with_vllm_omni.ipynb").read_text())
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
        replacements = {
            'REQUEST_PATH = AUTO_ROOT / "assets" / "daytime_scene" / "sim_auto.json"':
                f"REQUEST_PATH = Path({str(self.request_path)!r})",
            'JOINT_REQUEST_PATH = AUTO_ROOT / "assets" / "daytime_scene" / "sim_auto_joint.json"':
                f"JOINT_REQUEST_PATH = Path({str(self.joint_path)!r})",
        }
        for old, new in replacements.items():
            self.assertEqual(sum(old in cell for cell in cells.values()), 1, old)
            cells = {key: cell.replace(old, new) for key, cell in cells.items()}
        self.assertTrue(self.SKIPPED <= cells.keys())
        import IPython.display

        original_client = httpx.Client
        fail = False
        requests = []

        def handler(request):
            path = request.url.path
            requests.append((request.method, path))
            if path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": MODEL_ID}]})
            if path == "/v1/videos/sync":
                self.assertEqual(request.read().count(b'name="input_references"'), 14)
                return (
                    httpx.Response(400, text="mock generation failed") if fail else
                    httpx.Response(200, content=b"mock video", headers={"content-type": "video/mp4"})
                )
            if path == "/v1/videos" and request.method == "POST":
                body = request.read()
                self.assertEqual(body.count(b'name="input_references"'), 15)
                self.assertIn(b'filename="lidar_control.safetensors"', body)
                return httpx.Response(200, json={"id": "video-1", "status": "completed", "lidar": {"fps": 10.0}})
            if path == "/v1/videos/video-1/content":
                return httpx.Response(200, content=b"mock joint video")
            if path == "/v1/videos/video-1/lidar":
                return httpx.Response(200, content=b"mock lidar")
            raise AssertionError(f"Unexpected request {request.method} {path}")

        def client_factory(*args, **kwargs):
            return original_client(*args, transport=httpx.MockTransport(handler), **kwargs)

        def split(video, output, cameras, frames, fps):
            self.assertEqual((frames, fps), (297, 30))
            self.assertEqual(tuple(cameras), CAMERA_ORDER)
            paths = {camera: output / f"{camera}.mp4" for camera in cameras}
            for path in paths.values():
                path.write_bytes(b"mock camera")
            return paths

        def stub_media(scope):
            # Replace the FFmpeg-backed notebook helpers; the fixture videos are placeholders.
            scope.update(
                validate_video=lambda path, **expected: {
                    "frames": expected["num_frames"], "fps": float(expected["fps"]),
                    "width": expected["width"], "height": expected["height"],
                },
                read_frame=lambda path, index: PILImage.new("RGB", (8, 8)),
                video_row=lambda videos, **kwargs: "",
            )

        from helpers import auto_helpers
        environment = {
            "COSMOS_COOKBOOK_ROOT": str(AUTO_ROOT.parents[3]),
            "VLLM_BASE_URL": "http://test", "VLLM_API_KEY": "",
            "COSMOS3_OUTPUT_DIR": str(self.root / "runs"),
            "AUTO_SAMPLE_ROOT": str(self.sample_root),
        }
        controls = [self.sample_root / view["control_path"] for view in self.manifest["multiview"]["views"]]
        lidar_info = {"shape": [3, 100, 128, 1800], "dtype": "float32", "metadata": {}}
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.dict("os.environ", environment),
            patch.object(IPython.display, "display"),
            patch.object(IPython.display, "HTML"),
            patch.object(httpx, "Client", side_effect=client_factory),
            patch.object(auto_helpers, "validate_rgb_conditions"),
            patch.object(auto_helpers, "prepare_controls", return_value=controls),
            patch.object(auto_helpers, "split_cameras", side_effect=split),
            patch.object(auto_helpers, "validate_lidar", return_value=lidar_info) as validate_lidar,
            patch.object(auto_helpers, "validate_lidar_output", return_value=lidar_info) as validate_output,
        ):
            scope = {}
            for cell_id, cell in cells.items():
                if cell_id in self.SKIPPED:
                    continue
                exec(cell, scope)
                if "def validate_video" in cell:
                    stub_media(scope)
            self.assertEqual(scope["SAMPLE_ROOT"], self.sample_root)

            sample_dir = scope["sample_dir"]
            record, = json.loads((sample_dir / "sample_outputs.json").read_text())
            self.assertEqual(record["status"], "success")
            self.assertEqual(len(record["cameras"]), 7)

            joint_dir = scope["JOINT_OUTPUT_DIR"]
            self.assertNotEqual(joint_dir, sample_dir)
            joint, = json.loads((joint_dir / "sample_outputs.json").read_text())
            self.assertEqual((joint["status"], joint["job_id"], len(joint["cameras"])), ("success", "video-1", 7))
            self.assertEqual((joint_dir / "multiview_camera_major.lidar.safetensors").read_bytes(), b"mock lidar")
            self.assertEqual(json.loads((joint_dir / "multiview_camera_major.lidar.json").read_text()), {"fps": 10.0})
            validate_lidar.assert_called_once()
            self.assertEqual(validate_lidar.call_args.args[1], 99)
            self.assertEqual(validate_output.call_args.args[2], 99)
            joint_extra = json.loads(json.loads((joint_dir / "request.json").read_text())["extra_params"])
            self.assertEqual(joint_extra["lidar"], {"control_reference_index": 14, "return_output": True})
            self.assertIn(("POST", "/v1/videos"), requests)

            first_session = json.loads(json.loads((sample_dir / "request.json").read_text())["extra_params"])["session_id"]
            exec(cells["sim-auto-16"], scope)
            exec(cells["sim-auto-18"], scope)
            second_session = json.loads(json.loads((sample_dir / "request.json").read_text())["extra_params"])["session_id"]
            self.assertNotEqual(first_session, second_session)

            fail = True
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                exec(cells["sim-auto-16"], scope)
            self.assertIsNone(scope["video"])
            self.assertEqual((sample_dir / "response_error.txt").read_text(), "mock generation failed")
            with self.assertRaisesRegex(RuntimeError, "Complete generation"):
                exec(cells["sim-auto-18"], scope)


class SimAutoSglangRequestTests(SimAutoFixture, unittest.TestCase):
    def test_server_visible_paths_without_uploads_or_sessions(self):
        manifest = load_request(self.request_path, self.sample_root)
        original = copy.deepcopy(manifest)
        form, paths = prepare_sglang_request(manifest)
        extra = json.loads(form["extra_params"])
        self.assertEqual(manifest, original)
        self.assertEqual(paths, [])
        self.assertTrue(all(isinstance(value, str) for value in form.values()))
        self.assertEqual(form["model"], MODEL_ID)
        self.assertEqual(form["prompt"], f"Daytime caption for {CAMERA_ORDER[0]}.")
        self.assertEqual((form["guidance_scale"], form["num_inference_steps"], form["seed"]), ("1.0", "4", "0"))
        self.assertEqual((form["num_frames"], form["fps"], form["size"]), ("297", "30", "832x480"))
        for field in ("negative_prompt", "flow_shift", "enable_frame_interpolation"):
            self.assertNotIn(field, form)
        self.assertEqual(set(extra), {"wsm", "multiview", "resolution", "aspect_ratio"})
        self.assertEqual((extra["wsm"], extra["resolution"], extra["aspect_ratio"]), ({}, "480", "16,9"))
        self.assertEqual(set(extra["multiview"]), {"views", "condition_video_as_image", "num_frames"})
        self.assertTrue(extra["multiview"]["condition_video_as_image"])
        self.assertEqual(extra["multiview"]["num_frames"], 297)
        for source, mapped in zip(self.manifest["multiview"]["views"], extra["multiview"]["views"], strict=True):
            self.assertEqual(set(mapped), {"camera_key", "prompt", "control_path", "vision_path"})
            self.assertEqual(mapped["camera_key"], source["camera_key"])
            self.assertEqual(mapped["prompt"], f"Daytime caption for {source['camera_key']}.")
            for role in ("control", "vision"):
                self.assertEqual(Path(mapped[f"{role}_path"]), self.sample_root / source[f"{role}_path"])
                self.assertTrue(Path(mapped[f"{role}_path"]).is_absolute())
        self.assertEqual(prepare_sglang_request(manifest)[0], form)  # No per-request session identifiers.

    def test_joint_request_names_the_converted_lidar_control(self):
        lidar = self.root / "lidar_control.safetensors"
        manifest = load_request(self.joint_path, self.sample_root, lidar_output=lidar)
        form, paths = prepare_sglang_request(manifest)
        extra = json.loads(form["extra_params"])
        self.assertEqual(paths, [])
        self.assertEqual(extra["lidar"], {"control_path": str(lidar), "decode": True})
        self.assertEqual(form["fps"], "30")
        lidar.unlink()
        with self.assertRaisesRegex(ValueError, "LiDAR control"):
            prepare_sglang_request(manifest)

    def test_contract_and_missing_inputs_reject_before_submission(self):
        manifest = load_request(self.request_path, self.sample_root)
        for field, value in (("guidance_scale", 3.0), ("num_inference_steps", 8), ("num_frames", 296)):
            with self.assertRaises(ValueError):
                prepare_sglang_request({**manifest, field: value})
        reordered = load_request(self.request_path, self.sample_root)
        reordered["multiview"]["views"].reverse()
        with self.assertRaisesRegex(ValueError, "canonical"):
            prepare_sglang_request(reordered)
        Path(manifest["multiview"]["views"][0]["control_path"]).write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaisesRegex(ValueError, "Git LFS pointer"):
            prepare_sglang_request(manifest)


class SimAutoSglangNotebookTests(SimAutoFixture, unittest.TestCase):
    def test_execution_order_jobs_and_failed_run_cannot_reuse_prior_video(self):
        notebook = json.loads((AUTO_ROOT / "run_sim-auto_with_sglang.ipynb").read_text())
        cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        bundled = 'REQUEST_PATH = AUTO_ROOT / "assets" / "daytime_scene" / "sim_auto.json"'
        self.assertEqual(sum(bundled in cell for cell in cells), 1)
        cells = [cell.replace(bundled, f"REQUEST_PATH = Path({str(self.request_path)!r})") for cell in cells]
        generate = next(cell for cell in cells if "job = submit_video_job(" in cell)
        validate = next(cell for cell in cells if "Complete generation" in cell)
        import IPython.display

        original_client = httpx.Client
        state = {"fail": None, "jobs": 0, "polls": 0}

        def handler(request):
            path = request.url.path
            if path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": MODEL_ID}]})
            if path == "/v1/videos" and request.method == "POST":
                body = request.read()
                self.assertIn(b'name="extra_params"', body)
                self.assertIn(b'name="num_inference_steps"', body)
                self.assertNotIn(b'name="input_references"', body)
                self.assertNotIn(b"_reference_index", body)
                self.assertNotIn(b"session_id", body)
                if state["fail"] == "submit":
                    return httpx.Response(400, text="mock submission rejected")
                state["jobs"] += 1
                return httpx.Response(200, json={"id": f"job-{state['jobs']}", "status": "queued", "progress": 0})
            job_id = f"job-{state['jobs']}"
            if path == f"/v1/videos/{job_id}" and request.method == "GET":
                state["polls"] += 1
                if state["fail"] == "job":
                    return httpx.Response(200, json={"id": job_id, "status": "failed", "error": {"message": "mock rollout failed"}})
                return httpx.Response(200, json={"id": job_id, "status": "completed", "progress": 100})
            if path == f"/v1/videos/{job_id}/content" and request.method == "GET":
                return httpx.Response(200, content=b"mock video", headers={"content-type": "video/mp4"})
            self.fail(f"Unexpected request {request.method} {path}")

        def client_factory(*args, **kwargs):
            return original_client(*args, transport=httpx.MockTransport(handler), **kwargs)

        def split(video, output, cameras, frames, fps):
            self.assertEqual((frames, fps), (297, 30))
            self.assertEqual(tuple(cameras), CAMERA_ORDER)
            paths = {camera: output / f"{camera}.mp4" for camera in cameras}
            for path in paths.values():
                path.write_bytes(b"mock camera")
            return paths

        def stub_media(scope):
            # Replace the FFmpeg-backed notebook helpers; the fixture videos are placeholders.
            scope.update(
                validate_video=lambda path, **expected: {
                    "frames": expected["num_frames"], "fps": float(expected["fps"]),
                    "width": expected["width"], "height": expected["height"],
                },
                read_frame=lambda path, index: PILImage.new("RGB", (8, 8)),
                video_row=lambda videos, **kwargs: "",
            )

        from helpers import auto_helpers
        environment = {
            "COSMOS_COOKBOOK_ROOT": str(AUTO_ROOT.parents[3]),
            "SGLANG_BASE_URL": "http://test", "SGLANG_API_KEY": "",
            "COSMOS3_OUTPUT_DIR": str(self.root / "runs"),
            "AUTO_SAMPLE_ROOT": str(self.sample_root),
        }
        controls = [self.sample_root / view["control_path"] for view in self.manifest["multiview"]["views"]]
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.dict("os.environ", environment),
            patch("time.sleep"),
            patch.object(IPython.display, "display"),
            patch.object(IPython.display, "HTML"),
            patch.object(httpx, "Client", side_effect=client_factory),
            patch.object(auto_helpers, "validate_rgb_conditions"),
            patch.object(auto_helpers, "prepare_controls", return_value=controls) as prepare_controls,
            patch.object(auto_helpers, "split_cameras", side_effect=split),
        ):
            scope = {}
            for cell in cells:
                exec(cell, scope)
                if "def validate_video" in cell:
                    stub_media(scope)
            self.assertEqual(scope["SAMPLE_ROOT"], self.sample_root)
            # SGLang opens the controls by path, so exact-length controls must be copied under the mount.
            self.assertTrue(prepare_controls.call_args.kwargs.get("materialize"))
            sample_dir = scope["sample_dir"]
            staged = sample_dir / "staged_conditions"
            self.assertEqual({path.name for path in staged.iterdir()}, {f"{camera}.mp4" for camera in CAMERA_ORDER})
            record, = json.loads((sample_dir / "sample_outputs.json").read_text())
            self.assertEqual((record["status"], record["job_id"], len(record["cameras"])), ("success", "job-1", 7))
            self.assertEqual(len(record["inputs"]), 14)
            self.assertEqual(state["polls"], 1)  # Queued at submission, completed at the first poll.
            self.assertEqual(json.loads((sample_dir / "job.json").read_text())["status"], "completed")
            request = json.loads((sample_dir / "request.json").read_text())
            extra = json.loads(request["extra_params"])
            self.assertNotIn("session_id", extra)
            for view, control in zip(extra["multiview"]["views"], controls, strict=True):
                self.assertEqual(Path(view["control_path"]), control)
                self.assertEqual(Path(view["vision_path"]), staged / f"{view['camera_key']}.mp4")
                self.assertEqual(view["prompt"], f"Daytime caption for {view['camera_key']}.")

            exec(generate, scope)
            exec(validate, scope)
            record, = json.loads((sample_dir / "sample_outputs.json").read_text())
            self.assertEqual(record["job_id"], "job-2")

            state["fail"] = "submit"
            with self.assertRaisesRegex(RuntimeError, "HTTP 400"):
                exec(generate, scope)
            self.assertIsNone(scope["video"])
            self.assertEqual((sample_dir / "response_error.txt").read_text(), "mock submission rejected")
            with self.assertRaisesRegex(RuntimeError, "Complete generation"):
                exec(validate, scope)

            state["fail"] = "job"
            with self.assertRaisesRegex(RuntimeError, "Video generation failed"):
                exec(generate, scope)
            self.assertIsNone(scope["video"])
            self.assertEqual(json.loads((sample_dir / "job.json").read_text())["status"], "failed")
            self.assertIn("mock rollout failed", (sample_dir / "response_error.txt").read_text())
            with self.assertRaisesRegex(RuntimeError, "Complete generation"):
                exec(validate, scope)


if __name__ == "__main__":
    unittest.main()
