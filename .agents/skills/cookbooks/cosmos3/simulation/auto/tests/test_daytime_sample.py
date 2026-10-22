"""CPU checks for the shared daytime sample requests and the LiDAR range-map conversion."""

import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

AUTO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUTO_ROOT))

from helpers.auto_helpers import (
    CAMERA_ORDER, prepare_request, stage_conditions, submit_joint, validate_sglang_lidar,
)
from helpers.daytime_sample import SAMPLE_CAMERAS, read_caption, resolve_request, sample_files, verify_sample
from helpers.lidar_rangemap import convert_control, pool_nearest_return

ASSET_DIR = AUTO_ROOT / "assets" / "daytime_scene"
REQUESTS = ("transfer_auto_wsm.json", "transfer_auto_joint.json", "sim_auto.json", "sim_auto_joint.json")


def mock_convert(archive: Path, output: Path) -> dict:
    """Keep HTTP/request tests independent of large tensor allocation."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"mock numeric control")
    return {}


def make_sample(root: Path) -> Path:
    """Write placeholder media, real-format captions, and a matching SHA256SUMS."""
    lines = []
    for relative in sample_files():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(json.dumps({"caption_structured": {"chunk_0_300": {
                "caption": f"Daytime caption for {path.stem}.", "start_frame": 0, "end_frame": 300,
            }}}))
        elif path.suffix == ".jsonl":
            views = [{"camera_key": key, "control_path": f"daytime_scene/wsm/{key}.mp4",
                      "vision_path": f"daytime_scene/rgb/{key}.mp4",
                      "caption_path": f"daytime_scene/captions/{key}.json"} for key in reversed(SAMPLE_CAMERAS)]
            path.write_text(json.dumps({
                "name": "daytime_scene", "model_mode": "image2video", "wsm": {}, "prompt": "",
                "num_frames": 297, "num_video_frames_per_chunk": 297, "fps": 30, "seed": 0,
                "resolution": "720", "aspect_ratio": "16,9", "num_conditional_frames": 1,
                "multiview": {"views": views, "load_caption_from_data": True,
                              "condition_video_as_image": True, "use_first_chunk_only": True},
                "lidar": {"control_path": "daytime_scene/lidar/control.tar"},
            }) + "\n")
        elif path.suffix == ".tar":
            make_archive(path, {i * 3: sparse([0], [0], [1.0]) for i in range(100)}, clip="daytime_scene")
        else:
            path.write_bytes(f"placeholder {relative}".encode())
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{relative}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return root


def npz(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, array)
    return stream.getvalue()


def make_archive(path: Path, sweeps: dict[int, dict[str, np.ndarray]], clip: str = "scene") -> Path:
    with tarfile.open(path, "w") as archive:
        for frame, members in sweeps.items():
            for kind, array in members.items():
                data = npz(array)
                info = tarfile.TarInfo(f"{clip}.{frame:06d}.lidar_{kind}.npz")
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    return path


def sparse(rows, cols, ranges):
    return {"row": np.array(rows, np.uint8), "col": np.array(cols, np.uint16), "range": np.array(ranges, np.float16)}


class SampleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = make_sample(Path(temporary.name).resolve())
        converter = patch("helpers.daytime_sample.convert_control", side_effect=mock_convert)
        converter.start()
        self.addCleanup(converter.stop)

    def test_verification_reports_every_missing_modified_or_unhydrated_file(self):
        self.assertEqual(verify_sample(self.root), self.root)
        wsm = self.root / f"daytime_scene/wsm/{SAMPLE_CAMERAS[0]}.mp4"
        rgb = self.root / f"daytime_scene/rgb/{SAMPLE_CAMERAS[1]}.mp4"
        caption = self.root / f"daytime_scene/captions/{SAMPLE_CAMERAS[2]}.json"
        wsm.write_bytes(b"edited")
        rgb.unlink()
        caption.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
        with self.assertRaises(ValueError) as raised:
            verify_sample(self.root)
        for path in (wsm, rgb, caption):
            self.assertIn(str(path), str(raised.exception))

    def test_captions_must_cover_the_requested_frames(self):
        caption = self.root / f"daytime_scene/captions/{SAMPLE_CAMERAS[0]}.json"
        self.assertEqual(read_caption(caption, 297), f"Daytime caption for {SAMPLE_CAMERAS[0]}.")
        with self.assertRaisesRegex(ValueError, "covering"):
            read_caption(caption, 301)

    def test_request_files_share_the_sample_cameras_and_media(self):
        for name in REQUESTS:
            with self.subTest(request=name):
                template = json.loads((ASSET_DIR / name).read_text())
                views = template["multiview"]["views"]
                self.assertEqual(tuple(view["camera_key"] for view in views), SAMPLE_CAMERAS)
                self.assertEqual(SAMPLE_CAMERAS, CAMERA_ORDER[:7])
                self.assertEqual((template["name"], template["fps"], template["seed"], template["resolution"]),
                                 ("daytime_scene", 30, 0, "480"))
                self.assertNotIn("negative_prompt", template)
                self.assertNotIn("per_view_negative_prompt", template)
                joint = "lidar" in template
                self.assertEqual(joint, name.endswith("_joint.json"))
                rgb = name != "transfer_auto_wsm.json"
                for view in views:
                    key = view["camera_key"]
                    self.assertEqual(view["control_path"], f"daytime_scene/wsm/{key}.mp4")
                    self.assertEqual(view["caption_path"], f"daytime_scene/captions/{key}.json")
                    self.assertEqual(view.get("vision_path"), f"daytime_scene/rgb/{key}.mp4" if rgb else None)
                if joint:
                    self.assertEqual(template["lidar"], {"control_path": "daytime_scene/lidar/control.tar", "return_output": True})
                    self.assertTrue((self.root / template["lidar"]["control_path"]).is_file())
                frames = 297 if name.startswith("sim_auto") else 201
                self.assertEqual(template["num_frames"], frames)

    def test_transfer_requests_upload_seven_cameras_and_the_lidar_control(self):
        for name, uploads, lidar in (("transfer_auto_wsm.json", 7, False), ("transfer_auto_joint.json", 15, True)):
            with self.subTest(request=name):
                manifest = resolve_request(ASSET_DIR / name, self.root, lidar_output=self.root / "prepared/lidar_control.safetensors")
                original = copy.deepcopy(manifest)
                form, paths = prepare_request(manifest, self.root, "model")
                extra = json.loads(form["extra_params"])
                self.assertEqual(manifest, original)
                self.assertEqual(len(paths), uploads)
                self.assertEqual(form["prompt"], f"Daytime caption for {SAMPLE_CAMERAS[0]}.")
                self.assertEqual((form["num_frames"], form["seed"], form["guidance_scale"], form["size"]),
                                 ("201", "0", "3", "832x480"))
                self.assertNotIn("per_view_negative_prompt", extra)
                for index, view in enumerate(extra["multiview"]["views"]):
                    self.assertEqual(view["prompt"], f"Daytime caption for {view['camera_key']}.")
                    if lidar:
                        self.assertEqual((view["control_reference_index"], view["vision_reference_index"]),
                                         (2 * index, 2 * index + 1))
                    else:
                        self.assertEqual(view["control_reference_index"], index)
                if lidar:
                    self.assertEqual(extra["lidar"], {"return_output": True, "control_reference_index": 14})
                    self.assertEqual(paths[-1], self.root / "prepared/lidar_control.safetensors")

    def test_hf_archive_symlink_still_selects_tar_conversion(self) -> None:
        archive = self.root / "daytime_scene/lidar/control.tar"
        blob = self.root / "blob-without-extension"
        archive.rename(blob)
        archive.symlink_to(blob)
        output = self.root / "converted.safetensors"
        with patch("helpers.daytime_sample.convert_control", side_effect=mock_convert) as convert:
            request = resolve_request(ASSET_DIR / "transfer_auto_joint.json", self.root, lidar_output=output)
        convert.assert_called_once_with(archive, output)
        self.assertEqual(request["lidar"]["control_path"], str(output))

    def test_transfer_requests_accept_ordered_subsets_only(self):
        manifest = resolve_request(ASSET_DIR / "transfer_auto_wsm.json", self.root)
        manifest["multiview"]["views"] = manifest["multiview"]["views"][::3]
        form, paths = prepare_request(manifest, self.root, "model")
        self.assertEqual([view["camera_key"] for view in json.loads(form["extra_params"])["multiview"]["views"]],
                         [CAMERA_ORDER[0], CAMERA_ORDER[3], CAMERA_ORDER[6]])
        for edit in (lambda views: views.reverse(), lambda views: views.append(copy.deepcopy(views[0])), lambda views: views.clear()):
            manifest = resolve_request(ASSET_DIR / "transfer_auto_wsm.json", self.root)
            edit(manifest["multiview"]["views"])
            with self.assertRaisesRegex(ValueError, "canonical camera order"):
                prepare_request(manifest, self.root, "model")

    def test_inline_requests_pass_through_with_resolved_paths(self):
        request = {"name": "own", "prompt": "Top", "num_frames": 201, "multiview": {"views": [
            {"camera_key": CAMERA_ORDER[0], "control_path": "wsm.mp4", "prompt": "Inline"},
        ]}}
        path = self.root / "own.json"
        path.write_text(json.dumps(request))
        manifest = resolve_request(path, self.root)
        self.assertEqual((manifest["prompt"], manifest["multiview"]["views"][0]["prompt"]), ("Top", "Inline"))
        self.assertEqual(manifest["multiview"]["views"][0]["control_path"], str(self.root / "wsm.mp4"))


class SglangRequestTests(unittest.TestCase):
    """The SGLang branch of the shared request helpers: server-visible paths, async jobs, LiDAR artifacts."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = make_sample(Path(temporary.name).resolve())
        converter = patch("helpers.daytime_sample.convert_control", side_effect=mock_convert)
        converter.start()
        self.addCleanup(converter.stop)
        self.lidar = self.root / "prepared" / "lidar_control.safetensors"

    def resolve(self, name):
        return resolve_request(ASSET_DIR / name, self.root, caption_folder="captions", lidar_output=self.lidar)

    def test_sglang_requests_carry_server_visible_paths_instead_of_uploads(self):
        for name in ("transfer_auto_wsm.json", "transfer_auto_joint.json"):
            with self.subTest(request=name):
                template = json.loads((ASSET_DIR / name).read_text())
                manifest = self.resolve(name)
                original = copy.deepcopy(manifest)
                form, uploads = prepare_request(manifest, self.root, "model", backend="sglang")
                vllm_form, vllm_uploads = prepare_request(manifest, self.root, "model")
                self.assertEqual(manifest, original)
                self.assertEqual(uploads, [])
                self.assertEqual(len(vllm_uploads), 15 if "lidar" in template else 7)
                self.assertEqual(
                    {key: value for key, value in form.items() if key != "extra_params"},
                    {key: value for key, value in vllm_form.items() if key != "extra_params"},
                )
                extra = json.loads(form["extra_params"])
                self.assertNotIn("model_mode", extra)
                for source, view in zip(template["multiview"]["views"], extra["multiview"]["views"], strict=True):
                    self.assertEqual([key for key in view if key.endswith("_reference_index")], [])
                    self.assertEqual(view["prompt"], f"Daytime caption for {source['camera_key']}.")
                    for role in ("control", "vision"):
                        if f"{role}_path" in source:
                            self.assertEqual(Path(view[f"{role}_path"]), self.root / source[f"{role}_path"])
                        else:
                            self.assertNotIn(f"{role}_path", view)
                if "lidar" in template:
                    self.assertEqual(extra["lidar"], {"control_path": str(self.lidar), "decode": True})
                    self.assertEqual(extra["control_guidance"], 1)
                else:
                    self.assertNotIn("lidar", extra)

    def test_sglang_lidar_decode_follows_return_output_and_unknown_backends_reject(self):
        manifest = self.resolve("transfer_auto_joint.json")
        manifest["lidar"]["return_output"] = False
        form, _ = prepare_request(manifest, self.root, "model", backend="sglang")
        self.assertFalse(json.loads(form["extra_params"])["lidar"]["decode"])
        with self.assertRaisesRegex(ValueError, "return_output=true"):
            prepare_request(manifest, self.root, "model")
        manifest["lidar"]["decode"] = True
        with self.assertRaisesRegex(ValueError, "agree"):
            prepare_request(manifest, self.root, "model", backend="sglang")
        with self.assertRaisesRegex(ValueError, "backend"):
            prepare_request(self.resolve("transfer_auto_wsm.json"), self.root, "model", backend="trt")
        with self.assertRaisesRegex(ValueError, "backend"):
            submit_joint(None, form, [], self.root / "unsubmitted", backend="trt")
        self.assertFalse((self.root / "unsubmitted").exists())  # Rejected before any request or output.

    def test_stage_conditions_copies_each_camera_under_its_key(self):
        views = self.resolve("transfer_auto_joint.json")["multiview"]["views"]
        staged = stage_conditions(views, self.root / "staged")
        self.assertEqual([path.name for path in staged], [f"{view['camera_key']}.mp4" for view in views])
        for view, path in zip(views, staged, strict=True):
            self.assertEqual(path.read_bytes(), Path(view["vision_path"]).read_bytes())
        self.assertEqual(stage_conditions(views, self.root / "staged"), staged)  # Idempotent.
        changed = Path(views[0]["vision_path"])
        changed.write_bytes(b"X" * changed.stat().st_size)  # Same length, different content.
        self.assertEqual(stage_conditions(views, self.root / "staged")[0].read_bytes(), changed.read_bytes())
        Path(views[0]["vision_path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            stage_conditions(views, self.root / "staged-again")

    def test_stage_conditions_keeps_the_request_suffix_for_snapshot_symlinks(self):
        # A Hugging Face snapshot names files normally but symlinks them into an extension-less blob store.
        views = self.resolve("transfer_auto_joint.json")["multiview"]["views"]
        blobs = self.root / "blobs"
        blobs.mkdir()
        for index, view in enumerate(views):
            named = Path(view["vision_path"])
            blob = blobs / f"{index:064x}"
            named.replace(blob)
            named.symlink_to(blob)
        staged = stage_conditions(views, self.root / "staged")
        self.assertEqual([path.name for path in staged], [f"{view['camera_key']}.mp4" for view in views])
        for view, path in zip(views, staged, strict=True):
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.read_bytes(), Path(view["vision_path"]).read_bytes())

    def test_sglang_joint_job_downloads_the_video_and_rangemap(self):
        import httpx

        manifest = self.resolve("transfer_auto_joint.json")
        form, uploads = prepare_request(manifest, self.root, "model", backend="sglang")
        descriptor = {"sweeps": 67, "fps": 10.0, "files": {"rangemap": "/srv/job-1.lidar.safetensors"}}
        polls = []

        def handler(request):
            path = request.url.path
            if path == "/v1/videos" and request.method == "POST":
                body = request.read()
                self.assertIn(b'name="extra_params"', body)
                self.assertNotIn(b'name="input_references"', body)
                return httpx.Response(200, json={"id": "job-1", "status": "queued"})
            if path == "/v1/videos/job-1":
                polls.append(request.method)
                return httpx.Response(200, json={"id": "job-1", "status": "completed", "lidar": descriptor})
            if path == "/v1/videos/job-1/content":
                if request.url.params.get("variant") == "rangemap":
                    return httpx.Response(200, content=b"mock rangemap")
                return httpx.Response(200, content=b"mock video", headers={"content-type": "video/mp4"})
            self.fail(f"Unexpected request {request.method} {request.url}")

        output_dir = self.root / "out"
        with (
            patch("time.sleep"),
            contextlib.redirect_stdout(io.StringIO()),
            httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)) as client,
        ):
            result = submit_joint(client, form, uploads, output_dir, backend="sglang")
        self.assertEqual((result["job_id"], result["descriptor"]), ("job-1", descriptor))
        self.assertEqual(result["video"].read_bytes(), b"mock video")
        self.assertEqual(result["lidar"].read_bytes(), b"mock rangemap")
        self.assertEqual(json.loads(result["lidar"].with_suffix(".json").read_text()), descriptor)
        self.assertEqual(json.loads((output_dir / "job.json").read_text())["status"], "completed")
        self.assertEqual(polls, ["GET"])

        # The vLLM-Omni contract is unchanged: uploads travel as input_references and no rangemap is fetched.
        vllm_form, vllm_uploads = prepare_request(manifest, self.root, "model")
        seen = []

        def vllm_handler(request):
            seen.append(request.url.path)
            if request.method == "POST":
                self.assertIn(b'name="input_references"', request.read())
                return httpx.Response(200, json={"id": "job-2", "status": "completed", "lidar": descriptor})
            return httpx.Response(200, content=b"mock video", headers={"content-type": "video/mp4"})

        with (
            patch("time.sleep"),
            contextlib.redirect_stdout(io.StringIO()),
            httpx.Client(base_url="http://test", transport=httpx.MockTransport(vllm_handler)) as client,
        ):
            result = submit_joint(client, vllm_form, vllm_uploads, self.root / "vllm")
        self.assertEqual(result, {
            "video": self.root / "vllm" / "multiview_camera_major.mp4", "descriptor": descriptor, "job_id": "job-2",
        })
        self.assertEqual(seen, ["/v1/videos", "/v1/videos/job-2/content"])

    def test_validate_sglang_lidar_checks_layout_metadata_and_masked_values(self):
        from safetensors.numpy import save_file

        sweeps = 2
        validity = np.zeros((sweeps, 128, 1800), dtype=np.uint8)
        validity[:, :64] = 1
        ranges = np.where(validity == 1, 12.5, 0.0).astype(np.float32)
        intensity = np.where(validity == 1, 0.25, 0.0).astype(np.float16)
        metadata = {
            "layout": "[sweeps, beams, azimuth]", "fps": "10.0",
            "min_range_m": "0.5", "max_range_m": "120.0", "invalid_range_m": "0",
        }
        path = self.root / "generated.safetensors"
        save_file({"range_m": ranges, "intensity": intensity, "validity": validity}, path, metadata=metadata)
        info = validate_sglang_lidar(path, sweeps)
        self.assertEqual((info["shape"], info["metadata"]["fps"]), ([sweeps, 128, 1800], "10.0"))
        with self.assertRaisesRegex(ValueError, "metadata"):
            validate_sglang_lidar(path, sweeps, fps=30.0)
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_sglang_lidar(path, sweeps + 1)
        broken = ranges.copy()
        broken[0, 0, 0] = 0.0  # A valid ray must have a positive range.
        save_file({"range_m": broken, "intensity": intensity, "validity": validity}, path, metadata=metadata)
        with self.assertRaisesRegex(ValueError, "masked"):
            validate_sglang_lidar(path, sweeps)


class LidarRangemapTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_pooling_keeps_the_nearest_nonzero_return(self):
        dense = np.zeros((128, 3600), np.float32)
        dense[0, 0:2] = 30, 12  # nearest return wins
        dense[1, 2:4] = 0, 40  # an empty neighbor is ignored
        dense[2, 3598:3600] = 7, 9  # the last group
        pooled = pool_nearest_return(dense)
        self.assertEqual(pooled.shape, (128, 1800))
        self.assertEqual((pooled[0, 0], pooled[1, 1], pooled[2, 1799]), (12, 40, 7))
        self.assertEqual(np.count_nonzero(pooled), 3)

    def test_conversion_writes_unit_intensity_and_validity_on_the_camera_timeline(self):
        archive = make_archive(self.root / "control.tar", {
            0: sparse([5, 5, 127], [10, 11, 3599], [20.5, 8.25, 99.0]),
            3: sparse([0], [0], [6.0]),
        })
        output = self.root / "out" / "control.safetensors"
        summary = convert_control(archive, output)
        self.assertEqual((summary["shape"], summary["frame_ids"], summary["clip"]), ([3, 2, 128, 1800], [0, 3], "scene"))
        from safetensors import safe_open

        with safe_open(output, framework="numpy") as handle:
            self.assertEqual(list(handle.keys()), ["frames"])
            frames = handle.get_tensor("frames")
            source = json.loads(handle.metadata()["source"])
        self.assertEqual(frames.dtype, np.float32)
        self.assertEqual((frames[0, 0, 5, 5], frames[0, 0, 127, 1799], frames[0, 1, 0, 0]), (8.25, 99.0, 6.0))
        np.testing.assert_array_equal(frames[1], frames[2])
        np.testing.assert_array_equal(frames[2], (frames[0] > 0).astype(np.float32))
        self.assertEqual(source["archive_sha256"], hashlib.sha256(archive.read_bytes()).hexdigest())

    def test_malformed_archives_are_rejected(self):
        good = sparse([1], [1], [10.0])
        cases = {
            "advance by 3": {0: good, 6: good},
            "missing": {0: {"row": good["row"], "col": good["col"]}},
            "repeat a pixel": {0: sparse([1, 1], [2, 2], [10.0, 11.0])},
            "outside": {0: sparse([1], [3600], [10.0])},
            "positive": {0: sparse([1], [1], [0.0])},
        }
        for message, sweeps in cases.items():
            with self.subTest(message=message):
                archive = make_archive(self.root / f"{len(message)}.tar", sweeps)
                with self.assertRaisesRegex(ValueError, message):
                    convert_control(archive, self.root / "out.safetensors")
        archive = make_archive(self.root / "target.tar", {0: {**good, "intensity": np.array([0.5], np.float16)}})
        with self.assertRaisesRegex(ValueError, "only row, col, and range"):
            convert_control(archive, self.root / "out.safetensors")



class TransferNotebookTests(unittest.TestCase):
    # Cells that decode real safetensors LiDAR; the fixture's LiDAR file is a placeholder.
    SKIPPED = {"lidar-preview", "lidar-visualize", "lidar-video"}

    def test_both_examples_run_against_a_mock_server(self):
        import contextlib
        import os
        from unittest.mock import patch

        import httpx
        import IPython.display
        from PIL import Image as PILImage

        from helpers import auto_helpers

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        sample = make_sample(root / "sample")
        requests = root / "requests"
        requests.mkdir()
        for name in ("transfer_auto_wsm.json", "transfer_auto_joint.json"):
            (requests / name).write_text((ASSET_DIR / name).read_text())
        (requests / "lidar_control.safetensors").write_bytes(b"lidar placeholder")

        notebook = json.loads((AUTO_ROOT / "run_auto_with_vllm_omni.ipynb").read_text())
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"}
        for variable, name in (("WSM_REQUEST", "transfer_auto_wsm.json"), ("LIDAR_REQUEST", "transfer_auto_joint.json")):
            old = f'{variable} = AUTO_ROOT / "assets" / "daytime_scene" / "{name}"'
            self.assertEqual(sum(old in cell for cell in cells.values()), 1, old)
            cells = {key: cell.replace(old, f"{variable} = Path({str(requests / name)!r})") for key, cell in cells.items()}
        self.assertTrue(self.SKIPPED <= cells.keys())

        original_client = httpx.Client

        def handler(request):
            path = request.url.path
            if path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": "nvidia/Cosmos3-Nano-Transfer-Auto"}]})
            if path == "/v1/videos/sync":
                self.assertEqual(request.read().count(b'name="input_references"'), 7)
                return httpx.Response(200, content=b"mock video", headers={"content-type": "video/mp4"})
            if path == "/v1/videos" and request.method == "POST":
                self.assertEqual(request.read().count(b'name="input_references"'), 15)
                return httpx.Response(200, json={"id": "video-1", "status": "completed", "lidar": {"fps": 10.0}})
            if path == "/v1/videos/video-1/content":
                return httpx.Response(200, content=b"mock joint video")
            if path == "/v1/videos/video-1/lidar":
                return httpx.Response(200, content=b"mock lidar")
            raise AssertionError(f"Unexpected request {request.method} {path}")

        def split(video, output, cameras, frames, fps):
            self.assertEqual((tuple(cameras), frames, fps), (SAMPLE_CAMERAS, 201, 30))
            paths = {camera: output / f"{camera}.mp4" for camera in cameras}
            for path in paths.values():
                path.write_bytes(b"mock camera")
            return paths

        controls = [sample / f"daytime_scene/wsm/{camera}.mp4" for camera in SAMPLE_CAMERAS]
        lidar_info = {"shape": [3, 100, 128, 1800], "dtype": "float32", "metadata": {}}
        environment = {
            "COSMOS_COOKBOOK_ROOT": str(AUTO_ROOT.parents[3]), "MPLBACKEND": "Agg",
            "VLLM_BASE_URL": "http://test", "VLLM_API_KEY": "",
            "COSMOS3_OUTPUT_DIR": str(root / "runs"), "AUTO_SAMPLE_ROOT": str(sample),
        }
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch.dict(os.environ, environment),
            patch.object(IPython.display, "display"),
            patch.object(IPython.display, "HTML"),
            patch.object(httpx, "Client", side_effect=lambda *a, **k: original_client(*a, transport=httpx.MockTransport(handler), **k)),
            patch("helpers.daytime_sample.convert_control", side_effect=mock_convert),
            patch.object(auto_helpers, "validate_rgb_conditions"),
            patch.object(auto_helpers, "prepare_controls", return_value=controls),
            patch.object(auto_helpers, "split_cameras", side_effect=split),
            patch.object(auto_helpers, "validate_lidar", return_value=lidar_info),
            patch.object(auto_helpers, "validate_lidar_output", return_value=lidar_info) as validate_output,
        ):
            scope = {}
            for cell_id, cell in cells.items():
                if cell_id in self.SKIPPED:
                    continue
                exec(cell, scope)
                if "def validate_video" in cell:
                    scope.update(
                        validate_video=lambda path, **expected: {
                            "frames": expected["num_frames"], "fps": float(expected["fps"]),
                            "width": expected["width"], "height": expected["height"],
                        },
                        probe_video=lambda path: {"width": 1280, "height": 720, "fps": 30, "frames": 201},
                        read_frame=lambda path, index: PILImage.new("RGB", (8, 8)),
                        video_row=lambda videos, **kwargs: "",
                    )
        output = root / "runs"
        for example in ("wsm", "lidar"):
            record, = json.loads((output / example / "sample_outputs.json").read_text())
            self.assertEqual((record["status"], len(record["cameras"]), record["frames_per_camera"]), ("success", 7, 201))
        self.assertEqual((output / "lidar" / "multiview_camera_major.lidar.safetensors").read_bytes(), b"mock lidar")
        self.assertEqual(validate_output.call_args.args[2], 67)
        extra = json.loads(json.loads((output / "lidar" / "request.json").read_text())["extra_params"])
        self.assertEqual(extra["lidar"], {"return_output": True, "control_reference_index": 14})


if __name__ == "__main__":
    unittest.main()
