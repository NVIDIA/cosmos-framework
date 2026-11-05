"""Exercise native notebook preparation with portable sample-shaped inputs, without GPUs."""

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import venv
from pathlib import Path
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

AUTO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AUTO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers.auto_helpers import validate_framework_outputs
from helpers.daytime_sample import CAPTION_FOLDERS, native_request
from helpers.sim_auto_helpers import validate_sim_auto_outputs
from test_daytime_sample import make_sample

NOTEBOOKS = ("run_auto_with_cosmos_framework.ipynb", "run_sim-auto_with_cosmos_framework.ipynb")


def notebook_cells(name: str) -> dict[str, str]:
    doc = json.loads((AUTO_ROOT / name).read_text())
    return {cell["id"]: "".join(cell["source"]) for cell in doc["cells"] if cell["cell_type"] == "code"}


class NotebookRootDiscoveryTests(unittest.TestCase):
    def test_vllm_notebook_imports_shared_helpers_in_a_fresh_process(self) -> None:
        # Execute the real notebook setup without a service/model. A new process
        # catches missing modules and prevents a previously imported helper from
        # hiding a broken package path in a long-lived notebook kernel.
        script = r"""
import ast
import json
from pathlib import Path
import sys

notebook = Path(sys.argv[1])
is_sim = "sim_auto" in notebook.name
document = json.loads(notebook.read_text())
namespace = {}
setup_id = "sim-auto-05" if is_sim else "shared-imports"
setup = next(cell for cell in document["cells"] if cell.get("id") == setup_id)
exec(compile("".join(setup["source"]), "shared-imports", "exec"), namespace)
cell_ids = ("sim-auto-joint-visualize", "sim-auto-joint-video") if is_sim else ("lidar-visualize", "lidar-video")
for cell_id in cell_ids:
    cell = next(cell for cell in document["cells"] if cell.get("id") == cell_id)
    imports = [node for node in ast.parse("".join(cell["source"])).body
               if isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(ast.Module(body=imports, type_ignores=[]), cell_id, "exec"), namespace)
helper_root = notebook.parent / "helpers"
for name, expected_module in {
    "prepare_request": "helpers.sim_auto_vllm_omni_helpers" if is_sim else "helpers.auto_helpers",
    "prepare_controls": "helpers.auto_helpers",
    "split_cameras": "helpers.auto_helpers",
    "prepare_lidar_sweep": "helpers.lidar_visualization",
    "draw_lidar_sweep": "helpers.lidar_visualization",
    "render_lidar_video": "helpers.lidar_visualization",
}.items():
    function = namespace[name]
    assert callable(function) and function.__module__ == expected_module, name
    assert Path(sys.modules[expected_module].__file__).resolve().parent == helper_root.resolve()
import numpy as np
import lidar_visualization
from cosmos_framework.model.generator.tokenizers.lidar.geometry import PANDAR128_ELEVATIONS_DEG, range_map_to_xyz

sweep = np.zeros((3, 128, 1800), dtype=np.float32)  # [3,H,W]
sweep[:, (0, 64, 127), (0, 900, 1799)] = np.array([[10.0], [0.5], [1.0]])  # [3,H,W]
expected, _ = range_map_to_xyz(sweep[0], sweep[2] > 0)  # [N,3]
for prepare in (namespace["prepare_lidar_sweep"], lidar_visualization.prepare_lidar_sweep):
    assert prepare.__globals__["PANDAR128_ELEVATIONS_DEG"] is PANDAR128_ELEVATIONS_DEG
    preview = prepare(sweep, {"range_projection": {}, "validity_threshold": 0.5}, smooth_geometry=False)
    np.testing.assert_allclose(preview.xyz, expected, rtol=1e-6, atol=1e-6)
assert "torch" not in sys.modules, "CPU notebook imports must not load the GPU runtime"
"""
        for name in ("run_auto_with_vllm_omni.ipynb", "run_sim_auto_with_vllm_omni.ipynb"):
            notebook = AUTO_ROOT / name
            for start in (AUTO_ROOT.parents[5], AUTO_ROOT.parents[3], AUTO_ROOT):
                with self.subTest(notebook=name, start=start), tempfile.TemporaryDirectory() as temporary:
                    env = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=temporary, COSMOS3_OUTPUT_DIR=temporary)
                    env.pop("COSMOS_COOKBOOK_ROOT", None)
                    env.pop("PYTHONPATH", None)
                    result = subprocess.run(
                        [sys.executable, "-c", script, str(notebook)],
                        cwd=start,
                        env=env,
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

class NativeNotebookTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sample = make_sample(self.root / "sample")

    def test_notebook_interpreters_keep_the_selected_virtual_environment(self) -> None:
        environment = self.root / "venv"
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
        python = environment / "bin/python"
        for name in (*NOTEBOOKS, "generate_wsm.ipynb"):
            variable = "WSM_PYTHON" if name == "generate_wsm.ipynb" else "FRAMEWORK_PYTHON"
            binding = next(
                node
                for node in ast.parse(notebook_cells(name)["configure"]).body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == variable for target in node.targets)
            )
            with (
                self.subTest(notebook=name),
                patch.dict(
                    os.environ,
                    {"WSM_PYTHON": str(python), "COSMOS_AUTO_PYTHON": str(python)},
                ),
            ):
                scope = {"os": os, "Path": Path, "sys": sys}
                exec(compile(ast.Module(body=[binding], type_ignores=[]), name, "exec"), scope)
                prefix = subprocess.check_output(
                    [str(scope[variable]), "-c", "import sys; print(sys.prefix)"],
                    text=True,
                ).strip()
                self.assertEqual(Path(prefix), environment)

    def test_native_selection_preserves_camera_order_captions_and_archive(self) -> None:
        path = self.sample / "input-transfer-auto.jsonl"
        original = path.read_text()
        keys = [view["camera_key"] for view in json.loads(original)["multiview"]["views"]]
        for mode in ("camera", "joint", "text2video"):
            for folder in CAPTION_FOLDERS:
                with self.subTest(mode=mode, folder=folder):
                    request = native_request(path, mode=mode, caption_folder=folder)
                    self.assertEqual([v["camera_key"] for v in request["multiview"]["views"]], keys)
                    self.assertEqual("lidar" in request, mode == "joint")
                    for view in request["multiview"]["views"]:
                        self.assertEqual(Path(view["caption_path"]).parent.name, folder)
                        self.assertEqual("vision_path" in view, mode != "text2video")
                        self.assertTrue(Path(view["control_path"]).is_absolute())
                    if mode == "joint":
                        self.assertEqual(
                            request["lidar"]["control_path"], str(self.sample / "daytime_scene/lidar/control.tar")
                        )
        self.assertEqual(path.read_text(), original)
        custom = native_request(path, mode="joint")
        self.assertEqual(Path(custom["multiview"]["views"][0]["caption_path"]).parent.name, "captions")

    def test_joint_without_rgb_keeps_lidar_and_disables_image_conditioning(self) -> None:
        path = self.sample / "input-transfer-auto.jsonl"
        original = json.loads(path.read_text())
        for view in original["multiview"]["views"]:
            del view["vision_path"]
        path.write_text(json.dumps(original))
        request = native_request(path, mode="joint")
        self.assertEqual(request["model_mode"], "text2video")
        self.assertFalse(request["multiview"]["condition_video_as_image"])
        self.assertEqual(request["num_conditional_frames"], 0)
        self.assertEqual(request["condition_frame_indexes_vision"], [])
        self.assertEqual(request["lidar"]["control_path"], str(self.sample / "daytime_scene/lidar/control.tar"))
        self.assertTrue(all("vision_path" not in view for view in request["multiview"]["views"]))
        self.assertEqual(json.loads(path.read_text()), original)

    def test_joint_rejects_partial_rgb_conditioning(self) -> None:
        path = self.sample / "input-transfer-auto.jsonl"
        request = json.loads(path.read_text())
        del request["multiview"]["views"][0]["vision_path"]
        path.write_text(json.dumps(request))
        with self.assertRaisesRegex(ValueError, "every selected camera, or none"):
            native_request(path, mode="joint")

    def test_hf_snapshot_symlink_keeps_relative_media_at_the_snapshot_root(self) -> None:
        request_path = self.sample / "input-transfer-auto.jsonl"
        blob = self.root / "blobs" / "request"
        blob.parent.mkdir()
        blob.write_bytes(request_path.read_bytes())
        request_path.unlink()
        request_path.symlink_to(blob)
        archive = self.sample / "daytime_scene/lidar/control.tar"
        archive_blob = blob.parent / "range-data"
        archive_blob.write_bytes(archive.read_bytes())
        archive.unlink()
        archive.symlink_to(archive_blob)
        request = native_request(request_path, mode="joint")
        self.assertTrue(Path(request["multiview"]["views"][0]["control_path"]).is_file())
        self.assertEqual(request["lidar"]["control_path"], str(self.sample / "daytime_scene/lidar/control.tar"))

    def test_notebook_setup_prepare_and_launch_all_modes(self) -> None:
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text("{}")
        env = {
            "COSMOS_COOKBOOK_ROOT": str(AUTO_ROOT),
            "AUTO_SAMPLE_ROOT": str(self.sample),
            "COSMOS_AUTO_CHECKPOINT": str(checkpoint),
        }
        for name in NOTEBOOKS:
            cells = notebook_cells(name)
            for mode in ("camera", "joint", "text2video"):
                with (
                    self.subTest(notebook=name, mode=mode),
                    patch.dict(os.environ, env),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    scope = {}
                    exec(cells["configure"], scope)
                    scope.update(MODE=mode, OUTPUT_ROOT=self.root / name / mode)
                    exec(cells["download"], scope)
                    scope.update(
                        probe_video=lambda path: {"frames": 300, "fps": 30},
                        validate_rgb_conditions=lambda views, root: None,
                    )
                    exec(cells["prepare"], scope)
                    with patch("subprocess.run") as launch:
                        exec(cells["generate"], scope)
                    command = launch.call_args.args[0]
                    self.assertIn("cosmos_framework.scripts.inference", command)
                    self.assertEqual(command[command.index("--checkpoint-path") + 1], str(checkpoint))
                    request = json.loads(scope["prepared_request"].read_text())
                    self.assertEqual(len(request["multiview"]["views"]), 7)
                    self.assertEqual((request["num_frames"], request["resolution"]), (297, "720"))
                    self.assertEqual(request["guidance"], 1 if "sim-auto" in name else 3)
                    if "sim-auto" in name:
                        self.assertTrue(request["autoregressive"])
                        self.assertNotIn("num_steps", request)
                        self.assertNotIn("per_view_negative_prompt", request)
                    else:
                        self.assertEqual((request["num_steps"], request["shift"]), (35, 10))
                        self.assertTrue(request["per_view_negative_prompt"].strip())

    def test_checkout_discovery_and_clean_notebooks(self) -> None:
        for name in (*NOTEBOOKS, "generate_wsm.ipynb"):
            doc = json.loads((AUTO_ROOT / name).read_text())
            for cell in doc["cells"]:
                if cell["cell_type"] == "code":
                    self.assertFalse(cell["outputs"])
                    self.assertIsNone(cell["execution_count"])
            code = notebook_cells(name)["configure"]
            function = next(
                node
                for node in ast.parse(code).body
                if isinstance(node, ast.FunctionDef) and node.name == "find_auto_root"
            )
            for folder in (AUTO_ROOT, AUTO_ROOT.parents[3], AUTO_ROOT.parents[5]):
                with patch.dict(os.environ, {"COSMOS_COOKBOOK_ROOT": str(folder)}):
                    scope = {"os": os, "Path": Path}
                    exec(compile(ast.Module(body=[function], type_ignores=[]), name, "exec"), scope)
                    self.assertEqual(scope["find_auto_root"](), AUTO_ROOT)

    def test_output_validation_checks_resolution_lidar_and_causal_horizon(self) -> None:
        request = native_request(self.sample / "input-sim-auto.jsonl", mode="joint")
        request["num_frames"] = 9
        output = self.root / "generated"
        output.mkdir()
        for index, view in enumerate(request["multiview"]["views"]):
            (output / f"vision_view{index:02d}_{view['camera_key']}.mp4").write_bytes(b"video")
        (output / "sample_outputs.json").write_text(json.dumps({"status": "success"}))
        (output / "rolling_prompt.json").write_text(json.dumps({"num_frames": 9, "source_frame_range": [0, 8]}))
        (output / "caption_chunks.json").write_text(
            json.dumps([{"source_frame_range": [0, 8], "prompt": ["scene"] * 7}])
        )
        # LiDAR tensors are [sweeps, rows, columns], plus one source frame ID per sweep.
        tensors = {
            "range_m": np.full((3, 128, 1800), 20, np.float32),
            "intensity": np.full((3, 128, 1800), 0.5, np.float32),
            "frame_ids": np.array([0, 3, 6], np.int64),
        }
        save_file(tensors, str(output / "lidar.safetensors"))
        for resolution, width, height in (("720", 1280, 720), ("480", 832, 480)):
            request["resolution"] = resolution
            metadata = {"width": width, "height": height, "frames": 9, "fps": 30}
            with patch("helpers.auto_helpers.probe_video", return_value=metadata):
                self.assertEqual(validate_framework_outputs(output, request)["lidar_sweeps"], 3)
                self.assertEqual(validate_sim_auto_outputs(output, request)["source_frame_range"], [0, 8])
                metadata["frames"] = 8
                with self.assertRaisesRegex(ValueError, "Unexpected output video"):
                    validate_framework_outputs(output, request)
        with patch(
            "helpers.auto_helpers.probe_video", return_value={"width": 832, "height": 480, "frames": 9, "fps": 30}
        ):
            tensors["frame_ids"][2] = 7
            save_file(tensors, str(output / "lidar.safetensors"))
            with self.assertRaisesRegex(ValueError, "timestamps"):
                validate_sim_auto_outputs(output, request)
            tensors["frame_ids"][2] = 6
            save_file(tensors, str(output / "lidar.safetensors"))
            (output / "rolling_prompt.json").write_text(json.dumps({"num_frames": 5, "source_frame_range": [0, 4]}))
            with self.assertRaisesRegex(ValueError, "rolling schedule"):
                validate_sim_auto_outputs(output, request)


if __name__ == "__main__":
    unittest.main()
