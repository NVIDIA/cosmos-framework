# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exercise the public notebook's input and output checks without a server."""

import ast
import json
import shlex
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / ".agents/skills/cookbooks/cosmos3/simulation/bimanual/run_bimanual_with_vllm_omni.ipynb"
)


@pytest.fixture
def notebook():
    contents = json.loads(NOTEBOOK.read_text())
    cells = ["".join(cell["source"]) for cell in contents["cells"] if cell["cell_type"] == "code"]
    functions = []
    for cell in cells:
        functions.extend(
            node
            for node in ast.parse(cell).body
            if isinstance(node, ast.FunctionDef) and node.name in {"validate_actions", "expected_video_info"}
        )
    namespace = {"np": np, "Image": Image, "Fraction": Fraction}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    namespace["output_cell"] = next(cell for cell in cells if "for output in outputs:" in cell)
    return namespace


@pytest.fixture
def metadata():
    return {
        "temporal_compression_factor": 4,
        "vae_spatial_compression_factor": 16,
        "fixed_step_sampler_config": {"t_list": [1.0, 0.9375, 0.8333333333333334, 0.625]},
        "conditioning": {
            "schema_version": 5,
            "input_contract": {"action_space": "normalized_unified_v1", "action_dim": 59},
            "embodiments": {"agibotworld": {"domain_id": 15}, "camera_pose": {"domain_id": 2}},
        },
    }


def inputs(count=8, fps=30):
    row = {
        "name": "sample",
        "action_chunk_size": count,
        "num_frames": count + 1,
        "raw_action_dim": 59,
        "fps": fps,
        "num_steps": 4,
        "format_prompt_as_json": False,
    }
    sidecar = {
        "action_space": "normalized_unified_v1",
        "action": np.zeros((count, 59), dtype=np.float32).tolist(),
        "domain_names": "agibotworld",
    }
    return row, sidecar


def test_validation_server_keeps_the_guardrail_gate_in_the_stage_config():
    contents = json.loads(NOTEBOOK.read_text())
    blocks = [
        block.split("```", 1)[0]
        for cell in contents["cells"]
        if cell["cell_type"] == "markdown"
        for block in "".join(cell["source"]).split("```bash\n")[1:]
    ]
    command = next(block for block in blocks if "docker run" in block)
    arguments = shlex.split(command.replace("\\\n", " "))
    stages = json.loads(arguments[arguments.index("--stage-overrides") + 1])
    assert stages["0"]["model_config"]["guardrails"] is False


@pytest.mark.parametrize("count", [4, 300, 900, 1800])
@pytest.mark.parametrize("fps", [15, 24, 30])
@pytest.mark.parametrize("mixed", [False, True])
def test_action_length_and_domain_routing_follow_the_selected_row(notebook, metadata, count, fps, mixed):
    row, sidecar = inputs(count, fps)
    expected_domains = [15] * count
    if mixed:
        sidecar["domain_names"] = ["agibotworld"] * (count // 2) + ["camera_pose"] * (count // 2)
        expected_domains = [15] * (count // 2) + [2] * (count // 2)
    actions, domains = notebook["validate_actions"](row, sidecar, metadata)
    assert actions.shape == (count, 59)
    assert domains == expected_domains


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"num_frames": 1, "action_chunk_size": 0}, "positive"),
        ({"action_chunk_size": 4}, "num_frames - 1"),
        ({"num_frames": 8, "action_chunk_size": 7}, "temporal compression"),
        ({"raw_action_dim": 64}, "59D"),
        ({"num_steps": 2}, "sampling schedule"),
        ({"format_prompt_as_json": True}, "plain-text"),
        ({"fps": 0}, "positive and finite"),
        ({"fps": float("inf")}, "positive and finite"),
    ],
)
def test_invalid_row_contract_is_rejected(notebook, metadata, changes, message):
    row, sidecar = inputs()
    row.update(changes)
    with pytest.raises(ValueError, match=message):
        notebook["validate_actions"](row, sidecar, metadata)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"action": [[0.0] * 64] * 8}, "finite 59D"),
        ({"action": [[float("nan")] * 59] * 8}, "finite 59D"),
        ({"action": [[float("inf")] * 59] * 8}, "finite 59D"),
        ({"domain_names": ["agibotworld"]}, "one domain name per action"),
        ({"domain_names": "unknown"}, "Unknown action domains"),
        ({"action_space": "raw"}, "Action space"),
    ],
)
def test_invalid_sidecar_is_rejected(notebook, metadata, changes, message):
    row, sidecar = inputs()
    sidecar.update(changes)
    with pytest.raises(ValueError, match=message):
        notebook["validate_actions"](row, sidecar, metadata)


@pytest.mark.parametrize("count", [4, 900, 1800])
@pytest.mark.parametrize("fps", [15, 24, 30])
@pytest.mark.parametrize("size", [(640, 360), (640, 480), (832, 468)])
def test_output_metadata_follows_length_fps_and_image_geometry(notebook, metadata, count, fps, size):
    row, _ = inputs(count, fps)
    result = notebook["expected_video_info"](row, Image.new("RGB", size), metadata)
    assert result == {
        "width": size[0] // 16 * 16,
        "height": size[1] // 16 * 16,
        "frames": count + 1,
        "fps": Fraction(fps),
    }


@pytest.mark.parametrize("count", [4, 1800])
@pytest.mark.parametrize("incorrect", ["frames", "fps", "width", "height"])
def test_output_cell_checks_the_manifest_before_decoding(notebook, metadata, tmp_path, count, incorrect):
    row, _ = inputs(count, 24)
    image = Image.new("RGB", (832, 468))
    info = {"width": 832, "height": 464, "fps": Fraction(24), "frames": count + 1}
    info[incorrect] += 1
    decode_calls = []

    def decode(*args, **kwargs):
        decode_calls.append(args)
        return SimpleNamespace(returncode=0, stderr="")

    scope = {
        **notebook,
        "json": json,
        "metadata": metadata,
        "outputs": [
            {
                "row": row,
                "image": image,
                "video_path": tmp_path / "video.mp4",
                "sample_dir": tmp_path,
                "elapsed_seconds": 1.0,
            }
        ],
        "probe_video": lambda path: info,
        "subprocess": SimpleNamespace(run=decode),
        "Video": lambda **kwargs: None,
        "display": lambda *args: None,
    }
    with pytest.raises(ValueError, match="Unexpected video metadata"):
        exec(notebook["output_cell"], scope)
    assert not decode_calls
