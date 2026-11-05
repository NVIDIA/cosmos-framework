# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T rollout body: observation/action adaptation, startup guards, and the weight-version gate.

The body is built with ``object.__new__`` and given only the attributes each seam reads -- no engine.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from cosmos_framework.posttrain.config.model import Gr00tN1d7ModelConfig
from cosmos_framework.posttrain.model.gr00t.serving.body import Gr00tPolicyBody


def _body(**config_overrides) -> Gr00tPolicyBody:
    body = object.__new__(Gr00tPolicyBody)
    body.config = Gr00tN1d7ModelConfig(
        checkpoint_path="unused",
        camera_map={"exterior_image_1_left": "cam_a", "wrist_image_left": "cam_b"},
        state_map={"joint_position": "joint", "gripper_position": "gripper"},
        language_key="annotation.language.language_instruction",
        **config_overrides,
    )
    body._weight_version = 0
    body.ppo = False
    body._ppo_save_cfg_dir = ""
    body._model_path = body.config.checkpoint_path
    return body


def _obs(prompt: str = "pick up the cube") -> dict:
    return {
        "cam_a": np.zeros((8, 8, 3), dtype=np.uint8),
        "cam_b": np.ones((8, 8, 3), dtype=np.uint8),
        "joint": np.zeros(7, dtype=np.float32),
        "gripper": np.ones(1, dtype=np.float32),
        "prompt": prompt,
    }


def _body_reporting(described: dict) -> Gr00tPolicyBody:
    """A body whose worker RPC returns `described`, nested the way the real engine nests it."""
    body = _body()
    body._omni = type(
        "O", (), {"engine": type("E", (), {"collective_rpc": staticmethod(lambda m, **_kw: [[described]])})()}
    )()
    return body


@pytest.mark.L0
@pytest.mark.CPU
def test_ppo_startup_requires_the_worker_to_attest_bound_source_contents(tmp_path):
    from cosmos_framework.posttrain.model.gr00t.source import resolve_ppo_source
    from cosmos_framework.posttrain.tests.helpers.gr00t_source import SourceGaussianPolicy, write_source

    source = write_source(tmp_path / "source")
    policy = SourceGaussianPolicy(source)
    identity = policy.ppo_identity()["source"]
    report = {
        **_RL_REPORT,
        "ppo": True,
        "sdpa_math_effective": True,
        "model_path": str(source),
        "source_identity": identity,
    }
    body = _body_reporting(report)
    body.ppo = True
    body.config = policy.config
    body._model_path = str(source)
    body._ppo_source_identity = identity
    body.pipeline_info = body._verify_rl_pipeline()
    assert body.ppo_identity() == policy.ppo_identity()
    (source / "statistics.json").write_text('{"scale":2.0}')
    report["source_identity"] = resolve_ppo_source(str(source)).identity()
    with pytest.raises(RuntimeError, match="different model or processor source"):
        body._verify_rl_pipeline()


# What a healthy worker reports for `_body()`: the RL pipeline, the reference transform, the recipe's
# step count (default 4) and the path the body handed the engine.
_RL_REPORT = {
    "class": "Gr00tN1d7RLPipeline",
    "module": "cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline",
    "is_rl_pipeline": True,
    "image_transform": "reference-albumentations",
    "num_inference_timesteps": 4,
    "model_path": "unused",
}


@pytest.mark.L0
@pytest.mark.CPU
def test_build_robot_obs_keys_each_modality_by_gr00t_modality_key() -> None:
    """`Gr00tPolicy.check_observation` requires a DICT per modality, keyed by modality key."""
    body = _body()
    robot_obs, prompt = body._build_robot_obs([_obs(), _obs()])

    assert set(robot_obs["video"]) == {"exterior_image_1_left", "wrist_image_left"}
    for value in robot_obs["video"].values():
        assert value.shape == (2, 1, 8, 8, 3)  # [B, T=1, H, W, C]
        assert value.dtype == np.uint8

    assert set(robot_obs["state"]) == {"joint_position", "gripper_position"}
    assert robot_obs["state"]["joint_position"].shape == (2, 1, 7)  # [B, T=1, D]
    assert robot_obs["state"]["gripper_position"].shape == (2, 1, 1)
    assert robot_obs["state"]["joint_position"].dtype == np.float32

    # One prompt per batch element -- the stock helper only builds B=1.
    assert robot_obs["language"] == {"annotation.language.language_instruction": [["pick up the cube"]] * 2}
    assert prompt == "pick up the cube"


@pytest.mark.L0
@pytest.mark.CPU
def test_build_robot_obs_routes_env_keys_to_the_right_modality() -> None:
    """The camera map is a mapping, not a pair of parallel lists; a swap must be visible."""
    body = _body()
    obs = _obs()
    obs["cam_a"] = np.full((8, 8, 3), 3, dtype=np.uint8)
    obs["cam_b"] = np.full((8, 8, 3), 7, dtype=np.uint8)
    robot_obs, _ = body._build_robot_obs([obs])
    assert np.all(robot_obs["video"]["exterior_image_1_left"] == 3)
    assert np.all(robot_obs["video"]["wrist_image_left"] == 7)


@pytest.mark.L0
@pytest.mark.CPU
def test_build_robot_obs_keeps_per_element_prompts() -> None:
    body = _body()
    robot_obs, _ = body._build_robot_obs([_obs("open the drawer"), _obs("close the drawer")])
    assert robot_obs["language"]["annotation.language.language_instruction"] == [
        ["open the drawer"],
        ["close the drawer"],
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_missing_observation_key_names_the_keys_actually_present() -> None:
    """The DROID schema lives in robolab, which posttrain cannot import; report what the env sent."""
    body = _body()
    obs = _obs()
    del obs["cam_b"]
    with pytest.raises(KeyError) as excinfo:
        body._build_robot_obs([obs])
    message = str(excinfo.value)
    assert "cam_b" in message and "cam_a" in message and "gr00t.camera_map" in message


@pytest.mark.L0
@pytest.mark.CPU
def test_non_mapping_observation_rejected() -> None:
    body = _body()
    with pytest.raises(TypeError, match="must be a mapping"):
        body._build_robot_obs([["not", "a", "dict"]])


@pytest.mark.L0
@pytest.mark.CPU
def test_camera_frame_that_is_not_uint8_is_refused_not_cast() -> None:
    """`astype(uint8)` would turn a float image into plausible pixels; name the key and the element."""
    body = _body()
    bad = _obs()
    bad["cam_b"] = np.zeros((8, 8, 3), dtype=np.float32)
    with pytest.raises(TypeError) as excinfo:
        body._build_robot_obs([_obs(), bad])
    message = str(excinfo.value)
    assert "observation 1" in message and "'cam_b'" in message and "float32" in message


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("shape", [(3, 8, 8), (8, 8, 4), (8, 8)])
def test_camera_frame_must_be_hwc_rgb(shape: tuple[int, ...]) -> None:
    """CHW, RGBA and single-channel frames are refused before the eval image transform sees them."""
    body = _body()
    bad = _obs()
    bad["cam_a"] = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError) as excinfo:
        body._build_robot_obs([bad])
    message = str(excinfo.value)
    assert "observation 0" in message and "'cam_a'" in message and str(shape) in message


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_hand_the_engine_the_resolved_path_and_the_step_count() -> None:
    """The engine sees the pinned directory and the recipe's `denoising_steps`, and never a class name."""
    body = _body(denoising_steps=6)
    body._model_path = "/hf/snapshots/deadbeef"
    body._embodiment = type("Tag", (), {"name": "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"})()
    kwargs = body._engine_kwargs()
    assert kwargs["model"] == "/hf/snapshots/deadbeef"
    assert kwargs["model_config"]["denoising_steps"] == 6
    assert kwargs["model_config"]["embodiment_tag"] == "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"
    assert "model_class_name" not in kwargs


@pytest.mark.L0
@pytest.mark.CPU
def test_extract_actions_concatenates_dict_keys_in_sorted_order() -> None:
    """Keep the required-gate node ID; check both sorted and reversed processor orders."""

    class _Out:
        multimodal_output = {
            "actions": {
                "action.gripper": np.full((1, 4, 1), 9.0, dtype=np.float32),
                "action.arm": np.zeros((1, 4, 7), dtype=np.float32),
            }
        }

    actions = Gr00tPolicyBody._extract_actions(_Out(), ("action.arm", "action.gripper"))
    assert actions.shape == (1, 4, 8)
    # Reversing the declared modality must reverse the control-vector components.
    assert np.all(actions[..., :7] == 0.0)
    assert np.all(actions[..., 7] == 9.0)
    reverse = Gr00tPolicyBody._extract_actions(_Out(), ("action.gripper", "action.arm"))
    assert np.all(reverse[..., :1] == 9.0)
    assert np.all(reverse[..., 1:] == 0.0)


@pytest.mark.L0
@pytest.mark.CPU
def test_extract_actions_rejects_missing_output() -> None:
    class _Empty:
        multimodal_output = {}

    with pytest.raises(RuntimeError, match="multimodal_output"):
        Gr00tPolicyBody._extract_actions(_Empty())


@pytest.mark.L0
@pytest.mark.CPU
def test_split_actions_truncates_to_execution_horizon() -> None:
    body = _body(execution_horizon=4, training_prefix=16, model_action_horizon=16)
    out = body._split_actions(np.arange(2 * 16 * 3, dtype=np.float32).reshape(2, 16, 3), 2)
    assert len(out) == 2
    assert out[0]["action"].shape == (4, 3)
    assert np.array_equal(out[1]["action"], np.arange(2 * 16 * 3, dtype=np.float32).reshape(2, 16, 3)[1, :4])


@pytest.mark.L0
@pytest.mark.CPU
def test_split_actions_rejects_short_horizon_and_wrong_batch() -> None:
    body = _body(execution_horizon=8, training_prefix=16, model_action_horizon=16)
    with pytest.raises(RuntimeError, match="< execution_horizon"):
        body._split_actions(np.zeros((1, 4, 3), dtype=np.float32), 1)
    with pytest.raises(RuntimeError, match="action chunks"):
        body._split_actions(np.zeros((1, 16, 3), dtype=np.float32), 2)


@pytest.mark.L0
@pytest.mark.CPU
def test_native_latent_horizon_and_processor_action_horizon_are_distinct() -> None:
    body = _body(model_action_horizon=40, training_prefix=16, execution_horizon=16)
    body._embodiment = type("Tag", (), {"name": body.config.embodiment_tag})()
    modality = {
        "video_keys": list(body.config.camera_map),
        "state_keys": list(body.config.state_map),
        "language_key": body.config.language_key,
        "embodiment_tag": body.config.embodiment_tag,
        "native_action_horizon": 40,
        "action_horizon": 16,
    }
    body._verify_against_checkpoint(modality)
    with pytest.raises(ValueError, match="native action head"):
        body._verify_against_checkpoint({**modality, "native_action_horizon": 16})
    with pytest.raises(ValueError, match="processor action horizon"):
        body._verify_against_checkpoint({**modality, "action_horizon": 8})


@pytest.mark.L0
@pytest.mark.CPU
def test_weight_version_gate_rejects_a_mismatch() -> None:
    """Silently serving a stale version is how off-policy data enters a training batch."""
    body = _body()
    body._require_weight_version(None)  # unspecified: caller opts out
    body._require_weight_version(0)
    with pytest.raises(RuntimeError, match="weight_version=3"):
        body._require_weight_version(3)


@pytest.mark.L0
@pytest.mark.CPU
def test_body_module_defers_its_heavy_imports() -> None:
    """Importing the body must not pull vLLM / vLLM-Omni / Isaac-GR00T (the driver imports it)."""
    code = (
        "import sys;"
        "import cosmos_framework.posttrain.model.gr00t.serving.body;"
        "heavy=sorted(k for k in sys.modules if k.split('.')[0] in ('vllm','vllm_omni','gr00t'));"
        "assert not heavy, heavy"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "raw",
    [
        {"a": 1},  # flat
        [{"a": 1}],  # per-worker
        [[{"a": 1}]],  # per-stage then per-worker -- what vLLM-Omni actually returns
        [[[{"a": 1}]]],  # deeper, for safety
    ],
)
def test_worker_rpc_unwraps_however_deeply_collective_rpc_nests(raw) -> None:
    """`collective_rpc` aggregates per stage AND per worker; descend until a mapping appears."""
    body = _body()
    body._omni = type("O", (), {"engine": type("E", (), {"collective_rpc": staticmethod(lambda m, **_kw: raw)})()})()
    assert body._worker_rpc("describe_modality") == {"a": 1}


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("raw, expected", [([], "no worker results"), ([["not-a-mapping"]], "expected a mapping")])
def test_worker_rpc_reports_a_useless_reply_plainly(raw, expected: str) -> None:
    body = _body()
    body._omni = type("O", (), {"engine": type("E", (), {"collective_rpc": staticmethod(lambda m, **_kw: raw)})()})()
    with pytest.raises(RuntimeError, match=expected):
        body._worker_rpc("describe_modality")


@pytest.mark.L0
@pytest.mark.CPU
def test_startup_refuses_a_worker_that_built_the_stock_pipeline() -> None:
    """A stock-pipeline fallback is invisible from the output (seeding silently gone); fail at startup."""
    body = _body_reporting(
        {
            "class": "Gr00tN1d7Pipeline",
            "module": "vllm_omni.diffusion.models.gr00t.pipeline_gr00t",
            "is_rl_pipeline": False,
        }
    )
    with pytest.raises(RuntimeError) as excinfo:
        body._verify_rl_pipeline()
    message = str(excinfo.value)
    # The message must name what was actually built, or it sends the reader hunting.
    assert "vllm_omni.diffusion.models.gr00t.pipeline_gr00t.Gr00tN1d7Pipeline" in message
    assert "Gr00tN1d7RLPipeline" in message


@pytest.mark.L0
@pytest.mark.CPU
def test_startup_accepts_the_rl_pipeline_and_returns_what_the_worker_reported() -> None:
    assert _body_reporting(_RL_REPORT)._verify_rl_pipeline() == _RL_REPORT


@pytest.mark.L0
@pytest.mark.CPU
def test_startup_refuses_a_reply_that_omits_the_flag() -> None:
    """An older worker extension without `is_rl_pipeline` must not read as a pass."""
    body = _body_reporting({"class": "Gr00tN1d7Pipeline", "module": "somewhere"})
    with pytest.raises(RuntimeError, match="not Gr00tN1d7RLPipeline"):
        body._verify_rl_pipeline()


@pytest.mark.L0
@pytest.mark.CPU
def test_startup_refuses_the_rl_pipeline_without_the_reference_image_transform() -> None:
    """A correct pipeline class is not sufficient: the stock transform crops to a fixed size."""
    body = _body_reporting(
        {
            "class": "Gr00tN1d7RLPipeline",
            "module": "cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline",
            "is_rl_pipeline": True,
            "image_transform": None,
        }
    )
    with pytest.raises(RuntimeError, match="reference eval image transform"):
        body._verify_rl_pipeline()


@pytest.mark.L0
@pytest.mark.CPU
def test_a_stock_pipeline_reports_the_class_not_the_missing_transform() -> None:
    """Both guards would fire; the pipeline class is the cause and must be the message."""
    body = _body_reporting(
        {"class": "Gr00tN1d7Pipeline", "module": "vllm_omni", "is_rl_pipeline": False, "image_transform": None}
    )
    with pytest.raises(RuntimeError, match="not Gr00tN1d7RLPipeline"):
        body._verify_rl_pipeline()


@pytest.mark.L0
@pytest.mark.CPU
def test_startup_refuses_an_action_head_on_a_different_step_count() -> None:
    """The recipe's `denoising_steps` must be what the live Euler loop reads, not merely what was sent."""
    body = _body_reporting({**_RL_REPORT, "num_inference_timesteps": 8})
    with pytest.raises(RuntimeError, match="num_inference_timesteps=8.*denoising_steps=4"):
        body._verify_rl_pipeline()


@pytest.mark.L0
@pytest.mark.CPU
def test_startup_refuses_a_reply_without_the_step_count() -> None:
    """An older worker extension that does not report the step count must not read as a pass."""
    described = {k: v for k, v in _RL_REPORT.items() if k != "num_inference_timesteps"}
    with pytest.raises(RuntimeError, match="num_inference_timesteps=None"):
        _body_reporting(described)._verify_rl_pipeline()


@pytest.mark.L0
@pytest.mark.CPU
def test_startup_refuses_a_worker_that_loaded_a_different_artifact() -> None:
    """`revision` is honoured only if the engine served the path the body resolved."""
    body = _body_reporting({**_RL_REPORT, "model_path": "nvidia/GR00T-N1.7-DROID"})
    with pytest.raises(RuntimeError, match="not serving the pinned artifact"):
        body._verify_rl_pipeline()


@pytest.mark.L0
@pytest.mark.CPU
def test_startup_accepts_the_resolved_directory_under_another_spelling(tmp_path) -> None:
    """The worker may report the snapshot directory with a different spelling of the same path."""
    body = _body_reporting({**_RL_REPORT, "model_path": str(tmp_path) + "/"})
    body._model_path = str(tmp_path)
    assert body._verify_rl_pipeline()["model_path"] == str(tmp_path) + "/"
