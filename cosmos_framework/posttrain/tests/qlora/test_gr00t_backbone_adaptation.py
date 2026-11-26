# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T backbone adaptation: configuration, inputs, dataset and lifecycle on CPU."""

from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from cosmos_framework.posttrain.config.gr00t_bc import Gr00tBackboneAdaptation
from cosmos_framework.posttrain.model.gr00t.action_sft import Gr00tActionDataset, Gr00tActionPacker, evaluate_actions
from cosmos_framework.posttrain.model.gr00t.adaptation import LANGUAGE_PROJECTIONS, resolve_language_targets
from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def architecture():
    model = nn.Module()
    model.config = SimpleNamespace(
        model_type="Gr00tN1d7",
        select_layer=16,
        backbone_embedding_dim=2048,
        action_horizon=40,
        max_action_dim=132,
        num_inference_timesteps=4,
    )
    model.backbone = nn.Module()
    model.backbone.model = nn.Module()
    language = model.backbone.model.language_model = nn.Module()
    language.layers = nn.ModuleList()
    with torch.device("meta"):
        for _ in range(16):
            layer = nn.Module()
            layer.self_attn, layer.mlp = nn.Module(), nn.Module()
            for relative in LANGUAGE_PROJECTIONS:
                parent, name = relative.split(".")
                setattr(getattr(layer, parent), name, nn.Linear(2048, 128, bias=False))
            language.layers.append(layer)
        model.backbone.model.visual = nn.Linear(2048, 2048, bias=False)
        model.action_head = nn.Linear(2048, 2048, bias=False)
    return model


def test_exact_language_ownership_excludes_vision_and_action():
    targets = resolve_language_targets(architecture())
    assert len(targets) == 112
    assert all(name.startswith("backbone.model.language_model.layers.") for name in targets)
    assert set(targets.values()) == {(128, 2048)}


@pytest.mark.parametrize("kind", ["alias", "bias", "custom", "profile"])
def test_incompatible_architectures_fail_before_install(kind):
    model = architecture()
    parent = model.backbone.model.language_model.layers[0].self_attn
    if kind == "alias":
        model.action_head.proj = parent.q_proj
    elif kind == "bias":
        parent.q_proj = nn.Linear(2048, 128, device="meta")
    elif kind == "custom":

        class Custom(nn.Linear):
            pass

        parent.q_proj = Custom(2048, 128, bias=False, device="meta")
    else:
        model.config.action_horizon = 16
    with pytest.raises(ValueError):
        resolve_language_targets(model)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "nf4"},
        {"mode": "nf4_frozen", "train_action_head": False},
        {"rank": 0},
        {"alpha": float("nan")},
        {"dropout": 1.0},
    ],
)
def test_invalid_trainability_and_adapter_configuration(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Gr00tBackboneAdaptation(**kwargs)


def test_legacy_identity_stays_without_optional_adaptation():
    from cosmos_framework.posttrain.configs.gr00t_n1d7_droid import make_model_config

    config = make_model_config()
    assert "backbone_adaptation" not in config.identity()
    config.backbone_adaptation = Gr00tBackboneAdaptation()
    assert config.identity()["backbone_adaptation"]["mode"] == "nf4_lora"


def trajectory(tmp_path, identifier="sample", group="episode"):
    image = tmp_path / "image.png"
    Image.new("RGB", (20, 18), (10, 20, 30)).save(image)
    row = {
        "id": identifier,
        "episode_id": group,
        "observation": {"state": [0, 1, 2, 3, 4, 5, 0.2, 0.3], "prompt": "open the drawer"},
        "actions": [[0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1]],
        "media": {"image": str(image), "wrist_image": str(image)},
    }
    return row, Trajectory(
        id=identifier,
        status=Status.COMPLETED,
        transitions=[Transition(obs=ObsBundle(representations={"flow_sample": row}), action=row["actions"])],
    )


def test_native_packer_preserves_camera_state_and_gripper_semantics(monkeypatch, tmp_path):
    import sys

    tags = ModuleType("gr00t.data.embodiment_tags")
    tags.EmbodimentTag = SimpleNamespace(resolve=lambda value: value)
    types = ModuleType("gr00t.data.types")
    types.MessageType = SimpleNamespace(EPISODE_STEP=SimpleNamespace(value="episode_step"))
    types.VLAStepData = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, tags.__name__, tags)
    monkeypatch.setitem(sys.modules, types.__name__, types)
    captured = []

    class Processor:
        def __call__(self, messages):
            native = messages[0]["content"]
            captured.append(native)
            action = torch.zeros(40, 132)
            for index, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw", "gripper")):
                action[:2, index] = torch.tensor(native.actions[key][:, 0])
            return {"action": action, "vlm_content": {"conversation": "removed"}}

        def collator(self, items):
            assert all("conversation" not in item["vlm_content"] for item in items)
            return {"inputs": {key: torch.stack([item[key] for item in items]) for key in ("action", "action_mask")}}

    packer = object.__new__(Gr00tActionPacker)
    packer.training_prefix, packer.action_dim, packer.processor = 2, 7, Processor()
    _, example = trajectory(tmp_path)
    inputs = packer([example])
    assert inputs["action_mask"].sum() == 14
    assert inputs["action_mask"][:, 2:].sum() == 0
    assert inputs["action_mask"][:, :, 7:].sum() == 0
    assert inputs["action"][0, :2, 6].tolist() == [1, 0]
    assert captured[0].states["gripper"].shape == (1, 2)
    assert captured[0].states["gripper"][0].tolist() == pytest.approx([0.2, 0.3])
    assert captured[0].images["image"][0].shape == (18, 20, 3)
    assert captured[0].text == "open the drawer"


def test_action_sampling_does_not_receive_ground_truth_and_reports_decoded_units(tmp_path):
    _, example = trajectory(tmp_path)
    keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    expected = torch.zeros(1, 40, 132)
    mask = torch.zeros_like(expected)
    mask[:, :2, :7] = 1
    inputs = {"action": expected, "action_mask": mask, "state": torch.zeros(1, 132)}

    class Packer:
        processor = SimpleNamespace(decode_action=lambda action, tag: {key: np.zeros((1, 16, 1)) for key in keys})

        def __call__(self, batch):
            return inputs

    def sample(actual):
        assert "action" not in actual and "action_mask" not in actual
        return {"action_pred": expected.clone()}

    policy = SimpleNamespace(
        packer=Packer(),
        compute_ctx=nullcontext(),
        gr00t=SimpleNamespace(get_action=sample),
        config=SimpleNamespace(training_prefix=2, resolve_embodiment_tag=lambda: "libero_sim"),
        evaluation_backbone_reuse=None,
    )
    from cosmos_framework.posttrain.model.gr00t.policy import Gr00tN1d7TrainPolicy

    policy.predict_actions = lambda value: Gr00tN1d7TrainPolicy.predict_actions(policy, value)
    metrics = evaluate_actions(policy, [example], None)
    assert metrics["normalized_action_mse"] == {"sum": 0, "count": 14}
    assert metrics["action_mae_gripper_environment_units"] == {"sum": 2, "count": 2}


@pytest.mark.parametrize("reuse_active", [False, True])
@pytest.mark.parametrize("key", ["action", "action_mask"])
def test_prediction_refuses_action_conditioning_with_or_without_backbone_reuse(reuse_active, key):
    from cosmos_framework.posttrain.model.gr00t.policy import Gr00tN1d7TrainPolicy

    policy = SimpleNamespace(
        compute_ctx=nullcontext(),
        gr00t=SimpleNamespace(get_action=lambda _: pytest.fail("must refuse before sampling")),
        evaluation_backbone_reuse=SimpleNamespace(active=reuse_active),
    )
    with pytest.raises(ValueError, match="action conditioning"):
        Gr00tN1d7TrainPolicy.predict_actions(policy, {key: torch.zeros(1), "state": torch.zeros(1, 132)})


def test_rollout_samples_through_the_policy_prediction_path():
    from cosmos_framework.posttrain.model.gr00t.rollout import ACTION_KEYS, predict_processed

    inputs, calls = {"state": torch.zeros(1, 132)}, []

    def predict_actions(value):
        calls.append(value)
        assert not torch.is_grad_enabled()
        return {"action_pred": torch.zeros(1, 40, 132)}

    processor = SimpleNamespace(
        decode_action=lambda action, tag, states: {key: np.zeros((1, 16, 1)) for key in ACTION_KEYS}
    )
    policy = SimpleNamespace(
        predict_actions=predict_actions,
        packer=SimpleNamespace(processor=processor),
        config=SimpleNamespace(execution_horizon=8, resolve_embodiment_tag=lambda: "libero_sim"),
    )
    _, action = predict_processed(policy, inputs)
    assert calls == [inputs] and action.shape == (1, 16, 7)


def test_dataset_content_identity_and_episode_namespace(tmp_path):
    import json

    row, _ = trajectory(tmp_path)
    source = tmp_path / "data.jsonl"
    source.write_text(json.dumps(row) + "\n")
    dataset = Gr00tActionDataset(source)
    assert dataset.groups == {"episode"}
    assert dataset[0].transitions[0].obs.representations["flow_sample"]["actions"] == row["actions"]
    original = dataset.identity
    row["observation"]["prompt"] = "close the drawer"
    source.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="changed"):
        dataset[0]
    assert Gr00tActionDataset(source).identity != original


def test_dataset_rows_split_only_on_newline(tmp_path):
    import json

    first, _ = trajectory(tmp_path)
    second = json.loads(json.dumps(first))
    first["observation"]["prompt"] = "pick\u2028up the bowl\u0085now"
    second["id"], second["episode_id"] = "second", "other-episode"
    source = tmp_path / "data.jsonl"
    source.write_bytes(b"\n".join(json.dumps(row, ensure_ascii=False).encode() for row in (first, second)) + b"\r\n\n")
    dataset = Gr00tActionDataset(source)
    assert len(dataset) == 2
    assert dataset[0].transitions[0].obs.representations["flow_sample"]["observation"]["prompt"] == (
        "pick\u2028up the bowl\u0085now"
    )
    assert dataset[1].id == "second"


def test_native_recipe_loads_and_precise_modes_remain_explicit():
    from cosmos_framework.posttrain.entrypoints.action_common import validate_config
    from cosmos_framework.utils.config import load_config

    config = load_config("cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py", ["--", "rl.max_steps=3"])
    assert validate_config(config).max_steps == 3
    assert config.model.backbone_adaptation.mode == "nf4_lora"
    assert config.rl.optimizer.impl == "fused"
    assert config.model.training_prefix == 10


def test_fp32_task_inputs_preserve_native_noise_schedule_precision():
    from cosmos_framework.posttrain.model.gr00t.adaptation import prepare_task_inputs

    inputs = {
        "action": torch.tensor([1.0001]),
        "state": torch.tensor([0.123456]),
        "action_mask": torch.ones(1),
        "input_ids": torch.tensor([7]),
    }

    def native_prepare(value):
        moved = {
            key: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor for key, tensor in value.items()
        }
        return dict(moved), dict(moved)

    backbone, action = prepare_task_inputs(SimpleNamespace(device=torch.device("cpu")), inputs, native_prepare)
    assert backbone["action"].dtype == torch.bfloat16
    for key in ("state", "action", "action_mask"):
        assert action[key].dtype == torch.float32
        torch.testing.assert_close(action[key], inputs[key], rtol=0, atol=0)
    assert action["input_ids"].dtype == torch.int64


@pytest.mark.parametrize(
    "field,value",
    [
        ("denoising_steps", 2),
        ("model_action_horizon", 16),
        ("camera_map", {"image": "different"}),
        ("state_map", {"x": "different"}),
        ("backbone_buffer_dtype", "bfloat16"),
    ],
)
def test_selected_native_task_config_does_not_silently_ignore_overrides(field, value):
    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_qlora import make_config
    from cosmos_framework.posttrain.model.gr00t.adaptation import validate_task_config

    config = make_config().model
    validate_task_config(config)
    setattr(config, field, value)
    with pytest.raises(ValueError):
        validate_task_config(config)


def test_weight_reader_matches_source_and_hf_single_file_precedence(tmp_path):
    import json

    from safetensors.torch import save_file

    from cosmos_framework.posttrain.model.gr00t.adaptation import checkpoint_weight_map

    save_file({"actual": torch.ones(2)}, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"unbound": "wrong.safetensors"}}))
    assert checkpoint_weight_map(tmp_path) == {"actual": "model.safetensors"}
    (tmp_path / "model.safetensors").unlink()
    assert checkpoint_weight_map(tmp_path) == {"unbound": "wrong.safetensors"}
