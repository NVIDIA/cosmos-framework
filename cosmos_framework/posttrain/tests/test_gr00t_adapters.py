# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real small Isaac GR00T/Qwen3-VL/DiT graphs; separate from checkpoint qualification."""

import copy
import json

import pytest
import torch

pytest.importorskip("gr00t")
import transformers

pytestmark = pytest.mark.skipif(transformers.__version__ != "4.57.6", reason="pinned robotics runtime")

from cosmos_framework.posttrain.checkpoint.adapters import save_adapter_artifact
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig as C
from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.model.adapters import assert_trainability
from cosmos_framework.posttrain.model.gr00t.sft import Gr00tSFTModelConfig
from cosmos_framework.posttrain.model.gr00t.sft_policy import Gr00tSFTPolicy, build_policy, construct, load_base


def source(tmp_path):
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models
    from transformers import (
        Qwen2TokenizerFast,
        Qwen2VLImageProcessorFast,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _dense_config

    backbone = tmp_path / "Qwen/Qwen3-VL-tiny"
    backbone.mkdir(parents=True)
    vocab = {f"token{i}": i for i in range(128)}
    for name, index in [
        ("<unk>", 0),
        ("<|vision_start|>", 118),
        ("<|vision_end|>", 119),
        ("<|image_pad|>", 120),
        ("<|video_pad|>", 121),
    ]:
        del vocab[f"token{index}"]
        vocab[name] = index
    tokenizer = Qwen2TokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")), unk_token="<unk>", pad_token="<unk>"
    )
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"]}
    )
    tokenizer.chat_template = "{% for message in messages %}{% for item in message['content'] %}{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|> {% else %}{{ item['text'] }} {% endif %}{% endfor %}{% endfor %}"
    media = dict(min_pixels=16, max_pixels=64, patch_size=2, temporal_patch_size=2, merge_size=2)
    Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessorFast(**media),
        video_processor=Qwen3VLVideoProcessor(**media),
    ).save_pretrained(backbone)
    cfg = Qwen3VLConfig(**_dense_config().to_dict())
    cfg.text_config.rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 2]}
    torch.manual_seed(123)
    Qwen3VLForConditionalGeneration(cfg).save_pretrained(backbone)
    root = tmp_path / "base"
    root.mkdir()
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

    native = Gr00tN1d7Config(
        model_name=str(backbone),
        select_layer=2,
        backbone_embedding_dim=32,
        max_state_dim=8,
        max_action_dim=7,
        action_horizon=4,
        hidden_size=32,
        input_embedding_dim=32,
        max_num_embodiments=3,
        max_seq_len=16,
        use_flash_attention=False,
        state_dropout_prob=0.2,
        use_alternate_vl_dit=True,
        vl_self_attention_cfg={
            "num_layers": 1,
            "num_attention_heads": 4,
            "attention_head_dim": 8,
            "dropout": 0.1,
            "final_dropout": True,
            "positional_embeddings": None,
        },
        diffusion_model_cfg={
            "num_layers": 4,
            "num_attention_heads": 4,
            "attention_head_dim": 8,
            "output_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.1,
            "final_dropout": True,
            "interleave_self_attention": True,
            "positional_embeddings": None,
        },
    )
    (root / "config.json").write_text(json.dumps(native.to_dict()))
    keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    modalities = {
        "libero_sim": {
            "video": {"delta_indices": [0], "modality_keys": ["image", "wrist_image"]},
            "state": {"delta_indices": [0], "modality_keys": keys},
            "action": {
                "delta_indices": list(range(4)),
                "modality_keys": keys,
                "action_configs": [{"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"}] * 7,
            },
            "language": {"delta_indices": [0], "modality_keys": ["annotation.human.action.task_description"]},
        }
    }
    processor_config = {
        "processor_class": "Gr00tN1d7Processor",
        "processor_kwargs": {
            "modality_configs": modalities,
            "model_name": str(backbone),
            "max_state_dim": 8,
            "max_action_dim": 7,
            "max_action_horizon": 4,
            "image_crop_size": [8, 8],
            "image_target_size": [8, 8],
            "use_albumentations": False,
            "formalize_language": False,
            "use_percentiles": False,
            "state_dropout_prob": 0.2,
        },
    }
    statistics = {
        "libero_sim": {
            modality: {
                key: {
                    stat: [value] * (2 if modality == "state" and key == "gripper" else 1)
                    for stat, value in {"min": -1.0, "max": 1.0, "mean": 0.0, "std": 1.0}.items()
                }
                for key in keys
            }
            for modality in ("state", "action")
        }
    }
    (root / "processor_config.json").write_text(json.dumps(processor_config))
    (root / "statistics.json").write_text(json.dumps(statistics))
    (root / "embodiment_id.json").write_text('{"libero_sim": 2}')
    config = Gr00tSFTModelConfig(str(backbone), "none")
    model = construct(str(root), config)
    save_file({n: v.contiguous().clone() for n, v in model.state_dict().items()}, str(root / "model.safetensors"))
    return str(root), config


def adapters(method, rs=False):
    settings = dict(method=method, use_rslora=rs, rank=2, alpha=4)
    return AdapterConfig(
        components=[
            C(
                component="language",
                targets=[
                    "backbone.model.model.language_model.layers.*.self_attn.q_proj",
                    "backbone.model.model.language_model.layers.*.self_attn.v_proj",
                ],
                **settings,
            ),
            C(
                component="projector",
                targets=[
                    "backbone.model.model.visual.merger.linear_fc2",
                    "backbone.model.model.visual.deepstack_merger_list.*.linear_fc2",
                ],
                **settings,
            ),
            C(component="vision", targets=["backbone.model.model.visual.blocks.*.attn.qkv"], layers=[0], **settings),
            C(
                component="action",
                targets=[
                    "action_head.model.transformer_blocks.*.attn1.to_q",
                    "action_head.model.transformer_blocks.*.attn1.to_v",
                ],
                **settings,
            ),
        ]
    )


def batch():
    from cosmos_framework.posttrain.tests.test_model_qwen3_vl_gpu import _batch

    torch.manual_seed(71)
    image = _batch()
    image.update(
        input_ids=torch.tensor([[1, 118, 120, 120, 119, 2]]),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        pixel_values=torch.randn(8, 24),
        image_grid_thw=torch.tensor([[1, 2, 4]]),
    )
    inputs = {
        **image,
        "state": torch.randn(1, 1, 8),
        "action": torch.randn(1, 4, 7),
        "action_mask": torch.ones(1, 4, 7),
        "embodiment_id": torch.tensor([2]),
    }
    return dict(inputs=inputs, actions=inputs["action"], action_mask=inputs["action_mask"])


def prediction(policy, data):
    policy.net.eval()
    with torch.no_grad():
        return policy.flow_matching_loss({**data, "control_seed": 17}).prediction.detach()


@pytest.mark.parametrize("method,rs", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
@pytest.mark.parametrize("ac", ["none", "full"])
def test_native_components_update_and_reload(tmp_path, method, rs, ac):
    base, config = source(tmp_path)
    config.activation_checkpointing = ac
    adapter_config = adapters(method, rs)
    policy = build_policy(base, config, adapters=adapter_config, device="cpu")
    data = batch()
    reference = Gr00tSFTPolicy(load_base(base, config)[0])
    torch.testing.assert_close(prediction(policy, data), prediction(reference, data), atol=2e-5, rtol=2e-5)
    before = {n: p.clone() for n, p in policy.net.named_parameters()}
    initial = {n: v.clone() for n, v in policy.net.named_buffers() if n.endswith(("initial_A", "initial_B"))}
    optimizer = torch.optim.AdamW([p for p in policy.net.parameters() if p.requires_grad], lr=0.003, weight_decay=0)
    policy.net.train()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = policy.flow_matching_loss(data)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in policy.net.parameters() if p.requires_grad
        )
        optimizer.step()
    assert_trainability(policy.net, optimizer)
    for n, p in policy.net.named_parameters():
        assert (not torch.equal(p, before[n])) if p.requires_grad else torch.equal(p, before[n]), n
    assert all(torch.equal(v, dict(policy.net.named_buffers())[n]) for n, v in initial.items())
    expected = prediction(policy, data)
    out = str(tmp_path / "portable")
    save_adapter_artifact(policy.net, out)
    fresh = copy.deepcopy(adapter_config)
    fresh.warm_start = out
    loaded = build_policy(base, config, adapters=fresh, device="cpu")
    torch.testing.assert_close(prediction(loaded, data), expected, atol=2e-5, rtol=2e-5)


def test_checkpoint_constructor_matches_official_graph(tmp_path):
    from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

    base, config = source(tmp_path)
    net, _ = load_base(base, config)
    official = Gr00tN1d7(
        Gr00tN1d7Config(**net.config.to_dict()), transformers_loading_kwargs={"local_files_only": True}
    ).float()
    assert dict(net.state_dict()).keys() == dict(official.state_dict()).keys()
    official.load_state_dict(net.state_dict(), strict=True)
    net.eval()
    official.eval()
    data = batch()
    with torch.random.fork_rng(devices=[]), torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        torch.manual_seed(17)
        expected = official(data["inputs"])["loss"]
        torch.manual_seed(17)
        actual = net(data["inputs"]).loss
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_reject_category_tensor_and_wrong_component(tmp_path):
    from cosmos_framework.posttrain.model.adapters import resolve_targets

    base, config = source(tmp_path)
    net, _ = load_base(base, config)
    with pytest.raises(ValueError, match="Linear|unsupported"):
        resolve_targets(
            net,
            AdapterConfig(
                components=[C(component="action", targets=["action_head.action_encoder.W1"], rank=2, alpha=4)]
            ),
            family="Gr00tN1d7",
        )
    with pytest.raises(ValueError, match="outside language"):
        resolve_targets(
            net,
            AdapterConfig(
                components=[
                    C(component="language", targets=["backbone.model.model.visual.merger.linear_fc2"], rank=2, alpha=4)
                ]
            ),
            family="Gr00tN1d7",
        )


@pytest.mark.parametrize("method,rs", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
def test_native_stochastic_checkpoint_next_update(tmp_path, monkeypatch, method, rs):
    from cosmos_framework.posttrain.algorithm.flow_bc import make_objective
    from cosmos_framework.posttrain.checkpoint.io import load_component
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer

    base, model_config = source(tmp_path)
    cfg = adapters(method, rs)
    data = batch()

    def trainer(policy):
        objective = make_objective(lambda rows: data, training_prefix=4, action_dim=7)
        objective.identity = {**objective.identity, "adapters": policy.net._posttrain_adapter_identity}
        optimizer = torch.optim.AdamW([p for p in policy.net.parameters() if p.requires_grad], lr=0.003, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        return LLMTrainer(
            policy,
            objective,
            optimizer,
            checkpoint_format="dcp",
            save_async=False,
            lr_scheduler=scheduler,
            require_gradients=True,
        )

    original = trainer(build_policy(base, model_config, adapters=cfg, device="cpu"))
    original.train([0])
    original.train([0])
    checkpoint = str(tmp_path / "checkpoint")
    original.save(checkpoint, data_state={"next_row": 2})
    original.finalize_checkpoints()
    expected_report = original.train([0])
    expected = copy.deepcopy(original.model.net.state_dict())
    expected_opt = copy.deepcopy(original.opt.state_dict())
    monkeypatch.setattr(torch, "svd_lowrank", lambda *a, **kw: pytest.fail("resume repeated PiSSA decomposition"))
    resumed = trainer(build_policy(base, model_config, adapters=cfg, adapter_resume=True, device="cpu"))
    assert resumed.load(checkpoint) == 2
    actual_report = resumed.train([0])
    assert actual_report["train/loss_avg"] == expected_report["train/loss_avg"]
    assert load_component(checkpoint, "data") == {"next_row": 2}
    assert resumed.lr_scheduler.state_dict() == original.lr_scheduler.state_dict()
    for name, value in resumed.model.net.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    for index, values in resumed.opt.state_dict()["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, expected_opt["state"][index][name], atol=0, rtol=0)
    assert_trainability(resumed.model.net, resumed.opt)


@pytest.mark.parametrize("method,rs", [("lora", False), ("lora", True), ("dora", False), ("pissa", False)])
def test_merged_file_preserves_live_model_and_reloads(tmp_path, method, rs):
    from cosmos_framework.posttrain.model.gr00t.adapter_export import export_merged, load_merged

    base, config = source(tmp_path)
    policy = build_policy(base, config, adapters=adapters(method, rs), device="cpu")
    with torch.no_grad():
        for p in policy.net.parameters():
            if p.requires_grad:
                p.add_(0.001)
    portable = str(tmp_path / "portable")
    save_adapter_artifact(policy.net, portable)
    before = {n: v.clone() for n, v in policy.net.state_dict().items()}
    export_merged(portable, base, str(tmp_path / "merged"), processor=config.backbone_path)
    assert all(torch.equal(v, policy.net.state_dict()[n]) for n, v in before.items())
    policy.autocast = False
    loaded = load_merged(str(tmp_path / "merged"))
    data = batch()
    torch.testing.assert_close(prediction(loaded, data), prediction(policy, data), atol=5e-5, rtol=5e-5)
    with pytest.raises(ValueError, match="merged artifacts require"):
        load_base(str(tmp_path / "merged"), config)
    (tmp_path / "merged/backbone/config.json").write_text("{}")
    with pytest.raises(ValueError, match="missing or changed"):
        load_merged(str(tmp_path / "merged"))
    (tmp_path / "base/statistics.json").write_text('{"changed": 1}')
    cfg = adapters(method, rs)
    cfg.warm_start = portable
    with pytest.raises(ValueError, match="compatibility mismatch"):
        build_policy(base, config, adapters=cfg, device="cpu")


def test_real_processor_camera_state_action_convention_and_padding(tmp_path):
    import numpy as np
    from PIL import Image

    from cosmos_framework.posttrain.model.gr00t.sft import Gr00tSFTPacker
    from cosmos_framework.posttrain.schema.trajectory import ObsBundle, Trajectory, Transition

    base, config = source(tmp_path)
    packer = Gr00tSFTPacker(base, config, training_prefix=4, action_dim=7)
    media = {}
    for camera, color in (("image", "red"), ("wrist_image", "green")):
        path = tmp_path / (camera + ".png")
        Image.new("RGB", (8, 8), color).save(path)
        media[camera] = str(path)
    action = np.zeros((4, 7), dtype=np.float32)
    action[:, -1] = [-1, 1, -1, 1]
    sample = {"observation": {"state": [0.0] * 8, "prompt": "token3"}, "actions": action.tolist(), "media": media}
    transition = Transition(
        obs=ObsBundle(representations={"flow_sample": sample}), algo_extra={"validation_weight": 0.0}
    )
    rows = [Trajectory(transitions=[transition])]
    batch = packer(rows)
    assert batch["inputs"]["pixel_values"].numel() > 0
    assert batch["inputs"]["image_grid_thw"].shape[0] == 2
    # Min/max [-1,1] means normalized values equal the raw converted values.
    torch.testing.assert_close(batch["actions"][0, :, -1], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert batch["action_mask"].sum() == 0
    assert torch.equal(packer(rows)["inputs"]["pixel_values"], batch["inputs"]["pixel_values"])
    policy = build_policy(base, config, adapters=adapters("lora"), device="cpu")
    assert torch.isfinite(prediction(policy, batch)).all()
    identity = packer.validation_semantics_identity()
    assert identity["gripper_to_native"] == "(1-environment_gripper)/2"
    with pytest.raises(ValueError, match="evaluation probe only"):
        policy.net.train()
        policy.flow_matching_loss(packer.controlled_probe(batch, 321))
