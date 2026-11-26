# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared fixtures for the native action (Alpamayo-R1, GR00T N1.7 and family artifact) tests."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from cosmos_framework.posttrain.checkpoint.adapter import save_adapter_artifact
from cosmos_framework.posttrain.checkpoint.family_dense import export_family_dense
from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
from cosmos_framework.posttrain.model.alpamayo.sft import PHYSICALAI_CAMERAS
from cosmos_framework.posttrain.model.quantization.family import FamilyBackboneConfig, install_family_backbone

REPO_ROOT = Path(__file__).resolve().parents[4]
SUBPROCESS_TIMEOUT_SECONDS = 300
IMAGE_TOKEN_ID = 120
VISION_START_TOKEN_ID = 118
VISION_END_TOKEN_ID = 119
BUILD = {"code_revision": "1" * 40, "wheel_sha256": "2" * 64}
NATIVE_ACTION_TRANSFORMERS = "4.57.6"


def require_native_action(module: str, *, transformers: bool = False):
    """Import an optional native-action package (docker/posttrain/native_action runtime).

    Skips when it is unavailable, or fails instead when ``COSMOS_REQUIRE_NATIVE_ACTION=1`` so that a
    native-action environment enforces this coverage. ``transformers=True`` also requires the
    runtime's pinned Transformers version.
    """
    try:
        result = importlib.import_module(module)
        import transformers as library

        if transformers and library.__version__ != NATIVE_ACTION_TRANSFORMERS:
            raise ImportError(f"requires transformers=={NATIVE_ACTION_TRANSFORMERS}, found {library.__version__}")
    except ImportError as error:
        reason = f"native action test requires the optional {module} runtime: {error}"
        if os.environ.get("COSMOS_REQUIRE_NATIVE_ACTION") == "1":
            pytest.fail(f"{reason} (COSMOS_REQUIRE_NATIVE_ACTION=1)", pytrace=False)
        pytest.skip(reason, allow_module_level=True)
    return result


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_state_equal(a, b)
    else:
        assert left == right


def physicalai_sample():
    """A PhysicalAI four-camera observation row with placeholder media paths."""
    return {
        "media": {camera: [f"{camera}-{i}" for i in range(4)] for camera in PHYSICALAI_CAMERAS},
        "observation": {
            "camera_order": list(PHYSICALAI_CAMERAS),
            "camera_indices": [0, 1, 2, 6],
            "frame_timestamps_us": [[4800000, 4900000, 5000000, 5100000]] * 4,
            "t0_us": 5100000,
            "coordinate_frame": "vehicle_at_t0",
            "trajectory_dt_seconds": 0.1,
            "calibration": {
                "camera_intrinsics": {
                    camera: {
                        "width": 4,
                        "height": 2,
                        "cx": 1,
                        "cy": 1,
                        **{f"{side}_poly_{i}": float(i == 1) for side in ("bw", "fw") for i in range(5)},
                    }
                    for camera in PHYSICALAI_CAMERAS
                },
                "sensor_extrinsics": {
                    camera: {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
                    for camera in PHYSICALAI_CAMERAS
                },
            },
        },
    }


def tiny_qwen3_vl_config():
    from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 512,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=IMAGE_TOKEN_ID,
        video_token_id=121,
        vision_start_token_id=VISION_START_TOKEN_ID,
        vision_end_token_id=VISION_END_TOKEN_ID,
        tie_word_embeddings=False,
    )


def tiny_qwen3_vl_batch():
    return {
        "input_ids": torch.tensor(  # [B,T]
            [[1, VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, VISION_END_TOKEN_ID, 2]],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),  # [B,T]
        "pixel_values": torch.randn((4, 24), dtype=torch.float32),  # [N_patch,D_patch]
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),  # [N_image,3]
    }


def alpamayo_source(tmp_path):
    """Write a tiny public-architecture Alpamayo base and processor under ``tmp_path``; return it in BF16."""
    from alpamayo_r1.config import AlpamayoR1Config
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models
    from transformers import (
        Qwen2TokenizerFast,
        Qwen2VLImageProcessorFast,
        Qwen3VLConfig,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    from cosmos_framework.posttrain.model.alpamayo.policy import AlpamayoSFTNet

    processor = tmp_path / "processor"
    processor.mkdir()
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
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    media = dict(min_pixels=16, max_pixels=64, patch_size=2, temporal_patch_size=2, merge_size=2)
    Qwen3VLProcessor(
        tokenizer=tokenizer,
        image_processor=Qwen2VLImageProcessorFast(**media),
        video_processor=Qwen3VLVideoProcessor(**media),
    ).save_pretrained(processor)
    qwen = Qwen3VLConfig(**tiny_qwen3_vl_config().to_dict())
    # The public 8B processor declares a nested BF16 text dtype. Transformers
    # honors it even inside an outer from_pretrained(dtype=float32) call.
    qwen.text_config.dtype = torch.bfloat16
    qwen.text_config.rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 2]}
    qwen.save_pretrained(processor)
    cfg = AlpamayoR1Config(
        vlm_name_or_path=str(processor),
        traj_vocab_size=64,
        add_special_tokens=True,
        attn_implementation="sdpa",
        model_dtype="float32",
        min_pixels=16,
        max_pixels=64,
        action_space_cfg={"_target_": "alpamayo_r1.action_space.UnicycleAccelCurvatureActionSpace", "n_waypoints": 4},
        diffusion_cfg={"_target_": "alpamayo_r1.diffusion.flow_matching.FlowMatching"},
        action_in_proj_cfg={
            "_target_": "alpamayo_r1.models.action_in_proj.PerWaypointActionInProjV2",
            "hidden_size": 32,
            "num_enc_layers": 2,
            "num_fourier_feats": 4,
        },
        action_out_proj_cfg={"_target_": "torch.nn.Linear"},
        expert_cfg={"hidden_size": 32, "intermediate_size": 64, "dtype": "float32"},
    )
    torch.manual_seed(13)
    model = AlpamayoSFTNet(cfg)
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text(json.dumps(cfg.to_dict()))
    save_file({n: v.contiguous() for n, v in model.state_dict().items()}, str(base / "model.safetensors"))
    return model.bfloat16()


class TaskNet(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(39)
            self.backbone = nn.Linear(3, 4, bias=False, device=device, dtype=torch.bfloat16)
            self.head = nn.Linear(4, 2, device=device, dtype=torch.float32)
        self.register_buffer("action_scale", torch.tensor([0.2, 0.3], dtype=torch.float32), persistent=False)
        self.register_buffer("frame_id", torch.tensor([17], dtype=torch.int64))

    def forward(self, x):
        return self.head(self.backbone(x.to(self.backbone.weight.dtype)).float()) * self.action_scale


class AliasedTaskNet(TaskNet):
    def __init__(self, device="cpu", *, tied=True):
        super().__init__(device)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(53)
            self.embedding = nn.Embedding(5, 3, device=device, dtype=torch.bfloat16)
            self.output = nn.Linear(3, 5, bias=False, device=device, dtype=torch.bfloat16)
        if tied:
            self.output.weight = self.embedding.weight

    def forward(self, x):
        return super().forward(x + self.embedding(torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)))


def prepare_family_artifact(tmp_path, *, mode="bf16_lora", net=None, dropout=0.0):
    net = TaskNet() if net is None else net
    state = install_family_backbone(
        net,
        config=FamilyBackboneConfig(mode=mode, rank=2, alpha=4, seed=7, dropout=dropout),
        targets={"backbone": (4, 3)},
        auxiliary_parameters=("head.weight", "head.bias"),
        source_identity={"repository": "fixture/native-action", "revision": "a" * 40},
        task_identity={"family": "fixture", "action_schema": "two-scaled-axes", "horizon": 1},
        device=torch.device("cpu"),
    )
    state.codec.bind_processor_identity({"format": "fixture", "pixels": [4, 4]})
    state.codec.bind_training_identity({"steps": 3, "seed": 9})
    optimizer = torch.optim.AdamW(state.codec.parameters.values(), lr=0.01)
    data = torch.tensor([[0.1, 0.2, -0.3], [-0.3, 0.1, 0.2]])
    target = torch.tensor([[0.01, -0.02], [-0.02, 0.01]])
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        (net(data) - target).square().mean().backward()
        optimizer.step()
    source = save_adapter_artifact(
        str(tmp_path / "checkpoints/iter_000003"),
        state.codec.parameters,
        state.codec.metadata(optimizer),
        manager=CheckpointManager(checkpoint_format="consolidated"),
        publish=False,
    )
    return net, state, source, data


def export_family_artifact(tmp_path, *, mode="bf16_lora", dtype="float32", max_bytes=60, sidecars=None):
    net, state, source, data = prepare_family_artifact(tmp_path, mode=mode)
    output = tmp_path / "dense"
    manifest = export_family_dense(
        net,
        checkpoint=source,
        output=str(output),
        merged_backbone_dtype=dtype,
        max_shard_size_bytes=max_bytes,
        exporter_build=BUILD,
        sidecars=sidecars,
    )
    return net, state, source, data, output, manifest


class HeadTaskNet(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(39)
            self.backbone = nn.Sequential(
                nn.Linear(3, 4, bias=False, device=device, dtype=torch.bfloat16),
                nn.Tanh(),
                nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16),
            )
            self.action_head = nn.Module()
            self.action_head.q = nn.Linear(4, 4, device=device, dtype=torch.float32)
            self.action_head.out = nn.Linear(4, 2, device=device, dtype=torch.float32)
            # A custom category-shaped parameter remains completely outside the allowlist.
            self.action_head.category = nn.Parameter(torch.randn(2, 3, 4, device=device, dtype=torch.float32))
            self.embedding = nn.Embedding(5, 3, device=device, dtype=torch.bfloat16)
            self.output = nn.Linear(3, 5, bias=False, device=device, dtype=torch.bfloat16)
            self.output.weight = self.embedding.weight
        self.register_buffer("scale", torch.tensor([0.2, 0.3], dtype=torch.float32), persistent=False)

    def forward(self, inputs):
        x = self.backbone(inputs.to(torch.bfloat16))
        return self.action_head.out(torch.tanh(self.action_head.q(x.float()))) * self.scale
