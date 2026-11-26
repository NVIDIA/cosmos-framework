# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared models and builders for the native QLoRA tests."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.checkpoint import dense_adapter as dense
from cosmos_framework.posttrain.config.model import NativeLoRAConfig, QuantizedBaseTrainingConfig
from cosmos_framework.posttrain.model.checkpoint import build_adapter_checkpoint_codec
from cosmos_framework.posttrain.model.quantization import loader
from cosmos_framework.posttrain.model.quantization.family import FamilyBackboneConfig, install_family_backbone
from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
from cosmos_framework.posttrain.utils.rank_state import collect_rank_state
from cosmos_framework.utils.generator.lora import LoraInjectedLinear, NativeLoRAMixin

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
EXPORTER_BUILD = {"code_revision": "a" * 40, "wheel_sha256": "b" * 64}

# ---------------------------------------------------------------------------
# A tiny adapter model and LLMTrainer for checkpoint/resume tests.
# ---------------------------------------------------------------------------


class FrozenBase(nn.Linear):
    def state_dict(self, *args, **kwargs):
        raise AssertionError("full frozen-base state extraction is forbidden")


class TinyAdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = FrozenBase(4, 4, bias=False).requires_grad_(False)
        self.base.register_buffer("packed_state", torch.arange(8, dtype=torch.uint8))
        self.lora_A = nn.Linear(4, 2, bias=False)
        self.lora_B = nn.Linear(2, 4, bias=False)
        nn.init.zeros_(self.lora_B.weight)
        self._cosmos_adapter_parameters = {
            "language.q_proj.lora_A.weight": self.lora_A.weight,
            "language.q_proj.lora_B.weight": self.lora_B.weight,
        }
        self._cosmos_adapter_checkpoint_codec = build_adapter_checkpoint_codec(
            self,
            identity={"base": {"repo": "test/base", "revision": "a" * 40}, "quantizer": "test-only"},
        )
        self._cosmos_adapter_checkpoint_codec.bind_training_identity({"schedule": "linear", "seed": 17})
        self._cosmos_adapter_checkpoint_codec.bind_processor_identity({"template": "controlled", "media": "native"})

    def _apply(self, *args, **kwargs):
        raise AssertionError("blanket movement/conversion is forbidden")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_B(self.lora_A(x))


def tiny_objective(model, batch):
    x = torch.randn(2, 4)
    loss = model(x).square().mean()
    return loss, {"loss": float(loss.detach())}


def make_trainer(seed: int = 17, **kwargs) -> LLMTrainer:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = TinyAdapterModel()
    optimizer = torch.optim.AdamW([model.lora_A.weight, model.lora_B.weight], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=10)
    return LLMTrainer(
        model,
        Objective(tiny_objective, reduction=Reduction.MEAN),
        optimizer,
        checkpoint_format="dcp",
        lr_scheduler=scheduler,
        **kwargs,
    )


def save_checkpoint(trainer: LLMTrainer, tmp_path: Path, *, step: int | None = None) -> Path:
    path = tmp_path / "checkpoints" / f"iter_{trainer._iter if step is None else step:06d}"
    trainer.save(str(path), data_state=collect_rank_state({"cursor": trainer._iter}))
    return path


def adapter_values(trainer: LLMTrainer) -> dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in trainer._adapter_checkpoint_codec().parameters.items()}


# ---------------------------------------------------------------------------
# A dense adapter model and its export (dense-export tests).
# ---------------------------------------------------------------------------


class TinyDenseAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = LoraInjectedLinear(
            nn.Linear(4, 3, dtype=torch.bfloat16),
            rank=2,
            alpha=4,
            adapter_dtype=torch.float32,
        )
        self.embedding = nn.Embedding(5, 4, dtype=torch.bfloat16)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_("lora_" in name)
        self._cosmos_adapter_parameters = {
            f"linear.{name}": parameter for name, parameter in self.linear.adapter_parameters().items()
        }
        frozen = {name: p.detach().clone() for name, p in self.named_parameters() if not p.requires_grad}

        def validate():
            for name, parameter in self.named_parameters():
                if name in frozen:
                    assert torch.equal(parameter, frozen[name])

        self._cosmos_validate_frozen_base = validate

    def state_dict(self, *args, **kwargs):
        raise AssertionError("full-model state extraction is forbidden")


def export_dense(net, source, checkpoint, output, **kw):
    return dense.export_dense_adapter(
        net,
        source=str(source),
        checkpoint=str(checkpoint),
        output=str(output),
        output_dtype=kw.pop("output_dtype", "bfloat16"),
        exporter_build=kw.pop("exporter_build", EXPORTER_BUILD),
        **kw,
    )


# ---------------------------------------------------------------------------
# The all-252-target native Qwen lifecycle on tiny matrices.
# ---------------------------------------------------------------------------


def make_lifecycle_fixture(tmp_path, monkeypatch, layers=(), *, real_cuda=False):
    """Run actual prepare/materialize/load/finalize on all252 names and a tiny source."""
    size = 128 if real_cuda else 2
    targets = {name: (size, size) for name in loader.canonical_targets()}
    monkeypatch.setattr(loader, "canonical_targets", lambda: targets)
    device = torch.device("cuda", torch.cuda.current_device()) if real_cuda else torch.device("cpu")
    monkeypatch.setattr(loader, "validate_runtime", lambda *a: device)
    from cosmos_framework.posttrain.model.quantization import source as source_module

    monkeypatch.setattr(source_module, "validate_adapter_source", lambda cfg: {"fixture": "bounded-tiny-source"})
    monkeypatch.delenv("COSMOS_QLORA_METRICS_FILE", raising=False)
    loaded = []
    if not real_cuda:

        class Packed(NativeLoRAMixin, torch.nn.Linear):
            def __init__(self, base, adapter, **kwargs):
                super().__init__(size, size, bias=False, device="meta", dtype=torch.bfloat16)
                self.weight.requires_grad_(False)
                self.lora_A = torch.nn.Linear(size, adapter.rank, bias=False, device="meta", dtype=torch.float32)
                self.lora_B = torch.nn.Linear(adapter.rank, size, bias=False, device="meta", dtype=torch.float32)

            def load_dense_weight_once(self, value, target):
                assert self.weight.is_meta and value.dtype == torch.bfloat16 and target == device
                loaded.append(tuple(value.shape))
                self.weight = torch.nn.Parameter(torch.ones(size * size // 2, dtype=torch.uint8), requires_grad=False)

        monkeypatch.setitem(
            sys.modules,
            "cosmos_framework.posttrain.model.quantization.bnb",
            SimpleNamespace(NativeLoraLinear4bit=Packed, kernel_preflight=lambda *a, **k: None),
        )
        monkeypatch.setattr(
            loader, "quantized_state_record", lambda m: {"weight": loader.tensor_digest(m.weight), "fixture": "not-NF4"}
        )
        monkeypatch.setattr(loader, "version", lambda name: "cpu-fixture")
        monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "cpu-fixture")
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (0, 0))
    net = torch.nn.Module()
    for name in targets:
        parent = net
        for part in name.split(".")[:-1]:
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent._modules[part]
        parent.add_module(
            name.split(".")[-1], torch.nn.Linear(size, size, bias=False, device="meta", dtype=torch.bfloat16)
        )
    path = tmp_path / ("source-" + str(len(list(tmp_path.iterdir()))))
    path.mkdir()
    generator = torch.Generator().manual_seed(19)
    source = {
        name + ".weight": torch.randn(size, size, generator=generator, dtype=torch.bfloat16) * 0.02 for name in targets
    }
    save_file(source, path / "model.safetensors")
    cfg = SimpleNamespace(
        name_or_path=str(path),
        adapter=NativeLoRAConfig(rank=1 if not real_cuda else 8, alpha=2 if not real_cuda else 16),
        quantized_base=QuantizedBaseTrainingConfig(excluded_language_layers=layers),
        adapter_compute_dtype="float32",
        activation_checkpointing="full",
        sft_logits_mode="full",
        compile_language_model=False,
    )
    life = loader.NativeAdapterLifecycle(cfg, object(), None)
    life.prepare(net)
    assert len(life.targets) == 252 and len(life.quantized_targets) == 252 - 7 * len(layers)
    life.materialize(net)
    for name, module in life.targets.items():
        assert module.weight.is_meta == (name in life.quantized_targets)
    life.load(net, str(path), None, None)
    life.finalize(net)
    return net, life, source, loaded


# ---------------------------------------------------------------------------
# A tiny family task policy (family-backbone tests and their subprocesses).
# ---------------------------------------------------------------------------

FAMILY_TARGETS = {"backbone.0": (4, 4), "backbone.2": (4, 4)}
FAMILY_AUXILIARY = ["projector.weight", "projector.bias", "head.weight", "head.bias"]
FAMILY_SOURCE = {"repository": "fixture/native-task", "revision": "a" * 40}
FAMILY_TASK = {"objective": "fixture_action_mse", "action_horizon": 2, "units": "normalized", "embodiment": "fixture"}


class TaskPolicy(nn.Module):
    def __init__(self, *, device="cpu"):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(314)
            self.vision = nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16)
            self.projector = nn.Linear(4, 4, device=device, dtype=torch.bfloat16)
            self.backbone = nn.Sequential(
                nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16),
                nn.Tanh(),
                nn.Linear(4, 4, bias=False, device=device, dtype=torch.bfloat16),
            )
            self.head = nn.Linear(4, 2, device=device, dtype=torch.bfloat16)
        self.register_buffer("scale", torch.tensor(0.5, dtype=torch.float32), persistent=False)

    def forward(self, images):
        visual = self.vision(images)
        projected = nn.functional.linear(visual.float(), self.projector.weight.float(), self.projector.bias.float())
        hidden = self.backbone(projected.to(torch.bfloat16))
        return nn.functional.linear(hidden.float(), self.head.weight.float(), self.head.bias.float()) * self.scale


def install_task_backbone(net, *, mode="bf16_lora", targets=FAMILY_TARGETS, auxiliary=FAMILY_AUXILIARY, **kwargs):
    return install_family_backbone(
        net,
        config=FamilyBackboneConfig(mode=mode, rank=2, alpha=4.0, seed=17, dropout=kwargs.pop("dropout", 0.0)),
        targets=targets,
        auxiliary_parameters=auxiliary,
        source_identity=kwargs.pop("source_identity", FAMILY_SOURCE),
        task_identity=kwargs.pop("task_identity", FAMILY_TASK),
        device=torch.device("cpu"),
        **kwargs,
    )


def task_batch():
    generator = torch.Generator().manual_seed(91)
    return torch.randn(6, 4, dtype=torch.bfloat16, generator=generator), torch.randn(6, 2, generator=generator)


# ---------------------------------------------------------------------------
# A tiny native Qwen3-VL with one image (CUDA tests).
# ---------------------------------------------------------------------------

IMAGE_TOKEN_ID = 120
VISION_START_TOKEN_ID = 118
VISION_END_TOKEN_ID = 119


def tiny_qwen3_vl_config() -> Any:
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


def tiny_qwen3_vl_batch() -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(  # [B,T]
            [[1, VISION_START_TOKEN_ID, IMAGE_TOKEN_ID, VISION_END_TOKEN_ID, 2]],
            dtype=torch.long,
        ),
        "attention_mask": torch.ones((1, 5), dtype=torch.bool),  # [B,T]
        "pixel_values": torch.randn((4, 24), dtype=torch.float32),  # [N_patch,D_patch]
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),  # [N_image,3]
    }
