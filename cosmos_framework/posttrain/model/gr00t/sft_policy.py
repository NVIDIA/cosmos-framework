# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""GR00T adapters on the complete native forward, with checkpoint-owned construction."""

from __future__ import annotations

import json
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator
from gr00t.model.modules.qwen3_backbone import Qwen3Backbone
from torch import nn
from transformers import PreTrainedModel, Qwen3VLConfig, Qwen3VLForConditionalGeneration

from cosmos_framework.posttrain.model.gr00t.policy import OFFICIAL_PRECISION
from cosmos_framework.posttrain.model.gr00t.sft import REFERENCE_REVISION, require_runtime
from cosmos_framework.posttrain.model.protocol import FlowMatchingLossOutput


class Gr00tSFTBackbone(Qwen3Backbone):
    """Reuse Isaac's forward; full GR00T weights supply all backbone parameters.

    The upstream constructor downloads redundant base weights before overwriting
    them from the full GR00T checkpoint. Construct the *same* modules from the
    exact dependency config, then strictly restore the whole GR00T checkpoint.
    This requires exact metadata; it never infers a gated model's config. The
    staging tool also accepts byte-identical files published in a public snapshot.
    """

    def __init__(self, config, backbone_path):
        nn.Module.__init__(self)
        backbone_config = Qwen3VLConfig.from_pretrained(backbone_path, local_files_only=True)
        if not 1 <= config.select_layer <= backbone_config.text_config.num_hidden_layers:
            raise ValueError("GR00T select_layer is outside its actual backbone")
        backbone_config._attn_implementation = "sdpa"
        backbone_config.text_config.use_cache = False
        self.model = Qwen3VLForConditionalGeneration(backbone_config).float()
        while len(self.model.language_model.layers) > config.select_layer:
            self.model.language_model.layers.pop(-1)
        self.select_layer = config.select_layer
        self.set_trainable_parameters(True, True, 0)
        self._reset_rotary_inv_freq()
        self._apply_vision_patch_embed_channels_last()
        # Isaac's input hook retains the fast channels-last Conv3d activation
        # path. FSDP2 requires contiguous parameter storage, however. cuDNN can
        # convert the weight for that input layout inside the convolution; never
        # change parameter storage during a sharded forward or replace its owner.
        for module in self.model.visual.modules():
            if isinstance(module, nn.Conv3d):
                module.weight.data = module.weight.data.contiguous()


class Gr00tSFTNet(Gr00tN1d7):
    def __init__(self, config, backbone_path):
        PreTrainedModel.__init__(self, config)
        self.backbone = Gr00tSFTBackbone(config, backbone_path)
        self.action_head = Gr00tN1d7ActionHead(config)
        self.collator = Gr00tN1d7DataCollator(backbone_path, transformers_loading_kwargs={"local_files_only": True})

    def forward(self, inputs, *, control_seed=None):
        if control_seed is not None and self.training:
            raise ValueError("GR00T control_seed is an evaluation probe only; training uses checkpointed native RNG")
        devices = [self.device.index] if self.device.type == "cuda" else []
        context = torch.random.fork_rng(devices=devices) if control_seed is not None else nullcontext()
        predictions = []
        # Expose the native prediction without duplicating its loss/noise math.
        # Root FSDP observes both the scalar loss and this tensor in the pytree.
        handle = self.action_head.action_decoder.register_forward_hook(
            lambda _module, _args, value: predictions.append(value)
        )
        try:
            with context:
                if control_seed is not None:
                    torch.random.default_generator.manual_seed(control_seed)
                    if devices:
                        torch.cuda.manual_seed(control_seed)
                output = super().forward(dict(inputs))
            if len(predictions) != 1:
                raise ValueError("GR00T native forward did not produce exactly one action prediction")
            prediction = predictions[0][:, -inputs["action"].shape[1] :]
            return FlowMatchingLossOutput(output["loss"], output["action_mask"], prediction)
        finally:
            handle.remove()


class Gr00tSFTPolicy:
    precision = OFFICIAL_PRECISION

    def __init__(self, net, *, autocast=True):
        self.net, self.autocast = net, autocast

    def flow_matching_loss(self, batch):
        with torch.autocast(device_type=self.net.device.type, dtype=torch.bfloat16, enabled=self.autocast):
            return self.net(batch["inputs"], control_seed=batch.get("control_seed"))


def external_identity():
    import gr00t

    from cosmos_framework.posttrain.utils.identity import canonical_json_sha256
    from cosmos_framework.posttrain.utils.object_store import file_sha256

    root = Path(gr00t.__file__).parent
    files = {
        str(p.relative_to(root)): file_sha256(str(p))
        for directory in ("model", "configs", "data")
        for p in (root / directory).rglob("*.py")
    }
    digest = canonical_json_sha256(files, context="GR00T source")
    if digest != "819b113eb4b47483d759127c6ffbf9f397241fc479070daea2d929fd371d7b35":
        raise ValueError(f"GR00T adapter SFT requires Isaac source at {REFERENCE_REVISION}; installed source differs")
    return {
        "reference_revision": REFERENCE_REVISION,
        "source_sha256": digest,
        "files": files,
    }


def source_identity(model_id, config):
    from cosmos_framework.posttrain.checkpoint.adapters import snapshot_identity
    from cosmos_framework.posttrain.utils.identity import canonical_json_sha256
    from cosmos_framework.posttrain.utils.object_store import file_sha256

    root = Path(model_id)
    processor = {
        name: file_sha256(str(root / name))
        for name in ("processor_config.json", "statistics.json", "embodiment_id.json")
    }
    backbone = snapshot_identity(config.backbone_path, weights=False)
    return {
        "base": snapshot_identity(model_id, weights=True),
        "processor": {
            "files": processor,
            "backbone": backbone,
            "sha256": canonical_json_sha256({"files": processor, "backbone": backbone}, context="GR00T processor"),
        },
        "model_contract": {
            "family": "gr00t-n1d7-full-flow-v1",
            "external": external_identity(),
            "construction": "full-checkpoint-from-exact-backbone-config",
            "base_precision": "float32",
            "compute": "bfloat16-autocast",
            "state_dropout_prob": json.loads((root / "config.json").read_text()).get("state_dropout_prob", 0.8),
        },
    }


def construct(model_id, config):
    require_runtime()
    payload = json.loads((Path(model_id) / "config.json").read_text())
    if payload.get("model_type") != "Gr00tN1d7":
        raise ValueError("GR00T SFT only binds Gr00tN1d7")
    payload.update(
        tune_llm=True,
        tune_visual=True,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        use_flash_attention=False,
        load_bf16=False,
    )
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(42)
        net = Gr00tSFTNet(Gr00tN1d7Config(**payload), config.backbone_path).float()
    return net


def load_base(model_id, config):
    from safetensors.torch import load_file

    if (Path(model_id) / "merged_model.json").exists():
        raise ValueError("GR00T merged artifacts require model.gr00t.adapter_export.load_merged")
    identity = source_identity(model_id, config)
    net = construct(model_id, config)
    root = Path(model_id)
    index = root / "model.safetensors.index.json"
    files = (
        sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
    )
    state = {}
    for name in files:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid GR00T shard name")
        shard = load_file(str(root / name))
        if set(state) & set(shard):
            raise ValueError("duplicate GR00T checkpoint tensor")
        state.update(shard)
    net.load_state_dict(state, strict=True)
    if any(v.is_meta for v in [*net.parameters(), *net.buffers()]):
        raise ValueError("GR00T materialization left meta state")
    if source_identity(model_id, config) != identity:
        raise ValueError("GR00T source changed during materialization")
    return net, identity


def build_policy(
    model_id, config, pdims=None, *, adapters=None, adapter_resume=False, adapter_media_identity=None, device="cuda"
):
    from gr00t.model.modules.dit import BasicTransformerBlock
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock

    from cosmos_framework.posttrain.model.adapters import (
        assert_trainability,
        prepare_adapter_model,
        validate_adapter_parallelism,
    )
    from cosmos_framework.posttrain.model.parallelize import parallelize
    from cosmos_framework.posttrain.model.specs import ShardPlan
    from cosmos_framework.posttrain.utils.dist import run_rank_converged

    config.__post_init__()
    if adapters is None:
        raise ValueError("GR00T full SFT requires explicit adapter components")
    validate_adapter_parallelism(pdims)
    net, identity = run_rank_converged(lambda: load_base(model_id, config), "GR00T base materialization")
    net.to(device)
    blocks = (Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock, BasicTransformerBlock)
    run_rank_converged(
        lambda: prepare_adapter_model(
            net,
            adapters,
            base=model_id,
            processor=config.backbone_path,
            base_dtype="float32",
            media=adapter_media_identity or {},
            resume=adapter_resume,
            distributed=pdims is not None and pdims.world_size > 1,
            family="Gr00tN1d7",
            fsdp_block_types=blocks,
            source_identity=identity,
        ),
        "GR00T adapter installation",
    )
    if config.activation_checkpointing == "full":
        apply_activation_checkpointing(
            net,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda m: isinstance(m, blocks),
        )
    parallelize(net, ShardPlan(blocks, None, torch.float32), pdims)
    assert_trainability(net)
    return Gr00tSFTPolicy(net)
