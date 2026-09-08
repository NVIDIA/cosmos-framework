# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import MISSING
from tqdm import tqdm

from cosmos_framework.model.generator.tokenizers.dc_ae.dc_ae_v_ops import (
    ChannelDuplicatingPixelShuffleUpSampleLayer3d,
    CompilableOpSequential3d,
    CompilableRMSNorm2d,
    ConvLayer3d,
    ConvPixelShuffleUpSampleLayer3d,
    ConvPixelUnshuffleDownSampleLayer3d,
    CustomConv3d,
    IdentityLayer,
    OpSequential3d,
    PixelUnshuffleChannelAveragingDownSampleLayer3d,
    ResBlock3d,
    ResidualBlock3d,
    TritonRMSNorm2d,
    build_act,
    build_norm,
)
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution


@dataclass
class BlockConfig:
    block_name: str = MISSING
    spatial_kernel_size: int = 3
    temporal_kernel_size: int = 1
    causal_chunk_length: Optional[int] = None
    spatial_padding_mode: Optional[str] = None
    temporal_padding_mode: Optional[str] = None


@dataclass
class SampleBlockConfig(BlockConfig):
    spatial_factor: int = 2
    temporal_factor: int = 1


@dataclass
class DCAEVEncoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING

    project_in_block_type: Any = field(
        default_factory=lambda: SampleBlockConfig(
            block_name="ConvPixelUnshuffle",
            spatial_factor=2,
            temporal_factor=1,
            spatial_kernel_size=3,
            temporal_kernel_size=1,
        )
    )
    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = field(
        default_factory=lambda: BlockConfig(block_name="ResBlock3d", spatial_kernel_size=3, temporal_kernel_size=1)
    )
    norm: Any = "trms2d"
    act: str = "silu"
    downsample_block_type: Any = field(
        default_factory=lambda: SampleBlockConfig(
            block_name="ConvPixelUnshuffle",
            spatial_factor=2,
            temporal_factor=1,
            spatial_kernel_size=3,
            temporal_kernel_size=1,
        )
    )
    downsample_shortcut: Optional[str] = "averaging"
    project_out_block_type: Any = field(
        default_factory=lambda: BlockConfig(block_name="ConvLayer3d", spatial_kernel_size=3, temporal_kernel_size=1)
    )

    zero_out: bool = MISSING


@dataclass
class DCAEVDecoderConfig:
    in_channels: int = MISSING
    latent_channels: int = MISSING

    project_in_block_type: Any = field(
        default_factory=lambda: BlockConfig(block_name="ConvLayer3d", spatial_kernel_size=3, temporal_kernel_size=1)
    )

    width_list: tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    depth_list: tuple[int, ...] = (2, 2, 2, 2, 2, 2)
    block_type: Any = field(
        default_factory=lambda: BlockConfig(block_name="ResBlock3d", spatial_kernel_size=3, temporal_kernel_size=1)
    )
    norm: Any = "trms2d"
    act: Any = "silu"
    upsample_block_type: Any = field(
        default_factory=lambda: SampleBlockConfig(
            block_name="ConvPixelShuffle",
            spatial_factor=2,
            temporal_factor=1,
            spatial_kernel_size=3,
            temporal_kernel_size=1,
        )
    )
    upsample_shortcut: str = "duplicating"
    project_out_block_type: Any = field(
        default_factory=lambda: SampleBlockConfig(
            block_name="ConvPixelShuffle",
            spatial_factor=2,
            temporal_factor=1,
            spatial_kernel_size=3,
            temporal_kernel_size=1,
        )
    )
    out_norm: str = "trms2d"
    out_act: str = "silu"

    zero_out: bool = MISSING


@dataclass
class DCAEVConfig:
    in_channels: int = 3
    latent_channels: int = 32
    encoder: DCAEVEncoderConfig = field(
        default_factory=lambda: DCAEVEncoderConfig(
            in_channels="${..in_channels}",
            latent_channels="${..latent_channels}",
            zero_out="${..zero_out}",
        )
    )
    decoder: DCAEVDecoderConfig = field(
        default_factory=lambda: DCAEVDecoderConfig(
            in_channels="${..in_channels}",
            latent_channels="${..latent_channels}",
            zero_out="${..zero_out}",
        )
    )

    num_pad_frames: int = 0
    temporal_remainder: int = 0

    pretrained_path: Optional[str] = None
    pretrained_source: str = "dc-ae-v"
    pretrained_ema: bool = True
    zero_out: bool = False
    use_feature_cache: bool = False
    # Back the encoder feature cache with statically addressed pool buffers that are
    # updated in place, instead of reallocating and cloning the cache for every tile.
    reuse_feature_cache_buffers: bool = True
    # Drop the feature cache when the whole input fits in one tile, as images do.
    skip_feature_cache_for_single_tile: bool = True
    # Write tiles straight into a preallocated output instead of collecting them in a
    # list and concatenating. Only applies when tiles do not overlap.
    preallocate_tiled_output: bool = True

    encode_temporal_tile_size: int | Mapping[str, int] | None = None
    encode_temporal_tile_latent_size: int | Mapping[str, int] | None = None
    tile_buckets: Mapping[str, Sequence[int]] | None = None
    no_tile_padding: bool = False
    decode_temporal_tile_size: Optional[int] = None
    decode_temporal_tile_latent_size: Optional[int] = None
    encode_temporal_tile_overlap_factor: float = 0.0
    decode_temporal_tile_overlap_factor: float = 0.0

    spatial_tile_size: Optional[int] = None
    spatial_tile_overlap_factor: float = 0.25

    scaling_factor: float = MISSING

    compilable: bool = False

    verbose: bool = False


def build_downsample_block(
    block_type: SampleBlockConfig, in_channels: int, out_channels: int, shortcut: Optional[str], zero_out: bool = False
) -> nn.Module:
    block_name = block_type.block_name
    kernel_size = (block_type.temporal_kernel_size, block_type.spatial_kernel_size, block_type.spatial_kernel_size)
    kwargs = {}
    if block_type.spatial_padding_mode is not None:
        kwargs["spatial_padding_mode"] = block_type.spatial_padding_mode
        kwargs["temporal_padding_mode"] = block_type.temporal_padding_mode
    if block_name in ["ConvPixelUnshuffle", "CausalConvPixelUnshuffle"]:
        block = ConvPixelUnshuffleDownSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            spatial_factor=block_type.spatial_factor,
            temporal_factor=block_type.temporal_factor,
            zero_out=zero_out,
            causal=block_name == "CausalConvPixelUnshuffle",
            **kwargs,
        )
    elif block_name == "ChunkCausalConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            spatial_factor=block_type.spatial_factor,
            temporal_factor=block_type.temporal_factor,
            causal_chunk_length=block_type.causal_chunk_length,
            **kwargs,
        )
    else:
        raise ValueError(f"block_name {block_name} is not supported for downsampling")
    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_factor=block_type.spatial_factor,
            temporal_factor=block_type.temporal_factor,
        )
        block = ResidualBlock3d(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block


def build_upsample_block(
    block_type: SampleBlockConfig, in_channels: int, out_channels: int, shortcut: Optional[str], zero_out: bool = False
) -> nn.Module:
    block_name = block_type.block_name
    kernel_size = (block_type.temporal_kernel_size, block_type.spatial_kernel_size, block_type.spatial_kernel_size)
    kwargs = {}
    if block_type.spatial_padding_mode is not None:
        kwargs["spatial_padding_mode"] = block_type.spatial_padding_mode
        kwargs["temporal_padding_mode"] = block_type.temporal_padding_mode
    if block_name in ["ConvPixelShuffle", "CausalConvPixelShuffle"]:
        block = ConvPixelShuffleUpSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            spatial_factor=block_type.spatial_factor,
            temporal_factor=block_type.temporal_factor,
            zero_out=zero_out,
            causal=block_name == "CausalConvPixelShuffle",
            **kwargs,
        )
    elif block_name in ["ChunkCausalConvPixelShuffle"]:
        block = ConvPixelShuffleUpSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            spatial_factor=block_type.spatial_factor,
            temporal_factor=block_type.temporal_factor,
            zero_out=zero_out,
            causal_chunk_length=block_type.causal_chunk_length,
            **kwargs,
        )
    else:
        raise ValueError(f"block_name {block_name} is not supported for upsampling")
    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelShuffleUpSampleLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_factor=block_type.spatial_factor,
            temporal_factor=block_type.temporal_factor,
        )
        block = ResidualBlock3d(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_block(
    block_type: BlockConfig, channels: int, norm: Optional[str], act: Optional[str], zero_out: bool
) -> nn.Module:
    block_name = block_type.block_name
    kernel_size = (block_type.temporal_kernel_size, block_type.spatial_kernel_size, block_type.spatial_kernel_size)
    kwargs = {}
    if block_type.spatial_padding_mode is not None:
        kwargs["spatial_padding_mode"] = block_type.spatial_padding_mode
        kwargs["temporal_padding_mode"] = block_type.temporal_padding_mode
    if block_name in ["ResBlock3d", "CausalResBlock3d"]:
        main_block = ResBlock3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            zero_out=zero_out,
            causal=block_name == "CausalResBlock3d",
            **kwargs,
        )
        block = ResidualBlock3d(main_block, IdentityLayer())
    elif block_name in ["ChunkCausalResBlock3d"]:
        main_block = ResBlock3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            use_bias=(True, False),
            norm=(None, norm),
            act_func=(act, None),
            zero_out=zero_out,
            causal_chunk_length=block_type.causal_chunk_length,
            **kwargs,
        )
        block = ResidualBlock3d(main_block, IdentityLayer())
    else:
        raise ValueError(f"block_name {block_name} is not supported")
    return block


def build_stage_main(
    width: int, depth: int, block_type: BlockConfig | list[BlockConfig], norm: str, act: str, zero_out: bool = False
) -> list[nn.Module]:
    assert isinstance(block_type, BlockConfig) or (isinstance(block_type, list) and depth == len(block_type))
    stage = []
    for d in range(depth):
        current_block_type = block_type[d] if isinstance(block_type, list) else block_type
        block = build_block(
            block_type=current_block_type,
            channels=width,
            norm=norm,
            act=act,
            zero_out=zero_out,
        )
        stage.append(block)
    return stage


def build_encoder_project_in_block(block_type: SampleBlockConfig, in_channels: int, out_channels: int):
    block = build_downsample_block(
        block_type=block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None
    )
    return block


def build_encoder_project_out_block(block_type: BlockConfig, in_channels: int, out_channels: int):
    block_name = block_type.block_name
    kernel_size = (block_type.temporal_kernel_size, block_type.spatial_kernel_size, block_type.spatial_kernel_size)
    kwargs = {}
    if block_type.spatial_padding_mode is not None:
        kwargs["spatial_padding_mode"] = block_type.spatial_padding_mode
        kwargs["temporal_padding_mode"] = block_type.temporal_padding_mode
    if block_name in ["ConvLayer3d", "CausalConvLayer3d"]:
        block = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal=block_name == "CausalConvLayer3d",
            **kwargs,
        )
    elif block_name in ["ChunkCausalConvLayer3d"]:
        block = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal_chunk_length=block_type.causal_chunk_length,
            **kwargs,
        )
    else:
        raise ValueError(f"encoder project out block name {block_name} is not supported")
    return block


def build_decoder_project_in_block(block_type: BlockConfig, in_channels: int, out_channels: int):
    block_name = block_type.block_name
    kernel_size = (block_type.temporal_kernel_size, block_type.spatial_kernel_size, block_type.spatial_kernel_size)
    kwargs = {}
    if block_type.spatial_padding_mode is not None:
        kwargs["spatial_padding_mode"] = block_type.spatial_padding_mode
        kwargs["temporal_padding_mode"] = block_type.temporal_padding_mode
    if block_name in ["ConvLayer3d", "CausalConvLayer3d"]:
        block = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal=block_name == "CausalConvLayer3d",
            **kwargs,
        )
    elif block_name in ["ChunkCausalConvLayer3d"]:
        block = ConvLayer3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            use_bias=True,
            norm=None,
            act_func=None,
            causal_chunk_length=block_type.causal_chunk_length,
            **kwargs,
        )
    else:
        raise ValueError(f"decoder project in block name {block_name} is not supported")
    return block


def build_decoder_project_out_block(
    block_type: SampleBlockConfig, in_channels: int, out_channels: int, norm: Optional[str], act: Optional[str]
):
    layers: list[nn.Module] = [
        build_norm(norm, in_channels),
        build_act(act),
        build_upsample_block(block_type=block_type, in_channels=in_channels, out_channels=out_channels, shortcut=None),
    ]
    return OpSequential3d(layers)


class DCAEVEncoder(nn.Module):
    def __init__(self, cfg: DCAEVEncoderConfig):
        super().__init__()
        self.cfg = cfg

        start_stage = 0
        while cfg.depth_list[start_stage] == 0:
            start_stage += 1
        self.project_in = build_encoder_project_in_block(
            block_type=cfg.project_in_block_type,
            in_channels=cfg.in_channels,
            out_channels=cfg.width_list[start_stage],
        )

        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, BlockConfig) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (isinstance(cfg.norm, list) and len(cfg.norm) == num_stages)
        assert isinstance(cfg.downsample_block_type, SampleBlockConfig) or (
            isinstance(cfg.downsample_block_type, list) and len(cfg.downsample_block_type) == num_stages - 1
        )

        self.stages: list[OpSequential3d] = []
        for stage_id, (width, depth) in enumerate(zip(cfg.width_list, cfg.depth_list)):
            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            stage = build_stage_main(
                width=width,
                depth=depth,
                block_type=block_type,
                norm=norm,
                act=cfg.act,
                zero_out=cfg.zero_out,
            )
            if stage_id < num_stages - 1 and depth > 0:
                downsample_block_type = (
                    cfg.downsample_block_type[stage_id]
                    if isinstance(cfg.downsample_block_type, list)
                    else cfg.downsample_block_type
                )
                downsample_block = build_downsample_block(
                    block_type=downsample_block_type,
                    in_channels=width,
                    out_channels=cfg.width_list[stage_id + 1],
                    shortcut=cfg.downsample_shortcut,
                    zero_out=cfg.zero_out,
                )
                stage.append(downsample_block)
            self.stages.append(OpSequential3d(stage))
        self.stages = nn.ModuleList(self.stages)

        self.project_out = build_encoder_project_out_block(
            block_type=cfg.project_out_block_type,
            in_channels=cfg.width_list[-1],
            out_channels=cfg.latent_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x, _ = self.project_in(x, feature_cache, feat_idx)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            x, _ = stage(x, feature_cache, feat_idx)
        x, _ = self.project_out(x, feature_cache, feat_idx)
        return x, {}


class DCAEVDecoder(nn.Module):
    def __init__(self, cfg: DCAEVDecoderConfig):
        super().__init__()
        self.cfg = cfg

        self.project_in = build_decoder_project_in_block(
            block_type=cfg.project_in_block_type,
            in_channels=cfg.latent_channels,
            out_channels=cfg.width_list[-1],
        )

        num_stages = len(cfg.width_list)
        self.num_stages = num_stages
        assert len(cfg.depth_list) == num_stages
        assert len(cfg.width_list) == num_stages
        assert isinstance(cfg.block_type, BlockConfig) or (
            isinstance(cfg.block_type, list) and len(cfg.block_type) == num_stages
        )
        assert isinstance(cfg.norm, str) or (isinstance(cfg.norm, list) and len(cfg.norm) == num_stages)
        assert isinstance(cfg.act, str) or (isinstance(cfg.act, list) and len(cfg.act) == num_stages)
        assert isinstance(cfg.upsample_block_type, SampleBlockConfig) or (
            isinstance(cfg.upsample_block_type, list) and len(cfg.upsample_block_type) == num_stages - 1
        )
        self.stages: list[OpSequential3d] = []
        self.spatial_compression_ratio = 1
        self.temporal_compression_ratio = 1
        for stage_id, (width, depth) in reversed(list(enumerate(zip(cfg.width_list, cfg.depth_list)))):
            stage = []
            if stage_id < num_stages - 1 and depth > 0:
                upsample_block_type = (
                    cfg.upsample_block_type[stage_id]
                    if isinstance(cfg.upsample_block_type, list)
                    else cfg.upsample_block_type
                )
                upsample_block = build_upsample_block(
                    block_type=upsample_block_type,
                    in_channels=cfg.width_list[stage_id + 1],
                    out_channels=width,
                    shortcut=cfg.upsample_shortcut,
                    zero_out=cfg.zero_out,
                )
                stage.append(upsample_block)
                self.spatial_compression_ratio *= upsample_block_type.spatial_factor
                self.temporal_compression_ratio *= upsample_block_type.temporal_factor

            block_type = cfg.block_type[stage_id] if isinstance(cfg.block_type, list) else cfg.block_type
            norm = cfg.norm[stage_id] if isinstance(cfg.norm, list) else cfg.norm
            act = cfg.act[stage_id] if isinstance(cfg.act, list) else cfg.act
            stage.extend(
                build_stage_main(
                    width=width, depth=depth, block_type=block_type, norm=norm, act=act, zero_out=cfg.zero_out
                )
            )
            self.stages.insert(0, OpSequential3d(stage))
        self.stages = nn.ModuleList(self.stages)

        start_stage = 0
        while cfg.depth_list[start_stage] == 0:
            start_stage += 1
        self.project_out = build_decoder_project_out_block(
            block_type=cfg.project_out_block_type,
            in_channels=cfg.width_list[start_stage],
            out_channels=cfg.in_channels,
            norm=cfg.out_norm,
            act=cfg.out_act,
        )
        self.spatial_compression_ratio *= cfg.project_out_block_type.spatial_factor
        self.temporal_compression_ratio *= cfg.project_out_block_type.temporal_factor

    def forward(
        self,
        x: torch.Tensor,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x, _ = self.project_in(x, feature_cache, feat_idx)
        for stage_id, stage in reversed(list(enumerate(self.stages))):
            if len(stage.op_list) == 0:
                continue
            x, _ = stage(x, feature_cache, feat_idx)
        x, _ = self.project_out(x, feature_cache, feat_idx)
        return x, {}


def _replace_with_compilable_ops(module: nn.Module) -> None:
    """Recursively replace compile-unfriendly ops throughout *module*:
    - TritonRMSNorm2d   -> CompilableRMSNorm2d  (pure PyTorch, no Triton kernel)
    - OpSequential3d    -> CompilableOpSequential3d (no isinstance dispatch)
    - CustomConv3d      -> torch.nn.Conv3d
    """
    for name, child in list(module.named_children()):
        if isinstance(child, TritonRMSNorm2d):
            compilable = CompilableRMSNorm2d(child.normalized_shape, eps=child.eps)
            compilable.weight = child.weight
            compilable.bias = child.bias
            setattr(module, name, compilable)
        elif isinstance(child, CustomConv3d):
            compilable_conv = torch.nn.Conv3d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
            )
            compilable_conv.weight = child.weight
            if child.bias is not None:
                compilable_conv.bias = child.bias
            setattr(module, name, compilable_conv)
        elif isinstance(child, OpSequential3d) and not isinstance(child, CompilableOpSequential3d):
            compilable_seq = CompilableOpSequential3d.from_op_sequential_3d(child)
            setattr(module, name, compilable_seq)
            _replace_with_compilable_ops(compilable_seq)
        else:
            _replace_with_compilable_ops(child)


class CompilableDCAEVEncoder(DCAEVEncoder):
    """DCAEVEncoder with compile-friendly ops."""

    def __init__(self, cfg: DCAEVEncoderConfig):
        super().__init__(cfg)
        _replace_with_compilable_ops(self)

    def forward(
        self,
        x: torch.Tensor,
        *,
        feature_cache: list[torch.Tensor | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x = x.to(memory_format=torch.channels_last_3d)
        x, _ = self.project_in(x, feature_cache, feat_idx)
        for stage in self.stages:
            if len(stage.op_list) == 0:
                continue
            x, _ = stage(x, feature_cache, feat_idx)
        x, _ = self.project_out(x, feature_cache, feat_idx)
        return x, {}


class CompilableDCAEVDecoder(DCAEVDecoder):
    """DCAEVDecoder with compile-friendly ops."""

    def __init__(self, cfg: DCAEVDecoderConfig):
        super().__init__(cfg)
        _replace_with_compilable_ops(self)


def _count_causal_convs(model: nn.Module) -> int:
    """Count ConvLayer3d instances with causal or causal_chunk_length set.
    Used to pre-allocate the flat feature cache list."""
    count = 0
    for m in model.modules():
        if isinstance(m, ConvLayer3d) and (m.causal or m.causal_chunk_length is not None):
            count += 1
    return count


class _FeatureCacheSlotSpec(NamedTuple):
    """Shape of one causal-conv cache slot, excluding the batch dimension."""

    channels: int
    pad_t: int
    height: int
    width: int


def _encoder_feature_cache_specs(
    encoder: nn.Module,
    height: int,
    width: int,
) -> list[_FeatureCacheSlotSpec]:
    """Describe the feature cache slots of a causal encoder, in forward order.

    Walks the encoder in forward order, emitting one spec for every causal
    ``ConvLayer3d`` and shrinking H/W whenever a
    ``ConvPixelUnshuffleDownSampleLayer3d`` is encountered.
    """
    specs: list[_FeatureCacheSlotSpec] = []
    h = height
    w = width

    def _visit(module: nn.Module) -> None:
        nonlocal h, w
        if isinstance(module, ConvPixelUnshuffleDownSampleLayer3d):
            _visit(module.conv)
            h //= module.spatial_factor
            w //= module.spatial_factor
        elif isinstance(module, ConvLayer3d):
            if module.causal:
                specs.append(
                    _FeatureCacheSlotSpec(
                        channels=module.conv.in_channels,
                        pad_t=module.custom_padding[4],
                        height=h,
                        width=w,
                    )
                )
        elif isinstance(module, ResBlock3d):
            _visit(module.conv1)
            _visit(module.conv2)
        elif isinstance(module, ResidualBlock3d):
            if module.main is not None:
                _visit(module.main)
        elif isinstance(module, (OpSequential3d, CompilableOpSequential3d)):
            for op in module.op_list:
                _visit(op)
        else:
            raise ValueError(f"Unsupported module: {type(module)}")

    _visit(encoder.project_in)
    for stage in encoder.stages:
        _visit(stage)
    _visit(encoder.project_out)

    return specs


def _build_encoder_feature_cache(
    encoder: nn.Module,
    batch_size: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    """Pre-allocate zero-filled feature cache for a causal encoder.

    Creates a ``(B, C_in, pad, H, W)`` zero tensor for every causal
    ``ConvLayer3d``. Each call returns freshly allocated tensors; use
    ``_EncoderFeatureCachePool`` when the buffers must live at stable addresses.
    """
    return [
        torch.zeros(
            batch_size,
            spec.channels,
            spec.pad_t,
            spec.height,
            spec.width,
            dtype=dtype,
            device=device,
        ).to(memory_format=torch.channels_last_3d)
        for spec in _encoder_feature_cache_specs(encoder, height, width)
    ]


def _encoder_pads_causal_head_with_zeros(encoder: nn.Module) -> bool:
    """Whether every causal conv fills its temporal head padding with zeros.

    A freshly built cache is all zeros, so on the first chunk each conv overwrites its
    zero head padding with zeros. Only when the padding is zeros to begin with is that
    overwrite a no-op, and therefore only then can the cache be dropped entirely for an
    input that is encoded as a single chunk.
    """
    return all(
        module.custom_padding_mode == "constant"
        for module in encoder.modules()
        if isinstance(module, ConvLayer3d) and module.causal
    )


def _restore_static_cache_slots(cache: list[torch.Tensor | None], static_slots: list[torch.Tensor]) -> None:
    """Copy any rebound cache entry back into its static buffer and restore the reference.

    Convs that cannot update their slot in place rebind it to a fresh tensor,
    which under ``torch.compile`` lives in CUDA-graph-managed memory that the
    next replay overwrites. Snapshotting those few entries here keeps the cache
    list pointing at the pool's static buffers for the following tile.
    """
    for idx, static_slot in enumerate(static_slots):
        rebound = cache[idx]
        if rebound is not static_slot:
            assert rebound is not None, f"Feature cache slot {idx} was cleared by the encoder"
            static_slot.copy_(rebound)
            cache[idx] = static_slot


# Slot offsets are aligned so that every slot view starts at an address suitable
# for reinterpreting the flat byte pool as the cache dtype.
_FEATURE_CACHE_ALIGNMENT_BYTES = 512


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _normalized_device(device: torch.device | str) -> torch.device:
    """Resolve a device to an indexed form so cache keys compare equal.

    ``torch.device("cuda")`` and the ``cuda:0`` reported by a tensor are distinct
    dict keys, which would otherwise split one logical pool into two.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


class _EncoderFeatureCacheKey(NamedTuple):
    """Identifies one set of encoder feature cache slots."""

    batch_size: int
    height: int
    width: int
    dtype: torch.dtype
    device: torch.device


class _EncoderFeatureCacheLayout(NamedTuple):
    """Byte offsets of every slot for one key, plus the total pool bytes required."""

    offsets: tuple[int, ...]
    specs: tuple[_FeatureCacheSlotSpec, ...]
    num_bytes: int


class _EncoderFeatureCachePool:
    """Flat byte pool backing the causal feature cache of an encoder.

    All slots for one key are carved from a single allocation at fixed offsets, so
    slot addresses stay stable across encode calls. That is what allows the cache
    to be updated in place instead of cloned per tile: ``torch.compile`` with CUDA
    graphs only captures mutations of inputs whose address is known to be static.

    Memory is bounded by the largest key ever requested rather than the sum over
    keys, which matters for mixed-resolution training where every resolution and
    aspect ratio is a separate key.
    """

    def __init__(self) -> None:
        self._pools: dict[torch.device, torch.Tensor] = {}
        self._layouts: dict[_EncoderFeatureCacheKey, _EncoderFeatureCacheLayout] = {}
        self._slots: dict[_EncoderFeatureCacheKey, list[torch.Tensor]] = {}

    def _layout(self, encoder: nn.Module, key: _EncoderFeatureCacheKey) -> _EncoderFeatureCacheLayout:
        layout = self._layouts.get(key)
        if layout is not None:
            return layout

        specs = _encoder_feature_cache_specs(encoder, key.height, key.width)
        offsets: list[int] = []
        offset = 0
        for spec in specs:
            offsets.append(offset)
            slot_bytes = key.batch_size * spec.channels * spec.pad_t * spec.height * spec.width * key.dtype.itemsize
            offset += _align_up(slot_bytes, _FEATURE_CACHE_ALIGNMENT_BYTES)
        layout = _EncoderFeatureCacheLayout(offsets=tuple(offsets), specs=tuple(specs), num_bytes=offset)
        self._layouts[key] = layout
        return layout

    def num_bytes(self, encoder: nn.Module, key: _EncoderFeatureCacheKey) -> int:
        """Return the pool bytes required to hold every slot of ``key``."""
        return self._layout(encoder, key).num_bytes

    def pool_num_bytes(self, device: torch.device) -> int:
        """Return the currently allocated pool size on ``device``, for tests and logging."""
        pool = self._pools.get(device)
        return 0 if pool is None else pool.numel()

    def reserve(self, encoder: nn.Module, keys: Sequence[_EncoderFeatureCacheKey]) -> None:
        """Grow the pool up front so later ``acquire`` calls never move slot addresses."""
        required: dict[torch.device, int] = {}
        for key in keys:
            required[key.device] = max(required.get(key.device, 0), self.num_bytes(encoder, key))
        for device, num_bytes in required.items():
            self._ensure_pool(device, num_bytes)

    def acquire(self, encoder: nn.Module, key: _EncoderFeatureCacheKey) -> list[torch.Tensor]:
        """Return the zero-filled slots for ``key``, reusing the same buffers every call.

        Keys on one device share the underlying storage, so only the most recently
        acquired key holds meaningful data. That is fine because every acquire starts
        from zeros; what has to stay stable is the addresses, not the contents.
        """
        layout = self._layout(encoder, key)
        pool = self._ensure_pool(key.device, layout.num_bytes)
        slots = self._slots.get(key)
        if slots is None:
            slots = self._carve(pool, key, layout)
            self._slots[key] = slots
        # One kernel for the whole key instead of one per slot; the alignment gaps
        # between slots are zeroed too, which is harmless.
        pool[: layout.num_bytes].zero_()
        return slots

    def clear(self) -> None:
        """Drop every buffer. Invalidates slot addresses, so CUDA graphs must be re-recorded."""
        self._pools.clear()
        self._layouts.clear()
        self._slots.clear()

    def _ensure_pool(self, device: torch.device, num_bytes: int) -> torch.Tensor:
        pool = self._pools.get(device)
        if pool is not None and pool.numel() >= num_bytes:
            return pool
        pool = torch.empty(num_bytes, dtype=torch.uint8, device=device)
        self._pools[device] = pool
        # Growing reallocates, so previously carved views of this device are stale.
        self._slots = {key: slots for key, slots in self._slots.items() if key.device != device}
        return pool

    def _carve(
        self,
        pool: torch.Tensor,
        key: _EncoderFeatureCacheKey,
        layout: _EncoderFeatureCacheLayout,
    ) -> list[torch.Tensor]:
        slots: list[torch.Tensor] = []
        for offset, spec in zip(layout.offsets, layout.specs, strict=True):
            numel = key.batch_size * spec.channels * spec.pad_t * spec.height * spec.width
            elements = pool[offset : offset + numel * key.dtype.itemsize].view(key.dtype)
            # Viewing as (B, T, H, W, C) and moving C back to dim 1 yields exactly the
            # strides of a channels_last_3d tensor, matching _build_encoder_feature_cache.
            slot = elements.view(key.batch_size, spec.pad_t, spec.height, spec.width, spec.channels).permute(
                0, 4, 1, 2, 3
            )
            assert slot.is_contiguous(memory_format=torch.channels_last_3d), (
                f"Feature cache slot {tuple(slot.shape)} is not channels_last_3d"
            )
            torch._dynamo.mark_static_address(slot)
            slots.append(slot)
        return slots


def _select_temporal_tile_size(
    num_frames: int,
    default_tile_size: int,
    tile_buckets: Sequence[int],
) -> int:
    """Select the smallest allowed tile that can contain ``num_frames``."""
    if num_frames > default_tile_size:
        raise ValueError(f"num_frames ({num_frames}) exceeds the default tile size ({default_tile_size})")
    return min(tile_size for tile_size in (*tile_buckets, default_tile_size) if tile_size >= num_frames)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _empty_like_with_num_frames(reference: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Allocate ``[B,C,num_frames,H,W]`` matching ``reference``'s dtype, device and layout.

    Matching the layout keeps every per-tile copy into the result a plain memcpy.
    """
    output = torch.empty(
        (reference.shape[0], reference.shape[1], num_frames, reference.shape[3], reference.shape[4]),
        dtype=reference.dtype,
        device=reference.device,
    )
    if reference.is_contiguous(memory_format=torch.channels_last_3d):
        output = output.to(memory_format=torch.channels_last_3d)
    return output


class _CompiledEncoderKey(NamedTuple):
    """Identifies one compiled encoder executable.

    Passing a feature cache or not changes the traced graph, so the two variants of a
    shape are separate executables: separate CUDA graph captures now, and separate
    ahead-of-time compiled artifacts later.
    """

    shape: tuple[int, ...]
    with_feature_cache: bool


class DCAEV(nn.Module):
    _cudagraph_encoders: dict[_CompiledEncoderKey, nn.Module]
    _fallback_compiled_encoder: nn.Module | None
    _feature_cache_pool: _EncoderFeatureCachePool

    def __init__(self, cfg: DCAEVConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.compilable:
            self.encoder = CompilableDCAEVEncoder(cfg.encoder)
            self.decoder = DCAEVDecoder(cfg.decoder)
        else:
            self.encoder = DCAEVEncoder(cfg.encoder)
            self.decoder = DCAEVDecoder(cfg.decoder)

        # Compiled executables are runtime-only wrappers around ``self.encoder``.
        # Bypass nn.Module registration to avoid duplicating encoder parameters in the state dict.
        object.__setattr__(self, "_cudagraph_encoders", {})
        object.__setattr__(self, "_fallback_compiled_encoder", None)
        # Same reasoning for the feature cache pool: runtime scratch, not model state.
        object.__setattr__(self, "_feature_cache_pool", _EncoderFeatureCachePool())

        if cfg.pretrained_path is not None:
            self.load_model()

    def set_fallback_compiled_encoder(self, encoder: nn.Module) -> None:
        """Install the dynamic compiled encoder used when no CUDA graph shape matches."""
        object.__setattr__(self, "_fallback_compiled_encoder", encoder)

    def _get_feature_cache_pool(self) -> _EncoderFeatureCachePool:
        """Return the feature cache pool, initializing it for lightweight test instances."""
        pool = self.__dict__.get("_feature_cache_pool")
        if pool is None:
            pool = _EncoderFeatureCachePool()
            object.__setattr__(self, "_feature_cache_pool", pool)
        return pool

    def may_skip_feature_cache(self) -> bool:
        """Whether some encode calls run the encoder without a feature cache at all.

        True when a single-tile input (an image, which is one frame padded up to one
        short tile) is eligible to skip the cache. Compiled callers need this because a
        cacheless call is traced as its own graph and has to be warmed up separately.
        """
        if not (self.cfg.use_feature_cache and self.cfg.skip_feature_cache_for_single_tile):
            return False
        pads_with_zeros = self.__dict__.get("_causal_head_padding_is_zeros")
        if pads_with_zeros is None:
            pads_with_zeros = _encoder_pads_causal_head_with_zeros(self.encoder)
            object.__setattr__(self, "_causal_head_padding_is_zeros", pads_with_zeros)
        return pads_with_zeros

    def _skips_feature_cache_for_tile(self, shape: tuple[int, ...]) -> bool:
        """Whether a lone tile of ``shape`` should be encoded without a feature cache.

        Once compiled executables are installed, this needs one that was compiled
        cacheless for this exact shape. Sending an unregistered shape down the cacheless
        path would instead cost another compile and CUDA graph capture, so only the image
        bucket gets that executable and every longer clip keeps its cache.
        """
        if not self.may_skip_feature_cache():
            return False
        if not self.has_compiled_encoders():
            return True
        return self.has_cudagraph_encoder(shape, with_feature_cache=False)

    def reserve_encoder_feature_cache(
        self,
        shapes: Sequence[tuple[int, int, int]],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Pre-size the feature cache pool for ``(batch_size, height, width)`` triples.

        Call this before compiling or capturing the encoder: growing the pool later
        moves slot addresses, which forces CUDA graphs to be re-recorded.
        """
        if not (self.cfg.use_feature_cache and self.cfg.reuse_feature_cache_buffers):
            return
        normalized_device = _normalized_device(device)
        keys = [
            _EncoderFeatureCacheKey(
                batch_size=batch_size, height=height, width=width, dtype=dtype, device=normalized_device
            )
            for batch_size, height, width in shapes
        ]
        self._get_feature_cache_pool().reserve(self.encoder, keys)

    def prepare_encoder_feature_cache(
        self,
        *,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[list[torch.Tensor | None] | None, list[torch.Tensor] | None]:
        """Return ``(working_cache, static_slots)`` for one encode pass.

        ``working_cache`` is the list handed to the encoder; ops that cannot update
        their slot in place rebind entries in it. ``static_slots`` is the pool-owned
        list those entries are restored to via ``_restore_static_cache_slots``, and is
        ``None`` when pooling is disabled (in which case the caller is responsible for
        cloning the cache between tiles).
        """
        if not self.cfg.use_feature_cache:
            return None, None
        if not self.cfg.reuse_feature_cache_buffers:
            unpooled_cache: list[torch.Tensor | None] = list(
                _build_encoder_feature_cache(
                    self.encoder,
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    dtype=dtype,
                    device=device,
                )
            )
            return unpooled_cache, None
        key = _EncoderFeatureCacheKey(
            batch_size=batch_size, height=height, width=width, dtype=dtype, device=_normalized_device(device)
        )
        static_slots = self._get_feature_cache_pool().acquire(self.encoder, key)
        working_cache: list[torch.Tensor | None] = list(static_slots)
        return working_cache, static_slots

    def _get_cudagraph_encoders(self) -> dict[_CompiledEncoderKey, nn.Module]:
        """Return the CUDA graph registry, initializing it for lightweight test instances."""
        encoders = self.__dict__.get("_cudagraph_encoders")
        if encoders is None:
            encoders = {}
            object.__setattr__(self, "_cudagraph_encoders", encoders)
        return encoders

    def set_cudagraph_encoder(
        self,
        shape: tuple[int, ...],
        encoder: nn.Module,
        *,
        with_feature_cache: bool,
    ) -> None:
        """Install a CUDA graph compiled encoder for an exact input shape.

        ``with_feature_cache`` states whether this executable is the one called with a
        feature cache; it is part of the key, so both variants of a shape can coexist.
        """
        key = _CompiledEncoderKey(shape=shape, with_feature_cache=with_feature_cache)
        self._get_cudagraph_encoders()[key] = encoder

    def get_cudagraph_encoder(self, shape: tuple[int, ...], *, with_feature_cache: bool) -> nn.Module | None:
        """Return the CUDA graph compiled encoder for an exact input shape and variant."""
        key = _CompiledEncoderKey(shape=shape, with_feature_cache=with_feature_cache)
        return self._get_cudagraph_encoders().get(key)

    def has_cudagraph_encoder(self, shape: tuple[int, ...], *, with_feature_cache: bool) -> bool:
        """Return whether an exact-shape CUDA graph encoder is installed for the variant."""
        return self.get_cudagraph_encoder(shape, with_feature_cache=with_feature_cache) is not None

    def clear_compiled_encoders(self) -> None:
        """Remove all runtime compiled encoder executables."""
        self._get_cudagraph_encoders().clear()
        object.__setattr__(self, "_fallback_compiled_encoder", None)

    def has_compiled_encoders(self) -> bool:
        """Whether any compiled encoder executable is installed."""
        return bool(self._get_cudagraph_encoders()) or self.__dict__.get("_fallback_compiled_encoder") is not None

    def _get_encoder_for_tile(
        self,
        tile: torch.Tensor,  # [B,C,T,H,W]
        *,
        with_feature_cache: bool,
    ) -> nn.Module:
        encoder = self.get_cudagraph_encoder(tuple(tile.shape), with_feature_cache=with_feature_cache)
        if encoder is not None:
            return encoder
        fallback_encoder = self.__dict__.get("_fallback_compiled_encoder")
        if fallback_encoder is not None:
            return fallback_encoder
        return self.encoder

    def load_model(self):
        if self.cfg.pretrained_source == "dc-ae-v-fsdp":
            checkpoint = torch.load(self.cfg.pretrained_path, map_location="cpu", weights_only=True)
            if self.cfg.pretrained_ema and "ema_model_state_dict" in checkpoint:
                state_dict = checkpoint["ema_model_state_dict"]
                state_dict = state_dict[list(state_dict)[0]]
            else:
                state_dict = checkpoint["model_state_dict"]
            self.load_state_dict(state_dict)
        else:
            raise NotImplementedError

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            blend_ratio = x / blend_extent
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - blend_ratio) + b[:, :, x, :, :] * blend_ratio
        return b

    def blend_w(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[..., y, :] = a[..., -blend_extent + y, :] * (1 - y / blend_extent) + b[..., y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[..., x] = a[..., -blend_extent + x] * (1 - x / blend_extent) + b[..., x] * (x / blend_extent)
        return b

    def _padded_tile_frames(self, actual_t: int, tile_size: int, tile_buckets: Sequence[int]) -> int:
        """Frame count a tile of ``actual_t`` frames is padded up to before encoding.

        Padding a short tile up to a bucket is what lets it reuse a compiled shape.
        """
        if actual_t < tile_size and self.cfg.compilable and not self.cfg.no_tile_padding:
            return _select_temporal_tile_size(actual_t, tile_size, tile_buckets)
        return actual_t

    def temporal_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        resolution = get_vision_data_resolution((x.shape[3], x.shape[4]))
        if isinstance(self.cfg.encode_temporal_tile_size, Mapping):
            tile_size = self.cfg.encode_temporal_tile_size[resolution]
        else:
            tile_size = self.cfg.encode_temporal_tile_size
        if isinstance(self.cfg.encode_temporal_tile_latent_size, Mapping):
            tile_latent_size = self.cfg.encode_temporal_tile_latent_size[resolution]
        else:
            tile_latent_size = self.cfg.encode_temporal_tile_latent_size
        if tile_size is None or tile_latent_size is None:
            raise ValueError("Encode temporal tile sizes must be configured for temporal tiled encoding")

        if self.cfg.tile_buckets is None:
            tile_buckets: Sequence[int] = ()
        else:
            tile_buckets = self.cfg.tile_buckets[resolution]

        overlap_size = int(tile_size * (1 - self.cfg.encode_temporal_tile_overlap_factor))
        blend_extent = int(tile_latent_size * self.cfg.encode_temporal_tile_overlap_factor)
        t_limit = tile_latent_size - blend_extent

        tile_starts = range(0, x.shape[2], overlap_size)
        # A lone tile has no successor to consume the causal cache, and the cache it
        # would read from is still all zeros, so the whole cache is dead work. This is
        # the image path: one frame padded up to a single short tile.
        if len(tile_starts) == 1 and self._skips_feature_cache_for_tile(
            (
                x.shape[0],
                x.shape[1],
                self._padded_tile_frames(x.shape[2], tile_size, tile_buckets),
                x.shape[3],
                x.shape[4],
            )
        ):
            feature_cache: list[torch.Tensor | None] | None = None
            static_slots: list[torch.Tensor] | None = None
        else:
            feature_cache, static_slots = self.prepare_encoder_feature_cache(
                batch_size=x.shape[0],
                height=x.shape[3],
                width=x.shape[4],
                dtype=x.dtype,
                device=x.device,
            )
        feat_idx: list[int] | None = None if feature_cache is None else [0]

        # Split the video into tiles and encode them separately. With tile padding
        # enabled, pad the final tile to an allowed CUDA graph shape. With
        # no_tile_padding, preserve its exact shape and use the dynamic fallback
        # unless an exact-shape CUDA graph executable is available.
        compression_factor = self.decoder.temporal_compression_ratio

        # Tiles only need to be kept around for blending, so when they do not overlap
        # they can be written straight into the final latent instead of being collected
        # and concatenated. That also removes the copy that protects each tile from the
        # next CUDA graph replay, since the write into the output already snapshots it.
        preallocate = self.cfg.preallocate_tiled_output and blend_extent == 0
        output: torch.Tensor | None = None
        num_latent_frames = 0
        row: list[torch.Tensor] = []
        if preallocate:
            num_latent_frames = sum(
                min(_ceil_div(min(tile_size, x.shape[2] - start), compression_factor), t_limit) for start in tile_starts
            )
        write_pos = 0

        for start in tqdm(tile_starts, desc="Tiled Encode", disable=not self.cfg.verbose):
            # Clone is required for compiled tokenizer to avoid recompilation (view has different memory strides).
            tile = x[:, :, start : start + tile_size, :, :].clone()
            actual_t = tile.shape[2]
            final_tile_size = self._padded_tile_frames(actual_t, tile_size, tile_buckets)
            remove_padding = final_tile_size > actual_t
            if remove_padding:
                tile = F.pad(tile, (0, 0, 0, 0, 0, final_tile_size - actual_t))
            assert tile.numel() < 1 << 31, "Tile size exceeds the int32 limit (torch compile and/or cudnn indexing)"

            if feat_idx is not None:
                feat_idx[0] = 0
            if static_slots is None and feature_cache is not None and self.cfg.compilable:
                old_feature_cache = feature_cache
                feature_cache = [f.clone() if f is not None else None for f in feature_cache]
                old_feature_cache.clear()
            encoder = self._get_encoder_for_tile(tile, with_feature_cache=feature_cache is not None)
            tile_latent = encoder(tile, feature_cache=feature_cache, feat_idx=feat_idx)[0]
            if static_slots is not None:
                assert feature_cache is not None
                _restore_static_cache_slots(feature_cache, static_slots)
            if not preallocate:
                tile_latent = tile_latent.clone()
            if remove_padding:
                valid_latent_t = _ceil_div(actual_t, compression_factor)
                tile_latent = tile_latent[:, :, :valid_latent_t, :, :]

            if preallocate:
                tile_latent = tile_latent[:, :, :t_limit, :, :]
                if output is None:
                    output = _empty_like_with_num_frames(tile_latent, num_latent_frames)
                output[:, :, write_pos : write_pos + tile_latent.shape[2]].copy_(tile_latent)
                write_pos += tile_latent.shape[2]
            else:
                row.append(tile_latent)

        if static_slots is None and feature_cache is not None:
            feature_cache.clear()

        if preallocate:
            assert output is not None and write_pos == num_latent_frames, (
                f"Wrote {write_pos} latent frames into a {num_latent_frames} frame output"
            )
            return output

        result_row = []
        for i, tile_latent in enumerate(row):
            if i > 0:
                tile_latent = self.blend_t(row[i - 1], tile_latent, blend_extent)
            result_row.append(tile_latent[:, :, :t_limit, :, :])

        return torch.cat(result_row, dim=2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.num_pad_frames > 0:
            x = F.pad(x, (0, 0, 0, 0, self.cfg.num_pad_frames, 0), mode="replicate")
        if self.cfg.spatial_tile_size is not None:
            raise NotImplementedError("Spatial tiling is not supported for DCAEV")
        elif self.cfg.encode_temporal_tile_size is not None:
            x = self.temporal_tiled_encode(x)
        else:
            x, _ = self.encoder(x)
        return x * self.cfg.scaling_factor

    def temporal_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        overlap_size = int(
            self.cfg.decode_temporal_tile_latent_size * (1 - self.cfg.decode_temporal_tile_overlap_factor)
        )
        blend_extent = int(self.cfg.decode_temporal_tile_size * self.cfg.decode_temporal_tile_overlap_factor)
        t_limit = self.cfg.decode_temporal_tile_size - blend_extent

        if self.cfg.use_feature_cache:
            feature_cache: list[torch.Tensor | None] | None = [None] * _count_causal_convs(self.decoder)
            feat_idx: list[int] | None = [0]
        else:
            feature_cache = None
            feat_idx = None

        tile_latent_size = self.cfg.decode_temporal_tile_latent_size
        expansion_factor = self.decoder.temporal_compression_ratio
        tile_starts = range(0, z.shape[2], overlap_size)
        preallocate = self.cfg.preallocate_tiled_output and blend_extent == 0
        output: torch.Tensor | None = None
        num_frames = 0
        row: list[torch.Tensor] = []
        if preallocate:
            num_frames = sum(
                min(min(tile_latent_size, z.shape[2] - start) * expansion_factor, t_limit) for start in tile_starts
            )
        write_pos = 0

        for start in tqdm(tile_starts, desc="Tiled Decode", disable=not self.cfg.verbose):
            tile = z[:, :, start : start + tile_latent_size, :, :]
            if feat_idx is not None:
                feat_idx[0] = 0
            decoded, _ = self.decoder(tile, feature_cache=feature_cache, feat_idx=feat_idx)
            if preallocate:
                decoded = decoded[:, :, :t_limit, :, :]
                if output is None:
                    output = _empty_like_with_num_frames(decoded, num_frames)
                output[:, :, write_pos : write_pos + decoded.shape[2]].copy_(decoded)
                write_pos += decoded.shape[2]
            else:
                row.append(decoded.clone())

        if preallocate:
            assert output is not None and write_pos == num_frames, (
                f"Wrote {write_pos} frames into a {num_frames} frame output"
            )
            return output

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :])

        return torch.cat(result_row, dim=2)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z / self.cfg.scaling_factor
        if self.cfg.spatial_tile_size is not None:
            raise NotImplementedError("Spatial tiling is not supported for DCAEV")
        elif self.cfg.decode_temporal_tile_size is not None:
            z = self.temporal_tiled_decode(z)
        else:
            z, _ = self.decoder(z)
        if self.cfg.num_pad_frames > 0:
            z = z[:, :, self.cfg.num_pad_frames :, :, :]
        return z

    @torch.no_grad()
    def reconstruct_image(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        x: (B, 3, H, W) [-1, 1]
        """
        x = x.unsqueeze(2)
        if self.cfg.num_pad_frames == 0:
            x = x.repeat(1, 1, self.decoder.temporal_compression_ratio, 1, 1)
        elif self.cfg.num_pad_frames == self.decoder.temporal_compression_ratio - 1:
            pass
        else:
            raise ValueError(
                f"num_pad_frames {self.cfg.num_pad_frames} and temporal_compression_ratio {self.decoder.temporal_compression_ratio} is not supported for image reconsruction"
            )
        z = self.encode(x)
        y = self.decode(z)
        return y[:, :, 0], {"latent": z}


def dc_ae_v_f32t4_encoder_causal_decoder_chunk_causal_4(
    name: str,
    pretrained_path: Optional[str],
) -> DCAEVConfig:
    if name in [
        "dcae4x32x32_c64_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.1",
    ]:
        latent_channels, num_pad_frames, temporal_remainder, scaling_factor = 64, 7, 1, 0.7103
        encoder_width_list = [128, 256, 512, 512, 1024, 1024, 1024]
    elif name in [
        "dcae4x32x32_c32_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_11_v0.1",
    ]:
        latent_channels, num_pad_frames, temporal_remainder, scaling_factor = 32, 11, 1, 0.6774
        encoder_width_list = [128, 256, 512, 512, 1024, 1024, 1024]
    elif name in [
        "dcae4x32x32_c64_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.2",
    ]:
        latent_channels, num_pad_frames, temporal_remainder, scaling_factor = 64, 7, 1, 0.5704
        encoder_width_list = [0, 64, 128, 512, 1024, 1024, 1024]
    elif name in [
        "dcae4x32x32_c96_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.2_lcr",
    ]:
        latent_channels, num_pad_frames, temporal_remainder, scaling_factor = 96, 7, 1, 0.4766
        encoder_width_list = [0, 64, 128, 512, 1024, 1024, 1024]
    elif name in [
        "dcae4x32x32_c128_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.2_lcr",
    ]:
        latent_channels, num_pad_frames, temporal_remainder, scaling_factor = 128, 7, 1, 0.5637
        encoder_width_list = [0, 64, 128, 512, 1024, 1024, 1024]
    else:
        raise ValueError(f"model {name} is not supported")

    def causal_downsample(sf, tf):
        return SampleBlockConfig(
            block_name="CausalConvPixelUnshuffle",
            spatial_factor=sf,
            temporal_factor=tf,
            spatial_kernel_size=3,
            temporal_kernel_size=3,
        )

    def chunk_causal_upsample(sf, tf, cl):
        return SampleBlockConfig(
            block_name="ChunkCausalConvPixelShuffle",
            spatial_factor=sf,
            temporal_factor=tf,
            spatial_kernel_size=3,
            temporal_kernel_size=3,
            causal_chunk_length=cl,
        )

    cfg = DCAEVConfig(
        latent_channels=latent_channels,
        use_feature_cache=True,
        encode_temporal_tile_size={"256": 16, "480": 16, "720": 16, "768": 16},
        encode_temporal_tile_latent_size={"256": 4, "480": 4, "720": 4, "768": 4},
        tile_buckets={"256": [8, 12], "480": [8, 12], "720": [8, 12], "768": [8, 12]},
        decode_temporal_tile_size=16,
        decode_temporal_tile_latent_size=4,
        num_pad_frames=num_pad_frames,
        temporal_remainder=temporal_remainder,
        scaling_factor=scaling_factor,
        pretrained_source="dc-ae-v-fsdp",
        pretrained_path=pretrained_path,
        encoder=DCAEVEncoderConfig(
            in_channels=3,
            latent_channels=latent_channels,
            zero_out=False,
            project_in_block_type=SampleBlockConfig(
                block_name="CausalConvPixelUnshuffle",
                spatial_factor=2,
                temporal_factor=1,
                spatial_kernel_size=3,
                temporal_kernel_size=3,
            ),
            depth_list=(0, 5, 10, 4, 4, 4, 4),
            width_list=tuple(encoder_width_list),
            block_type=BlockConfig(
                block_name="CausalResBlock3d",
                spatial_kernel_size=3,
                temporal_kernel_size=3,
            ),
            downsample_block_type=[
                causal_downsample(2, 1),
                causal_downsample(2, 1),
                causal_downsample(2, 1),
                causal_downsample(2, 1),
                causal_downsample(2, 1),
                causal_downsample(1, 4),
            ],
            project_out_block_type=BlockConfig(
                block_name="CausalConvLayer3d",
                spatial_kernel_size=3,
                temporal_kernel_size=3,
            ),
        ),
        decoder=DCAEVDecoderConfig(
            in_channels=3,
            latent_channels=latent_channels,
            zero_out=False,
            depth_list=(0, 5, 10, 4, 4, 4, 4),
            width_list=(128, 256, 512, 512, 1024, 1024, 1024),
            project_in_block_type=BlockConfig(
                block_name="ChunkCausalConvLayer3d",
                spatial_kernel_size=3,
                temporal_kernel_size=3,
                causal_chunk_length=1,
            ),
            block_type=[
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=4,
                ),
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=4,
                ),
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=4,
                ),
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=4,
                ),
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=4,
                ),
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=4,
                ),
                BlockConfig(
                    block_name="ChunkCausalResBlock3d",
                    spatial_kernel_size=3,
                    temporal_kernel_size=3,
                    causal_chunk_length=1,
                ),
            ],
            upsample_block_type=[
                chunk_causal_upsample(2, 1, 4),
                chunk_causal_upsample(2, 1, 4),
                chunk_causal_upsample(2, 1, 4),
                chunk_causal_upsample(2, 1, 4),
                chunk_causal_upsample(2, 1, 4),
                chunk_causal_upsample(1, 4, 1),
            ],
            project_out_block_type=SampleBlockConfig(
                block_name="ChunkCausalConvPixelShuffle",
                spatial_factor=2,
                temporal_factor=1,
                spatial_kernel_size=3,
                temporal_kernel_size=3,
                causal_chunk_length=4,
            ),
        ),
    )
    return cfg
