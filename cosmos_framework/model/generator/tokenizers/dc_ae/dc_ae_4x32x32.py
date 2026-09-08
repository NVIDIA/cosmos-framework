# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import itertools
from collections.abc import Mapping, Sequence
from typing import Optional

import torch

from cosmos_framework.utils import log
from cosmos_framework.utils.distributed import get_rank, sync_model_states
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.model.generator.tokenizers.dc_ae.dc_ae_v import (
    DCAEV,
    DCAEVConfig,
    _restore_static_cache_slots,
    _select_temporal_tile_size,
    dc_ae_v_f32t4_encoder_causal_decoder_chunk_causal_4,
)
from cosmos_framework.model.generator.tokenizers.interface import VideoTokenizerInterface
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution

DEFAULT_MODEL_NAME = "dcae4x32x32_c64_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.2"


def _get_warmup_spatial_shapes(
    warmup_resolutions: Sequence[str],
    aspect_ratio: str | None,
    spatial_alignment: int = 32,
) -> dict[str, dict[str, tuple[int, int]]]:
    """Return spatially aligned CUDA-graph shapes as ``(height, width)``."""
    if spatial_alignment <= 0:
        raise ValueError(f"spatial_alignment must be positive, got {spatial_alignment}")

    spatial_shapes: dict[str, dict[str, tuple[int, int]]] = {}
    for resolution in warmup_resolutions:
        resolution_sizes = VIDEO_RES_SIZE_INFO[resolution]
        if aspect_ratio is not None:
            if aspect_ratio not in resolution_sizes:
                raise ValueError(f"Aspect ratio {aspect_ratio} not found for resolution {resolution}")
            resolution_sizes = {aspect_ratio: resolution_sizes[aspect_ratio]}
        spatial_shapes[resolution] = {
            ratio: (
                ((height + spatial_alignment - 1) // spatial_alignment) * spatial_alignment,
                ((width + spatial_alignment - 1) // spatial_alignment) * spatial_alignment,
            )
            for ratio, (width, height) in resolution_sizes.items()
        }
    return spatial_shapes


class DCAE4x32x32Interface(VideoTokenizerInterface):
    def __init__(
        self,
        bucket_name: str = "",
        object_store_credential_path_pretrained: str = "",
        vae_path: str = "",
        checkpoint_path: str | None = None,
        chunk_duration: int = 16,
        model_name: str = DEFAULT_MODEL_NAME,
        spatial_compression_factor: int = 32,
        temporal_compression_factor: int = 4,
        encode_chunk_frames: int | Mapping[str, int] = 16,
        tile_buckets: Mapping[str, Sequence[int]] | None = None,
        no_tile_padding: bool = False,
        encode_bucket_multiple: int = 2,  # Placeholder
        device: str = "cuda",
        compilable: bool = True,
        causal: bool = True,
        encode_exact_durations: Optional[list[int]] = None,
        load_weights: bool = True,
    ) -> None:
        self._causal = causal
        assert self._causal, "DCAE4x32x32Interface is a causal tokenizer; causal must be True."
        assert encode_exact_durations is None, "DCAE4x32x32Interface does not support encode_exact_durations."
        if load_weights:
            if checkpoint_path is not None and (bucket_name or vae_path):
                raise ValueError("Pass either checkpoint_path or the S3 bucket_name/vae_path pair, not both.")
            if checkpoint_path is None and (not bucket_name or not vae_path):
                raise ValueError("Pass checkpoint_path or both bucket_name and vae_path.")
        self._spatial_compression_factor = spatial_compression_factor
        self._temporal_compression_factor = temporal_compression_factor
        self.chunk_duration = chunk_duration
        self.model_name = model_name
        self.resolutions = None
        if isinstance(encode_chunk_frames, int):
            encode_chunk_frames = {
                "256": encode_chunk_frames * 3,
                "480": encode_chunk_frames,
                "704": encode_chunk_frames,
                "720": encode_chunk_frames,
                "768": encode_chunk_frames,
            }
        if tile_buckets is None:
            tile_buckets = {
                "256": [4, 8, 12, 24, 36],
                "480": [4, 8, 12],
                "704": [4, 8, 12],
                "720": [4, 8, 12],
                "768": [4, 8, 12],
            }
        self.encode_chunk_frames = encode_chunk_frames
        self.tile_buckets = tile_buckets

        # Build config (without pretrained_path so DCAEV doesn't try to load itself).
        cfg: DCAEVConfig = dc_ae_v_f32t4_encoder_causal_decoder_chunk_causal_4(model_name, pretrained_path=None)
        cfg.compilable = compilable
        cfg.encode_temporal_tile_size = encode_chunk_frames
        cfg.encode_temporal_tile_latent_size = {
            resolution: tile_size // temporal_compression_factor
            for resolution, tile_size in encode_chunk_frames.items()
        }
        cfg.tile_buckets = tile_buckets
        cfg.no_tile_padding = no_tile_padding

        # Instantiate model on meta device to avoid double allocation.
        with torch.device("meta"):
            self.model = DCAEV(cfg)

        # Load checkpoint from S3 on rank 0 only, then broadcast.
        if not load_weights:
            # Runtime benchmarks only: the weight values do not affect how long
            # encode and decode take. 'to_empty' leaves uninitialized memory that
            # can hold NaNs, so seed the tensors instead of using them as-is.
            log.warning("Skipping checkpoint loading; the tokenizer runs on random weights.")
            self.model.to_empty(device=device)
            with torch.no_grad():
                for tensor in itertools.chain(self.model.parameters(), self.model.buffers()):
                    if tensor.is_floating_point():
                        tensor.normal_(std=0.02)
                    else:
                        tensor.zero_()
        elif get_rank() == 0:
            checkpoint_source = checkpoint_path or f"s3://{bucket_name}/{vae_path}"
            if checkpoint_path is not None:
                checkpoint = easy_io.load(checkpoint_path, map_location=device)
            else:
                backend_args = {
                    "backend": "s3",
                    "s3_credential_path": object_store_credential_path_pretrained,
                }
                checkpoint = easy_io.load(checkpoint_source, backend_args=backend_args, map_location=device)
            log.info(f"loading {checkpoint_source}")

            self.model.load_state_dict(checkpoint["model_state_dict"], assign=True)
        else:
            self.model.to_empty(device=device)

        self.model.eval().requires_grad_(False)
        self.model.to(dtype=torch.bfloat16)

        sync_model_states(self.model)
        self.model.encoder = self.model.encoder.to(memory_format=torch.channels_last_3d)
        self.is_compiled = False
        self.use_streaming_encode = False

    def compile_encode_for_cudagraphs(
        self,
        *,
        mode: str | None = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = "inductor",
    ) -> None:
        """Compile the dynamic non-CUDA-graph fallback encoder."""
        del dynamic
        self.model.clear_compiled_encoders()
        fallback_encoder = torch.compile(
            self.model.encoder,
            fullgraph=fullgraph,
            mode=None,
            dynamic=True,
            backend=backend,
        )
        self.model.set_fallback_compiled_encoder(fallback_encoder)
        self.is_compiled = True

    def _image_tile_size(self, encode_resolution: str) -> int | None:
        """Tile size a single image is encoded at, or None if there is no such tile.

        An image is one frame padded to ``num_pad_frames + 1``, which always fits in one
        tile and therefore runs without a feature cache. Only this one bucket gets a
        cacheless executable: in reality, every video clip spans several tiles and keeps its feature cache,
        so compiling the cacheless variant for the other buckets would capture CUDA
        graphs that never run.
        """
        if not self.model.may_skip_feature_cache():
            return None
        num_image_frames = 1 + self.model.cfg.num_pad_frames
        tile_size = self.encode_chunk_frames[encode_resolution]
        if num_image_frames > tile_size:
            return None
        return _select_temporal_tile_size(num_image_frames, tile_size, self.tile_buckets[encode_resolution])

    def _warmup_encoder(
        self,
        encoder: torch.nn.Module,
        encoder_shape: tuple[int, int, int, int, int],
        num_warmups: int = 2,
        *,
        with_feature_cache: bool = True,
    ) -> None:
        """Warm one exact-shape encoder executable without invoking the full VAE encode path.

        Whether a feature cache is passed changes what dynamo traces, so a shape that is
        encoded both as part of a longer clip and on its own needs warming both ways.
        """
        batch_size, channels, tile_size, height, width = encoder_shape
        tile = torch.randn(
            batch_size,
            channels,
            tile_size,
            height,
            width,
            device="cuda",
            dtype=torch.bfloat16,
        )  # [B,C,T,H,W]

        for _ in range(num_warmups):
            if with_feature_cache:
                # Take the cache from the model so the buffer addresses recorded into the
                # CUDA graph here are the same ones the encode path passes at runtime.
                feature_cache, static_slots = self.model.prepare_encoder_feature_cache(
                    batch_size=batch_size,
                    height=height,
                    width=width,
                    dtype=tile.dtype,
                    device=tile.device,
                )  # Per-layer [B,C,T,H,W]
                assert feature_cache is not None, "Cached warmup needs cfg.use_feature_cache"
            else:
                feature_cache, static_slots = None, None
            feat_idx = None if feature_cache is None else [0]
            warmup_tile = tile.clone()  # [B,C,T,H,W]
            encoded, _ = encoder(
                warmup_tile,
                feature_cache=feature_cache,
                feat_idx=feat_idx,
            )  # [B,C_latent,T_latent,H_latent,W_latent]
            if static_slots is not None:
                assert feature_cache is not None
                _restore_static_cache_slots(feature_cache, static_slots)
            del encoded

    @torch.inference_mode()
    def compile_encode(
        self,
        warmup_resolutions: Sequence[str],
        output_dir: str | None = None,
        aspect_ratio: str | None = None,
        backend: str | None = "inductor",
        mode: str | None = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
    ) -> None:
        """Compile the encode function for the given resolutions."""
        if self.is_compiled:
            log.warning("Tokenizer is already compiled, skipping compilation.")
            return

        if backend is None:
            raise ValueError("backend must be provided")

        self.compile_encode_for_cudagraphs(mode=mode, fullgraph=fullgraph, dynamic=dynamic, backend=backend)

        # Run warmup resolutions. VIDEO_RES_SIZE_INFO stores (width, height), while
        # encoder tensors and compiled shape keys use (height, width).
        warmup_spatial_shapes = _get_warmup_spatial_shapes(
            warmup_resolutions,
            aspect_ratio,
            spatial_alignment=self._spatial_compression_factor,
        )

        self.resolutions = warmup_resolutions
        self.aspect_ratios = list(
            dict.fromkeys(ratio for resolution_shapes in warmup_spatial_shapes.values() for ratio in resolution_shapes)
        )

        # Size the feature cache pool for every shape before the first capture: growing it
        # later moves the buffer addresses that the CUDA graphs were recorded against.
        self.model.reserve_encoder_feature_cache(
            [
                (1, height, width)
                for resolution_shapes in warmup_spatial_shapes.values()
                for height, width in resolution_shapes.values()
            ],
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
        )

        for resolution in warmup_resolutions:
            for aspect_ratio, (height, width) in warmup_spatial_shapes[resolution].items():
                log.info(f"Warming up {resolution} {aspect_ratio}")
                encode_resolution = get_vision_data_resolution((height, width))
                tile_sizes = sorted(
                    {
                        self.encode_chunk_frames[encode_resolution],
                        *self.tile_buckets[encode_resolution],
                    }
                )
                image_tile_size = self._image_tile_size(encode_resolution)
                for tile_size in tile_sizes:
                    encoder_shape = (1, 3, tile_size, height, width)
                    # Clips spanning several tiles run the cached variant; only the image
                    # bucket additionally needs the cacheless one.
                    variants = [self.model.cfg.use_feature_cache]
                    if tile_size == image_tile_size:
                        variants.append(False)
                    for with_feature_cache in variants:
                        cudagraph_encoder = self.model.get_cudagraph_encoder(
                            encoder_shape, with_feature_cache=with_feature_cache
                        )
                        if cudagraph_encoder is None:
                            cudagraph_encoder = torch.compile(
                                self.model.encoder,
                                fullgraph=fullgraph,
                                mode=mode,
                                dynamic=False,
                                backend=backend,
                            )
                            self.model.set_cudagraph_encoder(
                                encoder_shape, cudagraph_encoder, with_feature_cache=with_feature_cache
                            )
                        self._warmup_encoder(cudagraph_encoder, encoder_shape, with_feature_cache=with_feature_cache)

    @property
    def dtype(self) -> torch.dtype:
        return self.model.dtype

    def reset_dtype(self) -> None:
        pass

    @torch.inference_mode()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        if self.resolutions is not None:
            allowed_spatial_shapes = _get_warmup_spatial_shapes(
                self.resolutions,
                aspect_ratio=None,
                spatial_alignment=self._spatial_compression_factor,
            )
            state_spatial_shape = (state.shape[3], state.shape[4])
            if not any(
                state_spatial_shape in resolution_shapes.values()
                for resolution_shapes in allowed_spatial_shapes.values()
            ):
                raise ValueError(f"State shape {state.shape[2:]} is not in {self.resolutions}")
        in_dtype = state.dtype
        tcf = self._temporal_compression_factor
        # Add padding to the sequence length to make it divisible by
        # the temporal compression factor after num_pad_frames padding.
        seq_len = state.shape[2] + self.model.cfg.num_pad_frames
        if seq_len % tcf != 0:
            raise ValueError(f"Sequence length {seq_len} is not divisible by temporal compression factor {tcf}")
        return self.model.encode(state.to(torch.bfloat16)).to(in_dtype)

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        in_dtype = latent.dtype
        return self.model.decode(latent.to(torch.bfloat16)).to(in_dtype)

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        return (num_pixel_frames + self.model.cfg.num_pad_frames) // self._temporal_compression_factor

    def get_pixel_num_frames(self, num_latent_frames: int, **kwargs: object) -> int:
        return num_latent_frames * self._temporal_compression_factor - self.model.cfg.num_pad_frames

    @property
    def spatial_compression_factor(self) -> int:
        return self._spatial_compression_factor

    @property
    def temporal_compression_factor(self) -> int:
        return self._temporal_compression_factor

    @property
    def pixel_chunk_duration(self) -> int:
        return self.chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self) -> int:
        return self.model.cfg.latent_channels

    @property
    def spatial_resolution(self) -> int:
        return 512

    @property
    def name(self) -> str:
        return "dc_ae_4x32x32_tokenizer"
