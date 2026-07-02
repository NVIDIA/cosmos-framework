# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from collections.abc import Sequence

import torch

from cosmos_framework.utils import log
from cosmos_framework.utils.distributed import get_rank, sync_model_states
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.data.generator.utils import VIDEO_RES_SIZE_INFO
from cosmos_framework.model.generator.tokenizers.dc_ae.dc_ae_v import (
    DCAEV,
    DCAEVConfig,
    dc_ae_v_f32t4_encoder_causal_decoder_chunk_causal_4,
)
from cosmos_framework.model.generator.tokenizers.interface import VideoTokenizerInterface

DEFAULT_MODEL_NAME = "dcae4x32x32_c64_t120_256p_fps_all_encoder_causal_decoder_chunk_causal_4_nogan_cosmos_pad_7_v0.2"


class DCAE4x32x32Interface(VideoTokenizerInterface):
    def __init__(
        self,
        bucket_name: str = "",
        object_store_credential_path_pretrained: str = "",
        vae_path: str = "",
        chunk_duration: int = 16,
        model_name: str = DEFAULT_MODEL_NAME,
        spatial_compression_factor: int = 32,
        temporal_compression_factor: int = 4,
        encode_chunk_frames: int = 128,  # Placeholder
        encode_bucket_multiple: int = 2,  # Placeholder
        device: str = "cuda",
        compilable: bool = True,
        causal: bool = True,
    ):
        self._causal = causal
        assert self._causal, "DCAE4x32x32Interface is a causal tokenizer; causal must be True."
        vae_path_full = f"s3://{bucket_name}/{vae_path}"
        self._spatial_compression_factor = spatial_compression_factor
        self._temporal_compression_factor = temporal_compression_factor
        self.chunk_duration = chunk_duration
        self.model_name = model_name
        self.resolutions = None

        # Build config (without pretrained_path so DCAEV doesn't try to load itself).
        cfg: DCAEVConfig = dc_ae_v_f32t4_encoder_causal_decoder_chunk_causal_4(model_name, pretrained_path=None)
        cfg.compilable = compilable
        cfg.encode_temporal_tile_size = chunk_duration

        # Instantiate model on meta device to avoid double allocation.
        with torch.device("meta"):
            self.model = DCAEV(cfg)

        # Load checkpoint from S3 on rank 0 only, then broadcast.
        if get_rank() == 0:
            backend_args = {
                "backend": "s3",
                "s3_credential_path": object_store_credential_path_pretrained,
            }
            checkpoint = easy_io.load(vae_path_full, backend_args=backend_args, map_location=device)
            log.info(f"loading {vae_path_full}")

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
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = "inductor",
    ) -> None:
        self.model.encoder = torch.compile(self.model.encoder, fullgraph=True, mode=mode)
        self.is_compiled = True

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

        # Run warmup resolutions
        if aspect_ratio is None:
            aspect_ratios = list(VIDEO_RES_SIZE_INFO["256"].keys())
        else:
            if isinstance(aspect_ratio, str):
                if aspect_ratio not in VIDEO_RES_SIZE_INFO["256"]:
                    raise ValueError(f"Aspect ratio {aspect_ratio} not found in predefined aspect ratios")
                aspect_ratios = [aspect_ratio]
            else:
                raise ValueError(f"Aspect ratio {aspect_ratio} must be a string")

        self.resolutions = warmup_resolutions
        self.aspect_ratios = aspect_ratios

        T = self.chunk_duration - self.model.cfg.num_pad_frames
        for resolution in warmup_resolutions:
            for aspect_ratio in aspect_ratios:
                H, W = VIDEO_RES_SIZE_INFO[resolution][aspect_ratio]
                log.info(f"Warming up {resolution} {aspect_ratio}")
                for _ in range(2):
                    self.model.encode(torch.randn(1, 3, T, H, W).cuda().to(torch.bfloat16))

    @property
    def dtype(self):
        return self.model.dtype

    def reset_dtype(self):
        pass

    @torch.inference_mode()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        if self.resolutions is not None:
            for resolution in self.resolutions:
                if tuple(state.shape[3:]) in VIDEO_RES_SIZE_INFO[resolution].values():
                    break
            else:
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

    def get_pixel_num_frames(self, num_latent_frames: int, **kwargs) -> int:
        return num_latent_frames * self._temporal_compression_factor - self.model.cfg.num_pad_frames

    @property
    def spatial_compression_factor(self):
        return self._spatial_compression_factor

    @property
    def temporal_compression_factor(self):
        return self._temporal_compression_factor

    @property
    def pixel_chunk_duration(self):
        return self.chunk_duration

    @property
    def latent_chunk_duration(self):
        return self.get_latent_num_frames(self.chunk_duration)

    @property
    def latent_ch(self):
        return self.model.cfg.latent_channels

    @property
    def spatial_resolution(self):
        return 512

    @property
    def name(self):
        return "dc_ae_4x32x32_tokenizer"
