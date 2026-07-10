# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""UniAE S3 tokenizer wrapper for diffusion training (4x16x16 compression).

Wraps the UniAE sparse autoencoder with DenseAutoencoderRuntime (batched backend)
to provide a VideoTokenizerInterface compatible with diffusion model training.

Usage:
    from cosmos_framework.model.generator.tokenizers.uniae.noncausal_4x16x16 import UniAEVAE

    vae = UniAEVAE(
        vae_pth="s3://bucket0/pretrained/tokenizers/video/cosmos/...",
        object_store_credential_path_pretrained="credentials/gcp_checkpoint.secret",
    )
    latents = vae.encode(video)   # [B, 3, T, H, W] -> [B, 48, ceil(T/4), H//16, W//16]
    recon = vae.decode(latents)   # [B, 48, T_p, H//16, W//16] -> [B, 3, 4*T_p, H, W]
"""

import os
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import torch

from cosmos_framework.utils import log
from cosmos_framework.utils.distributed import get_rank, sync_model_states
from cosmos_framework.utils.easy_io import easy_io
from cosmos_framework.model.generator.tokenizers.interface import VideoTokenizerInterface
from cosmos_framework.model.generator.tokenizers.uniae.frame_math import (
    get_uniae_latent_num_frames,
    get_uniae_latent_temporal_positions,
    get_uniae_pixel_num_frames,
    normalize_resolution_int_mapping,
)
from cosmos_framework.utils.generator.data_utils import get_vision_data_resolution
from cosmos_framework.model.tokenizer.checkpoint_io import (
    DCP_MODEL_LOAD_INFO_KEY,
    DCPModelLoadInfo,
    load_dcp_model_checkpoint,
    load_torch_checkpoint,
)
from cosmos_framework.model.tokenizer.models.architecture import get_siglip2_so400m_common_arch
from cosmos_framework.model.tokenizer.models.dense_runtime import DenseAutoencoderRuntime
from cosmos_framework.model.tokenizer.models.sparse_autoencoder import AutoencoderKL

_S3_ARCH = dict(
    **get_siglip2_so400m_common_arch(),
    use_quantizer=False,
    use_dual_latent=False,
    use_text_alignment=False,
    use_post_text_alignment=False,
)

_IGNORED_LEGACY_CHECKPOINT_PREFIXES = (
    "ema.",
    "loss_fn.",
    "text_decoder_wrapper.",
    "tokenizer.",
)
_LEGACY_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt")


def _is_legacy_checkpoint_file(checkpoint_path: str) -> bool:
    """Distinguish torch checkpoint files from DCP component directories."""
    is_remote = checkpoint_path.startswith("s3://")
    if not is_remote and os.path.isdir(checkpoint_path):
        return False
    return checkpoint_path.lower().endswith(_LEGACY_CHECKPOINT_SUFFIXES) or (
        not is_remote and os.path.isfile(checkpoint_path)
    )


def _derive_legacy_latent_norm_path(checkpoint_path: str) -> str:
    """Derive the latent-stat sidecar for a legacy checkpoint file."""
    checkpoint_stem, checkpoint_suffix = os.path.splitext(checkpoint_path)
    if checkpoint_suffix.lower() in _LEGACY_CHECKPOINT_SUFFIXES:
        return checkpoint_stem + "_latent_norm.pt"
    return checkpoint_path + "_latent_norm.pt"


def _coerce_latent_norm_vector(
    values: object,
    *,
    field_name: str,
    z_dim: int,
    source: str,
    strictly_positive: bool = False,
) -> torch.Tensor:
    """Convert one tensor/JSON vector to validated CPU float64 statistics."""
    if isinstance(values, torch.Tensor):
        if values.dtype == torch.bool:
            raise ValueError(f"Latent-normalization {field_name} in {source} must contain real numbers, not booleans.")
        if values.is_complex():
            raise ValueError(
                f"Latent-normalization {field_name} in {source} must contain real numbers, not complex values."
            )
        vector = values.detach().to(device="cpu", dtype=torch.float64)  # [C_candidate]
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
            raise ValueError(f"Latent-normalization {field_name} in {source} must contain only real numbers.")
        vector = torch.tensor(list(values), dtype=torch.float64)  # [C_candidate]
    else:
        raise TypeError(
            f"Latent-normalization {field_name} in {source} must be a tensor or numeric sequence, "
            f"got {type(values).__name__}."
        )

    if vector.ndim != 1 or vector.shape[0] != z_dim:
        raise ValueError(
            f"Latent-normalization {field_name} in {source} must have shape ({z_dim},), got {tuple(vector.shape)}."
        )
    finite_mask = torch.isfinite(vector)  # [C]
    if not bool(finite_mask.all().item()):
        raise ValueError(f"Latent-normalization {field_name} in {source} contains non-finite values.")
    if strictly_positive:
        positive_mask = vector > 0  # [C]
        if not bool(positive_mask.all().item()):
            raise ValueError(f"Latent-normalization {field_name} in {source} must be strictly positive.")
    return vector


def _load_latent_norm_stats(
    norm_path: str,
    *,
    backend_args: dict[str, str] | None,
    z_dim: int,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load latent statistics and return validated runtime mean and inverse standard deviation."""
    if norm_path.lower().endswith(".json"):
        norm_stats = easy_io.load(norm_path, backend_args=backend_args)
    else:
        norm_stats = easy_io.load(norm_path, backend_args=backend_args, map_location="cpu", weights_only=True)
    if not isinstance(norm_stats, Mapping):
        raise TypeError(
            f"Latent-normalization sidecar {norm_path} must contain a mapping, got {type(norm_stats).__name__}."
        )

    stats_z_dim = norm_stats.get("z_dim")
    if stats_z_dim is not None and (isinstance(stats_z_dim, bool) or not isinstance(stats_z_dim, int)):
        raise ValueError(f"Latent-normalization z_dim in {norm_path} must be an integer, got {stats_z_dim!r}.")
    if isinstance(stats_z_dim, int) and stats_z_dim != z_dim:
        raise ValueError(f"Latent-normalization z_dim in {norm_path} is {stats_z_dim}, expected {z_dim}.")
    if "mean" not in norm_stats or "std" not in norm_stats:
        raise ValueError(f"Latent-normalization sidecar {norm_path} must contain mean and std entries.")

    mean_cpu = _coerce_latent_norm_vector(
        norm_stats["mean"],
        field_name="mean",
        z_dim=z_dim,
        source=norm_path,
    )  # [C]
    std_cpu = _coerce_latent_norm_vector(
        norm_stats["std"],
        field_name="std",
        z_dim=z_dim,
        source=norm_path,
        strictly_positive=True,
    )  # [C]
    mean = mean_cpu.to(dtype=dtype, device=device)  # [C]
    std = std_cpu.to(dtype=dtype, device=device)  # [C]
    mean_finite_mask = torch.isfinite(mean)  # [C]
    std_finite_positive_mask = torch.isfinite(std) & (std > 0)  # [C]
    if not bool(mean_finite_mask.all().item()):
        raise ValueError(f"Latent-normalization mean in {norm_path} is non-finite after conversion to {dtype}.")
    if not bool(std_finite_positive_mask.all().item()):
        raise ValueError(
            f"Latent-normalization std in {norm_path} must remain finite and positive after conversion to {dtype}."
        )
    inv_std = std.reciprocal()  # [C]
    inv_std_finite_mask = torch.isfinite(inv_std)  # [C]
    if not bool(inv_std_finite_mask.all().item()):
        raise ValueError(f"Latent-normalization inverse std in {norm_path} is non-finite after conversion to {dtype}.")
    return mean, inv_std


def _extract_visual_tokenizer_state_dict(model_state: Mapping[str, Any]) -> dict[str, Any]:
    """Extract bare AutoencoderKL state from a full TokenizerModel checkpoint."""
    network_state = {
        key.removeprefix("network."): value for key, value in model_state.items() if key.startswith("network.")
    }
    return network_state if network_state else dict(model_state)


def _get_dcp_unexpected_visual_keys(load_info: DCPModelLoadInfo) -> list[str]:
    """Return unexpected keys owned by the tokenizer network from a DCP report."""
    return [key.removeprefix("network.") for key in load_info.unexpected_checkpoint_keys if key.startswith("network.")]


def _validate_uniae_checkpoint_keys(missing: list[str], unexpected: list[str], checkpoint_path: str) -> None:
    """Reject promoted checkpoints that do not fully cover the visual tokenizer."""
    if missing:
        raise RuntimeError(
            f"UniAE checkpoint {checkpoint_path} is missing {len(missing)} visual model keys (sample: {missing[:5]})."
        )
    invalid_unexpected = [key for key in unexpected if not key.startswith(_IGNORED_LEGACY_CHECKPOINT_PREFIXES)]
    if invalid_unexpected:
        raise RuntimeError(
            f"UniAE checkpoint {checkpoint_path} has {len(invalid_unexpected)} unexpected visual model keys "
            f"(sample: {invalid_unexpected[:5]})."
        )


class UniAEVAE:
    """UniAE S3 VAE wrapper for diffusion training.

    Loads the UniAE sparse autoencoder checkpoint, wraps it with
    DenseAutoencoderRuntime (batched backend for compile-friendly inference),
    and provides encode/decode in the standard [B, C, T, H, W] format.

    Latents are normalized per-channel using statistics computed from 10K images
    and 10K videos: ``normalized = (latent - mean) / std``.
    """

    def __init__(
        self,
        z_dim: int = 48,
        vae_pth: str = "",
        object_store_credential_path_pretrained: str = "",
        latent_norm_path: str = "",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        backend: str = "batched",
        pad_frames: int = 1,
        pixel_trim: bool = True,
        chunk_size: int | Mapping[str, int] = 16,
        encode_chunk_batch_size: int | Mapping[str, int] = 1,
    ) -> None:
        if torch.device(device).type == "meta":
            raise ValueError("UniAEVAE requires a concrete CPU or CUDA device; device='meta' is not supported.")
        chunk_size = normalize_resolution_int_mapping(chunk_size, name="chunk_size")
        if any(chunk_frames % 4 != 0 for chunk_frames in chunk_size.values()):
            raise ValueError("chunk_size values must be multiples of 4.")
        if any(chunk_frames <= 2 * pad_frames for chunk_frames in chunk_size.values()):
            raise ValueError(
                f"chunk_size values must be greater than 2 * pad_frames, got {chunk_size=} and {pad_frames=}."
            )
        encode_chunk_batch_size = normalize_resolution_int_mapping(
            encode_chunk_batch_size,
            name="encode_chunk_batch_size",
            default_keys=chunk_size.keys(),
            required_keys=chunk_size.keys(),
        )
        if any(batch_size < 1 for batch_size in encode_chunk_batch_size.values()):
            raise ValueError("encode_chunk_batch_size values must be >= 1.")
        self.chunk_size = chunk_size
        self.encode_chunk_batch_size = encode_chunk_batch_size
        self.dtype = dtype
        self.device = device
        self.z_dim = z_dim
        self._pad_frames = pad_frames
        self._pixel_trim = pixel_trim
        self._spatial_compression_factor = 16
        self._temporal_compression_factor = 4

        # Per-channel latent normalization stats — loaded from a file paired with the
        # tokenizer checkpoint: <ckpt_stem>_latent_norm.pt (same directory, same bucket).
        # Storing stats alongside the checkpoint prevents silent divergence when the
        # checkpoint is updated.
        if not vae_pth:
            raise ValueError("vae_pth must be provided to load latent normalization stats")
        vae_pth_str = str(vae_pth)
        is_legacy_checkpoint = _is_legacy_checkpoint_file(vae_pth_str)
        # Legacy files use a paired <checkpoint_stem>_latent_norm.pt sidecar.
        if latent_norm_path:
            norm_pth = latent_norm_path
        elif is_legacy_checkpoint:
            norm_pth = _derive_legacy_latent_norm_path(vae_pth_str)
        else:
            raise ValueError("latent_norm_path is required when vae_pth points to a DCP checkpoint directory.")
        if norm_pth.startswith("s3://"):
            norm_backend_args = {
                "backend": "s3",
                "s3_credential_path": object_store_credential_path_pretrained,
            }
        else:
            norm_backend_args = None
        mean, inv_std = _load_latent_norm_stats(
            norm_pth,
            backend_args=norm_backend_args,
            z_dim=z_dim,
            dtype=dtype,
            device=device,
        )  # [C],[C]
        self._latent_mean = mean.view(1, z_dim, 1, 1, 1)  # [1,C,1,1,1]
        self._latent_inv_std = inv_std.view(1, z_dim, 1, 1, 1)  # [1,C,1,1,1]

        # Construct the autoencoder on the requested concrete device.
        autoencoder = AutoencoderKL(
            **_S3_ARCH,
            latent_channels=z_dim,
            quantizer_feature_dim=z_dim,
        )
        autoencoder.eval()
        autoencoder.to(device=device, dtype=dtype)

        # Load checkpoint
        if vae_pth and get_rank() == 0:
            dcp_unexpected_keys: list[str] = []
            if is_legacy_checkpoint:
                if str(vae_pth).startswith("s3://"):
                    backend_args = {
                        "backend": "s3",
                        "s3_credential_path": object_store_credential_path_pretrained,
                    }
                else:
                    backend_args = None
                state_dict = load_torch_checkpoint(
                    vae_pth,
                    backend_args=backend_args,
                    map_location="cpu",
                )
                if "model" in state_dict:
                    model_state = state_dict["model"]
                elif "state_dict" in state_dict:
                    model_state = state_dict["state_dict"]
                else:
                    model_state = state_dict
                model_state = _extract_visual_tokenizer_state_dict(model_state)
            else:
                state_dict = load_dcp_model_checkpoint(
                    autoencoder,
                    vae_pth,
                    checkpoint_key_prefix="network.",
                    s3_credential=object_store_credential_path_pretrained,
                )
                model_state = state_dict["model"]
                dcp_load_info = state_dict[DCP_MODEL_LOAD_INFO_KEY]
                if not isinstance(dcp_load_info, DCPModelLoadInfo):
                    raise TypeError(f"Invalid DCP model load info: {type(dcp_load_info).__name__}")
                dcp_unexpected_keys = _get_dcp_unexpected_visual_keys(dcp_load_info)
            missing, unexpected = autoencoder.load_state_dict(model_state, strict=False)
            all_unexpected = [*unexpected, *dcp_unexpected_keys]
            _validate_uniae_checkpoint_keys(missing, all_unexpected, str(vae_pth))
            if all_unexpected:
                log.info(f"Ignored {len(all_unexpected)} non-visual checkpoint keys (e.g., {all_unexpected[:3]}).")
            log.info(f"Loaded checkpoint from {vae_pth}")
        elif vae_pth:
            autoencoder.to_empty(device=device)
        if vae_pth:
            sync_model_states(autoencoder)

        # Wrap with dense runtime for fast inference
        self.dense_runtime = DenseAutoencoderRuntime.from_autoencoder(
            autoencoder,
            backend=backend,
            pad_frames=self._pad_frames,
            pixel_trim=self._pixel_trim,
            # passing of min value makes sense in order to verify padding is not bigger than smallest chunk size
            chunk_size=min(chunk_size.values()),
        )
        self.dense_runtime.eval()

        # Freeze all parameters
        for param in self.dense_runtime.parameters():
            param.requires_grad = False

        log.info(
            f"UniAE loaded: {self.count_param() / 1e6:.1f}M params, backend={backend}, dtype={dtype}, device={device}"
        )

    def count_param(self) -> int:
        return sum(p.numel() for p in self.dense_runtime.parameters())

    @torch.inference_mode()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """Encode image or video to latent space.

        Boundary padding and latent trimming are handled by DenseAutoencoderRuntime
        via pad_frames and pixel_trim. Non-image inputs use UniAE's noncausal
        first-frame-alone chunking: the first source frame forms its own latent
        and the remaining frames are encoded in resolution-specific chunks with
        pad_frames replicated on both sides.

        Args:
            video: [B, 3, T, H, W] or [B, 3, H, W] (image) in range [-1, 1].
                   For videos, (T - 1) must either fill whole content chunks or
                   leave a tail whose frame count plus 2 * pad_frames is divisible
                   by the temporal compression factor.

        Returns:
            latent: [B, z_dim, ceil(T/4), H//16, W//16]
                    For single-image input, ceil(T/4) = 1.
        """
        # Handle image input: [B, C, H, W] -> [B, C, 1, H, W].
        # Do NOT expand here — pass 1 frame so encode_moments detects is_image=True
        # and handles the temporal padding internally without noncausal chunking.
        if video.ndim == 4:
            video = video.unsqueeze(2)

        B, C, T, H, W = video.shape

        res_key = get_vision_data_resolution((H, W))
        if res_key not in self.chunk_size:
            raise ValueError(
                f"Unsupported resolution key '{res_key}' for input shape ({H}, {W}). "
                f"Supported keys: {list(self.chunk_size.keys())}"
            )
        full_chunk_size = self.chunk_size[res_key]
        chunk_size = full_chunk_size - 2 * self._pad_frames
        encode_chunk_batch_size = self.encode_chunk_batch_size[res_key]
        # Convert to channels-last [B, T, H, W, C] for dense runtime
        video_cl = video.permute(0, 2, 3, 4, 1).contiguous().to(dtype=self.dtype)

        # Encode with UniAE's content chunk size; dense_runtime adds pad_frames at
        # noncausal chunk boundaries and trims boundary latents internally.
        # Returns [B, T_p, H_p, W_p, 2*z_dim].
        moments = self.dense_runtime.encode(
            video_cl,
            sample_posterior=False,
            chunk_raw_frames=chunk_size,
            encode_chunk_batch_size=encode_chunk_batch_size,
        )

        # Take mean for deterministic encoding; convert to [B, z_dim, T_p, H_p, W_p]
        mean, _ = moments.chunk(2, dim=-1)
        latent = mean.permute(0, 4, 1, 2, 3).contiguous()
        # Normalize per-channel: (z - mean) * inv_std
        latent = (latent - self._latent_mean) * self._latent_inv_std
        return latent

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to image or video.

        Args:
            latent: [B, z_dim, T_p, H_p, W_p]

        Returns:
            video: [B, 3, T, H, W] in range [-1, 1]
        """
        # Denormalize per-channel: z / inv_std + mean
        latent = latent / self._latent_inv_std + self._latent_mean
        # Convert to channels-last [B, T_p, H_p, W_p, z_dim]
        latent_cl = latent.permute(0, 2, 3, 4, 1).contiguous().to(dtype=self.dtype)

        # Use the resolution-specific encoder chunk size so each chunk is decoded
        # independently with correct boundary trimming.  Decoding all latents at once
        # would apply trim only at the outer edges, producing wrong pixel counts for
        # multi-chunk videos.  Derive resolution from latent spatial dims.
        _, _, H_p, W_p = latent.shape[1:]
        res_key = get_vision_data_resolution(
            (H_p * self._spatial_compression_factor, W_p * self._spatial_compression_factor)
        )
        if res_key not in self.chunk_size:
            raise ValueError(
                f"Unsupported resolution key '{res_key}' for latent shape ({H_p}, {W_p}). "
                f"Supported keys: {list(self.chunk_size.keys())}"
            )
        chunk_raw_frames = self.chunk_size[res_key]
        decoded = self.dense_runtime.decode(latent_cl, chunk_raw_frames=chunk_raw_frames)

        # Convert to [B, C, T, H, W] and clamp
        video = decoded.permute(0, 4, 1, 2, 3).contiguous()
        return video.clamp(-1, 1).float()

    def get_latent_num_frames(self, num_pixel_frames: int, resolution: str | None = None) -> int:
        return get_uniae_latent_num_frames(
            num_pixel_frames,
            self.chunk_size,
            pad_frames=self._pad_frames,
            temporal_compression_factor=self._temporal_compression_factor,
            resolution=resolution,
            missing_resolution_message=(
                f"resolution must be provided when UniAE uses mixed encode_chunk_frames; got chunk_size={self.chunk_size}."
            ),
            invalid_frame_message_prefix="UniAE frame count is not valid for noncausal chunking",
        )

    def get_pixel_num_frames(self, num_latent_frames: int, resolution: str | None = None) -> int:
        return get_uniae_pixel_num_frames(
            num_latent_frames,
            self.chunk_size,
            pad_frames=self._pad_frames,
            temporal_compression_factor=self._temporal_compression_factor,
            resolution=resolution,
            missing_resolution_message=(
                f"resolution must be provided when UniAE uses mixed encode_chunk_frames; got chunk_size={self.chunk_size}."
            ),
        )

    def get_latent_temporal_positions(
        self,
        num_pixel_frames: int,
        resolution: str | None = None,
        num_latent_frames: int | None = None,
    ) -> torch.Tensor:
        """Return UniAE latent temporal coordinates in source-frame / tcf units.

        UniAE keeps noncausal padded boundary latents. Those latents should not be
        assigned uniformly increasing temporal IDs, because each latent summarizes
        the right edge of its padded temporal patch.
        """
        positions = get_uniae_latent_temporal_positions(
            num_pixel_frames,
            self.chunk_size,
            pad_frames=self._pad_frames,
            temporal_compression_factor=self._temporal_compression_factor,
            resolution=resolution,
            missing_resolution_message=(
                f"resolution must be provided when UniAE uses mixed encode_chunk_frames; got chunk_size={self.chunk_size}."
            ),
            num_latent_frames=num_latent_frames,
        )
        return torch.tensor(positions, dtype=torch.float32)  # [T_latent]


class UniAEVAEInterface(VideoTokenizerInterface):
    """Full VideoTokenizerInterface wrapper for diffusion training config integration."""

    def __init__(
        self,
        bucket_name: str = "",
        object_store_credential_path_pretrained: str = "",
        vae_path: str = "",
        latent_norm_path: str = "",
        encode_chunk_frames: int | Mapping[str, int] = 16,
        encode_chunk_batch_size: int | Mapping[str, int] = 1,
        spatial_compression_factor: int = 16,
        temporal_compression_factor: int = 4,
        pad_frames: int = 1,
        pixel_trim: bool = True,
        backend: str = "batched_with_padding",
        causal: bool = False,
    ) -> None:
        if spatial_compression_factor != 16:
            raise ValueError(
                f"UniAEVAEInterface requires spatial_compression_factor=16, got {spatial_compression_factor}."
            )
        if temporal_compression_factor != 4:
            raise ValueError(
                f"UniAEVAEInterface requires temporal_compression_factor=4, got {temporal_compression_factor}."
            )
        super().__init__(object_store_credential_path_pretrained or None)
        self._causal = causal
        assert not self._causal, "UniAEVAEInterface is a non-causal tokenizer; causal must be False."
        self._spatial_compression_factor = spatial_compression_factor
        self._temporal_compression_factor = temporal_compression_factor
        encode_chunk_frames = normalize_resolution_int_mapping(encode_chunk_frames, name="encode_chunk_frames")
        if any(chunk_frames % temporal_compression_factor != 0 for chunk_frames in encode_chunk_frames.values()):
            raise ValueError("encode_chunk_frames values must be multiples of temporal_compression_factor.")
        if any(chunk_frames <= 2 * pad_frames for chunk_frames in encode_chunk_frames.values()):
            raise ValueError(
                f"encode_chunk_frames values must be greater than 2 * pad_frames, "
                f"got {encode_chunk_frames=} and {pad_frames=}."
            )
        self.encode_chunk_frames = encode_chunk_frames
        encode_chunk_batch_size = normalize_resolution_int_mapping(
            encode_chunk_batch_size,
            name="encode_chunk_batch_size",
            default_keys=encode_chunk_frames.keys(),
            required_keys=encode_chunk_frames.keys(),
        )
        if any(batch_size < 1 for batch_size in encode_chunk_batch_size.values()):
            raise ValueError("encode_chunk_batch_size values must be >= 1.")
        self.encode_chunk_batch_size = encode_chunk_batch_size
        # unused parameter
        self.use_streaming_encode = False

        vae_full_path = vae_path
        latent_norm_full_path = latent_norm_path
        if bucket_name and not vae_path.startswith("s3://"):
            vae_full_path = f"s3://{bucket_name}/{vae_path}"
        if bucket_name and latent_norm_path and not latent_norm_path.startswith("s3://"):
            latent_norm_full_path = f"s3://{bucket_name}/{latent_norm_path}"

        self.vae = UniAEVAE(
            vae_pth=vae_full_path,
            object_store_credential_path_pretrained=object_store_credential_path_pretrained,
            latent_norm_path=latent_norm_full_path,
            pad_frames=pad_frames,
            pixel_trim=pixel_trim,
            backend=backend,
            chunk_size=self.encode_chunk_frames,
            encode_chunk_batch_size=self.encode_chunk_batch_size,
        )
        self.is_compiled = False

    def reset_dtype(self) -> None:
        pass

    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(state)

    def compile_encode_for_cudagraphs(
        self,
        *,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str = "inductor",
    ) -> None:
        """Compile the encode function for CUDA graphs."""
        compile_kwargs = dict(mode=mode, fullgraph=fullgraph, dynamic=dynamic, backend=backend)
        if backend == "cudagraphs":
            compile_kwargs.pop("mode", None)
        if backend == "cudagraphs" or compile_kwargs.get("mode", None) == "reduce-overhead":
            self.vae.dense_runtime.cg_compiled = True

        self.vae.dense_runtime._encode_chunk_core = torch.compile(
            self.vae.dense_runtime._encode_chunk_core, **compile_kwargs
        )
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

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latent)

    def get_latent_num_frames(self, num_pixel_frames: int, resolution: str | None = None) -> int:
        return self.vae.get_latent_num_frames(num_pixel_frames, resolution=resolution)

    def get_pixel_num_frames(self, num_latent_frames: int, resolution: str | None = None) -> int:
        return self.vae.get_pixel_num_frames(num_latent_frames, resolution=resolution)

    def get_latent_temporal_positions(
        self,
        num_pixel_frames: int,
        resolution: str | None = None,
        num_latent_frames: int | None = None,
    ) -> torch.Tensor:
        return self.vae.get_latent_temporal_positions(
            num_pixel_frames=num_pixel_frames,
            resolution=resolution,
            num_latent_frames=num_latent_frames,
        )

    @property
    def spatial_compression_factor(self) -> int:
        return self._spatial_compression_factor

    @property
    def temporal_compression_factor(self) -> int:
        return self._temporal_compression_factor

    @property
    def spatial_resolution(self):
        raise NotImplementedError(
            "spatial_resolution is deprecated for UniAEVAEInterface (resolution is input-dependent). "
            "Will be removed in a future MR."
        )

    @property
    def pixel_chunk_duration(self):
        raise NotImplementedError(
            "pixel_chunk_duration is deprecated for UniAEVAEInterface (chunk size is resolution-dependent). "
            "Use encode_chunk_frames[res_key] directly. Will be removed in a future MR."
        )

    @property
    def latent_chunk_duration(self):
        raise NotImplementedError(
            "latent_chunk_duration is deprecated for UniAEVAEInterface (chunk size is resolution-dependent). "
            "Use encode_chunk_frames[res_key] // temporal_compression_factor. Will be removed in a future MR."
        )

    @property
    def pad_frames(self) -> int:
        return self.vae._pad_frames

    @property
    def latent_ch(self) -> int:
        return self.vae.z_dim
