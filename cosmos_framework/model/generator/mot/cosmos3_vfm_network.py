# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import math
from collections.abc import Sequence
from typing import List, Tuple

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from cosmos_framework.utils import log
from cosmos_framework.configs.base.defaults.flex_attention import (
    AttentionScope,
    FlexBackendPreference,
)
from cosmos_framework.model.generator.mot.attention import SplitInfo, build_packed_sequence
from cosmos_framework.model.generator.mot.context_parallel_utils import (
    get_context_parallel_last_hidden_state,
    get_context_parallel_sharded_sequence,
)
from cosmos_framework.model.generator.mot.domain_aware_linear import DomainAwareLinear
from cosmos_framework.model.generator.mot.flex_attention import (
    FlexBackend,
    MaskItem,
    build_multiview_block_mask,
    resolve_flex_backend,
)
from cosmos_framework.model.generator.mot.modeling_utils import TimestepEmbedder, has_noisy_tokens
from cosmos_framework.model.generator.utils.memory import MemoryState
from cosmos_framework.data.generator.sequence_packing import ModalityData, PackedSequence
from cosmos_framework.data.generator.sequence_packing.natten import verify_natten_parameter_list
from cosmos_framework.data.generator.sequence_packing.runtime import get_causal_seq, get_full_only_seq


class Cosmos3VFMNetworkConfig(PretrainedConfig):
    def __init__(
        self,
        vision_gen=True,
        action_gen=False,
        sound_gen=False,
        vlm_config=None,
        latent_patch_size=2,
        latent_downsample_factor=8,
        latent_channel_size=16,
        lidar_latent_channel_size=None,
        max_latent_h=32,
        max_latent_w=32,
        max_latent_t=32,
        enable_fps_modulation=False,
        enable_vision_modality_embeddings: bool = False,
        enable_media_modality_embedding: bool = False,
        enable_sound_modality_embedding: bool = True,
        base_fps=24,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        timestep_scale=0.001,
        predict_text_tokens=False,
        joint_attn_implementation="two_way",
        use_multiview_flex_attention: bool = False,
        flex_attention_backend: FlexBackendPreference = "auto",
        attention_scope: AttentionScope = "all_views",
        action_dim=32,
        num_embodiment_domains=32,
        temporal_compression_factor_vision=4,
        temporal_compression_factor_action=1,
        natten_parameter_list=None,
        video_temporal_causal=False,
        # Sound generation parameters
        sound_dim: int | None = None,
        temporal_compression_factor_sound=1,
        sound_latent_fps: int = 25,
        enable_input_bias: bool = True,
        **kwargs,
    ):
        self.vision_gen = vision_gen
        self.sound_gen = sound_gen
        self.vlm_config = vlm_config
        self.latent_patch_size = latent_patch_size
        self.latent_downsample_factor = latent_downsample_factor
        self.latent_channel_size = latent_channel_size
        self.lidar_latent_channel_size = lidar_latent_channel_size
        self.max_latent_h = max_latent_h
        self.max_latent_w = max_latent_w
        self.max_latent_t = max_latent_t
        self.enable_fps_modulation = enable_fps_modulation
        self.enable_vision_modality_embeddings = enable_vision_modality_embeddings
        self.enable_media_modality_embedding = enable_media_modality_embedding
        self.enable_sound_modality_embedding = enable_sound_modality_embedding
        if self.enable_vision_modality_embeddings and self.enable_media_modality_embedding:
            raise ValueError(
                "enable_vision_modality_embeddings and enable_media_modality_embedding are mutually exclusive"
            )
        self.base_fps = base_fps
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.timestep_scale = timestep_scale
        self.predict_text_tokens = predict_text_tokens
        self.joint_attn_implementation = joint_attn_implementation
        self.use_multiview_flex_attention = use_multiview_flex_attention
        self.flex_attention_backend = flex_attention_backend
        self.attention_scope: AttentionScope = attention_scope
        self.temporal_compression_factor_vision = temporal_compression_factor_vision
        self.natten_parameter_list = natten_parameter_list
        self.video_temporal_causal = video_temporal_causal
        self.enable_input_bias = enable_input_bias

        # action related parameters
        self.action_gen = action_gen  # whether to generate action tokens
        self.action_dim = action_dim
        self.num_embodiment_domains = num_embodiment_domains
        self.temporal_compression_factor_action = temporal_compression_factor_action
        if self.action_gen:
            assert self.vision_gen, (
                "Action generation requires visual generation! We do NOT support action only training!"
            )

        # sound related parameters
        self.sound_dim = sound_dim
        self.temporal_compression_factor_sound = temporal_compression_factor_sound
        self.sound_latent_fps = sound_latent_fps
        if self.sound_gen:
            assert self.vision_gen, (
                "Sound generation requires visual generation! We do NOT support sound only training!"
            )

        super().__init__(**kwargs)


class Cosmos3VFMNetwork(PreTrainedModel):
    config_class = Cosmos3VFMNetworkConfig
    base_model_prefix = "cosmos3"

    def __init__(self, language_model, config: Cosmos3VFMNetworkConfig):
        super().__init__(config)
        self.language_model = language_model

        text_config = config.vlm_config.text_config if hasattr(config.vlm_config, "text_config") else config.vlm_config
        self.hidden_size = text_config.hidden_size
        self.num_heads = text_config.num_attention_heads
        self.num_kv_heads = text_config.num_key_value_heads
        self.head_dim = text_config.head_dim
        self.num_hidden_layers = text_config.num_hidden_layers
        self.attention_io_layout = "sequence_sharded"
        self.predict_text_tokens = config.predict_text_tokens

        if config.natten_parameter_list is not None and config.joint_attn_implementation != "three_way":
            raise NotImplementedError(
                f"Sparsity is only supported with 'three_way' attention, but got {config.joint_attn_implementation=}, "
                "and 'natten_parameter_list' was not None."
            )
        self.natten_parameter_list = verify_natten_parameter_list(
            config.natten_parameter_list, num_layers=self.num_hidden_layers
        )

        if config.video_temporal_causal and config.joint_attn_implementation != "three_way":
            raise ValueError(
                f"video_temporal_causal=True requires joint_attn_implementation='three_way', "
                f"but got {config.joint_attn_implementation!r}."
            )
        self.video_temporal_causal = config.video_temporal_causal
        self.pad_for_cuda_graphs = False

        # Which kernels the multiview FlexAttention path runs on, and the mask geometry that
        # forces. Resolved here rather than per forward because the answer depends on the host
        # and not on the batch, so a run's log records it once.
        self.flex_backend: FlexBackend | None = None
        if config.use_multiview_flex_attention:
            # The device is only read for the GPU architecture the FlashAttention-4 block size
            # follows from, which is the same for every device in this process, so the local one
            # stands in for the one the batch will arrive on.
            device = (
                torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            )
            self.flex_backend = resolve_flex_backend(device, config.flex_attention_backend)
            log.info(
                f"Multiview FlexAttention is running on the {self.flex_backend.name} backend "
                f"(flex_attention_backend={config.flex_attention_backend!r}), with a "
                f"{self.flex_backend.block_size} block mask over a GEN stream padded to "
                f"{self.flex_backend.full_seq_alignment} tokens. Noisy tokens attend to the "
                f"noisy tokens of their sample under scope {config.attention_scope!r}."
            )

        if config.vision_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.timestep_scale = config.timestep_scale
            self.latent_downsample = config.latent_downsample_factor * config.latent_patch_size
            self.max_latent_h = config.max_latent_h
            self.max_latent_w = config.max_latent_w
            self.max_latent_t = config.max_latent_t
            self.latent_channel = config.latent_channel_size
            self.patch_latent_dim = self.latent_patch_size**2 * self.latent_channel

            _input_bias = config.enable_input_bias
            self.time_embedder = TimestepEmbedder(self.hidden_size, bias=_input_bias)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size, bias=_input_bias)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)

            # LiDAR is its own modality: a range clip enters and leaves the sequence through
            # its own pair of projections, the way action and sound do. Its VAE is wider than
            # the camera's (128 vs 48), and these two matrices are what that costs -- a patch
            # count follows T, H and W, not channels, so the streams agree on everything else
            # and share the grid packing, patchify and timestep machinery below.
            self.lidar_latent_channel = config.lidar_latent_channel_size
            if self.lidar_latent_channel is not None:
                self.lidar_patch_latent_dim = self.latent_patch_size**2 * self.lidar_latent_channel
                self.lidar2llm = nn.Linear(self.lidar_patch_latent_dim, self.hidden_size, bias=_input_bias)
                self.llm2lidar = nn.Linear(self.hidden_size, self.lidar_patch_latent_dim)
            if config.enable_vision_modality_embeddings:
                self.image_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))
                self.video_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))
            if config.enable_media_modality_embedding:
                self.media_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        if config.action_gen:
            self.action_dim = config.action_dim
            self.num_embodiment_domains = config.num_embodiment_domains
            self.action2llm = DomainAwareLinear(self.action_dim, self.hidden_size, self.num_embodiment_domains)
            self.llm2action = DomainAwareLinear(self.hidden_size, self.action_dim, self.num_embodiment_domains)

            self.action_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        if config.sound_gen:
            self.sound_dim = config.sound_dim
            self.sound2llm = nn.Linear(config.sound_dim, self.hidden_size, bias=config.enable_input_bias)
            self.llm2sound = nn.Linear(self.hidden_size, config.sound_dim)
            if config.enable_sound_modality_embedding:
                self.sound_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))  # [hidden_size]

        self.config = config
        self.parallel_dims = None

    def init_weights(self, buffer_device: torch.device | None):
        if self.config.vision_gen or self.config.action_gen or self.config.sound_gen:
            self.time_embedder._init_weights(buffer_device=buffer_device)

        if self.config.vision_gen:
            std = 1.0 / math.sqrt(self.patch_latent_dim)
            torch.nn.init.trunc_normal_(self.vae2llm.weight, std=std, a=-3 * std, b=3 * std)
            if self.config.enable_input_bias:
                torch.nn.init.zeros_(self.vae2llm.bias)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.llm2vae.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.llm2vae.bias)

            if self.lidar_latent_channel is not None:
                # Fan-in scaling as for every other head; named separately because the
                # modality embeddings below read the llm2vae ``std``.
                lidar_in_std = 1.0 / math.sqrt(self.lidar_patch_latent_dim)
                torch.nn.init.trunc_normal_(
                    self.lidar2llm.weight, std=lidar_in_std, a=-3 * lidar_in_std, b=3 * lidar_in_std
                )
                if self.config.enable_input_bias:
                    torch.nn.init.zeros_(self.lidar2llm.bias)
                torch.nn.init.trunc_normal_(self.llm2lidar.weight, std=std, a=-3 * std, b=3 * std)
                torch.nn.init.zeros_(self.llm2lidar.bias)

            if self.config.enable_vision_modality_embeddings:
                torch.nn.init.trunc_normal_(self.image_modality_embed, std=std, a=-3 * std, b=3 * std)
                torch.nn.init.trunc_normal_(self.video_modality_embed, std=std, a=-3 * std, b=3 * std)
            if self.config.enable_media_modality_embedding:
                torch.nn.init.trunc_normal_(self.media_modality_embed, std=std, a=-3 * std, b=3 * std)

        if self.config.action_gen:
            # DomainAwareLinear uses embeddings for weights, so we initialize them differently
            # action2llm: input_size=action_dim, output_size=hidden_size
            std = 1.0 / math.sqrt(self.action_dim)
            torch.nn.init.trunc_normal_(self.action2llm.fc.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.action2llm.bias.weight)

            # llm2action: input_size=hidden_size, output_size=action_dim
            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.llm2action.fc.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.llm2action.bias.weight)

            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.action_modality_embed, std=std, a=-3 * std, b=3 * std)

        if self.config.sound_gen:
            # sound2llm: input_size=sound_dim, output_size=hidden_size
            std = 1.0 / math.sqrt(self.sound_dim)
            torch.nn.init.trunc_normal_(self.sound2llm.weight, std=std, a=-3 * std, b=3 * std)
            if self.config.enable_input_bias:
                torch.nn.init.zeros_(self.sound2llm.bias)

            # llm2sound: input_size=hidden_size, output_size=sound_dim
            std = 1.0 / math.sqrt(self.hidden_size)
            torch.nn.init.trunc_normal_(self.llm2sound.weight, std=std, a=-3 * std, b=3 * std)
            torch.nn.init.zeros_(self.llm2sound.bias)

            if self.config.enable_sound_modality_embedding:
                std = 1.0 / math.sqrt(self.hidden_size)
                torch.nn.init.trunc_normal_(self.sound_modality_embed, std=std, a=-3 * std, b=3 * std)

        self.language_model.init_weights(buffer_device=buffer_device)

    def generate_reasoner_text(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        eos_token_id: int | list[int] | None = None,
        pad_token_id: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        seed: int | None = None,
        return_only_new_tokens: bool = False,
    ) -> torch.Tensor:
        """Autoregressively generate text tokens using only the reasoner tower.

        Thin pass-through to ``self.language_model.generate_reasoner_text``
        (see ``unified_mot._impl_generate_reasoner_text`` for full argument
        documentation).  Handles both text-only and image-conditioned (I2V)
        prompts through this single entry point: pass
        ``pixel_values`` + ``image_grid_thw`` (and optionally
        ``attention_mask``) for image-conditioned prefill via the Qwen3-VL
        visual encoder, or omit them for text-only prefill.  Video
        conditioning is also supported via ``pixel_values_videos`` +
        ``video_grid_thw``; the image and video pairs are mutually exclusive.
        Uses the und-pathway weights (those WITHOUT the ``_moe_gen`` suffix)
        plus ``embed_tokens`` / ``norm`` / ``lm_head``; the generation pathway
        and all VFM-level multimodal embedders / heads (``vae2llm``,
        ``llm2vae``, ``sound2llm``, etc.) are bypassed.

        ``repetition_penalty`` / ``presence_penalty`` are pass-through
        sampling controls applied inside
        :func:`unified_mot._impl_generate_reasoner_text` as logit
        transformations *before* the ``do_sample`` argmax / multinomial
        branch (so they shift the greedy argmax too).  Identity defaults
        (``1.0`` / ``0.0``) keep the un-penalized fast path
        bit-identical.

        ``seed`` is a pass-through sampling-RNG knob: when provided,
        :func:`unified_mot._impl_generate_reasoner_text` allocates a
        device-local ``torch.Generator``, seeds it once with
        ``manual_seed(seed)``, and threads it into every
        ``torch.multinomial`` draw — making the decoded sequence a
        deterministic function of the seed, the prompt, and the
        penalty masks.  ``None`` (default) consumes the device's
        default RNG and is bit-identical to the pre-seed call surface.
        Has no effect under greedy decoding (the argmax branch never
        reads the generator).
        """

        return self.language_model.generate_reasoner_text(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
            return_only_new_tokens=return_only_new_tokens,
        )

    @property
    def lidar_gen(self) -> bool:
        """Whether this network carries the LiDAR stream's own projections."""
        return self.config.vision_gen and self.lidar_latent_channel is not None

    def patchify_and_pack_latents(
        self,
        tokens_vision: torch.Tensor,
        token_shapes_vision: Sequence[tuple[int, ...]],
        latent_channel: int | None = None,
    ) -> tuple[torch.Tensor, List[Tuple[int, int, int]]]:
        p = self.latent_patch_size
        # One channel count per call: the caller passes its stream's width, since patches of
        # different widths cannot pack into one tensor.
        latent_channel = self.latent_channel if latent_channel is None else latent_channel
        # Patchify and pack the latents
        packed_latent = []
        original_latent_shapes = []  # Store original shapes for unpadding later

        # C, T, H, W
        for latent, (t, h, w) in zip(tokens_vision, token_shapes_vision):
            latent = latent.squeeze(0)  # [C,T,H,W]

            # Get original latent dimensions
            _, t_actual, h_actual, w_actual = latent.shape
            original_latent_shapes.append((t_actual, h_actual, w_actual))

            # Compute padded dimensions (must be divisible by p)
            h_padded = ((h_actual + p - 1) // p) * p
            w_padded = ((w_actual + p - 1) // p) * p

            # Zero-pad if dimensions are not divisible by p
            if h_padded != h_actual or w_padded != w_actual:
                padded = torch.zeros(
                    (latent_channel, t_actual, h_padded, w_padded),
                    device=latent.device,
                    dtype=latent.dtype,
                )  # [C,T,H_padded,W_padded]
                padded[:, :, :h_actual, :w_actual] = latent
                latent = padded  # [C,T,H_padded,W_padded]

            # Compute number of patches after padding
            h_patches = h_padded // p
            w_patches = w_padded // p

            # Patchify
            latent = latent.reshape(
                latent_channel, t_actual, h_patches, p, w_patches, p
            )  # [C,T,h_patches,p,w_patches,p]
            latent = torch.einsum("cthpwq->thwpqc", latent).reshape(
                -1, p * p * latent_channel
            )  # [T*h_patches*w_patches,patch_latent_dim]
            packed_latent.append(latent)

        # We assumed latents we get to the network is already noised
        packed_latent = torch.cat(packed_latent, dim=0)  # [total_vision_patches,patch_latent_dim]
        return packed_latent, original_latent_shapes

    def unpatchify_and_unpack_latents(
        self,
        packed_mse_preds: torch.Tensor,
        token_shapes_vision: List[Tuple[int, int, int]],
        noisy_frame_indexes_vision: list[torch.Tensor],
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
        latent_channel: int | None = None,
    ) -> list[torch.Tensor]:
        p = self.latent_patch_size
        # One channel count per call, as in ``patchify_and_pack_latents``.
        latent_channel = self.latent_channel if latent_channel is None else latent_channel
        unpatchified_latents = []

        # Split packed_mse_preds back into individual latents based on token_shapes_vision
        start_idx = 0
        for i, (t_c, h_c, w_c) in enumerate(token_shapes_vision):
            # Get original shape for unpadding (if provided)
            if original_latent_shapes is not None:
                t_orig, h_orig, w_orig = original_latent_shapes[i]
                # Compute padded dimensions used during patchify
                h_padded = ((h_orig + p - 1) // p) * p
                w_padded = ((w_orig + p - 1) // p) * p
                h_patches = h_padded // p
                w_patches = w_padded // p
            else:
                # Fallback: use token shapes directly (assumes no padding was needed)
                t_orig, h_orig, w_orig = t_c, h_c * p, w_c * p
                h_patches, w_patches = h_c, w_c

            # noisy_frame_indexes_vision is a list of tensors, each with shape (T,),
            # where the values are the noisy frame indices.
            noisy_frame_indexes = noisy_frame_indexes_vision[i]
            t_n = len(noisy_frame_indexes)

            # Initialize with the original shape (after unpadding), zeros for clean frames
            output_tensor = torch.zeros(
                (latent_channel, t_c, h_orig, w_orig),
                device=packed_mse_preds.device,
                dtype=packed_mse_preds.dtype,
            )  # [C,T,H_orig,W_orig]
            num_patches = t_n * h_patches * w_patches
            if num_patches > 0:
                end_idx = start_idx + num_patches
                # Extract patches for this latent
                latent_patches = packed_mse_preds[start_idx:end_idx]  # [num_patches,patch_latent_dim]
                # Reshape back to [t_n, h_patches, w_patches, p, p, channels]
                latent_patches = latent_patches.reshape(
                    t_n, h_patches, w_patches, p, p, latent_channel
                )  # [T_n,h_patches,w_patches,p,p,C]
                # Invert the einsum operation: "thwpqc->cthpwq"
                latent = torch.einsum("thwpqc->cthpwq", latent_patches)  # [C,T_n,h_patches,p,w_patches,p]
                # Reshape back to [channels, t_n, h_padded, w_padded]
                latent = latent.reshape(latent_channel, t_n, h_patches * p, w_patches * p)  # [C,T_n,H_padded,W_padded]

                # Crop to original dimensions (unpad the zeros)
                latent = latent[:, :, :h_orig, :w_orig]  # [C,T_n,H_orig,W_orig]

                # Fill only the noisy frame positions using the actual mask indices
                output_tensor[:, noisy_frame_indexes] = latent

                start_idx = end_idx

            unpatchified_latents.append(output_tensor.unsqueeze(0))  # [1,C,T,H,W]

        # Return list of unpatchified latents (supports variable shapes)
        return unpatchified_latents

    def pack_action(
        self,
        tokens_action: list[torch.Tensor],
        token_shapes_action: list[tuple[int, ...]],
        domain_id_action: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack variable-length action tokens into a 1D sequence for transformer input.

        Args:
            tokens_action: List of action tensors, each [T_i, action_dim] (T_i may vary).
            token_shapes_action: List of (T_i,) tuples per sample.
            domain_id_action: List of domain ID tensors, each of shape [1].

        Returns:
            Tuple of (packed_tokens, per_token_domain_id):
                packed_tokens: [total_action_tokens, action_dim]
                per_token_domain_id: [total_action_tokens]
        """
        packed: list[torch.Tensor] = []
        domain_ids: list[torch.Tensor] = []
        for tokens, shape, d_id in zip(tokens_action, token_shapes_action, domain_id_action):
            T = shape[0]
            packed.append(tokens[:T])
            domain_ids.append(d_id.expand(T))
        return torch.cat(packed, dim=0), torch.cat(domain_ids, dim=0)

    def unpack_action(
        self,
        packed_action_preds: torch.Tensor,
        token_shapes_action: list[tuple[int, ...]],
        noisy_frame_indexes_action: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Unpack action predictions back into per-sample action tensors.

        Args:
            packed_action_preds: Packed action predictions of shape (total_noisy_tokens, action_dim)
            token_shapes_action: Per-sample token shapes, each (T_i,) tuple.
            noisy_frame_indexes_action: List of tensors, each with shape (Tn_i,), where the values
                are the noisy frame indices for sample i.

        Returns:
            List of per-sample tensors, each of shape (T_i, action_dim), with predictions
            placed at noisy positions. Clean positions are left as zeros.
        """
        unpacked: list[torch.Tensor] = []
        start_idx = 0
        for shape, noisy_frame_indexes in zip(token_shapes_action, noisy_frame_indexes_action):
            T = shape[0]
            output = torch.zeros(
                (T, self.action_dim),
                device=packed_action_preds.device,
                dtype=packed_action_preds.dtype,
            )
            t_n = len(noisy_frame_indexes)
            if t_n > 0:
                end_idx = start_idx + t_n
                output[noisy_frame_indexes] = packed_action_preds[start_idx:end_idx]
                start_idx = end_idx
            unpacked.append(output)
        return unpacked

    def pack_sound_latents(
        self,
        tokens_sound: list[torch.Tensor],
        token_shapes_sound: list[tuple[int, int, int]],
    ) -> torch.Tensor:
        """Pack sound latents into a 1D sequence for transformer input.

        Args:
            tokens_sound: List of sound latent tensors, each [C, T]
            token_shapes_sound: List of (T, 1, 1) tuples per sample

        Returns:
            Packed tensor of shape [total_sound_tokens, C]
        """
        packed = []
        for sound, shape in zip(tokens_sound, token_shapes_sound):
            T = shape[0]
            # sound: [C, T] → take first T frames → [C, T]
            # Then permute to [T, C] for packing
            sound_tokens = sound[:, :T].permute(1, 0)  # [T,C]
            packed.append(sound_tokens)
        return torch.cat(packed, dim=0)  # [total_sound_tokens,C]

    def unpack_sound_latents(
        self,
        packed_sound_preds: torch.Tensor,
        token_shapes_sound: list[tuple[int, int, int]],
        noisy_frame_indexes_sound: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Unpack sound predictions back into per-sample sound latents.

        Args:
            packed_sound_preds: Packed sound predictions of shape (total_noisy_tokens, sound_dim)
            token_shapes_sound: List of (T, 1, 1) tuples per sample
            noisy_frame_indexes_action: List of tensors, each with shape (T_i,), where the values
                are the noisy frame indices. T_i <= max_T.

        Returns:
            List of per-sample tensors, each [C, T], with predictions placed at noisy positions.
            Clean positions are left as zeros.
        """
        unpacked = []
        start_idx = 0
        for shape, noisy_frame_indexes in zip(token_shapes_sound, noisy_frame_indexes_sound):
            T = shape[0]
            # Initialize output with zeros for clean positions
            output = torch.zeros(
                (self.sound_dim, T),
                device=packed_sound_preds.device,
                dtype=packed_sound_preds.dtype,
            )

            t_n = len(noisy_frame_indexes)

            if t_n > 0:
                end_idx = start_idx + t_n
                # packed_sound_preds: [total_noisy_tokens, C] → transpose and fill at noisy positions
                output[:, noisy_frame_indexes] = packed_sound_preds[
                    start_idx:end_idx
                ].T  # packed_sound_preds[...]: [T_n,C] → .T: [C,T_n]
                start_idx = end_idx

            unpacked.append(output)
        return unpacked

    def _encode_text(
        self,
        packed_seq: PackedSequence,
    ) -> tuple[torch.Tensor, torch.dtype]:
        """Embed text tokens and initialize packed_sequence.

        Args:
            packed_seq: PackedSequence containing text_ids and text_indexes.

        Returns:
            tuple of (packed_sequence, target_dtype) where packed_sequence has text embeddings filled in.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_seq.text_ids)  # [N_text,hidden_size]
        packed_sequence = packed_text_embedding.new_zeros(
            size=(packed_seq.sequence_length, self.hidden_size)
        )  # [N_total,hidden_size]
        packed_sequence[packed_seq.text_indexes] = (
            packed_text_embedding  # [N_text,hidden_size] scattered into [N_total,hidden_size]
        )
        return packed_sequence, packed_text_embedding.dtype

    def _embed_packed_timesteps(self, timesteps: torch.Tensor, packed_seq: PackedSequence) -> torch.Tensor:
        """Embed noised-token timesteps, reusing work when packing proves they share one scalar."""
        if packed_seq.uses_single_timestep and timesteps.numel() > 1:
            timestep = timesteps[:1]  # [1]
            with torch.autocast("cuda", enabled=True, dtype=torch.float32):
                timestep_embed = self.time_embedder(timestep)  # [1,hidden_size]
            # Materialize: expand() aliases storage; in-place ops on a float32 no-op .to() would corrupt all rows.
            return timestep_embed.expand(timesteps.shape[0], -1).contiguous()  # [N_noisy_frames,hidden_size]

        # Timesteps are computed in FP32 for numerical stability.
        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            return self.time_embedder(timesteps)  # [N_noisy_frames,hidden_size]

    def _encode_vision(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> List[Tuple[int, int, int]] | None:
        """Project vision tokens and fill into packed_sequence.

        Args:
            packed_seq: PackedSequence containing vision tokens and metadata.
            packed_sequence: The packed sequence tensor to fill vision embeddings into (modified in-place).
            target_dtype: Target dtype for embeddings (typically from text embedding).

        Returns:
            Original latent shapes before padding (for unpadding during decode), or None if no vision tokens.
        """
        if self.config.enable_vision_modality_embeddings:
            modality_embed = self.image_modality_embed if packed_seq.is_image_batch else self.video_modality_embed
        elif self.config.enable_media_modality_embedding:
            modality_embed = self.media_modality_embed
        else:
            modality_embed = None
        return self._encode_grid_stream(
            packed_seq,
            packed_seq.vision,
            packed_sequence,
            vae2llm=self.vae2llm,
            latent_channel=self.latent_channel,
            modality_embed=modality_embed,
            target_dtype=target_dtype,
        )

    def _encode_lidar(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> List[Tuple[int, int, int]] | None:
        """Project LiDAR range-view tokens and fill into packed_sequence.

        Same treatment as the vision stream, through the LiDAR VAE's own width and its own
        pair of projections. No modality embedding: the two streams already differ by their
        projections and by where mRoPE puts them.
        """
        return self._encode_grid_stream(
            packed_seq,
            packed_seq.lidar,
            packed_sequence,
            vae2llm=self.lidar2llm,
            latent_channel=self.lidar_latent_channel,
            modality_embed=None,
            target_dtype=target_dtype,
        )

    def _encode_grid_stream(
        self,
        packed_seq: PackedSequence,
        modality: ModalityData | None,
        packed_sequence: torch.Tensor,
        *,
        vae2llm: nn.Linear,
        latent_channel: int,
        modality_embed: torch.Tensor | None,
        target_dtype: torch.dtype,
    ) -> List[Tuple[int, int, int]] | None:
        """Patchify, project and scatter one stream of VAE latent grids.

        Shared by the vision and LiDAR streams so a second grid modality cannot drift from
        the first on patchification, timestep embedding or where its tokens land.

        Returns:
            Original latent shapes before padding, for unpadding during decode, or ``None``
            when the stream holds no tokens.
        """
        if modality is None or modality.tokens is None:
            return None

        assert modality.token_shapes is not None
        assert isinstance(modality.sequence_indexes, torch.Tensor)
        assert isinstance(modality.timesteps, torch.Tensor)
        torch._assert(
            modality.timesteps.dtype in (torch.long, torch.float32),
            f"Timestep must be long/float32, got {modality.timesteps.dtype}",
        )
        assert isinstance(modality.mse_loss_indexes, torch.Tensor)

        packed_patches, original_latent_shapes = self.patchify_and_pack_latents(
            modality.tokens, modality.token_shapes, latent_channel=latent_channel
        )  # [total_patches,patch_latent_dim]
        packed_tokens = vae2llm(packed_patches.to(target_dtype))  # [total_patches,hidden_size]
        if modality_embed is not None:
            packed_tokens = packed_tokens + modality_embed.view(1, -1)  # [total_patches,hidden_size]

        if modality.mse_loss_indexes.numel() > 0:
            timesteps = modality.timesteps.to(dtype=torch.float32) * self.timestep_scale  # [N_noisy_frames]
            packed_timestep_embeds = self._embed_packed_timesteps(timesteps, packed_seq)  # [N_noisy_frames,hidden_size]
            packed_timestep_embeds = packed_timestep_embeds.to(target_dtype)  # [N_noisy_frames,hidden_size]

            packed_tokens = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens,
                packed_timestep_embeds=packed_timestep_embeds,
                noisy_frame_indexes=modality.noisy_frame_indexes,
                token_shapes=modality.token_shapes,
            )  # [total_patches,hidden_size]

        packed_sequence[modality.sequence_indexes] = (
            packed_tokens  # [total_patches,hidden_size] scattered into [N_total,hidden_size]
        )
        return original_latent_shapes

    def _decode_vision(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> None:
        """Decode vision tokens from hidden states and update output_dict.

        Args:
            packed_seq: PackedSequence containing mse_loss_indexes_vision and token_shapes_vision.
            last_hidden_state: Hidden states from the transformer.
            output_dict: Output dictionary to update with mse_preds (modified in-place).
            original_latent_shapes: Original latent shapes before padding (for unpadding).
        """
        output_dict.update(
            preds_vision=self._decode_grid_stream(
                packed_seq.vision,
                last_hidden_state,
                vae2llm=self.vae2llm,
                llm2vae=self.llm2vae,
                latent_channel=self.latent_channel,
                patch_latent_dim=self.patch_latent_dim,
                original_latent_shapes=original_latent_shapes,
            )
        )

    def _decode_lidar(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
        original_latent_shapes: List[Tuple[int, int, int]] | None = None,
    ) -> None:
        """Decode LiDAR tokens from hidden states and update output_dict."""
        output_dict.update(
            preds_lidar=self._decode_grid_stream(
                packed_seq.lidar,
                last_hidden_state,
                vae2llm=self.lidar2llm,
                llm2vae=self.llm2lidar,
                latent_channel=self.lidar_latent_channel,
                patch_latent_dim=self.lidar_patch_latent_dim,
                original_latent_shapes=original_latent_shapes,
            )
        )

    def _decode_grid_stream(
        self,
        modality: ModalityData | None,
        last_hidden_state: torch.Tensor,
        *,
        vae2llm: nn.Linear,
        llm2vae: nn.Linear,
        latent_channel: int,
        patch_latent_dim: int,
        original_latent_shapes: List[Tuple[int, int, int]] | None,
    ) -> list[torch.Tensor]:
        """Read one stream's noisy patches back out of the hidden states.

        Returns:
            One ``[1,C,T,H,W]`` prediction per item of the stream. A stream with nothing
            noised -- absent from the batch, or present as conditioning only -- still runs a
            zero-weighted pass through its own two projections, so every rank reaches the
            same parameters in a step, which is what FSDP requires.
        """
        has_noisy = (
            modality is not None
            and modality.tokens is not None
            and isinstance(modality.mse_loss_indexes, torch.Tensor)
            and modality.mse_loss_indexes.numel() > 0
        )
        if not has_noisy:
            probe = torch.zeros(
                [1, patch_latent_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )  # [1,patch_latent_dim]
            probe = llm2vae(vae2llm(probe))  # [1,patch_latent_dim]
            # Per-item zeros of the right shape, so callers iterating the predictions
            # (_get_velocity, _compute_flow_matching_loss) see the shapes they expect; the
            # probe is folded into the first so the projections stay in the autograd graph.
            if modality is not None and modality.tokens is not None:
                preds = [torch.zeros_like(tok) for tok in modality.tokens]
                preds[0] = preds[0] + 0.0 * probe.sum()
                return preds
            # The stream is absent, so nothing iterates this; it exists for the graph alone.
            return [probe]

        assert modality is not None  # Type narrowing
        assert isinstance(modality.mse_loss_indexes, torch.Tensor)
        assert modality.noisy_frame_indexes is not None
        noisy_patches = last_hidden_state[modality.mse_loss_indexes]  # [total_noisy_patches,hidden_size]
        preds = llm2vae(noisy_patches)  # [total_noisy_patches,patch_latent_dim]
        return self.unpatchify_and_unpack_latents(
            preds,
            token_shapes_vision=modality.token_shapes,
            noisy_frame_indexes_vision=modality.noisy_frame_indexes,
            original_latent_shapes=original_latent_shapes,
            latent_channel=latent_channel,
        )

    def _encode_action(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> None:
        """Encode action tokens and fill into packed_sequence."""
        if packed_seq.action is None or packed_seq.action.tokens is None:
            # No action tokens in this batch
            return

        action: ModalityData = packed_seq.action
        assert action.token_shapes is not None
        assert isinstance(action.sequence_indexes, torch.Tensor)
        assert isinstance(action.timesteps, torch.Tensor)
        assert isinstance(action.mse_loss_indexes, torch.Tensor)

        # Pack variable-length action tokens into a 1D sequence (same pattern as pack_sound_latents)
        packed_tokens_action, per_token_domain_id = self.pack_action(
            action.tokens, action.token_shapes, action.domain_id
        )
        # Flow interpolation keeps actions in FP32; cast here to match the action encoder's model dtype.
        packed_tokens_action = packed_tokens_action.to(target_dtype)  # [B_action*T_action,action_dim]
        packed_tokens_action = self.action2llm(packed_tokens_action, per_token_domain_id)

        packed_tokens_action = packed_tokens_action + self.action_modality_embed.view(
            1, -1
        )  # [B_action*T_action,hidden_size]

        has_noisy_actions = has_noisy_tokens(action)
        if has_noisy_actions:
            timesteps_action = action.timesteps * self.timestep_scale  # [N_noisy_frames_action]
            packed_timestep_embeds_action = self._embed_packed_timesteps(
                timesteps_action, packed_seq
            )  # [N_noisy_frames_action,hidden_size]
            packed_timestep_embeds_action = packed_timestep_embeds_action.to(
                target_dtype
            )  # [N_noisy_frames_action,hidden_size]

            packed_tokens_action = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_action,
                packed_timestep_embeds=packed_timestep_embeds_action,
                noisy_frame_indexes=action.noisy_frame_indexes,
                token_shapes=action.token_shapes,
            )  # [B_action*T_action,hidden_size]

        packed_sequence[action.sequence_indexes] = (
            packed_tokens_action  # [B_action*T_action,hidden_size] scattered into [N_total,hidden_size]
        )

    def _decode_action(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
    ) -> None:
        """Decode action tokens from hidden states and update output_dict."""
        action = packed_seq.action
        # Shared predicate with OmniMoTModel's has_noisy_actions gating: actions
        # are decodable targets only when the packer marked action tokens noisy.
        has_noisy_action = has_noisy_tokens(action)
        if not has_noisy_action:
            # dummy forward to maintain computation graph consistency across ranks
            preds_action = torch.zeros(
                [1, self.action_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )  # [1,action_dim]
            dummy_domain_id = torch.zeros([1], device=last_hidden_state.device, dtype=torch.long)  # [1]
            preds_action = self.action2llm(preds_action, dummy_domain_id) + self.action_modality_embed.view(
                1, -1
            )  # [1,hidden_size]
            preds_action = self.llm2action(preds_action, dummy_domain_id)  # [1,action_dim]
            # Return a list of per-sample zero tensors with correct shapes (e.g. (T, action_dim)),
            # so downstream code (_get_velocity, _compute_flow_matching_loss) that iterates over preds_action
            # gets properly-shaped tensors. Without this, the dummy tensor (1, action_dim)
            # would cause a size mismatch when concatenating vision+action velocities.
            if action is not None and action.tokens is not None:
                preds_action_list = [torch.zeros_like(tok) for tok in action.tokens]
                # Inject dummy forward's computation graph so DomainAwareLinear params
                # stay in the autograd graph (zeros_like creates detached tensors).
                preds_action_list[0] = preds_action_list[0] + 0.0 * preds_action.sum()
            # When action is None (no action in batch), fall back to [preds_action] purely for
            # gradient graph consistency — it won't be iterated over.
            else:
                preds_action_list = [preds_action]
            output_dict.update(preds_action=preds_action_list)
        else:
            assert action is not None  # Type narrowing
            assert isinstance(action.mse_loss_indexes, torch.Tensor)
            assert action.condition_mask is not None
            assert len(action.domain_id) > 0

            action_hidden_states = last_hidden_state[action.mse_loss_indexes]  # [total_noisy_action_tokens,hidden_size]

            # Build per-token domain IDs for the noisy tokens (same expansion logic as pack_action)
            domain_ids: list[torch.Tensor] = []
            for nfi, d_id in zip(action.noisy_frame_indexes, action.domain_id):
                domain_ids.append(d_id.expand(len(nfi)))
            per_token_domain_id = torch.cat(domain_ids, dim=0)

            preds_action = self.llm2action(
                action_hidden_states, per_token_domain_id
            )  # [total_noisy_action_tokens,action_dim]
            preds_action = self.unpack_action(preds_action, action.token_shapes, action.noisy_frame_indexes)
            output_dict.update(preds_action=preds_action)

    def _encode_sound(
        self,
        packed_seq: PackedSequence,
        packed_sequence: torch.Tensor,
        target_dtype: torch.dtype,
    ) -> None:
        """Encode sound tokens and fill into packed_sequence.

        Args:
            packed_seq: PackedSequence containing sound tokens and metadata.
            packed_sequence: The packed sequence tensor to fill sound embeddings into (modified in-place).
            target_dtype: Target dtype for embeddings (typically from text embedding).
        """
        if packed_seq.sound is None or packed_seq.sound.tokens is None:
            # No sound tokens in this batch
            return

        sound = packed_seq.sound
        assert sound.token_shapes is not None
        assert isinstance(sound.sequence_indexes, torch.Tensor)
        assert isinstance(sound.timesteps, torch.Tensor)
        assert isinstance(sound.mse_loss_indexes, torch.Tensor)

        # Pack sound latents: list of [C, T] tensors → [total_tokens, C]
        packed_tokens_sound = self.pack_sound_latents(
            sound.tokens, sound.token_shapes
        )  # [total_sound_tokens,sound_dim]
        packed_tokens_sound = packed_tokens_sound.to(target_dtype)  # [total_sound_tokens,sound_dim]

        # Project sound tokens and optionally add a modality embedding. Position info
        # comes from mRoPE position IDs in the attention layers.
        packed_tokens_sound = self.sound2llm(packed_tokens_sound)  # [total_sound_tokens,hidden_size]
        if self.config.enable_sound_modality_embedding:
            packed_tokens_sound = packed_tokens_sound + self.sound_modality_embed  # [total_sound_tokens,hidden_size]

        has_noisy_sound = sound.mse_loss_indexes.numel() > 0
        if has_noisy_sound:
            timesteps_sound = sound.timesteps * self.timestep_scale  # [N_noisy_frames_sound]
            packed_timestep_embeds_sound = self._embed_packed_timesteps(
                timesteps_sound, packed_seq
            )  # [N_noisy_frames_sound,hidden_size]
            packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(
                target_dtype
            )  # [N_noisy_frames_sound,hidden_size]

            packed_tokens_sound = _apply_timestep_embeds_to_noisy_tokens(
                packed_tokens=packed_tokens_sound,
                packed_timestep_embeds=packed_timestep_embeds_sound,
                noisy_frame_indexes=sound.noisy_frame_indexes,
                token_shapes=sound.token_shapes,
            )  # [total_sound_tokens,hidden_size]

        packed_sequence[sound.sequence_indexes] = (
            packed_tokens_sound  # [total_sound_tokens,hidden_size] scattered into [N_total,hidden_size]
        )

    def _decode_sound(
        self,
        packed_seq: PackedSequence,
        last_hidden_state: torch.Tensor,
        output_dict: dict,
    ) -> None:
        """Decode sound tokens from hidden states and update output_dict.

        Args:
            packed_seq: PackedSequence containing sound modality data.
            last_hidden_state: Hidden states from the transformer.
            output_dict: Output dictionary to update with preds_sound (modified in-place).
        """
        sound = packed_seq.sound
        # Check if no sound or no noisy sound tokens
        has_noisy_sound = (
            sound is not None
            and sound.tokens is not None
            and isinstance(sound.mse_loss_indexes, torch.Tensor)
            and sound.mse_loss_indexes.numel() > 0
        )
        if not has_noisy_sound:
            # dummy forward to maintain computation graph consistency across ranks
            preds_sound = torch.zeros(
                [1, self.sound_dim], device=last_hidden_state.device, dtype=last_hidden_state.dtype
            )  # [1,sound_dim]
            preds_sound = self.sound2llm(preds_sound)  # [1,hidden_size]
            if self.config.enable_sound_modality_embedding:
                preds_sound = preds_sound + self.sound_modality_embed  # [1,hidden_size]
            preds_sound = self.llm2sound(preds_sound)  # [1,sound_dim]
            if sound is not None and sound.tokens is not None:
                preds_sound_list = [torch.zeros_like(tok) for tok in sound.tokens]
                preds_sound_list[0] = preds_sound_list[0] + 0.0 * preds_sound.sum()
            else:
                preds_sound_list = [preds_sound]
            output_dict.update(preds_sound=preds_sound_list)
        else:
            assert sound is not None  # Type narrowing
            assert isinstance(sound.mse_loss_indexes, torch.Tensor)
            assert sound.condition_mask is not None
            preds_sound = self.llm2sound(
                last_hidden_state[sound.mse_loss_indexes]
            )  # [total_noisy_sound_tokens,sound_dim]
            preds_sound = self.unpack_sound_latents(
                preds_sound, sound.token_shapes, sound.noisy_frame_indexes
            )  # list of [C,T] per sample
            output_dict.update(preds_sound=preds_sound)

    def forward(
        self,
        packed_seq: PackedSequence,
        memory: MemoryState | None = None,
        video_temporal_causal: bool | None = None,
    ) -> dict:
        """
        Forward pass for Cosmos3VFMNetwork.

        Args:
            packed_seq: PackedSequence containing all packed tensors and metadata.
                See PackedSequence dataclass for field details.
            memory: Optional MemoryState for persistent KV-cache memory
                (AR inference or rolling-KV-cache training).  Built by
                ``OmniMoTModel.build_memory_state()``.
            video_temporal_causal: Per-call attention-mode override; ``None``
                (default) uses the config-selected ``self.video_temporal_causal``.

        Returns:
            dict with keys:
                - "preds_vision": list[Tensor[C,T,H,W]], one per sample.
                - "preds_lidar": Velocity predictions for LiDAR tokens (if the LiDAR stream is configured).
                - "preds_action": Velocity predictions for action tokens (if action_gen).
                - "preds_sound": Velocity predictions for sound tokens (if sound_gen).
                - "last_hidden_state": Last hidden state from the transformer.
                - "lbl_metadata_*": Load balancing metadata.
                - "ce_preds": Cross-entropy predictions (if predict_text_tokens is True).
        """
        # Note: During inference with @torch.no_grad(), model may be in training mode
        # This is intentional for proper batch norm / dropout behavior
        # assert self.training, "Cosmos3VFMNetwork only supports training mode"

        packed_sequence, target_dtype = self._encode_text(packed_seq)  # packed_sequence: [N_total,hidden_size]

        # encode vision tokens
        original_latent_shapes: List[Tuple[int, int, int]] | None = None
        original_latent_shapes_lidar: List[Tuple[int, int, int]] | None = None
        if self.config.vision_gen:
            original_latent_shapes = self._encode_vision(packed_seq, packed_sequence, target_dtype)

        # encode lidar tokens
        if self.lidar_gen:
            original_latent_shapes_lidar = self._encode_lidar(packed_seq, packed_sequence, target_dtype)

        # encode action tokens
        if self.config.action_gen:
            self._encode_action(packed_seq, packed_sequence, target_dtype)

        # encode sound tokens
        if self.config.sound_gen:
            self._encode_sound(packed_seq, packed_sequence, target_dtype)

        assert packed_seq.attn_modes is not None
        assert packed_seq.split_lens is not None
        use_video_temporal_causal = (
            self.video_temporal_causal if video_temporal_causal is None else video_temporal_causal
        )

        # Get all generation sequence indexes for MoE routing
        # IMPORTANT: Include ALL latent tokens (video + action + sound), not just generation targets.
        # Condition tokens still need to be routed to diffusion experts; they are excluded from
        # LOSS computation, not from routing.
        all_gen_indexes = []
        if packed_seq.vision is not None:
            assert packed_seq.vision.token_shapes is not None
            assert isinstance(packed_seq.vision.sequence_indexes, torch.Tensor)
            all_gen_indexes.append(packed_seq.vision.sequence_indexes)
        if packed_seq.lidar is not None and isinstance(packed_seq.lidar.sequence_indexes, torch.Tensor):
            all_gen_indexes.append(packed_seq.lidar.sequence_indexes)
        if packed_seq.action is not None and isinstance(packed_seq.action.sequence_indexes, torch.Tensor):
            all_gen_indexes.append(packed_seq.action.sequence_indexes)
        if packed_seq.sound is not None and isinstance(packed_seq.sound.sequence_indexes, torch.Tensor):
            all_gen_indexes.append(packed_seq.sound.sequence_indexes)
        vision_sequence_indexes = torch.cat(all_gen_indexes, dim=0) if all_gen_indexes else None  # [N_gen_tokens]

        # When temporal causal is enabled the buffer is [action_t0, vision_t0, action_t1, vision_t1, ...].
        # After torch.cat([vision_indexes, action_indexes]) the interleaved order is lost; sorting restores it.
        if self.video_temporal_causal:
            assert packed_seq.sound is None, "Sound generation is not supported with video_temporal_causal=True."
            if vision_sequence_indexes is not None:
                vision_sequence_indexes = vision_sequence_indexes.sort().values  # [N_gen_tokens]

        # ModalityData.token_shapes is arity-agnostic to cover action and sound as well, but the
        # temporal-causal metadata downstream reads these as (T, H, W); unpacking pins that here.
        vision_token_shapes: list[tuple[int, int, int]] | None = (
            [(t, h, w) for t, h, w in packed_seq.vision.token_shapes] if packed_seq.vision else None
        )

        # The packer is the single source of truth for the supertoken layout.
        # ``num_action_tokens_per_supertoken`` is stamped onto ``packed_seq`` by
        # ``pack_supertokens_temporal_causal`` (= tcf when actions are packed
        # inline, 0 otherwise) and read unchanged by the attention builder, the
        # NATTEN metadata generator, and the rolling KV-cache state — keeping
        # all downstream supertoken geometry automatically in sync with the pack.
        num_action_tokens_per_supertoken = packed_seq.num_action_tokens_per_supertoken

        replicated_attention_io_cp = (
            self.attention_io_layout == "replicated"
            and self.parallel_dims is not None
            and self.parallel_dims.cp_enabled
        )
        # ``sequence_sharded`` attention I/O shards the token sequence, so
        # packing must pad sequence lengths to the CP size and the input/output
        # sequence helpers need the CP mesh.  ``replicated`` attention I/O keeps
        # current-frame sequences replicated and uses the CP mesh later inside
        # attention to slice local heads, so the effective sequence-sharding
        # world size is 1 here.
        sequence_shard_parallel_dims = None if replicated_attention_io_cp else self.parallel_dims
        sequence_shard_world_size = (
            1 if replicated_attention_io_cp else (self.parallel_dims.cp_size if self.parallel_dims else 1)
        )
        prepared_sequence_pack_metadata = packed_seq.get_sequence_pack_metadata()

        input_pack, attention_meta, natten_metadata_list = build_packed_sequence(
            self.config.joint_attn_implementation,
            packed_sequence=packed_sequence,
            attn_modes=packed_seq.attn_modes,
            split_lens=packed_seq.split_lens,
            sample_lens=packed_seq.sample_lens,
            packed_und_token_indexes=packed_seq.text_indexes,
            packed_gen_token_indexes=vision_sequence_indexes,
            num_heads=self.num_heads,
            is_image_batch=packed_seq.is_image_batch,
            head_dim=self.head_dim,
            num_layers=self.num_hidden_layers,
            token_shapes=packed_seq.vision.token_shapes,
            natten_parameter_list=self.natten_parameter_list,
            cp_world_size=sequence_shard_world_size,
            video_temporal_causal=use_video_temporal_causal,
            skip_natten_metadata=memory is not None and not memory.requires_natten_metadata(),
            vision_token_shapes=vision_token_shapes,
            action_token_shapes=packed_seq.action.token_shapes if packed_seq.action else None,
            num_action_tokens_per_supertoken=num_action_tokens_per_supertoken,
            null_action_supertokens=packed_seq.null_action_supertokens,
            pad_for_cuda_graphs=self.pad_for_cuda_graphs,
            full_seq_alignment=self.flex_backend.full_seq_alignment if self.flex_backend else 1,
            causal_seq_alignment=self.flex_backend.causal_seq_alignment if self.flex_backend else 1,
            prepared_metadata=prepared_sequence_pack_metadata,
        )

        # Non-None exactly when use_multiview_flex_attention is on, per the resolution in __init__.
        if self.flex_backend is not None:
            if self.config.joint_attn_implementation != "two_way":
                raise ValueError("Multiview FlexAttention requires joint_attn_implementation='two_way'.")
            # natten_metadata_list is always None here (only the three-way packer builds it), so the
            # conflict to catch is the configured one: NATTEN parameters would be silently ignored.
            if self.natten_parameter_list:
                raise ValueError("Multiview FlexAttention and NATTEN cannot be enabled together.")
            if packed_seq.action is not None or packed_seq.sound is not None:
                raise ValueError(
                    "Multiview FlexAttention supports vision and LiDAR generation batches, not action or sound."
                )
            if packed_seq.vision is None and packed_seq.lidar is None:
                raise ValueError("Multiview FlexAttention needs a vision or LiDAR generation stream.")
            mask_items = _multiview_mask_items(packed_seq)
            full_only_seq, full_q_offsets = get_full_only_seq(input_pack)
            causal_seq, causal_offsets = get_causal_seq(input_pack)
            # The mask is built here, outside the compiled and activation-checkpointed
            # decoder layers, because the build syncs with the host on a data-dependent
            # group count, which Dynamo cannot trace inside the checkpoint HOP. All
            # layers then share the one mask, which is all the attention path needs.
            #
            # GEN tokens are the queries and [UND | GEN] the keys, so the UND stream's
            # padded length and per-sample offsets come along: they are what labels a UND
            # key with its sample, which is the whole of the gen->und rule.
            attention_meta.flex_block_mask = build_multiview_block_mask(
                seq_len=full_only_seq.shape[0],
                full_q_offsets=full_q_offsets,
                items_per_sample=mask_items,
                device=full_only_seq.device,
                block_size=self.flex_backend.block_size,
                num_und=causal_seq.shape[0],
                causal_offsets=causal_offsets,
                attention_scope=self.config.attention_scope,
            )
            # Carried with the mask because its kernels are only valid for the block size the
            # mask was built at; two_way_attention hands both to flex_attention, which
            # checks that agreement before running them.
            attention_meta.flex_backend = self.flex_backend

        # ── Multi-control transfer: annotate SplitInfo with per-item ranges ──────
        # This block is entered for any pack carrying control_weights, single-control
        # included; ``_annotate_multi_control_ranges`` is what narrows it to packs with
        # more than one weight, and leaves the ranges unset otherwise.
        #
        # That distinction decides the routing, so it is not cosmetic. dispatch_attention
        # sends a pack to multi_control_two_way_attention iff control_stream_token_ranges
        # is set, and that path is maskless by construction. A single-control multiview
        # pack must therefore leave the ranges unset and fall through to
        # two_way_attention, which is the only path that applies the multiview flex mask.
        # Annotating it here would silently drop that mask.
        #
        # multi_control_two_way_attention runs N independent maskless SDPA passes,
        # one per control.  For each pass i, KV = [text | ctrl_i | noisy].
        # The final noisy output is the weighted sum of the N pass outputs:
        #   noisy_out = w_1 * noisy_out_1 + ... + w_N * noisy_out_N
        # All SDPA calls are maskless → Flash Attention always active.
        # In the plain dense case, N=1, w=1.0 matches two_way_attention; in a
        # multiview FlexAttention batch, single-control packs must stay unannotated
        # so two_way_attention applies the flex mask.
        #
        # CP compatibility: control_stream_token_ranges are gen-relative global
        # offsets computed here, before CP sharding.  Ulysses CP restores the full
        # sequence on every rank (via all-to-all) before calling dispatch_attention,
        # so the global ranges are valid indices inside multi_control_two_way_attention.
        if (
            isinstance(attention_meta, SplitInfo)
            and packed_seq.control_weights is not None
            and packed_seq.vision_item_split_lens
        ):
            n_gen = int(vision_sequence_indexes.shape[0]) if vision_sequence_indexes is not None else 0
            _annotate_multi_control_ranges(attention_meta, packed_seq, n_gen=n_gen)

        input_pack, packed_position_ids = get_context_parallel_sharded_sequence(
            attn_implementation=self.config.joint_attn_implementation,
            input_pack=input_pack,
            position_ids=packed_seq.position_ids,
            parallel_dims=sequence_shard_parallel_dims,
        )

        packed_outputs, lbl_metadata = self.language_model(
            input_pack,
            attention_mask=attention_meta,
            position_ids=packed_position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
        )
        last_hidden_state = get_context_parallel_last_hidden_state(
            packed_outputs=packed_outputs,
            parallel_dims=sequence_shard_parallel_dims,
        )  # [N_total,hidden_size]
        output_dict = dict()

        # decode vision tokens
        if self.config.vision_gen:
            self._decode_vision(packed_seq, last_hidden_state, output_dict, original_latent_shapes)

        # decode lidar tokens
        if self.lidar_gen:
            self._decode_lidar(packed_seq, last_hidden_state, output_dict, original_latent_shapes_lidar)

        # decode action tokens
        if self.config.action_gen:
            self._decode_action(packed_seq, last_hidden_state, output_dict)

        # decode sound tokens
        if self.config.sound_gen:
            self._decode_sound(packed_seq, last_hidden_state, output_dict)

        output_dict.update(last_hidden_state=last_hidden_state)
        for lbl_metadata_key, lbl_metadata_value in lbl_metadata.items():
            output_dict.update({f"lbl_metadata_{lbl_metadata_key}": lbl_metadata_value})
        if self.predict_text_tokens:
            packed_ce_preds = self.language_model.lm_head(
                last_hidden_state[packed_seq.ce_loss_indexes]
            )  # [N_ce_tokens,vocab_size]
            output_dict["ce_preds"] = packed_ce_preds

        return output_dict


def _annotate_multi_control_ranges(attention_meta: SplitInfo, packed_seq: PackedSequence, *, n_gen: int) -> None:
    """Populate multi-control attention ranges only for true multi-control packs."""
    if packed_seq.control_weights is None or not packed_seq.vision_item_split_lens:
        return
    has_multiple_controls = any(len(weights) > 1 for weights in packed_seq.control_weights)
    if not has_multiple_controls:
        return

    # For multi-control, each sample must have N controls + 1 noisy item
    # (items 0..N-2 are controls, item N-1 is the noisy target).
    # Only batch_size=1 is supported; assert to catch misuse early.
    assert len(packed_seq.vision_item_split_lens) == 1, (
        f"Multi-control transfer requires batch_size=1, got {len(packed_seq.vision_item_split_lens)} samples."
    )
    item_lens = packed_seq.vision_item_split_lens[0]  # [L_ctrl0,L_ctrl1,...,L_noisy]
    weights = packed_seq.control_weights[0]  # [w_ctrl0,w_ctrl1,...]
    assert len(item_lens) > 1, (
        f"Multi-control requires at least 1 control + 1 noisy item; got vision_item_split_lens={item_lens}."
    )
    assert len(weights) == len(item_lens) - 1, (
        f"control_weights length ({len(weights)}) must equal number of control items ({len(item_lens) - 1})."
    )
    ctrl_ranges: list[tuple[int, int]] = []
    cursor = 0
    for lens in item_lens[:-1]:  # all but last = control streams
        ctrl_ranges.append((cursor, cursor + lens))
        cursor += lens
    noisy_range = (cursor, cursor + item_lens[-1])
    assert noisy_range[1] == n_gen, (
        f"vision_item_split_lens sums to {noisy_range[1]} gen tokens but packed tensor has "
        f"{n_gen}; packing inconsistency detected."
    )
    attention_meta.control_stream_token_ranges = ctrl_ranges
    attention_meta.noisy_token_range = noisy_range
    attention_meta.control_weights = weights


def _multiview_mask_items(packed_seq: PackedSequence) -> list[list[MaskItem]]:
    """Describe each sample to the multiview mask as its vision items, then its LiDAR items.

    The packer lays a sample out in exactly that order, so walking the two streams sample by
    sample reproduces the packed order the mask assumes.

    LiDAR items take a view offset past the cameras. A range clip is not one of the rig's
    views, and the offset is what keeps the rules that match on view from pairing a camera
    latent with the sweep that happens to share its frame index -- two different instants,
    since the streams run at different latent rates. A batch with no LiDAR, or a LiDAR-only
    batch, leaves every item on view 0, so its mask is bit-identical to the single-stream
    one.

    Control items are marked per stream, not per sample: within each of the two streams,
    every item but the last is a control item conditioning the one that follows it, which
    is the same convention the packer uses when it forces those items fully clean
    (``packers.py``). Doing it per stream is what keeps a camera item's position from
    deciding whether a LiDAR item is control, and vice versa. A stream contributing a
    single item per sample -- every T2V/I2V/V2V batch, and either stream of a one-item-each
    joint pack -- marks nothing, so ``flex_attention``'s control rules stay unreachable.
    """
    num_samples = len(packed_seq.sample_lens)
    vision = packed_seq.vision
    lidar = packed_seq.lidar
    if vision is None and lidar is None:
        raise ValueError("Multiview FlexAttention needs a vision or LiDAR generation stream.")

    # None means every sample owns exactly one vision item (standard T2V/I2V);
    # multi-item samples (image editing, transfer) carry explicit counts. A
    # LiDAR-only pack has no vision items.
    if vision is None:
        vision_counts = [0] * num_samples
        views_per_vision_item: list[int] = []
    else:
        vision_counts = packed_seq.num_vision_items_per_sample or [1] * num_samples
        views_per_vision_item = list(packed_seq.num_views_per_vision_item or [])
        if not views_per_vision_item:
            if lidar is None:
                raise ValueError(
                    "Multiview FlexAttention requires per-camera VAE metadata; "
                    "enable enable_per_camera_vae_encoding on the dataset."
                )
            # A pack carrying both streams cannot hold that metadata: it is written by the
            # camera-major uint8 encode path, which a range clip never takes. Such a pack is
            # single-camera by construction, so one view per item is the grid the mask needs,
            # and the only thing it needs the count for.
            views_per_vision_item = [1] * sum(vision_counts)

    lidar_counts = [0] * num_samples
    if lidar is not None:
        lidar_counts = packed_seq.num_lidar_items_per_sample or [1] * num_samples
    # Step past the widest camera item so no LiDAR item can land on a camera's view.
    lidar_view_offset = max(views_per_vision_item, default=0)

    items_per_sample: list[list[MaskItem]] = []
    vision_cursor = 0
    lidar_cursor = 0
    for sample_idx in range(num_samples):
        sample_items: list[MaskItem] = []
        num_vision, num_lidar = vision_counts[sample_idx], lidar_counts[sample_idx]
        if vision is not None:
            for item_in_stream in range(num_vision):
                sample_items.append(
                    MaskItem(
                        token_shape=vision.token_shapes[vision_cursor],
                        condition_mask=vision.condition_mask[vision_cursor],
                        num_views=views_per_vision_item[vision_cursor],
                        view_offset=0,
                        is_control=item_in_stream < num_vision - 1,
                    )
                )
                vision_cursor += 1
        if lidar is not None:
            for item_in_stream in range(num_lidar):
                sample_items.append(
                    MaskItem(
                        token_shape=lidar.token_shapes[lidar_cursor],
                        condition_mask=lidar.condition_mask[lidar_cursor],
                        num_views=1,
                        view_offset=lidar_view_offset,
                        is_control=item_in_stream < num_lidar - 1,
                    )
                )
                lidar_cursor += 1
        items_per_sample.append(sample_items)
    return items_per_sample


def _apply_timestep_embeds_to_noisy_tokens(
    packed_tokens: torch.Tensor,
    packed_timestep_embeds: torch.Tensor,
    noisy_frame_indexes: List[torch.Tensor],
    token_shapes: list[tuple[int, ...]],
) -> torch.Tensor:
    """Apply timestep embeddings to noisy tokens.
    Tn is the number of noisy frames for a given sample.
    Tc is the number of clean frames for a given sample.
    T is the total number of frames for a given sample.
    T = Tn + Tc

    Args:
        packed_tokens: The packed tokens to apply timestep embeddings to.
        packed_timestep_embeds: The packed timestep embeddings to apply.
        noisy_frame_indexes: The frame indices to apply timestep embeddings to
            (list of tensors, each with shape (Tn,)).
        token_shapes: The token shapes for each sample. Each entry is a tuple
            shaped like ``(T, ...)`` where trailing dimensions represent the spatial grid.

    Returns:
        The packed tokens with timestep embeddings applied to the noisy tokens.
    """

    # Handle variable token shapes by processing each sample's noisy_frame_indexes individually.
    # The noisy indices are first expanded to cover the entire spatial grid of each frame.
    #
    # For video frames, the spatial grid is (H, W).
    # For action frames, the spatial grid is ().
    # For sound frames, the spatial grid is (1, 1).
    #
    # The noisy indices are then flattened into a single tensor overall. When flattening,
    # we must ensure that the noisy indices from each sample are unique by adding the
    # cumulative sum of the token shapes of previous samples to the noisy indices for
    # a given sample.
    start_noisy_index = 0
    flattened_noisy_frame_indexes = []

    for noisy_indexes_i, token_shape_i in zip(noisy_frame_indexes, token_shapes):
        assert noisy_indexes_i.numel() <= token_shape_i[0]
        spatial_numel_i = math.prod(token_shape_i[1:])
        spatial_indexes_i = torch.arange(spatial_numel_i, device=packed_tokens.device)  # [spatial_numel_i]
        noisy_indexes_i = (
            (noisy_indexes_i * spatial_numel_i).unsqueeze(-1).expand(-1, spatial_numel_i)
        )  # [Tn_i,spatial_numel_i]
        noisy_indexes_i = noisy_indexes_i.clone() + spatial_indexes_i + start_noisy_index  # [Tn_i,spatial_numel_i]
        flattened_noisy_frame_indexes.append(noisy_indexes_i.flatten())  # [Tn_i*spatial_numel_i]
        start_noisy_index += math.prod(token_shape_i)

    flattened_noisy_frame_indexes = torch.cat(flattened_noisy_frame_indexes, dim=0)  # [total_noisy_patches]

    assert packed_tokens.dim() == 2
    assert packed_timestep_embeds.dim() == 2
    assert packed_timestep_embeds.shape[1] == packed_tokens.shape[1]
    assert packed_timestep_embeds.shape[0] <= packed_tokens.shape[0]
    assert flattened_noisy_frame_indexes.dim() == 1
    assert flattened_noisy_frame_indexes.shape[0] == packed_timestep_embeds.shape[0]

    flattened_noisy_frame_indexes = flattened_noisy_frame_indexes.unsqueeze(-1).expand(
        -1,
        packed_tokens.shape[1],
    )  # [total_noisy_patches,hidden_size]

    return packed_tokens.scatter_add(
        dim=0,
        index=flattened_noisy_frame_indexes,
        src=packed_timestep_embeds,
    )  # [total_tokens,hidden_size]
