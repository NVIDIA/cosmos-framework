# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributed import ProcessGroup

from cosmos_framework.model.attention import attention as imaginaire_attention
from cosmos_framework.model.attention.masks import CausalType
from cosmos_framework.utils import log
from cosmos_framework.model.generator.mot.attention import (
    AttentionMaskType,
    dispatch_attention,
)

# Nemotron 3 Dense VL imports
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.configuration_nemotron_3_dense_vl import (
    Nemotron3DenseVLTextConfig,
)
from cosmos_framework.model.generator.reasoner.nemotron_3_dense_vl.nemotron_3_dense_vl import (
    MultiModalRotaryEmbedding,
    Nemotron3DenseVLMLP,
    Nemotron3DenseVLPreTrainedModel,
    Nemotron3DenseVLRMSNorm,
    apply_rotary_pos_emb_partial,
)

# Qwen3-VL imports
from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import (
    Qwen3VLPreTrainedModel,
    Qwen3VLTextMLP,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionModel,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import (
    apply_rotary_pos_emb as qwen3_vl_apply_rotary_pos_emb,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import (
    prepare_multimodal_reasoner_inputs,
)

# Qwen3-VL-MoE imports
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.configuration_qwen3_vl_moe import (
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
)
from cosmos_framework.model.generator.reasoner.qwen3_vl_moe.qwen3_vl_moe import (
    LBLMetadata,
    Qwen3VLMoePreTrainedModel,
    Qwen3VLMoeTextMLP,
    Qwen3VLMoeTextRMSNorm,
    Qwen3VLMoeTextRotaryEmbedding,
    Qwen3VLMoeTextSparseMoeBlock,
    Qwen3VLMoeVisionModel,
)
from cosmos_framework.model.generator.utils.memory import KVToStore, MemoryState, MemoryValue
from cosmos_framework.data.generator.sequence_packing.runtime import (
    SequencePack,
    from_all_seq,
    from_und_gen_splits,
    get_device_and_dtype,
    get_gen_seq,
    get_und_seq,
    set_gen_seq,
    set_und_seq,
    zeros_like,
)

# Torch optimization settings
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 4096

# -----------------------------------------------------------------------------
# Unified MoT (Mixture of Transformers) implementation supporting:
#   - Qwen3-VL Dense, Qwen3-VL MoE, and Nemotron 3 Dense VL
#
# Shared components:
#   - PackedAttentionMoT (config-driven QK norm and RoPE)
#   - MoTDecoderLayer (used by all variants)
#   - _impl_* (shared init/forward)
#
# Variant-specific wrapper classes are needed for different PreTrainedModel bases.
# Sub-layer classes (MLP, RMSNorm, RotaryEmbedding, RoPE fn) are selected via LayerTypes.
# -----------------------------------------------------------------------------


class LayerTypes:
    """Architecture-family dispatch table for the shared MoT layers.

    A single ``LayerTypes(variant)`` instance bundles the four sub-layer
    classes / functions that the MoT decoder layers swap in based on the
    text-model family: the MLP block, the RMSNorm class, the rotary
    embedding module, and the ``apply_rotary_pos_emb`` function.  Passed
    through ``_impl_init`` to ``MoTDecoderLayer`` so a single decoder
    implementation works across all three families.

    Supported variants: ``"qwen3_vl_dense"``, ``"qwen3_vl_moe"``,
    ``"nemotron_dense"``.
    """

    def __init__(self, variant: str):
        self.variant = variant
        if variant == "qwen3_vl_dense":
            self.mlp = Qwen3VLTextMLP
            self.rms_norm = Qwen3VLTextRMSNorm
            self.rotary_embedding = Qwen3VLTextRotaryEmbedding
            self.apply_rotary_pos_emb = qwen3_vl_apply_rotary_pos_emb
        elif variant == "qwen3_vl_moe":
            self.mlp = Qwen3VLMoeTextMLP
            self.rms_norm = Qwen3VLMoeTextRMSNorm
            self.rotary_embedding = Qwen3VLMoeTextRotaryEmbedding
            self.apply_rotary_pos_emb = qwen3_vl_apply_rotary_pos_emb
        elif variant == "nemotron_dense":
            self.mlp = Nemotron3DenseVLMLP
            self.rms_norm = Nemotron3DenseVLRMSNorm
            self.rotary_embedding = MultiModalRotaryEmbedding
            self.apply_rotary_pos_emb = apply_rotary_pos_emb_partial
        else:
            raise ValueError(f"Unknown LayerTypes variant: {variant!r}")

    @property
    def is_moe(self) -> bool:
        return self.variant == "qwen3_vl_moe"


# -----------------------------------------------------------------------------
# MoT wrapper configs — one per architecture family
# -----------------------------------------------------------------------------


class _MoTConfigBase(object):
    """Shared MoT wrapper logic for all three architecture families.

    Concrete subclasses: :class:`Qwen3VLMoTConfig` (dense Qwen3-VL),
    :class:`Qwen3VLMoeMoTConfig` (Qwen3-VL MoE), and
    :class:`Nemotron3DenseVLMoTConfig`.

    Stores the **JSON config dict** (as loaded from the per-family
    config JSON) plus a small fixed set of MoT-specific fields, and
    materializes fresh HF configs on demand via :pyattr:`full_config`,
    :pyattr:`text_config`, and :pyattr:`vision_config`.  Three concerns
    are kept separate:

    1. The HF *text* config that ``*TextModel`` / decoder layers
       actually consume — built from the nested ``text_config``
       sub-section of the JSON (or the flat dict for LLM-only configs),
       optionally massaged by :meth:`_transform_text_dict`, then splatted
       into :pyattr:`_text_config_cls`.  MoT-specific fields are *not*
       merged in here; see point 3.
    2. The HF *vision* sub-section (``config_dict["vision_config"]``)
       that the visual tower needs but the text model must never see —
       surfaced via :pyattr:`vision_config`.
    3. MoT-specific knobs (``qk_norm_for_text`` / ``qk_norm_for_diffusion``,
       ``include_visual``) that are read directly off the wrapper:
       ``qk_norm_for_*`` are passed as constructor kwargs to the
       ``*TextModel`` (and from there to the decoder layers / packed
       attention), and ``include_visual`` gates whether the ViT is built
       at all by the ``*TextForCausalLM`` __init__.

    Constructor surface:

    - ``config_dict``: full JSON dict (nested-VLM with top-level
      ``text_config`` / ``vision_config`` / ``image_token_id``, or flat
      LLM-only).  Stored verbatim under ``self.config_dict`` and never
      mutated; the materialization properties build fresh dicts.
    - MoT kwargs (``qk_norm_for_text``, ``qk_norm_for_diffusion``,
      ``include_visual``): stored as plain attributes.  These do *not*
      flow into :pyattr:`text_config` — the ``*TextForCausalLM.__init__``
      flow reads them directly off the wrapper.
    - ``text_config_overrides``: optional ``{field: value}`` mapping
      that the :pyattr:`text_config` property merges into the text
      dict *after* :meth:`_transform_text_dict` (so SMOKE-style
      shrinks like ``num_hidden_layers=2`` win over both the JSON
      defaults and the Nemotron ``56 -> 28`` fold).  Stored as a
      plain attribute so the ``create_vlm_config`` ``setattr`` flow
      can drop a fresh mapping in post-construction; the next
      ``text_config`` access picks it up.

    Post-construction overrides via plain ``setattr`` (the
    ``create_vlm_config`` flow in ``configs/base/defaults/reasoner.py``)
    just update the same plain attributes, so the next property access
    picks up the latest values.  No cache, no ``__setattr__``
    interception, no override bucket — the property rebuild is cheap
    relative to model init and consumers (``*TextForCausalLM.__init__``)
    read each property once and store the result locally.

    Subclasses customize four things:

    - :pyattr:`_full_config_cls`: HF full-config class (e.g.
      ``Qwen3VLConfig``); used by :pyattr:`full_config` to build the
      base ``PreTrainedModel.config`` for the ``*ForCausalLM`` wrapper.
    - :pyattr:`_text_config_cls`: HF text-config class (e.g.
      ``Qwen3VLTextConfig``); used by :pyattr:`text_config`.
    - :pyattr:`_vision_config_cls`: HF vision-config class for the
      visual tower (only relevant when ``include_visual=True``).
    - :meth:`_transform_text_dict`: optional hook for architecture-
      specific massaging of the text-config dict (e.g. the Nemotron 3
      Dense VL ``56 -> 28`` layer fold).  Default is identity.
    """

    # Concrete HF config classes used at materialization time.
    # Subclasses override these to switch between dense and MoE; the
    # bases default to ``type(None)`` (i.e. ``NoneType``) so mis-
    # configured subclasses fail loudly the moment any of the
    # ``*_config`` properties is read — :pyattr:`full_config` /
    # :pyattr:`text_config` / :pyattr:`vision_config` each detect the
    # ``type(None)`` sentinel and raise ``ValueError`` with an
    # actionable message before the splat-into-NoneType ``TypeError``.
    _full_config_cls: type = type(None)
    _text_config_cls: type = type(None)
    _vision_config_cls: type = type(None)

    def __init__(
        self,
        config_dict: Mapping[str, Any],
        *,
        qk_norm_for_text: bool = True,
        qk_norm_for_diffusion: bool = True,
        include_visual: bool = False,
        gen_noisy_gating: bool = False,
        text_config_overrides: Mapping[str, Any] | None = None,
    ):
        # Defensive copy so downstream materialization can't mutate the
        # caller's input.
        self.config_dict = dict(config_dict)
        self.qk_norm_for_text = qk_norm_for_text
        self.qk_norm_for_diffusion = qk_norm_for_diffusion
        self.include_visual = include_visual
        # Noisy top-k gating on the generation-tower MoE blocks (Shazeer 2017).
        # Gen-tower only; the understanding tower never receives this flag.
        self.gen_noisy_gating = gen_noisy_gating
        # Plain attribute (not a property) so the ``create_vlm_config``
        # post-construction ``setattr`` flow can replace the whole
        # mapping in one shot; default to ``{}`` so the merge in
        # :pyattr:`text_config` is unconditionally a no-op when no
        # overrides were supplied.
        self.text_config_overrides: dict[str, Any] = dict(text_config_overrides) if text_config_overrides else {}

    @property
    def full_config(self) -> Any:
        """Materialize the full HF LLM/VLM config from the stored dict.

        Splats :pyattr:`config_dict` directly into :pyattr:`_full_config_cls`.
        Used by ``*TextForCausalLM.__init__`` to build the
        ``PreTrainedModel.config`` that the HF base class stores under
        ``self.config``.
        """
        if self._full_config_cls is type(None):
            raise ValueError(f"No _full_config_cls defined for {self.__class__.__name__}")
        return self._full_config_cls(**self.config_dict)

    @property
    def text_config(self) -> Any:
        """Materialize a fresh HF text config from the stored dict.

        Built from scratch on every access — cheap relative to model
        init and naturally correct under the ``create_vlm_config``
        post-init ``setattr`` flow.  Construction steps:

        1. Pick the source dict: the ``text_config`` sub-section of
           :pyattr:`config_dict` when nested, or the dict itself for
           flat / LLM-only configs.
        2. Run :meth:`_transform_text_dict` on it (default identity;
           the Nemotron variant uses it for the ``56 -> 28`` layer
           fold).
        3. Merge :pyattr:`text_config_overrides` on top (last write
           wins) so caller-supplied per-field overrides (SMOKE
           shrinks like ``num_hidden_layers=2``) beat both the JSON
           defaults and the Nemotron transform.  Applied as a
           post-transform overlay rather than a pre-transform merge
           so transform-aware overrides (final layer counts) don't
           get re-halved by the Nemotron fold.
        4. Splat into :pyattr:`_text_config_cls`.

        MoT-specific fields (``qk_norm_for_*``, ``include_visual``) are
        deliberately *not* injected into the returned HF text config —
        they're read directly off the wrapper by
        ``*TextForCausalLM.__init__`` and forwarded as ``*TextModel``
        constructor kwargs.  Likewise ``tie_word_embeddings`` is only
        consumed via :pyattr:`full_config` (it lives at the JSON
        top-level for nested VLM configs and is not duplicated into the
        text sub-section).
        """
        # Nested VLM configs put the text-tower fields under
        # ``text_config``; flat LLM-only configs (e.g. Qwen3-0.6B.json)
        # carry them at the top level.  A single ``isinstance`` check
        # keeps both paths working without an external helper.
        nested = self.config_dict.get("text_config")
        text_dict = nested if isinstance(nested, dict) else self.config_dict
        # Subclass hook (default identity) — gives architecture-specific
        # variants a single seam to massage the dict before HF
        # instantiation.  Used today by the Nemotron variant for the
        # 56 -> 28 layer fold.
        text_dict = self._transform_text_dict(text_dict)
        # Caller-supplied per-field overrides (SMOKE knobs, debug
        # shrinks).  ``getattr`` default keeps us robust against
        # ``setattr(wrapper, "text_config_overrides", None)`` from
        # the ``create_vlm_config`` flow — None means "no overrides".
        overrides = getattr(self, "text_config_overrides", None) or {}
        if overrides:
            text_dict = {**text_dict, **overrides}

        if self._text_config_cls is type(None):
            raise ValueError(f"No _text_config_cls defined for {self.__class__.__name__}")
        return self._text_config_cls(**text_dict)

    def _transform_text_dict(self, text_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        """Subclass hook: transform the text-config dict before MoT fields are merged.

        Default implementation is identity.  Override in subclasses to
        apply architecture-specific massaging (e.g. the Nemotron 3
        Dense VL ``56 -> 28`` layer fold).  The hook receives the dict
        already split out of the nested ``text_config`` sub-section
        (or the flat dict for LLM-only configs) and must return a
        dict-like object suitable for splatting into
        :pyattr:`_text_config_cls`.
        """
        return text_dict

    @property
    def vision_config(self) -> Any | None:
        """Materialize the HF vision config for the visual tower, or None.

        Mirrors :pyattr:`text_config`'s "build on demand" pattern:
        gates on :pyattr:`include_visual`, splats the JSON's
        ``vision_config`` sub-section into :pyattr:`_vision_config_cls`,
        and returns the built HF config (caller plugs it into
        ``Qwen3VL{,Moe}VisionModel._from_config``).

        Returns ``None`` when ``include_visual=False``.  When
        ``include_visual=True`` but the JSON has no ``vision_config``
        sub-section (e.g. an LLM-only checkpoint mistakenly used as a
        VLM), raises ``ValueError`` eagerly rather than silently
        building an empty ViT.
        """
        if not self.include_visual:
            return None
        vision_dict = self.config_dict.get("vision_config")
        if vision_dict is None:
            raise ValueError(
                "include_visual=True requires a vision_config sub-section in the language-model JSON config."
            )
        if self._vision_config_cls is type(None):
            raise ValueError(f"No _vision_config_cls defined for {self.__class__.__name__}")
        return self._vision_config_cls(**vision_dict)

    @classmethod
    def from_json_file(cls, json_file: str) -> "_MoTConfigBase":
        """Load the JSON file verbatim and wrap it.

        The full dict is stored under ``self.config_dict``; sibling
        ``vision_config`` / ``image_token_id`` / ``vision_*_token_id``
        fields (when present) are surfaced lazily via
        :pyattr:`vision_config` and by HF downstream consumers reading
        the dict directly.
        """
        with open(json_file, encoding="utf-8") as reader:
            config_dict = json.load(reader)
        return cls(config_dict=config_dict)


class Qwen3MoTConfig(_MoTConfigBase):
    """MoT wrapper config for the Qwen3 family."""

    _full_config_cls = Qwen3VLTextConfig
    _text_config_cls = Qwen3VLTextConfig


class Qwen3VLMoTConfig(_MoTConfigBase):
    """MoT wrapper config for the dense Qwen3-VL family."""

    _full_config_cls = Qwen3VLConfig
    _text_config_cls = Qwen3VLTextConfig
    _vision_config_cls = Qwen3VLVisionConfig


class Qwen3VLMoeMoTConfig(_MoTConfigBase):
    """MoT wrapper config for the Qwen3-VL MoE family."""

    _full_config_cls = Qwen3VLMoeConfig
    _text_config_cls = Qwen3VLMoeTextConfig
    _vision_config_cls = Qwen3VLMoeVisionConfig


class Nemotron3DenseVLMoTConfig(_MoTConfigBase):
    """MoT wrapper config for the Nemotron 3 Dense VL family.

    Inherits the shared ``config_dict``-storing wrapper, MoT field
    handling, nested-vs-flat ``text_config`` resolution, and
    ``from_json_file`` loader from :class:`_MoTConfigBase`.  Only two
    things are Nemotron-specific:

    1. :pyattr:`_text_config_cls` — the materialized HF text config
       class is :class:`Nemotron3DenseVLTextConfig`.
    2. :meth:`_transform_text_dict` — folds the upstream VLM's
       alternating ``(attn, MLP)`` blocks (``num_hidden_layers == 56``)
       into MoT's fused decoder layers (28 effective).

    Nemotron 3 Dense VL has no nested ``vision_config`` sub-section in
    its JSON, so the inherited ``vision_config`` property is unused —
    callers leave the inherited ``include_visual`` at its default of
    ``False`` and never access the property.
    """

    _full_config_cls = Nemotron3DenseVLTextConfig
    _text_config_cls = Nemotron3DenseVLTextConfig

    def _transform_text_dict(self, text_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        if text_dict.get("num_hidden_layers") == 56:
            # Upstream VLM stores attention and MLP as separate
            # alternating blocks (56 total); MoT combines each pair
            # into a single transformer layer (28 effective).
            return {**text_dict, "num_hidden_layers": 28}
        return text_dict


# -----------------------------------------------------------------------------
# Common layers between Qwen3VL Dense, MoE, and Nemotron 3 Dense VL models
# -----------------------------------------------------------------------------


def _apply_head_sharded_o_proj(
    local_attn_output: torch.Tensor,  # [N,H_local*D]
    projection: nn.Linear,
    feature_slice: slice,
    cp_group: ProcessGroup,
) -> torch.Tensor:  # [N,hidden_size]
    """Apply one local input-column slice of ``projection`` and sum partial outputs."""
    local_weight = projection.weight[:, feature_slice]  # [hidden_size,H_local*D]
    out = torch.nn.functional.linear(local_attn_output, local_weight, bias=None)  # [N,hidden_size]
    torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM, group=cp_group)
    if projection.bias is not None:
        out = out + projection.bias  # [N,hidden_size]
    return out


class PackedAttentionMoT(nn.Module):
    """
    Dual-pathway packed attention for MoT architectures.
    Implements understanding and generation pathways with separate projections.

    Used for Qwen3VL (Dense), Qwen3VL-MoE, and Nemotron 3 Dense VL variants.
    QK normalisation and RoPE function are selected via ``layer_types`` and the
    ``qk_norm_for_text`` config attribute (the generation pathway always installs
    QK norm).
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig | Qwen3VLMoeTextConfig | Nemotron3DenseVLTextConfig,
        *,
        layer_idx: int,
        layer_types: LayerTypes,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
        use_und_k_norm_for_gen: bool = False,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        eps = config.rms_norm_eps

        # Understanding pathway projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # Understanding pathway QK norm
        if qk_norm_for_text:
            self.q_norm = layer_types.rms_norm(self.head_dim, eps=eps)
            self.k_norm = layer_types.rms_norm(self.head_dim, eps=eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        # Generation pathway QK norm
        if qk_norm_for_diffusion:
            self.q_norm_moe_gen = layer_types.rms_norm(self.head_dim, eps=eps)
            self.k_norm_moe_gen = layer_types.rms_norm(self.head_dim, eps=eps)
        else:
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        # Cross-attention K norm: normalises und K tokens seen by the generator in the
        # gen→und cross-attention path.  Only needed when the generation pathway has QK
        # norm (qk_norm_for_diffusion=True) but the understanding pathway does not
        # (qk_norm_for_text=False), i.e. the Nemotron-3 configuration.  Without this,
        # the joint softmax computes norm(Q_gen) · K_und_raw^T where K_und_raw has large
        # uncontrolled magnitude and dominates attention over the gen self-attention path.
        # When both pathways share the same QK norm (or neither has one) k_norm_und_for_gen
        # is None and the standard packed K tensor is used for all paths unchanged.
        if use_und_k_norm_for_gen and qk_norm_for_diffusion and not qk_norm_for_text:
            self.k_norm_und_for_gen: nn.Module | None = layer_types.rms_norm(self.head_dim, eps=eps)
        else:
            self.k_norm_und_for_gen = None

        # Generation pathway linear projections
        self.q_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj_moe_gen = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj_moe_gen = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )

        self._apply_rotary_pos_emb = layer_types.apply_rotary_pos_emb
        self.dispatch_attention_fn = dispatch_attention
        self.replicated_attention_io_local_head_o_proj = False
        self.replicated_attention_io_cp_mesh: Any | None = None

    def _replicated_attention_io_q_feature_slice(self) -> slice:
        cp_mesh = self.replicated_attention_io_cp_mesh
        assert cp_mesh is not None, "replicated attention I/O requires a CP mesh"
        cp_group = cp_mesh.get_group()
        cp_rank = torch.distributed.get_rank(cp_group)
        cp_size = torch.distributed.get_world_size(cp_group)
        assert self.num_key_value_heads % cp_size == 0, (
            f"cp_size({cp_size}) must divide num_key_value_heads({self.num_key_value_heads})"
        )
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads({self.num_attention_heads}) must be divisible by "
            f"num_key_value_heads({self.num_key_value_heads})"
        )
        kv_heads_per_rank = self.num_key_value_heads // cp_size
        q_heads_per_kv_head = self.num_attention_heads // self.num_key_value_heads
        q_heads_per_rank = kv_heads_per_rank * q_heads_per_kv_head
        q_start = cp_rank * q_heads_per_rank
        q_end = q_start + q_heads_per_rank
        return slice(q_start * self.head_dim, q_end * self.head_dim)

    def _uses_replicated_attention_io_local_head_o_proj(self, memory_value: MemoryValue | None) -> bool:
        return (
            self.replicated_attention_io_local_head_o_proj
            and memory_value is not None
            and getattr(memory_value, "frame_idx", 0) > 0
            and not getattr(memory_value, "for_cuda_graphs", False)
        )

    def forward(
        self,
        pack: SequencePack,
        attention_mask: AttentionMaskType,
        packed_position_embeddings: tuple[SequencePack, SequencePack],
        natten_metadata: dict | None = None,
        memory_value: MemoryValue | None = None,
    ) -> tuple[SequencePack, KVToStore | None]:
        """Forward pass with optional memory-augmented attention.

        When ``memory_value`` is provided, ``dispatch_attention_fn`` routes to
        the appropriate attention kernel (e.g. three-way KV-cache attention
        for training, or AR inference concat + dense attention).

        ``kv_to_store`` is produced when ``memory_value`` is present:
        ``(gen_k, gen_v, und_k, und_v)`` for the caller to write back via
        ``MemoryState.write_for_layer()``.  The tensors are passed with
        gradients attached; each ``MemoryState`` decides whether to detach
        (e.g. for truncated BPTT) or keep gradients (e.g. teacher forcing).

        Args:
            pack: Packed sequence with und/gen tokens
            attention_mask: Attention metadata (SplitInfo).
            packed_position_embeddings: RoPE embeddings (cos, sin)
            natten_metadata: Optional NATTEN metadata for neighborhood attention.
            memory_value: Optional read-only tensor container for memory-augmented attention.
        """

        q_und_in = self.q_proj(get_und_seq(pack))  # [N_und,num_heads*head_dim]
        q_gen_in = self.q_proj_moe_gen(get_gen_seq(pack))  # [N_gen,num_heads*head_dim]

        k_und_in = self.k_proj(get_und_seq(pack))  # [N_und,num_kv_heads*head_dim]
        k_gen_in = self.k_proj_moe_gen(get_gen_seq(pack))  # [N_gen,num_kv_heads*head_dim]

        v_und_in = self.v_proj(get_und_seq(pack))  # [N_und,num_kv_heads*head_dim]
        v_gen_in = self.v_proj_moe_gen(get_gen_seq(pack))  # [N_gen,num_kv_heads*head_dim]

        q_und = q_und_in.view(-1, self.num_attention_heads, self.head_dim)  # [N_und,num_heads,head_dim]
        k_und = k_und_in.view(-1, self.num_key_value_heads, self.head_dim)  # [N_und,num_kv_heads,head_dim]
        v_und = v_und_in.view(-1, self.num_key_value_heads, self.head_dim)  # [N_und,num_kv_heads,head_dim]

        q_gen = q_gen_in.view(-1, self.num_attention_heads, self.head_dim)  # [N_gen,num_heads,head_dim]
        k_gen = k_gen_in.view(-1, self.num_key_value_heads, self.head_dim)  # [N_gen,num_kv_heads,head_dim]
        v_gen = v_gen_in.view(-1, self.num_key_value_heads, self.head_dim)  # [N_gen,num_kv_heads,head_dim]

        q_und = self.q_norm(q_und)  # [N_und,num_heads,head_dim]
        k_und = self.k_norm(k_und)  # [N_und,num_kv_heads,head_dim]

        q_gen = self.q_norm_moe_gen(q_gen)  # [N_gen,num_heads,head_dim]
        k_gen = self.k_norm_moe_gen(k_gen)  # [N_gen,num_kv_heads,head_dim]

        packed_cos = packed_position_embeddings[0]
        packed_sin = packed_position_embeddings[1]

        q_und_, k_und_ = self._apply_rotary_pos_emb(
            q_und,
            k_und,
            get_und_seq(packed_cos),
            get_und_seq(packed_sin),
            unsqueeze_dim=1,
        )  # q_und_: [N_und,num_heads,head_dim], k_und_: [N_und,num_kv_heads,head_dim]
        q_gen_, k_gen_ = self._apply_rotary_pos_emb(
            q_gen,
            k_gen,
            get_gen_seq(packed_cos),
            get_gen_seq(packed_sin),
            unsqueeze_dim=1,
        )  # q_gen_: [N_gen,num_heads,head_dim], k_gen_: [N_gen,num_kv_heads,head_dim]

        packed_query_states_ = from_und_gen_splits(q_und_, q_gen_, pack)  # [N_und+N_gen,num_heads,head_dim]
        packed_key_states_ = from_und_gen_splits(k_und_, k_gen_, pack)  # [N_und+N_gen,num_kv_heads,head_dim]
        packed_value_states_ = from_und_gen_splits(v_und, v_gen, pack)  # [N_und+N_gen,num_kv_heads,head_dim]

        # Build a separate K pack where the und tokens are normalised for gen→und
        # cross-attention (fixes scale mismatch when qk_norm_for_diffusion=True but
        # qk_norm_for_text=False, i.e. the Nemotron-3 config).  The raw k_und_ is kept
        # in packed_key_states_ for the reasoner's own causal self-attention.
        if self.k_norm_und_for_gen is not None:
            k_und_normalized = self.k_norm_und_for_gen(k_und)  # RMSNorm before RoPE
            _, k_und_for_gen_ = self._apply_rotary_pos_emb(
                q_und_,  # dummy q — already RoPE-applied, output discarded; only k is used
                k_und_normalized,
                get_und_seq(packed_cos),
                get_und_seq(packed_sin),
                unsqueeze_dim=1,
            )  # k_und_for_gen_: [N_und,num_kv_heads,head_dim]
            packed_key_states_normalized_: SequencePack | None = from_und_gen_splits(k_und_for_gen_, k_gen_, pack)
        else:
            packed_key_states_normalized_ = None

        packed_attn_output, kv_to_store = self.dispatch_attention_fn(
            packed_query_states_,
            packed_key_states_,
            packed_value_states_,
            attention_mask,
            natten_metadata=natten_metadata,
            memory_value=memory_value,
            packed_key_states_normalized=packed_key_states_normalized_,
        )

        # Produce kv_to_store for MemoryState.write_for_layer() when the
        # dispatch didn't already provide one (e.g. standard or AR frame-0
        # non-CP paths).  CP dispatch returns local KV-head kv_to_store
        # directly, so kv_to_store is already non-None in that case.
        #
        # Gradient detach is NOT done here; each MemoryState.write_for_layer()
        # decides its own gradient policy (e.g. detach for truncated BPTT,
        # keep gradients for teacher forcing).
        if memory_value is not None and kv_to_store is None:
            und_len = pack["_num_causal_tokens"]
            gen_len = pack["_num_full_tokens"]
            # When und K-norm is active, AR frame 1+ gen→und cross-attention uses
            # the normalised K, so cache k_und_for_gen_ (RMSNorm+RoPE applied) instead
            # of raw k_und_.  Without the norm, k_und_for_gen_ is not defined, so
            # fall back to k_und_.
            k_und_to_store = k_und_for_gen_ if self.k_norm_und_for_gen is not None else k_und_
            kv_to_store = (
                k_gen_[:gen_len].unsqueeze(0),
                v_gen[:gen_len].unsqueeze(0),
                k_und_to_store[:und_len].unsqueeze(0),
                v_und[:und_len].unsqueeze(0),
            )

        # Attention compute is local-head under both sequence-sharded and
        # replicated attention I/O layouts.  The difference here is the output
        # layout returned to this module.  Replicated attention I/O returns only
        # this rank's local heads from AR frame 1+ attention:
        # gen [N_gen,H_local*D] and und [0,H_local*D].  We therefore apply the
        # matching o_proj input-column slice and all-reduce partial outputs back
        # to replicated hidden states.  The else path receives full attention
        # heads at this boundary, so regular o_proj applies:
        # und [N_und,H*D] -> [N_und,hidden_size],
        # gen [N_gen,H*D] -> [N_gen,hidden_size].
        if self._uses_replicated_attention_io_local_head_o_proj(memory_value):
            local_und_attn = get_und_seq(packed_attn_output)  # [0,H_local*D]
            local_gen_attn = get_gen_seq(packed_attn_output)  # [N_gen,H_local*D]
            assert local_und_attn.shape[0] == 0, "replicated attention I/O only supports gen-only frame 1+ attention"
            feature_slice = self._replicated_attention_io_q_feature_slice()
            assert feature_slice.start is not None and feature_slice.stop is not None
            expected_local_features = feature_slice.stop - feature_slice.start
            assert local_gen_attn.shape[-1] == expected_local_features, (
                f"Expected local attention features {expected_local_features}, got {local_gen_attn.shape[-1]}"
            )
            cp_mesh = self.replicated_attention_io_cp_mesh
            assert cp_mesh is not None, "replicated attention I/O requires a CP mesh"
            cp_group = cp_mesh.get_group()
            und_seq = local_gen_attn.new_empty((0, self.hidden_size))  # [0,hidden_size]
            gen_seq = _apply_head_sharded_o_proj(
                local_gen_attn,
                self.o_proj_moe_gen,
                feature_slice,
                cp_group,
            )  # [N_gen,hidden_size]
        else:
            und_seq = self.o_proj(get_und_seq(packed_attn_output))  # [N_und,hidden_size]
            gen_seq = self.o_proj_moe_gen(get_gen_seq(packed_attn_output))  # [N_gen,hidden_size]
        return from_und_gen_splits(und_seq, gen_seq, pack), kv_to_store  # [N_und+N_gen,hidden_size]

    def reasoner_forward(
        self,
        hidden_states: torch.Tensor,  # [B,T,hidden_size]
        cos: torch.Tensor,  # [B,T,head_dim]
        sin: torch.Tensor,  # [B,T,head_dim]
        cache: "ReasonerKVCache | None",
        layer_idx: int,
    ) -> torch.Tensor:
        """Run the reasoner (und) projections with a per-layer KV cache.

        Operates in standard ``[B, T, hidden_size] -> [B, T, hidden_size]``
        layout (rather than the factored sequence-packed layout used during
        training in :meth:`forward`) so it can drive a per-layer KV cache
        in a clean AR loop.

        All attention compute is dispatched through
        ``cosmos_framework.model.attention.attention`` (per repo policy) which expects the
        heads-last contiguous layout ``[B, S, H, D]`` and natively handles GQA
        (``H_KV != H``) — no manual head expansion is needed.

        Causal masking semantics:
            - Prefill (``T>1`` with empty cache): ``is_causal=True`` with
              ``CausalType.DontCare`` (``seq_q == seq_kv``).
            - Decode (``T==1`` with non-empty cache): ``is_causal=False``;
              the single query attends to all cached keys (it is always the
              rightmost position, so no masking is needed).

        Multi-token incremental prefill on top of an existing cache is not
        supported here — ``_impl_generate_reasoner_text`` never triggers it.
        """
        B, T, _ = hidden_states.shape
        H = self.num_attention_heads
        H_kv = self.num_key_value_heads
        D = self.head_dim

        q = self.q_proj(hidden_states).view(B, T, H, D)  # [B,T,num_heads,head_dim]
        k = self.k_proj(hidden_states).view(B, T, H_kv, D)  # [B,T,num_kv_heads,head_dim]
        v = self.v_proj(hidden_states).view(B, T, H_kv, D)  # [B,T,num_kv_heads,head_dim]

        # qk_norm_for_text=False -> q_norm/k_norm are nn.Identity().
        q = self.q_norm(q)
        k = self.k_norm(k)

        # ``apply_rotary_pos_emb`` uses ``unsqueeze_dim`` to broadcast cos/sin
        # over the heads.  q/k are [B, T, n_heads, head_dim] so ``unsqueeze_dim=2``.
        q, k = self._apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        # q: [B,T,num_heads,head_dim], k: [B,T,num_kv_heads,head_dim]

        # The KV cache stores tensors in the same BSHD layout that
        # ``cosmos_framework.model.attention.attention`` expects, with the seq dim at axis 1.
        if cache is not None:
            k_full, v_full = cache.update(layer_idx, k, v)
        else:
            k_full, v_full = k, v
        # k_full / v_full: [B, T_total, num_kv_heads, head_dim]

        # is_causal must only be True when seq_q == seq_kv (prefill).  For
        # decode (T=1, T_total > 1) the single query at the rightmost position
        # attends to every cached key without any mask.
        is_causal = T > 1 and k_full.shape[1] == T

        out = imaginaire_attention(
            query=q,  # [B,T,num_heads,head_dim]
            key=k_full,  # [B,T_total,num_kv_heads,head_dim]
            value=v_full,  # [B,T_total,num_kv_heads,head_dim]
            is_causal=is_causal,
            causal_type=CausalType.DontCare if is_causal else None,
            scale=self.scaling,
        )  # [B,T,num_heads,head_dim] (return_lse=False -> single Tensor)

        return self.o_proj(out.reshape(B, T, H * D))  # type: ignore[union-attr]  # [B,T,hidden_size]


def _impl_init(
    self,
    config: Qwen3VLTextConfig | Qwen3VLMoeTextConfig | Nemotron3DenseVLTextConfig,
    *,
    layer_types: LayerTypes,
    qk_norm_for_text: bool,
    qk_norm_for_diffusion: bool,
    gen_noisy_gating: bool = False,
    use_und_k_norm_for_gen: bool = False,
):
    """Shared ``__init__`` body for the three MoT text-model variants.

    Used by ``Qwen3VLTextModel``, ``Qwen3VLMoeTextModel``, and
    ``Nemotron3DenseVLTextModel``.  Sub-layer classes (MLP, RMSNorm,
    rotary embedding) are dispatched through ``layer_types``.
    """
    self.padding_idx = getattr(config, "pad_token_id", None)
    self.vocab_size = config.vocab_size

    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

    self.layers = nn.ModuleList()
    for layer_idx in range(config.num_hidden_layers):
        self.layers.append(
            MoTDecoderLayer(
                config=config,
                layer_types=layer_types,
                layer_idx=layer_idx,
                qk_norm_for_text=qk_norm_for_text,
                qk_norm_for_diffusion=qk_norm_for_diffusion,
                gen_noisy_gating=gen_noisy_gating,
                use_und_k_norm_for_gen=use_und_k_norm_for_gen,
            )
        )

    # Reasoner-pathway final norm.
    self.norm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
    # Generation-pathway final norm (parallel to ``self.norm``).
    self.norm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    # Rotary embedding (text-only optimized)
    self.rotary_emb = layer_types.rotary_embedding(config)

    # ``post_init`` is provided by each subclass's PreTrainedModel base
    # (Qwen3VL / Qwen3VLMoe / Nemotron3DenseVL).
    self.post_init()


def _impl_init_taylorseer(self, cache_dic=None, current=None):
    """Initialize TaylorSeer acceleration attributes.

    Shared implementation for ``init_taylorseer`` on
    ``Qwen3VLTextModel``, ``Qwen3VLMoeTextModel``, and
    ``Nemotron3DenseVLTextModel``.
    """
    self.cache_dic = cache_dic or {}
    self.current = current or {
        "step": 0,
        "type": "full",
        "stream": "layers_stream",
        "layer": 0,
        "module": "total",
        "activated_steps": [0],
    }
    # Enable TaylorSeer flag
    self.enable_taylorseer = True


def _impl_forward(
    self,
    pack: SequencePack,
    attention_mask,
    position_ids: torch.Tensor,
    natten_metadata_list: list | None = None,
    memory: MemoryState | None = None,
) -> tuple[SequencePack, dict[str, LBLMetadata]]:
    """Shared training forward pass for the three MoT text models.

    Used by ``Qwen3VLTextModel``, ``Qwen3VLMoeTextModel``, and
    ``Nemotron3DenseVLTextModel``.

    Args:
        pack: Packed sequence with und/gen tokens.
        attention_mask: Attention metadata (SplitInfo).
        position_ids: Position IDs (1D ``[N]`` for standard RoPE or 2D
            ``[3, N]`` for mrope).
        natten_metadata_list: Optional per-layer NATTEN metadata.
        memory: Optional ``MemoryState`` for persistent memory across
            forward passes.
    """

    # Create position embeddings (Qwen3 style) - squeeze once at model level
    # tensor below is only used for its dtype and device
    device, dtype = get_device_and_dtype(pack)
    _meta_tensor = torch.tensor([], dtype=dtype, device=device)
    cos, sin = self.rotary_emb(
        _meta_tensor, position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1)
    )  # if ndim == 2, the mrope position_ids is (3, seq_len); inject the batch dim in the
    # middle to get (3, 1, seq_len) so the rotary_emb's mrope branch broadcasts correctly.
    # In both branches Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope collapses the
    # T/H/W axis, so cos / sin always come back as [1, N, head_dim].
    cos = cos.squeeze(0)  # [N,head_dim]
    sin = sin.squeeze(0)  # [N,head_dim]
    position_embeddings = (
        from_all_seq(cos, pack),
        from_all_seq(sin, pack),
    )

    # Tracking the load balancing loss across all layers. For dense models, lbl_metadata_all
    # will be a dictionary with empty lists for each pathway. For MoE models, the lists
    # for each pathway will be populated with the load balancing loss metadata for each layer.
    lbl_metadata_all = dict(und=[], gen=[])

    hidden_states = pack

    # --- MemoryState: per-step init (outside compile) ---
    if memory is not None:
        memory.init(hidden_states, device)

    # Derive gen_only once (outside compile) if using MemoryState
    memory_gen_only = memory.is_gen_only() if memory is not None else False

    for i, decoder_layer in enumerate(self.layers):
        # MemoryState: produce read-only MemoryValue for this layer (outside compile)
        memory_value = memory.read_for_layer(i) if memory is not None else None

        hidden_states, lbl_metadata_dict, kv_to_store = decoder_layer(
            hidden_states,
            attention_mask,
            position_embeddings,
            natten_metadata=None if natten_metadata_list is None else natten_metadata_list[i],
            memory_value=memory_value,
            gen_only=memory_gen_only,
        )

        # MemoryState: store K/V produced by this layer (outside compile)
        if kv_to_store is not None and memory is not None:
            memory.write_for_layer(i, kv_to_store)

        for pathway, lbl_metadata in lbl_metadata_dict.items():
            lbl_metadata_all[pathway].append(lbl_metadata)

    # Compute the load balancing loss across all layers. For dense models, final_lbl_metadata
    # will be an empty dictionary. For MoE models, it will be a dictionary with the stacked
    # load balancing loss metadata for each pathway.
    final_lbl_metadata: dict[str, LBLMetadata] = dict()
    for pathway, lbl_metadata_list in lbl_metadata_all.items():
        if len(lbl_metadata_list) > 0:
            num_tokens_per_expert = torch.stack(
                [lbl_metadata.num_tokens_per_expert for lbl_metadata in lbl_metadata_list]
            )  # [num_layers,num_experts]
            num_tokens = torch.stack([lbl_metadata.num_tokens for lbl_metadata in lbl_metadata_list])  # [num_layers]
            mean_router_prob_per_expert = torch.stack(
                [lbl_metadata.mean_router_prob_per_expert for lbl_metadata in lbl_metadata_list]
            )  # [num_layers,num_experts]
            final_lbl_metadata[pathway] = LBLMetadata(
                num_tokens_per_expert=num_tokens_per_expert,
                num_tokens=num_tokens,
                mean_router_prob_per_expert=mean_router_prob_per_expert,
            )

    hidden_states_out = zeros_like(hidden_states)
    set_und_seq(hidden_states_out, self.norm(get_und_seq(hidden_states)))  # [N_und,hidden_size]
    set_gen_seq(hidden_states_out, self.norm_moe_gen(get_gen_seq(hidden_states)))  # [N_gen,hidden_size]

    return hidden_states_out, final_lbl_metadata


def _run_mlp(
    mlp: torch.nn.Module,
    input: torch.Tensor,
) -> tuple[torch.Tensor, LBLMetadata | None]:
    """Run an MLP block and normalize the return shape across dense / MoE.

    ``Qwen3VLMoeTextSparseMoeBlock`` returns ``(Tensor, LBLMetadata)`` so the
    caller can aggregate the load-balancing-loss metadata across layers; every
    other MLP block (dense ``*MLP``) returns just the output tensor.  This
    helper unifies both into a single ``(output, lbl_metadata_or_None)`` shape
    so the decoder layers and the reasoner-tower forward don't need to branch
    on the MLP type.
    """
    if isinstance(mlp, Qwen3VLMoeTextSparseMoeBlock):
        (
            output_tensor,
            lbl_metadata,
        ) = mlp(input)
    else:
        output_tensor = mlp(input)
        lbl_metadata = None
    return output_tensor, lbl_metadata


class MoTDecoderLayer(nn.Module):
    """
    Unified MoT (Mixture of Transformers) decoder layer.
    Features dual-pathway attention for understanding vs generation.

    This is used for both Dense and MoE models.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig | Qwen3VLMoeTextConfig | Nemotron3DenseVLTextConfig,
        *,
        layer_idx: int,
        layer_types: LayerTypes,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
        gen_noisy_gating: bool = False,
        use_und_k_norm_for_gen: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = PackedAttentionMoT(
            config,
            layer_types=layer_types,
            layer_idx=layer_idx,
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
            use_und_k_norm_for_gen=use_und_k_norm_for_gen,
        )

        if (
            hasattr(config, "mlp_only_layers")
            and (layer_idx not in config.mlp_only_layers)
            and (config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0)
        ):
            self.mlp = Qwen3VLMoeTextSparseMoeBlock(config)
            # Noisy gating is gen-tower only.
            self.mlp_moe_gen = Qwen3VLMoeTextSparseMoeBlock(config, noisy_gating=gen_noisy_gating)
        else:
            self.mlp = layer_types.mlp(config)
            self.mlp_moe_gen = layer_types.mlp(config)

        self.input_layernorm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = layer_types.rms_norm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input: SequencePack,
        attention_mask,
        packed_position_embeddings: tuple[SequencePack, SequencePack],
        natten_metadata: dict | None = None,
        memory_value: MemoryValue | None = None,
        gen_only: bool = False,
    ) -> tuple[SequencePack, dict[str, LBLMetadata], KVToStore | None]:
        """Forward pass with MoT routing and optional memory-augmented attention.

        Returns a 3-tuple: ``(hidden_states, lbl_metadata_dict, kv_to_store)``.
        ``kv_to_store`` is non-None when ``memory_value`` is provided,
        containing ``(gen_k, gen_v, und_k, und_v)`` to be written back by
        ``MemoryState.write_for_layer()`` outside the ``torch.compile``
        boundary.

        Args:
            input: Packed sequence with und/gen tokens
            attention_mask: Attention mask
            packed_position_embeddings: RoPE embeddings (cos, sin)
            natten_metadata: Optional NATTEN metadata for neighborhood attention.
            memory_value: Read-only tensor container from MemoryState.read_for_layer().
            gen_only: When True, skip the understanding pathway (und K/V come from cache).
        """
        # Pre-Attention layernorm
        pack_norm_out = from_und_gen_splits(
            self.input_layernorm(get_und_seq(input)),  # [N_und,hidden_size]
            self.input_layernorm_moe_gen(get_gen_seq(input)),  # [N_gen,hidden_size]
            input,
        )  # [N_und+N_gen,hidden_size]

        # Self Attention + Residual
        kv_to_store: KVToStore | None = None
        if gen_only:
            assert natten_metadata is None
            # gen_only: skip und, compute gen tokens only (und K/V come from cache)
            _gen_norm = get_gen_seq(pack_norm_out)
            gen_pack = from_und_gen_splits(
                _gen_norm.new_empty(0, _gen_norm.shape[-1]),
                _gen_norm,
                pack_norm_out,
            )

            # Build position embeddings whose und length matches gen_pack's
            # und length (always 0).  Required when the outer pack carries
            # a padded causal_seq (``pad_for_cuda_graphs=True``): without
            # this, the und RoPE inside ``PackedAttentionMoT.forward``
            # would broadcast cos/sin of shape ``(MAX_CAUSAL_LEN, head_dim)``
            # onto a length-0 ``q_und`` / ``k_und`` and crash.  When the
            # outer pack is unpadded (eager AR path), the und cos/sin
            # already have length 0 and this slice is a no-op.
            _cos, _sin = packed_position_embeddings
            _empty_cos_und = get_und_seq(_cos)[:0]
            _empty_sin_und = get_und_seq(_sin)[:0]
            gen_position_embeddings = (
                from_und_gen_splits(_empty_cos_und, get_gen_seq(_cos), _cos),
                from_und_gen_splits(_empty_sin_und, get_gen_seq(_sin), _sin),
            )

            pack_attn_out, kv_to_store = self.self_attn(
                gen_pack,
                attention_mask,
                gen_position_embeddings,
                natten_metadata=natten_metadata,
                memory_value=memory_value,
            )
            gen_attn_out = get_gen_seq(pack_attn_out)
            # No residual_und here: the gen_only MLP branch below builds its own
            # length-0 und sequence for ``mlp_out_und_seq``; carrying one through
            # this branch is dead code.
            residual_gen = get_gen_seq(input) + gen_attn_out
        else:
            # STANDARD PATH: Process both und and gen tokens
            pack_attn_out, kv_to_store = self.self_attn(
                pack_norm_out,
                attention_mask,
                packed_position_embeddings,
                natten_metadata=natten_metadata,
                memory_value=memory_value,
            )
            residual_und = get_und_seq(input) + get_und_seq(pack_attn_out)  # [N_und,hidden_size]
            residual_gen = get_gen_seq(input) + get_gen_seq(pack_attn_out)  # [N_gen,hidden_size]

        # Pre-MLP layernorm and processing
        lbl_metadata_dict: dict[str, LBLMetadata] = dict()

        if gen_only:
            # gen_only: skip und, compute gen tokens only
            ln_out_und = residual_gen.new_empty(0, residual_gen.shape[-1])
            ln_out_gen = self.post_attention_layernorm_moe_gen(residual_gen)

            # UNPAD MLP INPUT (gen only)
            gen_len = pack_attn_out["_num_full_tokens"]
            ln_out_gen_unpadded = ln_out_gen[:gen_len]  # [N_gen_unpadded,hidden_size]

            # Run MLP (gen only)
            mlp_out_gen_unpadded, lbl_metadata_gen = _run_mlp(self.mlp_moe_gen, ln_out_gen_unpadded)
            # mlp_out_gen_unpadded: [N_gen_unpadded,hidden_size]

            # PAD MLP OUTPUT (gen only)
            mlp_out_gen = torch.cat([mlp_out_gen_unpadded, ln_out_gen[gen_len:]], dim=0)  # [N_gen,hidden_size]

            # Build metadata dict (no und metadata in optimized path)
            if lbl_metadata_gen is not None:
                lbl_metadata_dict["gen"] = lbl_metadata_gen

            # Final output with residual (gen only)
            mlp_out_und_seq = residual_gen.new_empty(0, residual_gen.shape[-1])
            mlp_out_gen_seq = residual_gen + mlp_out_gen
        else:
            # STANDARD PATH: Process both und and gen tokens
            ln_out_und = self.post_attention_layernorm(residual_und)  # [N_und,hidden_size]
            ln_out_gen = self.post_attention_layernorm_moe_gen(residual_gen)  # [N_gen,hidden_size]

            # UNPAD MLP INPUT ===============
            # NOTE: This is only need for the MoE auxiliary loss computation and to avoid
            #       artificial expert inbalance due to routing padding tokens.
            gen_len = pack_attn_out["_num_full_tokens"]
            und_len = pack_attn_out["_num_causal_tokens"]
            ln_out_und_unpadded = ln_out_und[:und_len]  # [N_und_unpadded,hidden_size]
            ln_out_gen_unpadded = ln_out_gen[:gen_len]  # [N_gen_unpadded,hidden_size]

            mlp_out_und_unpadded, lbl_metadata_und = _run_mlp(self.mlp, ln_out_und_unpadded)
            # mlp_out_und_unpadded: [N_und_unpadded,hidden_size]
            mlp_out_gen_unpadded, lbl_metadata_gen = _run_mlp(self.mlp_moe_gen, ln_out_gen_unpadded)
            # mlp_out_gen_unpadded: [N_gen_unpadded,hidden_size]

            # PAD MLP OUTPUT ===============
            mlp_out_und = torch.cat([mlp_out_und_unpadded, ln_out_und[und_len:]], dim=0)  # [N_und,hidden_size]
            mlp_out_gen = torch.cat([mlp_out_gen_unpadded, ln_out_gen[gen_len:]], dim=0)  # [N_gen,hidden_size]

            if lbl_metadata_und is not None:
                lbl_metadata_dict["und"] = lbl_metadata_und
            if lbl_metadata_gen is not None:
                lbl_metadata_dict["gen"] = lbl_metadata_gen

            mlp_out_und_seq = residual_und + mlp_out_und  # [N_und,hidden_size]
            mlp_out_gen_seq = residual_gen + mlp_out_gen  # [N_gen,hidden_size]

        return from_und_gen_splits(mlp_out_und_seq, mlp_out_gen_seq, input), lbl_metadata_dict, kv_to_store

    def reasoner_forward(
        self,
        hidden_states: torch.Tensor,  # [B,T,hidden_size]
        cos: torch.Tensor,  # [B,T,head_dim]
        sin: torch.Tensor,  # [B,T,head_dim]
        cache: "ReasonerKVCache | None",
        layer_idx: int,
    ) -> torch.Tensor:
        """Run this decoder layer through the reasoner (und) pathway only.

        Skips the ``*_moe_gen`` sub-modules — the dual-pathway training
        :meth:`forward` runs both und and gen towers, but autoregressive
        text decoding only needs und.  Called by
        :func:`_impl_reasoner_forward` (which iterates over
        ``model.layers`` in the AR loop).

        FSDP2 note: under ``data_parallel_shard_degree > 1`` each decoder
        layer is its own FSDP unit (see
        ``parallelize_unified_mot.apply_fsdp``) and its parameters live
        as ``DTensor`` shards until the layer's pre-forward hook
        materializes them.  This method is registered as a
        forward-equivalent via ``register_fsdp_forward_method`` in
        :func:`parallelize_unified_mot.apply_fsdp` so the layer's
        pre-forward unshard / post-forward reshard hooks fire here
        exactly as they would for :meth:`forward`, even though we call
        sub-modules (``input_layernorm`` / ``self_attn`` / ``mlp``)
        directly — without that registration,
        ``layer.input_layernorm.weight`` et al. stay sharded and
        ``self.weight * hidden_states`` raises ``RuntimeError:
        aten.mul.Tensor: got mixed torch.Tensor and DTensor``.  In
        single-rank settings the registration is a no-op and this
        method just runs the und pathway directly.
        """
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        attn_out = self.self_attn.reasoner_forward(h, cos, sin, cache, layer_idx)
        hidden = residual + attn_out  # [B,T,hidden_size]

        residual = hidden
        h = self.post_attention_layernorm(hidden)

        # ``_run_mlp`` transparently handles both dense MLPs (returns Tensor)
        # and ``Qwen3VLMoeTextSparseMoeBlock`` (returns ``(Tensor, LBLMetadata)``).
        # The MoE block expects flat ``[N, hidden_size]`` input, so we flatten and
        # reshape back.  In inference we discard the LBL metadata.
        B, T, H = h.shape
        mlp_out, _ = _run_mlp(self.mlp, h.reshape(B * T, H))  # [B*T,hidden_size]
        mlp_out = mlp_out.view(B, T, H)
        return residual + mlp_out  # [B,T,hidden_size]


class Qwen3VLTextModel(Qwen3VLPreTrainedModel):
    """
    Qwen3VL text model for MoT with dense MLPs.
    This is a wrapper around the _impl_forward defined above,
    specialized for dense models.
    """

    def __init__(
        self,
        config: Qwen3VLTextConfig,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
        use_und_k_norm_for_gen: bool,
    ):
        super().__init__(config)
        _impl_init(
            self,
            config=config,
            layer_types=LayerTypes("qwen3_vl_dense"),
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
            use_und_k_norm_for_gen=use_und_k_norm_for_gen,
        )

    def init_taylorseer(self, cache_dic=None, current=None):
        _impl_init_taylorseer(self, cache_dic=cache_dic, current=current)

    def forward(self, *args, **kwargs):
        return _impl_forward(self, *args, **kwargs)

    def reasoner_forward(self, *args, **kwargs) -> torch.Tensor:
        return _impl_reasoner_forward(self, *args, **kwargs)


class Qwen3VLMoeTextModel(Qwen3VLMoePreTrainedModel):
    """
    Qwen3VL text model for MoT with MoE MLPs.
    This is a wrapper around the _impl_* helpers defined above,
    specialized for MoE models.
    """

    def __init__(
        self,
        config: Qwen3VLMoeTextConfig,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
        gen_noisy_gating: bool = False,
        use_und_k_norm_for_gen: bool,
    ):
        super().__init__(config)
        _impl_init(
            self,
            config=config,
            layer_types=LayerTypes("qwen3_vl_moe"),
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
            gen_noisy_gating=gen_noisy_gating,
            use_und_k_norm_for_gen=use_und_k_norm_for_gen,
        )

    def init_taylorseer(self, cache_dic=None, current=None):
        _impl_init_taylorseer(self, cache_dic=cache_dic, current=current)

    def forward(self, *args, **kwargs):
        return _impl_forward(self, *args, **kwargs)

    def reasoner_forward(self, *args, **kwargs) -> torch.Tensor:
        return _impl_reasoner_forward(self, *args, **kwargs)


class Nemotron3DenseVLTextModel(Nemotron3DenseVLPreTrainedModel):
    """
    Nemotron 3 Dense VL text model adapted for MoT training.
    This is a wrapper around the _impl_* helpers defined above,
    specialized for Nemotron 3 Dense VL models.
    """

    def __init__(
        self,
        config: Nemotron3DenseVLTextConfig,
        *,
        qk_norm_for_text: bool,
        qk_norm_for_diffusion: bool,
        use_und_k_norm_for_gen: bool,
    ):
        super().__init__(config)
        _impl_init(
            self,
            config=config,
            layer_types=LayerTypes("nemotron_dense"),
            qk_norm_for_text=qk_norm_for_text,
            qk_norm_for_diffusion=qk_norm_for_diffusion,
            use_und_k_norm_for_gen=use_und_k_norm_for_gen,
        )

    def init_taylorseer(self, cache_dic=None, current=None) -> None:
        _impl_init_taylorseer(self, cache_dic=cache_dic, current=current)

    def forward(self, *args, **kwargs):
        return _impl_forward(self, *args, **kwargs)

    def reasoner_forward(self, *args, **kwargs) -> torch.Tensor:
        return _impl_reasoner_forward(self, *args, **kwargs)


# -----------------------------------------------------------------------------
# Reasoner-tower autoregressive text generation
# -----------------------------------------------------------------------------
#
# The MoT decoder has two parallel pathways with identical structure but
# disjoint weights:
#   - "Reasoner" / understanding tower: weights WITHOUT the ``_moe_gen``
#     suffix (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, mlp,
#     input_layernorm, post_attention_layernorm, plus the model-level
#     ``embed_tokens`` / ``norm`` / ``lm_head``).
#   - Generation tower: weights WITH the ``_moe_gen`` suffix.
#
# The helpers below run *only* the reasoner tower in standard ``[B, T, H]``
# layout with a per-layer KV cache, enabling an efficient prompt-prefill +
# token-by-token decode loop for next-token text generation.  Sequence
# packing (``SequencePack``) is intentionally not used here because
# AR text generation has no full-attention generation tokens to pack with.
# -----------------------------------------------------------------------------


@dataclass
class ReasonerKVCache:
    """Per-layer KV cache for the reasoner-tower autoregressive loop.

    Tensors are stored in the heads-last BSHD layout that
    ``cosmos_framework.model.attention.attention`` expects::

        keys[layer_idx]:   [B, T, num_kv_heads, head_dim]
        values[layer_idx]: [B, T, num_kv_heads, head_dim]

    where ``T`` grows monotonically as new tokens are appended each
    decode step.  ``None`` entries indicate an empty cache for that layer
    (set by ``empty()``); the first ``update`` populates them.
    """

    keys: list[torch.Tensor | None]
    values: list[torch.Tensor | None]

    @classmethod
    def empty(cls, num_layers: int) -> "ReasonerKVCache":
        return cls(keys=[None] * num_layers, values=[None] * num_layers)

    @property
    def num_layers(self) -> int:
        return len(self.keys)

    @property
    def seq_len(self) -> int:
        if self.keys[0] is None:
            return 0
        return int(self.keys[0].shape[1])

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,  # [B,T_new,num_kv_heads,head_dim]
        v: torch.Tensor,  # [B,T_new,num_kv_heads,head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append new K/V along the seq dim and return the full cached K/V."""
        cached_k = self.keys[layer_idx]
        cached_v = self.values[layer_idx]
        if cached_k is None or cached_v is None:
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
        else:
            self.keys[layer_idx] = torch.cat([cached_k, k], dim=1)  # [B,T_total,num_kv_heads,head_dim]
            self.values[layer_idx] = torch.cat([cached_v, v], dim=1)  # [B,T_total,num_kv_heads,head_dim]
        return self.keys[layer_idx], self.values[layer_idx]  # type: ignore[return-value]

    def reset(self) -> None:
        for i in range(self.num_layers):
            self.keys[i] = None
            self.values[i] = None


def _impl_reasoner_forward(
    self,  # Qwen3VLTextModel | Qwen3VLMoeTextModel | Nemotron3DenseVLTextModel
    input_ids: torch.Tensor | None,  # [B,T]
    cache: ReasonerKVCache | None,
    position_ids: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    visual_pos_masks: torch.Tensor | None = None,
    deepstack_visual_embeds: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    """Shared reasoner-tower forward used by all three MoT text-model variants.

    Returns the final post-norm hidden states ``[B, T, hidden_size]``.

    Each per-layer call goes through ``MoTDecoderLayer.reasoner_forward``,
    which is registered as an FSDP2 forward-equivalent by
    ``parallelize_unified_mot.apply_fsdp`` so per-layer FSDP collectives
    fire correctly under ``data_parallel_shard_degree > 1``.  The
    outer ``self.embed_tokens`` / ``self.norm`` / ``self.rotary_emb``
    parameters live on the top-level FSDP unit and are materialized by
    the corresponding ``register_fsdp_forward_method(top, "generate_reasoner_text")``
    in ``parallelize_vfm_network`` — together those two registrations
    cover every FSDP-wrapped weight touched on the AR path.

    Args:
        input_ids: ``[B, T]`` integer token ids for this step.
        cache: Per-layer KV cache (mutated in place when not ``None``).
        position_ids: ``[B, T]`` absolute positions for these tokens.  When
            ``None``, positions are inferred from the cache so that prompts
            are indexed ``[0..T-1]`` and decode steps continue from the
            cache length.
        inputs_embeds: Pre-embedded inputs ``[B, T, hidden_size]``.
            Mutually exclusive with ``input_ids``.
        visual_pos_masks: ``[B, T]`` bool mask of visual token positions.
            Required (and must be on the same device as ``inputs_embeds``)
            when ``deepstack_visual_embeds`` is provided; ignored otherwise.
        deepstack_visual_embeds: Optional per-layer deepstack visual
            embeddings for the I2V path.  When provided, layer ``i``'s
            output is additively updated at the visual positions.  Every
            element must already match ``inputs_embeds.device`` and
            ``inputs_embeds.dtype``; the canonical producer
            ``prepare_multimodal_reasoner_inputs`` aligns both.
    """
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

    if inputs_embeds is None:
        assert input_ids is not None
        B, T = input_ids.shape
        device = input_ids.device
        h = self.embed_tokens(input_ids)  # [B,T,hidden_size]
    else:
        B, T, _ = inputs_embeds.shape
        device = inputs_embeds.device
        h = inputs_embeds

    if position_ids is None:
        past_len = 0 if cache is None else cache.seq_len
        position_ids = torch.arange(past_len, past_len + T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)

    # The multimodal rotary embedding accepts ``[B, T]`` and returns
    # cos/sin in ``[B, T, head_dim]`` (mrope axes are collapsed inside).
    cos, sin = self.rotary_emb(h, position_ids=position_ids)

    # Contract: when ``deepstack_visual_embeds`` is provided, ``visual_pos_masks``
    # must be non-None and every tensor must already match ``h``'s device + dtype.
    # ``prepare_multimodal_reasoner_inputs`` (the canonical producer) enforces both
    # invariants, so no per-layer ``.to(h.device)`` coerce is needed here.
    if deepstack_visual_embeds is not None and visual_pos_masks is None:
        raise ValueError("visual_pos_masks is required when deepstack_visual_embeds is provided.")

    for layer_idx, decoder_layer in enumerate(self.layers):
        h = decoder_layer.reasoner_forward(h, cos, sin, cache, layer_idx)
        if deepstack_visual_embeds is not None and layer_idx < len(deepstack_visual_embeds):
            h = h.clone()
            h[visual_pos_masks, :] = h[visual_pos_masks, :] + deepstack_visual_embeds[layer_idx]

    # The reasoner tower's final norm is ``norm`` (NOT ``norm_moe_gen``).
    return self.norm(h)  # [B,T,hidden_size]


def _sample_next_token(
    logits: torch.Tensor,  # [B,vocab_size]
    *,
    do_sample: bool,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    seen_mask: torch.Tensor | None = None,  # [B,vocab_size] bool — prompt ∪ output, for repetition_penalty
    output_seen_mask: torch.Tensor | None = None,  # [B,vocab_size] bool — output only,    for presence_penalty
    generator: torch.Generator | None = None,
) -> torch.Tensor:  # [B]
    """Greedy / multinomial sampling with optional top-k, top-p, and presence/repetition penalties.

    Pipeline (order matters):
        1. Repetition penalty (CTRL/HF formula) — multiplicative rescale
           of every logit at a position seen in history (``seen_mask``).
           ``>1.0`` discourages repetition, ``<1.0`` encourages it,
           ``1.0`` is identity.
        2. Presence penalty (OpenAI semantics) — additive shift of every
           logit at a position seen in **output** (``output_seen_mask``).  ``>0``
           discourages, ``<0`` encourages, ``0`` is identity.
        3. ``do_sample=False`` short-circuits to argmax.  The two
           penalties above are applied *before* this branch so they
           legitimately shift the greedy argmax — they're logit
           transformations, not sampling-only tricks.
        4. ``do_sample=True``: temperature → top-k → top-p → multinomial.

    Mask semantics (match vLLM):
      * ``seen_mask``  is seeded with prompt tokens and updated with each
        generated token — penalizes prompt ∪ output (HF convention).
      * ``output_seen_mask`` is updated with each generated token only — penalizes
        output only.
    Both penalties default to identity; the fast path (both off) leaves the
    existing greedy/sampling logic bit-identical.

    ``generator`` is the only RNG-consuming primitive in this module:
    when provided, it is threaded into ``torch.multinomial`` so the
    sampling branch becomes reproducible across calls that share the
    same seeded generator (see :func:`_impl_generate_reasoner_text`'s
    ``seed`` argument).  Passing ``None`` (default) preserves the
    pre-seed behavior of consuming the device's default RNG and is
    bit-identical to the previous call signature.
    """
    if seen_mask is not None and repetition_penalty != 1.0:
        # CTRL/HF formula: divide positive logits, multiply negative.
        penalty_factor = torch.where(
            logits > 0,
            torch.full_like(logits, 1.0 / repetition_penalty),
            torch.full_like(logits, repetition_penalty),
        )
        logits = torch.where(seen_mask, logits * penalty_factor, logits)
    if output_seen_mask is not None and presence_penalty != 0.0:
        # OpenAI semantics: subtract a constant from every seen
        # token's logit, once per token (presence, not frequency).
        logits = torch.where(
            output_seen_mask,
            logits - presence_penalty,
            logits,
        )

    if not do_sample:
        return torch.argmax(logits, dim=-1)

    if temperature != 1.0:
        logits = logits / max(temperature, 1e-6)

    if top_k is not None and top_k > 0:
        k = min(int(top_k), logits.size(-1))
        kth_values = torch.topk(logits, k, dim=-1).values[..., -1:]  # [B,1]
        logits = torch.where(logits < kth_values, torch.full_like(logits, float("-inf")), logits)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        # Keep the smallest set of tokens whose cumulative prob >= top_p.
        # ``cumprobs - probs`` is the cumulative prob *strictly before* the
        # current token, so we drop tokens for which that prefix already
        # exceeded ``top_p``.  This always keeps at least one token.
        mask = (cumprobs - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _all_ranks_finished(finished: torch.Tensor) -> bool:
    """Reduce ``finished.all()`` across every rank in the default process group.

    Single-process / non-distributed runs return the local result.  Under
    multi-rank inference each rank may see different prompts (e.g. test
    fixtures that build prompts from per-rank ``torch.randint`` without
    seed sync, or batched serving where each rank handles a different
    request slice) and therefore reach the EOS-driven early-exit at
    different decode steps.      Letting any one rank ``break`` while the
    others keep running strands the still-decoding ranks on the next
    FSDP all-gather collective inside ``*TextModel.reasoner_forward``
    (the AR loop is sharded by ``parallelize_unified_mot`` whenever
    ``parallel_dims.dp_enabled`` is True), causing an NCCL hang.

    Reducing the per-rank ``finished.all()`` flag with a logical AND —
    implemented as ``ReduceOp.MIN`` on a 0/1 ``uint8`` tensor, which is
    natively supported by NCCL — guarantees every rank breaks at the
    same iteration: only when *every* rank's every sample has emitted
    EOS.  Ranks that finished earlier keep iterating in lockstep with
    the slowest rank; their per-sample pad bookkeeping
    (``torch.where(finished, pad_token_id, next_token)``) already pins
    finished samples to ``pad_token_id``, so the extra steps just
    extend the rank-local KV cache with pad-fill — no semantic change
    to the returned tokens.

    Returns the boolean global flag (True iff every rank reports every
    sample finished).
    """
    local = finished.all()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        flag = local.to(dtype=torch.uint8)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        return bool(flag.item())
    return bool(local)


@torch.no_grad()
def _impl_generate_reasoner_text(
    causal_lm: nn.Module,
    input_ids: torch.Tensor,  # [B,T_prompt]
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
    """Run a reasoner-tower autoregressive decode loop with a per-layer KV cache.

    The loop has two phases:
        1. Prefill — process the prompt in one forward pass. For I2V, this
           first reuses Qwen3-VL visual helpers to scatter image embeddings.
        2. Decode — repeatedly feed the just-sampled token (length 1) back
           through the model, attending to the cached K/V.

    Each sample maintains its own ``finished`` flag.  Once a sample emits an
    ``eos_token_id``, subsequent positions are filled with ``pad_token_id``;
    the loop exits early once every sample has finished.

    **Distributed semantics.**  Under multi-rank execution
    (``torch.distributed`` initialized, the model wrapped by FSDP via
    ``parallelize_unified_mot``), the early-exit decision is reduced
    across ranks via :func:`_all_ranks_finished` so every rank breaks
    out of the decode loop at the *same* iteration.  This guarantees
    the AR loop's FSDP all-gather collectives stay in lockstep even
    when ranks see different prompts (and therefore reach EOS at
    different decode steps).  Ranks that finish earlier keep iterating
    in lockstep with the slowest rank; their per-sample pad bookkeeping
    is unchanged, so already-finished samples on early-finishing ranks
    simply continue emitting ``pad_token_id`` until the slowest rank
    catches up.  The returned ``T_new`` therefore matches the
    *globally slowest* rank's decode length.

    Args:
        causal_lm: A ``*ForCausalLM`` instance providing ``embed_tokens``
            (via ``causal_lm.model``) and ``lm_head``.
        input_ids: ``[B, T_prompt]`` integer token ids of the prompt.
            For image conditioning, the prompt must contain
            ``model.config.image_token_id`` placeholder tokens (one per
            patch *after* spatial merging) where the image features get
            scattered in.
        pixel_values: Optional image inputs in the native Qwen3-VL image
            processor format: a single ``[N_patches, C, H, W]`` float
            tensor (``model.visual.dtype``) holding *all* image patches
            from *all* images in *all* samples in the batch, concatenated
            along the leading patch dimension.  ``N_patches`` equals
            ``sum_i(t_i * h_i * w_i)`` over the rows of
            ``image_grid_thw`` below.  This is the same layout that
            ``Qwen3VLProcessor`` emits — pass it through unchanged.
            Moved to the prompt's device internally.  ``None`` (default)
            means text-only prompt; in that case the multimodal prefill
            path is skipped entirely.  For video conditioning, pass ``pixel_values_videos`` +
            ``video_grid_thw`` instead (mutually exclusive with the image
            pair).
        image_grid_thw: Optional ``[num_images, 3]`` long tensor giving
            ``(t, h, w)`` — the temporal / height / width feature-grid
            size per image as produced by ``Qwen3VLProcessor`` (``t`` is
            typically 1 for still images).  ``num_images`` is the *total*
            image count across the entire batch, not per-sample.  Moved
            to the prompt's device internally.  Must be supplied together
            with ``pixel_values``; passing exactly one of the two raises
            ``ValueError``.  The number of image placeholder tokens in
            ``input_ids`` must equal
            ``sum_i(t_i * h_i * w_i) // model.visual.spatial_merge_size ** 2``;
            :func:`prepare_multimodal_reasoner_inputs` (which wraps
            :func:`get_placeholder_mask`) raises if this invariant is
            violated.  ``causal_lm.visual`` must exist when this is
            provided (a combined / language-only checkpoint without a
            vision tower will raise ``ValueError``).
        max_new_tokens: Maximum number of new tokens to generate per sample.
        eos_token_id: Token id (or list of ids) that terminates a sample.
            ``None`` disables early stopping.
        pad_token_id: Token id used to fill positions of finished samples
            so generated sequences can be returned as a single padded
            tensor.  Defaults to ``eos_token_id`` (or ``0`` if neither is
            provided).
        do_sample, temperature, top_k, top_p: Sampling controls.
        repetition_penalty: CTRL/HF-style multiplicative penalty
            applied to every logit at a vocab position seen in
            history (prompt + everything generated so far for this
            sample).  ``>1.0`` discourages repetition, ``<1.0``
            encourages it, ``1.0`` (default) is identity and skips
            all penalty bookkeeping.
        presence_penalty: OpenAI-style additive penalty applied once
            to every logit at a vocab position seen in history.
            ``>0`` discourages reuse, ``<0`` encourages it, ``0``
            (default) is identity.  Presence (binary), not frequency
            — appearing twice costs the same as appearing once.
            Both penalties are applied *before* the ``do_sample``
            argmax/multinomial branch, so they shift the greedy
            argmax too.  When both are at identity, no history mask
            is allocated and the loop is bit-identical to the
            un-penalized fast path. Repetition penalty uses prompt ∪
            output; presence penalty uses output only (OpenAI / vLLM
            convention).
        seed: Optional integer seed for the sampling RNG.  When provided
            (and ``do_sample=True``), a fresh ``torch.Generator`` is
            allocated on ``input_ids.device`` and seeded once with
            ``manual_seed(seed)``; this generator is threaded into every
            ``torch.multinomial`` call inside :func:`_sample_next_token`,
            making the decoded sequence reproducible across calls that
            share the same seed.  ``None`` (default) consumes the
            device's default RNG, preserving the pre-seed call surface
            and behavior.  Has no effect when ``do_sample=False``
            (greedy / argmax doesn't consume any RNG); the generator
            is still allocated in that case but is never read.  Under
            multi-rank execution, callers that need cross-rank agreement
            on the sampled tokens must pass the *same* ``seed`` on
            every rank — the generator is rank-local, so distinct seeds
            (or ``None``) produce per-rank divergent samples even
            though the prompts and logits agree.
        return_only_new_tokens: If ``True``, return only the generated
            suffix ``[B, T_new]``; otherwise return prompt + generated
            tokens ``[B, T_prompt + T_new]``.

    Returns:
        Token ids ``[B, T_prompt + T_new]`` (or ``[B, T_new]`` when
        ``return_only_new_tokens=True``).  ``T_new <= max_new_tokens``;
        early termination only occurs when every sample emits EOS.
    """
    if input_ids.dim() != 2 or input_ids.shape[1] < 1:
        raise ValueError(f"input_ids must have shape [B, T_prompt>=1], got {tuple(input_ids.shape)}")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")

    eos_ids: list[int] = []
    if isinstance(eos_token_id, int):
        eos_ids = [eos_token_id]
    elif eos_token_id is not None:
        eos_ids = [int(t) for t in eos_token_id]
    if pad_token_id is None:
        pad_token_id = eos_ids[0] if eos_ids else 0

    B = input_ids.shape[0]
    device = input_ids.device

    if max_new_tokens == 0:
        empty = input_ids.new_zeros((B, 0))
        return empty if return_only_new_tokens else input_ids.clone()

    model = causal_lm.model
    cache = ReasonerKVCache.empty(num_layers=len(model.layers))

    if (pixel_values is None) != (image_grid_thw is None):
        raise ValueError("pixel_values and image_grid_thw must be provided together.")
    if (pixel_values_videos is None) != (video_grid_thw is None):
        raise ValueError("pixel_values_videos and video_grid_thw must be provided together.")
    if pixel_values is not None and pixel_values_videos is not None:
        raise ValueError("Reasoner conditions on one medium at a time: pass image OR video, not both.")

    _prefill_start = time.time()

    mrope_position_deltas: torch.Tensor | None = None
    if pixel_values is None and pixel_values_videos is None:
        hidden = model.reasoner_forward(input_ids, cache=cache)  # [B,T_prompt,hidden_size]
    else:
        if not hasattr(causal_lm, "visual"):
            raise ValueError("Combined checkpoint does not include a visual module on the reasoner language model.")
        (
            inputs_embeds,
            visual_pos_masks,
            deepstack_visual_embeds,
            position_ids,
            mrope_position_deltas,
        ) = prepare_multimodal_reasoner_inputs(
            causal_lm,
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        hidden = model.reasoner_forward(
            input_ids=None,
            cache=cache,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )  # [B,T_prompt,hidden_size]
    logits = causal_lm.lm_head(hidden[:, -1, :])  # [B,vocab_size]

    # seen_mask is seeded with prompt tokens (HF convention).
    # output_seen_mask stays empty until output tokens accumulate (OpenAI convention).
    seen_mask: torch.Tensor | None = None
    output_seen_mask: torch.Tensor | None = None
    if repetition_penalty != 1.0:
        seen_mask = torch.zeros(B, logits.size(-1), dtype=torch.bool, device=device)
        seen_mask.scatter_(1, input_ids, True)
    if presence_penalty != 0.0:
        output_seen_mask = torch.zeros(B, logits.size(-1), dtype=torch.bool, device=device)

    # Build a device-local ``torch.Generator`` only when an explicit
    # seed is supplied.  ``torch.multinomial(generator=None)`` falls
    # back to the device's default RNG, matching the pre-seed behavior
    # exactly, so an un-seeded call is bit-identical to the previous
    # signature.  We seed once here (not inside the decode loop) so
    # every multinomial draw consumes the same generator state — the
    # sequence of sampled tokens is then a deterministic function of
    # ``seed``, the logits, and the penalty masks.  Greedy decoding
    # (``do_sample=False``) doesn't touch the generator, so the
    # allocation is essentially free in that case.
    generator: torch.Generator | None = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    next_token = _sample_next_token(
        logits,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        seen_mask=seen_mask,
        output_seen_mask=output_seen_mask,
        generator=generator,
    )  # [B]
    # Fold the just-sampled token into both penalty histories.
    if seen_mask is not None:
        seen_mask.scatter_(1, next_token.unsqueeze(1), True)
    if output_seen_mask is not None:
        output_seen_mask.scatter_(1, next_token.unsqueeze(1), True)

    # Hoist invariants used by every decode step out of the loop body so we
    # don't pay per-iter Python and allocator overhead for what is in fact
    # step-invariant state.  Each of these used to be (re-)materialized
    # every iteration:
    #   * ``pad_tensor``  — replaces ``torch.full_like(next_token, pad_token_id)``
    #     in the per-step pad-fill of finished samples.  Allocator-free reuse.
    #   * ``eos_tensor`` — replaces the Python-level
    #     ``for t in eos_ids: finished = finished | (next_token == t)`` chain
    #     with one vectorized ``any(dim=-1)`` over a small EOS-id dim.  Also
    #     used as a guard to skip ``_all_ranks_finished`` entirely when no
    #     EOS ids are configured (the check can never trigger in that case,
    #     so the per-step ``.item()`` host-device sync is pure waste).
    #   * ``base_mrope_position_ids`` — replaces the per-step
    #     ``torch.full((B,1), cache.seq_len) + deltas.to(long)`` +
    #     ``unsqueeze(0).expand(3,-1,-1)`` chain with one scalar-add against
    #     a pre-expanded ``[3,B,1]`` view.
    pad_tensor = torch.full((B,), pad_token_id, dtype=next_token.dtype, device=device)
    eos_tensor: torch.Tensor | None = None
    if eos_ids:
        eos_tensor = torch.as_tensor(eos_ids, dtype=next_token.dtype, device=device)
    base_mrope_position_ids: torch.Tensor | None = None
    if mrope_position_deltas is not None:
        base_mrope_position_ids = mrope_position_deltas.to(dtype=torch.long).unsqueeze(0).expand(3, -1, -1)

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    if eos_tensor is not None:
        finished |= (next_token.unsqueeze(-1) == eos_tensor).any(dim=-1)

    torch.cuda.synchronize()
    _prefill_time = time.time() - _prefill_start
    log.info(f"[generate_reasoner_text] prefill time: {_prefill_time:.2f} sec")

    _decode_start = time.time()
    # Pre-allocate the full ``[B, max_new_tokens]`` output buffer and write
    # per-step into it via scalar column indexing; this avoids the per-step
    # ``next_token.unsqueeze(1)`` alloc, the Python list append, and the
    # final ``torch.cat`` over the accumulated chunks.  We slice the buffer
    # at the end to honor any early-exit shrinkage.
    output_buf = torch.empty(B, max_new_tokens, dtype=next_token.dtype, device=device)
    output_buf[:, 0] = next_token
    num_generated = 1

    # Decode — feed one token at a time, attending to all cached K/V.
    for _ in range(max_new_tokens - 1):
        # Cross-rank early-exit: only break when every sample on every
        # rank has emitted EOS.  Under FSDP the loop body all-gathers
        # full weights every step, so a per-rank ``break`` would hang
        # the slower ranks on a collective the faster ranks never enter.
        # See ``_all_ranks_finished`` for the full rationale; on
        # single-process / non-distributed runs this reduces to the
        # original ``bool(finished.all())`` check.  When no ``eos_ids``
        # are configured ``finished`` stays all-False forever and the
        # check can never trigger, so we skip the per-step host-device
        # sync from ``_all_ranks_finished``'s ``.item()`` entirely.
        if eos_tensor is not None and _all_ranks_finished(finished):
            break
        step_input = next_token.unsqueeze(1)  # [B,1]
        position_ids = None
        if base_mrope_position_ids is not None:
            # Decode-step mrope position = cache.seq_len + per-sample delta.
            # cache.seq_len already grew by one with each previous decode step,
            # so the Nth decoded token (N>=0) lands at absolute position
            # ``T_prompt + N + delta``, matching HF's ``cache_position[0] + rope_deltas``.
            # Broadcasting a Python int onto the pre-expanded ``[3,B,1]``
            # view produces a fresh contiguous ``[3,B,1]`` long tensor in a
            # single elementwise kernel — strictly cheaper than the prior
            # ``torch.full + add + unsqueeze + expand`` chain.
            position_ids = base_mrope_position_ids + cache.seq_len
        hidden = model.reasoner_forward(step_input, cache=cache, position_ids=position_ids)  # [B,1,hidden_size]
        logits = causal_lm.lm_head(hidden[:, -1, :])  # [B,vocab_size]
        next_token = _sample_next_token(
            logits,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            seen_mask=seen_mask,
            output_seen_mask=output_seen_mask,
            generator=generator,
        )  # [B]
        # Force pad on already-finished samples; finished stays True afterwards.
        # ``pad_tensor`` is hoisted above so we avoid the per-step
        # ``torch.full_like(next_token, pad_token_id)`` allocation.
        next_token = torch.where(finished, pad_tensor, next_token)
        # Record (post-pad) emitted token in both penalty histories. Finished
        # samples write pad_token_id, which is dead state and harmless.
        if seen_mask is not None:
            seen_mask.scatter_(1, next_token.unsqueeze(1), True)
        if output_seen_mask is not None:
            output_seen_mask.scatter_(1, next_token.unsqueeze(1), True)
        if eos_tensor is not None:
            # Vectorized EOS comparison: broadcast ``next_token`` (``[B,1]``)
            # against ``eos_tensor`` (``[E]``) and reduce-any across the
            # small E dim.  Equivalent to the previous Python OR chain
            # over ``eos_ids`` but builds only one transient tensor and a
            # single reduce, and updates ``finished`` in place.
            finished |= (next_token.unsqueeze(-1) == eos_tensor).any(dim=-1)
        output_buf[:, num_generated] = next_token
        num_generated += 1

    new_tokens_tensor = output_buf[:, :num_generated]  # [B,T_new]

    torch.cuda.synchronize()
    _decode_time = time.time() - _decode_start
    log.info(f"[generate_reasoner_text] decode time: {_decode_time:.2f} sec, number of tokens: {num_generated}")
    log.info(f"[generate_reasoner_text] average decode time per token: {_decode_time * 1e3 / num_generated:.2f} ms")

    if return_only_new_tokens:
        return new_tokens_tensor
    return torch.cat([input_ids, new_tokens_tensor], dim=1)  # [B,T_prompt+T_new]


class Qwen3VLTextForCausalLM(Qwen3VLPreTrainedModel):
    """
    Qwen3VL text causal language model for MoT.
    This variant is used for dense-only MLP models.
    """

    # lm_head.weight is tied to model.embed_tokens.weight
    _tied_weights_keys: list[str] = ["lm_head.weight"]

    def __init__(self, config: Qwen3VLMoTConfig):
        # Materialize a fresh HF ``Qwen3VLTextConfig`` from the wrapper.
        # The wrapper folds MoT params + caller-supplied overrides
        # (``tie_word_embeddings``, etc.) into the merge, so the live
        # ``self.config`` carries every field the decoder layers and
        # the HF base class expect.  ``Qwen3VLTextModel`` and the
        # ``super().__init__`` PreTrainedModel base both consume the
        # text config — never the wrapper — keeping the HF
        # PretrainedConfig contract intact.

        super().__init__(config.full_config)

        text_config = config.text_config
        self.model = Qwen3VLTextModel(
            text_config,
            qk_norm_for_text=config.qk_norm_for_text,
            qk_norm_for_diffusion=config.qk_norm_for_diffusion,
            use_und_k_norm_for_gen=getattr(config, "use_und_k_norm_for_gen", False),
        )
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        # The wrapper's ``vision_config`` property gates on
        # ``include_visual`` and materializes the HF vision config from
        # the JSON's vision sub-section, so this is just a gate + plug.
        vision_config = config.vision_config
        if vision_config is not None:
            self.visual = Qwen3VLVisionModel._from_config(vision_config)

        # Initialize weights and apply final processing
        self.post_init()

    def init_moe(self) -> None:
        """Copy understanding-pathway weights into the generation-pathway parameters.

        Iterates over every parameter whose name contains ``moe_gen`` (i.e.
        every gen-tower weight) and copies its und-tower counterpart in.
        ``q_norm`` / ``k_norm`` get a special-case skip: when
        ``qk_norm_for_text=False`` the und tower's q_norm/k_norm are
        ``nn.Identity()`` (no parameters), but the gen tower may still have
        real ``q_norm_moe_gen`` / ``k_norm_moe_gen`` RMSNorm modules — those
        keep their default ``ones`` init and we just skip the copy rather
        than raising.
        """
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if "moe_gen" not in name:
                continue
            original_name = name.replace("_moe_gen", "").replace("_checkpoint_wrapped_module.", "")
            if original_name in state_dict:
                param.data.copy_(state_dict[original_name].data)
            else:
                raise ValueError(f"Could not find {original_name} in state_dict for initialization of {name}")

    def get_input_embeddings(self) -> nn.Embedding:
        # Explicit override matching upstream HF `Qwen3VLForCausalLM`.  We
        # could rely on `PreTrainedModel.get_input_embeddings`'s recursion
        # via `base_model_prefix="model"`, but defining the method here is
        # the canonical HF idiom and removes a hidden dependency on
        # `base_model_prefix` being correctly set.
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        pack: SequencePack,
        attention_mask,
        position_ids: torch.Tensor,
        natten_metadata_list: list | None = None,
        memory: MemoryState | None = None,
    ) -> tuple[SequencePack, dict[str, LBLMetadata]]:
        """Training forward pass — delegates to the dense text model."""
        outputs = self.model(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
        )
        return outputs

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

        Handles both text-only and image-conditioned (I2V) prompts through a
        single entry point.  Pass ``pixel_values`` + ``image_grid_thw`` (and
        optionally ``attention_mask``) to drive image-conditioned prefill via
        the Qwen3-VL visual encoder; omit them for text-only prefill.  The
        two arguments are mutually required: passing exactly one raises
        ``ValueError`` inside :func:`_impl_generate_reasoner_text`.
        Video conditioning is also supported via ``pixel_values_videos`` +
        ``video_grid_thw``; the image and video pairs are mutually exclusive.

        Uses the und-pathway weights (those WITHOUT the ``_moe_gen`` suffix)
        plus the model-level ``embed_tokens`` / ``norm`` / ``lm_head``, and —
        for the I2V path — the visual encoder under ``self.visual``.  The
        generation pathway and all VFM-level multimodal embedders / heads
        (``vae2llm``, ``llm2vae``, etc.) are bypassed.  See
        :func:`_impl_generate_reasoner_text` for full argument docs.
        """
        return _impl_generate_reasoner_text(
            self,
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


class Qwen3VLMoeTextForCausalLM(Qwen3VLMoePreTrainedModel):
    """
    Qwen3VL text causal language model for MoT with MoE on the generation pathway.
    This variant is used for MoE MLP models.
    """

    # lm_head.weight is tied to model.embed_tokens.weight
    _tied_weights_keys: list[str] = ["lm_head.weight"]

    def __init__(self, config: Qwen3VLMoeMoTConfig):
        super().__init__(config.full_config)

        text_config = config.text_config
        self.model = Qwen3VLMoeTextModel(
            text_config,
            qk_norm_for_text=config.qk_norm_for_text,
            qk_norm_for_diffusion=config.qk_norm_for_diffusion,
            gen_noisy_gating=config.gen_noisy_gating,
            use_und_k_norm_for_gen=getattr(config, "use_und_k_norm_for_gen", False),
        )
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        # The wrapper's ``vision_config`` property gates on
        # ``include_visual`` and materializes the HF vision config from
        # the JSON's vision sub-section, so this is just a gate + plug.
        vision_config = config.vision_config
        if vision_config is not None:
            self.visual = Qwen3VLMoeVisionModel._from_config(vision_config)

        # Initialize weights and apply final processing
        self.post_init()

    def init_moe(self) -> None:
        """Copy understanding-pathway weights into the generation-pathway parameters.

        See :meth:`Qwen3VLTextForCausalLM.init_moe` for the q_norm/k_norm
        Identity-tower handling shared with the dense variant.
        """
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if "moe_gen" not in name:
                continue
            original_name = name.replace("_moe_gen", "").replace("_checkpoint_wrapped_module.", "")
            if original_name in state_dict:
                param.data.copy_(state_dict[original_name].data)
            elif "gate_noise" in original_name:
                # Noisy-gating projection is gen-tower only (the und tower has no
                # gate_noise counterpart), so keep its zero-init rather than copy.
                pass
            else:
                raise ValueError(f"Could not find {original_name} in state_dict for initialization of {name}")

    def get_input_embeddings(self) -> nn.Embedding:
        # See note on `Qwen3VLTextForCausalLM.get_input_embeddings`.
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        pack: SequencePack,
        attention_mask,
        position_ids: torch.Tensor,
        natten_metadata_list: list | None = None,
        memory: MemoryState | None = None,
    ) -> tuple[SequencePack, dict[str, LBLMetadata]]:
        """Training forward pass — delegates to the MoE text model."""

        outputs = self.model(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
        )

        return outputs

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

        Handles both text-only and image-conditioned (I2V) prompts through a
        single entry point.  Pass ``pixel_values`` + ``image_grid_thw`` (and
        optionally ``attention_mask``) to drive image-conditioned prefill via
        the Qwen3-VL visual encoder; omit them for text-only prefill.  The
        two arguments are mutually required: passing exactly one raises
        ``ValueError`` inside :func:`_impl_generate_reasoner_text`.
        Video conditioning is also supported via ``pixel_values_videos`` +
        ``video_grid_thw``; the image and video pairs are mutually exclusive.

        Uses the und-pathway weights (those WITHOUT the ``_moe_gen`` suffix)
        plus the model-level ``embed_tokens`` / ``norm`` / ``lm_head``, and —
        for the I2V path — the visual encoder under ``self.visual``.  The
        MoE generation pathway is bypassed entirely; the reasoner MLP can
        itself be a dense MLP or an MoE block — both are handled by
        :func:`_run_mlp` inside the loop.  See
        :func:`_impl_generate_reasoner_text` for full argument docs.
        """
        return _impl_generate_reasoner_text(
            self,
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


class Nemotron3DenseVLTextForCausalLM(Nemotron3DenseVLPreTrainedModel):
    """Causal LM head on top of the Nemotron 3 Dense VL MoT text model."""

    _tied_weights_keys: list[str] = []

    def __init__(self, config: Nemotron3DenseVLMoTConfig) -> None:
        # Materialize a fresh HF ``Nemotron3DenseVLTextConfig`` from the
        # wrapper.  The wrapper folds MoT params + caller-supplied
        # overrides into the merge, so the live ``self.config`` carries
        # every field the decoder layers and the HF ``PreTrainedModel``
        # base class expect.  ``Nemotron3DenseVLTextModel`` and the
        # ``super().__init__`` ``PreTrainedModel`` base both consume the
        # text config — never the wrapper — keeping the HF
        # ``PretrainedConfig`` contract intact.
        super().__init__(config.full_config)

        text_config = config.text_config
        self.model = Nemotron3DenseVLTextModel(
            text_config,
            qk_norm_for_text=config.qk_norm_for_text,
            qk_norm_for_diffusion=config.qk_norm_for_diffusion,
            use_und_k_norm_for_gen=getattr(config, "use_und_k_norm_for_gen", False),
        )
        self.vocab_size = text_config.vocab_size
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)

        assert config.vision_config is None, "Nemotron 3 Dense VL has no vision config"

        self.post_init()

    def init_moe(self) -> None:
        """Copy understanding-pathway weights into the generation-pathway parameters."""
        state_dict = self.state_dict()
        for name, param in self.named_parameters():
            if "moe_gen" not in name:
                continue
            original_name = name.replace("_moe_gen", "").replace("_checkpoint_wrapped_module.", "")
            if original_name in state_dict:
                param.data.copy_(state_dict[original_name].data)
            elif any(norm_key in original_name for norm_key in ("q_norm", "k_norm")):
                # qk_norm_for_text=False → q_norm/k_norm are nn.Identity() with no parameters;
                # the moe_gen counterpart (q_norm_moe_gen) is a real RMSNorm, so skip init here.
                pass
            else:
                raise ValueError(f"Could not find {original_name} in state_dict for initialization of {name}")

    def get_input_embeddings(self) -> nn.Embedding:
        # See note on `Qwen3VLTextForCausalLM.get_input_embeddings`.
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        pack: SequencePack,
        attention_mask,
        position_ids: torch.Tensor,
        natten_metadata_list: list | None = None,
        memory: MemoryState | None = None,
    ) -> tuple[SequencePack, dict[str, LBLMetadata]]:
        return self.model(
            pack=pack,
            attention_mask=attention_mask,
            position_ids=position_ids,
            natten_metadata_list=natten_metadata_list,
            memory=memory,
        )

    def generate_reasoner_text(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
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
        raise NotImplementedError("This method is not implemented for Nemotron 3 Dense VL.")
