# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal HFModel for the vfm/ unified VLM training path.

Responsibilities:
  - ``__init__``: meta-init the underlying HF model via the appropriate
    AutoClass (``AutoModelForImageTextToText`` / ``AutoModel`` /
    ``AutoModelForCausalLM`` — see ``HFModel`` for selection rules);
    no weights are loaded.
  - ``apply_gradient_checkpointing``: wraps HF's standard
    ``gradient_checkpointing_enable`` API.
  - ``tie_embeddings``: re-establishes the input/output embedding tie after
    FSDP wrapping + meta-materialization.
  - ``load_weights``: dispatches to ``load_vlm_model`` (VLM) or
    ``load_language_model`` (LLM) from ``safetensors_loader.py`` based on
    ``vision_config``; returns the set of checkpoint keys that were loaded.
  - ``forward``: pass-through returning logits.

FSDP wrapping lives in ``vfm/models/parallelize_vlm.py::parallelize()``,
NOT here.
"""

import os
from collections import OrderedDict
from types import MethodType
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from accelerate import init_on_device
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

import cosmos_framework.model.generator.reasoner.cosmos3_edge  # noqa: F401  registers cosmos3_edge with transformers Auto classes
from cosmos_framework.model.generator.utils.safetensors_loader import load_language_model, load_vlm_model
from cosmos_framework.utils import log
from cosmos_framework.utils.generator.parallelism import ParallelDims

if TYPE_CHECKING:
    from cosmos_framework.configs.base.defaults.reasoner import SoundUnderstandingConfig


class _ValidationVideoFeatureCache:
    """Rank-local GPU LRU for deterministic, repeated validation videos."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.entries: OrderedDict[tuple, tuple] = OrderedDict()
        self.calls = 0
        self.hits = 0
        self.misses = 0
        self.sync_dummy_encodes = 0
        self.global_all_hit_calls = 0
        self.bypassed_calls = 0
        self.hit_attested = False

    @staticmethod
    def _distributed_flag(value: bool, device: torch.device, op: torch.distributed.ReduceOp) -> bool:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return value
        flag = torch.tensor([int(value)], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(flag, op=op)
        return bool(flag.item())

    def get_or_encode(
        self,
        cache_keys: list[str] | tuple[str, ...] | None,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        encode_fn: Callable,
        spatial_merge_size: int,
    ):
        local_cacheable = (
            grid_thw is not None
            and grid_thw.ndim == 2
            and isinstance(cache_keys, (list, tuple))
            and len(cache_keys) == int(grid_thw.shape[0])
        )
        grids: list[tuple[int, ...]] = []
        raw_sizes: list[int] = []
        if local_cacheable:
            grids = [tuple(int(value) for value in row.detach().cpu().tolist()) for row in grid_thw]
            raw_sizes = [grid[0] * grid[1] * grid[2] for grid in grids]
            local_cacheable = sum(raw_sizes) == int(pixel_values.shape[0])

        globally_cacheable = self._distributed_flag(
            local_cacheable, pixel_values.device, torch.distributed.ReduceOp.MIN
        )
        if not globally_cacheable:
            self.bypassed_calls += 1
            return encode_fn(pixel_values, grid_thw)

        merged_sizes = [size // (spatial_merge_size**2) for size in raw_sizes]
        resolved_keys = [(str(key), grid) for key, grid in zip(cache_keys, grids)]
        pixel_chunks = torch.split(pixel_values, raw_sizes, dim=0)
        available: dict[tuple, tuple] = {}
        missing: OrderedDict[tuple, int] = OrderedDict()
        call_hits = 0
        for index, key in enumerate(resolved_keys):
            entry = self.entries.get(key)
            if entry is None:
                missing.setdefault(key, index)
            else:
                self.entries.move_to_end(key)
                available[key] = entry
                call_hits += 1

        any_rank_missing = self._distributed_flag(
            bool(missing), pixel_values.device, torch.distributed.ReduceOp.MAX
        )
        if any_rank_missing and missing:
            miss_indices = list(missing.values())
            miss_pixels = torch.cat([pixel_chunks[index] for index in miss_indices], dim=0)
            miss_grids = torch.stack([grid_thw[index] for index in miss_indices], dim=0)
            fresh_main, fresh_deepstack = encode_fn(miss_pixels, miss_grids)
            fresh_main_splits = torch.split(
                fresh_main,
                [merged_sizes[index] for index in miss_indices],
                dim=0,
            )
            deepstack_splits = [
                torch.split(layer, [merged_sizes[index] for index in miss_indices], dim=0)
                for layer in fresh_deepstack
            ]
            for fresh_index, key in enumerate(missing):
                entry = (
                    fresh_main_splits[fresh_index].detach().clone(),
                    tuple(parts[fresh_index].detach().clone() for parts in deepstack_splits),
                )
                available[key] = entry
                self.entries[key] = entry
                self.entries.move_to_end(key)
                while len(self.entries) > self.capacity:
                    self.entries.popitem(last=False)
        elif any_rank_missing:
            encode_fn(pixel_chunks[0], grid_thw[:1])
            self.sync_dummy_encodes += 1
        else:
            self.global_all_hit_calls += 1

        ordered = [available[key] for key in resolved_keys]
        main = torch.cat([entry[0] for entry in ordered], dim=0)
        deepstack = [
            torch.cat([entry[1][layer] for entry in ordered], dim=0)
            for layer in range(len(ordered[0][1]) if ordered else 0)
        ]
        self.calls += 1
        self.hits += call_hits
        self.misses += len(missing)
        if call_hits and not self.hit_attested:
            print(
                "TAO_FRAMEWORK_VALIDATION_FEATURE_CACHE_HIT_ATTESTATION "
                f"rank={os.environ.get('RANK', '0')} capacity={self.capacity} hits={call_hits}",
                flush=True,
            )
            self.hit_attested = True
        return main, deepstack

    def clear(self) -> dict[str, int]:
        stats = {
            "calls": self.calls,
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self.entries),
            "sync_dummy_encodes": self.sync_dummy_encodes,
            "global_all_hit_calls": self.global_all_hit_calls,
            "bypassed_calls": self.bypassed_calls,
        }
        self.entries.clear()
        self.calls = self.hits = self.misses = 0
        self.sync_dummy_encodes = self.global_all_hit_calls = self.bypassed_calls = 0
        return stats


def _tensor_names_to_skip_for(model_type: str) -> list[str]:
    """Per-model-type tensor-name regex skip list for load_vlm_model.

    Mirrors the upstream HF-model ``tensor_names_to_skip`` property from the
    legacy VLM policy registry.  Patterns match the **resolved model key**
    (post-name_converter).  These patterns are concatenated with any
    caller-supplied ``extra_skip_patterns`` and forwarded as the unified
    ``skip_patterns`` kwarg of ``load_vlm_model``, where they drive both
    Phase-5 (skip copy of matched model keys) and Phase-6 (tolerate
    matched model keys absent from the checkpoint).

    Registered VLMs (see
    cosmos_framework/configs/base/reasoner/defaults/vlm_policy.py):
    - Qwen3-VL dense (2B/4B/8B/32B): no skips needed.
    - NemotronH_Nano_VL_V2 (nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16):
      RADIO backbone buffers — initialized by the module, not from ckpt.
    """
    table: dict[str, list[str]] = {
        "NemotronH_Nano_VL_V2": [
            r"vision_model\.radio_model\.summary_idxs",
            r"vision_model\.radio_model\.input_conditioner\.norm_mean",
            r"vision_model\.radio_model\.input_conditioner\.norm_std",
        ],
    }
    return table.get(model_type, [])


class HFModel(nn.Module):
    """Minimal HF model wrapper for the vfm/ unified VLM training path.

    Loads any HF causal LM or VL model on the meta device (no GPU memory)
    via the appropriate AutoClass — see selection rules below. Weights are NOT
    loaded in ``__init__``. Call :meth:`load_weights` after FSDP wrapping +
    explicit meta-tensor materialization so each rank only fills its own shard.

    AutoClass selection (by vision_config presence + ``auto_map``):
    - VLM with standard transformers registration (e.g. Qwen3-VL)
      → ``AutoModelForImageTextToText``.  Returns the full conditional-generation
      class (e.g. ``Qwen3VLForConditionalGeneration``), which exposes ``.logits``
      on forward output.  ``AutoModelForCausalLM`` raises ``ValueError`` for VLM
      configs (``Qwen3VLConfig`` is not registered for that auto class), so it
      cannot be used here.
    - VLM with custom ``auto_map`` (e.g. NemotronVL): the registered entry maps
      the full causal-LM class through ``AutoModel`` rather than
      ``AutoModelForImageTextToText`` — use ``AutoModel`` for this case only.
    - LLM (no ``vision_config``) → ``AutoModelForCausalLM``.  Standard causal LM
      with ``.logits``.

    Do NOT use ``AutoModel`` for the standard VLM path — it returns the backbone
    only (e.g. ``Qwen3VLModel``), which does NOT have ``.logits``.

    FSDP / TP wrapping is applied externally by ``parallelize()`` in
    ``vfm/models/parallelize_vlm.py``.
    """

    def __init__(
        self,
        model_name_or_path: str,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "cosmos",
        trust_remote_code: bool = True,
        sound_und: bool = False,
        sound_und_config: "SoundUnderstandingConfig | None" = None,
        configured_model_name_or_path: str | None = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.configured_model_name_or_path = configured_model_name_or_path or model_name_or_path
        self.sound_und = sound_und
        self.sound_und_config = sound_und_config
        self.sound_und_token_id: int | None = None
        if not isinstance(sound_und, bool):
            raise TypeError(f"sound_und must be a bool, got {type(sound_und).__name__}")
        if sound_und and sound_und_config is None:
            raise ValueError("sound_und_config is required when sound_und=True")
        hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.hf_config = hf_config

        # Register cosmos before from_config validates it. Gated so non-cosmos
        # paths don't import cosmos_framework.model.attention.
        if attn_implementation == "cosmos":
            from transformers import AttentionInterface

            from cosmos_framework.utils.generator.hf_attention_cosmos import hf_attention_cosmos

            AttentionInterface.register("cosmos", hf_attention_cosmos)

        # AutoClass selection by model type:
        # - Standard VLM (Qwen3-VL, etc.): AutoModelForImageTextToText returns the full causal
        #   LM with .logits (Qwen3VLForConditionalGeneration, etc.).
        # - Custom VLM with auto_map (e.g. NemotronVL): AutoModelForImageTextToText is not
        #   registered; use AutoModel instead which maps to the full causal LM via auto_map.
        # - LLM (no vision_config): AutoModelForCausalLM → standard causal LM with .logits.
        is_vlm = getattr(hf_config, "vision_config", None) is not None
        auto_map = getattr(hf_config, "auto_map", None) or {}
        if is_vlm:
            if "AutoModelForImageTextToText" in auto_map or not auto_map:
                # Standard VLM or no auto_map (rely on registered transformers type)
                model_cls = AutoModelForImageTextToText
            else:
                # Custom VLM: use AutoModel which maps to the full causal-LM class via auto_map
                model_cls = AutoModel
        else:
            model_cls = AutoModelForCausalLM

        # Meta init: allocates no GPU memory. FSDP2's ``fully_shard`` does NOT
        # auto-materialize meta tensors; the caller (see ``vlm_model._init_vlm``)
        # must explicitly materialize via ``_apply(empty_like, ...)`` between
        # ``parallelize()`` and ``load_weights()``.
        with init_on_device("meta", include_buffers=False):
            self.model = model_cls.from_config(
                hf_config,
                attn_implementation=attn_implementation,
                torch_dtype=dtype,
                trust_remote_code=trust_remote_code,
            )

        if sound_und:
            from cosmos_framework.data.generator.processors.parakeet_audio_processor import (
                add_reasoner_audio_special_tokens,
            )
            from cosmos_framework.model.generator.reasoner.parakeet.configuration_parakeet import ParakeetAudioConfig
            from cosmos_framework.model.generator.reasoner.parakeet.parakeet import ParakeetAudioModel
            from cosmos_framework.model.generator.reasoner.parakeet.utils import patch_reasoner_audio_forward

            input_embeddings = self.model.get_input_embeddings()
            if input_embeddings is None or not hasattr(input_embeddings, "weight"):
                raise ValueError("Audio understanding requires a Reasoner input embedding table")
            embedding_rows, reasoner_hidden_size = input_embeddings.weight.shape
            if embedding_rows <= 0 or reasoner_hidden_size <= 0:
                raise ValueError(
                    "Reasoner input embeddings must have positive vocabulary and hidden dimensions, got "
                    f"{tuple(input_embeddings.weight.shape)}"
                )

            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
            special_tokens = add_reasoner_audio_special_tokens(
                tokenizer,
                model_name_or_path=self.configured_model_name_or_path,
                audio_start_token=sound_und_config.audio_start_token,
                audio_pad_token=sound_und_config.audio_pad_token,
                audio_end_token=sound_und_config.audio_end_token,
            )
            invalid_ids = [token_id for token_id in special_tokens.token_ids if not 0 <= token_id < embedding_rows]
            if invalid_ids:
                raise ValueError(
                    f"Audio token IDs {invalid_ids} fall outside the Reasoner embedding table with "
                    f"{embedding_rows} rows; embedding resize is intentionally unsupported"
                )

            audio_config = ParakeetAudioConfig(
                projection_hidden_size=sound_und_config.projection_hidden_size,
                out_hidden_size=reasoner_hidden_size,
            )
            with init_on_device("meta", include_buffers=False):
                self.model.sound_und_model = ParakeetAudioModel(audio_config)
            self.model.sound_und_model.encoder.requires_grad_(False)
            self.model.sound_und_model.encoder.eval()
            if sound_und_config.freeze_projector:
                self.model.sound_und_model.projector.requires_grad_(False)
                self.model.sound_und_model.projector.eval()

            self.sound_und_token_id = special_tokens.token_ids[1]
            patch_reasoner_audio_forward(
                self.model,
                audio_token_id=self.sound_und_token_id,
                model_type=hf_config.model_type,
                trainable_token_ids=(
                    (special_tokens.token_ids[0], special_tokens.token_ids[2])
                    if sound_und_config.train_boundary_token_embeddings_only
                    else None
                ),
            )
        log.info(f"HFModel: {hf_config.model_type} ({'VLM' if is_vlm else 'LLM'}), dtype={dtype}")

        # Normalize floating-point *parameter* dtypes to ``dtype``. HF's
        # ``from_config`` installs ``torch.set_default_dtype(dtype)`` around the
        # init, but some HF submodules (and vendored remote-code classes) read
        # ``config.torch_dtype`` directly or build tensors with an explicit
        # ``dtype=`` kwarg, so their params can end up in the checkpoint's dtype
        # (typically bf16) while the rest of the model is in ``dtype``. FSDP2's
        # ``_init_mp_dtypes`` then asserts "uniform original parameter dtype …
        # {bf16, fp32}". Normalize on meta (no GPU memory) so all FSDP units see
        # a single original dtype. Buffers are left alone — ``inv_freq`` etc.
        # must stay fp32 (enforced by e.g. qwen3_vl.py's inv_freq assertion).
        n_cast = 0
        with torch.no_grad():
            for p in self.model.parameters(recurse=True):
                if p.is_floating_point() and p.dtype != dtype:
                    p.data = p.data.to(dtype)
                    n_cast += 1
        if n_cast:
            log.info(f"HFModel: normalized {n_cast} param(s) to {dtype} post-from_config")

        self._tao_validation_video_cache_keys = None
        self._tao_validation_video_cache_active = False
        self._tao_validation_video_feature_cache = None
        self._configure_validation_video_feature_cache()

        # Patch Qwen3-VL forward for text-only batches (no pixel_values / image_grid_thw).
        # Required to avoid errors when a batch contains only text: every FSDP rank must
        # call visual() each step for all-gather sync; the patch runs a lightweight dummy
        # image and slices the output to [0:0] so it contributes no features.
        # Must happen BEFORE parallelize() so FSDP captures the patched forward.
        if hf_config.model_type == "qwen3_vl" and hasattr(self.model, "model"):
            from cosmos_framework.utils.generator.monkey_patch import patch_qwen3_vl_forward

            patch_qwen3_vl_forward(self.model.model)
            log.info("HFModel: applied patch_qwen3_vl_forward for text-only batch support")

    def _configure_validation_video_feature_cache(self) -> None:
        capacity = int(os.environ.get("TAO_FRAMEWORK_VALIDATION_VIDEO_FEATURE_CACHE_SIZE", "0"))
        if capacity < 0:
            raise ValueError("TAO_FRAMEWORK_VALIDATION_VIDEO_FEATURE_CACHE_SIZE must be non-negative")
        if capacity == 0:
            return
        if self.hf_config.model_type != "qwen3_vl":
            raise RuntimeError("Framework validation feature cache currently supports only qwen3_vl")
        target = getattr(self.model, "model", None)
        visual = getattr(target, "visual", None)
        original = getattr(visual, "forward", None)
        spatial_merge_size = getattr(visual, "spatial_merge_size", None)
        if target is None or not callable(original) or not spatial_merge_size:
            raise RuntimeError("Framework validation feature cache could not resolve Qwen3-VL vision encoder")
        cache = _ValidationVideoFeatureCache(capacity)

        def cached_visual_forward(visual_self, pixel_values, grid_thw=None):
            del visual_self
            if not self._tao_validation_video_cache_active:
                return original(pixel_values, grid_thw)
            return cache.get_or_encode(
                self._tao_validation_video_cache_keys,
                pixel_values,
                grid_thw,
                original,
                int(spatial_merge_size),
            )

        # c312482 applies ``patch_qwen3_vl_forward``, whose active video path
        # invokes ``self.visual(...)`` directly and bypasses both
        # ``get_video_features`` and ``get_image_features``.  Hook the actual
        # visual boundary so repeated validation media can reuse deterministic
        # features.  FSDP's per-block collectives remain aligned by
        # ``get_or_encode``: if any rank misses, every rank enters the native
        # visual stack once; only a global all-hit batch skips it collectively.
        visual.forward = MethodType(cached_visual_forward, visual)
        self._tao_validation_video_feature_cache = cache
        print(
            "TAO_FRAMEWORK_VALIDATION_FEATURE_CACHE_ENABLED_ATTESTATION "
            f"rank={os.environ.get('RANK', '0')} capacity={capacity} boundary=visual_forward",
            flush=True,
        )

    def clear_validation_video_feature_cache(self) -> dict[str, int] | None:
        if self._tao_validation_video_feature_cache is None:
            return None
        return self._tao_validation_video_feature_cache.clear()

    def train(self, mode: bool = True) -> "HFModel":
        """Keep immutable audio modules in eval mode while training the Reasoner."""
        if mode and self._tao_validation_video_feature_cache is not None:
            if self._tao_validation_video_feature_cache.entries:
                self.clear_validation_video_feature_cache()
        super().train(mode)
        if self.sound_und:
            self.model.sound_und_model.encoder.eval()
            if self.sound_und_config.freeze_projector:
                self.model.sound_und_model.projector.eval()
        return self

    @property
    def net(self) -> nn.Module:
        """Alias for ``self.model``. Matches the ``.net`` attribute that
        ``OmniMoTModel`` exposes, so ``vfm/utils/optimizer.py`` can iterate
        ``model.net.named_parameters()`` uniformly across model families."""
        return self.model

    def apply_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing via HF's standard API."""
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log.info("HFModel: gradient checkpointing enabled")

    def tie_embeddings(self) -> None:
        """Tie output embedding weight to input embedding, matching post_to_empty_hook behavior.

        Must be called AFTER ``parallelize()`` and AFTER the explicit
        meta-tensor materialization step (FSDP2 does not auto-materialize —
        see ``vlm_model._init_vlm`` step e), and BEFORE ``load_weights()`` so
        the tied pointer survives weight loading.

        Two strategies, matching the HF API split:
        1. ``get_output_embeddings()`` path — standard for most HF models.
        2. ``_tied_weights_keys`` fallback — some VLMs (notably
           ``Qwen3VLForConditionalGeneration``) define ``lm_head`` and
           ``_tied_weights_keys = ["lm_head.weight"]`` but do NOT override
           ``get_output_embeddings``.  For those, walk the dotted key to the
           owning module and assign its ``.weight`` directly.  See spec §8.3.

        Reference: the legacy VLM HF-model tie_embeddings implementation.
        """
        if not getattr(self.hf_config, "tie_word_embeddings", False):
            return
        input_embeddings = self.model.get_input_embeddings()
        if input_embeddings is None:
            return
        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings.weight = input_embeddings.weight
            log.info("HFModel: tied input/output embeddings via get_output_embeddings")
            return
        # Fallback: HF models that use _tied_weights_keys instead of
        # overriding get_output_embeddings (e.g. Qwen3VLForConditionalGeneration
        # defines _tied_weights_keys = ["lm_head.weight"] but returns None
        # from the default get_output_embeddings).  Walk the dotted key to
        # the owning module and assign the Parameter directly.
        tied_keys = getattr(self.model, "_tied_weights_keys", None) or ()
        if not tied_keys:
            return
        for key in tied_keys:
            parts = key.split(".")
            *mod_path, attr = parts
            target = self.model
            for name in mod_path:
                target = getattr(target, name, None)
                if target is None:
                    log.warning(
                        f"HFModel.tie_embeddings: could not resolve path {key!r} on "
                        f"{type(self.model).__name__}; skipping tie (weights will "
                        f"remain untied for this key)."
                    )
                    break
            else:
                setattr(target, attr, input_embeddings.weight)
                log.info(f"HFModel: tied {key} via _tied_weights_keys fallback")

    def load_weights(
        self,
        checkpoint_path: str,
        credential_path: str | None = None,
        parallel_dims: ParallelDims | None = None,
        extra_skip_patterns: list[str] | None = None,
    ) -> set[str]:
        r"""Load weights from a HF model directory (safetensors format).

        Dispatches on model type:
        - VLM (vision_config present): ``load_vlm_model`` (universal
          suffix-lookup loader inherited from the legacy VLM path; MoE VLMs
          explicitly blocked — see spec §2.2).
        - LLM (no vision_config): ``load_language_model`` — handles VFM-specific
          per-family key remapping for Qwen3 / Nemotron (unchanged from today).

        Must be called AFTER ``parallelize()`` so parameters are DTensors with
        CUDA local views.  For tied-embedding models, ``tie_embeddings()`` must
        be called between ``parallelize()`` and this function.

        Args:
            checkpoint_path: Path to a directory containing .safetensors files.
                Local paths and S3 URIs are tried first; if no safetensors are
                found, explicit ``hf://org/model`` Hub URIs and bare
                ``org/model`` repo IDs fall back to Hugging Face.
            credential_path: S3 credential file, or None for local/HF.
            parallel_dims: ``ParallelDims`` instance (from
                ``cosmos_framework.utils.generator.parallelism``).  The loader uses
                it via :func:`~cosmos_framework.model.generator.utils.safetensors_loader._get_dp_shard_mesh`
                to obtain the 1-D ``dp_shard`` sub-mesh (or None when
                ``dp_shard <= 1``) for striping checkpoint reads across
                FSDP shard ranks.  When non-None, the caller MUST have
                called ``parallel_dims.build_meshes()`` first — neither
                this method nor ``load_vlm_model`` re-checks this.  Pass
                ``parallel_dims=None`` for the single-rank fallback used
                by single-process / non-distributed runs.
            extra_skip_patterns: Optional list of regex patterns appended to
                the model-type fixed list returned by
                :func:`_tensor_names_to_skip_for` and forwarded as the unified
                ``skip_patterns`` kwarg of ``load_vlm_model``.  Use when
                overlaying an LLM-only checkpoint onto a VLM model (e.g. swapping
                the language tower while preserving visual + projector params)
                — pass patterns like ``r"model\.visual\."`` so those keys are
                skipped during the overlay.  Only takes effect on the VLM
                dispatch path; ignored when the model is a pure LLM (no
                ``vision_config``).

        Returns:
            Set of model state-dict keys that were loaded from the checkpoint.
        """
        is_vlm = getattr(self.hf_config, "vision_config", None) is not None
        if is_vlm:
            merged_skip_patterns = _tensor_names_to_skip_for(self.hf_config.model_type) + (extra_skip_patterns or [])
            loader_kwargs = {}
            if self.sound_und:
                # The standalone artifact is authoritative for the encoder, so
                # a bundled copy must never overwrite it. Projector weights do
                # resume when present, while legacy/base VLM checkpoints that
                # predate the audio pathway remain valid.
                merged_skip_patterns.append(r"^sound_und_model\.encoder\..+$")
                loader_kwargs["optional_missing_patterns"] = [r"^sound_und_model\.projector\..+$"]
            keys_loaded = load_vlm_model(
                model=self.model,
                checkpoint_path=checkpoint_path,
                credential_path=credential_path,
                parallel_dims=parallel_dims,
                skip_patterns=merged_skip_patterns,
                **loader_kwargs,
            )
        else:
            keys_loaded = load_language_model(
                model=self.model,
                checkpoint_path=checkpoint_path,
                credential_path=credential_path if credential_path else "",
                parallel_dims=parallel_dims,
            )
        log.info(f"HFModel: weights loaded from {checkpoint_path} ({len(keys_loaded)} keys)")
        return keys_loaded

    # Keys added by the VLM collate_fn (vlm/datasets/collate_fn.py) that are NOT valid
    # HF model forward arguments. These must be stripped before calling self.model.forward().
    # A blocklist (not a whitelist) is used so that legitimate kwargs passed via the model's
    # **kwargs — e.g. second_per_grid_ts for Qwen3-VL temporal encoding, output_router_logits
    # for MoE load-balancing — are forwarded correctly even when not named in the signature.
    _COLLATE_NON_MODEL_KEYS: frozenset[str] = frozenset(
        {
            "token_mask",
            "pad_token_id",
            "ignore_index",
            "collated",
            # content_tokens: non-pad token count emitted by custom_collate for the
            # VLMTokensPerSec throughput callback; telemetry only, not a forward arg.
            "content_tokens",
            # Extended packing telemetry emitted by custom_collate (supervision density,
            # l_max, attention-quadratic waste) for VLMTokensPerSec; telemetry only.
            "supervised_tokens",
            "seq_max_len",
            "sum_len_sq",
            # predicted_runtime_ms: the FLOP packer's per-step runtime estimate, surfaced by
            # custom_collate for the VLMTokensPerSec realized-vs-predicted calibration; telemetry only.
            "predicted_runtime_ms",
            "raw_image",
            "raw_video",
            # image_sizes is collected by collate_fn but is NOT a Qwen3-VL forward arg
            # (Qwen3-VL uses image_grid_thw instead). Strip it so strict HF signatures
            # don't reject it. NOTE: image_sizes IS valid for LLaVA-style models — if
            # a future Phase extends to those, remove this entry.
            "image_sizes",
        }
    )

    def forward(self, **kwargs) -> torch.Tensor:
        """Pass-through forward. Returns logits (B, T, V).

        Strips collate-added non-model keys (see ``_COLLATE_NON_MODEL_KEYS``:
        token_mask, pad_token_id, ignore_index, collated, raw_image, raw_video,
        image_sizes) before forwarding. Forces use_cache=False for training.
        All remaining keys (including ``**kwargs`` pass-throughs such as
        second_per_grid_ts) are forwarded unchanged.

        For nemotron_vl: attention_mask is also dropped. NemotronVLModel.get_rope_index
        strips padding positions when attention_mask is present, returning position_ids
        shorter than inputs_embeds (padded_len). With right-padding + causal attention,
        valid tokens never attend to padding tokens regardless, so dropping attention_mask
        is equivalent and avoids the shape mismatch.
        """
        cache_keys = kwargs.pop("tao_video_cache_keys", None)
        filtered = {k: v for k, v in kwargs.items() if k not in self._COLLATE_NON_MODEL_KEYS}
        if self.hf_config.model_type == "nemotron_vl":
            filtered.pop("attention_mask", None)
        filtered["use_cache"] = False
        cache = self._tao_validation_video_feature_cache
        self._tao_validation_video_cache_active = (
            cache is not None and not self.training and not torch.is_grad_enabled()
        )
        self._tao_validation_video_cache_keys = cache_keys
        try:
            out = self.model(**filtered)
        finally:
            self._tao_validation_video_cache_keys = None
            self._tao_validation_video_cache_active = False
        return out.logits
