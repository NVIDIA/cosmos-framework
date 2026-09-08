# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Minimal HFModel for the vfm/ unified VLM training path.

Responsibilities:
  - ``__init__``: meta-init the underlying HF model via the appropriate
    AutoClass (``AutoModelForImageTextToText`` / ``AutoModel`` /
    ``AutoModelForCausalLM`` — see ``HFModel`` for selection rules);
    no weights are loaded.
  - ``tie_embeddings``: re-establishes the input/output embedding tie after
    FSDP wrapping + meta-materialization.
  - ``load_weights``: dispatches to ``load_vlm_model`` (VLM) or
    ``load_language_model`` (LLM) from ``safetensors_loader.py`` based on
    ``vision_config``; returns the set of checkpoint keys that were loaded.
  - ``forward``: pass-through returning logits.

FSDP wrapping lives in ``vfm/models/parallelize_vlm.py::parallelize()``,
NOT here — as does activation checkpointing, which ``parallelize_vlm.apply_ac``
owns via ``ptd_checkpoint_wrapper`` rather than HF's
``gradient_checkpointing_enable`` (see that function for why).
"""

import inspect
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from accelerate import init_on_device
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

import cosmos_framework.model.generator.reasoner.cosmos3_edge  # noqa: F401  registers cosmos3_edge with transformers Auto classes
from cosmos_framework.utils import log
from cosmos_framework.model.generator.utils.safetensors_loader import load_language_model, load_vlm_model
from cosmos_framework.utils.generator.input_probe import maybe_dump_pre_forward
from cosmos_framework.utils.generator.parallelism import ParallelDims

if TYPE_CHECKING:
    from cosmos_framework.configs.base.defaults.reasoner import SoundUnderstandingConfig


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
            from cosmos_framework.model.generator.reasoner.audio.registry import get_audio_encoder_backend
            from cosmos_framework.model.generator.reasoner.audio.utils import patch_reasoner_audio_forward
            from cosmos_framework.data.generator.processors.audio_utils import add_reasoner_audio_special_tokens

            audio_backend = get_audio_encoder_backend(sound_und_config.audio_encoder_type)

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

            with init_on_device("meta", include_buffers=False):
                self.model.sound_und_model = audio_backend.build_model(
                    sound_und_config.projection_hidden_size,
                    reasoner_hidden_size,
                )
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

        # Patch Qwen3-VL / Qwen3-VL-MoE forward for text-only batches (no pixel_values /
        # image_grid_thw). Required to avoid errors when a batch contains only text: every
        # FSDP rank must call visual() each step for all-gather sync; the patch runs a
        # lightweight dummy image and slices the output to [0:0] so it contributes no features.
        # Both backbones share the same patch (see patch_qwen3_vl_forward).
        # Must happen BEFORE parallelize() so FSDP captures the patched forward.
        if hf_config.model_type in ("qwen3_vl", "qwen3_vl_moe") and hasattr(self.model, "model"):
            from cosmos_framework.utils.generator.monkey_patch import patch_qwen3_vl_forward

            patch_qwen3_vl_forward(self.model.model)
            log.info(f"HFModel: applied patch_qwen3_vl_forward ({hf_config.model_type}) for text-only batch support")

        # Swap HF's per-expert Python loop for the fused grouped_mm expert kernel. HF's loop
        # is never the faster choice on the GPUs this trains on, and the patch reuses the
        # existing expert parameters as-is, so there is nothing to trade off and no knob:
        # checkpoint and FSDP layouts are unchanged. Also must precede parallelize().
        if hf_config.model_type == "qwen3_vl_moe":
            from cosmos_framework.utils.generator.monkey_patch import patch_qwen3_vl_moe_grouped_mm_experts

            n_moe_blocks = patch_qwen3_vl_moe_grouped_mm_experts(self.model)
            log.info(f"HFModel: applied grouped_mm experts to {n_moe_blocks} MoE block(s)")

        # Give the vision tower a single varlen attention call per block. HF only does that for
        # flash_attention_2 and otherwise splits the packed patches per image, which costs a
        # device-to-host sync (and a graph break) in every block; cosmos_framework.model.attention takes the
        # packed layout directly. Only the cosmos adapter reaches it, so the other
        # implementations keep HF's split path. Also must precede parallelize().
        if hf_config.model_type in ("qwen3_vl", "qwen3_vl_moe") and attn_implementation == "cosmos":
            from cosmos_framework.utils.generator.monkey_patch import patch_qwen3_vl_vision_varlen_attention

            n_vision_attns = patch_qwen3_vl_vision_varlen_attention(self.model)
            log.info(f"HFModel: applied varlen attention to {n_vision_attns} vision attention module(s)")

        if torch.are_deterministic_algorithms_enabled():
            from cosmos_framework.utils.generator.monkey_patch import patch_siglip2_pos_embed_antialias_off

            for m in self.model.modules():
                if type(m).__name__ == "Siglip2VisionTransformer":
                    patch_siglip2_pos_embed_antialias_off(m)

    def train(self, mode: bool = True) -> "HFModel":
        """Keep immutable audio modules in eval mode while training the Reasoner."""
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
          suffix-lookup loader inherited from the legacy VLM path; dense and
          MoE VLMs both supported).
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

    # Keys the forward signature does not name but the model still consumes out of its
    # ``**kwargs``, so the signature-derived allowlist below would wrongly drop them:
    # second_per_grid_ts drives Qwen-VL temporal encoding, output_router_logits switches on
    # MoE load-balancing bookkeeping. Both arrive as a tensor or a bool, so the guards
    # torch.compile installs on them are cheap shape/dtype guards rather than the per-sample
    # value guards that make the stray string keys ruinous (see forward()).
    _FORWARD_KWARGS_PASSTHROUGHS: frozenset[str] = frozenset(
        {
            "second_per_grid_ts",
            "output_router_logits",
            # Standard HF varlen-attention metadata. The monkey-patched Qwen text
            # forward consumes these from **kwargs and threads them to every decoder layer.
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_q",
            "max_length_k",
        }
    )

    # Both set once and read every step. Class-level defaults rather than __init__
    # assignments because the test suite builds instances through __new__:
    #   _forward_keys_cache: derived from the forward signature on first use.
    #   _logged_dropped_forward_keys: keeps the drop set auditable without a log line per step.
    _forward_keys_cache: frozenset[str] | None = None
    _logged_dropped_forward_keys: bool = False

    @property
    def _forward_keys(self) -> frozenset[str]:
        """Kwargs that may reach ``self.model.forward``: what it declares, plus the known
        :attr:`_FORWARD_KWARGS_PASSTHROUGHS`.

        Read off the BOUND method, so ``self`` is already excluded and an instance-level
        monkey patch is honoured. VAR_POSITIONAL/VAR_KEYWORD entries are skipped: a
        ``**kwargs`` parameter matches anything, so counting it would admit every key and
        turn the allowlist back into a pass-through.

        A forward-wrapping patch therefore has to advertise the inputs it adds, or they get
        dropped here. ``patch_reasoner_audio_forward`` does: it publishes a ``__signature__``
        listing audio_features and friends as keyword-only (``inspect.signature`` prefers
        ``__signature__`` over the ``__wrapped__`` chain that ``functools.wraps`` installs,
        so the audio inputs survive). A future patch that only uses ``@wraps`` and pops its
        arguments out of ``**kwargs`` would need adding to
        :attr:`_FORWARD_KWARGS_PASSTHROUGHS`.

        Derived lazily on the first forward, so ``__init__`` has finished patching by then,
        and memoized rather than re-derived per step.
        """
        if self._forward_keys_cache is None:
            params = inspect.signature(self.model.forward).parameters
            declared = {
                name
                for name, param in params.items()
                if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            }
            self._forward_keys_cache = frozenset(declared | self._FORWARD_KWARGS_PASSTHROUGHS)
        return self._forward_keys_cache

    def forward(self, **kwargs) -> torch.Tensor:
        """Pass-through forward. Returns logits (B, T, V).

        Forwards only the keys in :attr:`_forward_keys` and drops the rest, logging the
        dropped set once per process. What the data batch carries beyond model inputs:
        collate telemetry for VLMTokensPerSec (content_tokens, supervised_tokens,
        seq_max_len, sum_len_sq, predicted_runtime_ms), decode leftovers and collate
        scaffolding (raw_image, raw_video, token_mask, pad_token_id, ignore_index, collated),
        keys belonging to other architectures (image_sizes is LLaVA-style; Qwen-VL uses
        image_grid_thw), and per-sample WebDataset bookkeeping (__url__, __key__,
        dataset_name, dialog_str, sample_index, ...).

        An allowlist, not a blocklist, because HF models funnel unrecognized kwargs down
        into EVERY decoder layer, where torch.compile turns them into guards. ``__url__`` is
        a fresh string each step, so its equality guard fails every step: Dynamo recompiles
        until it trips ``recompile_limit`` (8) and then abandons that code object for the
        rest of the run. Under activation checkpointing all blocks share the single
        ``CheckpointWrapper.forward`` code object, so one stray string key silently reverts
        the whole model to eager after eight steps, having paid eight compilations for it.
        A blocklist reopens that hole the moment the data pipeline grows a field.

        Forces use_cache=False for training, applied after filtering because not every HF
        forward names use_cache in its signature (Qwen3-VL takes it via ``**kwargs``).

        For Nemotron VL (``nemotron_vl`` or its remote-code name
        ``nemotron_siglip2``): attention_mask is also dropped.
        NemotronVLModel.get_rope_index strips padding positions when attention_mask is
        present, returning position_ids shorter than inputs_embeds (padded_len). With
        right-padding + causal attention, valid tokens never attend to padding tokens
        regardless, so dropping attention_mask is equivalent and avoids the shape mismatch.
        """
        probe_step = kwargs.pop("_probe_step", None)
        probe_tag = kwargs.pop("_probe_tag", None)
        forward_keys = self._forward_keys
        filtered = {k: v for k, v in kwargs.items() if k in forward_keys}
        if not self._logged_dropped_forward_keys and len(filtered) != len(kwargs):
            dropped = sorted(set(kwargs) - forward_keys)
            log.info(f"HFModel: dropping non-forward batch keys {dropped} before {type(self.model).__name__}.forward")
            self._logged_dropped_forward_keys = True
        if self.hf_config.model_type in {"nemotron_vl", "nemotron_siglip2"}:
            filtered.pop("attention_mask", None)
        filtered["use_cache"] = False
        maybe_dump_pre_forward(self.model, filtered, probe_step, probe_tag)
        out = self.model(**filtered)
        return out.logits
