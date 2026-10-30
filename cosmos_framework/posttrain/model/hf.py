# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""HFModel — the HF causal-LM body over the general FSDP2/loading primitives.

This is the model-specific layer: it owns HuggingFace knowledge (AutoConfig, AutoClass
selection, meta-init via ``from_config``, dtype normalization, embedding tie, gradient
checkpointing) and DECLARES a ``ShardPlan`` (what to FSDP-wrap) + ``LoadPlan`` (how to
import the pretrained safetensors). The actual sharding and weight loading are delegated
to the model-agnostic primitives:

  - :func:`cosmos_framework.posttrain.model.parallelize.parallelize` (FSDP2 wrap)
  - :func:`cosmos_framework.posttrain.model.weight_loader.load_weights` (safetensors load)

Both are driven by an injected :class:`~cosmos_framework.posttrain.model.parallel_dims.ParallelDims`
(``None`` = single device). The lifecycle order in :meth:`setup` is load-bearing and
mirrors cosmos3 ``_init_vlm``: parallelize -> materialize meta tensors -> tie embeddings
-> load pretrained weights -> (optional) gradient checkpointing.

Scope: LLM (``AutoModelForCausalLM``) + VLM (vision_config -> ``AutoModelForImageTextToText``
/ ``AutoModel`` via ``auto_map``, Qwen3-VL text-only forward patch, collate-key stripping,
per-model RADIO buffer skips) parity with cosmos3 ``HFModel``, plus HF safetensors
:meth:`export`. See design spec section 4.5.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from accelerate import init_on_device
from transformers import AutoConfig, AutoModel, AutoModelForImageTextToText

from cosmos_framework.posttrain.model.parallel_dims import ParallelDims
from cosmos_framework.posttrain.model.parallelize import parallelize
from cosmos_framework.posttrain.model.precision import compute_context, fsdp_param_dtype
from cosmos_framework.posttrain.model.specs import LoadPlan, ShardPlan
from cosmos_framework.posttrain.model.weight_loader import load_weights
from cosmos_framework.posttrain.utils.log import logger

if TYPE_CHECKING:
    from cosmos_framework.posttrain.config.build import Precision
    from cosmos_framework.posttrain.config.model import HFModelConfig

    # Annotation-only. The RUNTIME import stays local to `forward_policy_batch`: `protocol`
    # is imported by the native adapter too, and hoisting it here would put a model module on
    # the import path of every consumer of this one.
    from cosmos_framework.posttrain.model.protocol import VLMPolicyOutput

# Class-name suffixes used by transformers for the repeated decoder block. Only consulted
# as a FALLBACK when ``_no_split_modules`` is absent/unresolvable (see _resolve_block_types).
_DECODER_LAYER_SUFFIXES = ("DecoderLayer", "Block")


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Move every tensor in one model input value while preserving its container shape."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device)  # [*]
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _resolve_block_types(inner: nn.Module) -> tuple[type, ...]:
    """Resolve the FSDP-shardable block classes for ``inner``.

    HF exposes the repeated decoder block(s) as ``_no_split_modules`` — a list of *type
    name strings* (e.g. ``["GPT2Block"]``, ``["Qwen3DecoderLayer"]``). We resolve those
    strings to the actual classes by walking the live module tree and matching
    ``type(m).__name__`` (there is no reliable name->class registry, and remote-code
    classes are not importable by a fixed path). De-duplicated, order-preserving.

    Fallback (when ``_no_split_modules`` is empty or none of its names appear in the tree):
    use the transformers decoder-layer convention — take the element type of any
    ``nn.ModuleList`` whose children's class name ends in ``DecoderLayer``/``Block``. This
    is exactly the "repeated block" FSDP wants to shard leaf-first.
    """
    wanted = list(getattr(inner, "_no_split_modules", None) or [])
    resolved: dict[str, type] = {}
    if wanted:
        wanted_set = set(wanted)
        for module in inner.modules():
            type_name = type(module).__name__
            if type_name in wanted_set:
                resolved.setdefault(type_name, type(module))
    if resolved:
        return tuple(resolved.values())

    # Fallback: transformers decoder-layer convention.
    fallback: dict[str, type] = {}
    for module in inner.modules():
        if isinstance(module, nn.ModuleList):
            for child in module:
                type_name = type(child).__name__
                if type_name.endswith(_DECODER_LAYER_SUFFIXES):
                    fallback.setdefault(type_name, type(child))
    if not fallback:
        # Do NOT raise: single-device runs (setup(None)) never shard, so an empty
        # block_types is harmless (parallelize() no-ops). A multi-GPU run of such a
        # model would fall back to a single root FSDP unit — coarse but functional —
        # so surface a warning rather than break single-device construction.
        logger.warning(
            f"HFModel: could not resolve FSDP block types for {type(inner).__name__} "
            f"(_no_split_modules={wanted!r} matched nothing, no decoder-layer ModuleList found); "
            "using an empty ShardPlan (single-device is unaffected; multi-GPU wraps the root only)."
        )
        return ()
    return tuple(fallback.values())


def _tensor_names_to_skip_for(model_type: str) -> list[str]:
    """Per-model-type tensor-name regex skip list for VLM weight loading.

    Ported verbatim from cosmos3 ``hf_model._tensor_names_to_skip_for``. Patterns match
    the resolved model key and drive BOTH skip-copy and tolerate-absent in the general
    ``load_weights`` (same dual meaning as ``LoadPlan.skip_patterns``). Registered VLMs:
      - Qwen3-VL dense (2B/4B/8B/32B): no skips needed.
      - NemotronH_Nano_VL_V2 (nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16): RADIO backbone
        buffers — initialized by the module, not present in the checkpoint.
    """
    table: dict[str, list[str]] = {
        "NemotronH_Nano_VL_V2": [
            r"vision_model\.radio_model\.summary_idxs",
            r"vision_model\.radio_model\.input_conditioner\.norm_mean",
            r"vision_model\.radio_model\.input_conditioner\.norm_std",
        ],
    }
    return table.get(model_type, [])


def _is_moe_vlm(model: nn.Module) -> bool:
    """Detect whether an HF VLM is a Mixture-of-Experts model (ported from cosmos3).

    MoE VLMs (Qwen3-VL-30B-A3B, Qwen3-VL-235B-A22B) need replicated-gate + FSDP-fused-
    expert shard rules the general ``load_weights`` does NOT implement; the caller raises
    a clear error before sharding. Detection sources (any one is sufficient):
      - ``model.config.text_config.num_experts`` / ``num_local_experts`` (non-None, non-0)
      - the same attrs on ``model.config`` directly (text-only fallback)
      - any state-dict key containing ``.mlp.experts.``
    """
    text_cfg = getattr(model.config, "text_config", None) or model.config
    for attr in ("num_experts", "num_local_experts"):
        value = getattr(text_cfg, attr, None)
        if value is not None and value != 0:
            return True
    for name in model.state_dict().keys():
        if ".mlp.experts." in name:
            return True
    return False


class HFModel(nn.Module):
    """HF causal-LM wrapper: meta-init in ``__init__``, shard+load in :meth:`setup`.

    ``__init__`` allocates NO real memory — the underlying HF model is built on the
    ``meta`` device via ``from_config``. Call :meth:`setup` (with an injected
    ``ParallelDims`` or ``None`` for single device) to FSDP-wrap, materialize, tie, and
    load pretrained weights before training.
    """

    def __init__(
        self,
        model_name_or_path: str,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "flash_attention_2",
        trust_remote_code: bool = False,
        auto_class: str = "AutoModelForCausalLM",
        grad_ckpt: bool = False,
        master_dtype: torch.dtype | None = None,
        reduce_dtype: torch.dtype = torch.float32,
        mixed_precision_mode: str = "fsdp",
    ) -> None:
        """Build the HF body on meta and declare its shard/load plans.

        Args:
            model_name_or_path: HF repo id or local checkpoint dir.
            dtype: the COMPUTE dtype — what the forward/backward runs in: delivered by the FSDP2
                parameter cast (`ShardPlan.param_dtype`) in mode "fsdp", by `torch.autocast` in
                mode "autocast".
            attn_implementation: HF attention backend ("eager" / "sdpa" / "flash_attention_2").
            trust_remote_code: forwarded to `AutoConfig`/`from_config`.
            auto_class: AutoModel class name used for the LLM path.
            grad_ckpt: apply gradient checkpointing in `setup()`.
            master_dtype: the STORAGE dtype — what parameters are held in between steps, and
                therefore what the optimizer's moments inherit. `None` ⇒ `dtype`, which is the
                pre-split behaviour (storage == compute) and keeps every existing caller
                bit-identical. Pass `torch.float32` with `dtype=torch.bfloat16` for the reference
                recipe: fp32 optimizer state, bf16 matmuls, no apex master-weights needed.
            reduce_dtype: gradient all-reduce dtype for FSDP2 mixed precision.
        """
        super().__init__()
        # Storage defaults to compute: the two were ONE knob before the split, so `None` reproduces
        # that exactly. Resolved once here so every use below reads the same value.
        master_dtype = dtype if master_dtype is None else master_dtype
        self.mixed_precision_mode = mixed_precision_mode
        # Built once, re-entered by every `forward`. "cuda" is where HF policies train; on a CPU-only
        # host torch disables the autocast half with a warning, and mode "fsdp" is a null context anyway.
        self.compute_ctx = compute_context(mixed_precision_mode, dtype, "cuda")
        self.ckpt_path = model_name_or_path
        self.grad_ckpt = grad_ckpt
        self.tokenizer = None

        hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        self.hf_config = hf_config

        # AutoClass selection (ported from cosmos3 HFModel). The presence of ``vision_config``
        # picks the VLM path; otherwise the requested ``auto_class`` (default
        # AutoModelForCausalLM) is resolved off the transformers namespace for the LLM path.
        #   - Standard VLM (Qwen3-VL, etc.) or a VLM with no ``auto_map``:
        #     ``AutoModelForImageTextToText`` returns the full conditional-generation class
        #     (e.g. Qwen3VLForConditionalGeneration), which exposes ``.logits`` on forward.
        #     (AutoModelForCausalLM raises for VLM configs, so it cannot be used here.)
        #   - Custom VLM with a non-ImageTextToText ``auto_map`` (e.g. NemotronVL): the map
        #     routes the full causal-LM class through ``AutoModel``; use it for that case only
        #     (never for the standard path — AutoModel returns the backbone without ``.logits``).
        #   - LLM (no ``vision_config``): the requested ``auto_class`` off transformers.
        is_vlm = getattr(hf_config, "vision_config", None) is not None
        if is_vlm:
            auto_map = getattr(hf_config, "auto_map", None) or {}
            if "AutoModelForImageTextToText" in auto_map or not auto_map:
                model_cls = AutoModelForImageTextToText
            else:
                model_cls = AutoModel
        else:
            import transformers

            try:
                model_cls = getattr(transformers, auto_class)
            except AttributeError as exc:
                raise ValueError(f"Unknown auto_class {auto_class!r} on transformers") from exc

        # Meta init: no GPU/CPU parameter memory. FSDP2's fully_shard does NOT auto-
        # materialize meta tensors — setup() materializes explicitly between parallelize()
        # and load_weights(). Buffers are created real (include_buffers=False) so tensors
        # like inv_freq keep their init-time (fp32) values.
        with init_on_device("meta", include_buffers=False):
            # `torch_dtype` is the dtype the module is MATERIALIZED in, i.e. storage — so it takes
            # `master_dtype`, not the compute dtype. (Identical when master_dtype is None.)
            self.model = model_cls.from_config(
                hf_config,
                attn_implementation=attn_implementation,
                torch_dtype=master_dtype,
                trust_remote_code=trust_remote_code,
            )
        self.is_vlm = is_vlm
        logger.info(
            f"HFModel: {hf_config.model_type} ({'VLM' if is_vlm else 'LLM'}), "
            f"master_dtype={master_dtype} param_dtype={dtype} reduce_dtype={reduce_dtype}"
        )

        # MoE-VLM gate (parity with cosmos3 load_vlm_model): the general weight loader does
        # not implement replicated-gate + FSDP-fused-expert shard rules, so refuse loudly
        # rather than silently mis-shard. State-dict keys are available on meta (buffers/
        # params exist as meta tensors), so the ``.mlp.experts.`` probe still works here.
        if is_vlm and _is_moe_vlm(self.model):
            raise NotImplementedError(
                f"HFModel: MoE VLM {hf_config.model_type!r} is not supported — the general "
                "weight_loader lacks replicated-gate + fused-expert (EP) shard handling. "
                "Only dense VLMs (e.g. Qwen3-VL 2B/4B/8B/32B, NemotronH_Nano_VL_V2) are supported."
            )

        # Normalize floating-point PARAMETER dtypes to ``master_dtype`` — the STORAGE dtype, which
        # is what the optimizer reads and writes and what its moments inherit
        # (`torch.optim.AdamW` allocates state with `zeros_like(p)`). Some HF submodules build
        # tensors with an explicit dtype= kwarg (often the checkpoint's bf16) even though
        # from_config installs set_default_dtype(...); FSDP2's mixed-precision init then
        # asserts a single uniform original parameter dtype. Normalize on meta (free).
        # Buffers are left untouched (inv_freq etc. must stay fp32).
        #
        # This used to normalize to ``dtype``, which is the COMPUTE dtype below — the conflation
        # that made fp32-storage/bf16-compute inexpressible and forced apex FusedAdam's
        # master_weights to rebuild the fp32 copy inside the optimizer. See config/rl.py's
        # `HFRunConfig`. ``master_dtype`` defaults to ``dtype``, so the unset path is unchanged.
        n_cast = 0
        with torch.no_grad():
            for p in self.model.parameters(recurse=True):
                if p.is_floating_point() and p.dtype != master_dtype:
                    p.data = p.data.to(master_dtype)
                    n_cast += 1
        if n_cast:
            logger.info(f"HFModel: normalized {n_cast} param(s) to {master_dtype} post-from_config")

        # Declarative plans for the general primitives. block_types are the repeated
        # decoder blocks resolved from _no_split_modules. A native LLM checkpoint needs no
        # key remap, so rename is identity; the only skips are the tied output-embedding
        # key(s): HF de-duplicates tied weights on save (e.g. GPT2/Qwen3 omit lm_head.weight
        # when tie_word_embeddings=True), so the checkpoint lacks them, yet state_dict()
        # still exposes them. setup() calls tie_embeddings() BEFORE load_weights(), so the
        # tied output already shares the (loaded) input-embedding storage — skip-copy +
        # tolerate-absent is exactly correct. Mirrors the cosmos3 safetensors_loader
        # tied-embedding fix-up.
        # `param_dtype` is what the FSDP wrap delivers (`fsdp_param_dtype`): the COMPUTE dtype in mode
        # "fsdp" — FSDP2 all-gathers the (master_dtype) shard, casts to it for the matmuls, and discards
        # the cast copy — or None in mode "autocast", where `torch.autocast` in `forward` owns the compute
        # dtype and the wrap must cast nothing. `reduce_dtype` is the gradient all-reduce dtype — taken
        # from the caller instead of a hardcoded float32, which is what made
        # `HFModelConfig.fsdp_reduce_dtype` a dead knob (every recipe set the value the hardcode
        # already used, so nothing ever noticed).
        self.shard_plan = ShardPlan(
            block_types=_resolve_block_types(self.model),
            param_dtype=fsdp_param_dtype(mixed_precision_mode, dtype),
            reduce_dtype=reduce_dtype,
        )
        # Merge the tied output-embedding skips with the per-model-type VLM buffer skips
        # (e.g. NemotronH_Nano_VL_V2 RADIO buffers) so load_weights both skip-copies AND
        # tolerates the absence of both classes of key. LLM models get an empty VLM list.
        vlm_skips = tuple(_tensor_names_to_skip_for(hf_config.model_type)) if is_vlm else ()
        self.load_plan = LoadPlan(skip_patterns=self._tied_output_skip_patterns() + vlm_skips)

        # Patch Qwen3-VL forward for text-only batches (no pixel_values / image_grid_thw).
        # Every FSDP rank must call visual() each step for all-gather sync; the patch runs a
        # lightweight dummy image and slices it to [0:0] so it contributes no features. Must
        # happen BEFORE setup()/parallelize() so FSDP captures the patched forward. Import is
        # guarded on the qwen3_vl model_type so the LLM path never imports the VLM monkey_patch.
        if hf_config.model_type == "qwen3_vl" and hasattr(self.model, "model"):
            from cosmos_framework.utils.generator.monkey_patch import patch_qwen3_vl_forward

            patch_qwen3_vl_forward(self.model.model)
            logger.info("HFModel: applied patch_qwen3_vl_forward for text-only batch support")
        elif hf_config.model_type == "qwen3_5":
            # Keep the Transformers 5 import behind the family gate; RL retains its TF4 runtime.
            # Install before FSDP captures forwards so mixed-media ranks share a vision schedule.
            from cosmos_framework.posttrain.model.qwen3_5 import patch_qwen3_5_forward

            patch_qwen3_5_forward(self.model.model)
            logger.info("HFModel: applied Qwen3.5 ordered image/video synchronization")

    def _tied_output_skip_patterns(self) -> tuple[str, ...]:
        """Full-key regexes for the tied output-embedding param(s), or ``()`` if untied.

        HF omits these from a saved checkpoint when ``tie_word_embeddings`` is set, so the
        loader must both skip-copy them and tolerate their absence. Resolve the exact
        dotted key(s) from ``_tied_weights_keys`` (what HF itself uses to de-dup on save);
        fall back to the dotted name of ``get_output_embeddings()`` when that list is empty.
        """
        if not getattr(self.hf_config, "tie_word_embeddings", False):
            return ()
        keys = list(getattr(self.model, "_tied_weights_keys", None) or [])
        if not keys:
            out = self.model.get_output_embeddings()
            if out is not None:
                for name, module in self.model.named_modules():
                    if module is out:
                        keys = [f"{name}.weight" if name else "weight"]
                        break
        return tuple(rf"^{re.escape(k)}$" for k in keys)

    @property
    def net(self) -> nn.Module:
        """The trainable module the engine/optimizer/checkpoint handle — ``self.model``.

        Symmetric with the diffusion body's ``.net``; objectives forward through it (it is
        the raw HF module exposing ``.logits``) and the optimizer iterates its params.
        """
        return self.model

    def apply_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing via HF's standard API (non-reentrant)."""
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("HFModel: gradient checkpointing enabled")

    def tie_embeddings(self) -> None:
        """Re-establish the input/output embedding tie after FSDP wrap + materialization.

        Must run AFTER parallelize() + meta materialization (FSDP2 does not auto-
        materialize) and BEFORE load_weights() so the tied pointer survives the load.
        Two strategies matching the HF API split:
          1. ``get_output_embeddings()`` — standard for most causal LMs.
          2. ``_tied_weights_keys`` fallback — models that declare tied keys (e.g.
             ``["lm_head.weight"]``) but return ``None`` from ``get_output_embeddings``;
             walk the dotted key to the owning module and assign ``.weight`` directly.
        Gated on ``hf_config.tie_word_embeddings``.
        """
        if not getattr(self.hf_config, "tie_word_embeddings", False):
            return
        input_embeddings = self.model.get_input_embeddings()
        if input_embeddings is None:
            return
        output_embeddings = self.model.get_output_embeddings()
        if output_embeddings is not None:
            output_embeddings.weight = input_embeddings.weight
            logger.info("HFModel: tied input/output embeddings via get_output_embeddings")
            return
        tied_keys = getattr(self.model, "_tied_weights_keys", None) or ()
        if not tied_keys:
            return
        for key in tied_keys:
            *mod_path, attr = key.split(".")
            target = self.model
            for name in mod_path:
                target = getattr(target, name, None)
                if target is None:
                    logger.warning(
                        f"HFModel.tie_embeddings: could not resolve path {key!r} on "
                        f"{type(self.model).__name__}; leaving that key untied."
                    )
                    break
            else:
                setattr(target, attr, input_embeddings.weight)
                logger.info(f"HFModel: tied {key} via _tied_weights_keys fallback")

    def _materialize_meta(self) -> None:
        """Materialize meta tensors onto the compute device (cuda if available, else cpu).

        FSDP2's ``fully_shard`` leaves parameters on ``meta``; we allocate real storage
        with ``empty_like`` (filled next by ``load_weights``) and move already-real tensors
        (buffers) to the same device. Runs between parallelize() and tie/load. Single
        device (pdims=None) uses the same path — on a CPU-only host it materializes to cpu
        (so the single-device CPU forward works), on a GPU host to cuda.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model._apply(
            lambda t: torch.empty_like(t, device=device) if t.device.type == "meta" else t.to(device),
            recurse=True,
        )

    def setup(self, pdims: ParallelDims | None) -> None:
        """Run the load-bearing lifecycle: parallelize -> materialize -> tie -> load [-> ckpt].

        The ordering is required (mirrors cosmos3 ``_init_vlm``): FSDP2 wrap first so
        params become DTensors, then materialize the meta storage, then re-tie embeddings
        (the tie must exist before the load so the shared pointer is filled once), then
        stripe the pretrained safetensors into each rank's shard. Single device (pdims
        None) short-circuits parallelize/striping inside the primitives and still works.
        """
        parallelize(self.net, self.shard_plan, pdims)
        self._materialize_meta()
        self.tie_embeddings()
        load_weights(self.net, self.ckpt_path, self.load_plan, pdims)
        if self.grad_ckpt:
            self.apply_gradient_checkpointing()

    # Keys added by the VLM collate_fn that are NOT valid HF model forward arguments; they
    # must be stripped before calling self.model(...). A blocklist (not a whitelist) is used
    # so legitimate **kwargs pass-throughs (e.g. second_per_grid_ts for Qwen3-VL temporal
    # encoding, output_router_logits for MoE) are still forwarded. Ported from cosmos3.
    _COLLATE_NON_MODEL_KEYS: frozenset[str] = frozenset(
        {
            "token_mask",
            "pad_token_id",
            "ignore_index",
            "collated",
            # throughput/telemetry-only keys emitted by custom_collate for VLMTokensPerSec.
            "content_tokens",
            "supervised_tokens",
            "seq_max_len",
            "sum_len_sq",
            "predicted_runtime_ms",
            "raw_image",
            "raw_video",
        }
    )

    def forward(self, **kwargs) -> torch.Tensor:
        """Pass-through forward returning logits ``(B, T, V)``; forces ``use_cache=False``.

        Strips collate-added non-model keys (see ``_COLLATE_NON_MODEL_KEYS``) before the
        forward. For ``nemotron_vl``, ``attention_mask`` is also dropped: its
        ``get_rope_index`` strips padding positions when an attention_mask is present,
        returning ``position_ids`` shorter than ``inputs_embeds``; with right-padding +
        causal attention, valid tokens never attend to padding regardless, so dropping the
        mask is equivalent and avoids the shape mismatch. Ported from cosmos3.
        """
        filtered = {k: v for k, v in kwargs.items() if k not in self._COLLATE_NON_MODEL_KEYS}
        if self.hf_config.model_type == "nemotron_vl":
            filtered.pop("attention_mask", None)
        filtered["use_cache"] = False
        with self.compute_ctx:
            return self.model(**filtered).logits

    def validate_policy_batch(self, batch: dict[str, Any]) -> None:
        """Validate HF VLM model inputs before any FSDP rank enters forward."""
        if "input_ids" not in batch:
            raise ValueError("HF VLM policy batch is missing input_ids")
        input_ids = batch["input_ids"]  # [B,L] or invalid
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError(
                f"HF VLM input_ids must be a [B,L] tensor, got {type(input_ids).__name__} "
                f"with shape {getattr(input_ids, 'shape', None)}"
            )
        attention_mask = batch.get("attention_mask")  # [B,L] or None
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
                raise ValueError(
                    f"HF VLM attention_mask must be [B,L], got {type(attention_mask).__name__} "
                    f"with shape {getattr(attention_mask, 'shape', None)}"
                )
            if tuple(attention_mask.shape) != tuple(input_ids.shape):
                raise ValueError(
                    f"HF VLM attention_mask shape {tuple(attention_mask.shape)} does not match "
                    f"input_ids {tuple(input_ids.shape)}"
                )

    def prepare_policy_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Validate and move a complete HF VLM batch during the converged preflight."""
        self.validate_policy_batch(batch)
        try:
            device = next(self.net.parameters()).device
        except StopIteration as error:
            raise ValueError("HF VLM policy has no parameters") from error
        non_model_keys = self._COLLATE_NON_MODEL_KEYS | {"logprob_masks"}
        return {key: _move_to_device(value, device) for key, value in batch.items() if key not in non_model_keys}

    def forward_policy_batch(self, batch: dict[str, Any]) -> "VLMPolicyOutput":
        """Score a collated policy batch, media included (`model/protocol.py::VLMPolicy`).

        Declaring this capability is what lets an objective hand over a batch carrying a VLM's
        processor outputs. Without it, `algorithm/primitives/policy_forward.py` REFUSES the batch
        rather than let `net(input_ids)` drop the conditioning silently -- correct, but it also
        refused every HF VLM, which is the one thing this model class is used for here.

        Validation and CPU-to-device transfer happen in `prepare_policy_batch`, while peers are
        still inside the rank-converged preflight. This method only invokes the model, matching the
        native policy capability contract.

        A text-only HF model handed image kwargs raises from its own `forward` signature. That is
        the intended outcome: loud, at the first batch, naming the unexpected argument.

        Args:
            batch: the already validated, device-local, model-ready batch.

        Returns:
            The logits in a `VLMPolicyOutput`, so objectives do not depend on an HF output class.
        """
        from cosmos_framework.posttrain.model.protocol import VLMPolicyOutput

        return VLMPolicyOutput(logits=self.forward(**batch))

    def export(self, out_dir: str, tokenizer=None) -> None:
        """Export the (possibly FSDP-sharded) model to a HF safetensors directory.

        The reverse of :func:`weight_loader.load_weights`. Gathers the full unsharded
        state dict via DCP's ``get_model_state_dict(full_state_dict=True, cpu_offload=True)``
        — a collective that every rank must enter — then writes on rank 0 only:
        ``save_pretrained(state_dict=..., safe_serialization=True)`` (emits ``config.json`` +
        ``*.safetensors``) and the tokenizer (the passed one, else ``self.tokenizer``) when
        present. A final ``barrier`` keeps ranks in step so a caller can immediately re-read.

        Args:
            out_dir: Destination directory for the HF checkpoint.
            tokenizer: Optional tokenizer to co-save; falls back to ``self.tokenizer``.
        """
        import torch.distributed as dist
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        # Collective — ALL ranks gather (cpu_offload keeps the full copy off the GPU).
        full_sd = get_model_state_dict(self.net, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        if is_rank0:
            self.model.save_pretrained(out_dir, state_dict=full_sd, safe_serialization=True)
            tok = tokenizer if tokenizer is not None else self.tokenizer
            if tok is not None:
                tok.save_pretrained(out_dir)
            logger.info(f"HFModel: exported HF checkpoint to {out_dir}")
        if dist.is_initialized():
            dist.barrier()


def build_hf_model(
    hf_cfg: "HFModelConfig", pdims: ParallelDims | None = None, precision: "Precision | None" = None
) -> HFModel:
    """Build the HuggingFace model body and run its model-owned FSDP lifecycle.

    Args:
        hf_cfg: the model spec (identity / attn / grad-ckpt, plus the deprecated dtype fields).
        pdims: this actor's `ParallelDims`, or None for single device.
        precision: the resolved `Precision` from `resolve_precision(config)`, which splits STORAGE
            (`rl.master_dtype`, what the optimizer and master weights live in) from COMPUTE
            (`rl.param_dtype`, what FSDP casts to for forward/backward). `None` ⇒ derive it from
            `hf_cfg` alone, giving `master == param == hf_cfg.dtype`. Optional so the example and
            test call sites that hold only a model config need no change.

            ONE DELIBERATE BEHAVIOUR CHANGE relative to `main`, where this function took no
            `precision` and passed no `reduce_dtype`: the fallback honours
            `hf_cfg.fsdp_reduce_dtype`. On `main` every caller got `HFModel`'s hard-coded fp32
            reduction and the config field did nothing. The default is `"float32"`, so nothing
            shipped moves; a config that set the field was silently ignored there and is obeyed
            here. `test_model_hf_precision.py` pins both halves.

    Returns:
        A set-up `HFModel`.
    """
    from cosmos_framework.posttrain.config.build import Precision

    if precision is None:
        precision = Precision(
            master=hf_cfg.dtype, param=hf_cfg.dtype, reduce=hf_cfg.fsdp_reduce_dtype, transfer=hf_cfg.dtype
        )
    model = HFModel(
        hf_cfg.name_or_path,
        dtype=getattr(torch, precision.param),
        attn_implementation=hf_cfg.attn_impl,
        trust_remote_code=hf_cfg.trust_remote_code,
        auto_class=hf_cfg.auto_class,
        grad_ckpt=hf_cfg.grad_ckpt,
        master_dtype=getattr(torch, precision.master),
        reduce_dtype=getattr(torch, precision.reduce),
        mixed_precision_mode=precision.mode,
    )
    model.setup(pdims)
    return model
