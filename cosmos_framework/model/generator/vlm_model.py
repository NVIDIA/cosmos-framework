# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""VLMModel: config-instantiable ImaginaireModel for VLM training.

Config usage (in vfm/configs/base/vlm/defaults/model.py):
    config.model = LazyCall(VLMModel)(
        config=VLMModelConfig(),
        checkpoint="${checkpoint}",
    )

Phase 0 — bootstrap via the legacy VLM init path, ParallelDims, and async_safe_ce.
Phase 1 — ParallelDims switches to vfm/utils/parallelism.py.
Phase 2 — legacy init replaced by direct HFModel path (_init_vlm); async_safe_ce
           replaced by vfm/algorithm/loss/cross_entropy.py::cross_entropy_loss.
Phase 3 — init_flash_attn_meta ported to vfm/utils/flash_attn.py;
           config unified under vfm/configs/base/vlm/config.py.
"""

import os
import re
from collections import OrderedDict
from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import torch
import torch.nn as nn
from torch.nn.modules.module import _IncompatibleKeys

from cosmos_framework.utils.lazy_config import instantiate
from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils import log
from cosmos_framework.model.generator.algorithm.loss.cross_entropy import cross_entropy_loss, weighted_cross_entropy_loss
from cosmos_framework.model.generator.algorithm.loss.load_balancing import compute_load_balancing_loss
from cosmos_framework.configs.base.defaults.parallelism import PRECISION_TO_TORCH_DTYPE
from cosmos_framework.configs.base.defaults.reasoner import validate_sound_understanding_config
from cosmos_framework.configs.base.reasoner.defaults.policy_config import VLMModelConfig
from cosmos_framework.model.generator.hf_model import HFModel
from cosmos_framework.model.generator.parallelize_vlm import parallelize
from cosmos_framework.model.generator.utils.moe_utils import collect_hf_moe_lbl_metadata, set_hf_moe_token_mask
from cosmos_framework.model.generator.utils.safetensors_loader import load_vlm_model
from cosmos_framework.utils.generator.input_probe import (
    maybe_dump_forward_result,
    maybe_dump_gradients,
    maybe_dump_model_inputs,
    maybe_dump_post_optimizer,
)
from cosmos_framework.utils.generator.optimizer import OptimizersContainer
from cosmos_framework.utils.generator.parallelism import ParallelDims
from cosmos_framework.utils.generator.reasoner.constant import IGNORE_INDEX
from cosmos_framework.utils.generator.reasoner.pretrained_models_downloader import (
    maybe_download_hf_model_from_s3,
)
from cosmos_framework.utils.generator.reasoner.true_packing import (
    TRUE_PACKING_CPU_PREPARED_KEY,
    assert_packing_temporal_inputs_supported,
)

# Model-type dispatch sets. Using hf_config.model_type (stable HF-defined string)
# rather than backbone.model_name avoids the brittleness of substring-matching a local
# filesystem path that VLMModel._init_vlm has already rewritten (see _init_vlm: the
# downloader returns a local cache path, so the configured model name is lost).
#
# ``qwen3_vl_moe`` covers the 30B-A3B / 235B-A22B variants: MoE dispatch in every
# family helper below is wired for them, and load_vlm_model loads their fused
# ``mlp.experts.*`` tensors through the dense dim-0 shard rule (dim 0 is the
# expert axis). Removing ``qwen3_vl_moe`` here would regress the family helpers.
_QWEN_VL_TYPES = {"qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}
# InternVL variants register both "internvl" and "internvl_chat" as model_type
# in the upstream InternVL HF policy registry.
_INTERNVL_TYPES = {"internvl", "internvl_chat"}

_SOUND_UND_ENCODER_STATE_PREFIX = "model.model.sound_und_model.encoder."


def _canonical_param_name(name: str) -> str:
    """Strip the wrapper prefixes ``parallelize`` inserts around each repeated block.

    ``torch.compile`` contributes ``_orig_mod.`` and the activation-checkpoint wrapper
    ``_checkpoint_wrapped_module.`` (see ``parallelize_vlm.apply_compile`` /
    ``apply_ac``). Unlike ``state_dict()``, ``named_parameters()`` called from the ROOT does
    not undo either rename — ``nn.Module._named_members`` reads each submodule's
    ``_parameters`` directly rather than dispatching to the wrapper's own override — so any
    name-based matching must canonicalize first or silently stop matching as soon as AC or
    compile is enabled.
    """
    return name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")


def _is_sound_und_encoder_state_dict_key(key: str) -> bool:
    """Match only the standalone VLM's audio encoder state namespace."""
    canonical_key = key.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")
    return canonical_key.startswith(_SOUND_UND_ENCODER_STATE_PREFIX)


def _set_sound_und_encoder_dtype_for_fsdp(
    hf_model: HFModel,
    *,
    precision: str,
    fsdp_enabled: bool,
) -> None:
    """Store the replicated encoder in forward dtype only with FSDP MP.

    Without FSDP, the projector stays in the model's master dtype, so the
    encoder must stay there too. With FSDP mixed precision, the raw root casts
    its projector to the forward dtype while the ignored encoder is not cast;
    explicitly matching the encoder avoids an encoder→projector dtype mismatch.
    """
    if hf_model.sound_und and fsdp_enabled:
        hf_model.model.sound_und_model.encoder.to(dtype=PRECISION_TO_TORCH_DTYPE[precision])


def _get_overlay_config(model_type: str) -> tuple[list[str], Callable[[str], bool]]:
    """Return ``(skip_patterns, is_lm_key)`` for the backbone.pretrained_weights overlay.

    ``skip_patterns`` are regex patterns for resolved model keys that are expected to
    be absent from the LLM overlay checkpoint (visual encoder + projector); they are
    passed as ``extra_skip_patterns`` to :meth:`HFModel.load_weights`, which merges
    them with the model-type fixed list and forwards the union as ``skip_patterns``
    to :func:`load_vlm_model` so its Phase-6 completeness check tolerates them.
    Every OTHER missing model key still raises.

    ``is_lm_key`` is a predicate that decides whether a key returned in ``keys_loaded``
    counts as a "language-model parameter" for VLMModel's post-overlay sanity check.
    Implemented as the inverse of ``skip_patterns`` — a loaded key counts as an LM key
    iff it does NOT match any of the visual/projector skip regexes. This mirrors
    exactly what ``load_vlm_model``'s Phase-5 skip logic does, so the two checks can
    never disagree under HF state-dict layout variations (e.g. ``model.model.*``
    vs. ``model.language_model.*``).

    Family-specific because non-LM params differ across VLM families (projectors may
    live outside ``model.visual.*``). Raises ``NotImplementedError`` for unsupported
    families — safer than silently mis-skipping. Add a new entry when onboarding a
    new VLM family.

    MoE note: ``qwen3_vl_moe`` shares the Qwen VL patterns — its experts live under
    the same ``visual.*`` / language-tower split (see the module comment on
    ``_QWEN_VL_TYPES``).
    """
    if model_type in _QWEN_VL_TYPES:
        # Qwen2.5-VL / Qwen3-VL dense + MoE: the visual encoder AND merger/projector
        # both live under a ``visual.*`` subtree (merger is a submodule of visual —
        # see Qwen3VLForConditionalGeneration / Qwen2_5_VLForConditionalGeneration).
        # Every non-visual resolved key counts as an LM key (language_model layers,
        # norm, embed_tokens, top-level lm_head).
        #
        # The ``(?:model\.)*`` prefix makes both the loader-side Phase-5 skip AND
        # the VLMModel-side LM predicate tolerate three layouts uniformly:
        #
        #   1. Bare  (Qwen2.5-VL official HF class)          — ``visual.merger.*``,
        #      ``model.embed_tokens.*`` / ``lm_head.weight``.  See
        #      projects/cosmos3/vlm/scripts/convert_qwenvl_ckpt.py:101-118 which
        #      inspects ``state_dict()`` for keys starting with ``visual.merger``.
        #   2. One wrapper (Qwen3-VL official HF class)      — ``model.visual.*``,
        #      ``model.language_model.*`` / ``lm_head.weight``.
        #   3. Two+ wrappers (HFModel-shim-wrapped callers)  — ``model.model.visual
        #      .*`` etc., e.g. hf_model_test.py::test_vlm_load_hf_native_keys:644.
        #
        # A narrower regex (e.g. requiring a leading ``model.``) would either
        # reject valid Qwen2.5 visual keys in Phase-6 completeness OR misclassify
        # wrapper-layout visual keys as LM keys in the post-overlay safety check.
        skip_patterns = [r"^(?:model\.)*visual\..*"]
        compiled_skips = [re.compile(p) for p in skip_patterns]
        return (
            skip_patterns,
            lambda k: not any(r.match(k) for r in compiled_skips),
        )
    # Nemotron / InternVL / etc: projectors live outside ``model.visual.*``
    # (e.g. ``model.multi_modal_projector.*``, ``model.projector.*``), and lm_head
    # may be nested (``model.lm_head.weight``). The Qwen-shaped skip list would fail
    # Phase-6 completeness on those families; the Qwen-shaped predicate would misreport
    # a successful overlay as "0 language-model parameters". Fail loudly rather than
    # silently.
    raise NotImplementedError(
        f"VLMModel: backbone.pretrained_weights overlay not yet supported for "
        f"model_type={model_type!r}. Supported types: {sorted(_QWEN_VL_TYPES)}. "
        f"Add a new entry in _get_overlay_config() when onboarding a new VLM family "
        f"(see docs/superpowers/specs/2026-04-20-vlm-pretrain-weights-path-llm-design.md §7)."
    )


def _get_vision_encoder_modules(model: nn.Module, model_type: str) -> list:
    if model_type in _QWEN_VL_TYPES:
        # NOTE: intentional semantic change from `model_utils.get_model_vision_encoder`,
        # which returns only [patch_embed, blocks]. Qwen3-VL adds a learnable `pos_embed`
        # (nn.Embedding — see qwen3_vl.py Qwen3VLVisionModel); leaving it trainable while
        # freezing the rest of the vision encoder contradicts the intent of
        # freeze_vision_encoder=True. `hasattr` gate preserves Qwen2.5-VL compatibility
        # (no pos_embed there).
        mods = [model.visual.patch_embed, model.visual.blocks]
        if hasattr(model.visual, "pos_embed"):
            mods.append(model.visual.pos_embed)
        return mods
    elif model_type in _INTERNVL_TYPES:
        return [model.vision_model]
    raise ValueError(f"freeze_vision_encoder not supported for model_type={model_type!r}")


def _get_mm_projector_modules(model: nn.Module, model_type: str) -> list:
    if model_type == "qwen2_5_vl":
        return [model.visual.merger]
    elif model_type in {"qwen3_vl", "qwen3_vl_moe"}:
        mods = [model.visual.merger]
        if hasattr(model.visual, "deepstack_merger_list"):
            mods.append(model.visual.deepstack_merger_list)
        return mods
    elif model_type in _INTERNVL_TYPES:
        # Legacy InternVL helper used `model.model.model.multi_modal_projector`
        # because it operated on a wrapped HFModel (ImaginaireModel -> HFModel ->
        # raw HF InternVL).  We receive the raw HF model directly
        # (hf_model.model), so drop the two wrapper hops.  Best-effort until L1
        # GPU validation on a real InternVL3_5 checkpoint.
        return [model.model.multi_modal_projector]
    raise ValueError(f"freeze_mm_projector not supported for model_type={model_type!r}")


def _get_llm_modules(model: nn.Module, model_type: str) -> list:
    if model_type in _QWEN_VL_TYPES:
        # model.language_model is a @property on Qwen3VLForConditionalGeneration /
        # Qwen2_5_VLForConditionalGeneration that delegates to self.model.language_model
        # — avoids accidentally freezing `visual` which also lives inside self.model.
        # model.lm_head is a top-level submodule on the conditional-generation class.
        return [model.language_model, model.lm_head]
    elif model_type in _INTERNVL_TYPES:
        # Legacy InternVL helper returned `[model.language_model, model.model.lm_head]`
        # for the wrapped HFModel.  Same raw-HF adjustment as mm_projector above:
        # the raw HF InternVL class exposes `.language_model` at the top level but
        # its `lm_head` lives one level deeper under `.model`.  Best-effort until
        # L1 validation.
        return [model.language_model, model.model.lm_head]
    raise ValueError(f"freeze_llm not supported for model_type={model_type!r}")


def _apply_freeze_config(model: nn.Module, model_type: str, cfg) -> int:
    """Apply freeze config in-place. Returns trainable parameter-tensor count.

    ``cfg`` is duck-typed: accepts a ``VLMFreezeConfig`` instance or a
    ``DictConfig`` (LazyCall-backed) where ``__attrs_post_init__`` may not
    have fired. The mutual-exclusivity check below mirrors the attrs
    validator so both paths fail loudly before any parameter is frozen.
    """
    trainable_params = getattr(cfg, "trainable_params", None)
    frozen_params = getattr(cfg, "frozen_params", None)

    # Defensive mutual-exclusivity guard — runs BEFORE any freeze, even on LazyCall path.
    if trainable_params is not None and frozen_params is not None:
        raise ValueError("VLMFreezeConfig: set at most one of trainable_params or frozen_params, not both.")

    # Step 1 — legacy named flags via module-probing
    if cfg.freeze_vision_encoder:
        for m in _get_vision_encoder_modules(model, model_type):
            for p in m.parameters():
                p.requires_grad = False

    if cfg.freeze_mm_projector:
        for m in _get_mm_projector_modules(model, model_type):
            for p in m.parameters():
                p.requires_grad = False

    if cfg.freeze_llm:
        for m in _get_llm_modules(model, model_type):
            for p in m.parameters():
                p.requires_grad = False

    # Step 2 — regex override (mutually exclusive; already validated above).
    #
    # Patterns are matched against the CANONICAL name, so an expression aimed inside a
    # repeated block (e.g. r"layers\.\d+\.self_attn") keeps matching after `parallelize`
    # wraps that block for activation checkpointing and torch.compile. Matching the raw
    # `named_parameters()` name would fail silently: the `assert n > 0` below still passes as
    # long as some other parameter matched.
    #
    # `remove_duplicate=False` is required for tied weights. Qwen3 configs set
    # `tie_word_embeddings=True`, so `hf_model.tie_embeddings()` makes
    # `lm_head.weight` and `model.embed_tokens.weight` the same tensor. The default
    # `named_parameters()` dedups by tensor id and keeps only the first traversed
    # name (`model.embed_tokens.weight`); a regex aimed at `lm_head` would silently
    # match nothing and user intent would be lost. Iterating with duplicates
    # preserves both names so either can trigger a match.
    if trainable_params is not None:
        # OR-semantics across tied names: first freeze everything, then unfreeze
        # any tensor whose *any* registered name matches. Cannot write
        # `requires_grad = any(...)` directly because a second visit could flip
        # True back to False on the same shared tensor.
        for p in model.parameters():
            p.requires_grad = False
        for param_name, p in model.named_parameters(remove_duplicate=False):
            if any(re.search(pat, _canonical_param_name(param_name)) for pat in trainable_params):
                p.requires_grad = True
    elif frozen_params is not None:
        for param_name, p in model.named_parameters(remove_duplicate=False):
            if any(re.search(pat, _canonical_param_name(param_name)) for pat in frozen_params):
                p.requires_grad = False

    n = sum(p.requires_grad for p in model.parameters())
    if not any([cfg.freeze_vision_encoder, cfg.freeze_mm_projector, cfg.freeze_llm, trainable_params, frozen_params]):
        log.warning("freeze config: no freeze mechanism set — all parameters are trainable (full fine-tune)")
    assert n > 0, "freeze config left 0 trainable parameters — check patterns"
    return n


class VLMModel(ImaginaireModel):
    """Config-instantiable ImaginaireModel for VLM training.

    Args:
        config:          VLMModelConfig (parallelism, compile, AC, precision,
                         policy, freeze, ema, deterministic).
        checkpoint:      root CheckpointConfig (load_path, load_from_object_store).
    """

    emits_exact_validation_stats: bool = True

    def __init__(self, config: VLMModelConfig, checkpoint):
        super().__init__()
        from cosmos_framework.utils.generator.flash_attn import init_flash_attn_meta

        self.config = config
        validate_sound_understanding_config(config.sound_und_config, sound_und=config.sound_und)
        # Expose model.precision so LowPrecisionCallback can read it (mirrors OmniMoTModel).
        self.precision = getattr(torch, config.precision)
        self._parity_probe_step: int = 0
        init_flash_attn_meta(config.deterministic)
        self._init_vlm(config, checkpoint)

        # Apply freeze before the optimizer is built — ``build_optimizer`` reads
        # ``requires_grad`` off ``named_parameters``.
        n_trainable = _apply_freeze_config(self.model.model, self.hf_config.model_type, self.config.freeze)
        if config.sound_und:
            # The standalone artifact is the sole source of encoder weights.
            # Keep it immutable even when a broad trainable_params expression
            # (or a full-finetune config) would otherwise re-enable it.
            self.model.model.sound_und_model.encoder.requires_grad_(False)
            self.model.model.sound_und_model.encoder.eval()
            if config.sound_und_config.freeze_projector:
                self.model.model.sound_und_model.projector.requires_grad_(False)
                self.model.model.sound_und_model.projector.eval()
            n_trainable = sum(parameter.requires_grad for parameter in self.model.model.parameters())
            assert n_trainable > 0, "audio freeze policy left 0 trainable parameters — check freeze patterns"
        log.info(
            f"freeze config applied (model_type={self.hf_config.model_type}): {n_trainable} trainable parameter tensors"
        )

        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            # Both CE variants normalize over every rank in the world, which is only the
            # count they want while each rank holds a different sample. CP breaks that:
            # its ranks hold segments of one sequence, and no reduction over those has
            # been verified against the objective trained here (see the loss module
            # docstring). ``_init_vlm`` already asserts cp == 1 for the attention path,
            # which makes this unreachable today — it is here so the loss keeps its own
            # requirement if CP is ever wired into attention.
            raise NotImplementedError(
                f"VLM loss does not support context parallelism (got cp={self.parallel_dims.cp}); "
                "set parallelism.context_parallel_shard_degree=1."
            )

        if config.policy.use_weighted_ce:
            log.info(f"Using weighted CE loss with exponent={config.policy.weighted_ce_exponent}")
            self._loss_fn = partial(
                weighted_cross_entropy_loss,
                exponent=config.policy.weighted_ce_exponent,
                loss_scaling_factor=1.0,
                ignore_index=IGNORE_INDEX,
            )
        else:
            self._loss_fn = partial(
                cross_entropy_loss,
                loss_scaling_factor=1.0,
                ignore_index=IGNORE_INDEX,
            )
        # Dense weighted CE is normalized once over the whole gradient-accumulation window.
        # The trainer averages microbatch losses by K; training_step therefore backprops the
        # unnormalized WORLD numerator and this hook applies K / sum(global denominator).
        self._window_normalize_weighted_ce = bool(
            config.policy.use_weighted_ce
            and config.policy.normalize_weighted_ce_over_accumulation_window
            and self.hf_config.model_type == "qwen3_vl"
        )
        self._weighted_ce_window_denominator: torch.Tensor | None = None
        self._weighted_ce_window_microbatches: int = 0

    def _init_vlm(self, config: VLMModelConfig, checkpoint) -> None:
        """Initialize VLM without the legacy ModelRegistry (Phase 2+).

        Sequence (ordering is critical — do not reorder):
          a. Download HF weights from S3 to local cache.
          b. Meta-init HFModel (params on meta, buffers on CPU via include_buffers=False;
          c. Build ParallelDims + device mesh.
          d. Apply activation checkpointing, torch.compile and FSDP2 via parallelize() —
             meta tensors are NOT auto-materialized.
          e. Explicitly materialize meta tensors; move CPU buffers to CUDA.
          f. Tie output embedding → input embedding if tie_word_embeddings=True.
          g. Load pretrain weights into sharded CUDA tensors.
        """
        policy = config.policy

        load_pretrain_weights = checkpoint.load_path == ""
        log.info(f"checkpoint.load_path: {checkpoint.load_path!r} | load_pretrain_weights: {load_pretrain_weights}")

        # ── a. Download HF model files (config + tokenizer; weights only if no ckpt) ──
        local_path = maybe_download_hf_model_from_s3(
            policy.backbone.model_name,
            checkpoint.load_from_object_store.credentials,
            checkpoint.load_from_object_store.bucket,
            include_model_weights=load_pretrain_weights,
        )
        # local_path is exposed below as self.model_name_or_path; the (frozen) policy
        # config is not mutated.

        # ── b. Meta-init HFModel ──
        # Allocate params in the FSDP master dtype (float32) so each rank's
        # sharded param storage matches ``MixedPrecisionPolicy.reduce_dtype``
        # (the same field); MixedPrecisionPolicy down-casts to ``precision``
        # (bfloat16) for forward/backward.
        hf_model = HFModel(
            model_name_or_path=local_path,
            dtype=PRECISION_TO_TORCH_DTYPE[config.parallelism.fsdp_master_dtype],
            # Default "cosmos" → cosmos_framework.model.attention (NATTEN/blackwell-fmha);
            # set policy.attn_implementation=flash_attention_2 to fall back.
            attn_implementation=policy.attn_implementation,
            sound_und=config.sound_und,
            sound_und_config=config.sound_und_config,
            # Token policy needs the configured identity because local_path is
            # a cache directory and no longer identifies the Edge Reasoner.
            configured_model_name_or_path=policy.backbone.model_name,
        )
        # ── b.1. Early family-gate for backbone.pretrained_weights ──
        # Fail-fast on unsupported VLM families BEFORE any expensive work
        # (parallelize, materialize, base-weight load, overlay download).
        # ``hf_config.model_type`` is populated by HFModel's meta-init; no
        # weights touched yet. Empty backbone_path == no overlay, matching
        # the later overlay guard at step g.2.
        if policy.backbone.pretrained_weights.backbone_path:
            _get_overlay_config(hf_model.hf_config.model_type)

        # ── c. Build ParallelDims + device mesh ──
        # Overlay-mesh design (see vfm/utils/parallelism.py): cp/cfgp do NOT
        # consume FSDP rank slots, so dp_replicate * dp_shard == world_size
        # alone. The VLM HFModel doesn't have a CP-aware attention path.
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        _dp_replicate = config.parallelism.data_parallel_replicate_degree
        # Single-process run: force dp_replicate=1 so ParallelDims doesn't
        # auto-infer it to world_size (which would equal 1 anyway, but guards
        # against environments where WORLD_SIZE is unset/inconsistent).
        if not torch.distributed.is_initialized():
            _dp_replicate = 1

        parallel_dims = ParallelDims(
            world_size=world_size,
            dp_shard=config.parallelism.data_parallel_shard_degree,
            dp_replicate=_dp_replicate,
            cp=config.parallelism.context_parallel_shard_degree,
            enable_inference_mode=False,
        )

        # VLM does not currently support cp or cfgp. CP needs a CP-aware
        # attention path (see ``vfm/models/mot/context_parallel_utils.py``) that
        # is not wired into the VLM HFModel; CFGP is inference-only.
        assert parallel_dims.cp == 1, f"VLM does not support CP (got cp={parallel_dims.cp})"
        assert parallel_dims.cfgp == 1, f"VLM does not support CFGP (got cfgp={parallel_dims.cfgp})"

        if torch.distributed.is_initialized():
            parallel_dims.build_meshes(device_type="cuda")

        # dp_enabled, not dp_shard_enabled: replicate-only (dp_shard == 1,
        # dp_replicate == world_size) still goes through fully_shard — on a 2-D
        # mesh whose shard dim is 1 — so its mixed-precision policy applies and
        # the encoder must be cast alongside the projector. Must track
        # ``apply_fsdp``'s own guard.
        _set_sound_und_encoder_dtype_for_fsdp(
            hf_model,
            precision=config.precision,
            fsdp_enabled=parallel_dims.dp_enabled,
        )

        # ── d. Apply activation checkpointing, torch.compile and FSDP2 ──
        # config.compile is threaded through so model.config.compile.enabled=True
        # actually compiles each block in place (was previously a dead config on
        # the VLM path — only the MoT path consumed it). See parallelize_vlm for why the
        # three passes must run in this order (AC wraps, compile compiles the wrapper, FSDP
        # shards it).
        # Called unconditionally, NOT under an is_initialized() guard: activation
        # checkpointing and compile are independent of distribution, and a single-process
        # run needs AC as much as a sharded one. apply_fsdp no-ops on its own when there is
        # no data-parallel axis at all, which is the only case reachable without dist, so no
        # mesh is touched here.
        parallelize(
            hf_model,
            parallel_dims,
            config.parallelism,
            config.precision,
            activation_checkpointing=config.activation_checkpointing,
            compile_config=config.compile,
        )

        # ── e. Materialize meta tensors on CUDA ──
        # FSDP2 fully_shard does not auto-materialize meta tensors, so allocate
        # empty CUDA tensors here for load_weights() to copy into.
        # To enable FSDP-2 CPU offload later: add a CPU-materialize branch and
        # pair with ``offload_policy=CPUOffloadPolicy()`` in ``parallelize_vlm``.
        hf_model.model._apply(
            lambda t: torch.empty_like(t, device="cuda") if t.device.type == "meta" else t.to("cuda"),
            recurse=True,
        )
        if config.sound_und:
            # ``to_empty`` deliberately discards meta initialization. Encoder
            # tensors are populated by the authoritative artifact below. Let
            # each audio backend restore its fresh state before checkpoint load;
            # a VLM/DCP checkpoint may overwrite the projector later.
            audio_model = hf_model.model.sound_und_model
            buffer_device = audio_model.projector.linear_fc1.weight.device
            audio_model.init_weights(buffer_device=buffer_device)

        # ── f. Tie embeddings (replaces the legacy post_to_empty_hook) ──
        hf_model.tie_embeddings()

        # ── g. Load pretrain weights ──
        if load_pretrain_weights:
            if policy.backbone.safetensors_path:
                safetensors_local_path = maybe_download_hf_model_from_s3(
                    policy.backbone.safetensors_path,
                    checkpoint.load_from_object_store.credentials,
                    checkpoint.load_from_object_store.bucket,
                    include_model_weights=True,
                )
            else:
                safetensors_local_path = local_path

            hf_model.load_weights(
                checkpoint_path=safetensors_local_path,
                credential_path=None,  # local path after download
                parallel_dims=parallel_dims if torch.distributed.is_initialized() else None,
            )

            # ── g.2. Optional LLM overlay (backbone.pretrained_weights) ──
            # Overlay the language tower with a separate LLM checkpoint.
            # Visual + projector params are preserved from the VLM load above
            # (the overlay's visual/projector keys are folded into load_vlm_model's
            # unified skip_patterns by HFModel.load_weights).  The existing name
            # converter in load_vlm_model tail-matches raw LLM keys into
            # model.language_model.*, so no temp-dir remap is needed.
            # Mirrors legacy vlm/train.py:221-233 semantics.
            llm_path = policy.backbone.pretrained_weights.backbone_path

            if llm_path:
                overlay_skip_patterns, is_lm_key = _get_overlay_config(hf_model.hf_config.model_type)
                llm_local_path = maybe_download_hf_model_from_s3(
                    llm_path,
                    checkpoint.load_from_object_store.credentials,
                    checkpoint.load_from_object_store.bucket,
                    include_model_weights=True,
                    require_s3_exists=True,
                )
                keys_loaded = hf_model.load_weights(
                    checkpoint_path=llm_local_path,
                    credential_path=None,
                    parallel_dims=parallel_dims if torch.distributed.is_initialized() else None,
                    extra_skip_patterns=overlay_skip_patterns,
                )
                lm_loaded = {k for k in keys_loaded if is_lm_key(k)}
                if not lm_loaded:
                    raise RuntimeError(
                        f"VLMModel overlay: loaded 0 language-model parameters from "
                        f"{llm_path!r} (local path: {llm_local_path!r}). The LLM "
                        "checkpoint did not match any language_model.* key in the "
                        "VLM; check model-family / layer-count compatibility."
                    )
                log.info(f"VLMModel: overlaid {len(lm_loaded)} language-model params from {llm_path}")

        # ── h. Load the immutable standalone audio encoder artifact ──
        # This runs for both fresh starts and DCP resumes. On resume, DCP may
        # subsequently restore the same encoder when it was checkpointed; when
        # exclusion is enabled, these artifact weights remain authoritative.
        if config.sound_und:
            audio_config = config.sound_und_config
            loaded_audio_keys = load_vlm_model(
                model=hf_model.model.sound_und_model.encoder,
                checkpoint_path=audio_config.encoder_checkpoint_path,
                credential_path=audio_config.encoder_checkpoint_credentials_path or None,
                parallel_dims=parallel_dims if torch.distributed.is_initialized() else None,
            )
            log.info(
                f"VLMModel: loaded {len(loaded_audio_keys)} audio encoder tensors from "
                f"{audio_config.encoder_checkpoint_path}"
            )

        self.model = hf_model
        self.parallel_dims = parallel_dims
        self.model_name_or_path = local_path
        self.hf_config = hf_model.hf_config

    def on_train_start(self, memory_format) -> None:
        """Called by trainer after model.to("cuda"). No device move needed here."""

    def on_after_backward(self, iteration: int = 0) -> None:
        """Capture exact pre-clip gradients when deep parity probing is enabled."""
        maybe_dump_gradients(self.model.model, self._parity_probe_step, tag="i4")

    def on_before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer | OptimizersContainer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int,
    ) -> None:
        """Finish exact weighted-CE normalization over a gradient-accumulation window."""
        del scheduler, iteration
        if self._weighted_ce_window_denominator is None:
            return
        if self._weighted_ce_window_microbatches <= 0:
            raise RuntimeError("weighted-CE denominator exists without accumulated microbatches")
        scale = self._weighted_ce_window_microbatches / self._weighted_ce_window_denominator.clamp(min=1)
        optimizers = optimizer.optimizers if isinstance(optimizer, OptimizersContainer) else [optimizer]
        for inner_optimizer in optimizers:
            for group in inner_optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(scale)
        self._weighted_ce_window_denominator = None
        self._weighted_ce_window_microbatches = 0

    def on_before_zero_grad(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int,
    ) -> None:
        """Capture post-step parameters and optimizer state before gradients are cleared."""
        del scheduler, iteration
        maybe_dump_post_optimizer(self.model.model, optimizer, self._parity_probe_step, tag="i4")

    def state_dict(
        self,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """Optionally omit the immutable artifact-backed audio encoder."""
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        exclude_encoder = (
            self.config.sound_und and self.config.sound_und_config.exclude_frozen_encoder_from_training_checkpoint
        )
        if not exclude_encoder:
            return state_dict

        for key in tuple(state_dict):
            if (not prefix or key.startswith(prefix)) and _is_sound_und_encoder_state_dict_key(
                key.removeprefix(prefix)
            ):
                del state_dict[key]
        return state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ) -> _IncompatibleKeys:
        """Restore trainable state while preserving an excluded encoder artifact."""
        exclude_encoder = (
            self.config.sound_und and self.config.sound_und_config.exclude_frozen_encoder_from_training_checkpoint
        )
        if not exclude_encoder:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

        filtered_state_dict = OrderedDict(
            (key, value) for key, value in state_dict.items() if not _is_sound_und_encoder_state_dict_key(key)
        )
        metadata = getattr(state_dict, "_metadata", None)
        if metadata is not None:
            filtered_state_dict._metadata = metadata  # type: ignore[attr-defined]

        incompatible = super().load_state_dict(filtered_state_dict, strict=False, assign=assign)
        missing_keys = [key for key in incompatible.missing_keys if not _is_sound_und_encoder_state_dict_key(key)]
        unexpected_keys = list(incompatible.unexpected_keys)
        if strict and (missing_keys or unexpected_keys):
            errors = []
            if missing_keys:
                errors.append(f"Missing key(s): {', '.join(repr(key) for key in missing_keys)}")
            if unexpected_keys:
                errors.append(f"Unexpected key(s): {', '.join(repr(key) for key in unexpected_keys)}")
            raise RuntimeError(f"Error(s) in loading state_dict for {type(self).__name__}: " + "; ".join(errors))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        """Build optimizer + scheduler from hydra-instantiated configs.

        Freeze was applied in ``__init__``; ``build_optimizer`` reads
        ``requires_grad`` off ``named_parameters``.

        Per-component LR multipliers (e.g. vision_encoder=0.1x in the legacy
        recipe) are not currently restored on this code path. Substring matching
        in ``_filter_params_grouped`` (``vfm/utils/optimizer.py:148-159``) would
        need correct substrings for Qwen3-VL param names (``model.visual.*``)
        — separate follow-up MR.
        """
        optimizer = instantiate(optimizer_config, model=self.model)
        scheduler = instantiate(scheduler_config, optimizer=optimizer)
        return optimizer, scheduler

    def _set_moe_token_mask(self, attention_mask: torch.Tensor | None) -> None:
        """Tell the MoE blocks which rows of this step are real tokens.

        HF's sparse MoE block takes ``forward(hidden_states)`` and nothing else, so the
        patched forward reads the mask off the module instead; see
        ``moe_utils.set_hf_moe_token_mask``. Without it the padded rows are dispatched to
        experts and counted in the routing statistics, and the auxiliary loss trains the
        router to balance rows that carry no supervision — ``ignore_index`` keeps them out of
        the cross-entropy, not out of this.

        Publishing it unconditionally (rather than only when ``config.lbl.coeff`` is set) is
        deliberate: the mask also decides which rows the experts compute, so skipping it would
        leave the expert GEMMs padding-dependent even with the aux loss off.
        """
        if self.hf_config.model_type != "qwen3_vl_moe":
            return

        set_hf_moe_token_mask(self.model, attention_mask)

    def _moe_load_balancing_loss(self) -> torch.Tensor | None:
        """Build the MoE load-balancing auxiliary loss from this forward's routing stats.

        Returns ``None`` for a dense backbone or when ``config.lbl.coeff`` is unset. The
        statistics are stashed per layer by the patched MoE block forward and popped here;
        see ``moe_utils.collect_hf_moe_lbl_metadata`` for why popping matters.

        The stash is a side channel out of the block's forward, so it only stays
        grad-connected under NON-reentrant activation checkpointing — reentrant recompute
        runs the first pass under ``no_grad`` and would silently detach the router
        probabilities, leaving the aux loss with no gradient. ``parallelize_vlm.apply_ac``
        wraps blocks with ``ptd_checkpoint_wrapper``, whose ``checkpoint_impl`` defaults to
        ``CheckpointImpl.NO_REENTRANT``.

        Must be called OUTSIDE any compiled region: ``method="global"`` issues DP
        collectives, which torch.compile may reorder into a deadlock (see
        ``compute_load_balancing_loss``). ``training_step`` is eager, so that holds.

        Imports are local because the MoE modules pull in Triton kernels that the dense
        Qwen3-VL path must not require.
        """
        if self.hf_config.model_type != "qwen3_vl_moe":
            return None

        return compute_load_balancing_loss(
            collect_hf_moe_lbl_metadata(self.model),
            coeff=self.config.lbl.coeff,
            method=self.config.lbl.method,
            device_mesh=self.parallel_dims.dp_mesh if self.parallel_dims is not None else None,
        )

    def _prepare_true_packing(self, data: dict[str, Any]) -> None:
        """Prepare dense Qwen3-VL true-packed positions and standard varlen metadata in place.

        Padded batches return immediately and continue using the current model-internal M-RoPE
        path. Packed batches are accepted only for the one backend/model combination whose
        block-diagonal attention and position parity are covered by this MR.
        """
        true_packing = data.get("true_packing", False)
        if true_packing is False:
            return
        if true_packing is not True:
            raise TypeError("true_packing must be a Python bool")
        if self.hf_config.model_type != "qwen3_vl":
            raise NotImplementedError(
                f"True packing is validated only for dense qwen3_vl; got model_type={self.hf_config.model_type!r}"
            )
        if self.config.sound_und:
            raise NotImplementedError(
                "True packing is not validated for audio understanding inputs; use padded batching."
            )
        if self.config.policy.attn_implementation != "cosmos":
            raise NotImplementedError(
                "True packing requires policy.attn_implementation='cosmos'; other backends may "
                "ignore varlen metadata and permit cross-sample attention"
            )
        if self.config.policy.use_weighted_ce and not self._window_normalize_weighted_ce:
            raise ValueError(
                "True packing with weighted CE requires "
                "policy.normalize_weighted_ce_over_accumulation_window=True. Set it explicitly "
                "in both packed and padded A/B arms so the optimizer objective is unchanged."
            )
        if data.pop(TRUE_PACKING_CPU_PREPARED_KEY, None) is not True:
            raise RuntimeError(
                "true-packed batches must carry CPU-precomputed position_ids; construct them in the "
                "packing dataloader before the trainer H2D copy"
            )
        assert_packing_temporal_inputs_supported(data)
        seq_lens = data.pop("seq_lens")
        if not isinstance(seq_lens, list) or not all(isinstance(length, int) for length in seq_lens):
            raise TypeError("seq_lens must be a list of Python ints")
        packed_cu_seq_lens = data.pop("packed_cu_seq_lens")
        packed_max_length = data.pop("packed_max_length")
        if not isinstance(packed_cu_seq_lens, torch.Tensor) or packed_cu_seq_lens.dtype != torch.int32:
            raise TypeError("packed_cu_seq_lens must be one int32 tensor")
        if not isinstance(packed_max_length, int):
            raise TypeError("packed_max_length must be a Python int")

        position_ids = data.get("position_ids")
        input_ids = data.get("input_ids")
        if not isinstance(position_ids, torch.Tensor) or position_ids.dtype != torch.long:
            raise TypeError("CPU-precomputed position_ids must be one int64 tensor")
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("true-packed input_ids must be a tensor")
        expected_position_shape = (3, 1, input_ids.shape[1])
        if tuple(position_ids.shape) != expected_position_shape:
            raise ValueError(
                f"true-packed position_ids must have shape {expected_position_shape}, got {tuple(position_ids.shape)}"
            )
        if position_ids.device != input_ids.device:
            raise ValueError(
                "position_ids and input_ids must be moved to the same device by the trainer; "
                f"got {position_ids.device} and {input_ids.device}"
            )

        # Assign the same cumulative-boundary tensor object to Q and K after its one H2D copy.
        # The cosmos adapter accepts the standard HF protocol directly.
        data["cu_seq_lens_q"] = packed_cu_seq_lens
        data["cu_seq_lens_k"] = packed_cu_seq_lens
        data["max_length_q"] = packed_max_length
        data["max_length_k"] = packed_max_length
        data.pop("true_packing", None)

    def training_step(self, data: dict, iteration: int) -> tuple[dict, torch.Tensor]:
        """forward → CE loss, plus the MoE load-balancing loss when ``config.lbl`` enables it.

        position_ids are intentionally NOT precomputed here: both the dense
        (Qwen3-VL) and MoE (Qwen3-VL-MoE) backbones derive multimodal-RoPE
        positions internally via their own ``get_rope_index`` when
        ``position_ids is None`` (their forward is monkey-patched — see
        ``hf_model`` / ``monkey_patch.patch_qwen3_vl_forward``). Relying on the
        model's built-in path keeps the native ``[3, B, N]`` mRoPE layout and
        avoids a redundant external reimplementation.

        ``attention_mask`` is forwarded rather than dropped. Under the ``cosmos`` attention
        implementation it changes nothing: ``hf_model`` registers that name in HF's
        ``ALL_ATTENTION_FUNCTIONS`` only, and ``masking_utils._preprocess_mask_arguments``
        returns no mask at all for an implementation absent from
        ``ALL_MASK_ATTENTION_FUNCTIONS``, so the adapter attends causally over the whole
        padded row — safe because ``custom_collate`` pads on the RIGHT and the pad rows are
        ``ignore_index`` in the labels.

        It is load-bearing for the sdpa / flash fallbacks (``policy.attn_implementation``).
        Those DO build a mask, and when ``attention_mask`` is None that same function reads
        the position ids as a packed batch (``find_packed_sequence_indices``: any step other
        than +1 starts a new sequence). Qwen3-VL's mRoPE temporal ids repeat across an
        image/video block, so every vision token would be taken for the start of a new
        sequence and attention would be severed at each one.

        It also reaches ``get_rope_index``, where it only keeps the scan from walking into
        the trailing pads: right padding puts the real tokens first, so their positions —
        and the loss — are the same either way. Left-padding would break that silently;
        ``unit_tests/test_monkey_patch.py`` pins it.
        """
        self._prepare_true_packing(data)
        labels = data.pop("labels")
        self._set_moe_token_mask(data.get("attention_mask"))

        maybe_dump_model_inputs(data, iteration, tag="i4", labels=labels)
        self._parity_probe_step = iteration

        logits = self.model(_probe_step=iteration, _probe_tag="i4", **data)
        loss_kwargs: dict[str, Any] = {}
        if self.config.policy.use_weighted_ce:
            loss_kwargs = {
                "probe_step": iteration,
                "probe_tag": "i4",
                "cu_seq_lens": data.get("cu_seq_lens_q"),
            }
        loss_result = self._loss_fn(
            logits,
            labels,
            return_stats=True,
            **loss_kwargs,
        )
        if not isinstance(loss_result, tuple):
            raise TypeError("training loss must return statistics when return_stats=True")
        loss, loss_stats = loss_result
        backward_loss: torch.Tensor
        if self._window_normalize_weighted_ce:
            denominator = loss_stats.global_objective_denominator
            backward_loss = loss * denominator
            if self._weighted_ce_window_denominator is None:
                self._weighted_ce_window_denominator = denominator.detach().clone()
            else:
                self._weighted_ce_window_denominator.add_(denominator)
            self._weighted_ce_window_microbatches += 1
        else:
            backward_loss = loss

        ce_loss = loss.detach().clone()
        load_balancing_loss = self._moe_load_balancing_loss()
        if load_balancing_loss is not None:
            loss = loss + load_balancing_loss
            backward_loss = backward_loss + load_balancing_loss
        maybe_dump_forward_result(logits, {"ce_loss": ce_loss, "total_loss": loss}, iteration, tag="i4")

        # Callbacks accumulate these primitives on every microbatch and reduce over WORLD only at
        # logging cadence. With explicit window normalization they are the local ratio-of-sums
        # primitives. Otherwise the trained objective is the historical mean of independently
        # normalized microbatch/rank losses, so emit (loss, 1) and aggregate that exact mean rather
        # than silently logging a different exposure-weighted objective.
        if self._window_normalize_weighted_ce:
            train_objective_numerator = loss_stats.objective_numerator
            train_objective_denominator = loss_stats.objective_denominator
        else:
            train_objective_numerator = loss.detach()
            train_objective_denominator = torch.ones_like(train_objective_numerator)
        output = {
            "loss": loss,
            "labels": labels,
            "train_objective_numerator": train_objective_numerator,
            "train_objective_denominator": train_objective_denominator,
        }
        if backward_loss is not loss:
            output["_backward_loss"] = backward_loss
        if load_balancing_loss is not None:
            # loss is the full objective once the aux term is on; report the two
            # components separately so a rising CE behind a falling total stays visible.
            output["ce_loss"] = ce_loss
            output["aux_loss"] = load_balancing_loss.detach()
        return output, loss

    @torch.no_grad()
    def validation_step(self, data: dict, iteration: int) -> tuple[dict, torch.Tensor]:
        """Required: VLM experiments enable validation by default (pre_exp01x.py:607).
        ImaginaireTrainer.validate() calls this — must not raise NotImplementedError.

        Like ``training_step``, position_ids are computed internally by the model and
        ``attention_mask`` is forwarded (see that method's notes).
        """
        self._prepare_true_packing(data)
        labels = data.pop("labels")
        self._set_moe_token_mask(data.get("attention_mask"))
        logits = self.model(**data)
        loss_kwargs: dict[str, Any] = {}
        if self.config.policy.use_weighted_ce:
            loss_kwargs["cu_seq_lens"] = data.get("cu_seq_lens_q")
        loss_result = self._loss_fn(logits, labels, return_stats=True, **loss_kwargs)
        if not isinstance(loss_result, tuple):
            raise TypeError("validation loss must return (loss, LossStatistics) when return_stats=True")
        loss, stats = loss_result
        output: dict[str, torch.Tensor] = {
            "loss": loss,
            "labels": labels,
            "val_objective_numerator": stats.objective_numerator,
            "val_objective_denominator": stats.objective_denominator,
            "val_token_ce_sum": stats.token_ce_sum,
            "val_n_valid_tokens": stats.valid_token_count,
        }
        return output, loss
