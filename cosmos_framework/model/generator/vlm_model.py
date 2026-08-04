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
from cosmos_framework.configs.base.defaults.parallelism import PRECISION_TO_TORCH_DTYPE
from cosmos_framework.configs.base.defaults.reasoner import validate_sound_understanding_config
from cosmos_framework.configs.base.reasoner.defaults.policy_config import VLMModelConfig
from cosmos_framework.model.generator.hf_model import HFModel
from cosmos_framework.model.generator.parallelize_vlm import parallelize
from cosmos_framework.model.generator.utils.safetensors_loader import load_vlm_model
from cosmos_framework.utils.generator.parallelism import ParallelDims
from cosmos_framework.utils.generator.reasoner.constant import IGNORE_INDEX
from cosmos_framework.utils.generator.reasoner.create_position_ids import get_position_ids

# Model-type dispatch sets. Using hf_config.model_type (stable HF-defined string)
# rather than backbone.model_name avoids the brittleness of substring-matching a local
# filesystem path that VLMModel._init_vlm has already rewritten (see _init_vlm: the
# downloader returns a local cache path, so the configured model name is lost).
#
# ``qwen3_vl_moe`` is listed here as forward-compat — MoE dispatch in every family
# helper below is already wired for the 30B-A3B / 235B-A22B variants. End-to-end
# training still fails earlier at load_vlm_model's MoE precheck
# (safetensors_loader.py _is_moe_vlm / NotImplementedError) because sharded MoE
# weight loading is unimplemented; see spec §2.2. Removing ``qwen3_vl_moe`` here
# would regress the family helpers the moment MoE load support lands.
_QWEN_VL_TYPES = {"qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe"}
# InternVL variants register both "internvl" and "internvl_chat" as model_type
# in the upstream InternVL HF policy registry.
_INTERNVL_TYPES = {"internvl", "internvl_chat"}

_SOUND_UND_ENCODER_STATE_PREFIX = "model.model.sound_und_model.encoder."


def _is_sound_und_encoder_state_dict_key(key: str) -> bool:
    """Match only the standalone VLM's Parakeet encoder state namespace."""
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

    MoE note: ``qwen3_vl_moe`` is accepted here but end-to-end MoE training still
    fails earlier at ``load_vlm_model``'s MoE precheck (see module docstring on
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
            if any(re.search(pat, param_name) for pat in trainable_params):
                p.requires_grad = True
    elif frozen_params is not None:
        for param_name, p in model.named_parameters(remove_duplicate=False):
            if any(re.search(pat, param_name) for pat in frozen_params):
                p.requires_grad = False

    n = sum(p.requires_grad for p in model.parameters())
    if not any([cfg.freeze_vision_encoder, cfg.freeze_mm_projector, cfg.freeze_llm, trainable_params, frozen_params]):
        log.warning("freeze config: no freeze mechanism set — all parameters are trainable (full fine-tune)")
    assert n > 0, "freeze config left 0 trainable parameters — check patterns"
    return n


def _assert_lora_initialized(model: nn.Module) -> None:
    """Fail loudly if adapter init left garbage behind.

    ``init_lora_weights_post_materialization`` runs on FSDP2-sharded params, so
    every write goes through DTensor. A silent no-op there would leave whatever
    ``torch.empty_like`` allocated and produce NaN losses several minutes into
    the run. Check the local shard of the first adapter pair instead:
    ``lora_A`` must be finite and non-zero, ``lora_B`` must be exactly zero.

    Ranks whose shard of a tensor is empty (uneven FSDP split) are skipped.
    """
    from cosmos_framework.utils.generator.lora import LoraInjectedLinear

    def _local(t: torch.Tensor) -> torch.Tensor:
        return t.to_local() if hasattr(t, "to_local") else t

    for name, module in model.named_modules():
        if not isinstance(module, LoraInjectedLinear):
            continue
        a = _local(module.lora_A.weight.detach())
        b = _local(module.lora_B.weight.detach())
        if a.numel() == 0:
            continue
        if not torch.isfinite(a).all():
            raise RuntimeError(f"LoRA init failed: {name}.lora_A contains non-finite values after init.")
        if not a.any():
            raise RuntimeError(
                f"LoRA init failed: {name}.lora_A is all-zero after init. "
                "kaiming_uniform_ did not reach the sharded tensor — the adapter would never learn."
            )
        if b.any():
            raise RuntimeError(f"LoRA init failed: {name}.lora_B is not zero-initialized.")
        log.info(f"LoRA init verified on {name} (lora_A std={a.float().std().item():.4g}, lora_B all-zero)")
        return


def _enforce_lora_only_trainable(model: nn.Module) -> None:
    """Freeze everything except LoRA adapters, in-place.

    ``inject_lora_pre_fsdp`` already does this at injection time, but
    ``_apply_freeze_config`` runs later and can flip base params back to
    trainable. This re-asserts LoRA-only and logs loudly when it had to undo
    something, so a mis-specified freeze config is visible rather than silently
    producing a partial full fine-tune.
    """
    reverted = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    if reverted:
        log.warning(
            f"LoRA: freeze config left {len(reverted)} non-adapter parameter tensor(s) trainable "
            f"(first up to 5: {reverted[:5]}); re-freezing them. Remove `trainable_params` from the "
            "freeze config if you did not intend this."
        )

    lora_numel = 0
    frozen_numel = 0
    for name, param in model.named_parameters():
        is_lora = "lora_" in name
        param.requires_grad_(is_lora)
        if is_lora:
            lora_numel += param.numel()
        else:
            frozen_numel += param.numel()

    assert lora_numel > 0, (
        "LoRA is enabled but 0 adapter parameters are trainable — check "
        "model.config.policy.lora_target_modules against the backbone's module names."
    )
    log.info(
        f"LoRA-only training: {lora_numel:,} trainable adapter params, "
        f"{frozen_numel:,} frozen base params "
        f"({100 * lora_numel / max(1, lora_numel + frozen_numel):.3f}% trainable)"
    )

    # Where the adapters actually landed. A non-zero adapter count is NOT enough
    # to conclude the targets were right: naming differs across model families
    # (Qwen3-VL puts q_proj/k_proj/v_proj/o_proj in the LLM, cosmos3_edge puts
    # those same names in the SigLIP2 vision tower and uses to_q/to_k/to_v/to_out
    # for the LLM). Mistargeting produces a healthy-looking run that trains the
    # wrong subnetwork, so print the placement and let the reader judge.
    placement: dict[str, int] = {}
    for name, _ in model.named_parameters():
        if "lora_" not in name:
            continue
        # Collapse layer indices so 28 layers report as one bucket.
        bucket = re.sub(r"\.\d+\.", ".*.", name.rsplit(".lora_", 1)[0])
        placement[bucket] = placement.get(bucket, 0) + 1
    for bucket, count in sorted(placement.items(), key=lambda kv: -kv[1]):
        log.info(f"LoRA placement: {count:4d} adapter tensors under {bucket}")


class VLMModel(ImaginaireModel):
    """Config-instantiable ImaginaireModel for VLM training.

    Args:
        config:          VLMModelConfig (parallelism, compile, AC, precision,
                         policy, freeze, ema, deterministic).
        checkpoint:      root CheckpointConfig (load_path, load_from_object_store).
    """

    def __init__(self, config: VLMModelConfig, checkpoint):
        super().__init__()
        from cosmos_framework.utils.generator.flash_attn import init_flash_attn_meta

        self.config = config
        validate_sound_understanding_config(config.sound_und_config, sound_und=config.sound_und)
        # Expose model.precision so LowPrecisionCallback can read it (mirrors OmniMoTModel).
        self.precision = getattr(torch, config.precision)
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

        # LoRA-only is authoritative over the freeze config. ``_apply_freeze_config``
        # runs AFTER ``_init_vlm`` (where the adapters were injected and every base
        # param frozen), and its ``trainable_params`` branch unfreezes by regex —
        # which would silently un-freeze base weights and turn a "LoRA run" into a
        # partial full fine-tune. Re-assert here.
        if config.policy.lora_enabled:
            _enforce_lora_only_trainable(self.model.model)

        dp_group = None
        cp_group = None
        if self.parallel_dims is not None:
            if self.parallel_dims.dp_shard_enabled:
                dp_group = self.parallel_dims.dp_shard_mesh.get_group()
            if self.parallel_dims.cp_enabled:
                cp_group = self.parallel_dims.cp_mesh.get_group()

        if config.policy.use_weighted_ce:
            log.info(f"Using weighted CE loss with exponent={config.policy.weighted_ce_exponent}")
            self._loss_fn = partial(
                weighted_cross_entropy_loss,
                exponent=config.policy.weighted_ce_exponent,
                loss_scaling_factor=1.0,
                dp_group=dp_group,
                cp_group=cp_group,
                ignore_index=IGNORE_INDEX,
            )
        else:
            self._loss_fn = partial(
                cross_entropy_loss,
                loss_scaling_factor=1.0,
                dp_group=dp_group,
                cp_group=cp_group,
                ignore_index=IGNORE_INDEX,
            )

    def _init_vlm(self, config: VLMModelConfig, checkpoint) -> None:
        """Initialize VLM without the legacy ModelRegistry (Phase 2+).

        Sequence (ordering is critical — do not reorder):
          a. Download HF weights from S3 to local cache.
          b. Meta-init HFModel (params on meta, buffers on CPU via include_buffers=False;
          c. Build ParallelDims + device mesh.
          d. Apply FSDP2 via parallelize() — meta tensors are NOT auto-materialized.
          e. Explicitly materialize meta tensors; move CPU buffers to CUDA.
          f. Tie output embedding → input embedding if tie_word_embeddings=True.
          g. Load pretrain weights into sharded CUDA tensors.
          h. Apply gradient checkpointing if configured.
        """
        from cosmos_framework.utils.generator.reasoner.pretrained_models_downloader import (
            maybe_download_hf_model_from_s3,
        )

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

        # ── b.2. Inject LoRA adapters (still on meta, still pre-FSDP) ──
        # Ordering is load-bearing: the injector must see UNSHARDED nn.Linear
        # shapes. Injecting after ``parallelize()`` builds ``lora_B`` at the
        # per-rank shard size (e.g. 8192/4=2048) and crashes at forward time.
        # ``lora_A``/``lora_B`` stay uninitialized on meta here; step g.3
        # initializes them once the tensors are real.
        if policy.lora_enabled:
            from cosmos_framework.utils.generator.lora import inject_lora_pre_fsdp

            inject_lora_pre_fsdp(
                hf_model.model,
                lora_rank=policy.lora_rank,
                lora_alpha=policy.lora_alpha,
                lora_target_modules=policy.lora_target_modules,
                lora_exclude_path_regex=policy.lora_exclude_path_regex or None,
            )

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

        _set_sound_und_encoder_dtype_for_fsdp(
            hf_model,
            precision=config.precision,
            fsdp_enabled=parallel_dims.dp_shard_enabled,
        )

        # Replicate-only (DDP) is not implemented in Phase 2's parallelize().
        # Raise early rather than running with no gradient synchronization and
        # silently producing wrong training results.
        if parallel_dims.dp_replicate_enabled and not parallel_dims.dp_shard_enabled:
            raise NotImplementedError(
                "VLMModel Phase 2 does not support replicate-only DDP "
                "(dp_replicate > 1, dp_shard == 1). "
                "Use dp_shard > 1 for FSDP2. DDP support is planned for Phase 3."
            )

        # ── d. Apply FSDP2 (+ optional torch.compile of the repeated blocks) ──
        # config.compile is threaded through so model.config.compile.enabled=True
        # actually compiles each block in place (was previously a dead config on
        # the VLM path — only the MoT path consumed it). See parallelize_vlm.
        if torch.distributed.is_initialized():
            parallelize(
                hf_model,
                parallel_dims,
                config.parallelism,
                config.precision,
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
            # tensors are populated by the authoritative artifact below; only
            # the projector needs a fresh initialization here. A VLM/DCP
            # checkpoint may overwrite it later.
            hf_model.model.sound_und_model.projector.reset_parameters()

        # ── f. Tie embeddings (replaces the legacy post_to_empty_hook) ──
        hf_model.tie_embeddings()

        # ── g. Load pretrain weights ──
        # LoRA adapters exist on the model but never in a pretrained checkpoint.
        # ``load_vlm_model``'s Phase-6 completeness check raises on any model key
        # the checkpoint lacks, so the adapter keys must be tolerated explicitly.
        # Patterns are applied with ``re.fullmatch`` against the resolved model
        # key, hence the leading ``.*``.
        lora_skip_patterns = [r".*\.lora_[AB]\.weight"] if policy.lora_enabled else []

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
                extra_skip_patterns=lora_skip_patterns or None,
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
                    extra_skip_patterns=overlay_skip_patterns + lora_skip_patterns,
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

        # ── g.3. Initialize LoRA adapters ──
        # Must run AFTER step e (meta -> real CUDA storage) and after every
        # load_weights (which skips these keys, leaving whatever ``empty_like``
        # allocated). ``lora_A ~ kaiming_uniform_``, ``lora_B = 0`` — so the
        # adapter contributes exactly zero on the first forward and the run starts
        # from the pretrained model's loss.
        #
        # Ordering vs step h: the Parakeet load below writes only into
        # ``sound_und_model.encoder`` and its checkpoint has no ``lora_*`` keys,
        # so it cannot clobber what this initializes.
        if policy.lora_enabled:
            from cosmos_framework.utils.generator.lora import init_lora_weights_post_materialization

            init_lora_weights_post_materialization(hf_model.model)
            _assert_lora_initialized(hf_model.model)

        # ── h. Load the immutable standalone Parakeet artifact ──
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
                f"VLMModel: loaded {len(loaded_audio_keys)} Parakeet encoder tensors from "
                f"{audio_config.encoder_checkpoint_path}"
            )

        # ── i. Gradient checkpointing ──
        # HF backbone supports only binary on/off via gradient_checkpointing_enable,
        # so VLMActivationCheckpointingConfig.mode is restricted to {"full", "none"}.
        if config.activation_checkpointing.mode == "full":
            hf_model.apply_gradient_checkpointing()

        self.model = hf_model
        self.parallel_dims = parallel_dims
        self.model_name_or_path = local_path
        self.hf_config = hf_model.hf_config

    def on_train_start(self, memory_format) -> None:
        """Called by trainer after model.to("cuda"). No device move needed here."""

    def on_after_backward(self, iteration: int = 0) -> None:
        """No-op — FSDP handles gradient synchronization internally."""

    def state_dict(
        self,
        destination: dict[str, Any] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, Any]:
        """Optionally omit the immutable artifact-backed Parakeet encoder."""
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

    def training_step(self, data: dict, iteration: int) -> tuple[dict, torch.Tensor]:
        """position_ids → forward → CE loss."""
        position_ids = get_position_ids(
            self.hf_config,
            input_ids=data["input_ids"],
            image_grid_thw=data.get("image_grid_thw"),
            video_grid_thw=data.get("video_grid_thw"),
            attention_mask=data.get("attention_mask"),
        )
        if position_ids is not None:
            data["position_ids"] = position_ids

        labels = data.pop("labels")
        data.pop("attention_mask", None)
        logits = self.model(**data)
        loss = self._loss_fn(logits, labels)

        # loss_avg: DP-averaged loss for logging (matches cosmos-rl ReduceOp.AVG).
        # Does not affect the backward scalar. Pick the same 1-D sub-mesh the
        # legacy single-mesh ``ParallelDims.dp_mesh`` returned — dp_shard if
        # sharding is on, else dp_replicate — so the reduction group is
        # byte-identical to pre-merge behavior.
        loss_avg = loss.detach().clone()
        pd = getattr(self, "parallel_dims", None)
        dp_mesh = pd.dp_mesh if pd is not None else None
        if torch.distributed.is_initialized() and dp_mesh is not None:
            sub_dim = "dp_shard" if pd.dp_shard_enabled else "dp_replicate"
            torch.distributed.all_reduce(
                loss_avg, op=torch.distributed.ReduceOp.AVG, group=dp_mesh[sub_dim].get_group()
            )
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            log.info(f"train/loss_avg: {loss_avg.item():.5f} (iteration {iteration})")

        return {"loss": loss, "loss_avg": loss_avg, "labels": labels}, loss

    @torch.no_grad()
    def validation_step(self, data: dict, iteration: int) -> tuple[dict, torch.Tensor]:
        """Required: VLM experiments enable validation by default (pre_exp01x.py:607).
        ImaginaireTrainer.validate() calls this — must not raise NotImplementedError."""
        position_ids = get_position_ids(
            self.hf_config,
            input_ids=data["input_ids"],
            image_grid_thw=data.get("image_grid_thw"),
            video_grid_thw=data.get("video_grid_thw"),
            attention_mask=data.get("attention_mask"),
        )
        if position_ids is not None:
            data["position_ids"] = position_ids

        labels = data.pop("labels")
        data.pop("attention_mask", None)
        logits = self.model(**data)
        loss = self._loss_fn(logits, labels)
        return {"loss": loss, "labels": labels}, loss
