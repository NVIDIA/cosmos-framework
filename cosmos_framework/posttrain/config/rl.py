# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Typed posttrain RL config schema (DESIGN: config system).

The **authoring + agent-facing schema** for the RL half of a posttrain experiment.
Dependency-light (no torch / no model imports) so it can be read, type-checked, and
round-tripped through OmegaConf on its own. General config mapping/dispatch lives in
`config/build.py`; concrete model construction lives in `model/*.py`.

Layering (the three-stakeholder synthesis):
  - The model/data half is the inherited LazyConfig tree at top-level `config.model`
    (researchers `defaults: [<i4 experiment>]`); reflected via `instantiate`.
  - Model backend selection and shared load specs are top-level too (`config.model_backend`,
    `config.model_hf`, and `config.model_qwen3_vl`; schemas in `config/model.py`), NOT an RL concern.
    `DPOConfig` is purely the RL objective + optimizer + ref + data.
  - Orchestration (`config.grid`) is also top-level. So `DPOConfig` holds neither model
    nor grid — only what the DPO objective needs.
  - The whole experiment dumps to one resolved YAML (upstream `LazyConfig.save_yaml`), reloadable.

`rl` owns the training PROCESS; `model_hf` owns model IDENTITY. The line above ("model backend
selection is not an RL concern") is about WHICH model to load — `name_or_path`, `auto_class`,
`attn_impl`, `grad_ckpt`. It is not about the PRECISION that model is trained at, which belongs
beside the optimizer for the reason `HFRunConfig` documents: the optimizer's moment dtype is
decided by the storage dtype, so splitting the two across config objects means one knob silently
determines another object's behaviour. This mirrors cosmos-rl, whose `[train]` section holds
`optm_*` AND `master_dtype`/`param_dtype`/`fsdp_reduce_dtype`/`transfer_dtype` together, while
`[policy]` (its model config) carries no dtype at all.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from numbers import Real
from pathlib import Path
from typing import Any, Literal

from cosmos_framework.posttrain.algorithm.reward import RULE_REWARDS
from cosmos_framework.posttrain.config.placement import normalize_gpu_placement
from cosmos_framework.posttrain.schema.checkpoint import (
    EMERGENCY_CHECKPOINT_PLANS,
    FULL_CHECKPOINT_COMPONENTS,
    CheckpointComponentPlan,
)


def _normalize_checkpoint_components(
    components: Sequence[str] | CheckpointComponentPlan | None,
) -> list[str]:
    """Validate and canonicalize one config-level logical component selection."""
    plan = CheckpointComponentPlan.from_value(components)
    return list(plan.names)


# ---------------------------------------------------------------------------
# Optimizer — owned by the RL config (freeze backbone, train the gen subset)
# ---------------------------------------------------------------------------

OPTIMIZER_IMPLS: tuple[str, ...] = ("fused", "foreach", "for-loop")
"""torch kernel choices for `OptimizerConfig.impl` — cosmos-rl's `optm_impl` values verbatim."""

OPTIMIZER_NAMES: tuple[str, ...] = ("adamw", "fusedadam", "muon")
"""Optimizer CLASSES `OptimizerConfig.name` selects (its docstring describes each); `build_optimizer` dispatches on them."""

FIXED_NORMALISER_OPTIMIZER_EPS_CEILING: float = 1e-7
"""Largest `OptimizerConfig.eps` that `GRPOConfig` accepts silently under
`loss_type="seq-mean-token-sum-fixed"`. That reduction divides every sequence's token-summed loss by
the fixed `rollout.max_tokens` instead of the tokens produced, so per-parameter gradients land tens
of times below their token-mean scale. The Adam denominator can become
eps-dominated for most parameters. The GRPOConfig validator warns above this ceiling (and refuses the
pairing under `GRPOConfig.recipe_guard_severity="error"`); 1e-8 (torch's AdamW default) is the
recommendation, together with an lr re-tune. `the Dr-GRPO recipe layer` is the recipe layer
that selects the fixed reduction together with that eps."""

RECIPE_GUARD_WARN: str = "warn"
RECIPE_GUARD_ERROR: str = "error"
RECIPE_GUARD_SEVERITIES: tuple[str, str] = (RECIPE_GUARD_WARN, RECIPE_GUARD_ERROR)
"""Values of `GRPOConfig.recipe_guard_severity` — how the recipe-hygiene hazards report at construction.
`warn` (the default) keeps each hazard the `UserWarning` it always was; `error` collects every hazard
found and refuses the config in ONE `ValueError` that lists them all."""

RECIPE_GUARD_REFUSAL_SENTENCE: str = (
    "Refused because rl.recipe_guard_severity=error; set it to warn to run this pairing with a warning instead."
)
"""The one sentence that closes a `recipe_guard_severity="error"` refusal, after the hazard texts."""


def _fixed_normaliser_eps_message(eps: float) -> str:
    """The one description of why `seq-mean-token-sum-fixed` and an eps above the ceiling do not mix.

    Shared by the default warning and the `recipe_guard_severity="error"` refusal, so the two paths
    cannot drift apart in what they tell the operator. Mechanism only: eps is honored and the
    objective stays unbiased; what breaks is the meaning of lr.
    """
    return (
        f"loss_type='seq-mean-token-sum-fixed' with rl.optimizer.eps={eps:g}: this "
        "reduction divides each sequence's token-summed loss by the fixed rollout.max_tokens rather "
        "than by the tokens produced, so per-parameter gradients are tens of times smaller than under "
        "token-mean and may fall below this eps. The Adam denominator can become eps-dominated and the update can become an SGD-like "
        "step whose size follows lr/eps, not lr, and which does not carry across max_tokens or eps. "
        "The objective stays unbiased; the step size is not the one lr names. Set rl.optimizer.eps "
        f"<= {FIXED_NORMALISER_OPTIMIZER_EPS_CEILING:g} — 1e-8, torch's AdamW default, is the "
        "recommendation — and re-tune lr: the same lr takes far larger steps once eps stops dominating."
    )


def _literal_quota_below_half_message(*, literal: int, groups: int, batch_per_replica: int, group_size: int) -> str:
    """The one description of a literal DAPO quota that covers less than half of `groups_per_step`.

    Shared by the default warning and the `recipe_guard_severity="error"` refusal. The count is
    honored as written; the message names `groups_per_step`, the count and the fraction key so the
    operator can move to the layout-proof spelling.
    """
    return (
        f"min_informative_groups={literal} is below half of groups_per_step={groups} "
        f"(batch_per_replica={batch_per_replica} // group_size={group_size}). The count is a "
        f"GLOBAL quota split across the data cells, so only {literal}/{groups} of each "
        "step's groups are guaranteed informative and the rest is whatever the base draw produced. If the "
        "quota was meant as a share of the batch, set rl.min_informative_groups_frac instead (1.0 = full "
        f"DAPO; this count is {literal / groups:.3g} here) and leave "
        "min_informative_groups at 0."
    )


def _recipe_guard_refusal(hazards: list[str]) -> str:
    """The one `ValueError` text under `recipe_guard_severity="error"`.

    Every hazard found, in the order the warnings always had, joined by a blank line, then
    `RECIPE_GUARD_REFUSAL_SENTENCE`. A single hazard's text therefore STARTS the error unchanged, so
    the warning and the refusal describe the same thing in the same words.
    """
    return "\n\n".join([*hazards, RECIPE_GUARD_REFUSAL_SENTENCE])


def _normalize_finite_real(value: object, *, field_name: str) -> float:
    """Return one canonical built-in float for a finite configuration scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite real, got {value!r}")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{field_name} must be a finite real, got {value!r}") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite real, got {value!r}")
    return normalized


_PROMPT_DATASET_BULK_METADATA_COLUMNS: frozenset[str] = frozenset(
    {"image", "images", "video", "videos", "media", "video_fps", "fps"}
)


def normalize_prompt_metadata_columns(
    metadata_columns: object,
    *,
    configured_columns: Sequence[object],
) -> tuple[str, ...]:
    """Validate ordered prompt metadata projection names without mutating the input sequence."""
    if isinstance(metadata_columns, (str, bytes)) or not isinstance(metadata_columns, Sequence):
        raise TypeError(f"metadata_columns must be a structured sequence, got {type(metadata_columns).__name__}")
    normalized = tuple(metadata_columns)
    if any(type(column) is not str or not column for column in normalized):
        raise ValueError("metadata_columns entries must be non-empty exact strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError("metadata_columns entries must be unique")
    configured = {column for column in configured_columns if isinstance(column, str) and column}
    collisions = [
        column for column in normalized if column in configured or column in _PROMPT_DATASET_BULK_METADATA_COLUMNS
    ]
    if collisions:
        raise ValueError(
            "metadata_columns must exclude configured projection columns and reserved media/FPS aliases, "
            f"got {collisions!r}"
        )
    return normalized


def normalize_video_fps(value: object) -> float:
    """Return one canonical built-in float for video prompt timing."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("video_fps must be a finite positive number")
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValueError("video_fps must be a finite positive number") from error
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise ValueError("video_fps must be a finite positive number")
    return normalized


@dataclass
class OptimizerConfig:
    """RL owns the optimizer: freeze everything except params whose name contains one of
    `train_keys`, then an optimizer over the trainable subset. `train_keys=[]` ⇒ train all
    `requires_grad` params — the model-agnostic DEFAULT. A freeze subset is a model+recipe choice
    (its param names are model internals), so the experiment sets it — e.g. the cosmos3 DPO
    experiments pass the OmniMoT gen-pathway keys; it is never baked into this generic schema.

    `name` picks the implementation:
    - "adamw" — plain `torch.optim.AdamW`. Container-free (CPU tests, toy models), BUT its
      moments live in the PARAM dtype: with a bf16 net at posttrain lr (~5e-6) the relative
      update (lr/|w| ≈ 2.5e-4) is below bf16 resolution (2^-8 ≈ 3.9e-3) and mostly rounds
      away — root-caused against the wallace 8B DPO reference (see utils/fused_adam.py).
      Fine for fp32 nets (toys/tests); do NOT use for real bf16 posttraining.
    - "fusedadam" — apex-fused AdamW with fp32 master weights + fp32 moments
      (`utils/fused_adam.py`). Requires apex + GPU (training containers ship it).
    - "muon" — the shared `MuonWithAuxAdamW`: Muon for hidden `nn.Linear` weights and explicitly
      declared stacked MoE expert matrices, with its auxiliary fused AdamW route for embeddings,
      output heads, routers, norms, biases, convolutions and other parameters. Fused expert gate/up
      tensors are split before Newton-Schulz and expert-axis FSDP shards are processed locally.
      `muon_rms_match` initializes the Muon LR scale but does not make an Adam-tuned LR equivalent;
      re-tune LR for the selected objective. `betas`/`eps` apply to the Adam route only. Requires
      FP32 parameter storage (`master_dtype=float32`) with BF16 compute supplied by FSDP mixed
      precision, plus Transformer Engine + GPU.

    `name` vs `impl` — TWO AXES, and conflating them is a mistake this port had to undo.
    `name` picks the optimizer CLASS; `impl` picks the torch KERNEL for the `adamw` class
    (`fused`/`foreach`/`for-loop`). cosmos-rl spells them `optm_name` and `optm_impl`, and its
    `optm_impl="fused"` means "pass `fused=True` to `torch.optim.AdamW`" — NOT "use apex". The
    GSM8K recipe originally mapped that TOML row onto `name="fusedadam"`, which silently swapped
    the optimizer class against the reference: cosmos-rl's LLM GRPO path never touches apex (its
    registry is `{Adam, AdamW, Adam8bit, AdamW8bit}`; apex lives only on the WFM/diffusion path).

    NOTE `name="fusedadam"` exists to rebuild an fp32 master copy INSIDE the optimizer, which is
    only necessary when parameters are stored in bf16. With `rl.master_dtype="float32"` (see
    `HFRunConfig`) the parameter is already fp32 and plain `adamw` gets fp32 moments for free —
    matching cosmos-rl exactly and dropping the apex dependency."""

    name: str = "adamw"
    # torch kernel selection for `name="adamw"`, mirroring cosmos-rl's `optm_impl`.
    # "" (DEFAULT) ⇒ pass NEITHER `fused` nor `foreach`, letting torch choose — byte-identical to
    # this field not existing, which is what keeps it a no-op for every current recipe.
    # Ignored by `name="fusedadam"` (apex has no such knob); set both and it raises, rather than
    # accepting a kernel request the chosen class cannot honor.
    impl: str = ""
    lr: float = 1e-6
    weight_decay: float = 0.0
    # list (not tuple) so the schema is OmegaConf-structured-safe; build_optimizer tuples it.
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    # Adam's denominator floor, update = lr * m / (sqrt(v) + eps); torch's default. This is a
    # gradient-SCALE knob, not a stability nicety: once sqrt(v) for a parameter sits below eps, that
    # parameter's update degenerates to (lr/eps) * m and lr stops naming the step size. The
    # cosmos-rl GRPO TOMLs carry 1e-6 (100x this default), sized for token-mean gradients;
    # GRPOConfig warns when that pairs with loss_type="seq-mean-token-sum-fixed", whose fixed
    # divisor pushes the gradients below it (FIXED_NORMALISER_OPTIMIZER_EPS_CEILING), and refuses
    # the pairing under recipe_guard_severity="error". The recipe layer that selects the fixed
    # reduction together with this eps is `the Dr-GRPO recipe layer`.
    eps: float = 1e-8
    # [] ⇒ train all requires_grad params (model-agnostic). The freeze subset (its names are model
    # internals) is a recipe choice set by the experiment, not defaulted here.
    train_keys: list[str] = field(default_factory=list)
    # Gradient hygiene: clip the global grad norm before each optimizer step (0 = off).
    # force_finite rejects the whole accumulation window if any rank has NaN/Inf grads;
    # disabling it is an explicit legacy opt-out. Non-finite loss is always rejected.
    grad_clip_norm: float = 0.0
    force_finite_grads: bool = True
    # Linear LR warmup step count (0 = no warmup), then constant. cosmos-rl `optm_warmup_steps`. Validated
    # `>= 0`: a negative value used to be accepted and silently meant "no warmup".
    warmup_steps: int = 0
    # Weight-decay grouping: "" = one group, every trainable parameter decays (the pre-existing behavior,
    # and the Cosmos post-training recipe's: `cosmos3/utils/optimizer.py::_build_params_with_metadata`
    # lines 326-331 decay all 882 tensors of the Qwen3-VL-MoE backbone, `disable_weight_decay_for_1d_params`
    # False and no skip pattern); "hf" = the [decay, no-decay] split of `transformers.Trainer.create_optimizer`
    # (`get_decay_parameter_names`: LayerNorm modules, biases and `*norm*` names undecayed; the embedding,
    # the head and the MoE router gates DECAY), needed to reproduce an HF-trained reference; "qwen_moe" =
    # the SFT seat's own rule: no decay on every
    # 1-D parameter (RMSNorm/LayerNorm scales and biases, as HF) AND on the token embedding (`embed_tokens`),
    # the output head (`lm_head`) and the MoE router gates (`<layer>.mlp.gate.weight`), which neither
    # reference exempts; every matrix (attention, experts, vision) decays. The native recipe's docstring
    # states what the three extra exemptions protect and cost.
    decay_groups: str = ""
    # Per-group learning-rate multipliers: {parameter-name substring: factor}. A parameter takes the
    # factor of the FIRST key (dict order) that occurs in its name, 1.0 when none does; the groups the
    # optimizer sees are the (decay, factor) classes, the 1.0x class first. The Cosmos post-training
    # recipe of the Qwen3-VL-MoE backbone trains its vision tower at 0.1x (`{"model.visual": 0.1}`,
    # `cosmos3/configs/base/reasoner/defaults/optimizer.py`). Empty = one learning rate (pre-existing).
    lr_multipliers: dict[str, float] = field(default_factory=dict)
    # `name="muon"` only (ignored by the other classes). Momentum β of the Muon buffer `m ← β·m + g`
    # (Nesterov direction `g + β·m`), the number of Newton-Schulz iterations, and the RMS the
    # orthogonalised update is scaled to (`update *= muon_rms_match · sqrt(max(rows, cols))`, the
    # Moonlight RMS rule). The inherited lr is a starting heuristic and must be re-tuned.
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    muon_rms_match: float = 0.2

    def __post_init__(self) -> None:
        if self.name not in OPTIMIZER_NAMES:
            raise ValueError(f"OptimizerConfig.name={self.name!r} must be one of {OPTIMIZER_NAMES}")
        if self.name == "muon":
            if (
                isinstance(self.muon_momentum, bool)
                or not isinstance(self.muon_momentum, Real)
                or not math.isfinite(float(self.muon_momentum))
                or not 0.0 <= float(self.muon_momentum) < 1.0
            ):
                raise ValueError(f"OptimizerConfig.muon_momentum={self.muon_momentum!r} must be in [0, 1)")
            if (
                isinstance(self.muon_ns_steps, bool)
                or not isinstance(self.muon_ns_steps, int)
                or self.muon_ns_steps < 1
            ):
                raise ValueError(f"OptimizerConfig.muon_ns_steps={self.muon_ns_steps!r} must be a positive integer")
            if (
                isinstance(self.muon_rms_match, bool)
                or not isinstance(self.muon_rms_match, Real)
                or not math.isfinite(float(self.muon_rms_match))
                or self.muon_rms_match <= 0.0
            ):
                raise ValueError(
                    f"OptimizerConfig.muon_rms_match={self.muon_rms_match!r} must be a finite positive real"
                )
        if self.decay_groups not in ("", "hf", "qwen_moe"):
            raise ValueError(f"OptimizerConfig.decay_groups={self.decay_groups!r} must be '', 'hf' or 'qwen_moe'")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError(f"OptimizerConfig.warmup_steps={self.warmup_steps!r} must be an integer >= 0")
        if not isinstance(self.lr_multipliers, Mapping):
            raise TypeError(
                f"OptimizerConfig.lr_multipliers must be a mapping, got {type(self.lr_multipliers).__name__}"
            )
        # A DictConfig (imaginaire load_config) becomes a plain dict so the optimizer builder and the identity see one type.
        self.lr_multipliers = {pattern: factor for pattern, factor in dict(self.lr_multipliers).items()}
        for pattern, factor in self.lr_multipliers.items():
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(
                    f"OptimizerConfig.lr_multipliers keys must be non-empty name substrings, got {pattern!r}"
                )
            if (
                isinstance(factor, bool)
                or not isinstance(factor, Real)
                or not math.isfinite(float(factor))
                or float(factor) <= 0.0
            ):
                raise ValueError(
                    f"OptimizerConfig.lr_multipliers[{pattern!r}]={factor!r} must be a finite positive real"
                )
        if self.impl and self.impl not in OPTIMIZER_IMPLS:
            raise ValueError(f"OptimizerConfig.impl={self.impl!r} must be one of {OPTIMIZER_IMPLS} (or '')")
        if self.impl and self.name != "adamw":
            # A kernel request the chosen class cannot honor is worse than no knob: apex FusedAdam
            # has neither a `fused` nor a `foreach` argument, so silently dropping `impl` would
            # leave the run reporting a kernel it never used.
            raise ValueError(
                f"OptimizerConfig.impl={self.impl!r} applies only to name='adamw' (torch kernel "
                f"selection), but name={self.name!r}. apex FusedAdam and Muon have no such knob."
            )


# ---------------------------------------------------------------------------
# Reference model (frozen anchor) + preference data
# ---------------------------------------------------------------------------

CURSOR_RESUME_AUTO: str = "auto"
CURSOR_RESUME_STRICT: str = "strict"
CURSOR_RESUME_OFF: str = "off"
CURSOR_RESUME_MODES: tuple[str, str, str] = (
    CURSOR_RESUME_AUTO,
    CURSOR_RESUME_STRICT,
    CURSOR_RESUME_OFF,
)


def _normalize_factory_kwargs(factory_kwargs: object) -> dict:
    """Return recursively materialized factory kwargs from plain or OmegaConf mappings."""
    # IMPORTED HERE, NOT AT MODULE SCOPE, and this is a boundary constraint rather than a style
    # preference. `roles/env/__init__.py` re-exports `preference`, which imports this
    # module, so anything imported at the top of this file is on the SIM-SIDE import path -- and
    # the behavior1k sim image deliberately carries neither omegaconf nor pyyaml. A module-level
    # `from omegaconf import OmegaConf` therefore breaks `import cosmos_framework.posttrain.
    # roles.env` in the sim venv with `ImportError: yaml`, which is what
    # tests/test_import_discipline.py exists to catch. OmegaConf is used by this one function
    # and nowhere else in the module, so the deferral costs a dict lookup per call.
    from omegaconf import OmegaConf

    if not isinstance(factory_kwargs, Mapping):
        raise TypeError(f"factory_kwargs must be a mapping, got {type(factory_kwargs).__name__}")
    config = factory_kwargs if OmegaConf.is_config(factory_kwargs) else OmegaConf.create(dict(factory_kwargs))
    normalized = OmegaConf.to_container(config, resolve=True)
    if not isinstance(normalized, dict):
        raise TypeError(f"factory_kwargs must resolve to a dict, got {type(normalized).__name__}")
    return normalized


# The symmetric log-ratio clamp the clipped surrogate and its k3 KL term apply before `exp`, in nats.
# Mirrors `algorithm/primitives/policy_gradient.py::TOKEN_LOG_RATIO_CLAMP` (pinned equal by
# `tests/test_algorithm_grpo_kl_reference.py`) rather than importing it: that module imports torch,
# and this config module is loaded where torch is not (login-node config validation).
KL_LOG_RATIO_CLAMP_NATS = 20.0


@dataclass
class RefConfig:
    """Frozen / moving reference: the contract DPO, NFT and GRPO share. `update_every=0` ⇒ frozen
    anchor (the validator-enforced DPO default); >0 ⇒ moving reference refreshed every N iters.

    Nothing here says WHERE the reference weights come from: DPO anchors through `rl.init_from`
    (loaded into policy and reference alike), NFT at the policy's initialization, GRPO at the
    policy's own starting checkpoint. The one algorithm with a separate anchor source, GRPO's warm
    restart, carries that knob on its own subclass (`GRPORefConfig.model_dir`), so a setting most
    consumers cannot use never appears at `rl.ref.*` for DPO or NFT — a dotted override of it there
    is refused by the structured schema as an unknown key.
    """

    build_ref: bool = True
    update_every: int = 0


@dataclass
class GRPORefConfig(RefConfig):
    """GRPO's frozen KL reference: the shared contract plus the warm-restart anchor `model_dir`.

    `model_dir`: build the frozen reference from THIS checkpoint directory instead of the policy's
    own model directory. A run restarted from a model-only save starts its policy from that save,
    but its KL anchor must stay the ORIGINAL base — name the base here. `None` (default) keeps the
    reference identical to the policy's starting weights. Only meaningful with `build_ref=true`;
    consumed by `entrypoints/grpo.py::build_trainer`, which also checks that the directory exists —
    on every rank, right before it is loaded — and binds its content identity into the objective's
    exact-resume identity, so this dataclass never touches the filesystem and a saved config reloads
    on a host without the checkpoint mount. `GRPOConfig.ref` is typed as this class; a plain
    `RefConfig` handed to it is adopted with `model_dir=None`.
    """

    model_dir: str | None = None

    def __post_init__(self) -> None:
        if self.model_dir is None:
            return
        if not isinstance(self.model_dir, str) or not self.model_dir.strip():
            raise ValueError(f"ref.model_dir must be None or a non-empty path string, got {self.model_dir!r}")
        if not self.build_ref:
            raise ValueError(
                f"ref.model_dir={self.model_dir!r} is only meaningful with ref.build_ref=true: it names the "
                "checkpoint the frozen reference is built from, and no reference is built"
            )


@dataclass
class OldPolicyConfig:
    """Periodically refreshed behavior snapshot used by DiffusionNFT.

    The moving old policy constructs the NFT update direction. It is distinct
    from :class:`RefConfig`, whose frozen network only regularizes policy drift
    when ``kl_beta`` is non-zero.
    """

    build_old: bool = True
    update_every: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.build_old, bool):
            raise TypeError(f"old-policy build_old must be a bool, got {self.build_old!r}")
        if not isinstance(self.update_every, int) or isinstance(self.update_every, bool):
            raise TypeError(f"old-policy update_every must be an integer, got {self.update_every!r}")
        if self.update_every < 0:
            raise ValueError(f"old-policy update_every must be >= 0, got {self.update_every}")


@dataclass
class PreferenceEnvConfig:
    """Offline preference source (JSONL pairs) → `DPOPreferenceEnv`."""

    jsonl_path: str = ""
    # Importable ``factory(**factory_kwargs)`` returning canonical map-style
    # ``(prompt, winner, loser)`` items. The Env still owns traversal and sharding.
    factory: str = ""
    factory_kwargs: dict = field(default_factory=dict)
    # Source-cursor behavior at checkpoint boundaries:
    #   auto   — exact when compatible; restart the source for missing/unsupported
    #            state or a changed world size, while preserving model resume.
    #   strict — require an exact source snapshot and compatible restore.
    #   off    — omit and ignore source state explicitly.
    cursor_resume: str = CURSOR_RESUME_AUTO
    # > 0 ⇒ use SyntheticPreferenceDataset with that many in-memory pairs (the
    # zero-dependency quick-start path; no files, no cross-rank coordination —
    # deterministic per-pair seeds give every rank identical data). Mutually
    # exclusive with jsonl_path/tar_path; build_env fails loud if neither is set.
    synthetic_pairs: int = 0
    media: str = "video"  # "video" | "image"
    resolution: str = "256"
    num_frames: int = 121  # video recipe; image path is T=1
    cfg_dropout_rate: float = 0.0
    shuffle: bool = True
    # Tar-backed variant (HPDv3 offset jsonl + one big images.tar, cosmos-rl tar_dpo
    # parity): setting tar_path switches the env to TarDPOPreferenceDataset.
    tar_path: str = ""
    caption_key: str = "prompt"
    min_score_gap: float = 0.0
    min_wm_gap: float = 0.0
    min_align_gap: float = 0.0

    def __post_init__(self) -> None:
        if self.cursor_resume not in CURSOR_RESUME_MODES:
            raise ValueError(f"cursor_resume must be 'auto', 'strict', or 'off', got {self.cursor_resume!r}")
        if self.synthetic_pairs < 0:
            raise ValueError(f"synthetic_pairs must be >= 0, got {self.synthetic_pairs}")
        self.factory_kwargs = _normalize_factory_kwargs(self.factory_kwargs)
        if self.factory_kwargs and not self.factory:
            raise ValueError("rl.data.factory_kwargs requires rl.data.factory")
        built_in_source = bool(self.jsonl_path or self.tar_path or self.synthetic_pairs > 0)
        if self.factory and built_in_source:
            raise ValueError("custom preference factory is mutually exclusive with jsonl_path/tar_path/synthetic_pairs")
        if self.synthetic_pairs > 0 and (self.jsonl_path or self.tar_path):
            raise ValueError("synthetic_pairs is mutually exclusive with jsonl_path/tar_path")
        if self.tar_path and not self.jsonl_path:
            raise ValueError("tar_path requires jsonl_path containing the pair offsets")


# ---------------------------------------------------------------------------
# vLLM rollout engine (`config.rollout`) — cosmos-rl RolloutConfig-aligned
# ---------------------------------------------------------------------------


# vLLM's multimodal modality vocabulary (`vllm.multimodal` input keys). `rollout.limit_mm_per_prompt` keys are
# refused at config time when outside it (reviewer thread on rl.py:549: an unknown or mistyped key such as
# "images" or " image" is otherwise converted to a generic dummy option and silently ignored by the model's
# `allowed_mm_limits`, which iterates only its supported modalities). The model-specific supported set is
# checked a second time against the BUILT engine in `roles/rollout/vllm.py::VLLMRollout`.
VLLM_MODALITIES: frozenset[str] = frozenset({"image", "video", "audio"})
VLLM_CUSTOM_ALL_REDUCE_WORLD_SIZES: frozenset[int] = frozenset({2, 4, 6, 8})


@dataclass
class RolloutConfig:
    """vLLM rollout engine spec (`config.rollout`), consumed by `build.vllm_engine_kwargs` /
    `build.build_vllm_rollout`; PPO process globals are applied by
    `build.configure_ppo_vllm_environment`. Field names follow cosmos-rl's ``RolloutConfig`` where
    they map 1:1 (``gpu_memory_utilization``, ``enforce_eager``, ``seed``; cosmos-rl's
    ``parallelism.tp_size`` flattens to ``tensor_parallel_size``; its ``max_response_length`` is
    our sampling ``max_tokens``); engine-only knobs keep vLLM's own names (``max_model_len``,
    ``dtype``). ``n``/``logprobs`` are deliberately NOT here — they are framework-owned
    (n = GRPO group size per generate call, logprobs = π_old) and VLLMRollout rejects them."""

    name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str | None = None
    tokenizer_revision: str | None = None
    processor_revision: str | None = None
    trust_remote_code: bool = False
    tensor_parallel_size: int = 1  # >1 ⇒ external_launcher SPMD over the per-replica rendezvous group
    # vLLM expert parallelism is opt-in because dense models do not need it. The exact released
    # Qwen3-VL-MoE presets use independent single-rank rollout replicas and leave it disabled to
    # avoid the larger sampled-action score drift measured with tensor/expert-parallel rollout execution.
    enable_expert_parallel: bool = False
    # Colocated default: the engine SHARES each GPU with the FSDP trainer (cosmos-rl's 0.8 default
    # assumes a dedicated rollout GPU).
    gpu_memory_utilization: float = 0.3
    max_model_len: int = 1024
    dtype: str = "bfloat16"
    # vLLM weight loader, auto|dummy. dummy skips the checkpoint read, and the rollout then refuses to
    # sample until a weight publication commits (docs/howto/qwen-scaleout.md, "Startup weight I/O").
    load_format: str = "auto"
    enforce_eager: bool = True
    # vLLM sleep level for the colocated hand-off cycle. Level 1 keeps a host copy of the
    # engine weights; level 2 drops them and is appropriate only when the next full weight sync
    # rewrites every engine parameter. GRPO's colocated loop does that before every generation.
    sleep_level: int = 1
    # vLLM kernel selection; ``auto`` is its default. Some valid MoE shapes require an explicit
    # general backend when an architecture-specific fast kernel has stricter tile divisibility.
    moe_backend: str = "auto"
    # Request vLLM's decode batch-invariant kernels. PPO applies this through the worker environment
    # before importing vLLM; it is intentionally not an ``LLM`` constructor keyword.
    batch_invariant: bool = False
    # Optional vLLM decoder attention backend. None preserves platform auto-selection; this is
    # independent from the multimodal encoder backend below.
    attention_backend: str | None = None
    # Optional vLLM multimodal encoder attention backend. The pinned CUDA-13
    # Some CUDA runtimes use TORCH_SDPA to avoid a CuTe/CUTLASS ABI mismatch.
    mm_encoder_attn_backend: str | None = None
    # Per-REQUEST multimodal item budget, forwarded as vLLM's ``limit_mm_per_prompt`` (e.g. ``{"image": 40}``)
    # only when set, so the engine construction identity of every other run is unchanged. VERIFIED default
    # semantics of the pinned runtime (probe in the pinned container, vLLM 0.24.0; the 0.23.0 wheel pinned in
    # pyproject.toml carries the same code): ``BaseDummyOptions.count`` defaults to 999 and
    # ``MultiModalConfig.get_limit_per_prompt()`` returns 999 for an unspecified modality, while Qwen2/2.5/3-VL
    # report image/video support as unbounded — so ``None`` is a 999-item budget per supported modality and a
    # configured value LOWERS it (``{"image": 40}`` -> 40 images, video stays 999). The earlier claim that the
    # default was one item per modality ("At most 1 image(s) may be provided") described an older vLLM and was
    # never observed in this runtime. The knob is therefore a BUDGET: it bounds the encoder memory the engine
    # profiles for at start-up (a smaller budget leaves more KV blocks under the same gpu_memory_utilization)
    # and refuses over-budget prompts at admission; ``0`` disables a modality. Keys must be bare lower-case
    # vLLM modality names (VLLM_MODALITIES) and are checked again against the built engine's model-specific
    # supported set in ``roles/rollout/vllm.py::VLLMRollout``.
    limit_mm_per_prompt: dict[str, int] | None = None
    # Optional vLLM engine-performance knobs. Each is forwarded under vLLM's own name only when set,
    # so the engine construction identity of every run that leaves them at None is unchanged.
    # ``compilation_config`` is vLLM's CompilationConfig as a mapping, e.g.
    # ``{"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4, 8]}``;
    # it configures torch.compile and CUDA-graph capture, so a non-empty mapping requires
    # ``enforce_eager=False``. Empty means unset: Hydra applies a dict override by merging, and
    # OmegaConf cannot merge into a None node.
    compilation_config: dict[str, Any] = field(default_factory=dict)
    # ``async_scheduling`` overlaps scheduling of the next engine step with the current step's GPU work.
    async_scheduling: bool | None = None
    # ``max_num_seqs`` caps the sequences one engine runs concurrently; it also bounds CUDA-graph capture sizes.
    max_num_seqs: int | None = None
    # ``kv_cache_memory_bytes`` fixes the per-GPU KV-cache size instead of deriving it from
    # ``gpu_memory_utilization`` after vLLM's memory profile.
    kv_cache_memory_bytes: int | None = None
    # vLLM's custom all-reduce for a supported full-node tensor-parallel engine. It maps every TP
    # peer's buffers over CUDA IPC, so the binder exposes exactly the node's GPUs to each rollout
    # process, its own GPU first so ``cuda:0`` stays its device. Off keeps NCCL and one visible GPU.
    # GRPO only: PPO preflight rejects it because PPO rollout actors keep one visible GPU.
    custom_all_reduce: bool = False
    seed: int | None = None
    # Sampling (vLLM SamplingParams subset; the framework adds n/logprobs per generate call).
    temperature: float = 1.0  # rollout SAMPLING temperature — SEPARATE from GRPOConfig.temperature
    top_p: float = 1.0
    max_tokens: int = 16
    # top_k / repetition_penalty MUST be fields, not omissions. HF `generate` merges kwargs onto the
    # MODEL OBJECT's generation_config, so a knob we do not pass takes whatever that object happens
    # to carry — which is neither a framework default nor visible from this config, and which changes
    # with the load path (building from config + safetensors leaves HF's stock values; switching to
    # `from_pretrained` would silently adopt the checkpoint's generation_config.json instead). Passing
    # both explicitly makes the sampled distribution a property of this config alone.
    # Both mappings live together in `config/build.py` (`sampling_params` / `hf_sampling_params`) so
    # the backends cannot drift; docs/design/grpo-cosmos-rl-parity.md §2.3 records what this measured.
    top_k: int = -1  # <= 0 disables top-k filtering (HF spells that None, vLLM -1)
    repetition_penalty: float = 1.0  # 1.0 == no penalty

    def __post_init__(self) -> None:
        if self.load_format not in ("auto", "dummy"):
            raise ValueError(f"rollout.load_format must be auto|dummy, got {self.load_format!r}")
        if isinstance(self.sleep_level, bool) or not isinstance(self.sleep_level, int):
            raise TypeError(f"rollout.sleep_level must be a non-bool integer, got {self.sleep_level!r}")
        if self.sleep_level not in (1, 2):
            raise ValueError(f"rollout.sleep_level must be 1 or 2, got {self.sleep_level!r}")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise TypeError(f"rollout.max_tokens must be a non-bool integer, got {self.max_tokens!r}")
        if self.max_tokens < 1:
            raise ValueError(f"rollout.max_tokens must be > 0, got {self.max_tokens!r}")
        if self.limit_mm_per_prompt is not None:
            if not isinstance(self.limit_mm_per_prompt, Mapping) or not self.limit_mm_per_prompt:
                raise TypeError(
                    f"rollout.limit_mm_per_prompt must be a non-empty mapping of modality -> max items per request or None, got {self.limit_mm_per_prompt!r}"
                )
            for modality, limit in self.limit_mm_per_prompt.items():
                if not isinstance(modality, str) or not modality.strip():
                    raise TypeError(
                        f"rollout.limit_mm_per_prompt keys must be modality names (non-empty strings), got {modality!r}"
                    )
                if modality != modality.strip() or not modality.isidentifier() or not modality.islower():
                    raise ValueError(
                        f"rollout.limit_mm_per_prompt key {modality!r} is not a bare lower-case modality name "
                        "(no surrounding whitespace, no punctuation): vLLM would treat it as an unknown modality and ignore it"
                    )
                if modality not in VLLM_MODALITIES:
                    raise ValueError(
                        f"rollout.limit_mm_per_prompt key {modality!r} is not a vLLM modality {sorted(VLLM_MODALITIES)}; an unknown "
                        "key is refused here rather than silently ignored by the engine (whose supported set is model-specific and "
                        "is checked again against the built engine in VLLMRollout)"
                    )
                if isinstance(limit, bool) or not isinstance(limit, int):
                    raise TypeError(
                        f"rollout.limit_mm_per_prompt[{modality!r}] must be a non-bool integer, got {limit!r}"
                    )
                if limit < 0:
                    raise ValueError(
                        f"rollout.limit_mm_per_prompt[{modality!r}] must be >= 0 (0 disables the modality), got {limit!r}"
                    )
            # Hydra/OmegaConf supplies nested mappings as DictConfig. Keep the public
            # dataclass field dependency-free and stable after accepting that authoring form.
            self.limit_mm_per_prompt = dict(self.limit_mm_per_prompt)
        if not isinstance(self.compilation_config, Mapping):
            raise TypeError(
                f"rollout.compilation_config must be a mapping of vLLM CompilationConfig fields, got {self.compilation_config!r}"
            )
        if not self.compilation_config:
            self.compilation_config = {}
        else:
            if self.enforce_eager:
                raise ValueError(
                    "rollout.compilation_config configures torch.compile and CUDA-graph capture, which "
                    "rollout.enforce_eager=True disables; set rollout.enforce_eager=False to use it"
                )
            compilation_config = _normalize_factory_kwargs(self.compilation_config)
            if any(not isinstance(key, str) or not key for key in compilation_config):
                raise TypeError(
                    f"rollout.compilation_config keys must be non-empty strings, got {list(compilation_config)!r}"
                )
            self.compilation_config = compilation_config
        if self.async_scheduling is not None and not isinstance(self.async_scheduling, bool):
            raise TypeError(f"rollout.async_scheduling must be None or a bool, got {self.async_scheduling!r}")
        if not isinstance(self.custom_all_reduce, bool):
            raise TypeError(f"rollout.custom_all_reduce must be a bool, got {self.custom_all_reduce!r}")
        if self.custom_all_reduce and self.tensor_parallel_size not in VLLM_CUSTOM_ALL_REDUCE_WORLD_SIZES:
            raise ValueError(
                "rollout.custom_all_reduce needs a vLLM-supported tensor-parallel world size in "
                f"{sorted(VLLM_CUSTOM_ALL_REDUCE_WORLD_SIZES)}, got {self.tensor_parallel_size!r}"
            )
        for field_name in ("max_num_seqs", "kv_cache_memory_bytes"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"rollout.{field_name} must be None or a non-bool integer, got {value!r}")
            if value < 1:
                raise ValueError(f"rollout.{field_name} must be >= 1, got {value!r}")


# ---------------------------------------------------------------------------
# Grid / orchestration topology (RL-framework layer; top-level `config.grid`)
# ---------------------------------------------------------------------------


@dataclass
class GridConfig:
    """Orchestration topology — the RL-framework layer ABOVE the model/engine mesh.

    R counts independent WG execution replicas and is distinct from Torch HSDP ``dp_replicate``.
    The policy always has one execution replica; ``num_replicas`` is the rollout fallback R. A
    per-role ``shapes`` entry overrides the fallback while retaining the policy-R-equals-one
    invariant. ``dp_replicate`` plus ``dp_shard`` live inside the policy's K-rank model world.
    """

    # Orchestration axes (see README "backend × actor/device grouping"):
    #   backend — WHO hosts the role bodies:
    #     "local" (default): bodies live in THIS process; launch the entry under torchrun
    #       for single-GPU / multi-rank FSDP (each rank runs the same in-process stack).
    #     "ray": bodies live in a mesh of Ray worker processes (K=`ranks` per group);
    #       single-node, or joined to a multinode cluster via RAY_ADDRESS.
    #   grouping — HOW roles pack into those hosts (process-level; NOT the GPU-level
    #     colocated/disaggregated placement — see DESIGN "GPU-level placement"):
    #     "single" (default): all roles of a cell share one process (one setup builds all).
    #     "per_role": each role gets its own process per cell (ray only — local+per_role
    #       is rejected at run()).
    backend: str = "local"
    # GPU placement is independent of transport. ``colocated`` puts compatible ActorGroups in one
    # GPU DeviceGroup; ``disaggregated`` assigns them disjoint GPU DeviceGroups. Entry points build
    # that structure first, then select a compatible weight mover (DESIGN §4.0).
    # Keep ``str`` for OmegaConf structured-config compatibility; __post_init__ narrows the value.
    gpu_placement: str = "colocated"
    # Fallback rollout execution-replica count. The fallback policy shape is `(1, ranks)` while the
    # rollout shape is `(num_replicas, ranks)`; model-level HSDP never changes either placement R.
    # A rollout count greater than one therefore needs a placement/transport capable of serving the
    # unequal policy and rollout products.
    #
    # `ranks` stays == the rollout TP rather than being independently configurable. Colocated policy
    # and rollout ActorGroups share one GPU DeviceGroup by flat index, and that IpcMover weight plane
    # is a bijection requiring equal products. Unequal products are valid only in independent GPU
    # DeviceGroups with NIXL; that placement-dependent validation belongs to the binder, not this schema.
    #
    # ray backend only — the local backend is one torchrun world with no replica axis to place.
    num_replicas: int = 1
    ranks: int = 1  # K: worker processes per group for the ray backend (ignored for local)
    grouping: str = "single"  # "single" | "per_role"
    gpus_per_role: dict = field(default_factory=dict)
    # PER-ROLE shapes (spec §8.2), `{role: [R, K]}`. Only `policy` and `rollout` are user-set; the
    # experience roles (data/rm/replay) are DERIVED at `R = R_rollout, K = 1` (that keeps `data →
    # rollout` at op="one" and leaves `replay → policy` the single asymmetric edge — §8.2). Empty
    # (the default) means NO per-role override: policy falls back to `(1, ranks)` and rollout to
    # `(num_replicas, ranks)`. When set, both roles are required and policy R must remain one.
    # Equal products remain required for same-GPU IPC; unequal products are admitted here and fail
    # at bind unless the WEIGHT edge uses independent GPU pools and NIXL. Matches the
    # `gpus_per_role: dict` precedent above.
    shapes: dict = field(default_factory=dict)
    # Explicit topology inventory: exact Ray node address/hostname -> NVLink/fabric domain.
    # Empty preserves unconstrained placement. Nonempty pins complete nodes in domain order.
    gpu_node_domains: dict[str, str] = field(default_factory=dict, kw_only=True)

    _SHAPE_ROLES = ("policy", "rollout")

    def __post_init__(self) -> None:
        if self.num_replicas < 1 or self.ranks < 1:
            raise ValueError(
                f"grid fallback dimensions must be positive, got num_replicas={self.num_replicas}, ranks={self.ranks}"
            )
        self.gpu_placement = normalize_gpu_placement(self.gpu_placement, field_name="grid.gpu_placement")
        from cosmos_framework.posttrain.control.placement.topology import validate_gpu_node_domains

        validate_gpu_node_domains(self.gpu_node_domains)
        if self.gpu_node_domains and (self.backend != "ray" or self.gpu_placement != "colocated"):
            raise ValueError("grid.gpu_node_domains requires Ray colocated GPU placement")
        if not self.shapes:
            return
        # Unknown keys RAISE — the "configured, then silently ignored" failure this branch has paid
        # for twice. Only policy/rollout are user-set here.
        unknown = [k for k in self.shapes if k not in self._SHAPE_ROLES]
        if unknown:
            raise ValueError(
                f"grid.shapes has unknown key(s) {unknown}: only {list(self._SHAPE_ROLES)} are "
                f"user-set (the experience roles derive at R_rollout). A typo here would otherwise "
                f"fall back to the uniform default without a word."
            )
        # Both, or neither: placement validates the policy→rollout edge from the complete pair, and
        # 'only policy' cannot say what it balances against.
        missing = [r for r in self._SHAPE_ROLES if r not in self.shapes]
        if missing:
            raise ValueError(
                f"grid.shapes is missing {missing}: set BOTH policy and rollout or leave shapes empty "
                f"for the policy=(1,ranks), rollout=(num_replicas,ranks) fallback topology."
            )
        # Normalise to (R, K) int tuples; OmegaConf / `--` dotlist hand back lists. A length != 2 is a
        # different, silently-wrong topology, so it fails rather than being truncated.
        norm: dict[str, tuple[int, int]] = {}
        for role in self._SHAPE_ROLES:
            pair = list(self.shapes[role])
            if len(pair) != 2:
                raise ValueError(
                    f"grid.shapes[{role!r}]={pair} must be exactly [R, K] (length 2); a stray value is "
                    f"a different topology, not a truncation."
                )
            r, k = int(pair[0]), int(pair[1])
            if r < 1 or k < 1:
                raise ValueError(
                    f"grid.shapes[{role!r}]=[{r}, {k}] must have positive [R, K]: a zero or negative "
                    f"axis is an empty/inverted topology."
                )
            norm[role] = (r, k)
        if norm["policy"][0] != 1:
            raise ValueError(
                f"grid.shapes['policy'][0]={norm['policy'][0]} is unsupported: POLICY_REPLICAS must be 1; "
                "configure HSDP with model dp_replicate and dp_shard inside POLICY_RANKS"
            )
        self.shapes = norm
        # Product compatibility is a WEIGHT-edge placement fact. Keeping it here used to reject the
        # intended FSDP64/128→TP4 topology before the binder could see that NIXL was selected, and it
        # also imposed an unrelated constraint on future frozen model groups. The binder still rejects
        # unequal products for same-process/host/IPC and admits them only for cross-device NIXL.

    def policy_shape(self) -> tuple[int, int]:
        """`(R, K)` for policy — from `shapes` if set, else `(1, ranks)`.

        Returns:
            The policy execution-replica count (always one) and rank width.
        """
        return self.shapes["policy"] if self.shapes else (1, self.ranks)

    def rollout_shape(self) -> tuple[int, int]:
        """`(R, K)` for rollout — from `shapes` if set, else `(num_replicas, ranks)`.

        Returns:
            The rollout replica count and rank width.
        """
        return self.shapes["rollout"] if self.shapes else (self.num_replicas, self.ranks)


@dataclass
class PPOGridConfig(GridConfig):
    """Shared grid placement plus the two PPO-only model-role widths."""

    backend: str = "ray"
    grouping: str = "per_role"
    ranks: int = 4
    reference_ranks: int = 2
    critic_ranks: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("reference_ranks", "critic_ranks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")


# ---------------------------------------------------------------------------
# DPO objective config — purely the RL objective (no model, no grid)
# ---------------------------------------------------------------------------


@dataclass
class CheckpointStoreConfig:
    """Primary checkpoint storage selected below the shared checkpoint manager.

    ``local`` preserves the existing on-disk checkpoint lifecycle. ``object_store`` writes DCP
    components directly through the S3-compatible backend, so no complete local checkpoint copy
    exists. An empty ``prefix`` mirrors the local
    ``<project>/<group>/<name>/<run_ts>/checkpoints`` layout.
    """

    backend: str = "local"
    bucket: str = "<checkpoint-experiments-bucket>"
    credentials: str = "credentials/<credential>.secret"
    prefix: str = ""

    def __post_init__(self) -> None:
        if self.backend not in ("local", "object_store"):
            raise ValueError(f"checkpoint_store.backend must be 'local' or 'object_store', got {self.backend!r}")
        for field_name in ("bucket", "credentials", "prefix"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"checkpoint_store.{field_name} must be a string, got {value!r}")


@dataclass
class CheckpointUploadConfig:
    """Post-save checkpoint upload (cosmos-rl `save_to_object_store` parity). When
    enabled, rank 0 pushes each just-saved iter dir to `s3://<bucket>/<prefix>/iter_*/`
    on the checkpoint write job (idempotent; failure is a warning, never kills
    ordinary cadence training) and drops a `.uploaded` marker that gates optimizer and
    whole-model pruning. Full-checkpoint validation completion is strict and remains
    publication-pending until the same archive operation succeeds. Checkpoint-free validation
    has no publication phase, while partial-artifact upload keeps ordinary best-effort semantics.

    The bucket is just an address — retention semantics live on the bucket itself
    (the default experiments bucket has a ~2-month lifecycle; promote milestones to
    the permanent bucket explicitly via `utils/object_store.py copy`). `prefix` empty
    ⇒ mirror the local layout (`<project>/<group>/<name>/<run_ts>/checkpoints`),
    which is exactly the reference-experiment bucket structure eval tooling expects."""

    enabled: bool = False
    bucket: str = "<checkpoint-experiments-bucket>"
    credentials: str = "credentials/<credential>.secret"
    prefix: str = ""


@dataclass
class DatasetConfig:
    """GRPO prompt source: arithmetic, JSONL, Hugging Face, or an importable custom factory."""

    name: str = ""  # Hugging Face dataset id, e.g. "openai/gsm8k"
    subset: str = ""  # e.g. "main"
    split: str = "train"
    path: str = ""  # local JSONL, one object per line
    # Dotted callable path returning a map-style raw-row dataset. Only strings and kwargs cross
    # into Ray; the data actor constructs the source locally.
    factory: str = ""
    factory_kwargs: dict = field(default_factory=dict)
    prompt_column: str = ""  # e.g. "question"
    # Ground truth for answer-checking rewards. Empty ⇒ prompts only; whether that suits the
    # configured reward is a cross-role contract, so bind_check decides it, not this object.
    response_column: str = ""  # e.g. "answer"
    shuffle: bool = False
    # Exact continuation of a run restarted from a MODEL-ONLY artifact (which carries no data cursor).
    # `cursor_offset`: prompt draws EACH DATA SHARD skips before serving its first prompt -- the number
    # of draws the source run's shard had consumed (steps x prompts per data cell, plus dynamic-sampling
    # extra draws). With the source run's pool, shuffle flag, resolved data seed and data-replica count,
    # the first prompt served is the source shard's NEXT draw, not its first one. 0 = start of the
    # sequence. `cursor_shards`: the data-replica count the offset was computed for; the data role
    # refuses a run whose shard count differs (the per-shard stride changes with it). Required (non-zero)
    # whenever `cursor_offset` is set, so that check is never off for a continuation; 0 only without one.
    # `checkpoint/continuation.py::ContinuationPlan.overrides()` emits both from the source run.
    cursor_offset: int = 0
    cursor_shards: int = 0
    # A MAP over raw rows, as a dotted path: "pkg.mod.fn", `fn(row: dict) -> dict`.
    #
    # The framework loads the dataset (`name`/`subset`/`split`) and calls this on each drawn row.
    # It rewrites VALUES -- typically turning the prompt column into a chat conversation -- and
    # keeps the schema: same keys in, same keys out. `prompt_column` and `response_column` are then
    # applied to the mapped row, so they mean exactly what they say whether or not a recipe sets
    # this. (They previously did not: a builder silently replaced `prompt_column` while
    # `response_column` beside it stayed live, with nothing to tell them apart.)
    #
    # Resolved with `cosmos_framework.utils.serialization.load_callable`, so it crosses to Ray actors as a
    # string -- the same requirement `--config` already imposes on the recipe module.
    #
    # There is deliberately NO system_prompt (or user_suffix, or few_shot_k) field. Config expresses
    # SELECTION -- which dataset, which split, which column; code expresses CONSTRUCTION -- how a row
    # becomes a prompt. A system prompt is construction: it is the first rung of an endless ladder,
    # and it is meaningless for base-model RL, which has no roles at all. A recipe that needs one
    # writes a short items_fn in its own module (see configs/grpo_qwen3b_gsm8k.py). The callable is
    # a PURE CPU row transform: no I/O, CUDA initialization, or mutable side effects. Validation
    # materializes retained mapped rows once under isolated common RNGs and fingerprints the actual
    # result, but arbitrary external state cannot be rolled back generically.
    #
    # This maps a row from either a framework-loaded HF dataset or ``factory``. It remains separate
    # from source construction: the factory selects storage; items_fn rewrites row values.
    items_fn: str = ""
    # NO prompt_style knob. A dataset env that adds a format instruction emits ONE wording, its own
    # (`configs/grpo_qwen3b_gsm8k.py::GSM8K_SYSTEM_PROMPT`). A switch to emit cosmos-rl's wording used
    # to live here; it was configured, printed into every run log, and then dropped on the floor by
    # the example's call sites, which is how a parity report came to quote numbers taken with the
    # other prompt. See that constant's comment.

    def __post_init__(self) -> None:
        self.factory_kwargs = _normalize_factory_kwargs(self.factory_kwargs)
        if self.factory_kwargs and not self.factory:
            raise ValueError("rl.data.factory_kwargs requires rl.data.factory")
        selected = [
            name for name, value in (("path", self.path), ("name", self.name), ("factory", self.factory)) if value
        ]
        if len(selected) > 1:
            raise ValueError(
                f"set either rl.data.path or rl.data.name or rl.data.factory; select at most one, got {selected}"
            )
        if self.items_fn and self.path:
            raise ValueError(
                f"set either rl.data.items_fn or rl.data.path, not both (got {self.items_fn!r}, "
                f"{self.path!r}): items_fn transforms a row from rl.data.name or rl.data.factory."
            )
        if self.items_fn and not (self.name or self.factory):
            raise ValueError(
                f"rl.data.items_fn={self.items_fn!r} needs rl.data.name or rl.data.factory: it "
                "transforms a row from a separately selected dataset source."
            )
        if self.path and not self.prompt_column:
            raise ValueError(f"rl.data.path={self.path!r} needs rl.data.prompt_column (the field holding the prompt)")
        if self.name and not self.prompt_column:
            # A named hub dataset needs its columns just as much as a JSONL path does: the column
            # mapping reads prompt_column/response_column off this config, so without them the config
            # VALIDATES and then fails (or serves empty prompts) at run time. Required WITH an
            # items_fn too -- that maps a row to a row without renaming, so the column still names
            # the prompt afterwards.
            raise ValueError(f"rl.data.name={self.name!r} needs rl.data.prompt_column (the field holding the prompt)")
        if self.factory and not self.prompt_column:
            raise ValueError(
                f"rl.data.factory={self.factory!r} needs rl.data.prompt_column (the raw-row field holding the prompt)"
            )
        if self.name and not self.subset:
            # `subset` is the HF *config* name. Empty is not "no config": it is passed through to
            # load_dataset as "", which the wired dataset rejects — so `--validate` succeeds and the
            # job dies later inside actor setup, after placement and model construction. Datasets
            # that genuinely have no config would need an explicit opt-out, not a silent empty
            # string that only fails once GPUs are held.
            raise ValueError(
                f"rl.data.name={self.name!r} needs rl.data.subset (the HF dataset config, e.g. 'main' "
                f'for openai/gsm8k); an empty subset reaches load_dataset as "" and fails at setup'
            )
        if self.name and not self.response_column:
            # `prompt_source` reports this source as carrying ground truth, which is what lets a
            # verifiable reward (gsm8k) pass `bind_check`. Without response_column that claim is
            # false: the env has no reference answer to attach, and the reward's contract check
            # would raise mid-run instead of at config time.
            raise ValueError(
                f"rl.data.name={self.name!r} needs rl.data.response_column: this source is declared "
                f"as carrying ground truth, which a verifiable reward relies on"
            )
        if not (self.path or self.name or self.factory) and (
            self.prompt_column or self.response_column or self.shuffle
        ):
            raise ValueError("rl.data.prompt_column/response_column/shuffle require rl.data.path/name/factory")
        if not self.name and (self.subset or self.split != "train"):
            raise ValueError("rl.data.subset and a non-default split require rl.data.name")
        for field_name in ("cursor_offset", "cursor_shards"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"rl.data.{field_name} must be an integer, got {value!r}")
            if value < 0:
                raise ValueError(f"rl.data.{field_name} must be >= 0, got {value}")
        if (self.cursor_offset or self.cursor_shards) and not (self.path or self.name or self.factory):
            # The arithmetic source synthesizes prompts from its seed and owns no cursor, so an offset
            # into it is meaningless; refuse at config time rather than silently serving the first prompts.
            raise ValueError(
                "rl.data.cursor_offset/cursor_shards require an explicit prompt source (rl.data.path/name/factory): "
                "the arithmetic source has no cursor to continue"
            )
        if self.cursor_offset and not self.cursor_shards:
            # An offset is per shard and exact only for the shard count it was computed for; with `cursor_shards=0`
            # the data role's topology check is off and a continuation that changed its replica count would serve a
            # different prompt sequence unnoticed.
            raise ValueError(
                f"rl.data.cursor_offset={self.cursor_offset} requires rl.data.cursor_shards (the data-replica count "
                "the offset was computed for; `ContinuationPlan.overrides()` emits both): with 0 the shard-topology "
                "check is off and a continuation that changes its replica count would serve a different prompt sequence"
            )


@dataclass
class ValidationScheduleConfig:
    """Shared train-time model-validation lifecycle.

    Validation is synchronous at committed optimizer-step boundaries. Once enabled, final
    validation is mandatory; ``every_steps=0`` disables only the periodic trigger. A validation
    dataset is never derived from the training source implicitly -- the algorithm-specific parent
    config validates its typed ``data`` field.
    """

    enabled: bool = False
    before_training: bool = False
    every_steps: int = 20
    save_checkpoint: bool = False
    checkpoint_components: list[str] = field(default_factory=lambda: list(FULL_CHECKPOINT_COMPONENTS))
    batch_size: int = 1
    max_samples: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"validation.enabled must be bool, got {self.enabled!r}")
        if not isinstance(self.before_training, bool):
            raise TypeError(f"validation.before_training must be bool, got {self.before_training!r}")
        if not isinstance(self.save_checkpoint, bool):
            raise TypeError(f"validation.save_checkpoint must be bool, got {self.save_checkpoint!r}")
        for field_name in ("every_steps", "batch_size", "seed"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"validation.{field_name} must be an integer, got {value!r}")
        if self.every_steps < 0:
            raise ValueError(f"validation.every_steps must be >= 0, got {self.every_steps}")
        if self.batch_size < 1:
            raise ValueError(f"validation.batch_size must be >= 1, got {self.batch_size}")
        if self.max_samples is not None:
            if isinstance(self.max_samples, bool) or not isinstance(self.max_samples, int):
                raise TypeError(f"validation.max_samples must be an integer or None, got {self.max_samples!r}")
            if self.max_samples < 1:
                raise ValueError(f"validation.max_samples must be >= 1 when set, got {self.max_samples}")
        self.checkpoint_components = _normalize_checkpoint_components(self.checkpoint_components)


@dataclass
class ValidationSamplingConfig:
    """Request-local sampling used only for rollout/reward validation."""

    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = 1
    repetition_penalty: float = 1.0
    n_generation: int = 1
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        for field_name in ("temperature", "repetition_penalty"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"validation.sampling.{field_name} must be a real number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"validation.sampling.{field_name} must be finite, got {value!r}")
        if self.temperature < 0.0:
            raise ValueError(f"validation.sampling.temperature must be >= 0, got {self.temperature}")
        if self.top_p is not None:
            if isinstance(self.top_p, bool) or not isinstance(self.top_p, (int, float)):
                raise TypeError(f"validation.sampling.top_p must be a real number or None, got {self.top_p!r}")
            if not math.isfinite(float(self.top_p)) or not 0.0 < self.top_p <= 1.0:
                raise ValueError(f"validation.sampling.top_p must be finite and in (0, 1], got {self.top_p}")
        for field_name in ("top_k", "n_generation", "max_tokens"):
            value = getattr(self, field_name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"validation.sampling.{field_name} must be an integer or None, got {value!r}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError("validation.sampling.top_k must be >= 1 when set")
        if self.repetition_penalty <= 0.0:
            raise ValueError(f"validation.sampling.repetition_penalty must be > 0, got {self.repetition_penalty}")
        if self.n_generation < 1:
            raise ValueError(f"validation.sampling.n_generation must be >= 1, got {self.n_generation}")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"validation.sampling.max_tokens must be >= 1 when set, got {self.max_tokens}")


@dataclass
class DPOValidationConfig(ValidationScheduleConfig):
    """Held-out preference-objective validation owned by the Trainer."""

    data: PreferenceEnvConfig = field(
        default_factory=lambda: PreferenceEnvConfig(
            shuffle=False,
            cursor_resume=CURSOR_RESUME_OFF,
            cfg_dropout_rate=0.0,
        )
    )


@dataclass
class GRPOValidationConfig(ValidationScheduleConfig):
    """Held-out rollout/reward validation using the existing data/rollout/rm roles."""

    data: DatasetConfig = field(default_factory=DatasetConfig)
    sampling: ValidationSamplingConfig = field(default_factory=ValidationSamplingConfig)
    # Empty inherits the training reward. A non-empty value names the same reward registry.
    reward: str = ""


@dataclass
class PPODatasetConfig:
    """PPO prompt source. Empty `name` selects the deterministic arithmetic source;
    otherwise the PPO data role loads the pinned Hugging Face dataset and preserves
    its configured text and media columns in each episode."""

    name: str = ""  # e.g. "openai/gsm8k"
    # Local rows read as written (roles/env/dataset.py::JsonlRows): JSONL or any whitespace-separated stream of
    # JSON objects (pretty-printed or not), one JSON array, or the same gzip-compressed (gzip only).
    path: str = ""
    factory: str = ""  # importable map-style dataset factory
    factory_kwargs: dict = field(default_factory=dict)
    items_fn: str = ""  # importable pure raw-row transform
    revision: str = ""  # immutable dataset revision when reproducibility matters
    subset: str = ""  # e.g. "main"
    split: str = "train"
    prompt_column: str = ""  # e.g. "question"
    response_column: str = ""  # e.g. "answer"
    image_column: str = ""
    video_column: str = ""
    id_column: str = ""
    # Selected finite-JSON metadata crosses under the collision-safe ``dataset_row`` mapping.
    # A list keeps the schema OmegaConf-structured-safe; consumers canonicalize to tuple.
    metadata_columns: list[str] = field(default_factory=list)
    shuffle: bool = False

    def __post_init__(self) -> None:
        # Imported here, not at module scope: the behavior1k sim image carries neither omegaconf nor
        # pyyaml, so a module-level import would break `import cosmos_framework.posttrain.config.rl`
        # there. Same discipline as `_materialize_factory_kwargs` above.
        from omegaconf import OmegaConf

        self.factory_kwargs = _normalize_factory_kwargs(self.factory_kwargs)
        selected = [
            name for name, value in (("name", self.name), ("path", self.path), ("factory", self.factory)) if value
        ]
        if len(selected) > 1:
            raise ValueError(f"set at most one PPO dataset source, got {selected}")
        if self.factory_kwargs and not self.factory:
            raise ValueError("rl.data.factory_kwargs requires rl.data.factory")
        if self.items_fn and not selected:
            raise ValueError("rl.data.items_fn requires an explicit PPO dataset source")
        if not self.name and (self.revision or self.subset):
            raise ValueError("rl.data.revision and subset require rl.data.name")
        if selected and not self.prompt_column:
            raise ValueError("an explicit PPO dataset source requires prompt_column")

        metadata_columns: object = self.metadata_columns
        if type(metadata_columns) is list:
            candidate = metadata_columns
        elif OmegaConf.is_list(metadata_columns):
            candidate = OmegaConf.to_container(metadata_columns, resolve=True)
            if type(candidate) is not list:
                raise TypeError(f"metadata_columns must resolve to a list, got {type(candidate).__name__}")
        else:
            raise TypeError(f"metadata_columns must be a list, got {type(metadata_columns).__name__}")
        normalized = normalize_prompt_metadata_columns(
            candidate,
            configured_columns=(
                self.prompt_column,
                self.response_column,
                self.image_column,
                self.video_column,
                self.id_column,
            ),
        )
        self.metadata_columns = list(normalized)


@dataclass
class RunConfig:
    """Lifecycle fields every algorithm's run loop + spine consume — hoisted so DPO and GRPO share the
    exact `rl.*` dotlist paths (incl. the RL-owned `optimizer`). Algorithm-specific knobs live on the
    subclasses."""

    grad_accum: int = 1
    max_steps: int = 1000
    save_every: int = 0  # 0 ⇒ never (the loop owns cadence)
    # Resume: "auto" (DEFAULT — matches upstream/cosmos-rl convention): pick up the latest
    # checkpoint under this job's out dir, fresh start if none — so an unchanged command
    # survives preemption/requeue with zero config. "off" (or "") = force a fresh run.
    # Or an explicit iter_* dir / legacy .pt (must exist, else fail loud).
    # See checkpoint.resolve_resume; a resume always logs "resumed from ...".
    resume: str = "auto"
    # Run-attempt timestamp ("YYYYmmddHHMMSS") — the upstream convention run identity: the
    # out_dir becomes <...>/<name>/<run_ts>/, the wandb id IS run_ts, and eval tooling
    # takes it as $RUN_TS. Default "" = resolve at startup (adopt the latest existing
    # attempt when resuming, mint a new one otherwise — see the checkpoint package.
    # resolve_run_ts). Set explicitly only by the framework (driver → ray workers).
    run_ts: str = ""
    # "consolidated" (rank-0 full-state training .pt; a DTensor-sharded trainer still writes its
    # OPTIMIZER as per-rank DCP shards, `optim/`, because the gathered fp32 moments do not fit one
    # host) | "dcp" (sharded parallel write; compatible state can reshard to a different world).
    # See the checkpoint package.
    checkpoint_format: str = "consolidated"
    # Logical state families written by cadence and normal-final saves. Partial selections are
    # checkpoint artifacts, not exact-resume checkpoints; emergency saves force the full plan unless
    # `emergency_checkpoint_components` below says otherwise.
    checkpoint_components: list[str] = field(default_factory=lambda: list(FULL_CHECKPOINT_COMPONENTS))
    # Component plan of the termination-triggered emergency save (preemption / timeout seen by
    # `control/termination.py` at a step boundary). Default "full" = the behaviour before this knob existed.
    #   "full": the emergency save is upgraded to the exact-resume plan `[model, optimizer, training_state]`
    #     whatever `checkpoint_components` selects, so the requeued run exact-resumes through `rl.resume=auto`.
    #   "run": the emergency save writes `checkpoint_components` plus `model` (an emergency save always carries
    #     the weights; for the usual model-only plan the two coincide) through the cadence path -- the same
    #     location and commit marker as the save the loop would have written at its next `save_every` boundary.
    #     It is skipped only when every component of that plan is already on disk for the step. A partial
    #     emergency artifact is NOT exact-resume capable (no optimizer moments, step clock, RNG or data cursor):
    #     `rl.resume=auto` will not adopt it, and continuing from it is a warm start, exactly as from a partial
    #     cadence save. The trade pays when the full plan cannot be written inside the scheduler's grace (two
    #     fp32 Adam moments are ~4x the bf16 weight bytes); the measured save times live in
    #     docs/reference/checkpoint.md, "Preemption / timeout (emergency save)".
    # Validation coverage rule (`__post_init__` below): under "run" a saving validation plan
    # (`validation.save_checkpoint`) must COVER `checkpoint_components` plus `model`, so the one artifact a step
    # may already hold stands in for the emergency save -- a committed artifact is never re-committed, and a
    # validation artifact missing any component of the plan would leave only the full plan to write.
    # Honoured by every loop on the single-role coordinator (GRPO, OPD, DPO, NFT, SFT) through
    # `CheckpointCoordinator.save_emergency`; the durable validation transaction (validation pending at the
    # boundary) always writes the full plan because its resume needs exact state. The PPO loops' boundaries
    # always write the full plan (multi-role boundary / `save_full`), so their config validation refuses "run"
    # (`require_full_emergency_plan`) instead of ignoring it.
    emergency_checkpoint_components: str = "full"
    # save_async: consolidated only — disk write on a background thread (the entry finalizes on exit).
    save_async: bool = True
    # keep_optim_last: None lets the checkpoint manager choose a backend-aware default; an integer
    # prunes optimizer state from all but the newest N iter dirs (0 = keep all).
    keep_optim_last: int | None = None
    # keep_model_last: prune ENTIRE iter dirs beyond the newest N; with upload, only after `.uploaded`
    # exists — the bucket is the archive, never delete the only copy (0 = keep all).
    keep_model_last: int = 0
    checkpoint_store: CheckpointStoreConfig = field(default_factory=CheckpointStoreConfig)
    upload: "CheckpointUploadConfig" = field(default_factory=lambda: CheckpointUploadConfig())
    # HF safetensors export, beside the checkpoint lifecycle it belongs to. Relocated from
    # HFModelConfig, which is the model-LOAD spec (identity/dtype/attn) — exporting is an OUTPUT
    # concern with the run's lifetime, not a property of the weights being loaded. Reserved: no
    # trainer reads these yet (same as before the move).
    export_dir: str = ""  # Reserved HF artifact destination; no trainer/entrypoint consumes it yet.
    export_every: int = 0  # Reserved export cadence; zero has no implicit end-of-run action today.
    # The run's single seed knob: every stochastic component derives from it. DPO seeds its dataset
    # from it; GRPO seeds env init. PPO rollout sampling derives topology-invariant per-request seeds
    # from the separately configured engine base seed (config.rollout.seed).
    seed: int = 0
    # Bit-reproducibility mode: the driver AND every Ray actor's setup call
    # `utils.determinism.enable_determinism()` before building a model when this is set. Default
    # False: the determinism knobs (cuDNN deterministic, use_deterministic_algorithms) cost throughput
    # and are opt-in, matching cosmos3's parity harness convention (see utils/determinism.py). Beside
    # `seed` because both algorithms (DPO and GRPO) benefit equally — reproducibility needs a fixed
    # seed AND a deterministic kernel path, neither alone is sufficient.
    deterministic: bool = False
    # RL owns the optimizer (see OptimizerConfig + build_optimizer). Shared by every algorithm; for PPO
    # this is the policy optimizer, paired with PPOConfig.critic_optimizer. Defaults to training all
    # params (train_keys=[]) — a freeze subset is a model+recipe choice set by the experiment, not baked
    # into the shared default.
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    def __post_init__(self) -> None:
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 0:
            raise ValueError(f"max_steps must be a non-negative integer, got {self.max_steps!r}")
        if isinstance(self.save_every, bool) or not isinstance(self.save_every, int) or self.save_every < 0:
            raise ValueError(f"save_every must be a non-negative integer, got {self.save_every!r}")
        for field_name in ("keep_optim_last", "keep_model_last"):
            value = getattr(self, field_name)
            if field_name == "keep_optim_last" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                expected = "a non-bool integer or None" if field_name == "keep_optim_last" else "a non-bool integer"
                raise TypeError(f"{field_name} must be {expected}, got {value!r}")
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0, got {value}")
        if self.grad_accum < 1:
            raise ValueError(f"grad_accum must be >= 1, got {self.grad_accum}")
        if self.checkpoint_format not in ("consolidated", "dcp"):
            raise ValueError(f"checkpoint_format must be 'consolidated' or 'dcp', got {self.checkpoint_format!r}")
        self.checkpoint_components = _normalize_checkpoint_components(self.checkpoint_components)
        if self.emergency_checkpoint_components not in EMERGENCY_CHECKPOINT_PLANS:
            raise ValueError(
                "emergency_checkpoint_components must be one of "
                f"{list(EMERGENCY_CHECKPOINT_PLANS)}, got {self.emergency_checkpoint_components!r}"
            )
        # Under "run" the only artifact a step may hold before the emergency save is a validation artifact (an
        # off-cadence validation with save_checkpoint). A committed artifact is never re-committed, so if that
        # artifact lacked ANY component of the emergency plan -- the run's plan UNION the model -- the emergency save
        # could only write the full plan: the TiB-scale save "run" exists to avoid. The validation plan must therefore
        # COVER the run plan plus the model (a superset), so its artifact stands in for the emergency save
        # (control/checkpointing.py, _save_run_plan_with_model). Requiring the model alone was not enough: a run plan
        # [model, training_state] with a validation plan [model] was accepted and its emergency save escalated to the
        # full plan, adding the optimizer.
        validation = getattr(self, "validation", None)
        if (
            self.emergency_checkpoint_components == "run"
            and validation is not None
            and bool(getattr(validation, "enabled", False))
            and bool(getattr(validation, "save_checkpoint", False))
        ):
            validation_plan = CheckpointComponentPlan.from_value(getattr(validation, "checkpoint_components", None))
            emergency_plan = frozenset(self.checkpoint_components) | {"model"}
            missing = sorted(emergency_plan - validation_plan.selected)
            if missing:
                raise ValueError(
                    "emergency_checkpoint_components='run' requires validation.checkpoint_components to cover "
                    "checkpoint_components plus 'model' when validation.save_checkpoint is set: an off-cadence "
                    "validation artifact missing any of them would leave the emergency save of that step only the "
                    "full plan to write (a committed artifact is never re-committed), the TiB-scale save 'run' exists "
                    f"to avoid; got validation.checkpoint_components={list(validation_plan.names)!r} under "
                    f"checkpoint_components={list(self.checkpoint_components)!r}, missing {missing}"
                )


def require_full_emergency_plan(rl: Any, *, algorithm: str, reason: str) -> None:
    """Refuse `rl.emergency_checkpoint_components` other than "full" for a loop whose boundary always writes the full plan.

    The selector is a shared `RunConfig` field, so every algorithm's config accepts it; a loop that cannot honour "run"
    (its save is the full plan at the cadence and at a termination trigger alike) must say so at config validation
    rather than write the full plan behind a config that asked for the run plan.

    Args:
        rl: the run config (any object with an `emergency_checkpoint_components` attribute; absent = "full").
        algorithm: the loop's name for the message.
        reason: why that loop always writes the full plan, with the code path.

    Raises:
        ValueError: the selector is not "full".
    """
    plan = str(getattr(rl, "emergency_checkpoint_components", "full"))
    if plan != "full":
        raise ValueError(
            f"{algorithm} ignores rl.emergency_checkpoint_components={plan!r}: {reason}; set 'full' (the plan it writes)"
        )


PRECISION_DTYPES: tuple[str, ...] = ("float32", "bfloat16", "float16")
"""Dtype names accepted by the `HFRunConfig` precision fields (cosmos-rl's same three choices)."""

PRECISION_MODES: tuple[str, ...] = ("fsdp", "autocast")
"""How the compute dtype reaches the forward: the FSDP2 MixedPrecisionPolicy parameter cast, or torch.autocast."""


@dataclass
class HFRunConfig(RunConfig):
    """`RunConfig` plus the FSDP2 precision block, for the algorithms whose backend is `hf`.

    WHY PRECISION LIVES HERE AND NOT ON `HFModelConfig`. A parameter has TWO dtypes under FSDP2
    mixed precision and they are independent knobs:

      - STORAGE (`master_dtype`) — what the parameter tensor is held in between steps. The optimizer
        reads and writes this, and optimizer state inherits it: `torch.optim.AdamW` allocates
        `exp_avg`/`exp_avg_sq` with `zeros_like(p)`, so the moment dtype is decided HERE.
      - COMPUTE (`param_dtype`) — what the forward/backward runs in.
        `MixedPrecisionPolicy(param_dtype=...)` all-gathers the sharded parameter, casts it for the
        matmuls, and discards the cast copy.

    `HFModelConfig.dtype` drives BOTH today (`model/hf.py` uses it for the parameter normalization
    AND for `ShardPlan.param_dtype`), which makes the useful combination — fp32 storage with bf16
    compute, i.e. the recipe both this framework and cosmos-rl actually want — INEXPRESSIBLE. That
    is why `rl.optimizer.name="fusedadam"` exists at all: apex `FusedAdam(master_weights=True)`
    rebuilds, inside the optimizer, the fp32 master copy the model layer could not provide. The
    apex dependency is a consequence of the missing knob, not an independent design choice.

    Splitting them belongs beside `optimizer` rather than on the model spec because the coupling is
    to the OPTIMIZER: the storage dtype decides the moment dtype. cosmos-rl draws the same line —
    its `[train]` section holds `optm_*` together with `master_dtype`/`param_dtype`/
    `fsdp_reduce_dtype`/`transfer_dtype`, and its `[policy]` model config carries no dtype at all
    (`policy/model/base.py` reads `config.train.master_dtype` at construction time).

    NOT ON `RunConfig`, deliberately. DPO's backend is `i4` (`config_dpo.py`), whose model comes from
    the inherited LazyConfig tree and carries its own dtype — these fields would be silently ignored
    there, and "a knob the layers below refuse to honor is worse than no knob". Declaring them one
    level down makes `rl.master_dtype=...` on a DPO run an unknown-key error at config load, which
    is the correct failure. `GRPOConfig` subclasses this; `OPDConfig` inherits it through `GRPOConfig`.

    A third, orthogonal knob says HOW the compute dtype is delivered: `mixed_precision_mode` — "fsdp"
    (the MixedPrecisionPolicy parameter cast, default) or "autocast" (`torch.autocast(param_dtype)`
    around the forward with parameters left in `master_dtype`, what HF `TrainingArguments(bf16=True)` does).

    EVERY FIELD DEFAULTS TO `""` — meaning "resolve from the deprecated `model_hf` fields", so this
    class is a pure schema addition with no behavioural change until a recipe opts in. The
    resolution chain itself lives in ONE place, `config/build.py::resolve_precision`, so the
    fallbacks and the deprecation warning cannot drift between call sites.
    """

    # "" ⇒ inherit the deprecated `model_hf.dtype`. The forward/backward compute dtype.
    param_dtype: str = ""
    # "" ⇒ inherit the RESOLVED `param_dtype` (i.e. today's behaviour, where storage == compute).
    # Set "float32" with `param_dtype="bfloat16"` for the reference recipe: fp32 optimizer state,
    # bf16 matmuls, and no apex needed.
    master_dtype: str = ""
    # "" ⇒ inherit the deprecated `model_hf.fsdp_reduce_dtype`. Gradient all-reduce dtype.
    reduce_dtype: str = ""
    # "" ⇒ the RESOLVED `master_dtype`, matching cosmos-rl's default. The dtype weights are cast to
    # when pushed policy -> rollout: the trainer's storage dtype need not be the inference engine's,
    # and without this the transfer plan would ship the trainer's (possibly fp32) buffers into a bf16
    # engine at double the wire bytes. cosmos-rl applies the same cast on both ends of that hop.
    transfer_dtype: str = ""
    # "" ⇒ "fsdp". How `param_dtype` reaches the forward. "fsdp": the FSDP2 MixedPrecisionPolicy cast
    # (needs an FSDP wrap whenever master_dtype != param_dtype). "autocast": parameters stay in
    # master_dtype and the forward runs under torch.autocast(param_dtype) at every world size, wrapped
    # or not. Not supported by the native backend, whose builders deliver the compute dtype only
    # through their own FSDP cast; float16 autocast is refused because the trainer has no GradScaler.
    mixed_precision_mode: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("param_dtype", "master_dtype", "reduce_dtype", "transfer_dtype"):
            value = getattr(self, name)
            if value and value not in PRECISION_DTYPES:
                # Checked at config load, not at model construction: a typo'd dtype otherwise
                # surfaces as an AttributeError inside `getattr(torch, ...)` after placement and
                # model build, i.e. once GPUs are already held.
                raise ValueError(
                    f"rl.{name}={value!r} must be one of {PRECISION_DTYPES} (or '' to inherit); "
                    f"see HFRunConfig for the storage-vs-compute split"
                )
        if self.mixed_precision_mode and self.mixed_precision_mode not in PRECISION_MODES:
            raise ValueError(
                f"rl.mixed_precision_mode={self.mixed_precision_mode!r} must be one of {PRECISION_MODES} "
                f"(or '' for 'fsdp')"
            )


@dataclass
class VideoRewardConfig:
    """Bounded decoded-video reward contract for one rank-local WFM trajectory.

    Endpoint URLs and credentials are actor-environment inputs and never enter resolved config.
    ``operation_timeout_s`` is a hard wall-clock bound for one external transport process, while
    ``total_deadline_s`` bounds decode, enqueue, polling, and accepted result time together.
    """

    provider_revision: str
    provider_name: str = "dance_grpo"
    score_key: str = "overall_reward"
    video_fps: float = 16.0
    # Actor-local filesystem root for the decoded uint8 request payload. None uses
    # tempfile's platform default; production recipes should name capacity-managed storage.
    spool_dir: str | None = None
    connect_timeout_s: float = 10.0
    operation_timeout_s: float = 30.0
    total_deadline_s: float = 300.0
    poll_interval_s: float = 1.0
    pull_max_attempts: int = 3
    reward_clip_min: float = -5.0
    reward_clip_max: float = 5.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        safe_name = re.fullmatch(r"[a-z][a-z0-9_]*", self.provider_name)
        if safe_name is None:
            raise ValueError(f"video reward provider_name must match '[a-z][a-z0-9_]*', got {self.provider_name!r}")
        for name in ("provider_revision", "score_key"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"video reward {name} must be a non-empty string, got {value!r}")
        if self.spool_dir is not None:
            if not isinstance(self.spool_dir, str):
                raise TypeError(f"video reward spool_dir must be a string or None, got {self.spool_dir!r}")
            if not self.spool_dir.strip():
                raise ValueError("video reward spool_dir must be a non-empty absolute path when configured")
            if not Path(self.spool_dir).is_absolute():
                raise ValueError(f"video reward spool_dir must be an absolute path, got {self.spool_dir!r}")
        for name in (
            "video_fps",
            "connect_timeout_s",
            "operation_timeout_s",
            "total_deadline_s",
            "poll_interval_s",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"video reward {name} must be a real number, got {value!r}")
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"video reward {name} must be finite and > 0, got {value!r}")
        if self.connect_timeout_s > self.operation_timeout_s:
            raise ValueError("video reward connect_timeout_s cannot exceed operation_timeout_s")
        if self.operation_timeout_s > self.total_deadline_s:
            raise ValueError("video reward operation_timeout_s cannot exceed total_deadline_s")
        if isinstance(self.pull_max_attempts, bool) or not isinstance(self.pull_max_attempts, int):
            raise TypeError(f"video reward pull_max_attempts must be an integer, got {self.pull_max_attempts!r}")
        if self.pull_max_attempts < 1:
            raise ValueError(f"video reward pull_max_attempts must be >= 1, got {self.pull_max_attempts}")
        for name in ("reward_clip_min", "reward_clip_max", "scale"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"video reward {name} must be a real number, got {value!r}")
            if not math.isfinite(float(value)):
                raise ValueError(f"video reward {name} must be finite, got {value!r}")
        if self.reward_clip_min > self.reward_clip_max:
            raise ValueError(
                "video reward reward_clip_min must be <= reward_clip_max, got "
                f"{self.reward_clip_min} > {self.reward_clip_max}"
            )


@dataclass
class ImageRewardConfig(VideoRewardConfig):
    """Bounded decoded-image reward contract for one rank-local T2I trajectory."""

    provider_name: str = "hpsv3"
    score_key: str = "hpsv3"
    reward_clip_min: float = -50.0
    reward_clip_max: float = 50.0


VIDEO_REWARD_REVISION_REQUIRED = "REQUIRED_DEPLOYMENT_REVISION"
"""Explicit recipe sentinel: operators must supply the immutable deployed scorer revision."""


@dataclass
class WFMLRSchedulerConfig:
    """Cosmos-RL WFM ``lambda_linear`` schedule, expressed without legacy imports.

    The schedule starts at ``f_start``, warms linearly to ``f_max``, then
    decays linearly toward ``f_min`` over one fixed cycle and holds ``f_min``
    thereafter. Keeping these knobs explicit makes effective learning rate part
    of the pinned reference-recipe comparison contract.
    """

    warm_up_steps: int = 500
    cycle_length: int = 500_000
    f_start: float = 1e-6
    f_max: float = 1.0
    f_min: float = 0.5

    def __post_init__(self) -> None:
        if self.warm_up_steps < 0:
            raise ValueError(f"WFM scheduler warm_up_steps must be >= 0, got {self.warm_up_steps}")
        if self.cycle_length <= self.warm_up_steps:
            raise ValueError(
                f"WFM scheduler cycle_length must exceed warm_up_steps, got {self.cycle_length} <= {self.warm_up_steps}"
            )
        for name in ("f_start", "f_max", "f_min"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"WFM scheduler {name} must be finite and >= 0, got {value}")
        if self.f_start > self.f_max:
            raise ValueError(f"WFM scheduler f_start must be <= f_max, got {self.f_start} > {self.f_max}")
        if self.f_min > self.f_max:
            raise ValueError(f"WFM scheduler f_min must be <= f_max, got {self.f_min} > {self.f_max}")


@dataclass
class NFTConfig(RunConfig):
    """`config.rl` for synchronous, colocated Cosmos3 online DiffusionNFT."""

    beta: float = 1.0
    kl_beta: float = 1.0
    policy_loss_scale: float = 5.0
    advantage_high: float = 5.0
    advantage_low: float = -5.0
    r_map: str = "linear"  # "linear" | "sigmoid"
    r_tau: float = 0.5
    uniform_timestep: bool = False
    modality: Literal["image", "video"] = "video"
    # Media type and qualified recipe identity are separate contracts. The native
    # video path has no named profile; image recipes select one explicitly.
    profile: str = ""
    group_size: int = 4
    sample_steps: int = 20
    sampler_shift: float = 5.0
    guidance: float = 3.0
    # Initial WFM recipe contract: sample std (correction=1) + 1e-4.
    advantage_eps: float = 1e-4
    # Model-only initialization and checkpoint lifecycle mirror the diffusion
    # trainer's DPO path. The moving old policy is checkpointed on every save;
    # the optional KL reference remains frozen at the initialization anchor.
    init_from: str = ""
    # Model-only copy-mode exceptions. Every declared source-only key must exist,
    # and exact resume never consults this list.
    init_keys_to_skip_loading: list[str] = field(default_factory=list)
    # The native default gives every source row a new sample. The qualified T2I
    # comparison recipe intentionally reuses one rank-local seed across its GA window.
    rollout_seed_clock: Literal["optimizer_step", "source_row"] = "source_row"
    save_async: bool = True
    keep_optim_last: int | None = None
    keep_model_last: int = 0
    upload: "CheckpointUploadConfig" = field(default_factory=lambda: CheckpointUploadConfig())
    old: OldPolicyConfig = field(default_factory=OldPolicyConfig)
    ref: RefConfig = field(default_factory=lambda: RefConfig(build_ref=True, update_every=0))
    scheduler: WFMLRSchedulerConfig = field(default_factory=WFMLRSchedulerConfig)
    reward: VideoRewardConfig = field(
        default_factory=lambda: VideoRewardConfig(provider_revision=VIDEO_REWARD_REVISION_REQUIRED)
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError(f"nft beta must be finite and > 0, got {self.beta}")
        if not math.isfinite(self.kl_beta) or self.kl_beta < 0:
            raise ValueError(f"nft kl_beta must be finite and >= 0, got {self.kl_beta}")
        if not math.isfinite(self.policy_loss_scale) or self.policy_loss_scale <= 0:
            raise ValueError(f"nft policy_loss_scale must be finite and > 0, got {self.policy_loss_scale}")
        if not (math.isfinite(self.advantage_low) and math.isfinite(self.advantage_high)):
            raise ValueError("nft advantage bounds must be finite")
        if self.advantage_high <= self.advantage_low:
            raise ValueError("nft advantage_high must exceed advantage_low")
        if self.r_map not in ("linear", "sigmoid"):
            raise ValueError(f"nft r_map must be 'linear' or 'sigmoid', got {self.r_map!r}")
        if self.r_map == "sigmoid" and (not math.isfinite(self.r_tau) or self.r_tau <= 0):
            raise ValueError(f"nft r_tau must be finite and > 0 for sigmoid mapping, got {self.r_tau}")
        if self.modality not in ("image", "video"):
            raise ValueError(f"nft modality must be 'image' or 'video', got {self.modality!r}")
        if not isinstance(self.profile, str):
            raise TypeError(f"nft profile must be a string, got {self.profile!r}")
        if self.profile != self.profile.strip():
            raise ValueError(f"nft profile must not contain surrounding whitespace, got {self.profile!r}")
        # The generic config reconstructor follows the default field's concrete
        # type, so an ImageRewardConfig round-trip arrives here as its
        # field-compatible VideoRewardConfig base. Modality is the explicit
        # discriminator; restore the image subtype before runtime validation.
        if self.modality == "image" and type(self.reward) is VideoRewardConfig:
            self.reward = ImageRewardConfig(**asdict(self.reward))
        if self.group_size < 2:
            raise ValueError(f"nft group_size must be >= 2, got {self.group_size}")
        if self.sample_steps < 1:
            raise ValueError(f"nft sample_steps must be >= 1, got {self.sample_steps}")
        if not math.isfinite(self.sampler_shift) or self.sampler_shift <= 0:
            raise ValueError(f"nft sampler_shift must be finite and > 0, got {self.sampler_shift}")
        if not math.isfinite(self.guidance) or self.guidance < 0:
            raise ValueError(f"nft guidance must be finite and >= 0, got {self.guidance}")
        if not math.isfinite(self.advantage_eps) or self.advantage_eps <= 0:
            raise ValueError(f"nft advantage_eps must be finite and > 0, got {self.advantage_eps}")
        if self.rollout_seed_clock not in ("optimizer_step", "source_row"):
            raise ValueError(
                f"nft rollout_seed_clock must be 'optimizer_step' or 'source_row', got {self.rollout_seed_clock!r}"
            )
        if isinstance(self.init_keys_to_skip_loading, (str, bytes)) or not isinstance(
            self.init_keys_to_skip_loading, Sequence
        ):
            raise TypeError("nft init_keys_to_skip_loading must be a list of non-empty strings")
        self.init_keys_to_skip_loading = list(self.init_keys_to_skip_loading)
        if any(not isinstance(key, str) or not key for key in self.init_keys_to_skip_loading):
            raise TypeError("nft init_keys_to_skip_loading must be a list of non-empty strings")
        if len(self.init_keys_to_skip_loading) != len(set(self.init_keys_to_skip_loading)):
            raise ValueError("nft init_keys_to_skip_loading must not contain duplicate keys")
        if not self.old.build_old:
            raise ValueError("online NFT requires rl.old.build_old=True")
        if self.old.update_every < 1:
            raise ValueError(f"online NFT requires rl.old.update_every >= 1, got {self.old.update_every}")
        if self.ref.update_every != 0:
            raise ValueError("online NFT keeps the KL reference frozen; rl.ref.update_every must be 0")
        if self.kl_beta == 0 and self.ref.build_ref:
            raise ValueError("rl.ref.build_ref=True requires kl_beta>0; set build_ref=False when kl_beta=0")
        if self.kl_beta > 0 and not self.ref.build_ref:
            raise ValueError("online NFT with kl_beta>0 requires rl.ref.build_ref=True")
        if self.max_steps > self.scheduler.cycle_length:
            raise ValueError(
                f"online NFT max_steps={self.max_steps} exceeds scheduler cycle_length={self.scheduler.cycle_length}"
            )


@dataclass
class DPOConfig(RunConfig):
    """`config.rl` for offline Diffusion-DPO: objective + optimizer + reference + data, over the shared
    RunConfig lifecycle. Model sourcing (`config.model`/`model_backend`) and orchestration
    (`config.grid`) are top-level, not here."""

    beta: float = 500.0
    pairs_per_step: int = 1
    # Init-from-base (Wallace-style DPO from a pretrained/midtrained checkpoint): load
    # ONLY net weights into policy AND frozen ref (copy_mode), no optimizer/iteration.
    # Runs on EVERY start — including resumed windows (the frozen ref is not saved in
    # checkpoints, so this is the only place it gets its anchor; the resume load then
    # overwrites net/optim/trainer state and leaves a frozen ref untouched). Accepts our
    # component/DCP layouts and cosmos-rl DCP dirs (same net.* key format). Accepts a
    # LOCAL path or `s3://bucket/prefix` — the s3 form is pre-pulled once by rank 0 into
    # a shared cache and reused by every later window (see control.checkpointing.localize_checkpoint).
    init_from: str = ""
    # Model-only copy-mode exceptions: base-checkpoint keys the DPO backbone does not build. Every
    # declared key must exist in the base, and exact resume never consults this list.
    init_keys_to_skip_loading: list[str] = field(default_factory=list)
    ref: RefConfig = field(default_factory=RefConfig)
    data: PreferenceEnvConfig = field(default_factory=PreferenceEnvConfig)
    validation: DPOValidationConfig = field(default_factory=DPOValidationConfig)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.beta <= 0:
            raise ValueError(f"dpo beta must be > 0, got {self.beta}")
        if isinstance(self.init_keys_to_skip_loading, (str, bytes)) or not isinstance(
            self.init_keys_to_skip_loading, Sequence
        ):
            raise TypeError("dpo init_keys_to_skip_loading must be a list of non-empty strings")
        self.init_keys_to_skip_loading = list(self.init_keys_to_skip_loading)
        if any(not isinstance(key, str) or not key for key in self.init_keys_to_skip_loading):
            raise TypeError("dpo init_keys_to_skip_loading must be a list of non-empty strings")
        if len(self.init_keys_to_skip_loading) != len(set(self.init_keys_to_skip_loading)):
            raise ValueError("dpo init_keys_to_skip_loading must not contain duplicate keys")
        if self.validation.enabled:
            validation_data = self.validation.data
            if not (validation_data.factory or validation_data.jsonl_path or validation_data.synthetic_pairs > 0):
                raise ValueError("rl.validation.enabled=true requires an explicit rl.validation.data preference source")
            if validation_data.shuffle:
                raise ValueError("DPO validation data must set shuffle=false")
            if validation_data.cfg_dropout_rate != 0.0:
                raise ValueError("DPO validation data requires cfg_dropout_rate=0")
            if not self.ref.build_ref:
                raise ValueError("DPO validation requires rl.ref.build_ref=true")
            same_synthetic = (
                self.data.synthetic_pairs > 0
                and validation_data.synthetic_pairs > 0
                and self.seed == self.validation.seed
            )
            same_tar = bool(
                (self.data.tar_path or validation_data.tar_path)
                and self.data.tar_path == validation_data.tar_path
                and self.data.jsonl_path == validation_data.jsonl_path
            )
            same_jsonl = bool(
                not (self.data.tar_path or validation_data.tar_path)
                and self.data.jsonl_path
                and self.data.jsonl_path == validation_data.jsonl_path
            )
            if same_synthetic or same_tar or same_jsonl:
                raise ValueError("training and validation preference sources must be distinct")


@dataclass
class DataPoolConfig:
    """Ray/NIXL page pools.

    ``page_bytes`` is the allocation and transfer granularity. Producer capacities reserve
    persistent source pages by memory tier. Cache capacities reserve persistent consumer pages.
    ``cpu_min_bytes`` keeps smaller CPU tensors in the normal payload path.
    """

    # Selects the payload transport.
    data_plane: str = "bulk_store"
    # Allocation and RDT transfer page size.
    page_bytes: int = 32 * 1024 * 1024
    # Producer-owned source-pool capacities.
    cuda_capacity_bytes: int = 1024 * 1024 * 1024
    cpu_capacity_bytes: int = 512 * 1024 * 1024
    # Consumer-owned persistent receive-cache capacities.
    cuda_cache_bytes: int = 512 * 1024 * 1024
    cpu_cache_bytes: int = 512 * 1024 * 1024
    # CPU tensors below this size bypass the page pool.
    cpu_min_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.data_plane not in ("bulk_store", "ray_nixl_pool"):
            raise ValueError(f"data_pool.data_plane must be bulk_store|ray_nixl_pool, got {self.data_plane!r}")
        for field_name in (
            "page_bytes",
            "cuda_capacity_bytes",
            "cpu_capacity_bytes",
            "cuda_cache_bytes",
            "cpu_cache_bytes",
            "cpu_min_bytes",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"data_pool.{field_name} must be a positive integer, got {value!r}")
        if self.page_bytes % 256:
            raise ValueError(
                f"data_pool.page_bytes must be a multiple of the 256-byte field alignment, got {self.page_bytes}"
            )
        for field_name in (
            "cuda_capacity_bytes",
            "cpu_capacity_bytes",
            "cuda_cache_bytes",
            "cpu_cache_bytes",
        ):
            if getattr(self, field_name) % self.page_bytes:
                raise ValueError(f"data_pool.{field_name} must be a multiple of data_pool.page_bytes")


# Accepted `GRPOConfig.routing_replay` values. Anything else is refused at construction.
# "off": natural dispatch. "rollout": Rollout Routing Replay -- the vLLM engine returns the top-k it
# used for every input position (`enable_return_routed_experts`), the rollout records it in
# `policy_info["rollout_routed_experts"]`, the packer aligns it to the policy batch and the MoE
# adapter pins every forward of that batch to it.
ROUTING_REPLAY_MODES: frozenset[str] = frozenset({"off", "rollout"})


@dataclass
class ScoreCenteringConfig:
    """`rl.score_centering`: centre the policy-gradient score under the SAMPLER's distribution.

    The rollout engine (vLLM, bf16, TP) samples completion tokens from `q`; the trainer scores them
    with its own `p`. When `q != p` the expected score `E_q[grad log p] = Σ_v q_v grad log p_v` is
    not zero, so even a constant advantage moves the policy along `q − p` (train/inference-mismatch
    drift). Score centering subtracts that expectation using the sampler's top-k head:

        grad L_t = −A · ( w_{y_t} grad log p_{y_t} − Σ_{v in H} (q_v w_v − α p_v) grad log p_v )

    with `H` the engine's top-k ids at each generated token, `ρ = q_tail / p_tail` modelling the
    unseen sampler tail as `ρ · p`, `w = f(p/q)` an optional importance weight and `α = ρ f(1/ρ)`
    (`algorithm/primitives/score_centering.py`, the reference's `score_centering_loss`).

    `enabled`    — False (DEFAULT): byte-identical to before this block existed; the engine returns
                   only the sampled token's logprob. True: the vLLM rollout is built with
                   `logprobs_mode="processed_logprobs"` and `max_logprobs=top_k`, asks for
                   `SamplingParams(logprobs=top_k)` and stamps `policy_info["rollout_topk_ids"]`
                   (`[T, k]` ids) and `policy_info["rollout_topk_logprobs"]` (`[T, k]` float32 log q)
                   on every training completion; the clipped objective gathers the trainer's log p at
                   those ids and adds the correction to the surrogate. PROCESSED logprobs are
                   `log_softmax(logits / rollout.temperature)` — the distribution the token was drawn
                   from — so with the on-policy contract (`rollout.temperature == rl.temperature`, no
                   top-p/top-k/penalty warp) `q` and the trainer's tempered `p` describe the same
                   temperature. Consequence for diagnostics: `train/logprob_gap_*` then compares the
                   TEMPERED trainer logprob against the engine's tempered one (not temperature-1).
    `top_k`      — head width `k` (128 default; the reference matched full centering at 32 and 128).
                   Memory on the wire: `T × k × (4 + 4)` bytes per completion (float32 log q + int32 ids; float32 is deliberate — a float16 head made q != p for an identical sampler and trainer, see `schema/trajectory.py`).
    `compose`    — `none` (DEFAULT): pure score centering, `f = 1`. `tis`: truncated importance
                   sampling, `f(r) = min(r, tis_clip)`. `mis`: masked IS, `f(r) = r` inside
                   `[mis_low, mis_high]`, else 0. The sampled-token weight uses the TRUE `q_{y_t}`
                   (the engine's logprob of the sampled token), the reference's shipped choice.
    `tis_clip`   — TIS ceiling (2.0). Must be >= 1 so `f(1) = 1`.
    `mis_low` / `mis_high` — MIS band (0.5 / 5.0). Must contain 1.

    Consumed by `grpo_surrogate="clipped"` at `n_mini == 1` with token-level importance sampling and
    `use_rollout_logprobs_for_loss=false` only; every other combination is refused in
    `GRPOConfig.__post_init__` (two off-policy corrections on the same tokens are not composed here).
    """

    enabled: bool = False
    top_k: int = 128
    compose: str = "none"  # none|tis|mis; OmegaConf structured configs reject Literal
    tis_clip: float = 2.0
    mis_low: float = 0.5
    mis_high: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError(f"score_centering.enabled must be a bool, got {self.enabled!r}")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError(f"score_centering.top_k must be a positive integer, got {self.top_k!r}")
        if self.compose not in ("none", "tis", "mis"):
            raise ValueError(f"score_centering.compose must be none|tis|mis, got {self.compose!r}")
        for name in ("tis_clip", "mis_low", "mis_high"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise TypeError(f"score_centering.{name} must be a finite non-bool real, got {value!r}")
        if float(self.tis_clip) < 1.0:
            raise ValueError(f"score_centering.tis_clip must be >= 1 so that f(1) == 1, got {self.tis_clip!r}")
        low, high = float(self.mis_low), float(self.mis_high)
        if not (0.0 <= low <= 1.0 <= high) or low >= high:
            raise ValueError(
                "score_centering.mis_low/mis_high must satisfy 0 <= low <= 1 <= high and low < high (the band "
                f"must contain the on-policy ratio 1), got {self.mis_low!r}/{self.mis_high!r}"
            )


@dataclass
class GRPOConfig(HFRunConfig):
    """`config.rl` for online GRPO (LLM/VLM), over the shared RunConfig lifecycle plus the
    `HFRunConfig` precision block (GRPO's backend is `hf`, so it owns the storage/compute dtype
    split — see that class for why precision sits beside the optimizer rather than on
    `config.model_hf`). The model is
    `config.model_hf` (top-level, backend=hf) and the vLLM rollout engine + sampling is `config.rollout`
    (`RolloutConfig`) — neither is an rl field. Holds only GRPO's own loop knobs. Checkpoint lifecycle
    controls (`save_async`, `keep_*`, `checkpoint_store`, and `upload`) come from `RunConfig`, so the LLM and diffusion
    trainers share the same `rl.*` schema. `init_from` remains DPO-only because it initializes both the
    policy and frozen reference rather than controlling the checkpoint lifecycle."""

    group_size: int = 8
    max_operand: int = 50  # ArithmeticEnv operand bound (the legacy policy-gradient prompt source)
    # Clipped-objective knobs, flat beside the loop knobs above (cosmos-rl keeps these flat under
    # [train.train_policy] too). They are the PPO-family surrogate shared by any clipped policy-
    # gradient method; `group_size` above is the GRPO-specific part (group-relative advantage).
    # Dtype for the log_softmax that produces per-token logprobs, INDEPENDENT of the model's compute
    # dtype. Mirrors cosmos-rl `train.logprob_dtype`, whose default is likewise float32 — so a bf16
    # model still scores in fp32 there. The clipped surrogate's ratio is a DIFFERENCE of two
    # logprobs, and bf16 cannot resolve the deviations that decide whether a token crosses the
    # epsilon boundary: measured at n_mini=8, bf16 logprobs made the clip engage on 0.089% of tokens
    # against 1.49% in fp32. Only matters at n_mini > 1 -- at n_mini == 1 the ratio is identically 1.
    logprob_dtype: str = "float32"

    epsilon_low: float = 0.2  # PPO clip lower bound: ratio clamped to [1-epsilon_low, ...]
    epsilon_high: float = 0.2  # PPO clip upper bound: ratio clamped to [..., 1+epsilon_high]
    # Reference-anchored KL penalty weight for the CLIPPED surrogate (0 = off, plain on-policy GRPO).
    # Positive values add `kl_beta * k3(logπ_ref − logπ_θ)` to every loss-masked completion token's
    # loss (DeepSeekMath GRPO eq. 4; `algorithm/primitives/policy_gradient.py`), pulling the policy
    # back toward a FROZEN reference — the weights the run started from, rebuilt as a sharded
    # gradient-free twin of the policy on every start (`algorithm/grpo.py::FrozenReference`, wired by
    # `entrypoints/grpo.py::build_trainer`). Requires `ref.build_ref=true` below, which costs one
    # extra copy of the network per policy rank.
    kl_beta: float = 0.0
    # Bound, in nats, on the KL term's per-token GRADIENT coefficient. None (default) = the exact k3
    # gradient `1 − exp(z)`, byte-identical to before this field existed. That coefficient is
    # unbounded below inside the ±20 nat log-ratio clamp (−4.85e8 just inside it), so a single token
    # whose reference forward disagrees with the policy forward by a few nats owns the whole step:
    # Routing nondeterminism can make a small number of token coefficients dominate a step. With
    # kl_coeff_clamp=c the penalty's VALUE (and every train/kl_* statistic) stays the exact k3 while
    # its gradient becomes that of k3's linear extension beyond |z| = c, i.e. the coefficient
    # `1 − exp(clamp(z, −c, c))`, bounded by exp(c) − 1 (5.0 recommended: |coef| ≤ 147). Must be
    # positive and below the ±20 clamp (`KL_LOG_RATIO_CLAMP_NATS`), which already zeroes the gradient
    # beyond it; requires kl_beta > 0. Keyword-only like `importance_sampling_level` below.
    kl_coeff_clamp: float | None = field(default=None, kw_only=True)
    # Inner PPO epochs per batch. PINNED AT 1 (pure on-policy: pi_old == pi, so the ratio is
    # identically 1 and the clip never binds). Ported from the cosmos-rl TOML for provenance and
    # W&B run config; >1 is REJECTED in __post_init__ until a driver re-uses a batch across
    # epochs -- the surrogate supports it, nothing calls it that way.
    mu_iterations: int = 1
    # Policy-loss (logit) temperature: divides the training-forward logits before log_softmax in the
    # objective. SEPARATE from RolloutConfig.temperature (the rollout SAMPLING temperature): the
    # fields are not auto-synchronized, while objective-specific preflight requires equality
    # for on-policy GRPO (the cosmos-rl reference TOML uses 0.9 for each).
    temperature: float = 1.0
    # SHARED-PREFIX loss filtering (cosmos-rl `min_filter_prefix_tokens`, same None default).
    # This is a THRESHOLD, not a count: within a group, maximal shared prefixes of at least this
    # many tokens are found, and where such a prefix set holds DIFFERENT rewards its tokens are
    # masked out of the loss — the prefix cannot explain the reward spread, so its advantage is
    # noise. The number of tokens actually skipped is COMPUTED per completion (the shared-prefix
    # length), never this value.
    #
    # A previous version of this port had `n_ignore_prefix_tokens: int = 0` here and mapped the
    # TOML's `min_filter_prefix_tokens = 1` onto it as a constant, which unconditionally deleted the
    # FIRST token of every completion from the loss mask — including the `####` marker the GSM8K
    # verifier keys on. See algorithm/annotators/prefix_filter.py.
    min_filter_prefix_tokens: int | None = None
    # π_old source for the clipped ratio. FALSE (default) = recompute from the CURRENT policy
    # (`cur.detach()`), which is cosmos-rl's default (`use_rollout_logprobs_for_loss=False`) and
    # makes the on-policy ratio exactly exp(0)==1. TRUE = use the rollout engine's own logprobs
    # (`policy_info["rollout_logprobs"]`), cosmos-rl's opt-in mode.
    #
    # Default False deliberately. The vLLM rollout records RAW logprobs (log-softmax of the model
    # logits at temperature 1, before any top-k/top-p warp), so they are log π_old only when nothing
    # warps sampling: on top of the top-p/top-k/repetition-penalty rejection every on-policy GRPO
    # run gets, the temperature contract (`control/temperature.py`) requires
    # rollout.temperature == rl.temperature == 1.0 in this mode (the HF rollout's processed scores
    # coincide with raw logprobs under it). Even then the ratio pairs the
    # inference engine's forward with the training forward, so the train/infer gap
    # (`train/logprob_gap_*`) enters every ratio. cosmos-rl, when it does use them, does NOT put
    # them in the ratio denominator: it keeps them as a separate behavioural-KL importance weight
    # bounded by `behav_imp_weight_cap`. We have no such cap, so the honest default is off.
    use_rollout_logprobs_for_loss: bool = False
    # Group-advantage variant. FALSE (default) = (r - mean)/(std + eps), standard GRPO. TRUE =
    # mean-centre only (r - mean), skipping the std normalization — cosmos-rl's `GRPO(unbiased=...)`
    # (`dispatcher/algo/grpo.py`), same default. Named `unbiased_advantage` rather than cosmos-rl's
    # bare `unbiased` because these knobs are FLAT here: next to `temperature`/`kl_beta` a lone
    # `unbiased` does not say unbiased *what*.
    #
    # Dividing by the group std rescales every group to unit variance, so a barely-disagreeing group
    # (one lucky success, std ~ 0) gets an advantage as large as a strongly-disagreeing one —
    # inflating the least informative groups (the Dr-GRPO critique). Worth trying if early updates
    # are unstable.
    unbiased_advantage: bool = False
    # DUAL-CLIP lower bound (cosmos-rl `lower_bound_ratio`, same default 3.0). Bounds the
    # NEGATIVE-advantage branch of the clipped surrogate at `lower_bound_ratio * |adv|`; for adv>=0
    # and on-policy (ratio==1) it is inert. 0 disables it, recovering the single-clip form.
    # Without it a large ratio on a negative advantage makes the penalty unbounded — one badly
    # off-policy token can dominate the batch (measured: 100.43 vs cosmos-rl's 15.0 at adv=-5).
    lower_bound_ratio: float = 3.0
    # GRANULARITY of the clipped surrogate's importance ratio (TRL's name for the same knob):
    #   "token"    (DEFAULT) — one ratio per token, r_t = exp(logπ_θ(y_t) − logπ_old(y_t)): GRPO/PPO,
    #                          byte-identical to before this field existed.
    #   "sequence" — one ratio per response, s_i = exp(mean_t(logπ_θ − logπ_old)) over the response's
    #                loss-masked tokens, shared by all of them so the epsilon clip admits or drops
    #                WHOLE responses: GSPO (arXiv 2507.18071), in the GSPO-token form of
    #                `algorithm/primitives/policy_gradient.py::sequence_log_ratio`.
    # Everything else about the clipped surrogate is inherited (frozen π_old, epsilon clip,
    # temperature, loss_type, fp32 logprobs). What sequence level rejects (dual clip, per-token
    # advantages) and warns about (an inert ratio at n_mini == 1, token-scale epsilons, bf16
    # logprobs) is enforced in __post_init__ below and documented in docs/reference/config.md. Keyword-only so
    # the option does not shift the positional constructor contract of the fields above (same
    # convention as `weight_sync_r2r_broadcast`).
    importance_sampling_level: str = field(default="token", kw_only=True)
    # Frozen KL reference for `kl_beta > 0`: the shared `RefConfig` contract DPO/NFT use plus GRPO's
    # own warm-restart anchor (`GRPORefConfig.model_dir`; a plain `RefConfig` is adopted). OFF by
    # default: an unanchored run is what every earlier GRPO checkpoint trained as, and the twin is
    # not free. Tied to `kl_beta` both ways in __post_init__ (build_ref without kl_beta wastes a
    # model copy; kl_beta without build_ref cannot be computed), and `update_every` must be 0: the
    # anchor is the initial policy, never refreshed — unless `ref.model_dir` names the checkpoint to
    # anchor to (warm restart from a model-only save: policy from the save, reference from the
    # original base). Keyword-only for the same reason as the field above.
    ref: GRPORefConfig = field(default_factory=lambda: GRPORefConfig(build_ref=False, update_every=0), kw_only=True)
    # Prompt source, named `data` to match DPOConfig.data so both algorithms expose their source at
    # the same `rl.data` path (the TYPE differs per algorithm — DatasetConfig here, PreferenceEnvConfig
    # there — but the slot does not). Unset keeps the v0 ArithmeticEnv; set `path` to read prompts
    # from a JSONL file.
    data: DatasetConfig = field(default_factory=DatasetConfig)
    validation: GRPOValidationConfig = field(default_factory=GRPOValidationConfig)
    # Which seed orders the prompt source. False keeps `rl.seed`: the cursor permutes the corpus
    # with `rl.seed + epoch` (`roles/env/cursor.py::IndexedCursor`) and the arithmetic source draws
    # from `rl.seed`, while vLLM samples under the separate top-level `rollout.seed`. A launcher that
    # replicates an arm by varying only `rollout.seed` therefore feeds every replicate the IDENTICAL
    # prompt sequence — the replicates differ in sampling alone — and a continuation pool that drops
    # a FILE prefix of such a run removes rows the parent never consumed in that order. True keys
    # the prompt source on `rollout.seed` (`resolve_data_seed`), so one per-arm seed varies the
    # prompt order and the sampling together; requires an integer `rollout.seed`. Resolved once at
    # config time — preflight (`config/build.py`) and the data role (`roles/env/prompt.py`) call the
    # same function — and recorded in the cursor checkpoint, so an exact resume refuses a flipped
    # value: set it from the first step of a run.
    data_seed_follows_rollout_seed: bool = False
    # The GRPO policy-gradient surrogate, deliberately distinct from OPD's ``distillation_mode``:
    #   "legacy_pg" — the v0 group-baseline policy gradient, with no ratio, clip or KL.
    #   "clipped"   — the production surrogate: frozen π_old ratio, epsilon/dual clip and optional KL.
    # Production defaults to clipped. The arithmetic quick-start pins legacy_pg in config_grpo.py
    # because that recipe intentionally demonstrates the minimal policy-gradient path.
    grpo_surrogate: str = "clipped"
    # Loss reduction for the CLIPPED objective, mirroring cosmos-rl's `loss_type`
    # (grpo_trainer.py:310-345), same default:
    #   "token-mean" (DEFAULT) — sum(per-token loss) / DP-GLOBAL token count. Every TOKEN weighted
    #                            equally, so a long completion contributes proportionally more.
    #   "seq-mean-token-sum"   — mean over sequences of each sequence's token SUM.
    #   "seq-mean-token-mean"  — mean over sequences of each sequence's token MEAN (every SEQUENCE
    #                            weighted equally regardless of length).
    #   "seq-mean-token-sum-fixed" — Dr-GRPO: mean over sequences of each sequence's token SUM,
    #                            divided by the fixed rollout.max_tokens response bound.
    # Not cosmetic: on the 4..160-token spread a real GSM8K run produces, token-mean and
    # seq-mean-token-mean differ by ~1.4x. All modes defer their DP-global token/sequence divisor
    # across micro-batches.
    loss_type: str = "token-mean"

    # Reward provider scoring the completions. Built-ins live in algorithm.reward; a recipe may add
    # providers through `reward_provider_catalog` without changing the framework package.
    reward: str = "arithmetic"
    # Optional dotted path to a zero-argument callable returning
    # `Mapping[str, RewardProviderSpec] | RewardProviderCatalog`. It is resolved independently on
    # the driver and each Ray actor from the same persisted string, then merged with built-ins;
    # collisions fail instead of silently replacing framework behavior.
    reward_provider_catalog: str = ""

    # The online trainer→rollout weight plane is shared by GRPO and OPD. Placement is selected
    # independently through ``grid.gpu_placement``; ``nixl`` serves its disaggregated cross-device
    # relationship, while an unset hint preserves the colocated IPC/default-host paths.
    weight_sync_transport: str | None = None
    weight_sync_bucket_bytes: int | None = None
    weight_sync_probe_bytes: int = 256 * 1024 * 1024
    weight_sync_min_gbps: float = 20.0

    # On a colocated card (``grid.gpu_placement=colocated``) the loop hands the GPU between the two
    # bodies every step: the vLLM engine sleeps while the trainer updates and the trainer parks its
    # state on the host while the engine generates (``control/loop/offload.py``). False keeps both
    # resident: the engine is built without sleep mode and no per-step sleep/wake or offload/onload
    # runs, trading GPU memory for step time (the cycle costs seconds per step on a large model
    # whose card was never memory-bound). Only valid when engine and trainer fit side by side, which
    # the existing bind-time memory checks and CUDA OOM surface; inert under ``disaggregated``, where
    # nothing is shared and no hand-off runs anyway. Keyword-only for the same positional-contract
    # reason as ``weight_sync_r2r_broadcast``.
    colocated_offload: bool = field(default=True, kw_only=True)

    # Replica-global batching (spec §5). These describe the sole policy execution replica's step and are
    # independent of how many DP ranks that replica has — a user sizing a run should not have to
    # restate the batch when the mesh changes shape.
    #
    # `batch_per_replica` is the retained public name for the sole policy execution replica's global
    # trajectory batch. The total prompt count derives from it; runtime then divides prompts across
    # rollout/data execution replicas. It is never per HSDP dp_replicate row.
    batch_per_replica: int = 32
    mini_batch_per_replica: int = 0  # per optimizer step; 0 ⇒ the whole batch (today)
    micro_batch_per_replica: int = 0  # per forward;        0 ⇒ one chunk    (today)
    # Data-plane backend. ``bulk_store`` retains the existing behavior. ``ray_nixl_pool`` stages
    # CUDA tensors and large CPU tensors into producer-owned RDT/NIXL pages and releases their
    # leases after the complete synchronous training step.
    data_pool: DataPoolConfig = field(default_factory=DataPoolConfig)

    # DAPO-style DYNAMIC SAMPLING (arXiv 2503.14476 §3.2, "uniform-group filtering"). A GRPO group
    # whose rewards all tie has zero group-relative advantage for every member: it contributes no
    # policy gradient but still dilutes the batch (and, under token-count normalization, shrinks
    # the effective step). The reward stage stamps every complete group's reward spread
    # (`algorithm.annotators.advantage.INFORMATIVE_GROUP_GATE`); the replay buffer gates on that
    # stamp alone, so the drop decision is per GROUP and placement-independent. The loop keeps
    # drawing whole prompt batches until the replay cells together hold `min_informative_groups`
    # informative groups — a GLOBAL count that `control.loop.batching.informative_quota_per_cell`
    # splits exactly across the data cells (the first N mod R cells owe one group more), so the
    # requirement does not grow with the replica count — then assembles the batch informative
    # groups first at its configured size, so every mini/micro
    # divisibility invariant holds. Tied groups fill the remainder whenever
    # `min_informative_groups` is below the batch's group capacity `batch_per_replica // group_size`;
    # full DAPO — an all-informative batch — is `min_informative_groups = batch_per_replica // group_size`.
    #
    #   min_informative_groups = 0 (default)  — feature OFF, collection loop unchanged.
    #   min_informative_groups = N > 0        — keep drawing until N informative groups are buffered.
    min_informative_groups: int = 0
    # The count is GLOBAL and placement-independent, which also means it does not scale with the
    # batch: `groups_per_step = batch_per_replica // group_size` is the capacity it is a share of, and
    # a literal tuned for one layout (8 = every group of b32g4) copied onto a wider one (8 of the 64
    # groups of b256g4) quietly turns dynamic sampling into a 12% floor while the run still reports it
    # ON (a literal below half of `groups_per_step` therefore warns). `min_informative_groups_frac`
    # spells the same quota as a share of `groups_per_step`, so it survives a change of batch layout;
    # `effective_min_informative_groups` resolves the pair into the ONE count the collection loop reads.
    #   min_informative_groups_frac = None (default) — the literal count above is the quota.
    #   min_informative_groups_frac = f, 0 < f <= 1  — quota = ceil(f * groups_per_step); 1.0 is full
    #                                                  DAPO on any layout. The literal must stay 0.
    min_informative_groups_frac: float | None = None
    # Top-up budget in extra ROUNDS per loop step (a round = one more draw of `prompts_per_data_cell`
    # prompts on each participating cell). How many rounds one `rollout.generate` call carries is
    # `dynamic_sampling_refill_dispatch` below; the calls a step issued are counted by
    # `train/grpo_dynamic_sampling_refill_dispatches`. Only consulted when the resolved quota
    # `effective_min_informative_groups` is > 0. Bounded so a
    # reward regime with NO spread (all-zero rewards on a too-hard task) cannot become an unbounded
    # generation loop; what happens at the bound is the knob below.
    dynamic_sampling_max_extra_rollouts: int = 4
    # all (default) refills every cell; short_cells idles cells that already meet their quota.
    # The initial draw remains full everywhere. This changes later source/engine histories and,
    # at fractional quotas, can retain tied filler that full refills would replace. It is an
    # opt-in collection policy, not a promise of identical batches or learning curves.
    dynamic_sampling_refill_mode: str = field(default="all", kw_only=True)
    # Where the resolved quota is enforced.
    #   cell   (default) — every replay cell must buffer its exact share of the quota
    #                      (`informative_quota_per_cell`), so particular cells owe particular groups.
    #   pooled           — stop once the batch assembled informative-first holds the quota: each cell
    #                      contributes at most its batch share, so it holds sum(min(informative, share))
    #                      informative trajectories. Refills target cells whose share is not yet
    #                      informative. With one prompt per cell (one data cell per rollout replica,
    #                      e.g. 256 TP4 engines at batch 1024), cell scope spends the whole budget
    #                      whenever any owing cell keeps drawing tied prompts.
    # Both scopes enforce the same minimum quota on the trained batch. The informative-group count it
    # retains and its composition can differ: cell scope keeps refilling owing cells after the
    # assembled batch already holds the quota, and pooled scope does not fix which cells supply it.
    dynamic_sampling_quota_scope: str = field(default="cell", kw_only=True)
    # How many top-up rounds one `rollout.generate` call carries. This is orthogonal to
    # `dynamic_sampling_refill_mode`, which decides which cells participate in that call.
    #   per_round    (default) — one round per call; re-count the short cells after every round.
    #   whole_budget           — all remaining rounds in one call. Each participating cell serves
    #                            `remaining_rounds * prompts_per_data_cell` prompts; cells idled by
    #                            `short_cells` serve none. Selection remains informative-first in draw order.
    # The wider call can advance the cursor farther and replace tied filler below full DAPO when
    # `per_round` would have stopped early; the field is therefore an explicit collection policy.
    dynamic_sampling_refill_dispatch: str = field(default="per_round", kw_only=True)
    # What the step does when the budget is spent and a replay cell is still short of its quota.
    #   raise   (default) — abort the run: the paper trains on informative groups only, and a
    #                       reward regime that cannot supply them within budget is a recipe fault
    #                       (reward too hard/easy, group_size too small), not something to train through.
    #   proceed           — train anyway with tied groups as zero-advantage filler and log the
    #                       shortfall; for users who need a fixed batch shape more than the guarantee.
    dynamic_sampling_on_budget_exhausted: str = "raise"  # proceed|raise; OmegaConf structured configs reject Literal
    # DAPO "overlong filtering" (arXiv 2503.14476 §3.3). A completion the engine cut at its token
    # budget (`rollout.max_tokens`; the vLLM rollout stamps `policy_info["finish_reason"] ==
    # "length"`, `schema.FINISH_REASON_KEY`) carries a reward for an answer it never finished, and
    # under the token-sum reductions its `max_tokens` tokens outweigh a median response many times
    # over (3072 tokens against a ~80-token median is ~39x the per-response gradient mass).
    #   False (DEFAULT) — it trains like any other completion, byte-identical to before this field.
    #   True            — the clipped objective zeroes its loss mask, so it is absent from the
    #                     surrogate, the KL term and their token/sequence divisors alike
    #                     (`algorithm/grpo.py::make_clipped_objective`).
    # `train/rollout_truncated_frac`, `train/response_len_mean` and `train/response_len_max` report
    # the rate either way. Consumed by `grpo_surrogate=clipped` only; legacy_pg and OPD refuse it.
    # Keyword-only so the field does not shift the positional constructor contract (the same
    # convention as `importance_sampling_level`).
    mask_truncated_completions: bool = field(default=False, kw_only=True)
    # How the recipe-hygiene checks report. Two pairings are checked at construction: a literal
    # `min_informative_groups` below half of `groups_per_step` (dynamic sampling in name only on a
    # wider layout) and `loss_type="seq-mean-token-sum-fixed"` paired with an `optimizer.eps` above
    # FIXED_NORMALISER_OPTIMIZER_EPS_CEILING (the reported lr is not the step size). Both are
    # honored as written, so the repo rule for un-honored knobs (reject) does not apply.
    #   "warn"  (DEFAULT) — each hazard WARNS at construction; existing fields and default runtime
    #                       behavior remain unchanged.
    #   "error"           — every hazard found is collected and refused in ONE ValueError listing
    #                       them all, for a recipe that wants a mis-pairing caught at preflight
    #                       (`--validate-manifest` re-instantiates `rl` after the dotted overrides)
    #                       rather than logged. `the Dr-GRPO recipe layer`, the layer that
    #                       selects the fixed reduction, sets it.
    # The checks, thresholds and messages are the same in both modes (RECIPE_GUARD_SEVERITIES). OPD
    # inherits the field but refuses both pairings outright, so it has nothing to do there.
    # Keyword-only so the field does not shift the positional constructor contract.
    recipe_guard_severity: str = field(default=RECIPE_GUARD_WARN, kw_only=True)
    # Score centering of the policy gradient under the SAMPLER's distribution (`ScoreCenteringConfig`
    # above). OFF by default; on, the vLLM rollout records its top-k head per generated token and the
    # clipped objective subtracts the sampler-expected score. Keyword-only like the fields above.
    score_centering: ScoreCenteringConfig = field(default_factory=ScoreCenteringConfig, kw_only=True)

    @property
    def prompts_per_data_replica(self) -> int:
        """Single-data-replica prompt count derived from ``batch_per_replica``.

        Policy execution R is one. A caller with multiple rollout/data execution replicas must use
        ``control.loop.batching.prompts_per_data_cell`` to divide this total prompt count across
        them. The property remains the exact R_data=1 convenience and compatibility surface.

        Returns:
            `batch_per_replica // group_size`.
        """
        return self.batch_per_replica // self.group_size

    @property
    def groups_per_step(self) -> int:
        """Prompt groups the trained batch holds per loop step: ``batch_per_replica // group_size``.

        Placement-independent: ``batch_per_replica`` is the sole policy replica's GLOBAL trajectory
        batch, which ``control.loop.batching.prompts_per_data_cell`` divides across the rollout/data
        cells, so more cells means a smaller share per cell, never more groups per step. This is the
        capacity ``min_informative_groups`` is a share of — the same number as
        ``prompts_per_data_replica``, under the name the dynamic-sampling share is spelled in, and
        defined through it so the two cannot drift.
        """
        return self.prompts_per_data_replica

    @property
    def effective_min_informative_groups(self) -> int:
        """The informative-group quota the collection loop enforces (GLOBAL, in groups).

        ``min_informative_groups_frac`` set ⇒ ``max(1, ceil(frac * groups_per_step))``, which lies in
        ``[1, groups_per_step]`` for ``0 < frac <= 1``; otherwise the literal ``min_informative_groups``
        (0 ⇒ dynamic sampling OFF). The fraction is read as the nearest rational with denominator
        ≤ 10^6 before the ceil, so a share that names a whole number of groups is exactly that
        number: 0.14 of 50 groups is 7, where the float product ``0.14 * 50 = 7.000000000000001``
        would ceil to 8. The floor of 1 keeps the promise that a set fraction is ON: that rational
        read rounds a share below 5e-7 to 0, whose ceil would silently switch dynamic sampling OFF
        while the config says otherwise. This is the ONE place the pair is resolved:
        ``entrypoints/grpo.py`` reads only this, and the stored fields keep what the author wrote so
        the YAML round trip carries the intent, not the arithmetic.
        """
        if self.min_informative_groups_frac is None:
            return int(self.min_informative_groups)
        share = Fraction(float(self.min_informative_groups_frac)).limit_denominator(1_000_000)
        return max(1, math.ceil(share * self.groups_per_step))

    @property
    def n_mini(self) -> int:
        """Optimizer steps per loop step. DP-INVARIANT, so every rank agrees without communicating.

        Returns:
            `batch_per_replica // mini_batch_per_replica`, or 1 if `mini_batch_per_replica` is unset.
        """
        return self.batch_per_replica // self.mini_batch_per_replica if self.mini_batch_per_replica else 1

    # EXPLICIT operator attestation that this model's forward is deterministic given its weights.
    # Required for n_mini > 1, and False by default, so the ahead pass is FAIL-CLOSED for an
    # unvetted model rather than relying on a screen that cannot certify one.
    #
    # WHY AN ATTESTATION AND NOT A CHECK. `LLMTrainer` is model-agnostic. The two automated screens
    # (a named-stochasticity scan and an empirical eval-vs-train probe, both in roles/trainer/llm.py)
    # catch the common cases, but neither can PROVE determinism for an arbitrary model: a layer that
    # is stochastic only on inputs the probed chunk does not exercise passes both. Retiring a
    # general-model safety guard on one sampled comparison is not a correctness contract, so the
    # contract is stated by whoever knows the model, and the screens become defence in depth.
    #
    # Setting this on a model with live dropout does NOT make the run correct — it makes it your
    # claim. Verify with the model's config (Qwen2.5 reports attention_dropout 0.0) before setting it.
    certified_deterministic_forward: bool = False
    # Optional relay mode: existing P2R loads only rollout replica 0, then corresponding live
    # rollout ranks broadcast in place across replicas over independent NCCL groups. Keyword-only
    # so adding the option does not shift the positional constructor contract of existing fields.
    weight_sync_r2r_broadcast: bool = field(default=False, kw_only=True)
    # Sparse-MoE ROUTING REPLAY (Rollout Routing Replay, Ma et al., arXiv 2510.11370; DeepSeek's
    # production variant is "Keep Routing"). On an MoE policy the router's top-k selection over
    # near-tied scores can vary across otherwise identical forwards, so the eval-vs-train determinism probe
    # refuses the π_old ahead pass at n_mini > 1 and, past it, the importance ratio would measure
    # routing churn instead of policy change. Replay pins the dispatch to a recorded selection
    # through the model's `ForwardReplayPolicy` capability (`model/protocol.py`); the router still
    # trains through the combine weights. "rollout" replays the INFERENCE engine's routing
    # (`ROUTING_REPLAY_MODES`): valid at any n_mini, including 1, where it closes the train/infer
    # routing gap that no π_old pass can, and together with `use_rollout_logprobs_for_loss`. It needs
    # a policy that can replay (assembly refuses dense/HF backends), the vLLM rollout body and the
    # clipped surrogate: only that objective prepares its batches through the model capability
    # (`prepare_policy_capability`); `legacy_pg` forwards `net(ids)` directly, so nothing would pin
    # and the run would die at its first forward instead of here.
    # Keyword-only for the same positional-contract reason as `weight_sync_r2r_broadcast`.
    routing_replay: str = field(default="off", kw_only=True)

    @property
    def needs_cached_old_logprobs(self) -> bool:
        """Whether the π_old ahead pass must run for this config — the ONE predicate, read by both
        the driver (`entrypoints/grpo.py`) and the bulk reader accounting (`workers/grid.py`).

        Mirrors cosmos-rl's `need_compute_old_ahead` EXACTLY, both clauses:
        `batch_size > per_optimize_batch_size and not use_rollout_logprobs_for_loss`
        (`policy/trainer/llm_trainer/grpo_trainer.py`).

        The second clause is not an optimization. Under `use_rollout_logprobs_for_loss` the objective
        takes the `_gather_old` branch and never touches `OldLogprobCache`, so an ahead pass there
        would be a wasted forward AND would apply the eval/train determinism guard to a mode that
        never uses an eval-mode π_old — rejecting a dropout-enabled model that is perfectly valid.

        It lives here, as one property, because the two call sites CANNOT diverge safely: the driver
        decides whether the batch is materialized twice, and grid.py decides the refcount that pays
        for it. Gate them differently and either the store pops mid-window (KeyError) or entries
        never reach zero and LEAK.

        Returns:
            True when π_old must be recorded ahead of the optimizer-step window.
        """
        return self.n_mini > 1 and not bool(getattr(self, "use_rollout_logprobs_for_loss", False))

    @property
    def n_micro(self) -> int:
        """Forward/backward chunks per optimizer step. Also the trainer's `grad_accum` (§5.2).

        Returns:
            `mini // micro_batch_per_replica` (where `mini` is `mini_batch_per_replica` or, if
            unset, the whole `batch_per_replica`), or 1 if `micro_batch_per_replica` is unset.
        """
        mini = self.mini_batch_per_replica or self.batch_per_replica
        return mini // self.micro_batch_per_replica if self.micro_batch_per_replica else 1

    def _objective_consults_pi_old(self) -> bool:
        """Does this config's objective form an importance ratio against a frozen π_old?

        Only such an objective runs the ahead pass, and only the ahead pass needs a deterministic
        forward — so only it justifies demanding the attestation. For GRPO that is the CLIPPED
        surrogate alone: `grpo_surrogate="legacy_pg"` is a plain policy gradient with no ratio and
        no cache (`config/build.py` hands it `make_objective`), and the driver already skips the
        ahead pass for it via `policy.needs_old_logprobs()`.

        `OPDConfig` overrides this again for its own two modes.

        Returns:
            True when the objective scores against a cached π_old.
        """
        return self.grpo_surrogate == "clipped"

    def __post_init__(self) -> None:
        if not isinstance(self.reward_provider_catalog, str):
            raise TypeError(
                f"reward_provider_catalog must be a dotted callable path string, got {self.reward_provider_catalog!r}"
            )
        if self.weight_sync_transport not in (None, "nixl"):
            raise ValueError(f"weight_sync_transport must be None|'nixl', got {self.weight_sync_transport!r}")
        if not isinstance(self.weight_sync_r2r_broadcast, bool):
            raise TypeError(f"weight_sync_r2r_broadcast must be bool, got {self.weight_sync_r2r_broadcast!r}")
        if not isinstance(self.data_seed_follows_rollout_seed, bool):
            raise TypeError(f"data_seed_follows_rollout_seed must be bool, got {self.data_seed_follows_rollout_seed!r}")
        if not isinstance(self.recipe_guard_severity, str):
            raise TypeError(f"recipe_guard_severity must be str, got {self.recipe_guard_severity!r}")
        if self.recipe_guard_severity not in RECIPE_GUARD_SEVERITIES:
            raise ValueError(f"recipe_guard_severity must be 'warn' or 'error', got {self.recipe_guard_severity!r}")
        if not isinstance(self.colocated_offload, bool):
            raise TypeError(f"colocated_offload must be bool, got {self.colocated_offload!r}")
        if self.weight_sync_bucket_bytes is not None and (
            isinstance(self.weight_sync_bucket_bytes, bool)
            or not isinstance(self.weight_sync_bucket_bytes, int)
            or self.weight_sync_bucket_bytes < 1
        ):
            raise ValueError(
                "weight_sync_bucket_bytes must be None or a non-bool positive integer, got "
                f"{self.weight_sync_bucket_bytes!r}"
            )
        if (
            isinstance(self.weight_sync_probe_bytes, bool)
            or not isinstance(self.weight_sync_probe_bytes, int)
            or self.weight_sync_probe_bytes < 1
        ):
            raise ValueError(
                f"weight_sync_probe_bytes must be a non-bool positive integer, got {self.weight_sync_probe_bytes!r}"
            )
        if (
            isinstance(self.weight_sync_min_gbps, bool)
            or not isinstance(self.weight_sync_min_gbps, Real)
            or not math.isfinite(float(self.weight_sync_min_gbps))
            or self.weight_sync_min_gbps < 0
        ):
            raise ValueError(
                f"weight_sync_min_gbps must be a non-bool finite real >= 0, got {self.weight_sync_min_gbps!r}"
            )
        if float(self.temperature) <= 0.0:
            # The policy-loss temperature DIVIDES the training-forward logits before log_softmax.
            # Zero yields inf/NaN there rather than an error, and a negative value inverts the
            # logits -- training on a flipped policy. Separate knob from rollout.temperature, and
            # easy to set on the wrong one, so it fails here rather than mid-forward.
            raise ValueError(
                f"GRPOConfig.temperature={self.temperature} must be > 0 (it divides the training "
                f"logits; 0 gives inf/NaN and negatives invert the policy). Did you mean "
                f"rollout.temperature, the SAMPLING temperature?"
            )
        super().__post_init__()
        for field_name in ("group_size", "batch_per_replica"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer, got {value!r}")
            if value < 1:
                raise ValueError(f"{field_name} must be positive, got {value}")
        for field_name in ("mini_batch_per_replica", "micro_batch_per_replica"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer, got {value!r}")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
        for field_name in ("min_informative_groups", "dynamic_sampling_max_extra_rollouts"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer, got {value!r}")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative, got {value}")
        if self.min_informative_groups * self.group_size > self.batch_per_replica:
            raise ValueError(
                f"min_informative_groups={self.min_informative_groups} asks for more informative "
                f"groups than the batch holds: batch_per_replica={self.batch_per_replica} is "
                f"{self.batch_per_replica // self.group_size} groups of group_size={self.group_size}"
            )
        frac = self.min_informative_groups_frac
        if frac is not None:
            if isinstance(frac, bool) or not isinstance(frac, Real):
                raise TypeError(f"min_informative_groups_frac must be None or a non-bool real in (0, 1], got {frac!r}")
            if not math.isfinite(float(frac)) or not 0.0 < float(frac) <= 1.0:
                raise ValueError(
                    f"min_informative_groups_frac={frac!r} must satisfy 0 < frac <= 1: it is the share of the "
                    f"groups_per_step={self.groups_per_step} groups the batch holds that must be informative "
                    "(1.0 is full DAPO); None leaves the literal min_informative_groups in charge"
                )
            if self.min_informative_groups > 0:
                raise ValueError(
                    f"min_informative_groups={self.min_informative_groups} and min_informative_groups_frac={frac!r} "
                    "size the same quota (a literal GLOBAL group count vs. a share of groups_per_step="
                    f"{self.groups_per_step}); set exactly one and leave min_informative_groups at 0 when using "
                    "the fraction"
                )
        if self.dynamic_sampling_on_budget_exhausted not in ("proceed", "raise"):
            raise ValueError(
                "dynamic_sampling_on_budget_exhausted must be proceed|raise, "
                f"got {self.dynamic_sampling_on_budget_exhausted!r}"
            )
        if self.dynamic_sampling_refill_mode not in ("all", "short_cells"):
            raise ValueError(
                f"dynamic_sampling_refill_mode must be all|short_cells, got {self.dynamic_sampling_refill_mode!r}"
            )
        if self.dynamic_sampling_quota_scope not in ("cell", "pooled"):
            raise ValueError(
                f"dynamic_sampling_quota_scope must be cell|pooled, got {self.dynamic_sampling_quota_scope!r}"
            )
        if not isinstance(self.dynamic_sampling_refill_dispatch, str):
            raise TypeError(
                f"dynamic_sampling_refill_dispatch must be a string, got {self.dynamic_sampling_refill_dispatch!r}"
            )
        if self.dynamic_sampling_refill_dispatch not in ("per_round", "whole_budget"):
            raise ValueError(
                "dynamic_sampling_refill_dispatch must be per_round|whole_budget, "
                f"got {self.dynamic_sampling_refill_dispatch!r}"
            )
        if not isinstance(self.mask_truncated_completions, bool):
            raise TypeError(f"mask_truncated_completions must be a bool, got {self.mask_truncated_completions!r}")
        if not isinstance(self.score_centering, ScoreCenteringConfig):
            raise TypeError(
                f"score_centering must be a ScoreCenteringConfig, got {type(self.score_centering).__name__}"
            )
        if self.score_centering.enabled:
            # Score centering corrects the sampler/trainer mismatch on the SAME tokens the clipped
            # ratio or the rollout-logprob denominator would correct for staleness; the estimators are
            # not composed here, so every second correction is refused rather than silently stacked.
            if self.use_rollout_logprobs_for_loss:
                raise ValueError(
                    "score_centering.enabled=true is incompatible with use_rollout_logprobs_for_loss=true: the "
                    "rollout-logprob loss mode puts the engine's RAW temperature-1 logprob in the ratio denominator, "
                    "while score centering switches the engine to PROCESSED (tempered) logprobs and centres the "
                    "score instead. Pick one."
                )
            if self.importance_sampling_level != "token":
                raise ValueError(
                    "score_centering.enabled=true requires importance_sampling_level='token': the correction is a "
                    f"per-token expectation over the sampler head, got {self.importance_sampling_level!r}"
                )
            if self.n_mini > 1:
                raise ValueError(
                    f"score_centering.enabled=true requires n_mini == 1 (one optimizer step per collected batch), got "
                    f"n_mini={self.n_mini}: the pi_old ahead-pass ratio and the centering would both correct the same "
                    "tokens. Leave mini_batch_per_replica unset."
                )
        if self.validation.enabled:
            validation_data = self.validation.data
            if not (validation_data.path or validation_data.name or validation_data.factory):
                raise ValueError("rl.validation.enabled=true requires an explicit rl.validation.data source")
            if validation_data.shuffle:
                raise ValueError("GRPO validation data must set shuffle=false")
            same_path = bool((self.data.path or validation_data.path) and self.data.path == validation_data.path)
            same_hf_split = bool(
                not (self.data.path or validation_data.path)
                and self.data.name
                and self.data.name == validation_data.name
                and self.data.subset == validation_data.subset
                and self.data.split == validation_data.split
            )
            if same_path or same_hf_split:
                raise ValueError("training and validation datasets must be distinct")
        if self.loss_type not in (
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-fixed",
        ):
            raise ValueError(
                "loss_type must be token-mean|seq-mean-token-sum|seq-mean-token-mean|"
                f"seq-mean-token-sum-fixed, got {self.loss_type!r}"
            )
        if self.importance_sampling_level not in ("token", "sequence"):
            raise ValueError(
                f"importance_sampling_level must be 'token' or 'sequence', got {self.importance_sampling_level!r}"
            )
        if self.routing_replay not in ROUTING_REPLAY_MODES:
            raise ValueError(
                f"routing_replay must be one of {sorted(ROUTING_REPLAY_MODES)}, got {self.routing_replay!r}: "
                "the rollout engine is the only source of recorded expert routing"
            )
        if isinstance(self.ref, RefConfig) and not isinstance(self.ref, GRPORefConfig):
            # The shared contract carries no anchor source; adopt its two fields with the GRPO
            # default (`model_dir=None`: the policy's own starting checkpoint).
            self.ref = GRPORefConfig(build_ref=self.ref.build_ref, update_every=self.ref.update_every)
        if not isinstance(self.ref, GRPORefConfig):
            raise TypeError(f"rl.ref must be a GRPORefConfig (or a plain RefConfig), got {type(self.ref).__name__}")
        if self.ref.update_every != 0:
            # The anchor is the INITIAL policy. A moving reference would need a refresh hook the LLM
            # trainer does not have (only the diffusion adapter's `refresh_ref` exists), and it would
            # silently weaken the anchor the run was asked for.
            raise ValueError(
                f"GRPO keeps the KL reference frozen at the initial policy; rl.ref.update_every must be 0, "
                f"got {self.ref.update_every}"
            )
        if self.grpo_surrogate == "clipped":
            # kl_beta and the reference are one feature with two knobs; refuse either half alone, before
            # the workers and the vLLM engine come up. Mirrors the NFT validators above.
            if self.kl_beta > 0 and not self.ref.build_ref:
                raise ValueError(
                    f"kl_beta={self.kl_beta} with grpo_surrogate='clipped' needs a frozen reference policy: "
                    "set rl.ref.build_ref=true (one extra sharded copy of the network per policy rank) "
                    "or kl_beta=0"
                )
            if self.kl_beta == 0 and self.ref.build_ref:
                raise ValueError(
                    "rl.ref.build_ref=true requires kl_beta>0; the reference is only consulted by the KL term, "
                    "so set build_ref=false when kl_beta=0"
                )
        if self.grpo_surrogate not in ("legacy_pg", "clipped"):
            raise ValueError(f"grpo_surrogate must be 'legacy_pg' or 'clipped', got {self.grpo_surrogate!r}")
        if self.grpo_surrogate == "legacy_pg":
            # `routing_replay` sits here because only the clipped objective prepares the policy batch
            # through the model capability that pins; legacy_pg would run natural under a replay identity.
            ignored: list[str] = []
            for field_name in (
                "logprob_dtype",
                "epsilon_low",
                "epsilon_high",
                "kl_beta",
                "kl_coeff_clamp",
                "lower_bound_ratio",
                "loss_type",
                "use_rollout_logprobs_for_loss",
                "importance_sampling_level",
                "routing_replay",
                "mask_truncated_completions",
            ):
                value = getattr(self, field_name)
                if value != getattr(type(self), field_name):
                    ignored.append(f"{field_name}={value!r}")
            if self.ref.build_ref:  # factory-default field: compare the knob, not a class attribute
                ignored.append("ref.build_ref=True")
            if self.score_centering.enabled:  # factory-default block: compare the knob, not a class attribute
                ignored.append("score_centering.enabled=True")
            if ignored:
                raise ValueError(
                    "grpo_surrogate='legacy_pg' does not consume clipped-surrogate settings: "
                    + ", ".join(ignored)
                    + ". Select grpo_surrogate='clipped' or leave those settings at their defaults."
                )
        if self.lower_bound_ratio < 0:
            raise ValueError(f"lower_bound_ratio must be >= 0 (0 disables dual-clip), got {self.lower_bound_ratio}")
        if self.importance_sampling_level == "sequence" and self.lower_bound_ratio != 0:
            # The default 3.0 is ON, so "reject non-default" would not catch it, and dropping it the
            # way the reference implementations do is the silent ignore this file refuses elsewhere.
            raise ValueError(
                "importance_sampling_level='sequence' applies no dual clip to the response ratio (GSPO "
                f"defines none); set rl.lower_bound_ratio=0 explicitly, got {self.lower_bound_ratio!r}"
            )
        if self.group_size < 2:
            raise ValueError(f"grpo group_size must be >= 2 (variance within a group), got {self.group_size}")
        if self.mu_iterations != 1:
            # `>1` is REJECTED, not merely unsupported-in-silence. The clipped surrogate is written
            # for it -- it reads pi_old from `rollout_logprobs`, so a second inner epoch would see a
            # ratio that is no longer identically 1 -- but the DRIVER that re-uses one collected
            # batch across epochs was never built (GSM8K spec, non-goals). Nothing in the loop or the
            # objective reads this field today, so setting 2 would train exactly as 1 and the run
            # would report `mu_iterations: 2` into its W&B config while doing single-epoch updates.
            # That is the failure this repo already names: "a knob the layers below refuse to honor
            # is worse than no knob" (GridConfig.num_replicas, above). Fail at config load, before
            # the cluster comes up, and delete this branch when the driver lands.
            #
            # cosmos-rl DOES implement it -- `policy/trainer/llm_trainer/grpo_trainer.py` loops
            # `for i_mu in range(self.mu_iterations)` -- so this is a gap against the reference, not
            # a shared limitation. Porting it is more than the loop: mu>1 is only PPO if pi_old stays
            # FROZEN at the rollout policy, and posttrain has no such tensor. Its two sources both
            # fail here -- `cur.detach()` moves with training (ratio returns to 1 on epoch 2, so the
            # clip silently stops binding and you get N unconstrained steps on one batch), and
            # `rollout_logprobs` is the inference engine's raw temperature-1 logprob, valid as pi_old
            # only under the unwarped T=1 contract and carrying the train/infer gap regardless.
            # cosmos-rl solves it with a dedicated `is_computing_old_ahead` pass: model in eval, one
            # no-grad forward over the batch producing `old_per_token_logps` BEFORE the mu loop.
            # That pass, plus the fact that mu>1 breaks "one loop step == one optimizer step" which
            # resume/checkpoint/weight-sync all key off, is the actual work.
            #
            # UPDATE: THE PASS NOW EXISTS (`algorithm/grpo.py::OldLogprobCache`, driven by
            # `entrypoints/grpo.py` when `n_mini > 1`), so the pi_old half of this is done. The
            # analysis above was also INCOMPLETE in a way that cost a run: `mini_batch_per_replica`
            # reaches the identical "N unconstrained steps on one batch" state WITHOUT mu>1, and it
            # was never guarded. Measured at n_mini=32: reward collapsed to 0.0 at step 29/100 and
            # the policy emitted token soup. cosmos-rl's trigger is `batch_size >
            # per_optimize_batch_size`, which covers BOTH doors; ours now does too. What remains
            # before mu>1 can be allowed is only the loop-step/optimizer-step bookkeeping named
            # above, not pi_old.
            raise ValueError(
                f"mu_iterations must be 1, got {self.mu_iterations}: inner PPO epochs are NOT "
                f"IMPLEMENTED. The clipped objective supports a non-unit ratio, but no driver "
                f"re-uses a batch across epochs, so any value >1 would be silently ignored and the "
                f"run would train on-policy while reporting otherwise."
            )
        if self.kl_beta < 0:
            raise ValueError(f"kl_beta must be >= 0, got {self.kl_beta}")
        if self.kl_coeff_clamp is not None:
            clamp = self.kl_coeff_clamp
            if isinstance(clamp, bool) or not isinstance(clamp, Real):
                raise TypeError(f"kl_coeff_clamp must be None or a non-bool real number of nats, got {clamp!r}")
            if not math.isfinite(float(clamp)) or clamp <= 0:
                raise ValueError(f"kl_coeff_clamp must be None or a positive finite number of nats, got {clamp!r}")
            if float(clamp) >= KL_LOG_RATIO_CLAMP_NATS:
                raise ValueError(
                    f"kl_coeff_clamp={clamp!r} is inert: the log-ratio is already clamped at "
                    f"±{KL_LOG_RATIO_CLAMP_NATS} nats with a zero gradient beyond it, so the coefficient bound must "
                    "sit below that (5.0 recommended)"
                )
            if self.kl_beta == 0:
                raise ValueError(
                    f"kl_coeff_clamp={clamp!r} requires kl_beta>0: it bounds the KL term's gradient, and there is no "
                    "KL term at kl_beta=0"
                )

        if self.grad_accum != 1:
            raise ValueError(
                "GRPO retires rl.grad_accum: it meant 'that many FRESH rollouts per step', which "
                "disappears once the batch is drawn once. Set rl.micro_batch_per_replica instead — "
                "the trainer's accumulation count is derived from it (spec §5.2)."
            )
        if self.batch_per_replica < self.group_size or self.batch_per_replica % self.group_size:
            # The sole policy batch contains `batch_per_replica // group_size` total prompts. Runtime
            # prompt sizing divides that exact total across rollout/data execution replicas. A
            # remainder would silently under- or over-fill the policy batch; `< group_size` is zero.
            raise ValueError(
                f"batch_per_replica={self.batch_per_replica} must be a positive multiple of "
                f"group_size={self.group_size}: the policy batch contains batch_per_replica // "
                f"group_size prompts, so a remainder silently mis-fills it."
            )
        mini = self.mini_batch_per_replica
        if mini:
            if mini > self.batch_per_replica:
                raise ValueError(
                    f"mini_batch_per_replica={mini} exceeds batch_per_replica={self.batch_per_replica} "
                    f"(n_mini would be 0)"
                )
            if self.batch_per_replica % mini:
                raise ValueError(
                    f"batch_per_replica={self.batch_per_replica} must divide evenly by "
                    f"mini_batch_per_replica={mini}: a ragged last mini-batch changes the token-mean "
                    f"of one optimizer step per loop step"
                )
        if self.validation.enabled and self.validation.every_steps % self.n_mini:
            raise ValueError(
                f"validation every_steps={self.validation.every_steps} must be divisible by "
                f"n_mini={self.n_mini}; validation runs only at collected-batch boundaries"
            )
        micro = self.micro_batch_per_replica
        if micro:
            span = mini or self.batch_per_replica
            if micro > span:
                raise ValueError(f"micro_batch_per_replica={micro} exceeds {span} (n_micro would be 0)")
            if span % micro:
                raise ValueError(
                    f"{span} must divide evenly by micro_batch_per_replica={micro}: see above, per forward"
                )

        # ATTESTATION, checked LAST of the batching rules. A malformed config must report the
        # malformation -- a non-dividing micro batch is a structural error and stays the message the
        # user sees. This is a POLICY gate on an otherwise-valid config, so it belongs after the
        # shape is known good; running it first made a ragged-chunk bug surface as a determinism
        # complaint, which is a worse error message and is what the config tests caught.
        #
        # The condition is `needs_cached_old_logprobs` AND the objective actually consulting pi_old --
        # the same question `needs_cached_old_logprobs` already owns, so route through it rather than
        # re-deriving `n_mini > 1` here. Without its `use_rollout_logprobs_for_loss` clause this gate
        # refused a config that demonstrably cannot have the hazard: under that flag the objective
        # takes the `_gather_old` branch, never touches `OldLogprobCache`, and no ahead pass runs --
        # so demanding a determinism attestation for it rejects a dropout-enabled model that is
        # perfectly valid here. Fail-closed rather than corrupting, but still a wrong refusal.
        if (
            self.needs_cached_old_logprobs
            and self._objective_consults_pi_old()
            and not self.certified_deterministic_forward
        ):
            raise ValueError(
                f"mini_batch_per_replica={mini} gives n_mini={self.batch_per_replica // mini} > 1, "
                "which enables the pi_old ahead pass. That requires a forward DETERMINISTIC given "
                "its weights (pi_old is recorded under eval(), `cur` is scored under train()), and "
                "that cannot be proven automatically for an arbitrary model. Confirm the model has "
                "no live dropout or other stochastic layers -- Qwen2.5 reports attention_dropout "
                "0.0 -- then set rl.certified_deterministic_forward=true. Leave "
                "mini_batch_per_replica unset for one optimizer step per batch, where pi_old comes "
                "from the same forward and no attestation is needed."
            )
        # Recipe-hygiene checks. Both pairings are honored as written, so under the default
        # recipe_guard_severity="warn" they WARN (below, after every other warning, in the order they
        # always had). Under "error" every hazard found is refused together in ONE ValueError, evaluated
        # here, after every unconditional rejection and before the first warning, so a refused config
        # does not also warn.
        quota_hazard = (
            _literal_quota_below_half_message(
                literal=self.min_informative_groups,
                groups=self.groups_per_step,
                batch_per_replica=self.batch_per_replica,
                group_size=self.group_size,
            )
            if self.min_informative_groups_frac is None and 0 < 2 * self.min_informative_groups < self.groups_per_step
            else None
        )
        eps_hazard = (
            _fixed_normaliser_eps_message(self.optimizer.eps)
            if (
                self.loss_type == "seq-mean-token-sum-fixed"
                and self.optimizer.eps > FIXED_NORMALISER_OPTIMIZER_EPS_CEILING
            )
            else None
        )
        recipe_hazards = [hazard for hazard in (quota_hazard, eps_hazard) if hazard is not None]
        if recipe_hazards and self.recipe_guard_severity == RECIPE_GUARD_ERROR:
            raise ValueError(_recipe_guard_refusal(recipe_hazards))
        if self.importance_sampling_level == "sequence":
            # WARNINGS, after every rejection so a refused config does not also warn. These knobs
            # are honored, so the repo rule for un-honored knobs (reject) does not apply; they are
            # warned because each one silently makes the sequence-level clip inert or noisy.
            if self.n_mini <= 1 and not self.use_rollout_logprobs_for_loss:
                warnings.warn(
                    "importance_sampling_level='sequence' with n_mini == 1 and no rollout logprobs: pi_old is "
                    "the current policy, so every response ratio is exactly 1 and this run trains as token-level "
                    "GRPO in value and gradient while its identity records GSPO. Set rl.mini_batch_per_replica "
                    "(n_mini > 1) or rl.use_rollout_logprobs_for_loss=true for the ratio to do anything.",
                    UserWarning,
                    stacklevel=3,
                )
            active_eps = [eps for eps in (self.epsilon_low, self.epsilon_high) if eps >= 0]
            if active_eps and max(active_eps) > 0.05:
                warnings.warn(
                    f"importance_sampling_level='sequence' with epsilon_low/high={self.epsilon_low}/{self.epsilon_high}: "
                    "a response-MEAN log-ratio must exceed ln(1+eps) for the clip to bind, and token-level "
                    "epsilons like 0.2 are ~two orders of magnitude above where that mean sits (GSPO uses "
                    "3e-4/4e-4). Watch train/seq_clip_frac; an unclipped run is sequence-level importance "
                    "sampling with no trust region.",
                    UserWarning,
                    stacklevel=3,
                )
            if str(self.logprob_dtype).removeprefix("torch.") != "float32":
                warnings.warn(
                    f"importance_sampling_level='sequence' with logprob_dtype={self.logprob_dtype!r}: a sub-1e-3 "
                    "clip band cannot be resolved from bf16 logprobs (ulp ~3e-2 at |logp|~4), so the clip "
                    "measures rounding noise. Leave logprob_dtype at float32.",
                    UserWarning,
                    stacklevel=3,
                )
        if self.grpo_surrogate == "clipped" and self.kl_beta > 0 and self.kl_coeff_clamp is None:
            # Honored knob, so warn rather than reject: the exact k3 gradient is what every earlier
            # anchored run trained under, but it is also what let one token own a step.
            warnings.warn(
                f"kl_beta={self.kl_beta} with kl_coeff_clamp unset: the k3 penalty's per-token gradient coefficient "
                "1 - exp(logpi_ref - logpi_theta) is unbounded below inside the +/-20 nat log-ratio clamp "
                "Routing nondeterminism can make a small number of token coefficients dominate a step. Set "
                "rl.kl_coeff_clamp (5.0 recommended) to bound the coefficient at exp(c)-1 while train/kl_mean keeps "
                "reporting the exact k3.",
                UserWarning,
                stacklevel=3,
            )
        if quota_hazard is not None:
            # WARNING, after every rejection. The literal count is GLOBAL and placement-independent, so
            # a count tuned for one batch layout (8 = every group of b32g4) copied onto a wider one
            # (8 of the 64 groups of b256g4) quietly turns dynamic sampling into a 12% floor while the
            # run still reports it ON. The knob is honored as written, hence a warning, not a
            # rejection (recipe_guard_severity="error" refused it above); the fraction knob is the
            # layout-proof spelling.
            warnings.warn(quota_hazard, UserWarning, stacklevel=3)
        if eps_hazard is not None:
            # WARNING, not a rejection: eps is honored and the objective stays unbiased; what breaks
            # is the meaning of lr. The fixed reduction divides each sequence's token-summed loss by
            # rollout.max_tokens (2048 in the Qwen3-VL recipes) instead of by the tokens actually
            # produced, which can leave per-parameter gradients far below their token-mean scale.
            # m/(sqrt(v)+eps) degenerates to (lr/eps)*m: an SGD-like step whose size is set by lr/eps
            # rather than lr, and which therefore does not port across max_tokens (the divisor) or
            # eps. The other three reductions divide by realised token or sequence counts and keep
            # the gradient scale a 1e-6 eps was sized for, so they stay silent
            # (`_fixed_normaliser_eps_message` is the shared text; recipe_guard_severity="error"
            # refused it above).
            warnings.warn(eps_hazard, UserWarning, stacklevel=3)


def resolve_data_seed(rl: Any, rollout: Any | None) -> int:
    """The seed that orders the GRPO prompt source, resolved once for preflight and the data role.

    `rl.seed` by default; `rollout.seed` under `rl.data_seed_follows_rollout_seed`, so one per-arm
    seed varies the prompt order together with the sampling. The preflight
    (`config/build.py::_validate_online_policy_identity`) and the data role
    (`roles/env/prompt.py::build_prompt_env`) both read this value, so the loop never sees two seeds.

    Args:
        rl: the GRPO `rl` section.
        rollout: the top-level `rollout` section, or None when the caller holds no root config.

    Returns:
        The seed the prompt cursor permutes with (`seed + epoch`) and the arithmetic source draws from.

    Raises:
        ValueError: the flag is set but `rollout.seed` is not an integer. `None` leaves the engine's
            own default in force, which would silently pin every arm to one order again.
    """
    if not bool(getattr(rl, "data_seed_follows_rollout_seed", False)):
        return int(rl.seed)
    rollout_seed = None if rollout is None else getattr(rollout, "seed", None)
    if isinstance(rollout_seed, bool) or not isinstance(rollout_seed, int):
        raise ValueError(
            "rl.data_seed_follows_rollout_seed=true requires an integer rollout.seed to key the data cursor on, "
            f"got {rollout_seed!r}"
        )
    return int(rollout_seed)


def check_data_order_seed(rl: Any, rollout: Any) -> int:
    """Cross-section preflight for the prompt-order seed; returns the resolved data seed.

    Warns when `rl.data.shuffle` is on, `rl.seed` is the default 0, `rollout.seed` is set to a
    non-zero value and the flag is off: that is the launcher pattern "one arm per rollout seed", under
    which every arm draws the identical prompt sequence and a seed replicate varies sampling alone.
    Raises when the flag cannot be honored (`resolve_data_seed`).
    """
    seed = resolve_data_seed(rl, rollout)
    if not bool(getattr(getattr(rl, "data", None), "shuffle", False)):
        return seed
    if bool(getattr(rl, "data_seed_follows_rollout_seed", False)) or int(rl.seed) != 0:
        return seed
    rollout_seed = getattr(rollout, "seed", None)
    if rollout_seed is None or rollout_seed == 0:
        return seed
    warnings.warn(
        f"rl.data.shuffle=true with rl.seed=0 and rollout.seed={rollout_seed!r}: the prompt order is drawn from "
        "rl.seed + epoch, so all arms share one prompt order and arms that differ only in rollout.seed vary "
        "vLLM sampling alone. Set rl.data_seed_follows_rollout_seed=true to key the data cursor on "
        "rollout.seed, or give each arm its own rl.seed.",
        UserWarning,
        stacklevel=2,
    )
    return seed


@dataclass
class TeacherConfig:
    """The frozen, posttrain-owned OPD teacher engine.

    ``input_mode`` states which canonical observation fields the teacher consumes, while
    ``signal_mode`` states the credit representation it produces. They are separate because
    modality ownership and objective arithmetic are different contracts even though the first
    production recipes pair ``shared_vocab_multimodal`` with ``topk`` and either
    ``cross_token_text`` or ``cross_token_multimodal`` with ``dpca``.
    """

    name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    tensor_parallel_size: int = 1
    # A teacher is usually larger than the rollout model. Colocated recipes must size this together
    # with rollout.gpu_memory_utilization; disaggregated placement can assign it dedicated memory.
    gpu_memory_utilization: float = 0.4
    max_model_len: int = 4096
    dtype: str = "bfloat16"
    enforce_eager: bool = True
    # vLLM MoE kernel selection for the teacher engine, same contract as ``RolloutConfig.moe_backend``:
    # ``auto`` is vLLM's default; an explicit general backend (``triton``) is needed where the
    # architecture-specific fast kernel has stricter tile rules or faults (the SM100 FlashInfer
    # TRT-LLM BF16 MoE path dies with an illegal address on GB300 while building a TP=2 teacher).
    moe_backend: str = "auto"
    input_mode: str = "shared_vocab_multimodal"
    signal_mode: str = "topk"
    logprobs_mode: str = "processed_logprobs"
    # Support width per scored position. DPCA requires zero because it scores only the realized
    # teacher tokens and has no shared support set.
    top_k: int = 32
    scoring_temperature: float = 1.0
    max_attempts: int = 2
    retry_backoff_s: float = 0.0
    conditional_think_prefix: str = "<think>\n\n</think>\n\n"
    max_alignment_chunk_tokens: int = 64
    min_abs_student_chunk_logprob: float = 1.0e-8


@dataclass
class OPDConfig(GRPOConfig):
    """``config.rl`` for on-policy distillation: GRPO plus a frozen teacher.

    ``advantage`` mode folds the per-token teacher divergence into GRPO's policy-gradient
    advantage, so it inherits GRPO's surrogate, strict mini/micro batching, and held-out
    rollout/reward validation. Validation intentionally evaluates only the current student through
    the existing data, rollout, and reward roles; it neither invokes the teacher nor defines a
    separate distillation-objective validation contract.

    All three ``loss_type`` modes use the objective's DP-global normalization-unit contract. For
    ``seq-mean-token-sum`` that unit is a sequence, so it computes the mean of per-sequence token
    sums; this matches the reference ``token-sum`` update up to its fixed mini-batch factor without
    introducing a rank-local or micro-batch-local mean.
    """

    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    # ``direct`` minimizes divergence; ``advantage`` subtracts it from the group advantage before
    # applying the on-policy GRPO surrogate. This selector is intentionally distinct from
    # ``grpo_surrogate``; the latter is fixed to ``clipped`` for OPD.
    distillation_mode: str = "advantage"
    # 0 = KL(teacher||student), 1 = KL(student||teacher); the reference uses 1.
    jsd_beta: float | None = 1.0
    kl_penalty_coef: float = 1.0

    def __post_init__(self) -> None:
        if self.kl_beta != 0:
            raise ValueError(
                f"OPD requires kl_beta=0, got {self.kl_beta!r}: direct mode does not read the inherited "
                "reference-policy KL knob, and advantage mode has no reference logprobs"
            )
        if self.importance_sampling_level != "token":
            # Before super(): GRPOConfig's own sequence-level rules would otherwise answer first with
            # GSPO advice for a knob OPD never reads. Advantage mode scores the token-level ratio
            # (`algorithm/opd.py`), direct mode forms no ratio, and `build_opd_objective` does not
            # forward the field, so any other value would be silently ignored.
            raise ValueError(
                "OPD does not consume importance_sampling_level: its advantage objective scores a token-level "
                f"ratio and direct mode forms none. Leave it at 'token', got {self.importance_sampling_level!r}"
            )
        if self.routing_replay != "off":
            # Before super() so OPD's own refusal answers first. `build_opd_objective` builds no
            # replay-aware trainer wiring, so any other value would be a silent no-op.
            raise ValueError(
                f"OPD does not consume the GRPO-only rl.routing_replay, got {self.routing_replay!r}; leave it at 'off'"
            )
        if self.min_informative_groups != 0 or self.min_informative_groups_frac is not None:
            # Before super(): GRPOConfig warns about a literal count below half of groups_per_step, and
            # a refused config must not also warn. Dynamic sampling drops reward-tied groups as
            # gradient-free, but under distillation every trajectory carries teacher signal
            # regardless of reward spread.
            raise ValueError(
                "OPD does not consume min_informative_groups / min_informative_groups_frac: dynamic sampling "
                "drops reward-tied groups as gradient-free, but under distillation every trajectory carries "
                "teacher signal regardless of reward spread. Leave them at 0 / None, got "
                f"min_informative_groups={self.min_informative_groups!r}, "
                f"min_informative_groups_frac={self.min_informative_groups_frac!r}"
            )
        if self.loss_type == "seq-mean-token-sum-fixed":
            # Before super() for the same reason as routing_replay: OPD's own refusal answers first,
            # and the inherited fixed-normaliser eps warning (GRPOConfig, issued after its
            # rejections) must not fire for a section OPD is about to refuse.
            raise ValueError(
                "OPD does not consume the Dr-GRPO-only loss_type='seq-mean-token-sum-fixed'; "
                "select one of token-mean|seq-mean-token-sum|seq-mean-token-mean."
            )
        super().__post_init__()
        teacher_tp = self.teacher.tensor_parallel_size
        if isinstance(teacher_tp, bool) or not isinstance(teacher_tp, int) or teacher_tp < 1:
            raise ValueError(f"teacher.tensor_parallel_size must be a non-bool positive integer, got {teacher_tp!r}")
        if self.distillation_mode not in ("direct", "advantage"):
            raise ValueError(f"distillation_mode must be direct|advantage, got {self.distillation_mode!r}")
        if self.teacher.input_mode not in (
            "shared_vocab_multimodal",
            "cross_token_text",
            "cross_token_multimodal",
        ):
            raise ValueError(
                "teacher.input_mode must be shared_vocab_multimodal|cross_token_text|"
                f"cross_token_multimodal, got {self.teacher.input_mode!r}"
            )
        if self.teacher.signal_mode not in ("topk", "dpca"):
            raise ValueError(f"teacher.signal_mode must be topk|dpca, got {self.teacher.signal_mode!r}")
        if self.teacher.logprobs_mode != "processed_logprobs":
            raise ValueError(
                "teacher.logprobs_mode must be 'processed_logprobs' for the declared likelihood "
                f"contract, got {self.teacher.logprobs_mode!r}"
            )
        teacher_top_k = self.teacher.top_k
        if isinstance(teacher_top_k, bool) or not isinstance(teacher_top_k, int) or teacher_top_k < 0:
            raise ValueError(f"teacher.top_k must be a non-bool integer >= 0, got {teacher_top_k!r}")
        if self.teacher.signal_mode == "topk":
            if self.teacher.input_mode != "shared_vocab_multimodal":
                raise ValueError("teacher.signal_mode='topk' requires input_mode='shared_vocab_multimodal'")
            if teacher_top_k < 1:
                raise ValueError("teacher.signal_mode='topk' requires teacher.top_k > 0")
            if self.jsd_beta is None or not 0.0 <= self.jsd_beta <= 1.0:
                raise ValueError(f"top-k OPD requires jsd_beta in [0, 1], got {self.jsd_beta!r}")
        else:
            if self.teacher.input_mode not in ("cross_token_text", "cross_token_multimodal"):
                raise ValueError(
                    "teacher.signal_mode='dpca' requires input_mode='cross_token_text' or 'cross_token_multimodal'"
                )
            if self.distillation_mode != "advantage":
                raise ValueError("teacher.signal_mode='dpca' requires distillation_mode='advantage'")
            if teacher_top_k != 0:
                raise ValueError("teacher.signal_mode='dpca' requires teacher.top_k=0")
            if self.jsd_beta is not None:
                raise ValueError("teacher.signal_mode='dpca' requires jsd_beta=None; it has no shared support set")
        kl_penalty_coef = self.kl_penalty_coef
        if (
            isinstance(kl_penalty_coef, bool)
            or not isinstance(kl_penalty_coef, Real)
            or not math.isfinite(float(kl_penalty_coef))
            or kl_penalty_coef < 0
        ):
            raise ValueError(f"kl_penalty_coef must be a non-bool finite real >= 0, got {kl_penalty_coef!r}")
        teacher_temperature = self.teacher.scoring_temperature
        if (
            isinstance(teacher_temperature, bool)
            or not isinstance(teacher_temperature, Real)
            or not math.isfinite(float(teacher_temperature))
            or teacher_temperature <= 0
        ):
            raise ValueError(
                f"teacher.scoring_temperature must be a non-bool finite real > 0, got {teacher_temperature!r}"
            )
        if (
            isinstance(self.teacher.max_attempts, bool)
            or not isinstance(self.teacher.max_attempts, int)
            or self.teacher.max_attempts < 1
        ):
            raise ValueError(
                f"teacher.max_attempts must be a non-bool positive integer, got {self.teacher.max_attempts!r}"
            )
        if (
            isinstance(self.teacher.retry_backoff_s, bool)
            or not isinstance(self.teacher.retry_backoff_s, Real)
            or not math.isfinite(float(self.teacher.retry_backoff_s))
            or self.teacher.retry_backoff_s < 0
            or self.teacher.retry_backoff_s > 60
        ):
            raise ValueError(
                "teacher.retry_backoff_s must be a non-bool finite real in [0, 60], got "
                f"{self.teacher.retry_backoff_s!r}"
            )
        if not isinstance(self.teacher.conditional_think_prefix, str):
            raise ValueError("teacher.conditional_think_prefix must be a string")
        if (
            isinstance(self.teacher.max_alignment_chunk_tokens, bool)
            or not isinstance(self.teacher.max_alignment_chunk_tokens, int)
            or self.teacher.max_alignment_chunk_tokens < 1
        ):
            raise ValueError(
                "teacher.max_alignment_chunk_tokens must be a non-bool positive integer, got "
                f"{self.teacher.max_alignment_chunk_tokens!r}"
            )
        student_floor = self.teacher.min_abs_student_chunk_logprob
        if (
            isinstance(student_floor, bool)
            or not isinstance(student_floor, Real)
            or not math.isfinite(float(student_floor))
            or student_floor <= 0
        ):
            raise ValueError(
                f"teacher.min_abs_student_chunk_logprob must be a non-bool finite real > 0, got {student_floor!r}"
            )
        # OPD has one distillation selector. Its advantage mode uses the production clipped GRPO
        # surrogate; direct mode does not form a policy-gradient surrogate at all. Refuse any other
        # inherited selection instead of silently ignoring it.
        if self.grpo_surrogate != "clipped":
            raise ValueError(
                "OPD requires grpo_surrogate='clipped'; select direct|advantage with "
                f"distillation_mode instead, got grpo_surrogate={self.grpo_surrogate!r}"
            )
        if self.use_rollout_logprobs_for_loss:
            raise ValueError(
                "OPD does not consume rollout logprobs for loss: advantage mode freezes pi_old "
                "through its policy-side old-logprob cache, while direct mode forms no importance "
                "ratio. This matches the reference config's `use_rollout_logprobs_for_loss=false`. "
                "Leave it False."
            )
        if self.logprob_dtype != type(self).logprob_dtype:
            raise ValueError(
                "OPD does not consume logprob_dtype in either distillation mode: both direct and "
                "advantage scoring use their objective-owned fp32 path. Leave it at the default."
            )
        if self.mask_truncated_completions:
            raise ValueError(
                "OPD does not consume mask_truncated_completions: neither distillation objective reads "
                "the rollout's finish_reason stamp, so a length-cut completion would train regardless. "
                "Leave it False."
            )
        if self.score_centering.enabled:
            raise ValueError(
                "OPD does not consume score_centering: the distillation objectives score the teacher's "
                "support, not a sampler-centred policy gradient. Leave score_centering.enabled False."
            )
        if self.distillation_mode == "direct":
            ignored: list[str] = []
            for field_name in (
                "epsilon_low",
                "epsilon_high",
                "lower_bound_ratio",
                "kl_penalty_coef",
                "unbiased_advantage",
            ):
                value = getattr(self, field_name)
                if value != getattr(type(self), field_name):
                    ignored.append(f"{field_name}={value!r}")
            if ignored:
                raise ValueError(
                    "distillation_mode='direct' does not consume advantage-surrogate settings: "
                    + ", ".join(ignored)
                    + ". Select distillation_mode='advantage' or leave those settings at their defaults."
                )
            warnings.warn(
                "distillation_mode='direct' does not use task reward or group advantage as a "
                "training signal; reward configuration still controls reward-stage metrics, "
                "held-out evaluation, and optional shared-prefix masking.",
                UserWarning,
                # 3 frames out of `warn`: __post_init__, the dataclass-generated __init__, then the
                # construction site the user actually wrote.
                stacklevel=3,
            )

    def _objective_consults_pi_old(self) -> bool:
        """Only OPD's ADVANTAGE objective forms an importance ratio against a frozen π_old.

        `direct` scores the target policy outright, never reads the cache (the driver already skips
        the ahead pass for it via `policy.needs_old_logprobs()`), and so cannot meet the eval/train
        hazard the determinism attestation guards. Requiring the attestation there rejected valid
        mini-batched direct-OPD configs at construction.

        Returns:
            True only in `advantage` mode.
        """
        return self.distillation_mode == "advantage"


# PPO's own deterministic matchers. Any other accepted value names a SHARED rule reward, so this
# tuple is the boundary between "PPO builds the provider" and "PPO binds a shared one".
PPO_BUILTIN_TERMINALS: tuple[str, ...] = ("arithmetic_exact_match", "none")


@dataclass
class PPORewardConfig:
    """Deterministic built-in reward adapter used by PPO recipes.

    Production reward models can replace the entrypoint callbacks while preserving
    the same terminal and position-aligned process-reward contract.
    """

    terminal: str = "arithmetic_exact_match"
    process: str = "none"
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    process_reward_scale: float = 0.0
    # Optional importable providers, formatted as ``package.module:callable``.
    # Each receives PPORewardContext plus its corresponding keyword mapping.
    terminal_callable: str = ""
    process_callable: str = ""
    terminal_kwargs: dict[str, Any] = field(default_factory=dict)
    process_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.terminal not in PPO_BUILTIN_TERMINALS and self.terminal not in RULE_REWARDS:
            raise ValueError(
                f"terminal reward must be one of {[*PPO_BUILTIN_TERMINALS, *sorted(RULE_REWARDS)]}, "
                f"got {self.terminal!r}"
            )
        if self.process not in ("linear", "none"):
            raise ValueError(f"process reward must be 'linear' or 'none', got {self.process!r}")
        for name in ("correct_reward", "incorrect_reward", "process_reward_scale"):
            setattr(self, name, _normalize_finite_real(getattr(self, name), field_name=name))
        if self.process_reward_scale < 0.0:
            raise ValueError(f"process_reward_scale must be >= 0, got {self.process_reward_scale}")
        for name in ("terminal_callable", "process_callable"):
            path = getattr(self, name)
            if not isinstance(path, str):
                raise TypeError(f"{name} must be a string, got {type(path).__name__}")
            module_name, separator, attribute = path.partition(":")
            if path and (not separator or not module_name or not attribute):
                raise ValueError(f"{name} must use the absolute 'package.module:callable' format")
        if self.terminal_callable and self.terminal not in PPO_BUILTIN_TERMINALS:
            raise ValueError(
                "terminal_callable overrides PPO's own matcher and cannot be combined with the shared "
                f"rule reward {self.terminal!r}"
            )
        for name in ("terminal_kwargs", "process_kwargs"):
            kwargs = getattr(self, name)
            if not isinstance(kwargs, Mapping) or any(not isinstance(key, str) for key in kwargs):
                raise TypeError(f"{name} must be a mapping with string keys")


@dataclass
class PPOBaseConfig(RunConfig):
    """PPO recurrence and clipping shared by text and decision-level recipes."""

    gamma: float = 1.0
    gae_lambda: float = 0.95
    policy_clip: float = 0.2
    value_clip: float = 0.2

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("gamma", "gae_lambda"):
            value = _normalize_finite_real(getattr(self, name), field_name=name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
        policy_clip = _normalize_finite_real(self.policy_clip, field_name="policy_clip")
        if not 0.0 < policy_clip < 1.0:
            raise ValueError(f"policy_clip must be finite and in (0, 1), got {policy_clip}")
        value_clip = _normalize_finite_real(self.value_clip, field_name="value_clip")
        if value_clip <= 0.0:
            raise ValueError(f"value_clip must be finite and > 0, got {value_clip}")


@dataclass
class JointPPOConfig(PPOBaseConfig):
    """Synchronous decision windows with one jointly optimized actor/value model."""

    num_envs: int = 8
    window_decisions: int = 16
    global_minibatch: int = 32
    microbatch: int = 2
    seed: int = 1234
    verify_behavior_logprobs: bool = True
    gamma: float = 0.99
    time_limit_bootstrap: Literal["value", "zero"] = "value"
    dual_clip: float | None = 3.0
    loss_reduction: Literal["decision_mean", "fixed_episode_mean"] = "decision_mean"
    huber_delta: float = 10.0
    value_lr: float = 1e-4
    max_steps: int = 800
    save_every: int = 80
    resume: str = "off"
    checkpoint_format: str = "dcp"
    save_async: bool = False
    keep_optim_last: int | None = 0
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            lr=5e-6,
            betas=[0.9, 0.95],
            eps=1e-8,
            weight_decay=0.01,
            grad_clip_norm=1.0,
        )
    )
    validation: ValidationScheduleConfig = field(
        default_factory=lambda: ValidationScheduleConfig(
            enabled=True,
            before_training=True,
            every_steps=80,
            save_checkpoint=True,
        )
    )

    @property
    def actor_lr(self) -> float:
        return self.optimizer.lr

    @property
    def decisions(self) -> int:
        return self.num_envs * self.window_decisions

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("num_envs", "window_decisions", "global_minibatch", "microbatch"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"rl.{name} must be a positive integer")
        if type(self.verify_behavior_logprobs) is not bool:
            raise ValueError("rl.verify_behavior_logprobs must be a boolean")
        if self.time_limit_bootstrap not in ("value", "zero"):
            raise ValueError("rl.time_limit_bootstrap must be 'value' or 'zero'")
        if self.loss_reduction not in ("decision_mean", "fixed_episode_mean"):
            raise ValueError("rl.loss_reduction must be 'decision_mean' or 'fixed_episode_mean'")
        if self.loss_reduction == "fixed_episode_mean" and self.time_limit_bootstrap != "zero":
            raise ValueError("rl.loss_reduction=fixed_episode_mean requires rl.time_limit_bootstrap=zero")
        for name in ("huber_delta", "value_lr") + (("dual_clip",) if self.dual_clip is not None else ()):
            value = _normalize_finite_real(getattr(self, name), field_name=f"rl.{name}")
            if value <= (1 if name == "dual_clip" else 0):
                raise ValueError(f"rl.{name} has an invalid value: {value}")
        if self.decisions % self.global_minibatch:
            raise ValueError("rl.num_envs * rl.window_decisions must be divisible by rl.global_minibatch")
        if self.checkpoint_format != "dcp" or self.save_async or self.grad_accum != 1:
            raise ValueError("joint PPO requires synchronous dcp checkpoints and rl.grad_accum=1")
        if self.optimizer.train_keys or self.optimizer.decay_groups:
            raise ValueError("joint PPO owns the complete actor/value parameter partition")

    def identity(self) -> dict:
        from dataclasses import asdict

        from cosmos_framework.posttrain.utils.identity import canonical_config_value

        return {
            "algorithm": "joint-ppo-v3",
            # Version 3 already requires verification; only the explicit opt-out
            # changes its identity, preserving existing full-check GR00T runs.
            **({"verify_behavior_logprobs": False} if not self.verify_behavior_logprobs else {}),
            **{
                name: getattr(self, name)
                for name in (
                    "num_envs",
                    "window_decisions",
                    "global_minibatch",
                    "microbatch",
                    "seed",
                    "gamma",
                    "time_limit_bootstrap",
                    "gae_lambda",
                    "policy_clip",
                    "dual_clip",
                    "loss_reduction",
                    "value_clip",
                    "huber_delta",
                    "value_lr",
                )
            },
            "advantage_normalization": (
                "actor-local-float32-correction1-std-plus-1e-5"
                if self.loss_reduction == "fixed_episode_mean"
                else "actor-local-correction1-std-plus-1e-5"
            ),
            "optimizer": canonical_config_value(asdict(self.optimizer)),
        }


@dataclass
class PPOConfig(PPOBaseConfig, HFRunConfig):
    """Clipped actor-critic PPO schema; inherited `optimizer` is the policy optimizer."""

    checkpoint_format: str = "dcp"
    save_every: int = 1
    ppo_epochs: int = 4
    num_minibatches: int = 1
    microbatch_size: int = 1
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    reference_kl_coef: float = 0.01
    jsd_coef: float = 0.0
    jsd_beta: float = 0.5
    jsd_top_k: int = 128
    prompts_per_iteration: int = 8
    # Leading OUTER iterations that collect, score, prepare targets and train the CRITIC ONLY. The
    # actor's parameters, optimizer state and LR scheduler are left untouched for exactly this many
    # iterations, so a freshly initialized value function can calibrate against the task's reward
    # distribution before its estimates start steering policy updates. `0` (the default) is the
    # historical behaviour: every iteration updates both roles.
    critic_warmup_iterations: int = 0
    max_operand: int = 50
    # The built-in source can attach synthetic media for connectivity tests.
    modality: str = "text"
    image_size: int = 64
    video_frames: int = 4
    video_fps: float = 24.0
    reward: PPORewardConfig = field(default_factory=PPORewardConfig)
    data: PPODatasetConfig = field(default_factory=PPODatasetConfig)
    critic_optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    weight_sync_transport: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        # Retain the text schema's numeric normalization; native recipes retain
        # their original serialized scalar types in checkpoint identities.
        for name in ("gamma", "gae_lambda", "policy_clip", "value_clip", "jsd_beta"):
            setattr(self, name, _normalize_finite_real(getattr(self, name), field_name=name))
        if not 0.0 <= self.jsd_beta <= 1.0:
            raise ValueError(f"jsd_beta must be finite and in [0, 1], got {self.jsd_beta}")
        for name in (
            "ppo_epochs",
            "num_minibatches",
            "microbatch_size",
            "jsd_top_k",
            "prompts_per_iteration",
            "max_operand",
            "image_size",
            "video_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if (
            isinstance(self.critic_warmup_iterations, bool)
            or not isinstance(self.critic_warmup_iterations, int)
            or self.critic_warmup_iterations < 0
        ):
            raise ValueError(
                f"critic_warmup_iterations must be a non-negative integer, got {self.critic_warmup_iterations!r}"
            )
        if self.modality not in ("text", "image", "video"):
            raise ValueError(f"modality must be 'text', 'image', or 'video', got {self.modality!r}")
        self.video_fps = normalize_video_fps(self.video_fps)
        batch_divisor = self.num_minibatches * self.microbatch_size
        if self.prompts_per_iteration % batch_divisor != 0:
            raise ValueError(
                f"prompts_per_iteration={self.prompts_per_iteration} must be divisible by "
                f"num_minibatches * microbatch_size ({batch_divisor})"
            )
        for name in ("value_coef", "entropy_coef", "reference_kl_coef", "jsd_coef"):
            value = _normalize_finite_real(getattr(self, name), field_name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value}")
            setattr(self, name, value)
        if not isinstance(self.optimizer, OptimizerConfig):
            raise TypeError(f"optimizer must be an OptimizerConfig, got {type(self.optimizer).__name__}")
        if not isinstance(self.critic_optimizer, OptimizerConfig):
            raise TypeError(f"critic_optimizer must be an OptimizerConfig, got {type(self.critic_optimizer).__name__}")
        if not isinstance(self.reward, PPORewardConfig):
            raise TypeError(f"reward must be a PPORewardConfig, got {type(self.reward).__name__}")
        if not isinstance(self.data, PPODatasetConfig):
            raise TypeError(f"data must be a PPODatasetConfig, got {type(self.data).__name__}")
