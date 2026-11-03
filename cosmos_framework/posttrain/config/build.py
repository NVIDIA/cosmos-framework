# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Shared construction primitives (DESIGN: config system).

`config/rl.py` is the typed *schema*; this module holds the **shared, algorithm-agnostic**
building blocks that entries compose:

  - `build_backbone(config, world)` — backend dispatch (legacy LazyConfig tree | generic native
    policy build spec | HF from_pretrained) → the model/capability.
  - `build_optimizer(model, opt_cfg)` — RL-owned optimizer (freeze backbone, train the subset);
    `opt_cfg.name` picks the class and `opt_cfg.impl` the torch kernel.
  - `resolve_precision(config)` — the ONE `rl.*_dtype` → `model_hf.*` fallback chain, splitting
    STORAGE (`master`, what the optimizer's moments inherit) from COMPUTE (`param`).
  - `build_hf_model(hf_cfg, pdims, precision=None)` — `HFModel` body + FSDP lifecycle over an
    injected mesh. `precision=None` derives from `hf_cfg` alone (pre-split behaviour).

There is intentionally **no per-algorithm `build_dpo_from_config`/assemble here** — each entry
(e.g. `entrypoints/dpo.py`) assembles its own stack from these primitives, so build.py never grows
a case per algorithm.
"""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.config.model import (
    HF,
    I4,
    QWEN3_VL,
    DataPackerConfig,
    HFModelConfig,
    NativePolicyBuildSpec,
    NativePolicyParallelismConfig,
)
from cosmos_framework.posttrain.config.rl import (
    OPTIMIZER_NAMES,
    GRPOConfig,
    OPDConfig,
    OptimizerConfig,
    RolloutConfig,
    TeacherConfig,
    WFMLRSchedulerConfig,
    check_data_order_seed,
)
from cosmos_framework.posttrain.utils.dist import world_size_or_1
from cosmos_framework.posttrain.utils.log import logger
from cosmos_framework.posttrain.utils.scheduler import bind_scheduler_resume_identity

if TYPE_CHECKING:
    # Type-only: this module builds algorithm objects through LAZY function-local imports so that
    # config/ carries no runtime dependency on algorithm/. Annotating the return must not undo that.
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective
    from cosmos_framework.posttrain.config.sft import SFTLRScheduleConfig
    from cosmos_framework.posttrain.data.packer.base import DataPacker
    from cosmos_framework.posttrain.roles.teacher import VLLMTeacher


_QWEN3_VL_NATIVE_BUILDER = "cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_policy"
_QWEN3_VL_NATIVE_VALUE_BUILDER = "cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_native_value_model"
_LOCAL_MODEL_CONFIG_RETRY_DELAYS_S = (1.0, 2.0, 4.0, 8.0, 16.0)


def _configure_vllm_runtime() -> None:
    """Force Posttrain's supported in-process vLLM V1 control path.

    Each Ray actor already provides process isolation. Keeping EngineCore in that actor makes
    ``apply_model`` a direct in-process call and avoids an unnecessary pickle/msgpack boundary for
    per-step weight synchronization.
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def _resolve_hf_config_with_local_retry(name_or_path: str, *, trust_remote_code: bool | None) -> Any:
    """Resolve one HF config, retrying transient reads only for an absolute local path.

    A large rollout or teacher build resolves the same shared-filesystem config on every rank. A
    transient Lustre visibility error otherwise makes Transformers reinterpret that absolute path as a
    Hub repository ID and aborts the complete setup. Remote repository errors remain single-attempt
    failures so authentication and revision mistakes are not hidden. A relative local directory
    (``models/x``) also gets one attempt: while it is invisible it is indistinguishable from a Hub ID.
    ``trust_remote_code=None`` keeps Transformers' default by not passing the argument.
    """
    from transformers import AutoConfig

    options = {} if trust_remote_code is None else {"trust_remote_code": trust_remote_code}
    retry_delays = _LOCAL_MODEL_CONFIG_RETRY_DELAYS_S if os.path.isabs(name_or_path) else ()
    for attempt in range(len(retry_delays) + 1):
        try:
            return AutoConfig.from_pretrained(name_or_path, **options)
        except OSError as exc:
            if attempt == len(retry_delays):
                raise
            delay_s = retry_delays[attempt]
            logger.warning(
                f"local model config read failed for {name_or_path!r}; "
                f"retrying in {delay_s:g}s ({attempt + 1}/{len(retry_delays)}): {exc}"
            )
            time.sleep(delay_s)
    raise AssertionError("unreachable local model config retry state")


def normalize_model_backend(backend: str) -> str:
    """Normalize a public backend name to the generic ``i4 | hf`` online split.

    ``qwen3_vl`` is retained only as a compatibility alias for existing OPD configs. Concrete
    model-family selection belongs to ``config.model``'s native build spec, never to an entrypoint.

    Args:
        backend: Raw root ``model_backend`` value.

    Returns:
        ``"i4"`` or ``"hf"``.

    Raises:
        ValueError: The backend is unknown.
    """
    normalized = I4 if backend == QWEN3_VL else backend
    if normalized not in (I4, HF):
        raise ValueError(
            f"unknown model_backend {backend!r} (expected {I4!r} or {HF!r}; {QWEN3_VL!r} is a compatibility alias)"
        )
    return normalized


@dataclass(frozen=True)
class PolicyBackendSelection:
    """One normalized view of the policy backend fields consumed by online wiring."""

    backend: str
    model_id: str
    model_config: Any | None
    native_spec: Any | None

    @property
    def trust_remote_code(self) -> bool:
        """Whether policy-owned HF consumers may execute model-repository code."""
        return self.backend == HF and bool(getattr(self.model_config, "trust_remote_code", False))


@dataclass(frozen=True)
class DataPackerBuildSpec:
    """Serializable, immutable identity for constructing one actor-local DataPacker."""

    model_id: str
    factory: str = ""
    model_backend: str | None = None
    trust_remote_code: bool = False

    def build(self) -> "DataPacker":
        """Construct a fresh packer from this shared identity."""
        from cosmos_framework.posttrain.data.packer import init_data_packer_from_factory

        return init_data_packer_from_factory(
            self.model_id,
            self.factory,
            trust_remote_code=self.trust_remote_code,
            model_backend=self.model_backend,
        )


def policy_data_packer_build_spec(
    config: Any,
    data_packer_config: DataPackerConfig,
) -> DataPackerBuildSpec:
    """Resolve the one policy-owned packer identity shared by every online actor."""
    selection = select_policy_backend(config)
    return DataPackerBuildSpec(
        model_id=selection.model_id,
        factory=data_packer_config.factory,
        model_backend=selection.backend,
        trust_remote_code=selection.trust_remote_code,
    )


def native_policy_build_spec(config: Any) -> Any | None:
    """Return the generic native-policy spec selected by ``config``, if one is present.

    The legacy ``model_backend='qwen3_vl'`` + ``model_qwen3_vl`` pair is projected onto the generic
    seam here. This is the only compatibility boundary; callers and entrypoints see ``i4``.
    """
    raw_backend = str(getattr(config, "model_backend", I4))
    if raw_backend == QWEN3_VL:
        model_config = getattr(config, "model_qwen3_vl", None)
        if model_config is None:
            raise AttributeError("model_backend='qwen3_vl' requires config.model_qwen3_vl")
        return NativePolicyBuildSpec(
            builder=_QWEN3_VL_NATIVE_BUILDER,
            value_builder=_QWEN3_VL_NATIVE_VALUE_BUILDER,
            model_id=str(getattr(model_config, "name_or_path", "")),
            model_config=model_config,
            parallelism=NativePolicyParallelismConfig(dp_replicate=1, dp_shard=-1),
            requires_collated_policy_forward=True,
        )
    if normalize_model_backend(raw_backend) != I4:
        return None
    model = getattr(config, "model", None)
    if model is None:
        return None
    builder = getattr(model, "builder", None)
    model_id = getattr(model, "model_id", None)
    model_config = getattr(model, "model_config", None)
    parallelism = getattr(model, "parallelism", None)
    if builder is None and model_config is None:
        return None
    if not isinstance(builder, str) or not builder.strip() or "." not in builder:
        raise ValueError("config.model.builder must be a non-empty absolute dotted callable")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("config.model.model_id must be a non-empty model/tokenizer artifact identity")
    if model_config is None:
        raise ValueError("config.model.model_config must not be None for a native policy build spec")
    config_model_id = str(getattr(model_config, "name_or_path", "") or "")
    if config_model_id and config_model_id != model_id:
        raise ValueError(
            "config.model.model_id and config.model.model_config.name_or_path must identify the same "
            f"policy artifact when both are set, got {model_id!r} and {config_model_id!r}"
        )
    if parallelism is None:
        raise ValueError("config.model.parallelism must be configured for a native policy build spec")
    return model


def select_policy_backend(config: Any) -> PolicyBackendSelection:
    """Resolve backend, identity, model config, and native spec exactly once."""
    backend = normalize_model_backend(str(getattr(config, "model_backend", I4)))
    if backend == HF:
        model_config = getattr(config, "model_hf", None)
        model_id = getattr(model_config, "name_or_path", "") or getattr(
            getattr(config, "rollout", None),
            "name_or_path",
            "",
        )
        return PolicyBackendSelection(
            backend=backend,
            model_id=str(model_id or ""),
            model_config=model_config,
            native_spec=None,
        )
    spec = native_policy_build_spec(config)
    return PolicyBackendSelection(
        backend=backend,
        model_id=str(getattr(spec, "model_id", "") or "") if spec is not None else "",
        model_config=getattr(spec, "model_config", None) if spec is not None else None,
        native_spec=spec,
    )


def build_lr_scheduler(optimizer: Any, opt_cfg: OptimizerConfig):
    """Linear LR warmup for `opt_cfg.warmup_steps`, then constant — cosmos-rl `optm_warmup_steps`.

    Returns None when `warmup_steps <= 0` so the no-warmup path stays exactly as it was (no
    scheduler object, no per-step call).

    Ramp: step 0 uses lr * 1/warmup, step warmup-1 uses lr * warmup/warmup. The first step is
    therefore NOT zero-LR.

    An earlier revision claimed cosmos-rl "ramps on the same 1-based convention". MEASURED FALSE.
    Its default schedule (`warmup_stable_decay`, `policy/trainer/optm/__init__.py`) is
    `start + (1-start) * step/(warmup-1)` on a 0-BASED step, and `optm_warmup_start_factor`
    defaults to 0.0 — so cosmos-rl DOES burn a zero-LR first step, and its ramp sits below this one
    at every step but the last.

    The two ARE reconcilable, which the parity doc previously denied: setting cosmos-rl's
    `optm_warmup_start_factor = 1/optm_warmup_steps` makes its factor `1/w + k/w == (k+1)/w`,
    identical to this one at every step. Verified for w=20. See docs/design/grpo-cosmos-rl-parity.md.

    Args:
        optimizer: the optimizer whose param-group LRs are scaled.
        opt_cfg: supplies `warmup_steps`.

    Returns:
        A `LambdaLR`, or None when warmup is disabled.
    """
    import torch

    w = int(getattr(opt_cfg, "warmup_steps", 0) or 0)
    if w <= 0:
        return None
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / w))


def warmup_cosine_factor(
    step: int, *, warmup_steps: int, cycle_steps: int, f_start: float, f_min: float, f_max: float = 1.0
) -> float:
    """The LR multiplier of ``imaginaire/functional/lr_scheduler.py::LambdaWarmUpCosineScheduler`` for one cycle.

    ``step`` is 0-based (the scheduler's ``last_epoch``): ``f_start + (f_max - f_start) * step / warmup_steps`` for
    ``step < warmup_steps`` (so the first update runs at ``f_start``, the backbone's own convention), then
    ``f_min + 0.5 * (f_max - f_min) * (1 + cos(pi * t))`` with ``t = min(1, (step - warmup) / (cycle - warmup))``; the
    factor stays at ``f_min`` past the cycle. This is the schedule the Cosmos post-training recipe of the Qwen3-VL-MoE
    backbone trains with (``warm_up_steps [1000]``, ``f_start 0.01``, ``f_min 0.1``, cycle = ``trainer.max_iter``).
    """
    import math

    if warmup_steps > 0 and step < warmup_steps:
        return f_start + (f_max - f_start) * step / warmup_steps
    span = cycle_steps - warmup_steps
    t = 1.0 if span <= 0 else min(1.0, (step - warmup_steps) / span)
    return f_min + 0.5 * (f_max - f_min) * (1.0 + math.cos(math.pi * t))


def build_warmup_cosine_lr_scheduler(
    optimizer: Any, *, warmup_steps: int, cycle_steps: int, f_start: float, f_min: float, f_max: float = 1.0
) -> Any:
    """A ``LambdaLR`` over :func:`warmup_cosine_factor`, stepped once per applied optimizer update, with its resume
    identity bound like the WFM schedule. A cycle shorter than the warmup (a 2-step canary of a 1,000-step warmup) is
    legal and stays on the ramp, as the backbone's scheduler does; ``0 < f_start, f_min <= f_max``."""
    import torch

    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError(f"warmup_steps={warmup_steps!r} must be an integer >= 0")
    if isinstance(cycle_steps, bool) or not isinstance(cycle_steps, int) or cycle_steps < 1:
        raise ValueError(f"cycle_steps={cycle_steps!r} must be an integer >= 1")
    for name, value in (("f_start", f_start), ("f_min", f_min), ("f_max", f_max)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < float(value) <= float(f_max):
            raise ValueError(f"{name}={value!r} must satisfy 0 < {name} <= f_max={f_max!r}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: warmup_cosine_factor(
            step, warmup_steps=warmup_steps, cycle_steps=cycle_steps, f_start=f_start, f_min=f_min, f_max=f_max
        ),
    )
    bind_scheduler_resume_identity(
        scheduler,
        {
            "version": 1,
            "kind": "imaginaire_lambda_warmup_cosine",
            "step_timing": "after_applied_optimizer_update_v1",
            "warm_up_steps": int(warmup_steps),
            "cycle_length": int(cycle_steps),
            "f_start": float(f_start),
            "f_max": float(f_max),
            "f_min": float(f_min),
        },
    )
    return scheduler


def build_sft_lr_scheduler(
    optimizer: Any, opt_cfg: OptimizerConfig, schedule: SFTLRScheduleConfig, max_steps: int
) -> Any:
    """The SFT seat's schedule factory, beside the builders it dispatches to.

    ``schedule.kind``: ``constant`` = :func:`build_lr_scheduler` (linear warmup over ``opt_cfg.warmup_steps`` then
    constant; ``None`` at warmup 0, the pre-existing path); ``cosine`` = :func:`build_warmup_cosine_lr_scheduler` (the
    backbone's warmup + cosine, cycle = ``schedule.cycle_steps`` or ``max_steps``).
    """
    if schedule.kind == "constant":
        return build_lr_scheduler(optimizer, opt_cfg)
    cycle = int(schedule.cycle_steps or max_steps)
    return build_warmup_cosine_lr_scheduler(
        optimizer,
        warmup_steps=opt_cfg.warmup_steps,
        cycle_steps=cycle,
        f_start=schedule.f_start,
        f_min=schedule.f_min,
    )


def build_wfm_lr_scheduler(optimizer: Any, scheduler_cfg: WFMLRSchedulerConfig) -> Any:
    """Build the pinned Cosmos-RL WFM schedule and bind its exact-resume semantics."""
    import torch

    warmup = int(scheduler_cfg.warm_up_steps)
    cycle = int(scheduler_cfg.cycle_length)
    f_start = float(scheduler_cfg.f_start)
    f_max = float(scheduler_cfg.f_max)
    f_min = float(scheduler_cfg.f_min)

    def factor(step: int) -> float:
        if step < warmup:
            progress = step / warmup if warmup else 1.0
            return f_start + (f_max - f_start) * progress
        progress = max(0.0, (cycle - step) / (cycle - warmup))
        return f_min + (f_max - f_min) * progress

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    bind_scheduler_resume_identity(
        scheduler,
        {
            "version": 1,
            "kind": "cosmos_rl_wfm_lambda_linear",
            "step_timing": "after_applied_optimizer_update_v1",
            "warm_up_steps": warmup,
            "cycle_length": cycle,
            "f_start": f_start,
            "f_max": f_max,
            "f_min": f_min,
        },
    )
    return scheduler


MUON_EXPERT_PARAM_KEYWORDS: tuple[str, ...] = (
    "experts.gate_up_proj",
    "experts.down_proj",
)
"""Raw stacked expert parameters that the shared Muon processes per local expert shard."""

MUON_ORTHOGONALIZE_SKIP_PATTERNS: tuple[str, ...] = (
    r"(?:^|\.)(?:gate|router|cosine_router)(?:\.|$)",
    r"(?:^|\.)gate_up_proj\.weight$",
    r"(?:^|\.)qkv_proj\.weight$",
)
"""Ambiguous or coupled Linear weights that stay on the auxiliary AdamW route."""


@dataclass(frozen=True)
class OptimizerParameterGroup:
    """Explicit named partition of trainable parameters, resolved by the model owner."""

    name: str
    parameter_names: tuple[str, ...]
    lr: float


def _explicit_parameter_groups(
    net: Any, opt_cfg: OptimizerConfig, parameter_groups: tuple[OptimizerParameterGroup, ...]
) -> list[dict[str, Any]]:
    """Validate a model-owned partition of the trainable parameters and emit its optimizer groups."""
    import math

    if opt_cfg.train_keys or opt_cfg.decay_groups or opt_cfg.lr_multipliers:
        raise ValueError(
            "explicit optimizer parameter_groups cannot be combined with train_keys, decay_groups or lr_multipliers"
        )
    names = dict(net.named_parameters())
    selected = [name for group in parameter_groups for name in group.parameter_names]
    if len({group.name for group in parameter_groups}) != len(parameter_groups):
        raise ValueError("optimizer parameter group names must be unique")
    if len(set(selected)) != len(selected):
        raise ValueError("optimizer parameter_groups contain duplicate parameters")
    expected = {name for name, parameter in names.items() if parameter.requires_grad}
    if set(selected) != expected:
        raise ValueError(
            f"optimizer parameter_groups coverage mismatch: missing={sorted(expected - set(selected))}, "
            f"unexpected={sorted(set(selected) - expected)}"
        )
    for group in parameter_groups:
        if not group.name or not group.parameter_names or not math.isfinite(group.lr) or group.lr <= 0:
            raise ValueError(f"invalid optimizer parameter group {group.name!r}")
    return [
        {"params": [names[name] for name in group.parameter_names], "lr": group.lr, "name": group.name}
        for group in parameter_groups
    ]


QWEN_MOE_NO_DECAY_PATTERNS: tuple[str, ...] = (r"(?:^|\.)embed_tokens\.", r"(?:^|\.)lm_head\.", r"\.mlp\.gate\.weight$")
"""Name rules of ``decay_groups="qwen_moe"`` beside the shape rule (every 1-D parameter): the token embedding, the
output head and the MoE router gate of every sparse layer stay undecayed. The seat's own choice, not a convention:
the backbone's recipe (``cosmos3/utils/optimizer.py::_build_params_with_metadata``) decays all 882
tensors and transformers' ``Trainer.get_decay_parameter_names`` decays these three classes; the native recipe's
docstring (knob 4) states what the exemption protects and costs."""


def qwen_moe_decays(name: str, parameter: Any) -> bool:
    """Whether ``decay_groups="qwen_moe"`` applies weight decay to ``parameter`` (see ``OptimizerConfig``).

    Activation-checkpoint and compile wrappers insert ``_checkpoint_wrapped_module.`` / ``_orig_mod.`` segments into
    the names ``named_parameters`` yields; the rules are suffix/segment matches, so they hold with or without them.
    """
    import re

    if getattr(parameter, "ndim", 2) < 2:
        return False
    return not any(re.search(pattern, name) for pattern in QWEN_MOE_NO_DECAY_PATTERNS)


def lr_multiplier_for(name: str, lr_multipliers: Mapping[str, float]) -> float:
    """The factor of the first ``lr_multipliers`` key (dict order) that occurs in ``name``; 1.0 when none does."""
    for pattern, factor in lr_multipliers.items():
        if pattern in name:
            return float(factor)
    return 1.0


def _group_by_decay_and_lr(net: Any, params: list, opt_cfg: OptimizerConfig) -> list[dict[str, Any]]:
    """Partition the trainable parameters into (decays, lr factor) classes, each one optimizer group.

    ``decay_groups="qwen_moe"`` decides the decay side with :func:`qwen_moe_decays`; ``""`` keeps every parameter
    decayed (the one-group behaviour) and only the lr factors split the groups. Group order: the 1.0x classes first
    (decayed before undecayed), then each other DISTINCT factor once, in first-seen ``lr_multipliers`` order (two
    patterns sharing a factor name one class), so ``param_groups[0]["lr"]`` (the reported ``train/learning_rate``) is
    the recipe's base lr. Every group records ``name``, ``lr_multiplier`` and ``n_params`` so a resolved run can be
    read off the optimizer.
    """
    trainable = {id(p) for p in params}
    decay_rule = qwen_moe_decays if opt_cfg.decay_groups == "qwen_moe" else (lambda name, parameter: True)
    classes: dict[tuple[float, bool], list] = {}
    for name, parameter in net.named_parameters():
        if id(parameter) not in trainable:
            continue
        key = (lr_multiplier_for(name, opt_cfg.lr_multipliers), bool(decay_rule(name, parameter)))
        classes.setdefault(key, []).append(parameter)
    # Each distinct factor once, in first-seen order: a factor two patterns share names ONE class; iterating it twice
    # would append its groups twice and trip the count check below.
    order = [1.0]
    for value in opt_cfg.lr_multipliers.values():
        factor = float(value)
        if factor not in order:
            order.append(factor)
    groups: list[dict[str, Any]] = []
    for factor in order:
        for decays in (True, False):
            members = classes.get((factor, decays))
            if not members:
                continue
            groups.append(
                {
                    "params": members,
                    "weight_decay": opt_cfg.weight_decay if decays else 0.0,
                    "lr": opt_cfg.lr * factor,
                    "lr_multiplier": factor,
                    "name": f"{'decay' if decays else 'no_decay'}_lr{factor:g}x",
                    "n_params": len(members),
                }
            )
    if sum(len(group["params"]) for group in groups) != len(params):
        raise ValueError(
            "optimizer decay/lr grouping lost parameters; every trainable parameter must land in one group"
        )
    return groups


def build_optimizer(
    model: Any, opt_cfg: OptimizerConfig, *, parameter_groups: tuple[OptimizerParameterGroup, ...] | None = None
) -> Any:
    """RL owns the optimizer: freeze everything except params whose name contains a
    `train_keys` substring, then the optimizer `opt_cfg.name` selects (`OPTIMIZER_NAMES`) over the
    trainable subset. `train_keys=[]` ⇒ train all `requires_grad` params. Operates on `model.net` (native policy) or the module itself (hf).
    Explicit ``parameter_groups`` must partition every trainable parameter exactly once and
    cannot be combined with train-key filtering or automatic decay grouping.
    """
    import torch

    # Single-mesh FSDP: all of a body's params share one device mesh, so a plain optimizer
    # over `net.named_parameters()` is correct. Mesh-aware per-bucket optimizers (grouping
    # params by device_mesh) are a follow-up for multi-mesh parallelism (TP/EP).
    net = getattr(model, "net", model)
    # An explicit partition is complete by construction; the train-key and decay-group paths
    # below see already-populated groups (and validated-empty knobs) and leave them alone.
    params: list = [] if parameter_groups is None else _explicit_parameter_groups(net, opt_cfg, parameter_groups)
    if opt_cfg.train_keys:
        keys = tuple(opt_cfg.train_keys)
        for name, p in net.named_parameters():
            if any(k in name for k in keys):
                params.append(p)
            else:
                p.requires_grad_(False)
        # train_keys was set but matched nothing → we just froze everything; the all-params
        # fallback below would also be empty and AdamW([]) errors opaquely. Fail loud instead
        # (a model rename / wrong keys is the usual cause).
        if not params:
            raise ValueError(
                f"OptimizerConfig.train_keys={list(opt_cfg.train_keys)} matched no parameter in "
                f"{type(net).__name__} — every param was frozen. Check the keys against the model's "
                f"parameter names, or set train_keys=[] to train all params."
            )
    if not params:
        params = [p for p in net.parameters() if p.requires_grad]
    if opt_cfg.decay_groups == "hf" and opt_cfg.lr_multipliers:
        raise ValueError(
            "OptimizerConfig.decay_groups='hf' cannot be combined with lr_multipliers; use decay_groups='qwen_moe'"
        )
    if opt_cfg.decay_groups == "qwen_moe" or opt_cfg.lr_multipliers:
        params = _group_by_decay_and_lr(net, params, opt_cfg)
    if opt_cfg.decay_groups == "hf":
        # transformers.Trainer.get_decay_parameter_names (trainer.py:1274): everything decays except
        # nn.LayerNorm instances and these name patterns; HF emits the groups as [decay, no-decay].
        import torch.nn as nn
        from transformers.trainer_pt_utils import get_parameter_names

        patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_names = set(get_parameter_names(net, [nn.LayerNorm], patterns))
        trainable = {id(p) for p in params}
        params = [
            {
                "params": [p for n, p in net.named_parameters() if id(p) in trainable and n in decay_names],
                "weight_decay": opt_cfg.weight_decay,
            },
            {
                "params": [p for n, p in net.named_parameters() if id(p) in trainable and n not in decay_names],
                "weight_decay": 0.0,
            },
        ]
    name = getattr(opt_cfg, "name", "adamw")
    if name == "fusedadam":
        # fp32 master weights + fp32 moments (bf16-net correctness — see OptimizerConfig
        # docstring). cosmos-rl reference construction: capturable=True, master_weights=True.
        from cosmos_framework.posttrain.utils.fused_adam import FusedAdam

        return FusedAdam(
            params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=tuple(opt_cfg.betas),
            eps=opt_cfg.eps,
            capturable=True,
            master_weights=True,
        )
    if name == "muon":
        # Reuse the model-stack Muon rather than maintaining a posttrain fork. Its shared
        # classifier admits hidden Linear weights, keeps embeddings/heads/non-Linears on AdamW,
        # and opts the native fused expert tensors into a local expert-shard path. Router weights
        # and coupled 2-D fusions stay on AdamW unless a future model capability declares their
        # matrix semantics explicitly.
        from cosmos_framework.utils.generator.muon_with_aux_adamw import MuonWithAuxAdamW

        optimizer = MuonWithAuxAdamW(
            params,
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            muon_momentum=opt_cfg.muon_momentum,
            muon_lr_scale=opt_cfg.muon_rms_match,
            ns_steps=opt_cfg.muon_ns_steps,
            adam_betas=tuple(opt_cfg.betas),
            eps=opt_cfg.eps,
            use_distributed=True,
            capturable=True,
            expert_param_keywords=MUON_EXPERT_PARAM_KEYWORDS,
            orthogonalize_skip_patterns=MUON_ORTHOGONALIZE_SKIP_PATTERNS,
            split_expert_gate_up=True,
            batch_split_expert_ns=True,
            max_moe_expert_ns_matrices=0,
        )
        optimizer.categorize_params(net)
        if not optimizer.muon_params and not optimizer.stacked_muon_params:
            raise ValueError(
                "OptimizerConfig.name='muon' found no supported hidden Linear weight or stacked expert matrix; "
                "use adamw/fusedadam or revise the model-owned parameter selection"
            )
        return optimizer
    if name != "adamw":
        raise ValueError(f"OptimizerConfig.name={name!r} — expected one of {OPTIMIZER_NAMES}")
    # Kernel selection (cosmos-rl `optm_impl`). Only the REQUESTED flag is passed: torch treats
    # `fused=False`/`foreach=False` as an explicit refusal, not "unspecified", so sending both would
    # pin the for-loop path whenever one was unset. "" sends neither and lets torch choose, which is
    # exactly what this call did before the knob existed.
    impl = str(getattr(opt_cfg, "impl", "") or "")
    kernel: dict[str, bool] = {}
    if impl == "fused":
        kernel["fused"] = True
    elif impl == "foreach":
        kernel["foreach"] = True
    elif impl == "for-loop":
        kernel["foreach"] = False
    return torch.optim.AdamW(
        params,
        lr=opt_cfg.lr,
        weight_decay=opt_cfg.weight_decay,
        betas=tuple(opt_cfg.betas),
        eps=opt_cfg.eps,
        **kernel,
    )


def refuse_rounding_or_unavailable_optimizer(config: Any, opt_cfg: OptimizerConfig) -> None:
    """Refuse an optimizer that cannot train the resolved parameter storage, on either seat (the SFT preflight calls
    it before any model is built; it lives beside :func:`build_optimizer`).

    Storage dtype: on the native seat ``Qwen3VLModelConfig.dtype`` (``build_qwen3_vl_model`` uses it as the
    FSDP master dtype and requires bfloat16 for qwen3_vl_moe); on the HF seat ``resolve_precision(config).master``
    (``rl.master_dtype``, falling back to ``rl.param_dtype`` and then ``model_hf.dtype``). torch ``adamw``
    keeps its moments and the update in the parameter dtype, so below float32 most updates at lr 1e-5 round
    to zero (84.5 % measured on std-0.02 weights; ``config/rl.py::OptimizerConfig``). ``fusedadam`` (apex)
    keeps fp32 master weights and moments inside the optimizer but imports apex only when the trainer
    builds it, so a runtime without apex is refused here, before any model is built.
    """
    import importlib.util

    selection = select_policy_backend(config)
    if selection.backend == "hf":
        storage = str(resolve_precision(config).master)
        seat = "the HF seat (rl.master_dtype, else rl.param_dtype, else model_hf.dtype)"
    elif selection.native_spec is not None:
        storage = str(getattr(selection.native_spec.model_config, "dtype", "") or "the model dtype")
        seat = "the native seat (Qwen3VLModelConfig.dtype is the builder's FSDP master dtype)"
    else:
        return
    name = getattr(opt_cfg, "name", "adamw")
    if name == "adamw" and storage != "float32":
        raise ValueError(
            f"OptimizerConfig(name='adamw') with parameter storage in {storage} on {seat} keeps its moments and "
            "update in that dtype, so most updates at lr 1e-5 round to zero; store parameters in float32 "
            "(rl.master_dtype='float32' on the HF seat) or use OptimizerConfig(name='fusedadam') (apex fp32 master weights)"
        )
    if name == "fusedadam" and importlib.util.find_spec("apex") is None:
        raise ValueError(
            "OptimizerConfig(name='fusedadam') requires apex (apex.optimizers.FusedAdam) in the training runtime and "
            "this runtime has no 'apex' module; use a runtime that carries apex (the RL image's /opt/venv/cosmos_rl does) "
            "or store parameters in float32 and use OptimizerConfig(name='adamw')"
        )


@dataclass(frozen=True)
class Precision:
    """The four resolved dtype NAMES for one training run (strings, not `torch.dtype`), plus how the
    compute dtype is delivered.

    Strings so this stays importable without torch, matching `config/rl.py`'s dependency-light
    contract; `build_hf_model` does the `getattr(torch, ...)`.
    A model-owned native mixed-precision preset may use None for storage, compute and reduction:
    the model declares its per-module dtypes and FSDP preserves them without a blanket cast.
    The shared HF resolver always returns concrete dtype names.

    Attributes:
        master: STORAGE dtype — parameters between steps, and therefore optimizer moment dtype.
        param: COMPUTE dtype — `MixedPrecisionPolicy.param_dtype` for the forward/backward.
        reduce: gradient all-reduce dtype.
        transfer: dtype weights are cast to on the policy -> rollout hop, or None to preserve
            each parameter's storage dtype for a model-owned mixed-precision preset.
        mode: "fsdp" (MixedPrecisionPolicy parameter cast, the default) or "autocast"
            (`torch.autocast(param)` around the forward, parameters left in `master`).
    """

    master: str | None
    param: str | None
    reduce: str | None
    transfer: str | None
    mode: str = "fsdp"


def resolve_precision(config: Any) -> Precision:
    """Resolve ``rl.*_dtype`` against the selected model config — the ONE fallback chain.

    Single implementation on purpose: the `""`-means-inherit rule and the deprecation warning
    would otherwise be re-derived at each call site and drift, which is exactly how
    `model_hf.fsdp_reduce_dtype` became a knob nothing read.

    Chain (see `config/rl.py::HFRunConfig`):
        param    <- rl.param_dtype    or model_hf.dtype              (compute)
        master   <- rl.master_dtype   or the resolved param          (storage; pre-split behaviour)
        reduce   <- rl.reduce_dtype   or model_hf.fsdp_reduce_dtype  (grad all-reduce)
        transfer <- rl.transfer_dtype or the resolved master         (cosmos-rl's own default)
        mode     <- rl.mixed_precision_mode or "fsdp"                (how `param` reaches the forward)

    Args:
        config: The root config. ``config.model_config`` may provide an already-selected native or
            HF model config; otherwise the root ``model_hf`` compatibility field is used.

    Returns:
        A `Precision` of dtype NAMES.
    """
    rl = getattr(config, "rl", None)
    selected_model_config = getattr(config, "model_config", None)
    fallback_model_config = selected_model_config or getattr(config, "model_hf", None)

    def _rl(name: str) -> str:
        return str(getattr(rl, name, "") or "")

    # The typed model configs share these fallback field names. A config missing either section
    # retains the historical bfloat16/float32 defaults.
    param = _rl("param_dtype") or str(getattr(fallback_model_config, "dtype", "") or "bfloat16")
    master = _rl("master_dtype") or param
    reduce = _rl("reduce_dtype") or str(getattr(fallback_model_config, "fsdp_reduce_dtype", "") or "float32")
    transfer = _rl("transfer_dtype") or master
    mode = _rl("mixed_precision_mode") or "fsdp"
    for field_name in ("optimizer", "critic_optimizer"):
        optimizer = getattr(rl, field_name, None)
        if getattr(optimizer, "name", None) == "muon" and master != "float32":
            raise ValueError(
                f"rl.{field_name}.name='muon' requires float32 parameter storage, but the resolved "
                f"master dtype is {master!r}. Set rl.master_dtype=float32 and keep BF16 compute with "
                "rl.param_dtype=bfloat16, or choose adamw/fusedadam."
            )
    if mode == "autocast" and param == "float16":
        # Checked on the RESOLVED dtype: an unset rl.param_dtype inherits model_hf.dtype, so
        # HFRunConfig.__post_init__ cannot see it. fp16 autocast underflows gradients without a
        # GradScaler, which the trainer does not have.
        raise ValueError(
            "rl.mixed_precision_mode='autocast' with a float16 compute dtype needs a GradScaler; "
            "use bfloat16, or mode 'fsdp'"
        )

    backend = normalize_model_backend(str(getattr(config, "model_backend", HF)))
    if (
        selected_model_config is None
        and backend == HF
        and not _rl("param_dtype")
        and getattr(fallback_model_config, "dtype", None)
    ):
        # Not an error: the deprecated field is still the documented fallback. Warned once per
        # build so a run that silently depends on it is visible in its own log.
        logger.warning(
            "model_hf.dtype is DEPRECATED as a precision knob (it sets storage AND compute at "
            "once, which cannot express fp32-master/bf16-compute). Set rl.param_dtype "
            "(compute) and rl.master_dtype (storage) instead — see config/rl.py::HFRunConfig."
        )
    return Precision(master=master, param=param, reduce=reduce, transfer=transfer, mode=mode)


REPLICA_INDEX_ENV = "POSTTRAIN_REPLICA_INDEX"
"""Env var carrying THIS cell's data-parallel replica index (see `replica_coords`)."""

REPLICA_COUNT_ENV = "POSTTRAIN_REPLICA_COUNT"
"""Env var carrying the colocation's replica count (see `replica_coords`)."""


def replica_coords() -> tuple[int, int]:
    """`(replica_index, replica_count)` for the cell this process hosts; `(0, 1)` when unplaced.

    Role bodies that own per-replica state must shard it on these. A dataset is the clearest case:
    every replica constructing the same reader with the same cursor emits the SAME rows, so the
    data-parallel axis multiplies compute without adding data and the cross-replica all-reduce
    averages gradients computed from duplicate samples. That is not a crash — it is a run that looks
    healthy and means nothing.

    `(0, 1)` is the honest answer outside a placed actor (torchrun, local backend, plain tests):
    one replica, so no sharding, which is what the single-replica path has always done.

    Returns:
        `(replica_index, replica_count)` with `0 <= replica_index < replica_count`.

    Raises:
        ValueError: if the published coordinates are inconsistent.
    """
    import os

    index = int(os.environ.get(REPLICA_INDEX_ENV, "0"))
    count = int(os.environ.get(REPLICA_COUNT_ENV, "1"))
    if count < 1 or not (0 <= index < count):
        raise ValueError(
            f"inconsistent replica coordinates {REPLICA_INDEX_ENV}={index}, {REPLICA_COUNT_ENV}={count}: "
            f"need 0 <= index < count and count >= 1"
        )
    return index, count


def _resolve_data_parallelism(
    dp_replicate: int,
    dp_shard: int,
    role_ranks: int,
    *,
    field_name: str,
    ranks_field_name: str,
) -> tuple[int, int]:
    """Resolve one model-owned data-parallel factorization inside a role rank world."""
    if type(role_ranks) is not int or role_ranks < 1:
        raise ValueError(f"{ranks_field_name} must be a positive integer, got {role_ranks!r}")
    for dimension, degree in (("dp_replicate", dp_replicate), ("dp_shard", dp_shard)):
        if type(degree) is not int or degree < -1 or degree == 0:
            raise ValueError(
                f"{field_name}.{dimension} must be a positive integer or -1 for automatic inference, got {degree!r}"
            )
    if dp_replicate == -1 and dp_shard == -1:
        dp_replicate, dp_shard = 1, role_ranks
    elif dp_replicate == -1:
        if role_ranks % dp_shard:
            raise ValueError(
                f"cannot infer {field_name}.dp_replicate: {ranks_field_name}={role_ranks} is not divisible by "
                f"{field_name}.dp_shard={dp_shard}"
            )
        dp_replicate = role_ranks // dp_shard
    elif dp_shard == -1:
        if role_ranks % dp_replicate:
            raise ValueError(
                f"cannot infer {field_name}.dp_shard: {ranks_field_name}={role_ranks} is not divisible by "
                f"{field_name}.dp_replicate={dp_replicate}"
            )
        dp_shard = role_ranks // dp_replicate
    if dp_replicate * dp_shard != role_ranks:
        raise ValueError(
            f"{field_name} dp_replicate={dp_replicate} * dp_shard={dp_shard} must equal {ranks_field_name}={role_ranks}"
        )
    return dp_replicate, dp_shard


def resolve_hf_model_parallelism(
    model_config: HFModelConfig,
    role_ranks: int,
    *,
    field_name: str,
    ranks_field_name: str,
) -> tuple[int, int]:
    """Resolve ``HFModelConfig.parallelism`` inside one model role's K-rank world.

    Args:
        model_config: Hugging Face model configuration owned by the role.
        role_ranks: K, the worker ranks in the role's sole execution replica.
        field_name: Fully qualified parallelism field used in validation errors.
        ranks_field_name: Fully qualified K field used in validation errors.

    Returns:
        The normalized positive ``(dp_replicate, dp_shard)`` dimensions.

    Raises:
        ValueError: A degree is invalid, cannot be inferred, or does not factor ``role_ranks``.
    """
    parallelism = model_config.parallelism
    return _resolve_data_parallelism(
        parallelism.dp_replicate,
        parallelism.dp_shard,
        role_ranks,
        field_name=field_name,
        ranks_field_name=ranks_field_name,
    )


def resolve_policy_parallelism(config: Any, policy_ranks: int) -> tuple[int, int]:
    """Normalize the configured policy DeviceMesh to ``(dp_replicate, dp_shard)``.

    The model backend owns the factorization. The HF adapter reads its typed ``model_hf.parallelism``
    section. A generic native-policy spec reads ``model.parallelism``; the legacy native model tree
    retains its ``model.config.parallelism`` source. Placement supplies only ``policy_ranks`` for
    validation and never invents a configurable dimension.

    Args:
        config: resolved posttrain root config.
        policy_ranks: K in the policy WorkerGroup shape ``(1, K)``.

    Returns:
        The normalized positive ``(dp_replicate, dp_shard)`` dimensions.

    Raises:
        ValueError: a dimension is invalid or their product does not equal ``policy_ranks``.
        AttributeError: the selected model backend has no parallelism section.
    """
    selection = select_policy_backend(config)
    backend = selection.backend
    native_spec = selection.native_spec
    if backend == HF:
        model_config = getattr(config, "model_hf")
        return resolve_hf_model_parallelism(
            model_config,
            policy_ranks,
            field_name="config.model_hf.parallelism",
            ranks_field_name="POLICY_RANKS",
        )
    elif native_spec is not None:
        parallelism = getattr(native_spec, "parallelism")
        replicate = int(getattr(parallelism, "dp_replicate"))
        shard = int(getattr(parallelism, "dp_shard"))
    else:
        model = getattr(config, "model")
        if model is None:
            raise AttributeError("model_backend='i4' requires config.model before resolving policy parallelism")
        parallelism = getattr(getattr(model, "config"), "parallelism")
        for field_name in ("context_parallel_shard_degree", "cfg_parallel_shard_degree"):
            degree = int(getattr(parallelism, field_name, 1))
            if degree != 1:
                raise ValueError(
                    f"config.model.config.parallelism.{field_name}={degree} is unsupported in posttrain: "
                    "POLICY_RANKS currently contains only dp_replicate * dp_shard"
                )
        replicate = int(getattr(parallelism, "data_parallel_replicate_degree"))
        shard = int(getattr(parallelism, "data_parallel_shard_degree"))

    replicate, shard = _resolve_data_parallelism(
        replicate,
        shard,
        policy_ranks,
        field_name="policy DeviceMesh",
        ranks_field_name="POLICY_RANKS",
    )
    # A legacy native tree must consume the same normalized factorization the trainer WG carries.
    # Generic native and HF specs retain their inference sentinel so this helper remains repeatable
    # when a caller validates the same config against another policy width.
    if backend == I4 and native_spec is None:
        parallelism.data_parallel_replicate_degree = replicate
        parallelism.data_parallel_shard_degree = shard
    return replicate, shard


def resolve_pdims(
    device: str | None = None,
    *,
    parallelism: tuple[int, int] | None = None,
) -> Any:
    """The policy data-parallel mesh: injected, explicitly configured, or pure-FSDP fallback.

    Resolution order:

    1. The Ray placement layer's per-actor mesh (`current_pdims()`), which `run_setup` injects.
    2. An explicit ``(dp_replicate, dp_shard)`` factorization over the live process group.
    3. Pure FSDP ``(1, world_size)`` for compatibility callers that supply no factorization.
    4. `None` — an uninjected single device, so there is nothing to shard.

    Placement R is intentionally absent. A policy has one execution replica and its complete Torch
    world is K; model config owns how K factors into the two DP dimensions. The fallback preserves
    existing plain-torchrun and direct-test behavior but cannot express HSDP; production entrypoints
    pass the normalized model-owned factor explicitly.

    Args:
        device: device type for `build_device_mesh`; defaults to cuda when available, else cpu. The
            Ray path builds the mesh in `set_mesh`; the local path must build it here so
            `parallelize()`/`load_weights()` see a live mesh.
        parallelism: explicit ``(dp_replicate, dp_shard)`` factorization. ``None`` means pure FSDP.

    Returns:
        A `ParallelDims`; its DeviceMesh is built for multi-rank worlds, while Ray's injected
        single-rank descriptor is mesh-free. Returns None for an uninjected single-device process.

    Raises:
        ValueError: the explicit or injected factorization disagrees with the live world.
    """
    try:
        from cosmos_framework.posttrain.workers.ray import current_pdims
    except ImportError:  # ray is optional — the local/torchrun backend never needs it
        pass
    else:
        if (injected := current_pdims()) is not None:
            if parallelism is not None and (injected.dp_replicate, injected.dp_shard) != parallelism:
                raise ValueError(
                    "injected policy DeviceMesh disagrees with model config: "
                    f"injected={(injected.dp_replicate, injected.dp_shard)}, configured={parallelism}"
                )
            return injected

    import torch
    import torch.distributed as dist

    # The mesh must span the LIVE process group (init_device_mesh tiles the whole group), so the
    # world comes from dist, never from a caller-supplied argument that may be stale.
    world = dist.get_world_size() if dist.is_initialized() else 1
    replicate, shard = parallelism or (1, world)
    if replicate < 1 or shard < 1 or replicate * shard != world:
        raise ValueError(
            f"policy DeviceMesh dp_replicate={replicate} * dp_shard={shard} must equal "
            f"the live policy world size {world}"
        )
    if world <= 1:
        return None

    from cosmos_framework.posttrain.model.parallel_dims import ParallelDims

    pdims = ParallelDims(world_size=world, dp_replicate=replicate, dp_shard=shard)
    pdims.build_device_mesh(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return pdims


def _resolve_dotted_callable(path: str) -> Any:
    """Resolve one absolute dotted callable without importing model families at config import time."""
    import importlib

    module_name, separator, attribute_name = path.rpartition(".")
    if not separator or not module_name or not attribute_name:
        raise ValueError(f"native policy builder must be an absolute dotted callable, got {path!r}")
    module = importlib.import_module(module_name)
    builder = getattr(module, attribute_name, None)
    if not callable(builder):
        raise TypeError(f"native policy builder {path!r} is not callable")
    return builder


def _build_native_from_spec(config: Any, pdims: Any, *, builder_field: str) -> tuple[Any, Any]:
    """Invoke one model-family builder from the selected native spec."""
    from cosmos_framework.utils.lazy_config import instantiate

    spec = select_policy_backend(config).native_spec
    if spec is None:
        raise TypeError(
            "online model_backend='i4' requires config.model=NativePolicyBuildSpec; "
            "legacy config.model trees remain supported by build_backbone for non-online callers"
        )
    builder_path = str(getattr(spec, builder_field, "") or "")
    if not builder_path:
        raise ValueError(f"config.model.{builder_field} must be configured for this native role")
    model_config = instantiate(getattr(spec, "model_config"))
    builder = _resolve_dotted_callable(builder_path)
    return builder(str(getattr(spec, "model_id")), model_config, pdims), model_config


def build_native_policy(config: Any, pdims: Any = None) -> tuple[Any, Any]:
    """Build the generic native online policy and return the exact builder config used."""
    spec = select_policy_backend(config).native_spec
    if spec is None:
        raise TypeError(
            "online model_backend='i4' requires config.model=NativePolicyBuildSpec; "
            "legacy config.model trees remain supported by build_backbone for non-online callers"
        )
    policy_world = int(pdims.world_size) if pdims is not None else world_size_or_1()
    configured_parallelism = resolve_policy_parallelism(config, policy_world)
    if pdims is None:
        pdims = resolve_pdims(parallelism=configured_parallelism)
    elif (int(pdims.dp_replicate), int(pdims.dp_shard)) != configured_parallelism:
        raise ValueError(
            "explicit policy mesh disagrees with config.model.parallelism: "
            f"pdims={(pdims.dp_replicate, pdims.dp_shard)}, configured={configured_parallelism}"
        )
    return _build_native_from_spec(config, pdims, builder_field="builder")


def build_native_model(config: Any, pdims: Any = None) -> tuple[Any, Any]:
    """Build a non-policy native role on its already-injected role mesh."""
    return _build_native_from_spec(config, pdims, builder_field="builder")


def build_native_value_model(config: Any, pdims: Any = None) -> tuple[Any, Any]:
    """Build a native scalar-value model on its already-injected critic mesh."""
    return _build_native_from_spec(config, pdims, builder_field="value_builder")


def build_backbone(config: Any, world_size: int = 1, pdims: Any = None) -> Any:
    """Backend dispatch to a legacy native model, generic native policy, or HF model.

    A legacy ``model_backend='i4'`` config still instantiates ``config.model`` and moves it to CUDA.
    When ``config.model`` is a :class:`NativePolicyBuildSpec`, its dotted builder owns construction
    and receives the resolved policy mesh. ``qwen3_vl`` is normalized to that same generic seam.

    For HF and generic native policies, normalized backend parallelism is authoritative. An
    explicitly passed ``pdims`` must match it; otherwise the helper builds the configured mesh from
    the live world.

    `world_size` is accepted for the entry's positional call signature but is NOT the source of
    truth for the FSDP mesh: the mesh MUST span the live process group, so a stale/default
    parameter (e.g. an entry calling `build_backbone(config)` in a 4-GPU torchrun run) can't be
    allowed to silently drop FSDP and build an unsharded single-device model on every rank."""
    from cosmos_framework.utils.lazy_config import instantiate

    selection = select_policy_backend(config)
    backend = selection.backend
    if backend == I4:
        if selection.native_spec is None:
            return instantiate(config.model).cuda()
        # Native construction owns its mesh resolution. Keeping that in one place avoids validating
        # the same generic spec once here and again immediately inside build_native_policy().
        return build_native_policy(config, pdims)[0]

    policy_world = pdims.world_size if pdims is not None else world_size_or_1()
    parallelism = resolve_policy_parallelism(config, int(policy_world))
    if pdims is None:
        pdims = resolve_pdims(parallelism=parallelism)
    elif (pdims.dp_replicate, pdims.dp_shard) != parallelism:
        raise ValueError(
            "explicit policy mesh disagrees with model_hf.parallelism: "
            f"pdims={(pdims.dp_replicate, pdims.dp_shard)}, configured={parallelism}"
        )

    from cosmos_framework.posttrain.model.hf import build_hf_model

    # `precision` splits STORAGE from COMPUTE; without it the one `model_hf.dtype` drives both
    # and fp32-master/bf16-compute -- what the reference GRPO recipe runs -- is inexpressible.
    return build_hf_model(instantiate(config.model_hf), pdims, precision=resolve_precision(config))


def vllm_engine_kwargs(
    cfg: RolloutConfig,
    *,
    trust_remote_code: bool = False,
    return_routed_experts: bool = False,
    topk_logprobs: int = 0,
) -> dict:
    """Pure config → ``vllm.LLM(**kwargs)`` mapping (no vllm import — CPU-testable). TP>1 adds the
    SPMD wiring the grid relies on: ``external_launcher`` reuses the per-replica rendezvous group as
    the TP group (every rank returns the same full output), and custom all-reduce stays disabled
    unless ``cfg.custom_all_reduce``, because per-actor GPU isolation breaks vLLM's P2P probe; the
    entry that enables it must expose the TP peers' GPUs (``_require_visible_tensor_parallel_peers``).

    Args:
        cfg: rollout config supplying the engine knobs (`name_or_path`, `gpu_memory_utilization`,
            `max_model_len`, `dtype`, `load_format`, `enforce_eager`, `moe_backend`, `trust_remote_code`,
            optional decoder `attention_backend`, `seed`, `tensor_parallel_size`,
            `enable_expert_parallel`, the optional per-request multimodal budget
            `limit_mm_per_prompt`, and the optional performance knobs `compilation_config`,
            `async_scheduling`, `max_num_seqs` and `kv_cache_memory_bytes`).
        trust_remote_code: the policy-owned HF trust decision forwarded to vLLM. It is OR-ed with
            `cfg.trust_remote_code`: the rollout config and the policy each own an explicit opt-in
            (both default to False), so neither can silently cancel the other's trust grant.
        return_routed_experts: make the engine return the per-token top-k expert ids it routed
            through (``CompletionOutput.routed_experts``), the record `rl.routing_replay='rollout'`
            replays. Emitted only when True, so the engine construction identity of every other run
            is unchanged.
        topk_logprobs: > 0 makes the engine able to return that many top-k logprobs per generated
            token (``max_logprobs``, vLLM's default is 20) in ``processed_logprobs`` mode — the
            log-softmax of the logits AFTER temperature (and any top-p/top-k mask, which the on-policy
            contract forbids), i.e. the distribution the token was sampled from. This is what
            `rl.score_centering` needs for its sampler head; 0 (default) leaves both keys absent, so
            the engine identity of every other run is unchanged and the recorded sampled-token
            logprobs stay vLLM's raw temperature-1 values.

    Returns:
        The `vllm.LLM(**kwargs)` keyword dict (prefix caching forced off; TP>1 adds the SPMD wiring)."""
    kw: dict = dict(
        model=cfg.name_or_path,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,
        enforce_eager=cfg.enforce_eager,
        trust_remote_code=bool(trust_remote_code or cfg.trust_remote_code),
        runner="generate",
        convert="none",
        enable_expert_parallel=cfg.enable_expert_parallel,
        moe_backend=cfg.moe_backend,
        # Forced off (vLLM defaults it to True): weight sync does NOT invalidate the prefix cache,
        # so a cached prefix (system prompt) would reuse KV computed under PRE-sync weights —
        # silently off-policy logprobs. Deliberately NOT a RolloutConfig knob until apply_weights
        # resets the cache after each sync.
        enable_prefix_caching=False,
    )
    if cfg.seed is not None:
        kw["seed"] = cfg.seed
    if cfg.load_format != "auto":
        kw["load_format"] = cfg.load_format
    if cfg.revision is not None:
        kw["revision"] = cfg.revision
    if cfg.tokenizer_revision is not None:
        kw["tokenizer_revision"] = cfg.tokenizer_revision
    if cfg.attention_backend is not None:
        kw["attention_config"] = {"backend": cfg.attention_backend}
    if cfg.mm_encoder_attn_backend is not None:
        kw["mm_encoder_attn_backend"] = cfg.mm_encoder_attn_backend
    if cfg.limit_mm_per_prompt is not None:
        # vLLM's own name; validated in RolloutConfig.__post_init__. A copy, so the engine never aliases the config.
        kw["limit_mm_per_prompt"] = dict(cfg.limit_mm_per_prompt)
    if cfg.compilation_config:
        # A deep copy: vLLM builds its CompilationConfig from this mapping and its capture-size lists.
        kw["compilation_config"] = copy.deepcopy(cfg.compilation_config)
    for knob in ("async_scheduling", "max_num_seqs", "kv_cache_memory_bytes"):
        value = getattr(cfg, knob)
        if value is not None:
            kw[knob] = value
    if return_routed_experts:
        kw["enable_return_routed_experts"] = True
    if isinstance(topk_logprobs, bool) or not isinstance(topk_logprobs, int) or topk_logprobs < 0:
        raise ValueError(f"topk_logprobs must be a non-negative integer, got {topk_logprobs!r}")
    if topk_logprobs > 0:
        from cosmos_framework.posttrain.roles.rollout.sampler_head import SamplerHeadSpec

        kw.update(SamplerHeadSpec(topk_logprobs).engine_kwargs())  # max_logprobs=k, logprobs_mode=processed_logprobs
    if cfg.tensor_parallel_size > 1:
        kw.update(
            tensor_parallel_size=cfg.tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=not cfg.custom_all_reduce,
        )
    return kw


def ppo_vllm_engine_kwargs(cfg: RolloutConfig, *, behavior_top_k: int) -> dict:
    """Add PPO's behavior-distribution requirements to ordinary vLLM kwargs.

    vLLM's raw-logprob mode reports logits before sampling temperature and its
    default top-k cap is too small for the default JSD support. PPO ratios score
    HF logits at the shared temperature, so the engine must report processed
    log-probabilities under the same distribution.
    """
    if isinstance(behavior_top_k, bool) or not isinstance(behavior_top_k, int) or behavior_top_k < 1:
        raise ValueError(f"behavior_top_k must be a positive integer, got {behavior_top_k!r}")
    if cfg.custom_all_reduce:
        # PPO rollout actors keep one visible GPU (`validate_ppo_config` rejects this before workers start).
        raise ValueError("PPO rollout engines do not support rollout.custom_all_reduce")
    from cosmos_framework.posttrain.roles.rollout.sampler_head import SamplerHeadSpec

    return {**vllm_engine_kwargs(cfg), **SamplerHeadSpec(behavior_top_k).engine_kwargs()}


def sampling_params(cfg: RolloutConfig) -> dict:
    """The per-call SamplingParams subset a VLLMRollout owns from config. ``n``/``logprobs`` are
    framework-owned (n = GRPO group size per generate call, logprobs = π_old) and never appear here
    — VLLMRollout rejects them in a user sampling dict.

    Every knob is emitted, including neutral ones — see `hf_sampling_params` for why, and
    `RolloutConfig.top_k` for the underlying hazard.

    Args:
        cfg: the rollout config.

    Returns:
        kwargs for ``vllm.SamplingParams``.
    """
    return {
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        "top_k": cfg.top_k if cfg.top_k > 0 else -1,  # vLLM spells "disabled" as -1
        "repetition_penalty": cfg.repetition_penalty,
        "max_tokens": cfg.max_tokens,
    }


def hf_sampling_params(cfg: RolloutConfig) -> dict:
    """The same `RolloutConfig` mapped to HF ``generate`` kwargs — the sibling of `sampling_params`.

    Lives HERE, next to the vLLM mapping, rather than beside whichever example happens to build an
    HFRollout: the two backends must sample from the SAME distribution for a given config, and that
    invariant is only checkable when both mappings are read together. They differ in spelling only
    (HF wants ``max_new_tokens``/``top_k=None``, vLLM wants ``max_tokens``/``top_k=-1``).

    EVERY knob is emitted, including ones at their neutral value, because `generate` merges these
    kwargs onto the model object's ``generation_config`` — an omitted knob silently inherits from
    that object rather than from this config (see `RolloutConfig.top_k`).

    Args:
        cfg: the rollout config.

    Returns:
        kwargs for ``model.generate``.
    """
    return {
        "do_sample": True,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        # None (not 0/-1) is HF's spelling of "disabled": `_get_logits_processor` adds a
        # TopKLogitsWarper unless top_k is None or 0, and passing None OVERRIDES the model object's
        # value rather than falling back to it.
        "top_k": cfg.top_k if cfg.top_k > 0 else None,
        "repetition_penalty": cfg.repetition_penalty,
        "max_new_tokens": cfg.max_tokens,
    }


def _require_visible_tensor_parallel_peers(cfg: RolloutConfig) -> None:
    """Refuse custom all-reduce in an engine process that can see only its own GPU.

    vLLM maps every tensor-parallel peer's buffers over CUDA IPC and fails under Ray's default
    one-GPU isolation, so an entry that enables it must build the engine's actors with
    ``ActorGroup(expose_peer_gpus=True)``, as GRPO's experience group does.
    """
    if not (cfg.custom_all_reduce and cfg.tensor_parallel_size > 1):
        return
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = len([entry for entry in raw.split(",") if entry.strip()]) if raw is not None else None
    if visible is not None and visible != cfg.tensor_parallel_size:
        raise ValueError(
            "rollout.custom_all_reduce needs exactly its tensor-parallel world visible to each engine process, "
            f"but tensor_parallel_size={cfg.tensor_parallel_size} and CUDA_VISIBLE_DEVICES={raw!r}; "
            "use a full-node ActorGroup(expose_peer_gpus=True)"
        )


def build_vllm_rollout(
    cfg: RolloutConfig,
    packer: Any,
    *,
    trust_remote_code: bool = False,
    return_routed_experts: bool = False,
    enable_sleep_mode: bool = True,
    topk_logprobs: int = 0,
) -> Any:
    """Build the vLLM engine + `VLLMRollout` body from config. Import vllm lazily so this module
    stays importable without it; call INSIDE the hosting actor (the engine must live where the GPU
    is, and under TP the per-replica rendezvous group must already be up — spawn_grid Pass 2).

    Args:
        cfg: the rollout config (engine kwargs + sampling params).
        packer: REQUIRED DataPacker (`init_data_packer(cfg.model.name_or_path)`) — the single
            prompt-shaping path. Required, not defaulted: a `build_vllm_rollout(cfg)` with no
            packer would build the engine and every actor around it and only fail on the first
            generate(), and it would silently skip the packer's special-token consistency guard.
        trust_remote_code: the policy-owned HF trust decision used by both vLLM and the model config
            recorded in the engine construction identity.
        return_routed_experts: build the engine to return its per-token expert routing and the body
            to record it (`rollout_records_routing`); part of the engine construction identity.
        enable_sleep_mode: build the engine on vLLM's sleep-mode allocator so the loop can release
            its weights and KV cache between phases (`VLLMRollout.sleep`). The GRPO entry passes
            False only for a colocated run that opted out of the hand-off cycle
            (`rl.colocated_offload=False`); disaggregated placement keeps the allocator it always
            had (sleep mode on, never slept).
        topk_logprobs: `rl.score_centering.top_k` when score centering is on, else 0: build the
            engine to return processed top-k logprobs (`vllm_engine_kwargs`) and the body to request
            and record the sampler head on every training completion (`rollout_topk_logprobs`).

    Returns:
        A `VLLMRollout` bound to a freshly built `vllm.LLM` engine.
    """
    from cosmos_framework.posttrain.roles.rollout import VLLMRollout
    from cosmos_framework.posttrain.roles.rollout.validation import (
        normalize_validation_semantics_identity,
        validation_semantics_equal,
    )
    from cosmos_framework.posttrain.utils.dist import run_rank_converged
    from cosmos_framework.posttrain.utils.identity import normalize_typed_json_dict
    from cosmos_framework.posttrain.utils.rank_state import gather_rank_states

    def prepare() -> tuple[Any, dict[str, Any], dict[str, Any]]:
        _require_visible_tensor_parallel_peers(cfg)
        _configure_vllm_runtime()
        from vllm import LLM

        engine_config = normalize_validation_semantics_identity(
            vllm_engine_kwargs(
                cfg,
                trust_remote_code=trust_remote_code,
                return_routed_experts=return_routed_experts,
                topk_logprobs=topk_logprobs,
            ),
            context="vLLM engine construction config",
        )
        resolved_model_config = normalize_typed_json_dict(
            _resolve_hf_config_with_local_retry(
                cfg.name_or_path,
                trust_remote_code=trust_remote_code,
            ).to_dict(),
            context="vLLM resolved model config",
        )
        return LLM, engine_config, resolved_model_config

    llm_class, engine_config, resolved_model_config = run_rank_converged(
        prepare,
        "vLLM engine identity preparation",
    )

    construction_identity = {
        "engine_config": engine_config,
        "resolved_model_config": resolved_model_config,
    }
    by_rank = gather_rank_states(
        construction_identity,
        context="vLLM engine construction identity preparation",
    )
    if any(not validation_semantics_equal(candidate, construction_identity) for candidate in by_rank.values()):
        raise RuntimeError(f"vLLM engine construction identity differs across ranks: {by_rank!r}")

    # Sleep mode only swaps vLLM's allocator so the loop can release the engine between phases; it
    # does not change generation semantics, so it stays out of the recorded engine identity.
    engine = llm_class(**engine_config, enable_sleep_mode=bool(enable_sleep_mode))

    def verify_model_config() -> None:
        post_config = normalize_typed_json_dict(
            _resolve_hf_config_with_local_retry(
                cfg.name_or_path,
                trust_remote_code=trust_remote_code,
            ).to_dict(),
            context="post-construction vLLM resolved model config",
        )
        if not validation_semantics_equal(post_config, resolved_model_config):
            raise RuntimeError("vLLM model config changed while the engine was being constructed")

    run_rank_converged(verify_model_config, "vLLM post-construction model identity verification")
    return VLLMRollout(
        engine,
        packer=packer,
        sampling=sampling_params(cfg),
        model_identifier=cfg.name_or_path,
        engine_config=engine_config,
        resolved_model_config=resolved_model_config,
        record_routed_experts=return_routed_experts,
        sleep_level=cfg.sleep_level,
        topk_logprobs=topk_logprobs,
    )


def configure_ppo_vllm_environment(cfg: RolloutConfig) -> None:
    """Apply PPO-owned vLLM process globals before the first vLLM import in this process."""
    import os

    if type(cfg.batch_invariant) is not bool:
        raise TypeError(f"PPO rollout.batch_invariant must be a bool, got {type(cfg.batch_invariant).__name__}")
    if cfg.batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    else:
        os.environ.pop("VLLM_BATCH_INVARIANT", None)


def build_ppo_vllm_rollout(
    cfg: RolloutConfig,
    packer: Any,
    *,
    behavior_top_k: int,
) -> Any:
    """Build the strict PPO vLLM adapter inside its rollout TP actor."""
    if cfg.seed is not None and type(cfg.seed) is not int:
        raise ValueError(f"PPO sampling seed must be an exact integer or None, got {cfg.seed!r}")
    configure_ppo_vllm_environment(cfg)
    from vllm import LLM

    from cosmos_framework.posttrain.roles.rollout import PPOVLLMRollout

    engine_config = ppo_vllm_engine_kwargs(cfg, behavior_top_k=behavior_top_k)
    return PPOVLLMRollout(
        LLM(**engine_config),
        packer=packer,
        sampling=sampling_params(cfg),
        behavior_top_k=behavior_top_k,
        seed=cfg.seed,
        engine_config=dict(engine_config),  # the construction identity the head contract is verified against
    )


def _dp_coords(pdims: Any) -> tuple[int, int]:
    """`(dp_rank, dp_world)` from a `ParallelDims`, or `(0, 1)` when there is no mesh.

    `(0, 1)` makes `select_chunk`'s dp interleave an exact no-op, so the single-device / CPU path is
    unchanged. The caller is `workers/ray.py::_HostImpl.invoke` (actor-side, from the actor's own
    injected mesh) — the driver cannot resolve this itself: it knows the cell `(r, k)` but not how to
    map `k -> dp_rank` without the parallelism factorization `ParallelDims` carries.

    Reports the complete policy-DP extent ``dp_replicate * dp_shard``. Policy placement has exactly
    one execution replica, so replay dispatch produces one batch and this function is its sole DP
    partitioning owner. Returning only ``dp_shard`` would hand the same samples to corresponding
    shard coordinates in different ``dp_replicate`` rows.

    Args:
        pdims: this actor's `ParallelDims`, or None.

    Returns:
        The flattened ``(dp_rank, dp_world)`` over both DeviceMesh DP dimensions.
    """
    if pdims is None or not getattr(pdims, "dp_enabled", False):
        return 0, 1
    mesh = pdims.dp_mesh
    if mesh is None:
        return 0, 1
    # `get_local_rank()` REQUIRES an explicit mesh_dim once a mesh is multi-dimensional (a no-arg
    # call raises on ndim > 1). Fold row-major over `(dp_replicate, dp_shard)` so every policy rank
    # owns one stable coordinate in `[0, dp_world)`.
    rank = 0
    for dim in range(mesh.ndim):
        rank = rank * mesh.size(dim) + mesh.get_local_rank(dim)
    return rank, mesh.size()


def build_objective(
    rl: GRPOConfig,
    name_or_path: str,
    pdims: Any = None,
    packer: Any = None,
    *,
    max_response_length: int | None = None,
) -> "Objective":
    """Map a GRPO config to its `Objective` — the single config-to-surrogate dispatch.

    Why this lives in config/ and not in algorithm/: `algorithm/` takes primitives and imports no
    config, so the algorithm is usable without the config system at all. Keeping the config→primitive
    mapping here preserves that and keeps both surrogate mappings readable together.

    `pdims` no longer changes what this function returns (Task 9 deleted `dp_group_sharded`, the
    objective wrapper that used to divide the dp axis here when `_dp_coords(pdims)` reported an
    extent > 1). Partitioning the dp axis is now the ingress predicate's job alone
    (`workers/dispatch.py::select_chunk`, applied by the WorkerGroup `train` edge when its
    `mini`/`micro`/`n_mini`/`n_micro` control kwargs are passed — spec 2026-08-12 §3.4); wrapping the
    objective here too would divide the same axis twice, training on `1/dp_world**2` of the batch
    while every metric reported the full count. `pdims` is kept as a parameter (rather than dropped)
    so existing callers that resolve and pass it need no signature change, and because a future
    per-flavor consumer of the mesh is plausible; today it is accepted and unused.

    Args:
        rl: the resolved GRPO config (`config/rl.py::GRPOConfig`).
        name_or_path: model id used to build the packer for the clipped flavor.
        pdims: this actor's `ParallelDims`, or None for single-device. Unused (see above) — kept for
            call-site compatibility.
        packer: a DataPacker already built by the caller. Passed through rather than rebuilt so a
            caller that owns one (the examples build it for the rollout too) does not load a second
            tokenizer for the same model. None builds one from `name_or_path`.
        max_response_length: Fixed rollout response bound consumed only by Dr-GRPO's loss mode.

    Returns:
        The bound `Objective`, unwrapped — scoring exactly the batch it is called with.

    Raises:
        ValueError: `rl.grpo_surrogate` names no known surrogate.
        AttributeError: the config is missing a field this reads — deliberately loud, so a rename
            cannot silently fall back to a default that disables a check.
    """
    from cosmos_framework.posttrain.algorithm.grpo import make_clipped_objective, make_objective
    from cosmos_framework.posttrain.data.packer import init_data_packer

    if rl.grpo_surrogate == "clipped":
        # The clipped flavor NEEDS a packer: it shapes through `get_policy_input` and collates a
        # padded batch, so unlike legacy_pg it cannot run off stored ids alone.
        obj = make_clipped_objective(
            packer or init_data_packer(name_or_path),
            rl,
            max_response_length=max_response_length,
        )
    elif rl.grpo_surrogate == "legacy_pg":
        # Forward the packer. It is OPTIONAL here -- unlike the clipped flavor, legacy_pg
        # can shape off stored `policy_info['prompt_ids']` alone -- but dropping a packer the caller
        # SUPPLIED is not the same as not needing one: trajectories that lack stored ids, or carry
        # decoded-string actions, then fail in `shape_policy_input` while the tokenizer that would
        # have shaped them sat unused in this function's argument. Silently ignoring a supplied
        # argument is the failure mode this branch has hit repeatedly; honor it and let
        # `make_objective` keep its own None default for callers that genuinely have none.
        # Prefix filtering is NOT wired here any more -- it is a reward-stage annotation now, built
        # by `build_reward_annotators` below, for the same reason the advantage estimator is: both
        # are computed over a WHOLE group, and only the reward stage is guaranteed to hold one.
        obj = make_objective(packer=packer, temperature=rl.temperature)
    else:
        raise ValueError(f"rl.grpo_surrogate={rl.grpo_surrogate!r} is not a known surrogate ('legacy_pg' | 'clipped')")

    identity: dict[str, Any] = {
        "schema_version": 1,
        "algorithm": "grpo",
        "surrogate": str(rl.grpo_surrogate),
        "temperature": float(rl.temperature),
        "loss_type": "token_sum" if rl.grpo_surrogate == "legacy_pg" else str(rl.loss_type),
    }
    if rl.grpo_surrogate == "clipped":
        identity.update(
            logprob_dtype=str(rl.logprob_dtype),
            epsilon_low=float(rl.epsilon_low),
            epsilon_high=float(rl.epsilon_high),
            kl_beta=float(rl.kl_beta),
            lower_bound_ratio=float(rl.lower_bound_ratio),
            use_rollout_logprobs_for_loss=bool(rl.use_rollout_logprobs_for_loss),
        )
        # `kl_beta` names the KL SCALE; the anchor's WEIGHTS join the identity as `reference_artifact`
        # — the content identity of `rl.ref.model_dir` (`utils/artifact_identity.py`) — bound by
        # `entrypoints/grpo.py::build_trainer`, the one place that has the directory on every policy
        # rank and a process group to read it once. Absent when the reference is the policy's own
        # initial weights (every earlier anchored checkpoint), so those stay resumable; present, it
        # refuses an exact resume whose anchor directory changed path OR content.
        if rl.loss_type == "seq-mean-token-sum-fixed":
            if max_response_length is None:
                raise ValueError("seq-mean-token-sum-fixed requires max_response_length in the objective identity")
            identity["max_response_length"] = int(max_response_length)
        # Recorded only when it changes the loss. Token level is the objective every earlier clipped
        # checkpoint was trained under; an unconditional key would refuse their exact resume on a
        # byte that carries no new information. Absent therefore means "token".
        if rl.importance_sampling_level != "token":
            identity["importance_sampling_level"] = str(rl.importance_sampling_level)
        # Same rule for the KL gradient bound: absent means the exact k3 gradient every earlier
        # anchored checkpoint trained under.
        if rl.kl_coeff_clamp is not None:
            identity["kl_coeff_clamp"] = float(rl.kl_coeff_clamp)
        # Same absence convention: overlong filtering changes which tokens the loss sees, so it joins
        # the identity when on; off is what every earlier clipped checkpoint trained under.
        if rl.mask_truncated_completions:
            identity["mask_truncated_completions"] = True
        # Same absence convention: centering changes the gradient every token sees, so it joins the
        # identity when on, with the knobs that shape the correction; off is every earlier checkpoint.
        sc = getattr(rl, "score_centering", None)
        if sc is not None and bool(sc.enabled):
            identity["score_centering"] = {
                "top_k": int(sc.top_k),
                "compose": str(sc.compose),
                "tis_clip": float(sc.tis_clip),
                "mis_low": float(sc.mis_low),
                "mis_high": float(sc.mis_high),
            }
    obj.identity = identity

    # No dp-axis wrapping here any more (Task 9) — see the docstring above. `obj` scores exactly the
    # batch its caller hands it; partitioning that batch across dp ranks is the ingress predicate's
    # job (`workers/dispatch.py::select_chunk`), applied before the objective ever runs.
    return obj


def effective_model_id(config: Any) -> str:
    """The model id the selected policy will load, falling back to the rollout identity.

    `HFModelConfig.name_or_path` defaults to `""` and `build_trainer` falls back to
    `config.rollout.name_or_path`, because the trainer and the inference engine must load the same
    weights or the weight sync is meaningless. Anything else that needs "the model" — building a
    tokenizer for shared-prefix filtering, for instance — has to resolve it the SAME way. Reading
    `config.model_hf.name_or_path` directly gives `""` on the default config, and
    `init_data_packer("")` is a confusing failure a long way from its cause.

    Args:
        config: The resolved online root config.

    Returns:
        The non-empty model id, or `""` if neither is set (the caller decides whether that matters).
    """
    model_id = select_policy_backend(config).model_id
    if model_id:
        return model_id
    hf_model_id = getattr(getattr(config, "model_hf", None), "name_or_path", "")
    return str(hf_model_id or getattr(getattr(config, "rollout", None), "name_or_path", ""))


def prefix_filter_enabled(rl) -> bool:
    """Is shared-prefix loss filtering on for this config?

    ONE predicate, read by everything that has to agree about it: `build_reward_annotators` (whether
    to build the filter at all), the GRPO entry (whether the reward WorkerGroup must declare
    `needs_bulk`, since the filter reads `transition.action`), and any caller deciding whether a
    tokenizer is worth loading. These previously each spelled the test out, and it is not the obvious
    one: enablement follows `assign_shared_prefix_ignore`'s contract -- `None` or `< 1` disables --
    NOT Python truthiness, so a negative threshold is truthy and yet off. Two copies of a subtle
    predicate is how the shape and the body come to disagree.

    Args:
        rl: the resolved GRPO config.

    Returns:
        True when the filter will actually mask anything.
    """
    return rl.min_filter_prefix_tokens is not None and int(rl.min_filter_prefix_tokens) >= 1


def rollout_topk_logprobs(rl) -> int:
    """How many sampler top-k logprobs the rollout must return and record per generated token.

    The rollout-side reading of `rl.score_centering`: `top_k` when enabled, else 0 (nothing
    requested, nothing recorded, engine identity unchanged). The objective reads the same block, so
    a run cannot centre on a head the engine never returned or record a head nothing consumes.

    Args:
        rl: the resolved GRPO config.

    Returns:
        `rl.score_centering.top_k` when `rl.score_centering.enabled`, else 0.
    """
    sc = getattr(rl, "score_centering", None)
    if sc is None or not bool(getattr(sc, "enabled", False)):
        return 0
    return int(sc.top_k)


def rollout_records_routing(rl) -> bool:
    """Whether the rollout engine must return, and its body record, the per-token expert routing.

    The rollout-side reading of `rl.routing_replay`; `forward_replay_source` is the trainer-side
    one. Both read the same knob so a run cannot record routing it never replays or replay routing
    that was never recorded.

    Args:
        rl: the resolved GRPO config.

    Returns:
        True for `rl.routing_replay == 'rollout'`.
    """
    return str(rl.routing_replay) == "rollout"


def forward_replay_source(rl, model: Any) -> str | None:
    """Bind `rl.routing_replay` to the built policy's replay capability, refusing an incapable one.

    Assembly-time so a dense or HF policy asked to replay routing fails before any worker or
    engine comes up, and generic so the entry never names a model class: the capability is
    `model/protocol.py::ForwardReplayPolicy`, implemented by the model adapters that route.

    Args:
        rl: the resolved GRPO config.
        model: the assembled policy (`HFModel` or a native adapter).

    Returns:
        The routing source the trainer must hold every forward to (`LLMTrainer(forward_replay=...)`,
        which also binds it into the exact-resume identity), or None for natural dispatch.

    Raises:
        ValueError: routing replay is requested on a model that cannot replay its routing.
    """
    from cosmos_framework.posttrain.model.protocol import ForwardReplayPolicy

    if rl.routing_replay == "off":
        return None
    if not isinstance(model, ForwardReplayPolicy):
        raise ValueError(
            f"rl.routing_replay={rl.routing_replay!r} requires a policy that can replay its forward routing "
            f"(ForwardReplayPolicy); {type(model).__name__} cannot. Routing replay applies to the native "
            "sparse-MoE policy only; dense and HF backends must leave it at 'off'."
        )
    return str(rl.routing_replay)


def build_reward_annotators(rl, name_or_path: str, packer=None) -> tuple:
    """Build the reward stage's group annotators from the GRPO config. Sibling of `build_objective`.

    These are the computations that need a WHOLE group and therefore cannot live in the objective:
    the group-relative advantage (`algorithm/annotators/advantage.py`) and shared-prefix loss filtering
    (`algorithm/annotators/prefix_filter.py`). Both run once per reward batch, where a group is intact by
    construction — one rollout call produces a prompt's G completions and the reward stage sees
    exactly that. Their results ride on the trajectory, so dispatch is then free to split a group
    across ranks, replicas or steps.

    ONE mapping for both, for the reason `build_objective` is one mapping for both flavors: these
    knobs have been silently dropped on one of two routes before. `RewardWorker` calls whatever it
    is given and cannot tell that something is missing; the objective's `require_advantages` catches
    a missing advantage, but a missing prefix filter degrades quietly, so it must be wired here
    rather than at each call site.

    Args:
        rl: the resolved GRPO config (`config/rl.py::GRPOConfig`).
        name_or_path: model id, used to obtain the tokenizer's EOS when prefix filtering is on.
        packer: a DataPacker the caller already owns, so a second tokenizer is not loaded for the
            same model. None builds one ONLY when prefix filtering is actually enabled.

    Returns:
        The annotators, in the order `RewardWorker` should run them.
    """
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
    from cosmos_framework.posttrain.algorithm.annotators.prefix_filter import shared_prefix_ignore
    from cosmos_framework.posttrain.data.packer import init_data_packer

    annotators: list = []
    if prefix_filter_enabled(rl):
        # Without EOS, `assign_shared_prefix_ignore` cannot tell an EOS-terminated shared prefix
        # from a continuing one, and one config yields different loss masks depending on whether a
        # packer happened to be supplied.
        eff_packer = packer or init_data_packer(name_or_path)
        eos = getattr(getattr(eff_packer, "tokenizer", None), "eos_token_id", None)
        annotators.append(shared_prefix_ignore(int(rl.min_filter_prefix_tokens), eos))
    annotators.append(
        group_relative(unbiased=bool(rl.unbiased_advantage), group_size=int(rl.group_size)),
    )
    return tuple(annotators)


def vllm_teacher_engine_kwargs(cfg: TeacherConfig) -> dict:
    """Pure config -> ``vllm.LLM(**kwargs)`` for the OPD teacher (no vllm import — CPU-testable).

    Differs from `vllm_engine_kwargs` in exactly two ways, both because a teacher SCORES rather than
    generates (the MoE kernel knob is shared: ``moe_backend`` passes through exactly as for the rollout):

    - ``max_logprobs`` is raised to ``cfg.top_k``. vLLM defaults it to 20 while the cosmos-rl
      reference config asks for 32, and the resulting failure is the nastiest one on this path: the
      served equivalent 400s every scoring request, cosmos-rl's client turns that into ``None``, and
      its trainer logs "All teacher logprobs are None ... skipping distillation loss calculation" and
      carries on as plain GRPO. Setting it here means the engine cannot be built into that state.
    - Prefix caching stays ON (the rollout forces it off because weight sync would leave a cached
      prefix holding pre-sync KV). A teacher is FROZEN, so its cache is always valid, and completions
      in one GRPO group share a prompt — exactly the case a prefix cache is for.

    Args:
        cfg: the teacher config.

    Returns:
        The `vllm.LLM(**kwargs)` keyword dict.
    """
    kw: dict = dict(
        model=cfg.name_or_path,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,
        enforce_eager=cfg.enforce_eager,
        # DPCA asks vLLM only for the realized token (prompt_logprobs=0), but the engine capacity
        # still has to be positive on vLLM versions that validate this construction knob.
        max_logprobs=max(1, cfg.top_k),
        logprobs_mode=cfg.logprobs_mode,
        enable_prefix_caching=True,
        moe_backend=cfg.moe_backend,
    )
    if cfg.tensor_parallel_size > 1:
        kw.update(
            tensor_parallel_size=cfg.tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=True,
        )
    return kw


def _validate_native_policy_precision(rl: Any, model_config: Any) -> None:
    """Refuse precision overrides that a native builder cannot observe.

    A native builder receives only its model config and policy mesh. When that config exposes the
    conventional ``dtype``/``fsdp_reduce_dtype`` fields, storage and compute must resolve to the
    single native dtype and reduction must match the configured reduction dtype. Transfer dtype is
    independent because the weight publisher performs that cast explicitly. The delivery mode must be
    "fsdp" whatever the config exposes: a native builder has no autocast path, so "autocast" would be
    silently ignored.
    """
    native_dtype = getattr(model_config, "dtype", None)
    native_reduce_dtype = getattr(model_config, "fsdp_reduce_dtype", None)
    from types import SimpleNamespace

    precision = resolve_precision(
        SimpleNamespace(rl=rl, model_config=model_config, model_backend=I4),
    )
    mismatches: list[str] = []
    if native_dtype is not None:
        if precision.master != str(native_dtype):
            mismatches.append(f"master={precision.master!r}, native dtype={native_dtype!r}")
        if precision.param != str(native_dtype):
            mismatches.append(f"param={precision.param!r}, native dtype={native_dtype!r}")
    if native_reduce_dtype is not None and precision.reduce != str(native_reduce_dtype):
        mismatches.append(f"reduce={precision.reduce!r}, native reduce dtype={native_reduce_dtype!r}")
    if precision.mode != "fsdp":
        mismatches.append(
            f"mode={precision.mode!r}; native builders deliver the compute dtype only via their FSDP cast"
        )
    if mismatches:
        raise ValueError(
            "model_backend='i4' cannot apply the resolved precision: "
            + "; ".join(mismatches)
            + ". Align rl.*_dtype with config.model.model_config, or use model_backend='hf' when a "
            "storage/compute precision split is required."
        )


def _validate_online_policy_identity(
    rl: Any,
    rollout: RolloutConfig,
    model_backend: str | None,
    model_config: Any | None,
    policy_model_id: str | None = None,
) -> None:
    """Validate generic online policy selection and policy/rollout identity."""
    max_tokens = rollout.max_tokens
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise TypeError(f"rollout.max_tokens must be a non-bool integer, got {max_tokens!r}")
    if max_tokens < 1:
        raise ValueError(f"rollout.max_tokens must be > 0, got {max_tokens!r}")
    # rl/rollout seed contract: resolves the one prompt-order seed the data role will use, and warns
    # when arms that vary only rollout.seed would all read one prompt order.
    check_data_order_seed(rl, rollout)
    if model_backend is None:
        return
    backend = normalize_model_backend(str(model_backend))
    configured_policy_id = str(
        policy_model_id
        if policy_model_id is not None
        else ("" if model_config is None else getattr(model_config, "name_or_path", ""))
    )
    if backend == I4 and not configured_policy_id:
        raise ValueError("model_backend='i4' requires config.model.model_id for online training")
    policy_model_id = configured_policy_id or str(rollout.name_or_path)
    if policy_model_id != str(rollout.name_or_path):
        raise ValueError(
            f"{backend} policy and rollout must start from the same model/tokenizer artifact before weight sync, "
            f"got policy={policy_model_id!r} rollout={rollout.name_or_path!r}"
        )
    if backend == I4:
        if model_config is not None:
            _validate_native_policy_precision(rl, model_config)
    else:
        from types import SimpleNamespace

        # Resolve once on the driver so the combinations `resolve_precision` refuses (fp16 autocast) fail
        # here, before Ray placement, instead of inside the trainer worker -- which resolves the same
        # namespace (`entrypoints/grpo.py`) and would otherwise be the first to see them.
        resolve_precision(SimpleNamespace(rl=rl, model_config=model_config, model_backend=backend))


def _raise_temperature_errors(contract: Any) -> None:
    from cosmos_framework.posttrain.control.temperature import check_temperature

    checked = check_temperature(contract)
    if checked.errors:
        raise ValueError("; ".join(checked.errors))


def validate_online_policy_backend(
    rl: Any,
    rollout: RolloutConfig,
    model_backend: str | None,
    model_config: Any | None,
    policy_model_id: str | None = None,
) -> None:
    """Validate GRPO policy selection and objective-distribution identity."""
    _validate_online_policy_identity(
        rl,
        rollout,
        model_backend,
        model_config,
        policy_model_id=policy_model_id,
    )
    from cosmos_framework.posttrain.control.temperature import grpo_temperature_contract

    _raise_temperature_errors(grpo_temperature_contract(rl, rollout))


def validate_opd_policy_backend(
    rl: OPDConfig,
    rollout: RolloutConfig,
    model_backend: str | None = None,
    model_config: Any | None = None,
    policy_model_id: str | None = None,
) -> None:
    """Validate OPD policy selection and backend capability before model construction."""
    _validate_online_policy_identity(
        rl,
        rollout,
        model_backend,
        model_config,
        policy_model_id=policy_model_id,
    )
    normalized_backend = None if model_backend is None else normalize_model_backend(str(model_backend))
    if rl.teacher.signal_mode == "dpca" and normalized_backend is not None and normalized_backend != I4:
        raise ValueError(
            f"DPCA requires model_backend={I4!r}, got {normalized_backend!r}; the generic HF path "
            "does not provide the native multimodal policy capability"
        )


def validate_opd_runtime(
    rl: OPDConfig,
    rollout: RolloutConfig,
    model_backend: str | None = None,
    model_config: Any | None = None,
    policy_model_id: str | None = None,
) -> None:
    """Validate OPD couplings that cross the algorithm and rollout config trees.

    These checks are intentionally CPU-only and run before placement or model construction. A
    syntactically valid DPCA config can still define the wrong likelihood ratio if rollout sampling
    is warped, while direct OPD is wrong when teacher and policy score at different effective
    temperatures.

    Args:
        rl: The resolved OPD algorithm config.
        rollout: The resolved student rollout config.
        model_backend: Optional root backend. DPCA requires a native policy capability.
        model_config: Optional selected policy config used to prove policy/rollout artifact identity.
        policy_model_id: Explicit generic native policy/tokenizer identity, when selected by a
            :class:`NativePolicyBuildSpec`.
    """
    validate_opd_policy_backend(
        rl,
        rollout,
        model_backend,
        model_config,
        policy_model_id=policy_model_id,
    )
    from cosmos_framework.posttrain.control.temperature import opd_temperature_contract

    _raise_temperature_errors(opd_temperature_contract(rl, rollout))


def _shared_vocab_tokenizer_identity(tokenizer: Any, *, owner: str) -> dict[str, Any]:
    """Project the exact vocabulary, template, and special-token semantics."""
    from cosmos_framework.posttrain.utils.identity import (
        canonical_json_sha256,
        normalize_json_dict,
    )

    get_vocab = getattr(tokenizer, "get_vocab", None)
    if not callable(get_vocab):
        raise TypeError(f"{owner} tokenizer exposes no get_vocab()")
    vocab = get_vocab()
    if not isinstance(vocab, dict) or any(
        not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int)
        for token, token_id in vocab.items()
    ):
        raise TypeError(f"{owner} tokenizer.get_vocab() must return dict[str, int]")

    tokenizer_fields: dict[str, Any] = {
        "vocab_sha256": canonical_json_sha256(vocab, context=f"{owner} tokenizer vocabulary"),
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    for field_name in (
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "unk_token_id",
        "additional_special_tokens_ids",
        "all_special_ids",
    ):
        tokenizer_fields[field_name] = getattr(tokenizer, field_name, None)
    return normalize_json_dict(
        tokenizer_fields,
        context=f"{owner} shared-vocabulary tokenizer identity",
    )


def _shared_vocab_processor_identity(processor: Any, model_config: Any, *, owner: str) -> dict[str, Any]:
    """Project only tokenizer/processor/placeholder semantics, never model capacity fields."""
    from cosmos_framework.posttrain.utils.identity import normalize_typed_json_dict, qualified_type

    if processor is None:
        raise TypeError(f"{owner} processor is unavailable")
    tokenizer_identity = _shared_vocab_tokenizer_identity(getattr(processor, "tokenizer", None), owner=owner)

    component_fields: dict[str, Any] = {}
    for field_name in ("image_processor", "video_processor"):
        component = getattr(processor, field_name, None)
        if component is None:
            component_fields[field_name] = None
            continue
        to_dict = getattr(component, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(f"{owner} processor {field_name} exposes no to_dict()")
        component_fields[field_name] = {
            "type": qualified_type(component),
            "config": normalize_typed_json_dict(
                to_dict(),
                context=f"{owner} processor {field_name} config",
            ),
        }

    vision_config = getattr(model_config, "vision_config", None)
    placeholder_fields: dict[str, Any] = {}
    for field_name in (
        "image_token_id",
        "video_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "spatial_merge_size",
        "patch_size",
        "temporal_patch_size",
    ):
        value = getattr(model_config, field_name, None)
        if value is None and vision_config is not None:
            value = getattr(vision_config, field_name, None)
        placeholder_fields[field_name] = value

    return normalize_typed_json_dict(
        {
            "processor_type": qualified_type(processor),
            "tokenizer": tokenizer_identity,
            "components": component_fields,
            "placeholders": placeholder_fields,
        },
        context=f"{owner} shared-vocabulary processor identity",
    )


def _cross_token_multimodal_teacher_identity(packer: DataPacker) -> dict[str, Any]:
    """Project the teacher-owned processor contract for multimodal DPCA.

    Cross-token OPD intentionally permits different student and teacher processors. The teacher
    identity is still fail-closed: it must expose a vision config and the image placeholder IDs
    needed to render each media-bearing prompt. Text-only samples remain legal because this is a
    model capability check; the per-sample packer contract decides whether a particular prompt has
    matching placeholders and media. The full model config is already normalized separately in the
    teacher construction identity, so duplicating ``vision_config.to_dict()`` here adds no safety
    and rejects valid Hugging Face metadata such as integer-keyed ``id2label`` mappings.
    """
    from cosmos_framework.posttrain.data.packer.base import PackerProcessorIdentity
    from cosmos_framework.posttrain.utils.identity import normalize_json_dict

    processor = getattr(packer, "processor", None)
    model_config = getattr(packer, "model_config", None)
    if processor is None or model_config is None:
        raise TypeError("cross-token multimodal teacher packer must expose processor and model_config")
    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is None:
        raise ValueError("cross-token multimodal teacher config must expose vision_config")
    if not isinstance(packer, PackerProcessorIdentity):
        raise TypeError(
            "cross-token multimodal teacher packer must implement "
            "PackerProcessorIdentity.processor_semantics_identity()"
        )
    processor_identity = packer.processor_semantics_identity()
    if not isinstance(processor_identity, dict):
        raise TypeError("cross-token multimodal teacher processor identity must be a dictionary")

    image_ids: dict[str, int] = {}
    for field_name in ("image_token_id", "vision_start_token_id", "vision_end_token_id"):
        value = getattr(model_config, field_name, None)
        if value is None:
            value = getattr(vision_config, field_name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"cross-token multimodal teacher config requires non-negative integer {field_name}, got {value!r}"
            )
        image_ids[field_name] = value
    if getattr(packer, "image_token_id", None) != image_ids["image_token_id"]:
        raise RuntimeError(
            "cross-token multimodal teacher packer image_token_id differs from its resolved model config"
        )

    return normalize_json_dict(
        {
            "processor": processor_identity,
            "image_ids": image_ids,
        },
        context="cross-token multimodal teacher processor identity",
    )


def build_vllm_teacher(
    cfg: TeacherConfig,
    packer: DataPacker,
    data_packer_config: DataPackerConfig | None = None,
) -> VLLMTeacher:
    """Build one ranked teacher lane after its per-replica rendezvous is live.

    Every TP lane proves the same engine/model identity before construction. This mirrors
    ``build_vllm_rollout``: ``external_launcher`` is collective, so rank-local config drift must
    fail before any lane starts building the engine. There is deliberately no second native
    ``AutoConfig`` read afterwards: vLLM registers its own config parser while constructing some
    models, so that would compare two parser implementations rather than attest the built engine.

    Args:
        cfg: the teacher config.
        packer: the student DataPacker, threaded through to `shape_policy_input` so the teacher
            derives the scored span exactly as the objective does. Multimodal cross-token mode also
            constructs a teacher-owned packer from ``cfg.name_or_path``.
        data_packer_config: optional model-aware factory configuration for the teacher model's own
            packer. None preserves the built-in model-type registry behavior.

    Returns:
        A `VLLMTeacher` bound to a freshly built `vllm.LLM`.
    """
    from cosmos_framework.posttrain.roles.rollout.validation import (
        normalize_validation_semantics_identity,
        validation_semantics_equal,
    )
    from cosmos_framework.posttrain.roles.teacher import VLLMTeacher
    from cosmos_framework.posttrain.utils.dist import run_rank_converged
    from cosmos_framework.posttrain.utils.identity import normalize_typed_json_dict
    from cosmos_framework.posttrain.utils.rank_state import gather_rank_states

    def prepare() -> tuple[
        Any,
        dict[str, Any],
        dict[str, Any],
        dict[str, Any] | None,
        dict[str, Any] | None,
        DataPacker | None,
    ]:
        _configure_vllm_runtime()
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM

        engine_config = normalize_validation_semantics_identity(
            vllm_teacher_engine_kwargs(cfg),
            context="vLLM teacher engine construction config",
        )
        teacher_model_config = _resolve_hf_config_with_local_retry(cfg.name_or_path, trust_remote_code=None)
        resolved_model_config = normalize_typed_json_dict(
            teacher_model_config.to_dict(),
            context="vLLM teacher resolved model config",
        )
        shared_vocab_identity: dict[str, Any] | None = None
        teacher_processor_identity: dict[str, Any] | None = None
        teacher_packer: DataPacker | None = None
        if cfg.input_mode == "shared_vocab_multimodal":
            packer_model_config = getattr(packer, "model_config", None)
            packer_processor = getattr(packer, "processor", None)
            if (packer_model_config is None) != (packer_processor is None):
                raise TypeError(
                    "shared-vocabulary VLM packer must expose both processor and model_config, or neither for text"
                )
            if packer_processor is None:
                student_identity = _shared_vocab_tokenizer_identity(
                    getattr(packer, "tokenizer", None),
                    owner="student",
                )
                teacher_identity = _shared_vocab_tokenizer_identity(
                    AutoTokenizer.from_pretrained(cfg.name_or_path),
                    owner="teacher",
                )
            else:
                student_identity = _shared_vocab_processor_identity(
                    packer_processor,
                    packer_model_config,
                    owner="student",
                )
                teacher_identity = _shared_vocab_processor_identity(
                    AutoProcessor.from_pretrained(cfg.name_or_path),
                    teacher_model_config,
                    owner="teacher",
                )
            if not validation_semantics_equal(student_identity, teacher_identity):
                raise RuntimeError(
                    "shared-vocabulary teacher/student tokenizer, processor, or placeholder semantics differ"
                )
            shared_vocab_identity = teacher_identity
        elif cfg.input_mode == "cross_token_multimodal":
            teacher_packer = DataPackerBuildSpec(
                model_id=cfg.name_or_path,
                factory=(data_packer_config or DataPackerConfig()).teacher_model_factory,
            ).build()
            teacher_processor_identity = _cross_token_multimodal_teacher_identity(teacher_packer)
        return (
            LLM,
            engine_config,
            resolved_model_config,
            shared_vocab_identity,
            teacher_processor_identity,
            teacher_packer,
        )

    (
        llm_class,
        engine_config,
        resolved_model_config,
        shared_vocab_identity,
        teacher_processor_identity,
        teacher_packer,
    ) = run_rank_converged(prepare, "vLLM teacher engine identity preparation")
    construction_identity = {
        "engine_config": engine_config,
        "resolved_model_config": resolved_model_config,
        "shared_vocab_identity": shared_vocab_identity,
        "teacher_processor_identity": teacher_processor_identity,
        "input_mode": cfg.input_mode,
        "signal_mode": cfg.signal_mode,
        "logprobs_mode": cfg.logprobs_mode,
        "scoring_temperature": cfg.scoring_temperature,
        "conditional_think_prefix": cfg.conditional_think_prefix,
        "max_alignment_chunk_tokens": cfg.max_alignment_chunk_tokens,
    }
    by_rank = gather_rank_states(
        construction_identity,
        context="vLLM teacher construction identity preparation",
    )
    if any(not validation_semantics_equal(candidate, construction_identity) for candidate in by_rank.values()):
        raise RuntimeError(f"vLLM teacher construction identity differs across ranks: {by_rank!r}")

    engine = llm_class(**engine_config)

    return VLLMTeacher(
        engine,
        packer=packer,
        teacher_packer=teacher_packer,
        top_k=cfg.top_k,
        input_mode=cfg.input_mode,
        signal_mode=cfg.signal_mode,
        logprobs_mode=cfg.logprobs_mode,
        scoring_temperature=cfg.scoring_temperature,
        max_attempts=cfg.max_attempts,
        retry_backoff_s=cfg.retry_backoff_s,
        conditional_think_prefix=cfg.conditional_think_prefix,
        max_alignment_chunk_tokens=cfg.max_alignment_chunk_tokens,
    )


def build_opd_objective(rl: OPDConfig, packer: DataPacker, pdims: Any = None) -> Objective:
    """Build the OPD objective selected by teacher-signal and objective modes.

    Same-token direct OPD uses the normalized top-k-plus-residual divergence. The historical
    same-token advantage route retains the raw-slice divergence for Cosmos-RL parity. Cross-token
    DPCA constructs detached additive credit and never passes through a divergence adapter.

    Args:
        rl: the OPD run config.
        packer: the shaping seam, shared with the objective's GRPO siblings.
        pdims: this actor's `ParallelDims`, or None for single-device. As in `build_objective`, it no
            longer changes what this returns — Task 9 deleted `dp_group_sharded` and moved dp-axis
            partitioning to the ingress predicate (`workers/dispatch.py::select_chunk`), applied by
            the WorkerGroup `train` edge before the objective ever runs. Kept so the two builders
            stay signature-compatible behind `build_trainer`'s seam.

    Returns:
        An `Objective`, built exactly as `build_objective` builds the GRPO ones.

    Raises:
        ValueError: `distillation_mode` is neither `direct` nor `advantage` (also rejected at config
            construction; kept here so a hand-built config cannot slip past).
    """
    from cosmos_framework.posttrain.algorithm.dpca import DPCATeacherCreditSource
    from cosmos_framework.posttrain.algorithm.opd import make_advantage_objective, make_direct_objective

    if rl.distillation_mode == "direct":
        if rl.teacher.signal_mode != "topk" or rl.jsd_beta is None:
            raise ValueError("direct OPD requires a top-k teacher signal and jsd_beta in [0, 1]")
        obj = make_direct_objective(
            packer,
            jsd_beta=float(rl.jsd_beta),
            loss_type=rl.loss_type,
            temperature=rl.temperature,
        )
    elif rl.distillation_mode == "advantage":
        teacher_credit = (
            DPCATeacherCreditSource(
                min_abs_student_chunk_logprob=rl.teacher.min_abs_student_chunk_logprob,
            )
            if rl.teacher.signal_mode == "dpca"
            else None
        )
        obj = make_advantage_objective(
            packer,
            jsd_beta=rl.jsd_beta,
            kl_penalty_coef=rl.kl_penalty_coef,
            loss_type=rl.loss_type,
            temperature=rl.temperature,
            epsilon_low=rl.epsilon_low,
            epsilon_high=rl.epsilon_high,
            kl_beta=rl.kl_beta,
            lower_bound_ratio=rl.lower_bound_ratio,
            teacher_credit=teacher_credit,
        )
    else:
        raise ValueError(f"distillation_mode must be direct|advantage, got {rl.distillation_mode!r}")

    identity: dict[str, int | float | bool | str] = {
        "schema_version": 1,
        "algorithm": "opd",
        "distillation_mode": str(rl.distillation_mode),
        "teacher_signal_mode": str(rl.teacher.signal_mode),
        "temperature": float(rl.temperature),
        "loss_type": str(rl.loss_type),
        "teacher_scoring_temperature": float(rl.teacher.scoring_temperature),
    }
    if rl.distillation_mode == "direct":
        identity.update(
            divergence="topk_plus_other",
            jsd_beta=float(rl.jsd_beta),
            teacher_top_k=int(rl.teacher.top_k),
        )
    else:
        identity.update(
            kl_penalty_coef=float(rl.kl_penalty_coef),
            epsilon_low=float(rl.epsilon_low),
            epsilon_high=float(rl.epsilon_high),
            kl_beta=float(rl.kl_beta),
            lower_bound_ratio=float(rl.lower_bound_ratio),
        )
        if rl.teacher.signal_mode == "topk":
            if rl.jsd_beta is None:
                raise ValueError("top-k advantage OPD requires jsd_beta in [0, 1]")
            identity.update(
                divergence="topk_slice",
                jsd_beta=float(rl.jsd_beta),
                teacher_top_k=int(rl.teacher.top_k),
            )
        else:
            identity.update(
                teacher_credit="dpca",
                min_abs_student_chunk_logprob=float(rl.teacher.min_abs_student_chunk_logprob),
            )
    obj.identity = identity

    # No dp wrapping: `select_chunk` on the train edge already handed this rank its own slice.
    return obj
