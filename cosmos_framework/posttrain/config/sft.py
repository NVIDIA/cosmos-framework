# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Offline supervised-conversation configuration, independent of rollout algorithms."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from cosmos_framework.posttrain.config.adapters import AdapterConfig
from cosmos_framework.posttrain.config.rl import HFRunConfig, ValidationScheduleConfig, _normalize_factory_kwargs

SFT_SERVE_CONTRACT_DEFAULT = "tools_no_prefill"
SFT_SERVE_CONTRACT_PREFILLS_OPENER = "prefills_opener"
SFT_SERVE_CONTRACTS = (SFT_SERVE_CONTRACT_DEFAULT, SFT_SERVE_CONTRACT_PREFILLS_OPENER)
"""The serving contracts a run declares through ``rl.serve_contract`` (a typed field of the resolved config and of
the objective identity, in place of the environment variable
``SFT_SERVE_PREFILLS_OPENER``, which two byte-identical resolved configs could differ by without a trace).
``tools_no_prefill``: the serving path of record (Codex ``--thinking high`` -> a Responses request with
``reasoning.effort`` and the harness's tools -> vLLM 0.24), which prefills NO ``<think>`` after ``<|im_start|>assistant\n``,
so the student emits the opener itself. ``prefills_opener``: a serve that prefills ``<think>`` for the
student (a tool-less request with ``reasoning.effort`` or ``enable_thinking``), the only contract under which
``mask_think_opener=True`` is accepted."""


def refuse_unacknowledged_opener_mask(serve_contract: str) -> None:
    """Raise ``ValueError`` for ``mask_think_opener=True`` under any serve contract but ``prefills_opener``.

    Called from :meth:`SFTConfig.__post_init__`, so the native recipe's ``prefilled_opener`` variant, a YAML and a ``--``
    override on the ``default`` variant are judged by the same typed field
    instead of training the opener unsupervised under ``recipe_variant: default``.
    """
    if serve_contract != SFT_SERVE_CONTRACT_PREFILLS_OPENER:
        raise ValueError(
            f"rl.mask_think_opener=True is refused under rl.serve_contract={serve_contract!r}: it unsupervises the "
            "<think> opener, which fits only a serve that prefills <think> for the student, and the serving path of "
            "record (Codex --thinking high -> a Responses request with reasoning.effort and the harness's 8 tools -> "
            "vLLM 0.24) prefills NO opener with tools present (review R138-01: the served prefix is "
            "<|im_start|>assistant\\n, ids 151644 77091 198; tests/fixtures/served_prefix_with_tools.json), so the "
            "student must emit <think> itself and its loss is the only training signal for that token. Set "
            f"rl.serve_contract={SFT_SERVE_CONTRACT_PREFILLS_OPENER} only for a student that will be served by a path "
            "that prefills the opener (a tool-less request with reasoning.effort or enable_thinking); the field is "
            "recorded in the resolved config and in the objective identity."
        )


@dataclass
class SFTDatasetConfig:
    """Local/S3 JSONL or a map-style conversation factory; exactly one at run time."""

    path: str = ""
    factory: str = ""
    factory_kwargs: dict[str, Any] = field(default_factory=dict)
    media_root: str = ""
    shuffle: bool = True
    credentials: str = "credentials/<credential>.secret"

    def __post_init__(self) -> None:
        if self.path and self.factory:
            raise ValueError("SFT data selects either path or factory, not both")
        if self.factory_kwargs and not self.factory:
            raise ValueError("SFT factory_kwargs requires factory")
        if not isinstance(self.shuffle, bool):
            raise TypeError("SFT data.shuffle must be bool")
        self.factory_kwargs = _normalize_factory_kwargs(self.factory_kwargs)
        json.dumps(self.factory_kwargs, allow_nan=False)

    def require_source(self) -> None:
        self.__post_init__()
        if not (self.path or self.factory):
            raise ValueError("SFT data requires a JSONL path or dataset factory")
        if self.path.startswith("s3://"):
            bucket, _, key = self.path[len("s3://") :].partition("/")
            if not bucket or not key or key.endswith("/"):
                raise ValueError("SFT S3 data requires an object URI: s3://bucket/path/file.jsonl")
            if not self.credentials:
                raise ValueError("SFT S3 data requires a credentials path")


@dataclass
class SFTValidationConfig(ValidationScheduleConfig):
    data: SFTDatasetConfig = field(default_factory=lambda: SFTDatasetConfig(shuffle=False))


SFT_LR_SCHEDULE_KINDS = ("constant", "cosine")


@dataclass
class SFTLRScheduleConfig:
    """The learning-rate schedule after ``rl.optimizer.warmup_steps`` linear warmup steps.

    ``kind="constant"``: ``config/build.py::build_lr_scheduler`` (warmup ramp ``(step+1)/warmup`` from the full lr's
    first fraction, then constant; no scheduler object at ``warmup_steps=0``) -- the behaviour every recipe had and the
    A1/A1a arms trained with. ``kind="cosine"``: ``build_warmup_cosine_lr_scheduler``, the Cosmos post-training recipe's
    ``LambdaWarmUpCosineScheduler``: lr * ``f_start`` at the first update, linear to lr at ``warmup_steps``, cosine down to
    lr * ``f_min`` at ``cycle_steps`` (``0`` = ``rl.max_steps``), ``f_min`` afterwards.
    """

    kind: str = "constant"
    f_start: float = 0.01
    f_min: float = 0.1
    cycle_steps: int = 0

    def __post_init__(self) -> None:
        if self.kind not in SFT_LR_SCHEDULE_KINDS:
            raise ValueError(f"SFT lr_schedule.kind={self.kind!r} must be one of {SFT_LR_SCHEDULE_KINDS}")
        for name in ("f_start", "f_min"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 < float(value) <= 1.0:
                raise ValueError(f"SFT lr_schedule.{name}={value!r} must satisfy 0 < value <= 1")
        if isinstance(self.cycle_steps, bool) or not isinstance(self.cycle_steps, int) or self.cycle_steps < 0:
            raise ValueError(
                f"SFT lr_schedule.cycle_steps={self.cycle_steps!r} must be an integer >= 0 (0 = max_steps)"
            )


SFT_MOE_LBL_METHODS = ("local", "global")


@dataclass
class SFTConfig(HFRunConfig):
    """One step is one optimizer update; batch size is per rank and microbatch."""

    data: SFTDatasetConfig = field(default_factory=SFTDatasetConfig)
    validation: SFTValidationConfig = field(default_factory=SFTValidationConfig)
    adapters: AdapterConfig | None = None
    micro_batch_size: int = 1
    max_length: int = 4096
    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    # The objective-level knobs the Cosmos post-training recipe of the Qwen3-VL-MoE backbone sets and the SFT seat
    # did not expose. Framework defaults = the pre-existing behaviour; the native recipe
    # (``configs/sft_qwen3_vl_moe_native.py``) documents its own values per knob.
    lr_schedule: SFTLRScheduleConfig = field(default_factory=SFTLRScheduleConfig)
    # Cross-entropy reduction exponent ``e`` of ``algorithm/sft.py``: every row contributes ``sum_tokens(CE) / n^e`` over
    # the DP-mean of ``sum_rows n^(1-e)``. ``0.0`` = the global supervised-token mean (every token weighs the same, a long
    # trajectory weighs more; the pre-existing objective). ``0.5`` = the backbone's square-root weighting
    # (``weighted_ce_exponent=0.5``, ``cosmos3/algorithm/loss/cross_entropy.py::weighted_cross_entropy_loss``): a row
    # weighs ``sqrt(n)``. ``1.0`` = every row weighs the same.
    ce_exponent: float = 0.0
    # Unsupervise the ``<think>`` token that opens an assistant turn, and only that token.
    # Keep ``False``: on the lab's serving path of record Codex ``--thinking high`` sends
    # ``reasoning.effort`` AND a tools array on every call, and with tools present vLLM 0.24 prefills no opener -- the
    # served prefix is ``<|im_start|>assistant\n`` with NO ``<think>`` (the effort field adds 0 input tokens; measured
    # live, fixture ``tests/fixtures/served_prefix_with_tools.json``) -- so the student must emit ``<think>`` itself as
    # the first token of every turn, then the reasoning or the closer ``\n\n</think>\n\n`` of the empty block a turn
    # without reasoning (a no-think turn) is rendered with; the opener's loss is the only training signal for that
    # token. ``True`` would leave it unsupervised; what a ``True``-trained student emits there is UNMEASURED with tools
    # present (no such student exists; the first token after ``assistant\n`` with tools present was never sampled), so
    # the consequence is a RISK, not a measurement: C-04 (the lab's live probe record of the served
    # generation prompt on the A1a deployment, 2026-10-02, outside the repo; primary pointer vLLM 0.24
    # ``reasoning/basic_parsers.py`` 173-181: ``extract_reasoning`` returns all-reasoning only when ``</think>`` is absent)
    # section 3's measured trigger is "an output that carries no ``</think>``" being returned as a reasoning item (a
    # no-think TEXT answer never reaches Codex;
    # tool-call turns unaffected), and the closer stays supervised under ``True``. The ``<think>`` prefill exists only
    # for requests without tools (a present effort -> ``enable_thinking`` -> the template appends ``<think>``; 10 input
    # tokens instead of 9), not a path of record; there the supervised opener costs one deterministic token per turn.
    # ``True`` is accepted only under ``serve_contract="prefills_opener"`` (the field below), checked in
    # :meth:`__post_init__` through :func:`refuse_unacknowledged_opener_mask`, so the native recipe's ``prefilled_opener``
    # variant, a YAML and a ``--`` override on the ``default`` variant are judged by the same typed field.
    mask_think_opener: bool = False
    # The serving contract the trained student is declared for (a typed,
    # persisted field, never an environment variable). ``tools_no_prefill`` (default): the path of record, where vLLM
    # 0.24 prefills no ``<think>`` with tools present and the student emits the opener itself, so ``mask_think_opener``
    # stays ``False``. ``prefills_opener``: a serve that prefills ``<think>`` for the student (a tool-less request with
    # ``reasoning.effort`` or ``enable_thinking``), the only contract under which ``mask_think_opener=True`` is accepted.
    # The value is in the resolved config and, when not the default, in the objective identity, so a checkpoint explains
    # why its opener was or was not supervised; a launcher variable may be translated into this field by deployment
    # glue, and nothing in the framework reads one.
    serve_contract: str = SFT_SERVE_CONTRACT_DEFAULT
    # MoE load-balancing auxiliary loss (Switch/GShard: ``coeff * E * mean_layers(sum_e f_e * P_e)``, with ``f_e`` the share
    # of routed tokens and ``P_e`` the mean router probability of expert ``e``) added to the CE objective; ``0.0`` = off
    # (pre-existing). The backbone post-trains with ``8e-3`` and method ``global`` (token counts summed over the DP ranks
    # before the shares are formed; ``local`` balances each rank's own batch).
    moe_lbl_coeff: float = 0.0
    moe_lbl_method: str = "global"

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.adapters, dict):
            self.adapters = AdapterConfig(**self.adapters)
        if self.adapters is not None:
            self.adapters.__post_init__()
        for name in ("micro_batch_size", "max_length"):
            value = getattr(self, name)
            minimum = 2 if name == "max_length" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"SFT {name} must be an integer >= {minimum}")
        self.lr_schedule.__post_init__()
        if (
            isinstance(self.ce_exponent, bool)
            or not isinstance(self.ce_exponent, (int, float))
            or not 0.0 <= float(self.ce_exponent) <= 1.0
        ):
            raise ValueError(f"SFT ce_exponent={self.ce_exponent!r} must be a real in [0, 1]")
        if not isinstance(self.mask_think_opener, bool):
            raise TypeError("SFT mask_think_opener must be bool")
        if self.serve_contract not in SFT_SERVE_CONTRACTS:
            raise ValueError(f"SFT serve_contract={self.serve_contract!r} must be one of {SFT_SERVE_CONTRACTS}")
        if self.mask_think_opener:
            refuse_unacknowledged_opener_mask(self.serve_contract)
        if (
            isinstance(self.moe_lbl_coeff, bool)
            or not isinstance(self.moe_lbl_coeff, (int, float))
            or not 0.0 <= float(self.moe_lbl_coeff) < 1.0
        ):
            raise ValueError(f"SFT moe_lbl_coeff={self.moe_lbl_coeff!r} must be a real in [0, 1)")
        if self.moe_lbl_method not in SFT_MOE_LBL_METHODS:
            raise ValueError(f"SFT moe_lbl_method={self.moe_lbl_method!r} must be one of {SFT_MOE_LBL_METHODS}")
        self.processor_kwargs = _normalize_factory_kwargs(self.processor_kwargs)
        json.dumps(self.processor_kwargs, allow_nan=False)
        self.data.__post_init__()
        self.validation.__post_init__()
        if self.validation.enabled:
            self.validation.data.require_source()
        if self.export_dir or self.export_every:
            raise ValueError("SFT uses the offline checkpoint conversion tools, not reserved rl.export_* fields")
