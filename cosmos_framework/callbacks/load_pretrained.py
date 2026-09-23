# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from cosmos_framework.model._base import ImaginaireModel
from cosmos_framework.utils.callback import Callback


def _warm_start_skips_complete_ema(patterns: list[str], ema_state_fqns: list[str] | None = None) -> bool:
    """Return whether DCP substring filters cover every ``net_ema.*`` state leaf."""
    ema_root_fqn = "net_ema."
    if any(pattern and pattern in ema_root_fqn for pattern in patterns):
        return True
    if not ema_state_fqns:
        return False
    return all(any(pattern and pattern in fqn for pattern in patterns) for fqn in ema_state_fqns)


def _warm_start_partially_skips_ema(
    patterns: list[str],
    ema_state_fqns: list[str],
) -> bool:
    """Return whether DCP filters match some, but not all, EMA state leaves."""
    if not patterns or not ema_state_fqns or _warm_start_skips_complete_ema(patterns, ema_state_fqns):
        return False
    matched = sum(any(pattern and pattern in fqn for pattern in patterns) for fqn in ema_state_fqns)
    return 0 < matched < len(ema_state_fqns)


class LoadPretrained(Callback):
    """Load HF understanding-pathway weights after DCP resume, gated by checkpoint state.

    Decision table (config flags are *intent*; the runtime probes here decide):
      * Latest checkpoint exists in load dir → DCP loaded the full model. Skip HF load.
      * No latest checkpoint, ``load_path`` set → DCP loaded full model from warm-start.
        Reload HF understanding pathway (e.g. swap Qwen3-VL → Cosmos-Reason) but skip
        the understanding→generation copy.
      * Neither → fresh init: full HF load + understanding→generation copy.

    Reads ``self.config.checkpoint`` / ``self.config.job`` (injected by
    ``CallBackGroup`` after instantiate) to build a probe checkpointer.
    """

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        from cosmos_framework.checkpoint.dcp import DistributedCheckpointer

        probe = DistributedCheckpointer(self.config.checkpoint, self.config.job, callbacks=None, disable_async=True)
        # DCP matches these patterns as substrings of flattened FQNs. Only a
        # pattern broad enough to match the EMA root proves that the complete
        # warm-start EMA subtree was skipped; partial EMA skips must not trigger
        # an unconditional regular->EMA overwrite.
        skip_patterns = self.config.checkpoint.keys_to_skip_loading
        ema_state_fqns: list[str] = []
        net_ema = getattr(model, "net_ema", None)
        if net_ema is not None:
            ema_state_fqns.extend(f"net_ema.{name}" for name, _ in net_ema.named_parameters())
            ema_state_fqns.extend(f"net_ema.{name}" for name, _ in net_ema.named_buffers())
        warm_start_ema_skipped = _warm_start_skips_complete_ema(skip_patterns, ema_state_fqns)
        warm_start_ema_partially_skipped = _warm_start_partially_skips_ema(skip_patterns, ema_state_fqns)
        model.load_pretrained_model_if_needed(
            has_resumable_checkpoint=probe.has_resumable_checkpoint(),
            has_load_path=probe.load_path is not None,
            warm_start_ema_skipped=warm_start_ema_skipped,
            warm_start_ema_partially_skipped=warm_start_ema_partially_skipped,
            warm_start_strict_resume=self.config.checkpoint.strict_resume,
        )
