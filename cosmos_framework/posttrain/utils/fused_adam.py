# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Apex-fused Adam with fp32 master weights + fp32 moments (cosmos-rl parity).

Port of `cosmos_rl/utils/wfm/optimizer.FusedAdam` (itself a DTensor-aware fork of
`apex.optimizers.FusedAdam`). This exists because plain `torch.optim.AdamW` keeps its
moments in the PARAM dtype: with a bf16 net, `exp_avg`/`exp_avg_sq` are bf16 and the
weight update is applied in bf16 — at posttrain learning rates (~5e-6) the relative
update (lr/|w| ≈ 2.5e-4) is BELOW bf16 resolution (2^-8 ≈ 3.9e-3), so most updates
round away and training crawls. Root-caused 2026-07-06 against the wallace 8B DPO
reference run: identical data/objective/trainable-set, ~7x slower loser-side drift,
margin plateau at ~3 vs the reference's steady climb (see the wallace repro config
docstring). FusedAdam fixes both sinks: fp32 moments, and fp32 master params that
accumulate sub-resolution updates across steps (bf16 params are refreshed from the
master copy each step).

Deltas vs the cosmos-rl original (behavior-preserving):
- `exp_avg`/`exp_avg_sq` are stored as fp32 DTensors mirroring the param placement
  (the original stores raw local shards). The fused kernel still operates on the
  local shard view, but `get_optimizer_state_dict` / DCP checkpointing now see
  properly-placed DTensors, so posttrain's consolidated and dcp save paths work
  unchanged. The original's shard-local blobs are only loadable at identical topology.
- No GradScaler path (posttrain trains in bf16, no fp16 loss scaling).
- apex is imported lazily in `__init__` so importing this module stays container-free
  (CPU test collection, login nodes).

Caveat kept from the original: fp32 master weights are (re)built lazily on the first
`step()` and are NOT part of `state_dict()`. On resume they are re-materialized from
the (bf16) params, losing sub-bf16 accumulation since the last step — identical to
cosmos-rl's resume semantics, so reference parity is preserved there too.
"""

from __future__ import annotations

import torch


def _local(t: torch.Tensor) -> torch.Tensor:
    """Local shard view of a DTensor (shared storage), or the tensor itself."""
    return t.to_local() if hasattr(t, "to_local") else t


class FusedAdam(torch.optim.Optimizer):
    """Fused AdamW (`adam_w_mode=True`) with fp32 master weights, for bf16/fp32 params.

    Matches cosmos-rl's reference construction: `FusedAdam(params, lr=..., weight_decay=...,
    betas=(0.9, 0.99), eps=1e-8, capturable=True, master_weights=True)`.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        bias_correction: bool = True,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        adam_w_mode: bool = True,
        weight_decay: float = 0.0,
        capturable: bool = True,
        master_weights: bool = True,
    ) -> None:
        if master_weights and not capturable:
            raise RuntimeError("master_weights requires capturable=True (kernel choice)")
        from apex.multi_tensor_apply import multi_tensor_applier

        if not multi_tensor_applier.available:
            raise RuntimeError("FusedAdam requires apex CUDA extensions (amp_C)")
        import amp_C

        lr_v = torch.tensor(lr, dtype=torch.float32) if capturable else lr
        defaults = dict(lr=lr_v, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.capturable = capturable
        self.master_weights = master_weights
        self._applier = multi_tensor_applier
        self.param_groups_master: list[dict] | None = None

        if capturable:
            for group in self.param_groups:
                if not group["params"]:
                    continue
                device = group["params"][0].device
                if isinstance(group["lr"], float):
                    group["lr"] = torch.tensor(group["lr"], dtype=torch.float32)
                group["lr"] = group["lr"].to(device=device)

        self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
        self.multi_tensor_adam = amp_C.multi_tensor_adam
        self.multi_tensor_adam_capturable = amp_C.multi_tensor_adam_capturable
        self.multi_tensor_adam_capturable_master = amp_C.multi_tensor_adam_capturable_master

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None

        if self.param_groups_master is None:
            # fp32 master copies of the LOCAL shards, built lazily (see module caveat).
            self.param_groups_master = [
                {"params": [_local(p).clone().detach().float() if self.master_weights else None for p in g["params"]]}
                for g in self.param_groups
            ]

        for group, group_master in zip(self.param_groups, self.param_groups_master):
            if not group["params"]:
                continue
            device = group["params"][0].device
            bias_correction = 1 if group.get("bias_correction") else 0
            beta1, beta2 = group["betas"]

            # One shared step counter per group (tensor on device in capturable mode).
            if "step" in group:
                if self.capturable:
                    group["step"] = (
                        group["step"].to(device=device)
                        if isinstance(group["step"], torch.Tensor)
                        else torch.tensor(group["step"], dtype=torch.int32, device=device)
                    )
                    group["step"] += (self._dummy_overflow_buf != 1).to(torch.int)
                else:
                    group["step"] += 1
            else:
                group["step"] = 1 if not self.capturable else torch.tensor([1], dtype=torch.int, device=device)

            if self.capturable:
                group["lr"] = (
                    group["lr"].to(device=device)
                    if isinstance(group["lr"], torch.Tensor)
                    else torch.tensor(group["lr"], dtype=torch.float32, device=device)
                )

            # Buckets per param dtype for the multi-tensor kernels.
            buckets: dict[torch.dtype, list[list[torch.Tensor]]] = {}
            for p, p_master in zip(group["params"], group_master["params"]):
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    # fp32 moments as DTensors mirroring p (DCP-friendly; see module docstring).
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                if p.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise RuntimeError(f"FusedAdam supports fp16/bf16/fp32 params, got {p.dtype}")
                b = buckets.setdefault(p.dtype, [[], [], [], [], []])
                b[0].append(_local(p.grad))
                b[1].append(_local(p))
                b[2].append(_local(state["exp_avg"]))
                b[3].append(_local(state["exp_avg_sq"]))
                if self.master_weights:
                    b[4].append(p_master)

            for _dtype, (g_l, p_l, m_l, v_l, master_l) in buckets.items():
                if self.capturable:
                    inv_scale = torch.ones((1,), device=device, dtype=torch.float32)
                    self._applier(
                        self.multi_tensor_adam_capturable_master
                        if self.master_weights
                        else self.multi_tensor_adam_capturable,
                        self._dummy_overflow_buf,
                        [g_l, p_l, m_l, v_l, master_l] if self.master_weights else [g_l, p_l, m_l, v_l],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                        inv_scale,
                    )
                else:
                    self._applier(
                        self.multi_tensor_adam,
                        self._dummy_overflow_buf,
                        [g_l, p_l, m_l, v_l],
                        group["lr"],
                        beta1,
                        beta2,
                        group["eps"],
                        group["step"],
                        self.adam_w_mode,
                        bias_correction,
                        group["weight_decay"],
                    )

        return loss

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # torch's base load_state_dict CASTS floating state tensors to the param dtype —
        # on a bf16 net that silently downgrades the fp32 moments (the exact failure mode
        # this optimizer exists to fix; the cosmos-rl original has the same wart). Restore
        # the ORIGINAL fp32 tensors from the incoming state_dict, mapped the same way the
        # base class maps them (flat param order across groups vs the saved index keys).
        saved = state_dict["state"]
        saved_ids = [i for g in state_dict["param_groups"] for i in g["params"]]
        live = [p for g in self.param_groups for p in g["params"]]
        for pid, p in zip(saved_ids, live):
            src = saved.get(pid, {})
            for k in ("exp_avg", "exp_avg_sq"):
                if k in src and torch.is_tensor(src[k]) and src[k].is_floating_point():
                    self.state[p][k] = src[k].to(device=p.device, dtype=torch.float32)
        # Masters go stale relative to the loaded params — rebuild lazily on next step().
        self.param_groups_master = None
        for group in self.param_groups:
            if self.capturable:
                # Use the group's own param device, not a hardcoded cuda:0 — under
                # multi-device-visible processes the rank's params may live on cuda:N
                # (mirrors __init__'s device resolution).
                device = group["params"][0].device if group["params"] else torch.device("cuda")
                if isinstance(group["lr"], torch.Tensor):
                    group["lr"] = group["lr"].to(device=device)
                else:
                    group["lr"] = torch.tensor(group["lr"], dtype=torch.float32, device=device)
                if "step" in group:
                    step = group["step"]
                    if not isinstance(step, torch.Tensor):
                        step = torch.tensor([int(step)], dtype=torch.int32)
                    group["step"] = step.to(dtype=torch.int32, device=device)
            elif "step" in group and isinstance(group["step"], torch.Tensor):
                group["step"] = int(group["step"].item())
