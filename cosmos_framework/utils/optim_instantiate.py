# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import hydra
import torch
from torch import nn

from cosmos_framework.utils import log


def get_regular_param_group(net: nn.Module):
    """
    seperate the parameters of the network into two groups: decay and no_decay.
    based on nano_gpt codebase.
    """
    param_dict = {pn: p for pn, p in net.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    return decay_params, nodecay_params


def get_base_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    optim_type: str = "adamw",
    sharding: bool = False,
    **kwargs,
) -> torch.optim.Optimizer:
    net_decay_param, net_nodecay_param = get_regular_param_group(model)

    num_decay_params = sum(p.numel() for p in net_decay_param)
    num_nodecay_params = sum(p.numel() for p in net_nodecay_param)
    net_param_total = num_decay_params + num_nodecay_params
    log.critical(f"total num parameters : {net_param_total:,}")

    param_group = [
        {
            "params": net_decay_param + net_nodecay_param,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]

    if optim_type == "adamw":
        opt_cls = torch.optim.AdamW
    elif optim_type == "fusedadam":
        from cosmos_framework.utils.fused_adam import FusedAdam

        opt_cls = FusedAdam
    else:
        raise ValueError(f"Unknown optimizer type: {optim_type}")

    return opt_cls(param_group, **kwargs)


def get_base_scheduler(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    scheduler_config: dict,
):
    net_scheduler = hydra.utils.instantiate(scheduler_config)
    net_scheduler.model = model

    num_param_groups = len(optimizer.param_groups)

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            net_scheduler.schedule,
        ]
        * num_param_groups,
    )
