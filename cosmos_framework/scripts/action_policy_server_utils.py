# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Shared helpers for the standalone Action policy serving scripts.

Both ``cosmos_framework.scripts.action_policy_server_libero`` (HTTP) and
``cosmos_framework.scripts.action_policy_server_robolab`` (WebSocket) reuse the
helpers defined here:

- single-rank distributed init for FSDP-aware model loading;
- local IP / free-port discovery for log messages and PG bring-up;
- the default output directory shared across both servers;
- the ``OmniSetupArgs`` subclass that disables runtime EMA for frozen configs.

Keeping these in one module avoids ``action_policy_server_robolab`` importing
from ``action_policy_server_libero`` purely to share runtime utilities.
"""

import os
import socket
from pathlib import Path

import torch
from torch import distributed as dist

from cosmos_framework.inference.args import OmniSetupArgs
from cosmos_framework.inference.common.args import ConfigFileType
from cosmos_framework.utils import log

DEFAULT_FALLBACK_OUTPUT_DIR = Path("/tmp/cosmos3_action_server")


def _get_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def get_local_ip() -> str:
    """Get the local IP address of this machine."""
    try:
        # Connect to an external address to determine the local IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def maybe_init_distributed() -> None:
    """
    Initialize a one-rank process group when launched outside torchrun so FSDP
    and distributed utilities have the process-group state they expect.

    ``init_script()`` already inits the PG when ``WORLD_SIZE>1`` (under
    torchrun); this function fills in the single-process case used by the
    standalone server.
    """
    if not dist.is_available() or dist.is_initialized():
        return

    world_size_env = os.getenv("WORLD_SIZE")
    rank_env = os.getenv("RANK")
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    backend = "nccl" if torch.cuda.is_available() else "gloo"

    if world_size_env is not None and rank_env is not None:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    port = _get_free_local_port()
    dist.init_process_group(backend=backend, init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)


class _ActionPolicyServerSetupArgs(OmniSetupArgs):
    """Server-local setup args that avoid instantiating runtime EMA for frozen configs."""

    def load_model_config_dict(self) -> dict:
        model_dict = super().load_model_config_dict()
        model_dict.setdefault("config", {}).setdefault("ema", {})["enabled"] = False
        return model_dict


def disable_runtime_ema_for_frozen_config(setup_args: OmniSetupArgs) -> OmniSetupArgs:
    """Use server-local setup args to instantiate frozen configs without runtime EMA."""
    if setup_args.config_file_type == ConfigFileType.MODULE:
        return setup_args

    log.info("[action-server] disabled runtime EMA for frozen config model load")
    return _ActionPolicyServerSetupArgs.model_validate(setup_args.model_dump())
