# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small tensors returned at the installed official HF loading boundary."""

from __future__ import annotations

import gc
import json
import os
import weakref
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


class SmallOfficialModel(nn.Module):
    def __init__(self, *, mask_token: bool = True) -> None:
        from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config

        super().__init__()
        self.config = Gr00tN1d7Config()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            self.backbone = nn.Linear(2, 2).requires_grad_(False)
            self.action_head = nn.Linear(2, 2)
        self.action_head.mask_token = nn.Parameter(torch.zeros(1, 2)) if mask_token else None


def run_loading_failure(rank: int, root: str, failure: str) -> None:
    """Exercise the real policy constructor and installed Pipeline under rank-local HF faults."""
    import torch.distributed as dist
    from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
    from transformers import AutoModel

    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_model_config
    from cosmos_framework.posttrain.model.gr00t.policy import Gr00tN1d7TrainPolicy

    directory = Path(root)
    os.environ["GR00T_DIT_SDPA_MODE"] = "math"
    dist.init_process_group(
        "gloo", init_method=f"file://{directory / 'rendezvous'}", rank=rank, world_size=2, timeout=timedelta(seconds=8)
    )
    references = []
    loaded_processor = False
    output = directory / f"config-{rank}"
    output.mkdir()
    if failure == "publish" and rank == 0:
        (output / "final_model_config.json").mkdir()

    def load_model(*args, **kwargs):
        model = SmallOfficialModel()
        references.append(weakref.ref(model))
        if rank == 1 and failure == "model":
            raise ValueError("rank-local model loading fault")
        return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}

    def load_processor(checkpoint, **kwargs):
        nonlocal loaded_processor
        loaded_processor = True
        if rank == 1 and failure == "processor":
            raise ValueError("rank-local processor loading fault")
        return json.loads((Path(checkpoint) / "statistics.json").read_text())

    try:
        with (
            patch.object(AutoModel, "from_pretrained", side_effect=load_model),
            patch.object(Gr00tN1d7Processor, "from_pretrained", side_effect=load_processor),
        ):
            try:
                Gr00tN1d7TrainPolicy(
                    make_model_config(str(directory / "source")), device="cpu", save_cfg_dir=output, ppo=True
                )
            except RuntimeError as error:
                retained = error
            else:
                raise AssertionError("injected loading/publication failure did not fail the policy constructor")
        gc.collect()
        report = {
            "error": str(retained),
            "models_released": all(reference() is None for reference in references),
            "processor_loaded": loaded_processor,
            "configuration_published": (output / "final_model_config.json").is_file(),
        }
        (directory / f"rank-{rank}.json").write_text(json.dumps(report))
    finally:
        dist.destroy_process_group()
