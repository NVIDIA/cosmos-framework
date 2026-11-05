# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Official loading semantics and production PPO construction failures before collectives."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from cosmos_framework.posttrain.model.gr00t.policy import BF16_PRECISION, build_official_gr00t_model
from cosmos_framework.posttrain.tests.helpers.gr00t_source import write_source
from cosmos_framework.posttrain.tests.helpers.gr00t_source_loading import SmallOfficialModel, run_loading_failure

pytestmark = [pytest.mark.L1, pytest.mark.CPU]


@pytest.mark.parametrize("failure", ["model", "processor", "publish"])
def test_production_policy_builder_converges_loading_failures(tmp_path: Path, failure: str) -> None:
    # No replacement of the official Pipeline, its rank context or distributed state.
    # Only the external HF model/processor loading results are controlled.
    write_source(tmp_path / "source")
    torch.multiprocessing.spawn(run_loading_failure, args=(str(tmp_path), failure), nprocs=2, join=True)
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    expected = "configuration publication" if failure == "publish" else f"rank-local {failure} loading fault"
    assert all(expected in report["error"] for report in reports)
    assert all("timed out" not in report["error"].lower() for report in reports)
    assert all(report["models_released"] and not report["configuration_published"] for report in reports)
    if failure != "model":
        assert all(report["processor_loaded"] for report in reports)


@pytest.mark.parametrize("mask_token,missing_mask", [(True, False), (True, True), (False, True)])
def test_local_loading_matches_official_kwargs_and_optional_mask_initialization(
    tmp_path: Path, missing_mask: bool, mask_token: bool
) -> None:
    from transformers import AutoModel

    calls = []

    def load_model(*args, **kwargs):
        calls.append((args, kwargs))
        return SmallOfficialModel(mask_token=mask_token), {
            "missing_keys": ["action_head.mask_token"] if missing_mask else [],
            "unexpected_keys": [],
            "mismatched_keys": [],
        }

    models = []
    with patch.object(AutoModel, "from_pretrained", side_effect=load_model):
        for ppo in (False, True):
            output = tmp_path / str(ppo)
            output.mkdir()
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(29)
                models.append(
                    build_official_gr00t_model(
                        "controlled-checkpoint",
                        save_cfg_dir=output,
                        revision="resolved-nested-revision",
                        precision=BF16_PRECISION,
                        local_files_only=True,
                        ppo=ppo,
                    )
                )
    assert calls[0] == calls[1]
    for name, tensor in models[0].state_dict().items():
        torch.testing.assert_close(models[1].state_dict()[name], tensor, rtol=0, atol=0)
    assert (tmp_path / "False" / "final_model_config.json").is_file()
    assert not (tmp_path / "True" / "final_model_config.json").exists()


def test_local_loading_rejects_the_same_weight_mismatches_as_official(tmp_path: Path) -> None:
    from transformers import AutoModel

    information = {key: ["backbone.weight"] for key in ("missing_keys", "unexpected_keys", "mismatched_keys")}

    def load_model(*args, **kwargs):
        return SmallOfficialModel(), information

    errors = []
    with patch.object(AutoModel, "from_pretrained", side_effect=load_model):
        for ppo in (False, True):
            with pytest.raises(RuntimeError, match="Checkpoint weight mismatch") as caught:
                build_official_gr00t_model("controlled-checkpoint", save_cfg_dir=tmp_path, ppo=ppo)
            errors.append(str(caught.value))
    assert errors[0] == errors[1]
