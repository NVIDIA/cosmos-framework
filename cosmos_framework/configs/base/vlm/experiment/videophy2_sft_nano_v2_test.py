# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Registration smoke test: videophy2_sft_nano_v2 must appear in Hydra ConfigStore."""

from __future__ import annotations

from hydra.core.config_store import ConfigStore


def test_v2_experiment_is_registered():
    import cosmos_framework.configs.base.vlm.experiment.videophy2_sft_nano_v2_experiment  # noqa: F401

    repo = ConfigStore.instance().repo
    assert "experiment" in repo
    names = set(repo["experiment"].keys())
    assert "videophy2_sft_nano_v2.yaml" in names, sorted(names)
