# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Hermetic tests for ActionIterableShuffleDataset."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


@pytest.fixture
def action_sft_dataset_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "cosmos_framework.data.vfm.action.datasets.action_sft_dataset"

    fake_torch = types.ModuleType("torch")
    fake_torch_utils = types.ModuleType("torch.utils")
    fake_torch_utils_data = types.ModuleType("torch.utils.data")
    fake_torch_utils_data.Dataset = type("Dataset", (), {})
    fake_torch_utils_data.IterableDataset = type("IterableDataset", (), {})
    fake_torch_utils_data.get_worker_info = lambda: None
    fake_torch.utils = fake_torch_utils

    fake_datasets_package = types.ModuleType("cosmos_framework.data.vfm.action.datasets")
    fake_datasets_package.__path__ = [
        "/Users/hoangvu/Code/OSS/cosmos-framework/cosmos_framework/data/vfm/action/datasets"
    ]

    fake_droid_dataset = types.ModuleType("cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset")
    fake_droid_dataset.DROIDLeRobotDataset = type("DROIDLeRobotDataset", (), {})

    fake_transforms = types.ModuleType("cosmos_framework.data.vfm.action.transforms")
    fake_transforms.ActionTransformPipeline = type("ActionTransformPipeline", (), {})

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", fake_torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", fake_torch_utils_data)
    monkeypatch.setitem(sys.modules, "cosmos_framework.data.vfm.action.datasets", fake_datasets_package)
    monkeypatch.setitem(
        sys.modules, "cosmos_framework.data.vfm.action.datasets.droid_lerobot_dataset", fake_droid_dataset
    )
    monkeypatch.setitem(sys.modules, "cosmos_framework.data.vfm.action.transforms", fake_transforms)
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)
    yield module

    sys.modules.pop(module_name, None)


def test_action_iterable_shuffle_dataset_raises_when_shuffle_blocks_are_empty(action_sft_dataset_module) -> None:
    class _Dataset:
        def get_shuffle_blocks(self):
            return []

        def __len__(self):
            return 0

    dataset = action_sft_dataset_module.ActionIterableShuffleDataset(_Dataset())

    with pytest.raises(ValueError, match="No shuffle blocks found"):
        next(iter(dataset))
