# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json
from pathlib import Path
from typing import Any

import pytest
from torch.utils.data import Dataset

from cosmos_framework.inference.dataset import (
    DatasetArgs,
    _apply_dataset_kwargs,
    _fixed_resolution,
    create_dataset,
    disable_text_tokenizers,
)
from cosmos_framework.inference.dataset_samples import MapDatasetSamples
from cosmos_framework.inference.sample_selection import select_limited_indices


class _TransformWithTokenizer:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.text_tokenizer = object()

    def __call__(self, sample: dict[str, Any], resolution: str | None = None) -> dict[str, Any]:
        return sample


class _ListDataset(Dataset):
    """Hydra-instantiable map-style dataset that records its constructor arguments."""

    def __init__(self, size: int, transform: Any = None, **kwargs: Any) -> None:
        self.size = size
        self.transform = transform
        self.kwargs = kwargs

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {"index": idx}

    def transform_sample(self) -> None:
        pass


def _write_config(tmp_path: Path, dataset: dict[str, Any]) -> Path:
    config = {
        "dataloader_train": None,
        "dataloader_val": {"dataloaders": {"val": {"dataloader": {"dataset": dataset}}}},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def _dataset_config(**fields: Any) -> dict[str, Any]:
    return {
        "_target_": f"{__name__}._ListDataset",
        "size": 20,
        "transform": {"_target_": f"{__name__}._TransformWithTokenizer", "options": {"keep": 1, "drop": 2}},
        **fields,
    }


@pytest.mark.L0
@pytest.mark.CPU
def test_num_samples_selects_a_deterministic_random_subset_by_default(tmp_path: Path) -> None:
    args = DatasetArgs(config_file=_write_config(tmp_path, _dataset_config()), sample_name="robot", num_samples=5)

    samples = create_dataset(args)

    assert isinstance(samples, MapDatasetSamples)
    expected = select_limited_indices(range(20), limit=5, seed=args.sample_selection_seed, namespace="robot")
    assert samples._sample_ids == expected
    assert samples._sample_ids != list(range(5))
    assert create_dataset(args)._sample_ids == expected


@pytest.mark.L0
@pytest.mark.CPU
def test_num_samples_first_strategy_keeps_the_leading_samples(tmp_path: Path) -> None:
    args = DatasetArgs(
        config_file=_write_config(tmp_path, _dataset_config()),
        num_samples=5,
        sample_stride=2,
        sample_selection_strategy="first",
    )

    assert create_dataset(args)._sample_ids == [0, 2, 4, 6, 8]


@pytest.mark.L0
@pytest.mark.CPU
def test_create_dataset_disables_the_transform_text_tokenizer(tmp_path: Path) -> None:
    samples = create_dataset(DatasetArgs(config_file=_write_config(tmp_path, _dataset_config())))

    assert samples._transform is not None
    assert samples._transform.text_tokenizer is None


@pytest.mark.L0
@pytest.mark.CPU
def test_transform_kwargs_merge_into_the_outer_config(tmp_path: Path) -> None:
    args = DatasetArgs(
        config_file=_write_config(tmp_path, _dataset_config()),
        transform_kwargs={"transform": {"options": {"drop": 3}}},
    )

    samples = create_dataset(args)

    assert samples._transform.kwargs == {"options": {"keep": 1, "drop": 3}}


@pytest.mark.L0
@pytest.mark.CPU
def test_resolution_kwarg_is_written_only_when_declared(tmp_path: Path) -> None:
    # The override targets the inner entry of ``list_of_datasets``, as upstream.
    undeclared = create_dataset(
        DatasetArgs(
            config_file=_write_config(tmp_path, _dataset_config(list_of_datasets=[{"name": "robot"}])),
            dataset_kwargs={"resolution": "256"},
        )
    )
    assert "resolution" not in undeclared._dataset.kwargs["list_of_datasets"][0]
    assert undeclared._resolution == "256"

    declared = create_dataset(
        DatasetArgs(
            config_file=_write_config(
                tmp_path, _dataset_config(list_of_datasets=[{"name": "robot", "resolution": "480"}])
            ),
            dataset_kwargs={"resolution": "256"},
        )
    )
    assert declared._dataset.kwargs["list_of_datasets"][0]["resolution"] == "256"
    assert declared._resolution == "256"


@pytest.mark.L0
@pytest.mark.CPU
def test_multi_resolution_policy_is_resolved_per_sample(tmp_path: Path) -> None:
    config_file = _write_config(tmp_path, _dataset_config(resolution=["480", "256"]))

    samples = create_dataset(DatasetArgs(config_file=config_file))

    assert samples._resolution is None


@pytest.mark.L0
@pytest.mark.CPU
def test_apply_dataset_kwargs_merges_nested_dicts() -> None:
    target: dict[str, Any] = {"a": 1, "nested": {"keep": 1, "replace": 2}, "scalar": {"x": 1}}

    _apply_dataset_kwargs(target, {"nested": {"replace": 3, "add": 4}, "scalar": 5, "new": {"y": 1}})

    assert target == {"a": 1, "nested": {"keep": 1, "replace": 3, "add": 4}, "scalar": 5, "new": {"y": 1}}


@pytest.mark.L0
@pytest.mark.CPU
def test_text_tokenizer_is_disabled_on_a_bound_sample_transform() -> None:
    own = _TransformWithTokenizer()
    dataset = _ListDataset(1, transform=own)

    disable_text_tokenizers(dataset.transform_sample)

    assert own.text_tokenizer is None


@pytest.mark.L0
@pytest.mark.CPU
def test_disable_text_tokenizers_tolerates_none_and_missing_attributes() -> None:
    disable_text_tokenizers(None, object())


class TestFixedResolution:
    """A declared resolution is a tier to apply, or a policy to resolve per sample."""

    @pytest.mark.parametrize("declared", ["256", "480", "720"])
    def test_single_tier_is_applied_to_every_sample(self, declared: str) -> None:
        assert _fixed_resolution(declared) == declared

    def test_absent_resolution_stays_absent(self) -> None:
        assert _fixed_resolution(None) is None

    @pytest.mark.parametrize("declared", [("480", "256"), ["480", "256"], ("480",)])
    def test_multi_resolution_policy_defers_to_the_sample(self, declared: Any) -> None:
        assert _fixed_resolution(declared) is None

    def test_non_string_scalar_is_passed_through_unchanged(self) -> None:
        assert _fixed_resolution(480) == 480
