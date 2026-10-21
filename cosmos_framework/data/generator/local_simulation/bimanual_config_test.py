# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Stage-specific local data policy, without RGB decoding or model allocation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.data.generator.action.action_binary import get_action_binary_spec
from cosmos_framework.data.generator.action.agibot_grippers import _manifest, action_episode_gripper, episode_gripper
from cosmos_framework.data.generator.local_simulation import dataloader as loader_module
from cosmos_framework.data.generator.local_simulation.bimanual_config import make_bimanual_local_dataloader
from cosmos_framework.data.generator.local_simulation.dataset import LocalSimulationDataset, _model_constructors

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


def _request() -> dict[str, Any]:
    return {
        "draw_count": 0,
        "global_index": 0,
        "aux_seed": 0,
        "location": {"row_position": 0, "row_id": 0, "local_start_frame": 0},
        "source_frame_indices": list(range(901)),
        "row_ids_by_view": {},
        "source_version": 1,
        "source_index_version": 1,
        "dataset_length": 1,
        "index_space_fingerprint": "unit-test",
        "frame_count": 1000,
        "source_fps": 30.0,
        "shard_id": "unit-test",
        "episode_index": 0,
        "resolution": "480",
        "target_fps": 30.0,
        "caption": "Unit-test caption.",
        "caption_mode": "short",
    }


def _bundle(root: Path) -> None:
    legacy = "projects.cosmos3.cosmos3.datasets"
    entries = [
        (
            "agibot",
            f"{legacy}.action.canonical_lance_action_datasets.CanonicalLanceAgiBotDataset",
            "480",
            1.0,
            1,
        ),
        (
            "yam",
            f"{legacy}.action.canonical_lance_action_datasets.CanonicalLanceYAMDataset",
            "480",
            3.0,
            3,
        ),
    ]
    transforms: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for name, kind, resolution, weight, count in entries:
        transforms[name] = {
            "resolution": None,
            "entry": {"dataset": {"_target_": kind}, "ratio": weight, "resolution": resolution},
        }
        for index in range(count):
            request = _request()
            request["resolution"] = resolution
            records.append(
                {
                    "identity": f"{name}-{index}",
                    "stratum": name,
                    "request": request,
                    "assets": {},
                    "source_rows": {},
                }
            )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "model": "bimanual",
                "complete": True,
                "sample_count": len(records),
                "transforms": transforms,
            }
        )
    )
    (root / "samples.json").write_text(json.dumps(records))


class _CapturedChild:
    def __init__(self, dataset: LocalSimulationDataset, **kwargs: Any) -> None:
        self.dataset = dataset


class _CapturedJoint:
    def __init__(self, **kwargs: Any) -> None:
        self.streams: dict[str, Any] = kwargs["dataloaders"]


def _loader(monkeypatch: pytest.MonkeyPatch, root: Path, stage: int, **kwargs: Any) -> _CapturedJoint:
    # Test the real factory and real dataset descriptors; do not start packers/workers.
    monkeypatch.setattr(loader_module, "LocalSimulationDataLoader", _CapturedChild)
    monkeypatch.setattr(loader_module, "LocalSimulationJointDataLoader", _CapturedJoint)
    return make_bimanual_local_dataloader(str(root), stage, 16, 4, 2, None, 1, **kwargs)


def _retain_strata(root: Path, names: set[str]) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["transforms"] = {name: value for name, value in manifest["transforms"].items() if name in names}
    records = [record for record in json.loads((root / "samples.json").read_text()) if record["stratum"] in names]
    manifest["sample_count"] = len(records)
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "samples.json").write_text(json.dumps(records))


def test_exported_constructor_ids_bind_only_to_framework_classes() -> None:
    constructors = _model_constructors("bimanual")
    legacy = "projects.cosmos3.cosmos3.datasets"
    assert (
        constructors[f"{legacy}.action.canonical_lance_action_datasets.CanonicalLanceAgiBotDataset"].__name__
        == "CanonicalLanceAgiBotDataset"
    )
    assert (
        constructors[f"{legacy}.action.canonical_lance_action_datasets.CanonicalLanceYAMDataset"].__name__
        == "CanonicalLanceYAMDataset"
    )
    assert constructors[f"{legacy}.lance.lance_action_source.ActionSamplesColumns"].__name__ == "ActionSamplesColumns"


@pytest.mark.parametrize(
    ("source_dataset", "frame_values"),
    [
        ("agibot_gear__AgiBotWorld-GEAR_20260208_agibot-custom", 54),
        ("agibot_gear__AgiBotWorld-GEAR_20260208_agibot-offshelf", 54),
        ("agibot_gear_ext__AgiBotWorld-GEAR_20260208_agibot-custom-ext", 130),
    ],
)
def test_exported_action_binary_contract(source_dataset: str, frame_values: int) -> None:
    assert get_action_binary_spec(source_dataset).per_frame_value_count == frame_values


@pytest.mark.parametrize(
    ("legacy_dataset", "framework_dataset", "legacy_suffix"),
    [
        (
            "agibot_gear__AgiBotWorld-GEAR_20260208_agibot-custom",
            "embodiment_c__EmbodimentCWorld_20260208_agibot-custom",
            "AgiBotWorld-GEAR_20260208/agibot-custom",
        ),
        (
            "agibot_gear__AgiBotWorld-GEAR_20260208_agibot-offshelf",
            "embodiment_c__EmbodimentCWorld_20260208_agibot-offshelf",
            "AgiBotWorld-GEAR_20260208/agibot-offshelf",
        ),
        (
            "agibot_gear_ext__AgiBotWorld-GEAR_20260208_agibot-custom-ext",
            "embodiment_c_ext__EmbodimentCWorld_20260208_agibot-custom-ext",
            "AgiBotWorld-GEAR_20260208/agibot-custom-ext",
        ),
    ],
)
def test_exported_agibot_gripper_identity_aliases(
    legacy_dataset: str, framework_dataset: str, legacy_suffix: str
) -> None:
    dataset = _manifest()["datasets"][framework_dataset]
    shard_id, hardware_groups = next(iter(dataset["shards"].items()))
    hardware, episode_indices = next((name, indices) for name, indices in hardware_groups.items() if indices)
    episode_index = episode_indices[0]
    assert action_episode_gripper(legacy_dataset, shard_id, episode_index) == hardware
    source_id = f"bucket/prefix/{legacy_suffix}/{shard_id}/episode_{episode_index}/window"
    assert episode_gripper(source_id) == hardware


def test_unknown_captured_constructor_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["transforms"]["agibot"]["entry"]["dataset"]["_target_"] = "untrusted.UnknownDataset"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported Bimanual robot dataset constructor"):
        _loader(monkeypatch, tmp_path, 0)


def test_constructor_suffix_from_unknown_module_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _bundle(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["transforms"]["yam"]["entry"]["dataset"]["_target_"] = "untrusted.CanonicalLanceYAMDataset"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported Bimanual robot dataset constructor"):
        _loader(monkeypatch, tmp_path, 0)


@pytest.mark.parametrize("stage", range(4))
def test_default_stage_policy_combines_user_robot_streams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: int
) -> None:
    _bundle(tmp_path)
    loader = _loader(monkeypatch, tmp_path, stage)
    assert sum(stream["ratio"] for stream in loader.streams.values()) == pytest.approx(1.0)
    for family, family_weight in (("agibot", 0.25), ("yam", 0.75)):
        tiers = {"256": 1 / 3, "480": 2 / 3} if stage in (1, 2) else {"480": 1.0}
        for resolution, fraction in tiers.items():
            name = f"{family}_{resolution}p" if stage in (1, 2) else family
            stream = loader.streams[name]
            assert stream["ratio"] == pytest.approx(family_weight * fraction)
            assert stream["dataloader"].dataset.transform_overrides == {"resolution": resolution}
    assert len(loader.streams) == (4 if stage in (1, 2) else 2)


@pytest.mark.parametrize("stage", range(4))
def test_single_robot_source_is_a_complete_default_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: int
) -> None:
    _bundle(tmp_path)
    _retain_strata(tmp_path, {"agibot"})
    loader = _loader(monkeypatch, tmp_path, stage)
    assert sum(stream["ratio"] for stream in loader.streams.values()) == pytest.approx(1.0)
    assert set(loader.streams) == ({"agibot_256p", "agibot_480p"} if stage in (1, 2) else {"agibot"})


def test_default_bundle_requires_a_robot_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _bundle(tmp_path)
    _retain_strata(tmp_path, set())
    with pytest.raises(ValueError, match="at least one supported robot data source"):
        _loader(monkeypatch, tmp_path, 0)


@pytest.mark.parametrize("stage", range(4))
@pytest.mark.parametrize("observations", (17, 301, 901))
def test_robot_windows_accept_user_selected_lengths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: int, observations: int
) -> None:
    _bundle(tmp_path)
    records = json.loads((tmp_path / "samples.json").read_text())
    for record in records:
        record["request"]["source_frame_indices"] = list(range(observations))
    (tmp_path / "samples.json").write_text(json.dumps(records))
    loader = _loader(monkeypatch, tmp_path, stage)
    assert {
        len(record["request"]["source_frame_indices"])
        for stream in loader.streams.values()
        for record in stream["dataloader"].dataset.records
    } == {observations}


@pytest.mark.parametrize("resolution", (None, "256", "480"))
def test_resolution_override_reaches_decode_and_transform_without_changing_timing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resolution: str | None
) -> None:
    _bundle(tmp_path)
    dataset = LocalSimulationDataset(
        tmp_path, stratum="agibot", transform_overrides={} if resolution is None else {"resolution": resolution}
    )
    original = copy.deepcopy(dataset.records[0])
    seen: list[Any] = []

    def materialize(request: Any, rows: Any, backend: Any) -> dict[str, Any]:
        seen.append(request)
        return {}

    def transform(sample: dict[str, Any], request: Any) -> dict[str, Any]:
        seen.append(request)
        return sample

    pipeline = SimpleNamespace(
        dataset=SimpleNamespace(materialize_local_sample=materialize), transform_resolved_sample=transform
    )
    monkeypatch.setattr(dataset, "_pipeline", lambda stratum: pipeline)
    dataset.materialize(dataset.records[0], draw_seed=123)
    assert len(seen) == 2
    assert all(request.resolution == (resolution or "480") for request in seen)
    assert all(request.source_frame_indices == tuple(range(901)) and request.target_fps == 30.0 for request in seen)
    assert dataset.records[0] == original


def test_resume_identity_includes_effective_resolution(tmp_path: Path) -> None:
    _bundle(tmp_path)
    identities = []
    for resolution in ("256", "480"):
        dataset = LocalSimulationDataset(tmp_path, stratum="agibot", transform_overrides={"resolution": resolution})
        child = loader_module.LocalSimulationDataLoader(dataset, rank=0, world_size=1)
        identities.append(loader_module.LocalSimulationJointDataLoader._child_identity(child))
    assert identities[0]["bundle_digest"] == identities[1]["bundle_digest"]
    assert identities[0]["transform_overrides"] != identities[1]["transform_overrides"]
    assert identities[0] != identities[1]
