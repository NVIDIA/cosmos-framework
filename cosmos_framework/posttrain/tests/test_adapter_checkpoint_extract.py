# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Selective reads still enforce the full checkpoint and portable contracts."""

import pickle

import pytest
import torch
import torch.distributed.checkpoint as dcp

from cosmos_framework.posttrain.checkpoint.adapters import checkpoint_adapter_state, state_spec
from cosmos_framework.posttrain.config.adapters import AdapterComponentConfig, AdapterConfig
from cosmos_framework.posttrain.model.adapters import install_adapters
from cosmos_framework.posttrain.tests.test_adapters import FIRST, identity, tiny_net


def checkpoint(path, kind, method="pissa", *, alias=False):
    net = tiny_net()
    targets = [FIRST]
    if alias:
        net.model.language_model.layers[1].self_attn.q_proj = net.get_submodule(FIRST)
        targets.append(FIRST.replace("layers.0", "layers.1"))
    manifest = install_adapters(
        net, AdapterConfig(components=[AdapterComponentConfig(targets=targets, method=method, rank=2)])
    )
    contract = identity(manifest)
    state = net.state_dict()
    expected = {n: state[n].clone() for n in state_spec(contract)}
    path.mkdir()
    torch.save({"objective_identity": {"adapters": contract}, "optimizer_step": 3, "iter": 3}, path / "trainer.pt")
    # If extraction opens the optimizer this deliberately invalid payload fails.
    (path / "optim.pt").write_bytes(b"optimizer must never be deserialized")
    if kind == "pt":
        torch.save({"net": state}, path / "model.pt")
    else:
        dcp.save(state, checkpoint_id=path / "model")
    return contract, expected


@pytest.mark.parametrize("kind", ["pt", "dcp"])
@pytest.mark.parametrize("method", ["lora", "dora", "pissa"])
def test_exact_selected_inventory_including_aliases_and_initial_factors(tmp_path, kind, method, monkeypatch):
    contract, expected = checkpoint(tmp_path / "checkpoint", kind, method, alias=True)
    if kind == "dcp":
        original = dcp.FileSystemReader.read_data

        def selected_only(self, plan, planner):
            assert {item.dest_index.fqn for item in plan.items} == set(expected)
            return original(self, plan, planner)

        monkeypatch.setattr(dcp.FileSystemReader, "read_data", selected_only)
    actual_contract, actual, provenance = checkpoint_adapter_state(str(tmp_path / "checkpoint"))
    assert actual_contract == contract
    assert set(actual) == set(expected)
    assert provenance["source"] == str(tmp_path / "checkpoint")
    for name in expected:
        assert torch.equal(actual[name], expected[name])
        assert actual[name].untyped_storage().nbytes() == actual[name].numel() * 4


@pytest.mark.parametrize("kind", ["pt", "dcp"])
def test_incomplete_checkpoint_rejected_before_extraction(tmp_path, kind):
    checkpoint(tmp_path / "checkpoint", kind)
    (tmp_path / "checkpoint" / "optim.pt").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        checkpoint_adapter_state(str(tmp_path / "checkpoint"))


def test_dcp_unselected_frozen_extent_is_checked(tmp_path):
    checkpoint(tmp_path / "checkpoint", "dcp")
    path = tmp_path / "checkpoint/model/.metadata"
    metadata = pickle.loads(path.read_bytes())
    selected = next(item for key, item in metadata.storage_data.items() if key.fqn.endswith("q_proj.weight"))
    selected.length = (tmp_path / "checkpoint/model" / selected.relative_path).stat().st_size + 1
    path.write_bytes(pickle.dumps(metadata))
    with pytest.raises(ValueError, match="incomplete DCP storage extent"):
        checkpoint_adapter_state(str(tmp_path / "checkpoint"))


def test_dcp_missing_adapter_never_reinitialized(tmp_path):
    checkpoint(tmp_path / "checkpoint", "dcp")
    path = tmp_path / "checkpoint/model/.metadata"
    metadata = pickle.loads(path.read_bytes())
    del metadata.state_dict_metadata[FIRST + ".initial_A"]
    path.write_bytes(pickle.dumps(metadata))
    with pytest.raises(ValueError, match="missing/invalid checkpoint adapter tensor"):
        checkpoint_adapter_state(str(tmp_path / "checkpoint"))


def test_consolidated_alias_disagreement_rejected(tmp_path):
    path = tmp_path / "checkpoint"
    checkpoint(path, "pt", alias=True)
    state = torch.load(path / "model.pt", weights_only=True)
    key = FIRST.replace("layers.0", "layers.1") + ".initial_A"
    state["net"][key] = state["net"][key] + 1
    torch.save(state, path / "model.pt")
    with pytest.raises(ValueError, match="aliases disagree"):
        checkpoint_adapter_state(str(path))


@pytest.mark.parametrize("suffix", ["", ".tmp", ".previous", ".lock"])
@pytest.mark.parametrize("spelling", ["", "/", "/./"])
def test_public_output_cannot_mutate_immutable_checkpoint(tmp_path, suffix, spelling, monkeypatch):
    from cosmos_framework.posttrain.checkpoint.tools import adapters

    source = tmp_path / ("destination" + suffix)
    source.mkdir()
    monkeypatch.setattr(adapters, "checkpoint_adapter_state", lambda _: pytest.fail("must reject before reading"))
    with pytest.raises(ValueError, match="overlap the immutable checkpoint"):
        adapters.main(["--checkpoint", str(source), "--out", str(tmp_path / "destination") + spelling])
    assert not list(source.iterdir())


def test_public_checkpoint_child_output_and_temporary_directory_are_rejected(tmp_path, monkeypatch):
    from cosmos_framework.posttrain.checkpoint.tools import adapters

    source = tmp_path / "checkpoint"
    source.mkdir()
    with pytest.raises(ValueError, match="overlap the immutable checkpoint"):
        adapters.main(["--checkpoint", str(source), "--out", str(source / "adapter")])
    monkeypatch.setattr(adapters.tempfile, "gettempdir", lambda: str(source))
    with pytest.raises(ValueError, match="TMPDIR must be outside"):
        adapters.main(
            [
                "--checkpoint",
                str(source),
                "--out",
                str(tmp_path / "export"),
                "--merged",
                "--base",
                str(tmp_path / "base"),
            ]
        )
    assert not list(source.iterdir())
