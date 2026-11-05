# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Focused copy-mode inventory tests for consolidated and DCP checkpoints."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from cosmos_framework.posttrain.algorithm.dpo import make_objective
from cosmos_framework.posttrain.checkpoint import save_component_dcp
from cosmos_framework.posttrain.model.cosmos3 import validate_cosmos3_cold_lora_inventory
from cosmos_framework.posttrain.roles.trainer.diffusion import CopyInventoryPolicy, DiffusionTrainer
from cosmos_framework.posttrain.tests.helpers.toy_model import ToyFlowModel


class _InventoryNet(torch.nn.Module):
    """Small state-dict schema with optional source-only and target-only tensors."""

    proj: torch.nn.Linear
    sound: torch.nn.Linear | None
    extra: torch.nn.Parameter | None

    def __init__(
        self,
        *,
        width: int = 2,
        sound: bool = False,
        lora: str = "none",
        extra: bool = False,
    ) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(2, width)
        self.sound = torch.nn.Linear(2, 2, bias=False) if sound else None
        if lora in ("a", "full"):
            self.proj.lora_A = torch.nn.Linear(2, 1, bias=False)
        if lora in ("b", "full"):
            self.proj.lora_B = torch.nn.Linear(1, width, bias=False)
        if lora not in ("none", "a", "b", "full"):
            raise ValueError(f"unsupported synthetic LoRA schema {lora!r}")
        self.extra = torch.nn.Parameter(torch.ones(1)) if extra else None  # [1]

    def forward(self, value: torch.Tensor) -> torch.Tensor:  # value: [B,2], returns [B,W]
        return self.proj(value)  # [B,W]


def _fill(module: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        for tensor in module.state_dict().values():
            tensor.fill_(value)


def _make_trainer(target: _InventoryNet, *, ema_target: _InventoryNet | None = None) -> DiffusionTrainer:
    model = ToyFlowModel(dim=2, seed=0)
    model.net = target
    model.net_old = deepcopy(target)
    model.net_ref = deepcopy(target)
    model.net_ema = deepcopy(target if ema_target is None else ema_target)
    _fill(model.net, 10.0)
    _fill(model.net_old, 20.0)
    _fill(model.net_ref, 30.0)
    _fill(model.net_ema, 40.0)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=1e-3)
    return DiffusionTrainer(
        model,
        make_objective(beta=1.0),
        optimizer,
        update_ema_after_iter=True,
        defer_frozen_reference_identity=True,
    )


def _write_checkpoint(
    path: Path,
    source: _InventoryNet,
    *,
    checkpoint_format: str,
    add_sibling_namespace: bool = True,
) -> None:
    path.mkdir(parents=True)
    source_state = {key: value.detach().clone() for key, value in source.state_dict().items()}  # [*P]
    sibling_state = {key: torch.full_like(value, -500.0) for key, value in source_state.items()}  # [*P]
    if checkpoint_format == "consolidated":
        model_state: dict[str, dict[str, torch.Tensor]] = {"net": source_state}
        if add_sibling_namespace:
            model_state["net_ema"] = sibling_state
        torch.save(model_state, path / "model.pt")
        return
    dcp_state = {f"net.{key}": value for key, value in source_state.items()}
    if add_sibling_namespace:
        dcp_state.update({f"net_ema.{key}": value for key, value in sibling_state.items()})
    save_component_dcp(str(path), "model", dcp_state)


def _snapshot(trainer: DiffusionTrainer) -> dict[str, dict[str, torch.Tensor]]:
    return {
        role: {key: value.detach().clone() for key, value in module.state_dict().items()}  # [*P]
        for role, module in (
            ("policy", trainer.model.net),
            ("old", trainer.model.net_old),
            ("reference", trainer.model.net_ref),
            ("ema", trainer.model.net_ema),
        )
    }


def _assert_snapshot_equal(
    actual: dict[str, dict[str, torch.Tensor]],
    expected: dict[str, dict[str, torch.Tensor]],
) -> None:
    assert actual.keys() == expected.keys()
    for role in expected:
        assert actual[role].keys() == expected[role].keys()
        for key in expected[role]:
            assert torch.equal(actual[role][key], expected[role][key])


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_copy_inventory_loads_base_preserves_complete_cold_lora_and_ignores_siblings(
    tmp_path: Path,
    checkpoint_format: str,
) -> None:
    source = _InventoryNet(sound=True)
    _fill(source, 7.0)
    checkpoint = tmp_path / checkpoint_format
    _write_checkpoint(checkpoint, source, checkpoint_format=checkpoint_format)
    trainer = _make_trainer(_InventoryNet(lora="full"))
    before = _snapshot(trainer)

    trainer.load(
        str(checkpoint),
        copy_mode=True,
        copy_inventory_policy=CopyInventoryPolicy(
            source_only_keys=("sound.weight",),
            target_only_validator=validate_cosmos3_cold_lora_inventory,
        ),
    )

    after = _snapshot(trainer)
    for role, state in after.items():
        assert torch.equal(state["proj.weight"], torch.full_like(state["proj.weight"], 7.0))
        assert torch.equal(state["proj.bias"], torch.full_like(state["proj.bias"], 7.0))
        assert torch.equal(state["proj.lora_A.weight"], before[role]["proj.lora_A.weight"])
        assert torch.equal(state["proj.lora_B.weight"], before[role]["proj.lora_B.weight"])


@pytest.mark.L0
@pytest.mark.CPU
def test_consolidated_copy_inventory_uses_sharded_target_inventory_on_every_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inventory must not request the rank-zero-only full/offloaded FSDP state."""
    source = _InventoryNet()
    _fill(source, 7.0)
    checkpoint = tmp_path / "consolidated"
    _write_checkpoint(checkpoint, source, checkpoint_format="consolidated")
    trainer = _make_trainer(_InventoryNet())
    getter_options: list[Any] = []
    setter_options: list[Any] = []

    def fake_get_model_state_dict(module: torch.nn.Module, *, options: Any = None, **_kwargs: Any) -> dict[str, Any]:
        getter_options.append(options)
        # Torch returns the full/offloaded state on rank zero only. Simulate a peer rank so
        # the regression fails if inventory planning accidentally gathers again.
        if options is not None and options.full_state_dict and options.cpu_offload:
            return {}
        return dict(module.state_dict())  # [*P]

    def fake_set_model_state_dict(
        module: torch.nn.Module,
        model_state_dict: dict[str, Any],
        *,
        options: Any,
    ) -> None:
        setter_options.append(options)
        module.load_state_dict(model_state_dict, strict=False)

    monkeypatch.setattr(
        "cosmos_framework.posttrain.roles.trainer.diffusion.is_dtensor_sharded",
        lambda _module: True,
    )
    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.get_model_state_dict",
        fake_get_model_state_dict,
    )
    monkeypatch.setattr(
        "torch.distributed.checkpoint.state_dict.set_model_state_dict",
        fake_set_model_state_dict,
    )

    trainer.load(str(checkpoint), copy_mode=True)

    assert len(getter_options) == 4
    assert getter_options == [None] * 4
    assert len(setter_options) == 4
    assert all(options.full_state_dict and options.cpu_offload for options in setter_options)
    for state in _snapshot(trainer).values():
        assert torch.equal(state["proj.weight"], torch.full_like(state["proj.weight"], 7.0))
        assert torch.equal(state["proj.bias"], torch.full_like(state["proj.bias"], 7.0))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
@pytest.mark.parametrize(
    ("source_kwargs", "target_kwargs", "policy", "match"),
    [
        pytest.param(
            {"sound": True},
            {"lora": "full"},
            CopyInventoryPolicy(),
            "unapproved source-only",
            id="source-only",
        ),
        pytest.param({}, {"lora": "full"}, CopyInventoryPolicy(), "unapproved target-only", id="target-only"),
        pytest.param(
            {"sound": True},
            {"lora": "full"},
            CopyInventoryPolicy(
                source_only_keys=("missing.weight",),
                target_only_validator=validate_cosmos3_cold_lora_inventory,
            ),
            "absent from the source",
            id="missing-configured-key",
        ),
        pytest.param(
            {"sound": True},
            {"sound": True},
            CopyInventoryPolicy(source_only_keys=("sound.weight",)),
            "but they exist in the target",
            id="configured-key-in-target",
        ),
        pytest.param(
            {},
            {"extra": True},
            CopyInventoryPolicy(target_only_validator=validate_cosmos3_cold_lora_inventory),
            "non-LoRA target-only",
            id="non-lora-target-only",
        ),
        pytest.param(
            {},
            {"lora": "a"},
            CopyInventoryPolicy(target_only_validator=validate_cosmos3_cold_lora_inventory),
            "incomplete target-only LoRA",
            id="partial-lora",
        ),
        pytest.param(
            {"lora": "a"},
            {"lora": "full"},
            CopyInventoryPolicy(target_only_validator=validate_cosmos3_cold_lora_inventory),
            "source already contains LoRA",
            id="source-has-lora",
        ),
        pytest.param({}, {"width": 3}, CopyInventoryPolicy(), "tensor shape drift", id="shape-drift"),
    ],
)
def test_copy_inventory_rejects_drift_before_mutating_any_policy_copy(
    tmp_path: Path,
    checkpoint_format: str,
    source_kwargs: dict[str, object],
    target_kwargs: dict[str, object],
    policy: CopyInventoryPolicy,
    match: str,
) -> None:
    source = _InventoryNet(**source_kwargs)
    _fill(source, 7.0)
    checkpoint = tmp_path / checkpoint_format
    _write_checkpoint(checkpoint, source, checkpoint_format=checkpoint_format)
    trainer = _make_trainer(_InventoryNet(**target_kwargs))
    before = _snapshot(trainer)

    with pytest.raises(RuntimeError, match=match):
        trainer.load(str(checkpoint), copy_mode=True, copy_inventory_policy=policy)

    _assert_snapshot_equal(_snapshot(trainer), before)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("checkpoint_format", ["consolidated", "dcp"])
def test_copy_inventory_validates_later_policy_copies_before_loading_the_policy(
    tmp_path: Path,
    checkpoint_format: str,
) -> None:
    source = _InventoryNet()
    _fill(source, 7.0)
    checkpoint = tmp_path / checkpoint_format
    _write_checkpoint(checkpoint, source, checkpoint_format=checkpoint_format)
    trainer = _make_trainer(_InventoryNet(), ema_target=_InventoryNet(extra=True))
    before = _snapshot(trainer)

    with pytest.raises(RuntimeError, match="ema.*unapproved target-only"):
        trainer.load(str(checkpoint), copy_mode=True)

    _assert_snapshot_equal(_snapshot(trainer), before)


@pytest.mark.L0
@pytest.mark.CPU
def test_copy_inventory_policy_is_rejected_for_exact_resume(tmp_path: Path) -> None:
    trainer = _make_trainer(_InventoryNet())
    with pytest.raises(ValueError, match="valid only with copy_mode=True"):
        trainer.load(
            str(tmp_path / "unused"),
            copy_inventory_policy=CopyInventoryPolicy(),
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_copy_inventory_policy_rejects_non_callable_target_only_validator() -> None:
    invalid_validator: Any = object()
    with pytest.raises(TypeError, match="target_only_validator must be callable"):
        CopyInventoryPolicy(target_only_validator=invalid_validator)
