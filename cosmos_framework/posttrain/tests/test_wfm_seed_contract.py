# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Exact-resume tests for the online-WFM rollout seed contract."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch

from cosmos_framework.posttrain.roles.env import WFMDataEnv, WFMSeedContract, wfm_first_seed, wfm_seed_set

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


class _Loader:
    def __init__(self) -> None:
        self.start: int = 0
        self.set_calls: list[int] = []

    def set_start_iteration(self, iteration: int) -> None:
        self.start = iteration
        self.set_calls.append(iteration)

    def __iter__(self) -> Iterator[dict[str, object]]:
        index = self.start
        while True:
            yield {
                "video": [torch.full((1, 2), float(index))],  # one sample [C,T]
                "ai_caption": [f"prompt-{index}"],
            }
            index += 1

    def cursor_fingerprint(self) -> str:
        return "wfm-seed-contract-test-v1"


def _commit_one_ga4_window(contract: WFMSeedContract) -> tuple[WFMDataEnv, dict[str, object]]:
    source = WFMDataEnv(_Loader(), grad_accum=4, seed_contract=contract)
    for _ in range(4):
        source.get_samples(1)
        source.stage()
    source.commit_window(4)
    return source, source.state_dict()


def test_seed_clocks_define_distinct_ga4_sequences() -> None:
    source_clock = WFMSeedContract(base_seed=17, clock="source_row", group_size=8)
    optimizer_clock = WFMSeedContract(base_seed=17, clock="optimizer_step", group_size=8)

    source_sets = [tuple(wfm_seed_set(source_clock, source_row=row, optimizer_step=0)) for row in range(4)]
    optimizer_sets = [tuple(wfm_seed_set(optimizer_clock, source_row=row, optimizer_step=0)) for row in range(4)]

    assert [seeds[0] for seeds in source_sets] == [17, 10_017, 20_017, 30_017]
    assert len(set(source_sets)) == 4
    assert optimizer_sets == [tuple(range(17, 25))] * 4


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_rank_partitions_reconstruct_the_global_seed_set(world_size: int) -> None:
    contract = WFMSeedContract(base_seed=31, clock="optimizer_step", group_size=8)
    expected = tuple(wfm_seed_set(contract, source_row=7, optimizer_step=2))
    local_group_size = contract.group_size // world_size
    partitioned: list[int] = []
    for process_rank in range(world_size):
        first = wfm_first_seed(
            contract,
            source_row=7,
            optimizer_step=2,
            process_rank=process_rank,
            world_size=world_size,
        )
        partitioned.extend(range(first, first + local_group_size))
    assert tuple(partitioned) == expected


def test_seed_contract_is_bound_to_saved_cursor_and_next_seed() -> None:
    contract = WFMSeedContract(base_seed=23, clock="optimizer_step", group_size=8)
    _, state = _commit_one_ga4_window(contract)

    assert state["version"] == 2
    assert state["seed_contract"] == {
        "version": 1,
        "base_seed": 23,
        "clock": "optimizer_step",
        "stride": 10_000,
        "group_size": 8,
    }
    expected_next = tuple(wfm_seed_set(contract, source_row=4, optimizer_step=1))

    restored = WFMDataEnv(_Loader(), grad_accum=4, seed_contract=contract)
    assert restored.load_state_dict(state)
    restored_state = restored.state_dict()
    resumed_next = tuple(
        wfm_seed_set(
            contract,
            source_row=restored_state["committed_rows"],
            optimizer_step=restored_state["optimizer_step"],
        )
    )
    assert resumed_next == expected_next


def test_restore_rejects_changed_seed_contract_and_ga_before_loader_cursor_mutation() -> None:
    contract = WFMSeedContract(base_seed=23, clock="optimizer_step", group_size=8)
    _, state = _commit_one_ga4_window(contract)

    changed_contracts = (
        WFMSeedContract(base_seed=24, clock="optimizer_step", group_size=8),
        WFMSeedContract(base_seed=23, clock="source_row", group_size=8),
        WFMSeedContract(base_seed=23, clock="optimizer_step", group_size=4),
    )
    for changed_contract in changed_contracts:
        changed_loader = _Loader()
        changed = WFMDataEnv(changed_loader, grad_accum=4, seed_contract=changed_contract)
        with pytest.raises(ValueError, match="seed contract differs"):
            changed.load_state_dict(state)
        assert changed_loader.set_calls == [0]
        assert changed.state_dict()["committed_rows"] == 0

    changed_ga_loader = _Loader()
    changed_ga = WFMDataEnv(changed_ga_loader, grad_accum=2, seed_contract=contract)
    with pytest.raises(ValueError, match="grad_accum changed"):
        changed_ga.load_state_dict(state)
    assert changed_ga_loader.set_calls == [0]
    assert changed_ga.state_dict()["committed_rows"] == 0


def test_version_one_state_stays_default_only() -> None:
    legacy = WFMDataEnv(_Loader())
    state = legacy.state_dict()
    assert state["version"] == 1

    configured = WFMDataEnv(
        _Loader(),
        seed_contract=WFMSeedContract(base_seed=0, clock="source_row", group_size=8),
    )
    with pytest.raises(ValueError, match="version-1 state has no seed contract"):
        configured.load_state_dict(state)

    assert WFMDataEnv(_Loader()).load_state_dict(state)


def test_seed_contract_rejects_overlap_partition_errors_and_int64_overflow() -> None:
    with pytest.raises(ValueError, match="stride must be an int64 >= group_size"):
        WFMSeedContract(base_seed=0, clock="source_row", group_size=8, stride=7)

    indivisible = WFMSeedContract(base_seed=0, clock="source_row", group_size=8)
    with pytest.raises(ValueError, match="must be divisible"):
        wfm_first_seed(
            indivisible,
            source_row=0,
            optimizer_step=0,
            process_rank=0,
            world_size=3,
        )

    int64_edge = WFMSeedContract(
        base_seed=0,
        clock="source_row",
        group_size=1,
        stride=(1 << 63) - 1,
    )
    assert tuple(wfm_seed_set(int64_edge, source_row=1, optimizer_step=0)) == ((1 << 63) - 1,)
    with pytest.raises(ValueError, match="exceeds the non-negative int64 range"):
        wfm_seed_set(int64_edge, source_row=2, optimizer_step=0)
