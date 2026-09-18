# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from cosmos_framework.model.generator.mot.diffusion_cache import DiffusionCache, _extract_timestep_key


def _const_indicator(value: float = 1.0) -> list[torch.Tensor]:
    return [torch.full((1, 2, 2, 1), value)]


def _dummy_residual() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(1, 2), torch.zeros(1, 2)


class _FakeLanguageModel:
    DELTA_CAUSAL = 1.0
    DELTA_FULL = 2.0

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, pack: dict, **kwargs: Any) -> tuple[dict, dict]:
        self.calls += 1
        out = dict(pack)
        out["causal_seq"] = pack["causal_seq"] + self.DELTA_CAUSAL
        out["full_only_seq"] = pack["full_only_seq"] + self.DELTA_FULL
        return out, {}


class _FakeNet:
    def __init__(self) -> None:
        self.language_model = _FakeLanguageModel()

    def forward(self, packed_seq: Any, memory: Any = None) -> dict:
        del memory
        pack = {
            "causal_seq": packed_seq._causal,
            "full_only_seq": packed_seq._full,
            "is_sharded": False,
        }
        out_pack, _ = self.language_model(pack)
        return {"preds_vision": [out_pack["full_only_seq"].clone()]}


class _FakeModel:
    def __init__(self) -> None:
        self.net = _FakeNet()
        self.sampler = None

    def generate_samples_from_batch(self, data_batch: Any = None, **kwargs: Any) -> dict:
        del data_batch, kwargs
        return {"vision": []}

    def denoise(self, net: Any = None, data_batch_packed: Any = None, memory: Any = None) -> dict:
        del net
        return {"preds_vision": self.net.forward(data_batch_packed, memory=memory)["preds_vision"]}


def _make_batch(step: int, total: int, full_value: float) -> SimpleNamespace:
    timestep = 1.0 - step / total
    latent = torch.zeros(2, 1, 2, 2)
    batch = SimpleNamespace(
        vision=SimpleNamespace(
            tokens=[latent],
            token_shapes=[(1, 2, 2)],
            timesteps=torch.tensor([timestep]),
        )
    )
    batch._causal = torch.zeros(2, 4)
    batch._full = torch.full((3, 4), full_value)
    return batch


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("refresh_branch", [0, 1, 2])
def test_control_cfg_preflight_refreshes_all_without_replay(
    refresh_branch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _FakeModel()
    cache = DiffusionCache(num_steps=10)
    monkeypatch.setattr(cache, "_sea_ab", lambda timestep_key: (0.5, 0.5))
    cache.install(SimpleNamespace(model=model))

    def run_step(step: int) -> list[torch.Tensor]:
        batches = [_make_batch(step, 10, full_value=10.0) for _ in range(3)]
        if step > 0:
            batches[refresh_branch].vision.tokens[0].fill_(1.0)
        before = model.net.language_model.calls
        cache.prepare_control_cfg_step(
            {f"cfg{i}": batch.vision.tokens for i, batch in enumerate(batches)},
            _extract_timestep_key(batches[0]),
        )
        assert model.net.language_model.calls == before
        results: list[torch.Tensor] = []
        for branch, batch in enumerate(batches):
            model.net.language_model.DELTA_FULL = float(100 * branch + step)
            results.append(model.denoise(data_batch_packed=batch)["preds_vision"][0])
        assert not cache._control_cfg_pending
        return results

    run_step(0)
    assert model.net.language_model.calls == 3
    outputs = run_step(1)
    assert model.net.language_model.calls == 6
    assert set(cache._pathways) == {"cfg0", "cfg1", "cfg2"}
    for branch, output in enumerate(outputs):
        torch.testing.assert_close(output, torch.full_like(output, 11.0 + 100 * branch))
        pathway = cache._pathways[f"cfg{branch}"]
        assert [index for index, _ in pathway.history] == [0, 1]
        assert pathway.consecutive_cached == 0
        assert pathway.accumulated == 0.0

    run_step(2)
    run_step(3)
    assert model.net.language_model.calls == 6
    assert all(pathway.consecutive_cached == 2 for pathway in cache._pathways.values())
    run_step(4)
    assert model.net.language_model.calls == 9
    assert cache._step_full == 9
    assert cache._step_skipped == 6


@pytest.mark.L0
@pytest.mark.CPU
def test_control_cfg_preflight_peer_refreshes_all(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel()
    cache = DiffusionCache(num_steps=10)
    monkeypatch.setattr(cache, "_sea_ab", lambda timestep_key: (0.5, 0.5))
    cache.install(SimpleNamespace(model=model))
    for branch in range(3):
        model.denoise(data_batch_packed=_make_batch(0, 10, full_value=10.0))
        cache._pathways[f"cfg{branch}"].accumulated = 0.1
    cache._dp_shard_group = object()
    flags: list[int] = []

    def peer_requires_full(flag: torch.Tensor, **kwargs: Any) -> None:
        del kwargs
        flags.append(int(flag.item()))
        flag.fill_(1)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "all_reduce", peer_requires_full)
    batch = _make_batch(1, 10, full_value=10.0)
    cache.prepare_control_cfg_step(
        {f"cfg{i}": batch.vision.tokens for i in range(3)},
        _extract_timestep_key(batch),
    )
    assert flags == [0]
    assert all(pathway.accumulated == 0.0 for pathway in cache._pathways.values())
    for _ in range(3):
        model.denoise(data_batch_packed=batch)
    assert model.net.language_model.calls == 6
    assert flags == [0]


@pytest.mark.L0
@pytest.mark.CPU
def test_control_cfg_interval_preserves_branch_histories(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel()
    cache = DiffusionCache(num_steps=10)
    monkeypatch.setattr(cache, "_sea_ab", lambda timestep_key: (0.5, 0.5))
    cache.install(SimpleNamespace(model=model))

    for step, branches in enumerate([(0, 1, 2), (0, 2), (0, 1, 2)]):
        batch = _make_batch(step, 10, full_value=10.0)
        cache.prepare_control_cfg_step(
            {f"cfg{i}": batch.vision.tokens for i in branches},
            _extract_timestep_key(batch),
        )
        for branch in branches:
            model.net.language_model.DELTA_FULL = float(100 * branch)
            output = model.denoise(data_batch_packed=batch)["preds_vision"][0]
            torch.testing.assert_close(output, torch.full_like(output, 10.0 + 100 * branch))
    assert model.net.language_model.calls == 3


@pytest.mark.L0
@pytest.mark.CPU
def test_max_consecutive_cached_forces_full_per_pathway() -> None:
    cache = DiffusionCache(
        num_steps=20,
        config={
            "ret_steps": 0,
            "cutoff_from_end": 0,
            "diffusion_cache_thresh": 1.0,
            "max_consecutive_cached": 3,
        },
    )
    cache.state.step = 0
    assert cache._should_compute("cfg0", _const_indicator()) is True
    cache._pathways["cfg0"].history = [(0, _dummy_residual())]

    for step in range(1, 4):
        cache.state.step = step
        assert cache._should_compute("cfg0", _const_indicator()) is False
        cache._pathways["cfg0"].consecutive_cached += 1

    cache.state.step = 4
    assert cache._should_compute("cfg0", _const_indicator()) is True


@pytest.mark.L0
@pytest.mark.CPU
def test_diffusion_cache_uses_tuned_defaults() -> None:
    cache = DiffusionCache(num_steps=10)

    assert cache.config.diffusion_cache_thresh == pytest.approx(0.25)
    assert cache.config.max_consecutive_cached == 2


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_max_consecutive_cached_rejected(value: object) -> None:
    with pytest.raises(ValueError, match="max_consecutive_cached"):
        DiffusionCache(num_steps=10, config={"max_consecutive_cached": value})
