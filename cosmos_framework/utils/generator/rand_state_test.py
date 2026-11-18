# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import builtins
import pickle
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from cosmos_framework.utils.generator import rand_state
from cosmos_framework.utils.generator.rand_state import (
    MAX_PACKED_LEN,
    get_rand_state_dict,
    pad_packed_bytes,
    set_rand_state_dict,
    unpack_numpy_state,
    unpack_random_state,
)
from cosmos_framework.utils.safe_unpickle import ForbiddenPickleGlobalError

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


class _CallOnLoad:
    def __init__(self, func: Any, *args: Any) -> None:
        self.func = func
        self.args = args

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return self.func, self.args


@pytest.fixture(autouse=True)
def _cpu_only_cuda_rng(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep the CUDA generator out of these tests so they run on CPU-only machines."""
    monkeypatch.setattr(rand_state.torch.cuda, "get_rng_state", lambda: torch.zeros(16, dtype=torch.uint8))
    monkeypatch.setattr(rand_state.torch.cuda, "set_rng_state", lambda state: None)
    torch_state, numpy_state, random_state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    yield
    torch.set_rng_state(torch_state)
    np.random.set_state(numpy_state)
    random.setstate(random_state)


def _draws() -> list[tuple[float, float, float]]:
    return [(torch.randn(1).item(), float(np.random.rand()), random.random()) for _ in range(5)]


def _snapshot() -> tuple[torch.Tensor, tuple[Any, ...], tuple[Any, ...]]:
    return torch.get_rng_state(), np.random.get_state(), random.getstate()


def _assert_snapshot_equal(a: tuple[torch.Tensor, tuple[Any, ...], tuple[Any, ...]], b: Any) -> None:
    assert torch.equal(a[0], b[0])
    assert a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and a[1][2:] == b[1][2:]
    assert a[2] == b[2]


def _with_packed(state_dict: dict[str, torch.Tensor], prefix: str, packed: bytes) -> dict[str, torch.Tensor]:
    tampered = dict(state_dict)
    tampered[f"{prefix}_packed_len"] = torch.tensor(len(packed), dtype=torch.long)
    tampered[f"{prefix}_packed_bytes"] = torch.frombuffer(pad_packed_bytes(packed), dtype=torch.uint8)
    return tampered


def _seed_and_advance() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    for _ in range(10):
        torch.randn(1)
        np.random.standard_normal()
        random.gauss(0, 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_restore_reproduces_draws() -> None:
    _seed_and_advance()
    state_dict = get_rand_state_dict()
    expected = _draws()
    _draws()

    set_rand_state_dict(state_dict)

    assert _draws() == expected


@pytest.mark.L0
@pytest.mark.CPU
def test_saved_format_is_unchanged_and_plain_pickle_readable() -> None:
    # Older readers decode these buffers with plain pickle.loads, so the saved layout must not change.
    _seed_and_advance()
    state_dict = get_rand_state_dict()

    assert set(state_dict) == {
        "torch",
        "torch_cuda",
        "numpy_packed_len",
        "numpy_packed_bytes",
        "random_packed_len",
        "random_packed_bytes",
    }
    for prefix, expected_state in (("numpy", np.random.get_state()), ("random", random.getstate())):
        packed_len = state_dict[f"{prefix}_packed_len"]
        packed_bytes = state_dict[f"{prefix}_packed_bytes"]
        assert packed_len.dtype == torch.long and packed_len.shape == ()
        assert packed_bytes.dtype == torch.uint8 and packed_bytes.shape == (MAX_PACKED_LEN,)
        decoded = pickle.loads(bytes(packed_bytes.tolist())[: packed_len.item()])
        assert pickle.dumps(decoded) == pickle.dumps(expected_state)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_legacy_packed_states_decode(protocol: int) -> None:
    _seed_and_advance()
    np.random.standard_normal()  # leaves a cached gaussian, so has_gauss == 1
    numpy_state, random_state = np.random.get_state(), random.getstate()

    numpy_packed = pickle.dumps(numpy_state, protocol=protocol)
    random_packed = pickle.dumps(random_state, protocol=protocol)
    decoded_numpy = unpack_numpy_state(len(numpy_packed), numpy_packed)
    decoded_random = unpack_random_state(len(random_packed), random_packed)

    assert decoded_numpy[0] == numpy_state[0] and np.array_equal(decoded_numpy[1], numpy_state[1])
    assert decoded_numpy[2:] == numpy_state[2:]
    assert decoded_random == random_state


@pytest.mark.L0
@pytest.mark.CPU
def test_numpy_1x_module_path_decodes() -> None:
    packed = pickle.dumps(np.random.get_state(), protocol=2)
    assert b"numpy._core.multiarray" in packed or b"numpy.core.multiarray" in packed
    packed = packed.replace(b"numpy._core.multiarray", b"numpy.core.multiarray")

    decoded = unpack_numpy_state(len(packed), packed)

    assert np.array_equal(decoded[1], np.random.get_state()[1])


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("prefix", ["numpy", "random"])
def test_code_payload_is_rejected_and_state_untouched(prefix: str, tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    payload = pickle.dumps(_CallOnLoad(builtins.exec, f"open({str(marker)!r}, 'w').write('loaded')"))
    state_dict = _with_packed(get_rand_state_dict(), prefix, payload)
    torch.randn(1)
    before = _snapshot()

    with pytest.raises(ForbiddenPickleGlobalError, match="forbidden global builtins.exec"):
        set_rand_state_dict(state_dict)

    assert not marker.exists()
    _assert_snapshot_equal(before, _snapshot())


def _numpy_state(**overrides: Any) -> tuple[Any, ...]:
    algorithm, key, pos, has_gauss, cached_gaussian = np.random.get_state()
    fields = {"algorithm": algorithm, "key": key, "pos": pos, "has_gauss": has_gauss, "gauss": cached_gaussian}
    fields.update(overrides)
    return tuple(fields.values())


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "state",
    [
        pytest.param(lambda: _numpy_state(algorithm="PCG64"), id="algorithm"),
        pytest.param(lambda: _numpy_state(key=np.zeros(624, dtype=np.int64)), id="key-dtype"),
        pytest.param(lambda: _numpy_state(key=np.zeros(623, dtype=np.uint32)), id="key-shape"),
        pytest.param(lambda: _numpy_state(key=np.array([None] * 624, dtype=object)), id="key-object-dtype"),
        pytest.param(lambda: _numpy_state(pos=625), id="pos"),
        pytest.param(lambda: _numpy_state(has_gauss=2), id="has-gauss"),
        pytest.param(lambda: _numpy_state(gauss="0.5"), id="gauss-type"),
        pytest.param(lambda: _numpy_state()[:4], id="tuple-length"),
        pytest.param(lambda: list(_numpy_state()), id="not-tuple"),
    ],
)
def test_malformed_numpy_state_is_rejected(state: Any) -> None:
    packed = pickle.dumps(state())
    with pytest.raises(ValueError):
        unpack_numpy_state(len(packed), packed)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "state",
    [
        pytest.param(lambda: (2, *random.getstate()[1:]), id="version"),
        pytest.param(lambda: (3, random.getstate()[1][:-1], None), id="internal-length"),
        pytest.param(lambda: (3, (2**32,) + random.getstate()[1][1:], None), id="word-range"),
        pytest.param(lambda: (3, random.getstate()[1][:-1] + (625,), None), id="pos"),
        pytest.param(lambda: (3, random.getstate()[1], "x"), id="gauss-type"),
        pytest.param(lambda: random.getstate()[:2], id="tuple-length"),
    ],
)
def test_malformed_random_state_is_rejected(state: Any) -> None:
    packed = pickle.dumps(state())
    with pytest.raises(ValueError):
        unpack_random_state(len(packed), packed)


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("unpack", [unpack_numpy_state, unpack_random_state])
def test_packed_length_must_fit_buffer(unpack: Any) -> None:
    buffer = bytes(pad_packed_bytes(pickle.dumps(random.getstate())))
    for packed_len in (0, -1, MAX_PACKED_LEN + 1, 2.0):
        with pytest.raises(ValueError, match="Packed random state length"):
            unpack(packed_len, buffer)
    with pytest.raises(ValueError, match="Packed random state length"):
        unpack(64, buffer[:32])


@pytest.mark.L0
@pytest.mark.CPU
def test_truncated_packed_state_is_rejected() -> None:
    packed = pickle.dumps(np.random.get_state())
    with pytest.raises((pickle.UnpicklingError, EOFError)):
        unpack_numpy_state(len(packed) - 10, packed)
