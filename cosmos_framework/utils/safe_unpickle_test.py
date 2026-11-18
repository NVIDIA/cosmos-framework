# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import builtins
import collections
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

import pytest

from cosmos_framework.utils.safe_unpickle import ForbiddenPickleGlobalError, safe_loads

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


class _CallOnLoad:
    """Pickles as a call to ``func(*args)``, which a plain ``pickle.loads`` would execute."""

    def __init__(self, func: Any, *args: Any) -> None:
        self.func = func
        self.args = args

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return self.func, self.args


def _write_marker_code(marker: Path) -> str:
    return f"open({str(marker)!r}, 'w').write('loaded')"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_builtin_data_round_trips_without_globals(protocol: int) -> None:
    state = {
        0: {"epoch": 3, "index": 17},
        1: {"epoch": 2, "index": -1},
        "val": {"format": "lance", "workers": {"0": {"current": {"draw_count": 5}}}, "batch_count": 9},
        "nested": [1, 2.5, (3, "four"), None, True, False],
    }

    assert safe_loads(pickle.dumps(state, protocol=protocol)) == state


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize(
    ("func", "forbidden"),
    [
        # Protocol 2 writes Python 2 module names (``__builtin__``, ``commands``).
        (builtins.exec, r"(builtins|__builtin__)\.exec"),
        (os.system, rf"{os.system.__module__}\.system"),
        (subprocess.Popen, r"(subprocess|commands)\.Popen"),
    ],
)
def test_callable_payload_is_rejected_without_running(func: Any, forbidden: str, protocol: int, tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    if func is builtins.exec:
        args: tuple[Any, ...] = (_write_marker_code(marker),)
    elif func is os.system:
        args = (f"touch {marker}",)
    else:
        args = (["touch", str(marker)],)
    payload = pickle.dumps(_CallOnLoad(func, *args), protocol=protocol)

    with pytest.raises(ForbiddenPickleGlobalError, match=f"forbidden global {forbidden}"):
        safe_loads(payload)

    assert not marker.exists()


@pytest.mark.L0
@pytest.mark.CPU
def test_payload_nested_inside_plain_data_is_rejected(tmp_path: Path) -> None:
    marker = tmp_path / "marker"
    payload = pickle.dumps({0: {"epoch": 1, "index": _CallOnLoad(builtins.exec, _write_marker_code(marker))}})

    with pytest.raises(ForbiddenPickleGlobalError):
        safe_loads(payload)

    assert not marker.exists()


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("value", [b"raw", {1, 2}, frozenset({1})])
def test_protocol_2_values_needing_globals_are_rejected(value: Any) -> None:
    # easy_io's pkl writer uses protocol 2, where bytes and sets are encoded as calls to globals;
    # state loaded with no allowed globals must therefore stay free of them.
    with pytest.raises(ForbiddenPickleGlobalError):
        safe_loads(pickle.dumps({"value": value}, protocol=2))


@pytest.mark.L0
@pytest.mark.CPU
def test_only_listed_globals_resolve() -> None:
    allowed = frozenset({("collections", "OrderedDict")})
    ordered = collections.OrderedDict(a=1)

    assert safe_loads(pickle.dumps(ordered), allowed) == ordered
    with pytest.raises(ForbiddenPickleGlobalError, match="forbidden global collections.defaultdict"):
        safe_loads(pickle.dumps(collections.defaultdict(int)), allowed)


@pytest.mark.L0
@pytest.mark.CPU
def test_malformed_pickle_raises_unpickling_error() -> None:
    with pytest.raises((pickle.UnpicklingError, EOFError)):
        safe_loads(pickle.dumps({"a": 1})[:-3])
