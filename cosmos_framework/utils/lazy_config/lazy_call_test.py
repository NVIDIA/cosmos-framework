# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.utils.lazy_config import instantiate, lazy_call
from cosmos_framework.utils.lazy_config.lazy_call import LazyCall


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("convert_target_to_string", [False, True])
def test_string_target_is_kept_as_is(monkeypatch: pytest.MonkeyPatch, convert_target_to_string: bool) -> None:
    monkeypatch.setattr(lazy_call, "_CONVERT_TARGET_TO_STRING", convert_target_to_string)
    node = LazyCall("collections.OrderedDict")(a=1)
    assert node._target_ == "collections.OrderedDict"
    assert dict(instantiate(node)) == {"a": 1}


class _Target:
    def __init__(self, a: int, b: int = 2) -> None:
        self.a, self.b = a, b


@pytest.mark.L0
@pytest.mark.CPU
def test_callable_target_is_converted_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lazy_call, "_CONVERT_TARGET_TO_STRING", True)
    node = LazyCall(_Target)(a=1)
    assert node._target_ == f"{__name__}._Target"
    assert node.b == 2
