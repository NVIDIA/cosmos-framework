# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

from types import SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.utils.callback import Callback, CallBackGroup


def _foreign_callback_type() -> type[Any]:
    hooks = {name: member for name, member in vars(Callback).items() if name.startswith("on_") and callable(member)}
    return type("ForeignCallback", (), hooks)


@pytest.mark.L0
@pytest.mark.CPU
def test_callback_group_accepts_structurally_compatible_foreign_callback() -> None:
    foreign_callback = _foreign_callback_type()
    config = SimpleNamespace(trainer=SimpleNamespace(callbacks={"foreign": {"_target_": foreign_callback}}))
    trainer = SimpleNamespace()

    group = CallBackGroup(config=config, trainer=trainer)

    assert len(group._callbacks) == 1
    assert group._callbacks[0].config is config
    assert group._callbacks[0].trainer is trainer


@pytest.mark.L0
@pytest.mark.CPU
def test_callback_group_rejects_foreign_callback_missing_required_hooks() -> None:
    class IncompleteCallback:
        def on_train_start(self) -> None:
            pass

    config = SimpleNamespace(trainer=SimpleNamespace(callbacks={"incomplete": {"_target_": IncompleteCallback}}))

    with pytest.raises(TypeError, match="missing required callback hooks"):
        CallBackGroup(config=config, trainer=SimpleNamespace())


@pytest.mark.L0
@pytest.mark.CPU
def test_callback_group_rejects_foreign_callback_without_metadata_slots() -> None:
    hooks = {name: member for name, member in vars(Callback).items() if name.startswith("on_") and callable(member)}
    foreign_callback = type("SlottedForeignCallback", (), {"__slots__": (), **hooks})
    callback_config = {"_target_": foreign_callback}
    config = SimpleNamespace(trainer=SimpleNamespace(callbacks={"foreign": callback_config}))

    with pytest.raises(TypeError, match="required callback metadata") as error:
        CallBackGroup(config=config, trainer=SimpleNamespace())

    assert repr(callback_config) in str(error.value)
