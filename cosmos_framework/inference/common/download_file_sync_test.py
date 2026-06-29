# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Hermetic tests for distributed download synchronization in common.args."""

from __future__ import annotations

import importlib
import sys
import types
from enum import Enum
from pathlib import Path

import pytest

pytestmark = [pytest.mark.level(0), pytest.mark.gpus(0)]


@pytest.fixture
def common_args_module(monkeypatch: pytest.MonkeyPatch):
    module_name = "cosmos_framework.inference.common.args"

    fake_omegaconf = types.ModuleType("omegaconf")

    class _OmegaConf:
        @staticmethod
        def from_dotlist(values):
            return values

        @staticmethod
        def merge(*values):
            return values[-1] if values else {}

    fake_omegaconf.DictConfig = dict
    fake_omegaconf.OmegaConf = _OmegaConf

    fake_pydantic = types.ModuleType("pydantic")

    class _BaseModel:
        model_config = None

        def model_dump(self, **kwargs):
            return {}

    fake_pydantic.BaseModel = _BaseModel
    fake_pydantic.ValidationError = type("ValidationError", (Exception,), {})
    fake_pydantic.FilePath = str
    fake_pydantic.DirectoryPath = str
    fake_pydantic.PositiveInt = int
    fake_pydantic.NonNegativeInt = int
    fake_pydantic.NonNegativeFloat = float
    fake_pydantic.AfterValidator = lambda fn: fn
    fake_pydantic.ConfigDict = lambda **kwargs: kwargs
    fake_pydantic.model_validator = lambda *args, **kwargs: (lambda fn: fn)

    def _field(default=None, *, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    fake_pydantic.Field = _field

    fake_tyro = types.ModuleType("tyro")
    fake_tyro_conf = types.ModuleType("tyro.conf")

    class _Suppress:
        def __class_getitem__(cls, item):
            return item

    fake_tyro.cli = lambda cls, **kwargs: None
    fake_tyro_conf.Suppress = _Suppress
    fake_tyro_conf.arg = lambda *args, **kwargs: object()
    fake_tyro.conf = fake_tyro_conf

    fake_yaml = types.ModuleType("yaml")
    fake_yaml.safe_load = lambda text: {}

    fake_typing_extensions = types.ModuleType("typing_extensions")
    fake_typing_extensions.Self = object
    fake_typing_extensions.assert_never = lambda value: (_ for _ in ()).throw(
        AssertionError(f"Unexpected value: {value!r}")
    )

    fake_common_config = types.ModuleType("cosmos_framework.inference.common.config")
    fake_common_config.deserialize_config_dict = lambda value: value
    fake_common_config.structure_config = lambda value, target: value
    fake_common_config.unstructure_config = lambda value: value

    fake_common_init = types.ModuleType("cosmos_framework.inference.common.init")
    fake_common_init.is_rank0 = lambda: True

    fake_public_model_config = types.ModuleType("cosmos_framework.inference.common.public_model_config")
    fake_public_model_config.load_model_config_from_hf_config = lambda config_dict: {}

    fake_checkpoint_db = types.ModuleType("cosmos_framework.utils.checkpoint_db")
    fake_checkpoint_db.CheckpointDirHf = type("CheckpointDirHf", (), {})

    fake_utils_config = types.ModuleType("cosmos_framework.utils.config")
    fake_utils_config.Config = type("Config", (), {})

    fake_flags = types.ModuleType("cosmos_framework.utils.flags")

    class _StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value

        @staticmethod
        def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
            return name.lower()

    fake_flags.TRAINING = False
    fake_flags.StrEnum = _StrEnum

    dist_state = types.SimpleNamespace(available=False, initialized=False, calls=[], handler=lambda obj_list, src: None)
    fake_torch = types.ModuleType("torch")
    fake_dist = types.ModuleType("torch.distributed")
    fake_dist.is_available = lambda: dist_state.available
    fake_dist.is_initialized = lambda: dist_state.initialized

    def _broadcast_object_list(obj_list, src=0):
        dist_state.calls.append((list(obj_list), src))
        dist_state.handler(obj_list, src)

    fake_dist.broadcast_object_list = _broadcast_object_list
    fake_torch.distributed = fake_dist

    monkeypatch.setitem(sys.modules, "omegaconf", fake_omegaconf)
    monkeypatch.setitem(sys.modules, "pydantic", fake_pydantic)
    monkeypatch.setitem(sys.modules, "tyro", fake_tyro)
    monkeypatch.setitem(sys.modules, "tyro.conf", fake_tyro_conf)
    monkeypatch.setitem(sys.modules, "yaml", fake_yaml)
    monkeypatch.setitem(sys.modules, "typing_extensions", fake_typing_extensions)
    monkeypatch.setitem(sys.modules, "cosmos_framework.inference.common.config", fake_common_config)
    monkeypatch.setitem(sys.modules, "cosmos_framework.inference.common.init", fake_common_init)
    monkeypatch.setitem(sys.modules, "cosmos_framework.inference.common.public_model_config", fake_public_model_config)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.checkpoint_db", fake_checkpoint_db)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.config", fake_utils_config)
    monkeypatch.setitem(sys.modules, "cosmos_framework.utils.flags", fake_flags)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", fake_dist)
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)
    yield module, dist_state

    sys.modules.pop(module_name, None)


def test_download_file_waits_for_distributed_sync_before_returning(
    common_args_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, dist_state = common_args_module
    monkeypatch.setattr(module, "is_rank0", lambda: False)
    monkeypatch.setattr(
        module, "_download_file", lambda url, path: pytest.fail("non-rank0 should not download directly")
    )

    dist_state.available = True
    dist_state.initialized = True
    download_path = tmp_path / "vision.jpg"

    def _create_file_during_sync(obj_list, src):
        assert obj_list == [None]
        download_path.write_text("ready", encoding="utf-8")

    dist_state.handler = _create_file_during_sync

    resolved = module.download_file("https://example.com/vision.jpg", tmp_path, "vision")

    assert resolved == str(download_path)
    assert download_path.read_text(encoding="utf-8") == "ready"
    assert dist_state.calls == [([None], 0)]


def test_download_file_raises_on_other_ranks_when_rank0_download_fails(
    common_args_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, dist_state = common_args_module
    monkeypatch.setattr(module, "is_rank0", lambda: False)
    monkeypatch.setattr(
        module, "_download_file", lambda url, path: pytest.fail("non-rank0 should not download directly")
    )

    dist_state.available = True
    dist_state.initialized = True
    dist_state.handler = lambda obj_list, src: obj_list.__setitem__(0, "RuntimeError: boom")

    with pytest.raises(
        RuntimeError, match=r"Distributed download failed for https://example.com/vision.jpg: RuntimeError: boom"
    ):
        module.download_file("https://example.com/vision.jpg", tmp_path, "vision")


def test_download_file_raises_when_synced_file_is_still_missing(
    common_args_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module, dist_state = common_args_module
    monkeypatch.setattr(module, "is_rank0", lambda: False)
    monkeypatch.setattr(
        module, "_download_file", lambda url, path: pytest.fail("non-rank0 should not download directly")
    )

    dist_state.available = True
    dist_state.initialized = True
    dist_state.handler = lambda obj_list, src: None

    with pytest.raises(
        FileNotFoundError, match=r"Expected downloaded file to exist after synchronization: .*vision.jpg"
    ):
        module.download_file("https://example.com/vision.jpg", tmp_path, "vision")
