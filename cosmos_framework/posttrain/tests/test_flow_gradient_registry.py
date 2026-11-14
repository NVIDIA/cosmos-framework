# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Model-owned gradient factories stay lazy and reject undeclared builders."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.model import sft_source
from cosmos_framework.posttrain.model.gr00t import sft as gr00t
from cosmos_framework.posttrain.model.pi05 import sft as pi05

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


@pytest.mark.parametrize("module", [pi05, gr00t])
def test_native_family_declares_its_own_post_optimizer_factory(module):
    factory = sft_source.flow_sft_gradient_window_factory(SimpleNamespace(builder=module.BUILDER))
    assert factory is module.build_gradient_window


def test_gr00t_factory_resolution_does_not_import_isaac_model():
    code = """
from types import SimpleNamespace
from cosmos_framework.posttrain.model.sft_source import flow_sft_gradient_window_factory
import sys
factory = flow_sft_gradient_window_factory(SimpleNamespace(
    builder='cosmos_framework.posttrain.model.gr00t.sft.build_gr00t_sft_policy'))
assert callable(factory)
assert 'gr00t' not in sys.modules
assert 'cosmos_framework.posttrain.model.gr00t.sft_policy' not in sys.modules
"""
    subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize(
    "builder",
    [None, "missing_module.make", "cosmos_framework.posttrain.model.gr00t.sft.build_policy"],
)
def test_undeclared_or_unavailable_factory_cannot_install_hooks(builder):
    with pytest.raises(ValueError, match="gradient-window factory"):
        sft_source.flow_sft_gradient_window_factory(None if builder is None else SimpleNamespace(builder=builder))


def test_duplicate_family_registration_is_not_silently_replaced():
    original = sft_source.flow_sft_gradient_window_factory(SimpleNamespace(builder=gr00t.BUILDER))
    with pytest.raises(ValueError, match="already registered"):
        sft_source.register_flow_sft_gradient_window(gr00t.BUILDER)(lambda *args: None)
    assert sft_source.flow_sft_gradient_window_factory(SimpleNamespace(builder=gr00t.BUILDER)) is original


@pytest.mark.parametrize("module", [pi05, gr00t])
@pytest.mark.parametrize("topology", [(1, 4), (2, 2), (1, 2)])
def test_public_structure_admits_only_declared_family_and_topology(monkeypatch, module, topology):
    from cosmos_framework.posttrain.config import build
    from cosmos_framework.posttrain.config.sft import SFTDatasetConfig
    from cosmos_framework.posttrain.entrypoints import sft
    from cosmos_framework.posttrain.tests.test_gradient_window_trainer import config
    from cosmos_framework.utils import lazy_config

    rl = config(data=SFTDatasetConfig(path="frozen-demo.jsonl"))
    root = SimpleNamespace(grid=SimpleNamespace(backend="local", grouping="single"), rl=rl)
    monkeypatch.setattr(sft, "sft_policy_source", lambda _: ("fixture", False))
    monkeypatch.setattr(lazy_config, "instantiate", lambda value: value)
    monkeypatch.setattr(
        build,
        "select_policy_backend",
        lambda _: SimpleNamespace(native_spec=SimpleNamespace(builder=module.BUILDER)),
    )
    monkeypatch.setattr(build, "resolve_policy_parallelism", lambda *_: topology)
    monkeypatch.setattr(sft_source, "native_sft_task", lambda _: "flow_bc")
    if topology == (1, 2):
        with pytest.raises(ValueError, match="FSDP4 or HSDP2x2"):
            sft.validate_config_structure(root)
    else:
        assert sft.validate_config_structure(root) is rl
