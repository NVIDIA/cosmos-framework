# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Device namespace and import-lifecycle correctness, without a renderer or CUDA."""

import os
import sys

import pytest

from cosmos_framework.posttrain.embodied.sims.libero import renderer

pytestmark = [pytest.mark.L1, pytest.mark.CPU]
GPU = "GPU-11111111-1111-1111-1111-111111111111"
OTHER = "GPU-22222222-2222-2222-2222-222222222222"


@pytest.fixture
def mapping(monkeypatch):
    monkeypatch.setattr(renderer, "_binding", None)
    monkeypatch.setenv(renderer.PROFILE_ENV, renderer.UUID_PROFILE)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU)
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "4")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setattr(renderer.importlib.metadata, "version", lambda name: "1.4.1")
    monkeypatch.setattr(renderer.subprocess, "check_output", lambda *a, **k: f"0, {GPU}\n1, {OTHER}\n")
    monkeypatch.setattr(
        renderer,
        "_egl_devices",
        lambda: [
            {"egl_index": 0, "gpu_uuid": OTHER, "driver": "nvidia"},
            {"egl_index": 4, "gpu_uuid": GPU, "driver": "nvidia"},
        ],
    )
    for name in ["robosuite.utils.binding_utils", "robosuite.renderers.context.egl_context"]:
        monkeypatch.delitem(sys.modules, name, raising=False)


def test_resolve_disjoint_ordinals_and_uuid(mapping):
    first = renderer.resolve_renderer_binding()
    assert first.nvml_index == 0 and first.egl_index == 4 and first.gpu_uuid == GPU


@pytest.mark.parametrize("visible", ["", "0", "0,1", " 0", "GPU-1111", "MIG-any", "6"])
def test_invalid_visible_device_rejected(mapping, monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError):
        renderer.resolve_renderer_binding()


def test_wrong_egl_device_and_backend_rejected(mapping, monkeypatch):
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    with pytest.raises(ValueError, match="differs"):
        renderer.resolve_renderer_binding()
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "4")
    monkeypatch.setenv("MUJOCO_GL", "osmesa")
    with pytest.raises(ValueError, match="EGL renderer"):
        renderer.resolve_renderer_binding()


def test_missing_duplicated_or_wrong_driver_identity(mapping, monkeypatch):
    for rows in [
        [],
        [{"egl_index": 4, "gpu_uuid": GPU, "driver": "mesa"}],
        [{"egl_index": i, "gpu_uuid": GPU, "driver": "nvidia"} for i in [0, 4]],
    ]:
        monkeypatch.setattr(renderer, "_egl_devices", lambda: rows)
        with pytest.raises(RuntimeError):
            renderer.resolve_renderer_binding()


def test_import_binding_restores_validated_index_without_cuda_mutation(mapping):
    with renderer.vendor_renderer_import():
        assert "MUJOCO_EGL_DEVICE_ID" not in os.environ
        assert os.environ["CUDA_VISIBLE_DEVICES"] == GPU
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "4"
    assert renderer.renderer_identity()["gpu_uuid"] == GPU
    with renderer.vendor_renderer_import():
        assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "4"


@pytest.mark.parametrize("requested", [None, "4"])
def test_exception_restores_preimport_state(mapping, monkeypatch, requested):
    if requested is None:
        monkeypatch.delenv("MUJOCO_EGL_DEVICE_ID")
    with pytest.raises(ImportError):
        with renderer.vendor_renderer_import():
            raise ImportError("controlled vendor failure")
    assert os.environ.get("MUJOCO_EGL_DEVICE_ID") == requested
    assert renderer.renderer_identity() == {"profile": "legacy"}
    assert os.environ["CUDA_VISIBLE_DEVICES"] == GPU


@pytest.mark.parametrize("module", ["robosuite.utils.binding_utils", "robosuite.renderers.context.egl_context"])
def test_existing_vendor_refuses_late_binding(mapping, monkeypatch, module):
    monkeypatch.setitem(sys.modules, module, object())
    with pytest.raises(RuntimeError, match="before importing"):
        with renderer.vendor_renderer_import():
            pytest.fail("should not import")
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "4"


def test_reentry_profile_and_device_conflicts(mapping, monkeypatch):
    with renderer.vendor_renderer_import():
        pass
    monkeypatch.setenv(renderer.PROFILE_ENV, "legacy")
    with pytest.raises(RuntimeError, match="profile changed"):
        with renderer.vendor_renderer_import():
            pass
    monkeypatch.setenv(renderer.PROFILE_ENV, renderer.UUID_PROFILE)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", OTHER)
    monkeypatch.setenv("MUJOCO_EGL_DEVICE_ID", "0")
    with pytest.raises(RuntimeError, match="device changed"):
        with renderer.vendor_renderer_import():
            pass


def test_disabled_does_not_probe_devices(monkeypatch):
    monkeypatch.setattr(renderer, "_binding", None)
    monkeypatch.delenv(renderer.PROFILE_ENV, raising=False)
    monkeypatch.setattr(renderer, "resolve_renderer_binding", lambda: pytest.fail("legacy must not probe"))
    with renderer.vendor_renderer_import():
        pass
    assert renderer.renderer_identity() == {"profile": "legacy"}


def test_unknown_profile_or_vendor_version_rejected(mapping, monkeypatch):
    monkeypatch.setenv(renderer.PROFILE_ENV, "other")
    with pytest.raises(ValueError):
        with renderer.vendor_renderer_import():
            pass
    monkeypatch.setenv(renderer.PROFILE_ENV, renderer.UUID_PROFILE)
    monkeypatch.setattr(renderer.importlib.metadata, "version", lambda name: "1.5.0")
    with pytest.raises(RuntimeError, match="1.4.1"):
        with renderer.vendor_renderer_import():
            pass
