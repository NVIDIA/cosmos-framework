# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The vLLM-Omni seams the weight plane depends on, pinned to the revision the project installs.

The second hop rides engine internals that are not a public contract: how `collective_rpc` replies, how the
worker delegates `load_weights`, which fields `DiffusionOutput` carries. A different vLLM-Omni revision may move
any of them and pass every other test while the plane silently misbehaves, so this file fails first. L1: it
imports vLLM-Omni.
"""

from __future__ import annotations

import inspect
import json
from importlib import metadata

import pytest

PINNED_COMMIT = "2d0b1d8a"  # the operator runtime contract: vllm-omni@2d0b1d8a


@pytest.mark.L1
@pytest.mark.CPU
def test_installed_vllm_omni_is_the_pinned_revision() -> None:
    dist = metadata.distribution("vllm_omni")
    direct_url = dist.read_text("direct_url.json")
    assert direct_url is not None, (
        "vllm_omni is not a VCS install; the weight plane is qualified against a pinned commit"
    )
    commit = json.loads(direct_url)["vcs_info"]["commit_id"]
    assert commit.startswith(PINNED_COMMIT), f"installed vllm_omni commit {commit} != pinned {PINNED_COMMIT}"
    assert PINNED_COMMIT in dist.version, dist.version


@pytest.mark.L1
@pytest.mark.CPU
def test_collective_rpc_forwards_args_and_timeout() -> None:
    """The body sends the IPC descriptor as `args` and bounds the call with `timeout`."""
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    params = inspect.signature(AsyncOmniEngine.collective_rpc).parameters
    assert {"method", "timeout", "args", "kwargs"} <= set(params)


@pytest.mark.L1
@pytest.mark.CPU
def test_worker_delegates_load_weights_to_the_pipeline_and_forwards_extension_args() -> None:
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
    from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker, WorkerWrapperBase

    assert callable(getattr(DiffusionWorker, "load_weights", None))
    assert callable(getattr(DiffusionModelRunner, "load_weights", None))
    params = inspect.signature(WorkerWrapperBase.execute_method).parameters
    assert any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()), "extension methods take args"
    # The weight plane's RPC names (`export_live_parameters`, `begin_weight_update`, `finish_weight_update`) are
    # registered nowhere: they reach the extension only because the wrapper resolves a method name by attribute
    # lookup on the composed worker.
    assert "getattr(self.worker, method)" in inspect.getsource(WorkerWrapperBase.execute_method)
    assert "worker_extension_cls" in inspect.signature(WorkerWrapperBase.__init__).parameters or hasattr(
        WorkerWrapperBase, "_prepare_worker_class"
    )


@pytest.mark.L1
@pytest.mark.CPU
def test_stage_pool_reports_worker_failures_as_a_value_the_body_recognizes() -> None:
    """`_worker_rpc` treats `{"supported": False, "error": ...}` as a failure; that shape must still be what the
    stage pool emits, or a worker exception would come back as a successful-looking mapping."""
    from vllm_omni.engine import stage_pool

    source = inspect.getsource(stage_pool.StagePool.collective_rpc)
    assert '"supported": False' in source and '"error"' in source


@pytest.mark.L1
@pytest.mark.CPU
def test_diffusion_output_carries_the_stage_4_channels() -> None:
    """Chains and log-probabilities will ride these fields; the plane's contracts must not block them."""
    import dataclasses

    from vllm_omni.diffusion.data import DiffusionOutput

    fields = {f.name for f in dataclasses.fields(DiffusionOutput)}
    assert {"trajectory_latents", "trajectory_log_probs", "trajectory_timesteps", "custom_output"} <= fields


@pytest.mark.L1
@pytest.mark.CPU
def test_executor_selection_accepts_a_custom_class() -> None:
    """The in-process fallback the plan names if the IPC hop ever proves too slow."""
    from vllm_omni.diffusion.executor.abstract import DiffusionExecutor

    source = inspect.getsource(DiffusionExecutor.get_class)
    assert "resolve_obj_by_qualname" in source and "issubclass" in source


@pytest.mark.L1
@pytest.mark.CPU
def test_isaac_auto_classes_pins_the_shared_key_and_hands_it_back() -> None:
    """Both packages register `model_type="Gr00tN1d7"`; the last import wins. Inside the block the key must resolve to
    Isaac's config class (what the reference and the exporter build through), and the port's must be back afterwards."""
    import gr00t.model.gr00t_n1d7.gr00t_n1d7 as isaac
    import gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 as isaac_processing
    import vllm_omni.diffusion.models.gr00t.modeling.gr00t_n1d7 as omni
    import vllm_omni.diffusion.models.gr00t.modeling.processing_gr00t_n1d7 as omni_processing
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from transformers.models.auto.processing_auto import PROCESSOR_MAPPING, processor_class_from_name

    from cosmos_framework.posttrain.tests.helpers.gr00t_parity_gate import isaac_auto_classes, port_auto_classes

    before_config = CONFIG_MAPPING._extra_content.get("Gr00tN1d7")
    before_processor = PROCESSOR_MAPPING._extra_content.get("Gr00tN1d7")
    try:
        # The state after the serving side imported last: both keys point at the port.
        AutoConfig.register("Gr00tN1d7", omni.Gr00tN1d7Config, exist_ok=True)
        AutoProcessor.register("Gr00tN1d7", omni_processing.Gr00tN1d7Processor, exist_ok=True)
        with isaac_auto_classes():
            assert CONFIG_MAPPING["Gr00tN1d7"] is isaac.Gr00tN1d7Config
            assert processor_class_from_name("Gr00tN1d7Processor") is isaac_processing.Gr00tN1d7Processor
        assert CONFIG_MAPPING["Gr00tN1d7"] is omni.Gr00tN1d7Config
        assert processor_class_from_name("Gr00tN1d7Processor") is omni_processing.Gr00tN1d7Processor
        # And the mirror: after Isaac imported last (an engine body built first, then a reference), the port pins its own.
        AutoConfig.register("Gr00tN1d7", isaac.Gr00tN1d7Config, exist_ok=True)
        AutoProcessor.register("Gr00tN1d7", isaac_processing.Gr00tN1d7Processor, exist_ok=True)
        with port_auto_classes():
            assert CONFIG_MAPPING["Gr00tN1d7"] is omni.Gr00tN1d7Config
            assert processor_class_from_name("Gr00tN1d7Processor") is omni_processing.Gr00tN1d7Processor
        assert CONFIG_MAPPING["Gr00tN1d7"] is isaac.Gr00tN1d7Config
        assert processor_class_from_name("Gr00tN1d7Processor") is isaac_processing.Gr00tN1d7Processor
    finally:
        if before_config is not None:
            AutoConfig.register("Gr00tN1d7", before_config, exist_ok=True)
        if before_processor is not None:
            AutoProcessor.register("Gr00tN1d7", before_processor, exist_ok=True)
