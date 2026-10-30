# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.inference.args import DEFAULT_CHECKPOINT_NAME, OmniSetupOverrides
from cosmos_framework.inference.inference import OmniInference

pytestmark = [pytest.mark.L0, pytest.mark.CPU]


def _setup(tmp_path, **kwargs):
    return OmniSetupOverrides(
        checkpoint_path=DEFAULT_CHECKPOINT_NAME,
        output_dir=tmp_path / "outputs",
        **kwargs,
    ).build_setup(world_size=1, local_world_size=1, device_memory_bytes=192 * 1024**3)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("flag", ["enable_cpu_offload", "enable_layerwise_offload"])
def test_offload_args_round_trip(tmp_path, enabled, flag):
    setup = _setup(tmp_path, **{flag: enabled}, prefill_text_kv=enabled)
    restored = OmniSetupOverrides.model_validate(setup.model_dump())
    assert getattr(restored, flag) is enabled
    assert restored.prefill_text_kv is enabled


@pytest.mark.parametrize("training_experiment", [False, True])
@pytest.mark.parametrize("flag", [None, "enable_cpu_offload", "enable_layerwise_offload"])
def test_cpu_initialization_is_passed_explicitly_to_both_loaders(tmp_path, monkeypatch, training_experiment, flag):
    from types import SimpleNamespace

    from cosmos_framework.inference import inference as inference_module
    from cosmos_framework.utils.generator import model_loader

    setup = _setup(
        tmp_path,
        **({flag: True} if flag else {}),
        use_cuda_graphs=False,
        config_file=__file__,
        experiment="offload-test",
    )
    captured = []

    def load(*args, parallelism_config, **kwargs):
        value = (
            parallelism_config["inference_cpu_initialization"]
            if training_experiment
            else parallelism_config.inference_cpu_initialization
        )
        captured.append(value)
        raise RuntimeError("stop before checkpoint allocation")

    monkeypatch.setattr(inference_module, "_loads_training_experiment", lambda _: training_experiment)
    monkeypatch.setattr(inference_module, "_download_on_rank0", lambda _: tmp_path)
    monkeypatch.setattr(inference_module.Cosmos3OmniModel, "before_load_model", lambda: None)
    monkeypatch.setattr(inference_module.Cosmos3OmniConfig, "from_pretrained", lambda _: SimpleNamespace(model={}))
    monkeypatch.setattr(inference_module.Cosmos3OmniModel, "from_pretrained_dcp", load)
    monkeypatch.setattr(model_loader, "load_model_from_checkpoint", load)
    with pytest.raises(RuntimeError, match="stop before checkpoint allocation"):
        OmniInference._create(setup)
    assert captured == [flag is not None]


@pytest.mark.parametrize("flag", ["enable_cpu_offload", "enable_layerwise_offload", "prefill_text_kv"])
def test_offload_rejects_cuda_graphs_before_loading(tmp_path, flag):
    setup = _setup(tmp_path, **{flag: True}, use_cuda_graphs=True)
    with pytest.raises(ValueError, match="no-use-cuda-graphs"):
        OmniInference._create(setup)


@pytest.mark.parametrize("flag", ["enable_cpu_offload", "enable_layerwise_offload", "prefill_text_kv"])
def test_offload_rejects_multiple_ranks_before_loading(tmp_path, monkeypatch, flag):
    import cosmos_framework.inference.inference as inference_module

    setup = _setup(tmp_path, **{flag: True}, use_cuda_graphs=False)
    monkeypatch.setattr(inference_module, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="single-GPU"):
        OmniInference._create(setup)


@pytest.mark.parametrize("flag", ["enable_cpu_offload", "enable_layerwise_offload"])
def test_offload_rejects_runtime_quantization_before_loading(tmp_path, flag):
    setup = _setup(tmp_path, **{flag: True}, use_cuda_graphs=False, quantization_method="fp8")
    with pytest.raises(ValueError, match="not runtime quantization"):
        OmniInference._create(setup)


@pytest.mark.parametrize("flag", ["enable_cpu_offload", "enable_layerwise_offload"])
@pytest.mark.parametrize("cache", ["generation", "all", "cpu_block", "gpu_block"])
def test_offload_rejects_dense_weight_cache_before_loading(tmp_path, flag, cache):
    setup = _setup(tmp_path, **{flag: True}, use_cuda_graphs=False, mixed_precision_w8a16_cache=cache)
    with pytest.raises(ValueError, match="mixed-precision-w8a16-cache none"):
        OmniInference._create(setup)


def test_cpu_materialization_preserves_initialized_buffers():
    import torch
    from torch import nn

    from cosmos_framework.model.generator.omni_mot_model import _materialize_inference_net_on_cpu

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(4, 4, device="meta", dtype=torch.bfloat16))
            self.register_buffer("frequencies", torch.empty(4, device="meta"), persistent=False)

        def init_weights(self, buffer_device):
            assert self.weight.is_meta, "Do not initialize checkpoint weights on CPU or GPU"
            self.frequencies = torch.arange(4, device=buffer_device, dtype=torch.float32)

    net = Net()
    _materialize_inference_net_on_cpu(net)
    assert net.weight.device.type == "cpu"
    assert net.weight.dtype == torch.bfloat16
    assert net.frequencies.device.type == "cpu"
    assert net.frequencies.dtype == torch.float32
    torch.testing.assert_close(net.frequencies, torch.arange(4, dtype=torch.float32), rtol=0, atol=0)
    assert tuple(net.state_dict()) == ("weight",)


def test_offload_modes_are_mutually_exclusive(tmp_path):
    setup = _setup(tmp_path, enable_cpu_offload=True, enable_layerwise_offload=True, use_cuda_graphs=False)
    with pytest.raises(ValueError, match="not both"):
        OmniInference._create(setup)


@pytest.mark.parametrize("placement", ["resident", "inference", "fsdp"])
@pytest.mark.parametrize("skip_keys", [False, True])
def test_experiment_loader_preserves_cpu_initialization(monkeypatch, placement, skip_keys):
    import sys
    from types import ModuleType, SimpleNamespace

    import torch

    from cosmos_framework.utils.generator import model_loader

    config = SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(
                parallelism=SimpleNamespace(
                    inference_cpu_initialization=placement == "inference",
                    fsdp_cpu_offload=placement == "fsdp",
                ),
                activation_checkpointing=SimpleNamespace(mode="full"),
                ema=SimpleNamespace(enabled=True),
                exclude_reasoner_weights_from_checkpoint=False,
            )
        ),
        trainer=SimpleNamespace(
            cudnn=SimpleNamespace(
                deterministic=torch.backends.cudnn.deterministic,
                benchmark=torch.backends.cudnn.benchmark,
            )
        ),
        validate=lambda: None,
        freeze=lambda: None,
    )
    module = ModuleType("offload_loader_test_config")
    module.make_config = lambda: config
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(model_loader, "get_config_module", lambda _: module.__name__)
    monkeypatch.setattr(model_loader, "override", lambda value, _: value)
    monkeypatch.setattr(model_loader, "_is_safetensors_checkpoint", lambda *_: False)
    monkeypatch.setattr(model_loader.misc, "set_random_seed", lambda **_: None)
    model = torch.nn.Linear(2, 2)
    cuda_calls = []
    monkeypatch.setattr(model, "cuda", lambda: cuda_calls.append(True) or model)
    monkeypatch.setattr(model, "on_train_start", lambda: None, raising=False)
    monkeypatch.setattr(model_loader, "instantiate", lambda _: model)
    loaded = []
    monkeypatch.setattr(model_loader, "_load_model", lambda value, **_: loaded.append(value))
    monkeypatch.setattr(model_loader, "_reload_pretrained_reasoner_after_checkpoint_load", lambda _: None)
    kwargs = dict(
        experiment_name="test", checkpoint_path="checkpoint", keys_to_skip_loading=["head"] if skip_keys else None
    )
    if skip_keys and placement != "resident":
        with pytest.raises(ValueError, match="initialize every parameter"):
            model_loader.load_model_from_checkpoint(**kwargs)
        assert not loaded and not cuda_calls
    else:
        actual, actual_config = model_loader.load_model_from_checkpoint(**kwargs)
        assert actual is model and actual_config is config
        assert loaded == [model]
        assert cuda_calls == ([True] if placement == "resident" else [])
