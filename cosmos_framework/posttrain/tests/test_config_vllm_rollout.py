# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""RolloutConfig → vLLM engine kwargs / sampling mapping. No vLLM import — the mapping is a pure
function precisely so it is CPU-testable. (The GRPO entry's config plumbing moved to the shared config
system — `config_grpo.py`'s root carries `rollout: RolloutConfig` and the `cli` dotlist reaches it, and
the config schema itself is covered by tests/test_config_rl.py. The trainer↔engine same-weights default
now lives in `grpo.build_trainer`, pinned by `test_grpo_trainer_checkpoint_defaults_to_rollout` /
`_explicit_wins` in tests/test_entrypoints_grpo.py.)"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from cosmos_framework.posttrain.config.build import (
    _configure_vllm_runtime,
    hf_sampling_params,
    ppo_vllm_engine_kwargs,
    sampling_params,
    vllm_engine_kwargs,
)
from cosmos_framework.posttrain.config.rl import RolloutConfig


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_single_gpu_defaults() -> None:
    cfg = RolloutConfig()
    kw = vllm_engine_kwargs(cfg)
    assert kw["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert kw["gpu_memory_utilization"] == 0.3  # colocated default (engine shares the GPU)
    assert kw["max_model_len"] == 1024 and kw["dtype"] == "bfloat16" and kw["enforce_eager"] is True
    assert cfg.sleep_level == 1
    assert kw["runner"] == "generate" and kw["convert"] == "none"
    # stale-KV guard: prefix caching must be OFF by default (weight sync never invalidates the cache)
    assert kw["enable_prefix_caching"] is False
    assert kw["enable_expert_parallel"] is False
    assert kw["moe_backend"] == "auto"
    assert "batch_invariant" not in kw
    assert "attention_config" not in kw
    # TP=1: NO distributed wiring keys (a bare LLM must not get an executor backend)
    for k in (
        "tensor_parallel_size",
        "distributed_executor_backend",
        "disable_custom_all_reduce",
        "load_format",
        "seed",
    ):
        assert k not in kw, f"unexpected key {k} for TP=1"


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (True, TypeError, "must be a non-bool integer"),
        (1.0, TypeError, "must be a non-bool integer"),
        (0, ValueError, "must be 1 or 2"),
        (3, ValueError, "must be 1 or 2"),
    ],
)
def test_rollout_config_rejects_invalid_sleep_level(
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        RolloutConfig(sleep_level=value)  # type: ignore[arg-type]


@pytest.mark.L0
@pytest.mark.CPU
def test_rollout_config_threads_only_an_explicit_dummy_load_format() -> None:
    assert "load_format" not in vllm_engine_kwargs(RolloutConfig())
    assert vllm_engine_kwargs(RolloutConfig(load_format="dummy"))["load_format"] == "dummy"
    with pytest.raises(ValueError, match=r"load_format must be auto\|dummy"):
        RolloutConfig(load_format="safetensors")


@pytest.mark.L0
@pytest.mark.CPU
def test_local_hf_config_resolution_retries_transient_os_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    import transformers

    import cosmos_framework.posttrain.config.build as build

    calls: list[tuple[str, bool]] = []
    sleeps: list[float] = []

    def resolve(name_or_path: str, *, trust_remote_code: bool) -> Any:
        calls.append((name_or_path, trust_remote_code))
        if len(calls) < 3:
            raise OSError("transient shared-filesystem visibility failure")
        return SimpleNamespace(model_type="fake")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", resolve)
    monkeypatch.setattr(build.time, "sleep", sleeps.append)

    result = build._resolve_hf_config_with_local_retry(str(tmp_path), trust_remote_code=True)

    assert result.model_type == "fake"
    assert calls == [(str(tmp_path), True)] * 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.L0
@pytest.mark.CPU
def test_remote_hf_config_resolution_does_not_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    import cosmos_framework.posttrain.config.build as build

    calls: list[str] = []
    sleeps: list[float] = []

    def resolve(name_or_path: str, *, trust_remote_code: bool) -> Any:
        del trust_remote_code
        calls.append(name_or_path)
        raise OSError("remote repository failure")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", resolve)
    monkeypatch.setattr(build.time, "sleep", sleeps.append)

    with pytest.raises(OSError, match="remote repository failure"):
        build._resolve_hf_config_with_local_retry("org/model", trust_remote_code=False)

    assert calls == ["org/model"]
    assert sleeps == []


@pytest.mark.L0
@pytest.mark.CPU
def test_local_hf_config_resolution_reraises_after_the_last_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import transformers

    import cosmos_framework.posttrain.config.build as build

    failure = OSError("persistent shared-filesystem visibility failure")
    calls: list[str] = []
    sleeps: list[float] = []

    def resolve(name_or_path: str, *, trust_remote_code: bool) -> Any:
        del trust_remote_code
        calls.append(name_or_path)
        raise failure

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", resolve)
    monkeypatch.setattr(build.time, "sleep", sleeps.append)

    with pytest.raises(OSError) as raised:
        build._resolve_hf_config_with_local_retry(str(tmp_path), trust_remote_code=False)

    assert raised.value is failure
    assert calls == [str(tmp_path)] * 6
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]


@pytest.mark.L0
@pytest.mark.CPU
def test_local_hf_config_resolution_does_not_retry_non_os_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import transformers

    import cosmos_framework.posttrain.config.build as build

    calls: list[str] = []
    sleeps: list[float] = []

    def resolve(name_or_path: str, *, trust_remote_code: bool) -> Any:
        del trust_remote_code
        calls.append(name_or_path)
        raise ValueError("unrecognized model config")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", resolve)
    monkeypatch.setattr(build.time, "sleep", sleeps.append)

    with pytest.raises(ValueError, match="unrecognized model config"):
        build._resolve_hf_config_with_local_retry(str(tmp_path), trust_remote_code=False)

    assert calls == [str(tmp_path)]
    assert sleeps == []


def _build_with_fake_vllm(monkeypatch: pytest.MonkeyPatch, cfg: RolloutConfig) -> Any:
    """Run `build_vllm_rollout` against a recording `vllm.LLM`; the engine kwargs land on `.model.kwargs`."""
    import sys
    from types import ModuleType, SimpleNamespace

    import transformers

    import cosmos_framework.posttrain.utils.rank_state as rank_state
    from cosmos_framework.posttrain.config.build import build_vllm_rollout

    class _FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    vllm = ModuleType("vllm")
    vllm.LLM = _FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(to_dict=lambda: {"model_type": "fake"}),
    )
    monkeypatch.setattr(rank_state, "gather_rank_states", lambda state, *, context: {0: state})
    return build_vllm_rollout(cfg, SimpleNamespace(adds_special_tokens=lambda: False))


@pytest.mark.L0
@pytest.mark.CPU
def test_build_vllm_rollout_threads_sleep_level(monkeypatch: pytest.MonkeyPatch) -> None:
    rollout = _build_with_fake_vllm(monkeypatch, RolloutConfig(sleep_level=2))

    assert rollout._sleep_level == 2
    assert rollout.model.kwargs["enable_sleep_mode"] is True


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("validation", [False, True], ids=["training", "validation"])
def test_dummy_loaded_rollout_refuses_to_generate_before_a_publication_commits(
    monkeypatch: pytest.MonkeyPatch,
    validation: bool,
) -> None:
    from cosmos_framework.posttrain.roles.rollout.validation import ValidationRolloutContext
    from cosmos_framework.posttrain.schema import ObsBundle

    rollout = _build_with_fake_vllm(monkeypatch, RolloutConfig(load_format="dummy"))
    prompts = [ObsBundle(representations={"prompt": "1 + 1 ="}, id="p0")]
    context = ValidationRolloutContext(model_step=0) if validation else None

    # The initial publication also commits version 0, so the version alone cannot prove publication.
    assert rollout.weight_version() == 0
    with pytest.raises(RuntimeError, match="load_format='dummy' and holds random weights"):
        rollout.generate(prompts, group=1, validation=context)


@pytest.mark.L0
@pytest.mark.CPU
def test_dummy_loaded_rollout_may_generate_after_its_first_committed_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    _build_with_fake_vllm(monkeypatch, RolloutConfig())._require_published_weights()

    rollout = _build_with_fake_vllm(monkeypatch, RolloutConfig(load_format="dummy"))
    rollout._weight_backend = SimpleNamespace(mover=object())
    rollout.mark_weight_version_ready(0)
    rollout.prepare_weight_version(0, None)
    with pytest.raises(RuntimeError, match="no weight version has been committed"):
        rollout._require_published_weights()

    rollout.commit_weight_version(0)

    rollout._require_published_weights()
    assert rollout.weight_version() == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_dummy_loaded_rollout_refuses_stamp_only_apply_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    rollout = _build_with_fake_vllm(monkeypatch, RolloutConfig(load_format="dummy"))
    with pytest.raises(RuntimeError, match="needs a streaming weight edge"):
        rollout.apply_weights(None)
    assert rollout.weight_version() == 0

    static = _build_with_fake_vllm(monkeypatch, RolloutConfig())
    static.apply_weights(None)
    assert static.weight_version() == 1


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("load_format", "loaded", "destinations", "accepted"),
    [
        ("dummy", None, None, False),
        ("dummy", None, ("w",), False),
        ("dummy", {"w"}, None, True),
        ("auto", None, None, True),
    ],
    ids=["dummy-unverifiable", "dummy-mapped", "dummy-reported", "auto-unverifiable"],
)
def test_dummy_loaded_rollout_requires_a_verifiable_weight_stream(
    monkeypatch: pytest.MonkeyPatch,
    load_format: str,
    loaded: set[str] | None,
    destinations: tuple[str, ...] | None,
    accepted: bool,
) -> None:
    from types import SimpleNamespace

    rollout = _build_with_fake_vllm(monkeypatch, RolloutConfig(load_format=load_format))
    rollout._engine_names = {"w"}
    rollout._loaded_names = loaded
    backend = SimpleNamespace(bucket_count=lambda: 2, destination_names=lambda: destinations)

    rollout._coverage_check(backend, 0)
    if accepted:
        rollout._coverage_check(backend, 1)
    else:
        with pytest.raises(ValueError, match="cannot verify this weight stream"):
            rollout._coverage_check(backend, 1)


@pytest.mark.L0
@pytest.mark.CPU
def test_build_vllm_rollout_threads_performance_knobs_through_the_engine_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compilation = {"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]}
    rollout = _build_with_fake_vllm(
        monkeypatch,
        RolloutConfig(enforce_eager=False, compilation_config=compilation, async_scheduling=True, max_num_seqs=16),
    )

    kwargs = rollout.model.kwargs
    assert kwargs["enforce_eager"] is False
    assert kwargs["compilation_config"] == compilation
    assert kwargs["async_scheduling"] is True and kwargs["max_num_seqs"] == 16
    assert "kv_cache_memory_bytes" not in kwargs


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_tp_adds_spmd_wiring_and_seed():
    kw = vllm_engine_kwargs(
        RolloutConfig(tensor_parallel_size=2, seed=7, gpu_memory_utilization=0.5, moe_backend="triton")
    )
    assert kw["tensor_parallel_size"] == 2
    assert kw["distributed_executor_backend"] == "external_launcher"  # reuse the per-replica group as TP
    assert kw["disable_custom_all_reduce"] is True  # per-actor GPU isolation breaks vLLM's P2P probe
    assert kw["seed"] == 7 and kw["gpu_memory_utilization"] == 0.5
    assert kw["moe_backend"] == "triton"


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_enable_custom_all_reduce_only_when_opted_in() -> None:
    assert vllm_engine_kwargs(RolloutConfig(tensor_parallel_size=4))["disable_custom_all_reduce"] is True
    opted_in = vllm_engine_kwargs(RolloutConfig(tensor_parallel_size=4, custom_all_reduce=True))
    assert opted_in["disable_custom_all_reduce"] is False


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"custom_all_reduce": "yes", "tensor_parallel_size": 4}, TypeError, "must be a bool"),
        ({"custom_all_reduce": True, "tensor_parallel_size": 1}, ValueError, r"\[2, 4, 6, 8\]"),
        ({"custom_all_reduce": True, "tensor_parallel_size": 3}, ValueError, r"\[2, 4, 6, 8\]"),
    ],
    ids=["non-bool", "single-gpu-engine", "unsupported-world-size"],
)
def test_rollout_config_rejects_invalid_custom_all_reduce(
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        RolloutConfig(**overrides)  # type: ignore[arg-type]


@pytest.mark.L0
@pytest.mark.CPU
def test_custom_all_reduce_engine_refuses_an_inexact_visible_tp_world(monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM must see this TP replica exactly, never an isolated GPU or another replica's GPUs."""
    from cosmos_framework.posttrain.config.build import _require_visible_tensor_parallel_peers

    opted_in = RolloutConfig(tensor_parallel_size=4, custom_all_reduce=True)
    for visible in ("3", "3,0", "3,0,1,2,4"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
        with pytest.raises(ValueError, match="exactly its tensor-parallel world"):
            _build_with_fake_vllm(monkeypatch, opted_in)
    _require_visible_tensor_parallel_peers(RolloutConfig(tensor_parallel_size=4))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,0,1,2")
    _require_visible_tensor_parallel_peers(opted_in)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    _require_visible_tensor_parallel_peers(opted_in)


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_threads_expert_parallel_to_vllm() -> None:
    kw = vllm_engine_kwargs(RolloutConfig(tensor_parallel_size=4, enable_expert_parallel=True))

    assert kw["enable_expert_parallel"] is True


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_thread_model_and_tokenizer_revisions() -> None:
    kw = vllm_engine_kwargs(
        RolloutConfig(
            name_or_path="org/model",
            revision="model-commit",
            tokenizer_revision="tokenizer-commit",
        )
    )

    assert kw["revision"] == "model-commit"
    assert kw["tokenizer_revision"] == "tokenizer-commit"


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_thread_remote_code_contract() -> None:
    """The rollout config carries its own trust opt-in, independent of the policy-owned decision."""
    kw = vllm_engine_kwargs(RolloutConfig(trust_remote_code=True))

    assert kw["trust_remote_code"] is True


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_propagate_policy_remote_code_trust() -> None:
    """The policy-owned trust decision reaches vLLM even when the rollout config leaves it off."""
    kw = vllm_engine_kwargs(RolloutConfig(), trust_remote_code=True)

    assert kw["trust_remote_code"] is True


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_thread_multimodal_attention_backend() -> None:
    kw = vllm_engine_kwargs(RolloutConfig(mm_encoder_attn_backend="TORCH_SDPA"))

    assert kw["mm_encoder_attn_backend"] == "TORCH_SDPA"


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_thread_decoder_attention_backend() -> None:
    kw = vllm_engine_kwargs(RolloutConfig(attention_backend="TRITON_ATTN"))

    assert kw["attention_config"] == {"backend": "TRITON_ATTN"}


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_omit_limit_mm_per_prompt_by_default() -> None:
    # Unset -> the kwarg is absent, so every existing run's engine construction is byte-identical.
    assert "limit_mm_per_prompt" not in vllm_engine_kwargs(RolloutConfig())


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_thread_limit_mm_per_prompt_as_a_copy() -> None:
    limits = {"image": 40, "video": 0}
    cfg = RolloutConfig(limit_mm_per_prompt=limits)

    kw = vllm_engine_kwargs(cfg)

    assert kw["limit_mm_per_prompt"] == {"image": 40, "video": 0}
    assert kw["limit_mm_per_prompt"] is not limits  # the engine never aliases the config mapping


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("value", "error"),
    [
        ({}, TypeError),  # empty mapping says nothing
        ([("image", 4)], TypeError),  # not a mapping
        ({"": 4}, TypeError),  # modality must be a non-empty string
        ({"image": True}, TypeError),  # bool is not a count
        ({"image": 4.0}, TypeError),  # float is not a count
        ({"image": -1}, ValueError),  # negative budget
    ],
)
def test_rollout_config_rejects_malformed_limit_mm_per_prompt(value, error) -> None:
    with pytest.raises(error):
        RolloutConfig(limit_mm_per_prompt=value)


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_omit_performance_knobs_by_default() -> None:
    kw = vllm_engine_kwargs(RolloutConfig())

    assert RolloutConfig().compilation_config == {}
    assert "compilation_config" not in vllm_engine_kwargs(RolloutConfig(compilation_config={}))
    for key in ("compilation_config", "async_scheduling", "max_num_seqs", "kv_cache_memory_bytes"):
        assert key not in kw


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_thread_performance_knobs_under_vllm_names() -> None:
    compilation = {"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": [1, 2, 4, 8]}
    cfg = RolloutConfig(
        enforce_eager=False,
        compilation_config=compilation,
        async_scheduling=False,
        max_num_seqs=64,
        kv_cache_memory_bytes=8 << 30,
    )

    kw = vllm_engine_kwargs(cfg)

    assert kw["compilation_config"] == compilation
    # vLLM builds its CompilationConfig from the mapping, so neither it nor its lists alias the config.
    assert kw["compilation_config"] is not cfg.compilation_config
    assert kw["compilation_config"]["cudagraph_capture_sizes"] is not cfg.compilation_config["cudagraph_capture_sizes"]
    assert kw["async_scheduling"] is False
    assert kw["max_num_seqs"] == 64 and kw["kv_cache_memory_bytes"] == 8 << 30


@pytest.mark.L0
@pytest.mark.CPU
def test_compilation_config_accepts_hydra_mappings_and_requires_graph_capture() -> None:
    authored = OmegaConf.create({"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2]})

    cfg = RolloutConfig(enforce_eager=False, compilation_config=authored)

    assert cfg.compilation_config == {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2]}
    assert type(cfg.compilation_config) is dict and type(cfg.compilation_config["cudagraph_capture_sizes"]) is list
    with pytest.raises(ValueError, match="enforce_eager=True disables"):
        RolloutConfig(compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"})


@pytest.mark.L0
@pytest.mark.CPU
def test_compilation_config_takes_a_hydra_dict_override() -> None:
    """`++rollout.compilation_config={...}` is Hydra's struct-mode add-or-override: a force-added merge.

    The merge needs a dict node rather than None, and the struct flag Hydra sets on the composed
    config refuses new keys unless they are force-added, so the plain `key={...}` form fails.
    """
    from dataclasses import dataclass, field

    from omegaconf.errors import ConfigKeyError

    @dataclass
    class _Root:
        rollout: RolloutConfig = field(default_factory=RolloutConfig)

    cfg = OmegaConf.structured(_Root)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.update(cfg, "rollout.enforce_eager", False, merge=True)
    compilation = {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2]}
    with pytest.raises(ConfigKeyError):
        OmegaConf.update(cfg, "rollout.compilation_config", compilation, merge=True)
    OmegaConf.update(cfg, "rollout.compilation_config", compilation, merge=True, force_add=True)

    rollout = OmegaConf.to_object(cfg).rollout

    assert rollout.compilation_config == {"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2]}
    assert vllm_engine_kwargs(rollout)["compilation_config"] == rollout.compilation_config


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        pytest.param({"enforce_eager": False, "compilation_config": [("mode", 0)]}, TypeError, id="compilation-list"),
        pytest.param({"async_scheduling": 1}, TypeError, id="async-scheduling-int"),
        pytest.param({"max_num_seqs": True}, TypeError, id="max-num-seqs-bool"),
        pytest.param({"max_num_seqs": 0}, ValueError, id="max-num-seqs-zero"),
        pytest.param({"kv_cache_memory_bytes": 1.5}, TypeError, id="kv-cache-float"),
        pytest.param({"kv_cache_memory_bytes": -1}, ValueError, id="kv-cache-negative"),
    ],
)
def test_rollout_config_rejects_malformed_performance_knobs(overrides: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        RolloutConfig(**overrides)


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_runtime_forces_the_supported_inprocess_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    monkeypatch.delenv("VLLM_ALLOW_INSECURE_SERIALIZATION", raising=False)

    _configure_vllm_runtime()

    assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
    assert "VLLM_ALLOW_INSECURE_SERIALIZATION" not in os.environ


@pytest.mark.L0
@pytest.mark.CPU
def test_sampling_params_exclude_framework_owned_keys():
    sp = sampling_params(RolloutConfig(temperature=0.7, top_p=0.9, max_tokens=32))
    assert sp == {"temperature": 0.7, "top_p": 0.9, "top_k": -1, "repetition_penalty": 1.0, "max_tokens": 32}
    assert "n" not in sp and "logprobs" not in sp  # framework-owned; VLLMRollout rejects them


@pytest.mark.L0
@pytest.mark.CPU
def test_sampling_params_always_pins_top_k_and_repetition_penalty():
    """Both knobs must be EMITTED even at their neutral values, and must not disagree across flavors.

    Omitting them is not equivalent to passing the neutral value: `generate` merges kwargs onto the
    model object's generation_config, so an absent key takes whatever that object carries — which
    varies with the load path. Pinning them makes the rollout distribution a property of the config.
    """
    default = sampling_params(RolloutConfig())
    assert default["top_k"] == -1  # -1 is vLLM's "disabled"
    assert default["repetition_penalty"] == 1.0

    # A configured top_k passes through; any non-positive value normalizes to vLLM's -1 so the field
    # reads identically on both backends.
    assert sampling_params(RolloutConfig(top_k=20))["top_k"] == 20
    assert sampling_params(RolloutConfig(top_k=0))["top_k"] == -1
    assert sampling_params(RolloutConfig(repetition_penalty=1.05))["repetition_penalty"] == 1.05


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_sampling_matches_vllm_sampling_for_the_same_config():
    """The HF and vLLM rollout flavors must sample from the SAME distribution for one RolloutConfig.

    They spell "no top-k" differently (HF wants None, vLLM wants -1), which is exactly the kind of
    difference that hides a real divergence, so the mapping is pinned on both sides.
    """
    rc = RolloutConfig(temperature=0.9, top_p=1.0, max_tokens=384)
    hf, vl = hf_sampling_params(rc), sampling_params(rc)
    assert hf["temperature"] == vl["temperature"] and hf["top_p"] == vl["top_p"]
    assert hf["max_new_tokens"] == vl["max_tokens"]
    assert hf["repetition_penalty"] == vl["repetition_penalty"] == 1.0
    assert hf["top_k"] is None and vl["top_k"] == -1  # same meaning, each backend's spelling

    rc = RolloutConfig(top_k=20, repetition_penalty=1.05)
    hf, vl = hf_sampling_params(rc), sampling_params(rc)
    assert hf["top_k"] == vl["top_k"] == 20
    assert hf["repetition_penalty"] == vl["repetition_penalty"] == 1.05


@pytest.mark.L0
@pytest.mark.CPU
def test_dp_coords_reports_the_full_policy_dp_extent():
    """`_dp_coords` must span dp_replicate x dp_shard, not just the shard extent.

    Under HSDP both axes carry data parallelism. Reporting only `dp_shard` would give ranks in
    DIFFERENT replicas the same `dp_rank`, so the ingress predicate (`select_chunk`) would hand them
    the SAME groups and the cross-replica all-reduce would double-count them — silent, wrong gradients.
    """
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.build import _dp_coords

    def _mesh(*axes):
        """A stub with the REAL `DeviceMesh` contract: per-axis `get_local_rank(dim)`/`size(dim)`.

        The earlier stub exposed a NO-ARG `get_local_rank()`, which torch only permits on a 1-D
        mesh. That made a 2-D HSDP mesh look callable here while the real one raised
        "Optional kwarg `mesh_dim` needs to be specified when device_mesh.ndim > 1" — so this test
        passed against code that died on 4 GPUs. Axes are `(local_rank, size)` pairs, outermost
        first, matching the ('dp_replicate', 'dp_shard') order the real mesh reports.
        """
        import math

        return SimpleNamespace(
            ndim=len(axes),
            get_local_rank=lambda dim: axes[dim][0],
            size=lambda dim=None: axes[dim][1] if dim is not None else math.prod(a[1] for a in axes),
        )

    assert _dp_coords(None) == (0, 1)  # no mesh -> wrapper is a no-op
    assert _dp_coords(SimpleNamespace(dp_enabled=False)) == (0, 1)
    # Pure FSDP: the full DP mesh and shard mesh have the same four-rank extent.
    pure = SimpleNamespace(
        dp_enabled=True,
        dp_replicate=1,
        dp_shard_enabled=True,
        dp_shard_mesh=_mesh((2, 4)),
        dp_mesh=_mesh((0, 1), (2, 4)),
    )
    assert _dp_coords(pure) == (2, 4)
    # HSDP: policy has one execution replica, so both model-DP dimensions partition its one batch.
    hsdp = SimpleNamespace(
        dp_enabled=True,
        dp_replicate=2,
        dp_shard_enabled=True,
        dp_shard_mesh=_mesh((1, 2)),
        dp_mesh=_mesh((1, 2), (1, 2)),
    )
    assert _dp_coords(hsdp) == (3, 4)

    # Replicate-only still has a real four-rank DP extent.
    rep_only = SimpleNamespace(
        dp_enabled=True, dp_replicate=4, dp_shard_enabled=False, dp_shard_mesh=None, dp_mesh=_mesh((2, 4))
    )
    assert _dp_coords(rep_only) == (2, 4)


@pytest.mark.L0
@pytest.mark.CPU
def test_build_objective_ignores_pdims_for_both_flavors():
    """Task 9 deleted `dp_group_sharded`: a NON-TRIVIAL pdims must NOT change what `build_objective`
    returns — for BOTH flavors. Partitioning the dp axis is the ingress predicate's job now
    (`workers/dispatch.py::select_chunk`, applied by the WorkerGroup `train` edge); wrapping the
    objective on `pdims` here too would divide the same axis twice, training on `1/dp_world**2` of
    the batch while every metric reported the full count — the exact silent bug Task 9 closes.
    """
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
    from cosmos_framework.posttrain.config.build import build_objective
    from cosmos_framework.posttrain.config.rl import GRPOConfig
    from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

    class _TinyLM(nn.Module):
        def __init__(self, vocab: int = 16, d: int = 8):
            super().__init__()
            self.emb, self.head = nn.Embedding(vocab, d), nn.Linear(d, vocab)

        def forward(self, ids):
            return SimpleNamespace(logits=self.head(self.emb(ids)))

    mesh = SimpleNamespace(ndim=1, get_local_rank=lambda dim: 0, size=lambda dim=None: 2)
    # dp_shard_enabled=True: a genuinely non-trivial mesh, so a wrap (if build_objective still did
    # one) would be observable — the negative result below is a real assertion, not a vacuous one.
    pdims = SimpleNamespace(dp_enabled=True, dp_replicate=1, dp_shard_enabled=True, dp_shard_mesh=mesh, dp_mesh=mesh)

    def _batch():
        trajs = [
            Trajectory(
                transitions=[
                    Transition(
                        obs=ObsBundle(),
                        action=torch.tensor([3 + k]),
                        policy_info={"prompt_ids": torch.tensor([1, 2])},
                        reward={"final": float(k)},
                        done=True,
                    )
                ],
                id=f"a{k}",
                group_id="a",
                status=Status.COMPLETED,
            )
            for k in range(4)
        ]
        group_relative(group_size=4)(trajs)  # the reward stage's job, done here
        return trajs

    # Asserted on the count the objective actually scored, for BOTH a pdims-carrying call and a
    # pdims=None one — the two must now agree (both see all 4 samples).
    for flavor in ("legacy_pg", "clipped"):
        rl = GRPOConfig(group_size=4)
        rl.grpo_surrogate = flavor
        torch.manual_seed(0)
        with_mesh = build_objective(rl, "gpt2", pdims=pdims)
        _, m_with_mesh = with_mesh.fn(_TinyLM(), _batch())
        torch.manual_seed(0)
        without_mesh = build_objective(rl, "gpt2", pdims=None)
        _, m_without_mesh = without_mesh.fn(_TinyLM(), _batch())

        assert m_without_mesh["grpo_n"] == 4, f"{flavor}: pdims=None objective must see the whole batch"
        assert m_with_mesh["grpo_n"] == 4, (
            f"{flavor}: build_objective must not shard on pdims any more — scored "
            f"{m_with_mesh['grpo_n']}/4 samples; dp partitioning is the ingress's job now (select_chunk)"
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_ppo_engine_kwargs_use_processed_logprobs_and_expand_topk_cap() -> None:
    kw = ppo_vllm_engine_kwargs(
        RolloutConfig(tensor_parallel_size=4, attention_backend="TRITON_ATTN"), behavior_top_k=128
    )
    assert kw["max_logprobs"] == 128
    assert kw["logprobs_mode"] == "processed_logprobs"
    assert kw["tensor_parallel_size"] == 4
    assert kw["attention_config"] == {"backend": "TRITON_ATTN"}
    with pytest.raises(ValueError, match="behavior_top_k"):
        ppo_vllm_engine_kwargs(RolloutConfig(), behavior_top_k=0)


@pytest.mark.L0
@pytest.mark.CPU
def test_ppo_engine_kwargs_refuse_custom_all_reduce() -> None:
    with pytest.raises(ValueError, match="custom_all_reduce"):
        ppo_vllm_engine_kwargs(RolloutConfig(tensor_parallel_size=4, custom_all_reduce=True), behavior_top_k=128)


@pytest.mark.L0
@pytest.mark.CPU
def test_limit_mm_per_prompt_key_policy_refuses_mistyped_and_unknown_modalities():
    """vLLM converts an unknown or mistyped key into a generic dummy option and the
    model's `allowed_mm_limits` iterates only its supported modalities, so `{'images': 40}` or `{' image': 40}`
    would be silently ignored. FAIL-ON-OLD: the first revision accepted both."""
    from cosmos_framework.posttrain.config.rl import VLLM_MODALITIES, RolloutConfig

    assert VLLM_MODALITIES == frozenset({"image", "video", "audio"})
    ok = RolloutConfig(limit_mm_per_prompt={"image": 40, "video": 0})
    assert ok.limit_mm_per_prompt == {"image": 40, "video": 0}
    from_hydra = RolloutConfig(limit_mm_per_prompt=OmegaConf.create({"image": 40}))
    assert from_hydra.limit_mm_per_prompt == {"image": 40}
    assert isinstance(from_hydra.limit_mm_per_prompt, dict)
    for bad in ({"images": 40}, {" image": 40}, {"image ": 40}, {"Image": 40}, {"image-1": 40}, {"pixels": 1}):
        with pytest.raises(ValueError, match="limit_mm_per_prompt key"):
            RolloutConfig(limit_mm_per_prompt=bad)


@pytest.mark.L0
@pytest.mark.CPU
def test_limit_mm_per_prompt_is_checked_against_the_built_engines_supported_modalities():
    """The model-specific supported set is checked at rollout construction without importing vLLM."""
    from types import SimpleNamespace

    from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout, validate_limit_mm_against_engine

    class _Registry:
        def __init__(self, supported: dict, has_processor: bool = True) -> None:
            self.supported, self.has_processor = supported, has_processor

        def create_processor(self, model_config):
            if not self.has_processor:
                raise ValueError("not a multimodal model")
            return SimpleNamespace(info=SimpleNamespace(get_supported_mm_limits=lambda: self.supported))

    model = SimpleNamespace(llm_engine=SimpleNamespace(model_config=SimpleNamespace(model="qwen3-vl")))
    reg = _Registry({"image": None, "video": None})
    assert validate_limit_mm_against_engine(model, {"image": 40}, owner="t", registry=reg) == {"image", "video"}
    assert validate_limit_mm_against_engine(model, {"image": 0, "video": 0}, owner="t", registry=reg) == {
        "image",
        "video",
    }
    with pytest.raises(ValueError, match=r"\['audio'\] are not modalities"):
        validate_limit_mm_against_engine(model, {"image": 40, "audio": 1}, owner="t", registry=reg)
    with pytest.raises(ValueError, match="no registered multimodal processor"):
        validate_limit_mm_against_engine(model, {"image": 40}, owner="t", registry=_Registry({}, has_processor=False))
    with pytest.raises(ValueError, match="supports no multimodal inputs"):
        validate_limit_mm_against_engine(model, {"image": 40}, owner="t", registry=_Registry({}))
    with pytest.raises(ValueError, match="no llm_engine.model_config"):
        validate_limit_mm_against_engine(object(), {"image": 40}, owner="t", registry=reg)
    fake_packer = SimpleNamespace(adds_special_tokens=lambda: False)
    with pytest.raises(ValueError, match="no llm_engine.model_config"):
        VLLMRollout(
            object(),
            packer=fake_packer,
            engine_config={"model": "m", "limit_mm_per_prompt": {"image": 40}},
        )  # type: ignore[arg-type]
    VLLMRollout(object(), packer=fake_packer, engine_config={"model": "m"})  # type: ignore[arg-type]  # no key -> no check
