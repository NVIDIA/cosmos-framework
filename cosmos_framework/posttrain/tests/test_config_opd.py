# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""@L0 @CPU: `config/rl.py::OPDConfig`/`TeacherConfig` + their builders in `config/build.py`.

`vllm_teacher_engine_kwargs` is a pure config→kwargs mapping precisely so the two settings that are
easy to get wrong — and expensive to discover on a GPU — are checkable on CPU:

- `max_logprobs` must cover `top_k`; otherwise the teacher cannot return the requested support.
- prefix caching must stay ON for the teacher, where the rollout forces it OFF. Opposite defaults,
  same engine, and the reason is not symmetric: the rollout's cache would hold pre-weight-sync KV,
  while a teacher is frozen.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from cosmos_framework.posttrain.config.build import (
    build_opd_objective,
    build_vllm_teacher,
    effective_model_id,
    validate_opd_runtime,
    vllm_teacher_engine_kwargs,
)
from cosmos_framework.posttrain.config.config_grpo import PosttrainGRPOConfig
from cosmos_framework.posttrain.config.config_opd import PosttrainOPDConfig
from cosmos_framework.posttrain.config.model import DataPackerConfig, HFModelConfig, Qwen3VLModelConfig
from cosmos_framework.posttrain.config.rl import (
    DatasetConfig,
    GRPOConfig,
    GRPOValidationConfig,
    OPDConfig,
    RolloutConfig,
    TeacherConfig,
)
from cosmos_framework.posttrain.configs import opd_qwen1p5b_countdown as opd_recipe
from cosmos_framework.posttrain.model.qwen3_vl import build_qwen3_vl_model, build_qwen3_vl_native_policy
from cosmos_framework.posttrain.tests.helpers.config import skip_recipe_registration


@pytest.mark.L0
@pytest.mark.CPU
def test_opd_inherits_the_grpo_surrogate_knobs() -> None:
    """`advantage` mode IS the GRPO surrogate with a penalty folded in, so the knobs must be the
    same objects rather than a parallel set that can drift."""
    cfg = OPDConfig()

    assert isinstance(cfg, GRPOConfig)
    for field in ("epsilon_low", "epsilon_high", "kl_beta", "lower_bound_ratio", "loss_type", "group_size"):
        assert getattr(cfg, field) == getattr(GRPOConfig(), field), field
    assert cfg.distillation_mode == "advantage"
    assert cfg.jsd_beta == 1.0 and cfg.kl_penalty_coef == 1.0
    assert cfg.weight_sync_transport is None
    assert cfg.weight_sync_bucket_bytes is None
    assert cfg.weight_sync_probe_bytes == 256 * 1024 * 1024
    assert cfg.weight_sync_min_gbps == 20.0

    validated = OPDConfig(
        validation=GRPOValidationConfig(
            enabled=True,
            data=DatasetConfig(path="held-out.jsonl", prompt_column="prompt"),
        )
    )
    assert validated.validation.enabled is True
    assert validated.validation.data.path == "held-out.jsonl"


@pytest.mark.L0
@pytest.mark.CPU
def test_opd_root_and_countdown_demo_reuse_the_grpo_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    skip_recipe_registration(monkeypatch, opd_recipe._make_root_config, PosttrainOPDConfig)
    assert issubclass(PosttrainOPDConfig, PosttrainGRPOConfig)
    assert set(PosttrainOPDConfig.__annotations__) == {"algorithm", "model_backend", "rl"}

    cfg = opd_recipe.build_config()
    assert (cfg.algorithm, cfg.model_backend, cfg.rl.distillation_mode) == ("opd", "hf", "direct")
    assert cfg.job.name == "qwen1p5b_countdown"
    assert "${now" not in cfg.job.name
    assert cfg.rl.save_every == 5
    assert cfg.model_hf.name_or_path == cfg.rollout.name_or_path == "Qwen/Qwen2.5-1.5B-Instruct"
    assert cfg.rl.teacher.name_or_path == "Qwen/Qwen2.5-3B-Instruct"
    assert (cfg.rl.teacher.input_mode, cfg.rl.teacher.signal_mode, cfg.rl.teacher.tensor_parallel_size) == (
        "shared_vocab_multimodal",
        "topk",
        1,
    )

    mapped = opd_recipe.countdown_items(
        {
            "messages": [{"role": "user", "content": "solve"}, {"role": "assistant", "content": "gold"}],
            "nums": [2, 3, 4],
            "target": 14,
        }
    )
    assert mapped["messages"] == [{"role": "user", "content": "solve"}]
    assert mapped["reference"] == {"nums": [2, 3, 4], "target": 14}


@pytest.mark.L0
@pytest.mark.CPU
def test_invalid_settings_are_rejected_at_construction() -> None:
    """Cheap, decidable knobs fail at config time, not after Ray placement and a model load."""
    with pytest.raises(ValueError, match="distillation_mode"):
        OPDConfig(distillation_mode="reverse_kl")
    with pytest.raises(ValueError, match=r"jsd_beta in \[0, 1\]"):
        OPDConfig(jsd_beta=1.5)
    for invalid_top_k in (True, -1, 1.0):
        with pytest.raises(ValueError, match="teacher.top_k must be a non-bool integer >= 0"):
            OPDConfig(teacher=TeacherConfig(top_k=invalid_top_k))
    with pytest.raises(ValueError, match="signal_mode='topk'.*top_k > 0"):
        OPDConfig(teacher=TeacherConfig(top_k=0))
    for invalid_kl_penalty_coef in (True, -0.1, float("nan"), float("inf"), "1.0"):
        with pytest.raises(ValueError, match="kl_penalty_coef must be a non-bool finite real >= 0"):
            OPDConfig(kl_penalty_coef=invalid_kl_penalty_coef)
    boundary_cfg = OPDConfig(teacher=TeacherConfig(top_k=1), kl_penalty_coef=0.0)
    assert boundary_cfg.teacher.top_k == 1
    assert boundary_cfg.kl_penalty_coef == 0.0
    dpca = OPDConfig(
        distillation_mode="advantage",
        jsd_beta=None,
        teacher=TeacherConfig(input_mode="cross_token_text", signal_mode="dpca", top_k=0),
    )
    assert dpca.teacher.signal_mode == "dpca"
    multimodal_dpca = OPDConfig(
        distillation_mode="advantage",
        jsd_beta=None,
        teacher=TeacherConfig(input_mode="cross_token_multimodal", signal_mode="dpca", top_k=0),
    )
    assert multimodal_dpca.teacher.input_mode == "cross_token_multimodal"
    with pytest.raises(ValueError, match="requires input_mode='cross_token_text'.*'cross_token_multimodal'"):
        OPDConfig(jsd_beta=None, teacher=TeacherConfig(signal_mode="dpca", top_k=0))
    with pytest.raises(ValueError, match="signal_mode='topk'.*shared_vocab_multimodal"):
        OPDConfig(teacher=TeacherConfig(input_mode="cross_token_multimodal"))
    with pytest.raises(ValueError, match="teacher.input_mode must be"):
        OPDConfig(teacher=TeacherConfig(input_mode="cross_token_video"))
    with pytest.raises(ValueError, match="requires distillation_mode='advantage'"):
        OPDConfig(
            distillation_mode="direct",
            jsd_beta=None,
            teacher=TeacherConfig(input_mode="cross_token_text", signal_mode="dpca", top_k=0),
        )
    with pytest.raises(ValueError, match="requires teacher.top_k=0"):
        OPDConfig(
            jsd_beta=None,
            teacher=TeacherConfig(input_mode="cross_token_text", signal_mode="dpca", top_k=8),
        )
    with pytest.raises(ValueError, match="requires jsd_beta=None"):
        OPDConfig(
            teacher=TeacherConfig(input_mode="cross_token_text", signal_mode="dpca", top_k=0),
        )
    for distillation_mode in ("direct", "advantage"):
        with pytest.raises(ValueError, match="OPD requires kl_beta=0"):
            OPDConfig(distillation_mode=distillation_mode, kl_beta=0.1)
    # OPD owns one distillation selector. Advantage mode always uses the clipped surrogate and
    # freezes pi_old across a multi-mini update window; direct mode forms no policy-gradient
    # surrogate. An independently selectable inherited surrogate would therefore be either
    # contradictory or ignored.
    with pytest.raises(ValueError, match="OPD requires grpo_surrogate='clipped'"):
        OPDConfig(grpo_surrogate="legacy_pg")
    with pytest.raises(ValueError, match="does not consume rollout logprobs"):
        OPDConfig(use_rollout_logprobs_for_loss=True)
    with pytest.raises(ValueError, match="Dr-GRPO-only loss_type"):
        OPDConfig(loss_type="seq-mean-token-sum-fixed")
    for distillation_mode in ("direct", "advantage"):
        with pytest.raises(ValueError, match="does not consume logprob_dtype in either distillation mode"):
            OPDConfig(distillation_mode=distillation_mode, logprob_dtype="bfloat16")
    # Refused BEFORE the inherited GRPO rules: advantage mode scores a token-level ratio and direct
    # mode forms none, so GSPO advice about the dual clip would point at a knob OPD never reads.
    for distillation_mode in ("direct", "advantage"):
        with pytest.raises(ValueError, match="OPD does not consume importance_sampling_level"):
            OPDConfig(distillation_mode=distillation_mode, importance_sampling_level="sequence")
    with pytest.raises(ValueError, match=r"retry_backoff_s.*\[0, 60\]"):
        OPDConfig(teacher=TeacherConfig(retry_backoff_s=61.0))
    with pytest.raises(ValueError, match="must divide evenly by mini_batch_per_replica"):
        OPDConfig(batch_per_replica=8, group_size=4, mini_batch_per_replica=3)
    with pytest.raises(ValueError, match="weight_sync_transport"):
        OPDConfig(weight_sync_transport="nccl")
    for invalid_bucket_bytes in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="weight_sync_bucket_bytes"):
            OPDConfig(weight_sync_bucket_bytes=invalid_bucket_bytes)
    for invalid_probe_bytes in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="weight_sync_probe_bytes"):
            OPDConfig(weight_sync_probe_bytes=invalid_probe_bytes)
    for invalid_min_gbps in (True, -0.1, float("nan"), float("inf"), "20"):
        with pytest.raises(ValueError, match="weight_sync_min_gbps"):
            OPDConfig(weight_sync_min_gbps=invalid_min_gbps)


@pytest.mark.L0
@pytest.mark.CPU
def test_opd_selector_migration_is_one_way_and_separate_from_grpo() -> None:
    """GRPO surrogate and OPD algorithm selection have one spelling and different ownership."""
    import dataclasses

    from omegaconf import OmegaConf

    fields = {item.name for item in dataclasses.fields(OPDConfig)}
    assert {"grpo_surrogate", "distillation_mode"} <= fields
    assert "objective" not in fields
    assert "objective_mode" not in fields
    schema = OmegaConf.structured(OPDConfig)
    with pytest.raises(Exception):
        OmegaConf.merge(schema, OmegaConf.from_dotlist(["objective_mode=direct"]))


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon_low": 0.1},
        {"epsilon_high": 0.3},
        {"lower_bound_ratio": 0.0},
        {"kl_penalty_coef": 0.0},
        {"unbiased_advantage": True},
    ],
)
def test_direct_opd_rejects_configured_advantage_only_knobs(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="does not consume advantage-surrogate settings"):
        OPDConfig(distillation_mode="direct", **kwargs)


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_opd_warns_that_task_reward_is_not_its_training_signal() -> None:
    with pytest.warns(UserWarning, match="does not use task reward or group advantage"):
        OPDConfig(distillation_mode="direct")


@pytest.mark.L0
@pytest.mark.CPU
def test_engine_kwargs_raise_max_logprobs_to_top_k() -> None:
    """The teacher engine must return every support token requested by the objective."""
    kwargs = vllm_teacher_engine_kwargs(TeacherConfig(top_k=32))

    assert kwargs["max_logprobs"] == 32
    assert kwargs["logprobs_mode"] == "processed_logprobs"
    assert kwargs["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert vllm_teacher_engine_kwargs(TeacherConfig(top_k=8))["max_logprobs"] == 8


@pytest.mark.L0
@pytest.mark.CPU
def test_cross_token_multimodal_builder_constructs_and_passes_teacher_packer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact teacher packer used for rendering is part of construction identity and the body."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    captured: dict[str, Any] = {}

    class _Tokenizer:
        chat_template = "teacher-template"
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 2

        def get_vocab(self) -> dict[str, int]:
            return {"<pad>": 2, "teacher-only-token": 9}

    class _Component:
        def to_dict(self) -> dict[str, Any]:
            return {"size": {"longest_edge": 1_024}}

    class _VisionConfig:
        spatial_merge_size = 2
        patch_size = 14
        temporal_patch_size = 2

        def to_dict(self) -> dict[str, Any]:
            return {"model_type": "qwen3_5_moe", "spatial_merge_size": 2}

    class _ModelConfig:
        vision_config = _VisionConfig()
        image_token_id = 248_056
        vision_start_token_id = 248_053
        vision_end_token_id = 248_054

        def to_dict(self) -> dict[str, Any]:
            return {
                "model_type": "qwen3_5_moe",
                "vision_config": self.vision_config.to_dict(),
                "image_token_id": self.image_token_id,
                "vision_start_token_id": self.vision_start_token_id,
                "vision_end_token_id": self.vision_end_token_id,
            }

    teacher_packer = SimpleNamespace(
        processor=SimpleNamespace(
            tokenizer=_Tokenizer(),
            image_processor=_Component(),
            video_processor=None,
        ),
        model_config=_ModelConfig(),
        image_token_id=248_056,
        processor_semantics_identity=lambda: {
            "tokenizer": {"chat_template": "teacher-template"},
        },
    )
    student_packer = SimpleNamespace(processor=object(), model_config=object(), tokenizer=object())

    config_loads: list[str] = []

    def _load_config(path: str) -> _ModelConfig:
        config_loads.append(path)
        return _ModelConfig()

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _load_config)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.data.packer.init_data_packer_from_factory",
        lambda path, factory, **kwargs: (
            captured.setdefault("teacher_model", path),
            captured.setdefault("teacher_model_factory", factory),
            captured.setdefault("teacher_packer_kwargs", kwargs),
            teacher_packer,
        )[-1],
    )

    vllm = ModuleType("vllm")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")

    def _llm(**kwargs: Any) -> object:
        assert os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
        captured["engine_kwargs"] = kwargs
        # vLLM registers its own Qwen3.5 config class globally during construction, so a
        # post-construction AutoConfig read is not the same parser identity as the preflight read.
        monkeypatch.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda _path: pytest.fail("teacher construction must not re-read config after vLLM mutates the registry"),
        )
        return object()

    vllm.LLM = _llm
    monkeypatch.setitem(sys.modules, "vllm", vllm)

    def _teacher(model: Any, **kwargs: Any) -> SimpleNamespace:
        captured["engine"] = model
        captured["teacher_kwargs"] = kwargs
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("cosmos_framework.posttrain.roles.teacher.VLLMTeacher", _teacher)

    def _gather(identity: dict[str, Any], *, context: str) -> dict[int, dict[str, Any]]:
        captured["construction_identity"] = identity
        captured["construction_context"] = context
        return {0: identity}

    monkeypatch.setattr("cosmos_framework.posttrain.utils.rank_state.gather_rank_states", _gather)

    cfg = TeacherConfig(
        name_or_path="/models/qwen3.5-397b",
        input_mode="cross_token_multimodal",
        signal_mode="dpca",
        top_k=0,
    )
    data_packer = DataPackerConfig(
        teacher_model_factory="cosmos_framework.posttrain.tests.helpers.custom_data.build_default_packer"
    )
    teacher = build_vllm_teacher(cfg, student_packer, data_packer)

    assert captured["teacher_model"] == cfg.name_or_path
    assert captured["teacher_model_factory"] == data_packer.teacher_model_factory
    assert captured["teacher_packer_kwargs"] == {"trust_remote_code": False, "model_backend": None}
    assert config_loads == [cfg.name_or_path]
    assert teacher.packer is student_packer
    assert teacher.teacher_packer is teacher_packer
    identity = captured["construction_identity"]
    assert identity["shared_vocab_identity"] is None
    assert identity["teacher_processor_identity"]["image_ids"]["image_token_id"] == 248_056


@pytest.mark.L0
@pytest.mark.CPU
def test_teacher_retries_a_transient_local_model_config_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """The teacher's identity read gets the rollout's bounded retry for an absolute local path."""
    transformers = pytest.importorskip("transformers")
    import cosmos_framework.posttrain.config.build as build

    config_loads: list[str] = []
    sleeps: list[float] = []

    def _load_config(path: str) -> SimpleNamespace:
        config_loads.append(path)
        if len(config_loads) == 1:
            raise OSError("transient shared-filesystem visibility failure")
        return SimpleNamespace(to_dict=lambda: {"model_type": "fake"})

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", _load_config)
    monkeypatch.setattr(build.time, "sleep", sleeps.append)
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    vllm = ModuleType("vllm")
    vllm.LLM = lambda **_kwargs: object()
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setattr(
        "cosmos_framework.posttrain.roles.teacher.VLLMTeacher",
        lambda _model, **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.rank_state.gather_rank_states",
        lambda identity, *, context: {0: identity},
    )

    cfg = TeacherConfig(name_or_path="/models/teacher", input_mode="cross_token_text", signal_mode="dpca", top_k=0)
    build_vllm_teacher(cfg, SimpleNamespace())

    assert config_loads == [cfg.name_or_path] * 2
    assert sleeps == [1.0]


@pytest.mark.L0
@pytest.mark.CPU
def test_teacher_keeps_prefix_caching_where_the_rollout_disables_it() -> None:
    """Opposite defaults on the same engine knob, for an asymmetric reason.

    The rollout forces prefix caching off because weight sync does not invalidate the cache, so a
    cached prompt would serve KV computed under pre-sync weights. A teacher is frozen — nothing
    invalidates it — and a GRPO group's completions all share one prompt, which is exactly the case
    the cache exists for."""
    from cosmos_framework.posttrain.config.build import vllm_engine_kwargs
    from cosmos_framework.posttrain.config.rl import RolloutConfig

    assert vllm_teacher_engine_kwargs(TeacherConfig())["enable_prefix_caching"] is True
    assert vllm_engine_kwargs(RolloutConfig())["enable_prefix_caching"] is False


@pytest.mark.L0
@pytest.mark.CPU
def test_teacher_moe_backend_passes_through_like_the_rollout() -> None:
    """The teacher engine takes the same MoE kernel knob as the rollout, default ``auto``.

    Without it the teacher always runs vLLM's architecture-specific pick, which on GB300 (SM100
    FlashInfer TRT-LLM BF16 MoE) faults with an illegal address while the engine is built at TP=2,
    even though the student rollout beside it runs ``triton`` and trains."""
    assert vllm_teacher_engine_kwargs(TeacherConfig())["moe_backend"] == "auto"
    assert vllm_teacher_engine_kwargs(TeacherConfig(moe_backend="triton"))["moe_backend"] == "triton"
    assert TeacherConfig().moe_backend == RolloutConfig().moe_backend


@pytest.mark.L0
@pytest.mark.CPU
def test_ranked_teacher_selects_external_launcher() -> None:
    """The independent teacher WorkerGroup supplies one external-launcher lane per TP rank."""
    single = vllm_teacher_engine_kwargs(TeacherConfig(tensor_parallel_size=1))
    assert "distributed_executor_backend" not in single

    ranked = OPDConfig(teacher=TeacherConfig(tensor_parallel_size=8))
    kwargs = vllm_teacher_engine_kwargs(ranked.teacher)
    assert kwargs["tensor_parallel_size"] == 8
    assert kwargs["distributed_executor_backend"] == "external_launcher"


@pytest.mark.L0
@pytest.mark.CPU
def test_runtime_preflight_rejects_incompatible_opd_settings() -> None:
    """Model identity and likelihood semantics are checked together before runtime setup."""
    dpca = OPDConfig(
        distillation_mode="advantage",
        jsd_beta=None,
        teacher=TeacherConfig(input_mode="cross_token_text", signal_mode="dpca", top_k=0),
    )
    validate_opd_runtime(dpca, RolloutConfig())
    rollout = RolloutConfig()
    validate_opd_runtime(
        dpca,
        rollout,
        model_backend="qwen3_vl",
        model_config=Qwen3VLModelConfig(name_or_path=rollout.name_or_path),
    )
    with pytest.raises(ValueError, match="DPCA requires model_backend='i4'"):
        validate_opd_runtime(dpca, RolloutConfig(), model_backend="hf")
    for invalid_rollout, field_name in (
        (RolloutConfig(temperature=0.8), "temperature"),
        (RolloutConfig(top_p=0.9), "top_p"),
        (RolloutConfig(top_k=8), "top_k"),
        (RolloutConfig(repetition_penalty=1.1), "repetition_penalty"),
    ):
        with pytest.raises(ValueError, match=rf"DPCA requires rollout\.{field_name}"):
            validate_opd_runtime(dpca, invalid_rollout)
    with pytest.raises(ValueError, match="policy and rollout must start from the same model/tokenizer artifact"):
        validate_opd_runtime(
            OPDConfig(),
            RolloutConfig(name_or_path="Qwen/Qwen3-VL-8B-Instruct"),
            model_backend="qwen3_vl",
            model_config=Qwen3VLModelConfig(name_or_path="Qwen/Qwen3-VL-4B-Instruct"),
        )

    student = "Qwen/Qwen3-VL-8B-Instruct"
    validate_opd_runtime(
        OPDConfig(
            distillation_mode="direct",
            teacher=TeacherConfig(name_or_path="Qwen/Qwen3-VL-235B-A22B-Instruct", top_k=8),
        ),
        RolloutConfig(name_or_path=student),
        model_backend="qwen3_vl",
        model_config=Qwen3VLModelConfig(name_or_path=student),
    )

    with pytest.raises(ValueError, match="hf policy and rollout must start from the same model/tokenizer artifact"):
        validate_opd_runtime(
            OPDConfig(),
            RolloutConfig(name_or_path="Qwen/Qwen2.5-0.5B-Instruct"),
            model_backend="hf",
            model_config=HFModelConfig(name_or_path="Qwen/Qwen2.5-1.5B-Instruct"),
        )
    validate_opd_runtime(OPDConfig(), RolloutConfig())
    with pytest.raises(ValueError, match="scoring_temperature == rl.temperature"):
        validate_opd_runtime(
            OPDConfig(teacher=TeacherConfig(scoring_temperature=0.5)),
            RolloutConfig(),
        )
    with pytest.raises(ValueError, match="unknown model_backend"):
        validate_opd_runtime(OPDConfig(), RolloutConfig(), model_backend="reasner")


@pytest.mark.L0
@pytest.mark.CPU
def test_native_qwen3_vl_builder_dispatches_from_resolved_model_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The config path selects the direct native builder, never posttrain's generic HFModel."""
    captured: dict[str, object] = {}
    sentinel = object()

    resolved_config = SimpleNamespace(
        model_type="qwen3_vl",
        source="Qwen/Qwen3-VL-8B-Instruct",
        tie_word_embeddings=False,
    )
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda _path: resolved_config)

    def _build(**kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr("cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_policy", _build)
    cfg = Qwen3VLModelConfig(
        name_or_path="Qwen/Qwen3-VL-8B-Instruct",
        checkpoint_path="/weights",
        activation_checkpointing="none",
    )

    assert build_qwen3_vl_model(cfg, None) is sentinel
    assert captured["model_type"] == "qwen3_vl"
    assert captured["checkpoint_path"] == "/weights"
    assert captured["model_config"] is resolved_config
    hooks = captured["hooks"]
    assert hooks.prepare_model is not None
    assert hooks.parallelize is not None


@pytest.mark.L0
@pytest.mark.CPU
def test_native_qwen3_vl_builder_validates_resolved_model_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 235B MoE checkpoint cannot be meta-initialized accidentally as the dense family."""
    resolved_config = SimpleNamespace(model_type="qwen3_vl_moe", tie_word_embeddings=False)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", lambda _path: resolved_config)
    with pytest.raises(ValueError, match="dtype must be 'bfloat16'"):
        build_qwen3_vl_model(
            Qwen3VLModelConfig(
                name_or_path="Qwen/Qwen3-VL-235B-A22B-Instruct",
                dtype="float32",
            ),
            None,
        )

    resolved_config.model_type = "unsupported_vlm"
    with pytest.raises(ValueError, match=r"model_type qwen3_vl\|qwen3_vl_moe"):
        build_qwen3_vl_model(Qwen3VLModelConfig(name_or_path="unsupported/model"), None)


@pytest.mark.L0
@pytest.mark.CPU
def test_generic_native_qwen_builder_injects_the_single_model_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    captured: dict[str, object] = {}

    def _build(cfg: Qwen3VLModelConfig, pdims: object | None = None) -> object:
        captured.update(cfg=cfg, pdims=pdims)
        return sentinel

    monkeypatch.setattr("cosmos_framework.posttrain.model.qwen3_vl.build_qwen3_vl_model", _build)
    cfg = Qwen3VLModelConfig()
    pdims = object()

    assert build_qwen3_vl_native_policy("Qwen/Qwen3-VL-8B-Instruct", cfg, pdims) is sentinel
    assert cfg.name_or_path == "Qwen/Qwen3-VL-8B-Instruct"
    assert captured == {"cfg": cfg, "pdims": pdims}

    with pytest.raises(ValueError, match="must match"):
        build_qwen3_vl_native_policy(
            "Qwen/Qwen3-VL-8B-Instruct",
            Qwen3VLModelConfig(name_or_path="Qwen/Qwen3-VL-4B-Instruct"),
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_effective_model_id_prefers_native_qwen3_vl_config() -> None:
    """Rollout, packer, reward, and policy resolve one student identity on the native path."""
    config = type(
        "Config",
        (),
        {
            "model_backend": "qwen3_vl",
            "model_qwen3_vl": Qwen3VLModelConfig(name_or_path="Qwen/Qwen3-VL-8B-Instruct"),
            "model_hf": type("HF", (), {"name_or_path": "wrong-hf"})(),
            "rollout": RolloutConfig(name_or_path="wrong-rollout"),
        },
    )()

    assert effective_model_id(config) == "Qwen/Qwen3-VL-8B-Instruct"


@pytest.mark.L0
@pytest.mark.CPU
def test_distillation_mode_selects_the_objective_and_carries_the_knobs() -> None:
    """`build_opd_objective` is the one config→objective mapping; the modes differ in how the
    divergence enters the loss, not in what it is."""
    pytest.importorskip("torch")
    from cosmos_framework.posttrain.algorithm.primitives.objective import Reduction

    packer = object()  # never called: construction only

    direct = build_opd_objective(OPDConfig(distillation_mode="direct"), packer)
    assert direct.reduction is Reduction.TOKEN_SUM  # loss_type defaults to token-mean
    assert direct.identity["distillation_mode"] == "direct"
    assert direct.identity["divergence"] == "topk_plus_other"
    assert direct.identity["temperature"] == 1.0

    advantage = build_opd_objective(OPDConfig(distillation_mode="advantage"), packer)
    assert advantage.reduction is Reduction.TOKEN_SUM
    assert advantage.identity["distillation_mode"] == "advantage"
    assert advantage.identity["divergence"] == "topk_slice"

    dpca = build_opd_objective(
        OPDConfig(
            distillation_mode="advantage",
            jsd_beta=None,
            teacher=TeacherConfig(input_mode="cross_token_text", signal_mode="dpca", top_k=0),
        ),
        packer,
    )
    assert dpca.reduction is Reduction.TOKEN_SUM
    assert dpca.identity["teacher_credit"] == "dpca"
    assert "jsd_beta" not in dpca.identity

    seq_mean = build_opd_objective(OPDConfig(loss_type="seq-mean-token-sum"), packer)
    assert seq_mean.reduction is Reduction.TOKEN_SUM  # sequence count is its DP-global normalization unit


@pytest.mark.L0
@pytest.mark.CPU
def test_direct_opd_does_not_need_the_determinism_attestation():
    """`distillation_mode="direct"` forms no importance ratio, so it never reads a cached π_old.

    The attestation exists for the ahead pass — π_old recorded under `eval()`, `cur` scored under
    `train()` — and the driver already skips that pass for direct OPD via `needs_old_logprobs()`.
    Demanding the attestation anyway rejected valid mini-batched direct-OPD configs at construction
    for a hazard they cannot have.
    """
    from cosmos_framework.posttrain.config.rl import OPDConfig

    cfg = OPDConfig()
    cfg.distillation_mode = "direct"
    cfg.batch_per_replica = 8
    cfg.mini_batch_per_replica = 2  # n_mini = 4 > 1
    with pytest.warns(UserWarning, match="does not use task reward"):
        cfg.__post_init__()  # must not raise
    assert cfg.batch_per_replica // cfg.mini_batch_per_replica == 4


@pytest.mark.L0
@pytest.mark.CPU
def test_advantage_opd_still_needs_the_determinism_attestation():
    """The ADVANTAGE objective does form a ratio against a frozen π_old, so the gate must hold."""
    from cosmos_framework.posttrain.config.rl import OPDConfig

    cfg = OPDConfig()
    cfg.distillation_mode = "advantage"
    cfg.batch_per_replica = 8
    cfg.mini_batch_per_replica = 2
    with pytest.raises(ValueError, match="certified_deterministic_forward"):
        cfg.__post_init__()
    cfg.certified_deterministic_forward = True
    cfg.__post_init__()  # attested -> accepted


@pytest.mark.L0
@pytest.mark.CPU
def test_advantage_opd_actually_carries_the_cache_its_gate_promises():
    """The attestation gate and the objective must agree about whether an ahead pass exists.

    `OPDConfig._objective_consults_pi_old()` returns True for `advantage`, so a config with
    `n_mini > 1` is refused until the operator attests the forward is deterministic. That attestation
    only buys anything if an ahead pass then runs -- and it runs only when the Objective carries an
    `old_logp_cache`, which is what `LLMTrainer.needs_old_logprobs()` reads.

    Shipped, it did not: the objective passed `current.detach()` and built its `Objective` without a
    cache. An operator who set `certified_deterministic_forward=true` got the gate's demand and none
    of its protection -- n_mini unconstrained steps per batch, the exact hazard being guarded.

    `direct` stays exempt: it scores the target policy outright, forms no ratio, and its gate returns
    False, so it must NOT carry a cache either.
    """
    from unittest.mock import MagicMock

    from cosmos_framework.posttrain.algorithm.opd import make_advantage_objective, make_direct_objective
    from cosmos_framework.posttrain.config.rl import OPDConfig

    advantage_cfg = OPDConfig()
    advantage_cfg.distillation_mode = "advantage"
    direct_cfg = OPDConfig()
    direct_cfg.distillation_mode = "direct"

    assert advantage_cfg._objective_consults_pi_old() is True
    assert make_advantage_objective(MagicMock(), kl_beta=0.0).old_logp_cache is not None, (
        "advantage mode demands the determinism attestation, so it must actually freeze pi_old"
    )

    assert direct_cfg._objective_consults_pi_old() is False
    assert make_direct_objective(MagicMock()).old_logp_cache is None, (
        "direct mode forms no ratio; a cache here would make needs_old_logprobs() lie the other way"
    )
