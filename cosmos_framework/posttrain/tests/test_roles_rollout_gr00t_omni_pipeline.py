# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""The out-of-tree GR00T pipeline: registration, the seed guard, and the stage-3 weight sink.

L1, not L0: importing this module pulls in ``vllm`` and ``vllm_omni`` (~48 s), past the L0 budget
before a single assertion runs. No engine is started here.
"""

from __future__ import annotations

import pytest


@pytest.mark.L1
@pytest.mark.CPU
def test_registration_is_idempotent_and_returns_the_arch_name() -> None:
    """The rollout body may construct more than once per process; a second registration must not fight."""
    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import (
        MODEL_ARCH,
        register_gr00t_rl_pipeline,
    )

    assert register_gr00t_rl_pipeline() == MODEL_ARCH
    assert register_gr00t_rl_pipeline() == MODEL_ARCH


@pytest.mark.L1
@pytest.mark.CPU
def test_registered_arch_resolves_to_this_class() -> None:
    """`_try_load_model_cls` is the exact call the engine makes to turn the arch name into a class."""
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import (
        MODEL_ARCH,
        Gr00tN1d7RLPipeline,
        register_gr00t_rl_pipeline,
    )

    register_gr00t_rl_pipeline()
    assert DiffusionModelRegistry._try_load_model_cls(MODEL_ARCH) is Gr00tN1d7RLPipeline


@pytest.mark.L1
@pytest.mark.CPU
def test_process_level_noise_seed_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """GR00T_NOISE_SEED silently overrides the per-request seed; the guard runs before any model loading."""
    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import Gr00tN1d7RLPipeline

    monkeypatch.setenv("GR00T_NOISE_SEED", "1234")
    with pytest.raises(RuntimeError, match="GR00T_NOISE_SEED"):
        Gr00tN1d7RLPipeline(od_config=None)


def _pipeline_with_head():
    import torch

    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import Gr00tN1d7RLPipeline

    head = torch.nn.Module()
    head.w = torch.nn.Parameter(torch.zeros(2, 3))
    head.b = torch.nn.Parameter(torch.zeros(4))
    pipeline = object.__new__(Gr00tN1d7RLPipeline)
    pipeline.policy = type("Policy", (), {"model": type("Model", (), {"action_head": head})()})()
    return pipeline, head


@pytest.mark.L1
@pytest.mark.CPU
def test_load_weights_copies_by_wire_name_and_returns_exactly_what_it_loaded() -> None:
    """The sink: `action_head.*` names land on the live parameters; the return is the vLLM `set[str]` contract."""
    import torch

    pipeline, head = _pipeline_with_head()
    new_w = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert pipeline.load_weights([("action_head.w", new_w)]) == {"action_head.w"}
    assert torch.equal(head.w.detach(), new_w)
    assert torch.equal(head.b.detach(), torch.zeros(4))  # an unmentioned parameter is left alone
    assert pipeline.load_weights(iter([("action_head.b", torch.ones(4))])) == {"action_head.b"}


@pytest.mark.L1
@pytest.mark.CPU
@pytest.mark.parametrize(
    "name, tensor_fn, error, match",
    [
        ("action_head.nope", lambda t: t, KeyError, "not a live action-head parameter"),
        ("backbone.w", lambda t: t, KeyError, "outside the 'action_head.' wire scope"),
        ("action_head.w", lambda t: t.reshape(3, 2), ValueError, "shape"),
        ("action_head.w", lambda t: t.to(t.dtype if False else __import__("torch").bfloat16), ValueError, "dtype"),
    ],
)
def test_load_weights_rejects_the_whole_bucket_and_touches_nothing(name, tensor_fn, error, match) -> None:
    """A partial or coerced load would be a silent weight change, so the bucket is validated before any copy."""
    import torch

    pipeline, head = _pipeline_with_head()
    good = ("action_head.b", torch.ones(4))
    with pytest.raises(error, match=match):
        pipeline.load_weights([good, (name, tensor_fn(torch.arange(6, dtype=torch.float32).reshape(2, 3)))])
    assert torch.equal(head.b.detach(), torch.zeros(4)), "a rejected bucket must not have copied its valid entries"


@pytest.mark.L1
@pytest.mark.CPU
def test_load_weights_accepts_the_empty_startup_call() -> None:
    """The engine's model-load path calls `load_weights` with nothing to load; that must succeed."""
    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import Gr00tN1d7RLPipeline

    pipeline = object.__new__(Gr00tN1d7RLPipeline)
    assert pipeline.load_weights([]) == set()
    assert pipeline.load_weights(iter([])) == set()


@pytest.mark.L1
@pytest.mark.CPU
def test_seeded_forward_is_reproducible_and_restores_global_rng(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real ``forward`` must make the noise a function of the seed and leave the global RNG alone.

    The base class's ``forward`` needs a loaded checkpoint, so it is replaced with a stand-in that
    performs GR00T's unseeded ``torch.randn`` draw; everything else is this class's own code.
    """
    import torch
    from vllm_omni.diffusion.models.gr00t.pipeline_gr00t import Gr00tN1d7Pipeline

    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import Gr00tN1d7RLPipeline

    monkeypatch.setattr(Gr00tN1d7Pipeline, "forward", lambda self, req, **kwargs: torch.randn(4))
    pipeline = object.__new__(Gr00tN1d7RLPipeline)
    pipeline.ppo = False

    class _Req:
        def __init__(self, seed):
            self.sampling_params = type("SP", (), {"extra_args": ({"seed": seed} if seed is not None else {})})()

    assert torch.equal(pipeline.forward(_Req(7)), pipeline.forward(_Req(7))), "same seed must give the same noise"
    assert not torch.equal(pipeline.forward(_Req(7)), pipeline.forward(_Req(8))), "different seeds must differ"

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99)
        expected = torch.randn(4)
        torch.manual_seed(99)
        pipeline.forward(_Req(7))  # must not consume from or reseed the global stream
        assert torch.equal(torch.randn(4), expected), "seeding one request perturbed the global RNG"
        # An unseeded request falls through to the base forward and draws from the global stream.
        torch.manual_seed(99)
        assert torch.equal(pipeline.forward(_Req(None)), expected)


@pytest.mark.L1
@pytest.mark.CPU
def test_ppo_records_survive_the_real_omni_output_formatter() -> None:
    from types import SimpleNamespace

    import numpy as np
    import torch
    from vllm_omni.diffusion.output_formatter import (
        DiffusionStepTimings,
        format_diffusion_outputs,
        normalize_diffusion_postprocess_output,
    )
    from vllm_omni.diffusion.request import OmniDiffusionRequest
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    from cosmos_framework.posttrain.configs.gr00t_n1d7_libero_ppo import make_model_config
    from cosmos_framework.posttrain.model.gr00t.ppo import (
        PREFIX_SAMPLER_IDENTITY,
        Gr00tPPOCondition,
        Gr00tPPOSample,
        Gr00tPPOScores,
        Gr00tPPOTrace,
    )
    from cosmos_framework.posttrain.model.gr00t.serving.body import Gr00tPolicyBody
    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import Gr00tN1d7RLPipeline

    generator = torch.Generator().manual_seed(29)
    features = torch.randn((2, 3, 4), generator=generator)
    condition = Gr00tPPOCondition(
        features,
        torch.ones((2, 3), dtype=torch.bool),
        torch.ones((2, 3), dtype=torch.bool),
        torch.zeros((2, 1, 132)),
        torch.zeros(2, dtype=torch.long),
    )
    latent = torch.randn((2, 40, 132), generator=generator)
    trace = Gr00tPPOTrace(
        latent, latent + 0.1, torch.zeros(2, dtype=torch.long), torch.zeros(2), PREFIX_SAMPLER_IDENTITY, 8
    )
    scores = Gr00tPPOScores(torch.tensor([-11.0, -13.0]), torch.tensor([0.4, 0.8]))
    sample = Gr00tPPOSample(latent, trace, scores, torch.zeros((2, 8, 7)))
    keys = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    expected_actions = np.linspace(-1, 1, len(keys), dtype=np.float32)
    decoded = {
        key: np.full((2, 16, 1), expected_actions[index], dtype=np.float32)
        for index, key in reversed(list(enumerate(keys)))
    }

    def decode(normalized, states):
        assert normalized.shape == (2, 40, 132)
        return decoded

    pipeline = object.__new__(Gr00tN1d7RLPipeline)
    pipeline.policy = SimpleNamespace(decode=decode, modality_configs={"action": SimpleNamespace(modality_keys=keys)})
    pipeline.ppo_model = SimpleNamespace(config=make_model_config("test", 8))
    actions = pipeline._decode_ppo_actions(latent, None)
    wire = Gr00tN1d7RLPipeline._ppo_output(actions, condition, sample)
    request = OmniDiffusionRequest(["test"], OmniDiffusionSamplingParams(seed=0), "ppo-format")
    result = format_diffusion_outputs(
        request=request,
        od_config=SimpleNamespace(model_class_name="Gr00tN1d7Pipeline"),
        diffusion_output=wire,
        output_data=wire.output,
        postprocess_output=normalize_diffusion_postprocess_output(wire.output, wire.custom_output),
        timings=DiffusionStepTimings(0, 0, 0, 0),
    )[0]
    body = object.__new__(Gr00tPolicyBody)
    body.ppo = True
    body.config = pipeline.ppo_model.config
    body.modality = {"action_keys": keys}
    records = body._parse_output(result, 2)
    for index, record in enumerate(records):
        torch.testing.assert_close(record["condition"].backbone_features, features)
        torch.testing.assert_close(record["trace"].latent_after, trace.latent_after)
        assert record["sample_index"] == index
        assert record["old_logprob"] == float(scores.logprobs[index])
        assert record["old_value"] == float(scores.values[index])
        np.testing.assert_array_equal(record["action"], np.broadcast_to(expected_actions, (8, 7)))
    for key in keys:
        decoded[key] = decoded[key][:, :8]
    with pytest.raises(ValueError, match="processor must decode"):
        pipeline._decode_ppo_actions(latent, None)


@pytest.mark.L1
@pytest.mark.CPU
def test_worker_extension_is_importable_by_the_qualname_the_body_sends() -> None:
    """The body passes `worker_extension_cls` as a STRING; resolve it here the way vLLM-Omni does."""
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from cosmos_framework.posttrain.model.gr00t.serving.body import WORKER_EXTENSION_CLS
    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import Gr00tOmniWorkerExtension

    assert resolve_obj_by_qualname(WORKER_EXTENSION_CLS) is Gr00tOmniWorkerExtension


@pytest.mark.L1
@pytest.mark.CPU
def test_importing_the_pipeline_module_registers_it() -> None:
    """Registration must be an IMPORT SIDE EFFECT: spawned workers start with a pristine registry."""
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import MODEL_ARCH, Gr00tN1d7RLPipeline

    # No explicit register call here: the import above must have been sufficient.
    assert DiffusionModelRegistry._try_load_model_cls(MODEL_ARCH) is Gr00tN1d7RLPipeline


@pytest.mark.L1
@pytest.mark.CPU
def test_constants_restated_on_the_light_import_path_agree() -> None:
    """The body restates two pipeline-module constants to stay vLLM-Omni-free; pin them together."""
    from cosmos_framework.posttrain.model.gr00t.serving.body import ACTION_HEAD_PREFIX as light_prefix
    from cosmos_framework.posttrain.model.gr00t.serving.body import MODEL_ARCH_NAME as light_arch
    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import ACTION_HEAD_PREFIX as heavy_prefix
    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import MODEL_ARCH as heavy_arch

    assert light_prefix == heavy_prefix
    assert light_arch == heavy_arch


@pytest.mark.L1
@pytest.mark.CPU
def test_worker_extension_prefixes_names_to_match_trainer_wire_scope() -> None:
    """Worker-reported names must compare 1:1 with the trainer's `net.state_dict()` keys."""
    import torch

    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import (
        ACTION_HEAD_PREFIX,
        Gr00tOmniWorkerExtension,
    )

    head = torch.nn.Module()
    head.weight = torch.nn.Parameter(torch.zeros(2, 3))

    class _Runner:
        pipeline = type("P", (), {"policy": type("Q", (), {"model": type("M", (), {"action_head": head})()})()})()

    ext = Gr00tOmniWorkerExtension()
    ext.model_runner = _Runner()
    described = ext.describe_action_head()
    assert set(described) == {f"{ACTION_HEAD_PREFIX}weight"}
    assert described[f"{ACTION_HEAD_PREFIX}weight"]["shape"] == [2, 3]


@pytest.mark.L1
@pytest.mark.CPU
def test_denoising_steps_override_reaches_the_live_action_head() -> None:
    """The recipe's step count lands on the attribute the Euler loop reads; unset leaves the checkpoint's."""
    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import Gr00tN1d7RLPipeline

    head = type("Head", (), {"num_inference_timesteps": 4})()
    pipeline = object.__new__(Gr00tN1d7RLPipeline)
    pipeline.policy = type("Policy", (), {"model": type("Model", (), {"action_head": head})()})()

    pipeline._denoising_steps = 7
    pipeline._apply_denoising_steps()
    assert head.num_inference_timesteps == 7
    pipeline._denoising_steps = None
    pipeline._apply_denoising_steps()
    assert head.num_inference_timesteps == 7


@pytest.mark.L1
@pytest.mark.CPU
def test_describe_pipeline_reports_the_live_step_count_and_artifact() -> None:
    """The body pins both per recipe and checks them at startup, so they must come off the live objects."""
    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import (
        Gr00tN1d7RLPipeline,
        Gr00tOmniWorkerExtension,
    )

    head = type("Head", (), {"num_inference_timesteps": 6})()
    transform = type("Transform", (), {"name": "reference-albumentations"})()
    processor = type("Processor", (), {"eval_image_transform": transform})()
    pipeline = object.__new__(Gr00tN1d7RLPipeline)
    pipeline.policy = type(
        "Policy", (), {"model": type("Model", (), {"action_head": head})(), "processor": processor}
    )()
    pipeline.model_path = "/hf/snapshots/deadbeef"

    ext = Gr00tOmniWorkerExtension()
    ext.model_runner = type("Runner", (), {"pipeline": pipeline})()
    described = ext.describe_pipeline()
    assert described["is_rl_pipeline"] is True
    assert described["image_transform"] == "reference-albumentations"
    assert described["num_inference_timesteps"] == 6
    assert described["model_path"] == "/hf/snapshots/deadbeef"
    import os

    assert described["worker_pid"] == os.getpid()  # a hot update keeps this; a rebuild changes it


@pytest.mark.L1
@pytest.mark.CPU
def test_worker_extension_is_the_generic_weight_plane_plus_gr00t_naming(monkeypatch: pytest.MonkeyPatch) -> None:
    """The export, the update bracket and the digests are the generic extension's; GR00T only names its live
    parameters in wire naming."""
    import torch

    from cosmos_framework.posttrain.model.gr00t.serving.omni_pipeline import (
        ACTION_HEAD_PREFIX,
        Gr00tOmniWorkerExtension,
    )
    from cosmos_framework.posttrain.roles.rollout.vllm_omni_worker import OmniIpcWorkerExtension

    assert issubclass(Gr00tOmniWorkerExtension, OmniIpcWorkerExtension)
    for method in (
        "export_live_parameters",
        "begin_weight_update",
        "finish_weight_update",
        "digest_live_parameters",
        "describe_device",
    ):
        assert getattr(Gr00tOmniWorkerExtension, method) is getattr(OmniIpcWorkerExtension, method)

    head = torch.nn.Module()
    head.weight = torch.nn.Parameter(torch.ones(2, 3))
    ext = Gr00tOmniWorkerExtension()
    ext.model_runner = type(
        "Runner",
        (),
        {"pipeline": type("P", (), {"policy": type("Q", (), {"model": type("M", (), {"action_head": head})()})()})()},
    )()
    assert [name for name, _p in ext._live_named_parameters()] == [f"{ACTION_HEAD_PREFIX}weight"]
    monkeypatch.setenv("GR00T_DIT_SDPA_MODE", "math")
    described = ext.describe_device()
    assert described["sdpa_mode"] == "math" and described["pid"] == __import__("os").getpid()
