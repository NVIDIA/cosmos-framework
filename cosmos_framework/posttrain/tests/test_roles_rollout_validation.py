# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""@L0 @CPU: validation-only rollout context and pinned weight-version contract."""

from __future__ import annotations

import copy
import random
import sys
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cosmos_framework.posttrain.roles.rollout.hf import HFRollout
from cosmos_framework.posttrain.roles.rollout.validation import (
    ValidationRolloutContext,
    validation_generation_seed,
)
from cosmos_framework.posttrain.roles.rollout.vllm import VLLMRollout
from cosmos_framework.posttrain.schema import ObsBundle
from cosmos_framework.posttrain.utils.identity import json_identity_equal, normalize_json_dict, project_typed_json_value


class _StringTokenizer:
    name_or_path = "fake-tokenizer"
    model_max_length = 128
    padding_side = "right"
    truncation_side = "right"
    chat_template = "{{ messages }}"
    eos_token_id = 2
    pad_token_id = 2

    @staticmethod
    def __call__(text: str, **_kwargs: object) -> object:
        return SimpleNamespace(input_ids=[len(text)])

    @staticmethod
    def get_vocab() -> dict[str, int]:
        return {"<eos>": 2, "token": 3}


class _Packer:
    tokenizer: _StringTokenizer = _StringTokenizer()

    @staticmethod
    def adds_special_tokens() -> bool:
        return False

    @staticmethod
    def get_rollout_input(obs: ObsBundle) -> str:
        return str(obs.representations["prompt"])


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_validation_uses_its_sampling_and_omits_training_logprobs(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_contexts = (
        ({"model_step": True}, "model_step"),
        ({"temperature": float("nan")}, "temperature"),
        ({"top_p": True}, "top_p"),
        ({"top_k": 0.5}, "top_k"),
        ({"seed": True}, "seed"),
    )
    for kwargs, field in invalid_contexts:
        with pytest.raises(ValueError, match=field):
            ValidationRolloutContext(**{"model_step": 0, **kwargs})

    captured: list[dict[str, object]] = []

    class _SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            captured.append(kwargs)

    fake_vllm = ModuleType("vllm")
    fake_vllm.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setenv("POSTTRAIN_REPLICA_INDEX", "0")
    monkeypatch.setenv("POSTTRAIN_REPLICA_COUNT", "1")

    output = SimpleNamespace(token_ids=[7, 8], text="ok", logprobs=None)
    request = SimpleNamespace(prompt_token_ids=[1, 2], outputs=[output])

    class _Engine:
        @staticmethod
        def generate(prompts: list[dict[str, list[int]]], params: object) -> list[object]:
            assert isinstance(params, list)
            assert prompts == [{"prompt_token_ids": [8]}, {"prompt_token_ids": [8]}]
            return [request, request]

    rollout = VLLMRollout(
        _Engine(),
        packer=_Packer(),
        sampling={"temperature": 0.9, "top_p": 0.8, "max_tokens": 64},
    )
    context = ValidationRolloutContext(
        model_step=0,
        temperature=0.0,
        top_k=1,
        n_generation=2,
        max_tokens=5,
        seed=11,
    )
    trajectories = rollout.generate(
        [ObsBundle(representations={"prompt": "question"}, id="sample")],
        validation=context,
    )

    assert captured == [
        {
            "n": 1,
            "logprobs": None,
            "temperature": 0.0,
            "max_tokens": 5,
            "top_k": 1,
            "repetition_penalty": 1.0,
            "seed": validation_generation_seed(11, "sample", generation_index),
        }
        for generation_index in range(2)
    ]
    assert len(trajectories) == 2
    assert all("rollout_logprobs" not in t.transitions[-1].policy_info for t in trajectories)
    assert all(t.transitions[-1].policy_info["validation_step"] == 0 for t in trajectories)
    assert all(t.transitions[-1].policy_info["completion_token_count"] == 2 for t in trajectories)


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_validation_pretokenizes_only_prompt_and_preserves_multimodal_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_vllm = ModuleType("vllm")
    fake_vllm.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    image = object()
    tokenizer_calls: list[tuple[str, bool]] = []
    request_identities: list[dict[str, object]] = []

    def _capture_request_identity(value: dict[str, object], *, context: str) -> None:
        assert context == "vLLM validation request identity"
        request_identities.append(value)

    monkeypatch.setattr(
        "cosmos_framework.posttrain.roles.rollout.vllm._require_identical_rank_identity",
        _capture_request_identity,
    )

    class _TokenizerSpy:
        @staticmethod
        def __call__(text: str, *, add_special_tokens: bool) -> object:
            tokenizer_calls.append((text, add_special_tokens))
            return SimpleNamespace(input_ids=[31, 32])

    class _MultimodalPacker:
        tokenizer = _TokenizerSpy()

        @staticmethod
        def adds_special_tokens() -> bool:
            return False

        @staticmethod
        def get_rollout_input(_obs: ObsBundle) -> dict[str, object]:
            return {
                "prompt": "<rendered-image-prompt>",
                "multi_modal_data": {"image": [image]},
                "multi_modal_uuids": {"image": ["image-content-hash"]},
                "mm_processor_kwargs": {"max_pixels": 1024},
            }

    class _Engine:
        @staticmethod
        def generate(prompts: list[dict[str, object]], params: list[object]) -> list[object]:
            assert len(prompts) == len(params) == 2
            for prompt in prompts:
                assert "prompt" not in prompt
                assert prompt["prompt_token_ids"] == [31, 32]
                assert prompt["multi_modal_data"] == {"image": [image]}
                assert prompt["multi_modal_uuids"] == {"image": ["image-content-hash"]}
                assert prompt["mm_processor_kwargs"] == {"max_pixels": 1024}
            output = SimpleNamespace(token_ids=[7], text="ok", logprobs=None)
            return [
                SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], outputs=[output]) for prompt in prompts
            ]

    obs = ObsBundle(
        representations={
            "messages": [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what?"}]}],
            "images": [image],
        },
        modality_tags=("text", "image"),
        id="image-sample",
    )
    rollout = VLLMRollout(
        _Engine(),
        packer=_MultimodalPacker(),
        sampling={"temperature": 0.9, "max_tokens": 4},
    )

    trajectories = rollout.generate(
        [obs],
        validation=ValidationRolloutContext(model_step=0, n_generation=2, seed=13),
    )

    assert tokenizer_calls == [("<rendered-image-prompt>", True)]
    assert len(trajectories) == 2
    identity_requests = request_identities[0]["requests"]
    assert isinstance(identity_requests, list)
    assert [request["multi_modal_uuids"] for request in identity_requests] == [
        {"image": ["image-content-hash"]},
        {"image": ["image-content-hash"]},
    ]


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_validation_accepts_pretokenized_request_and_rejects_ambiguous_prompt() -> None:
    image = object()

    class _TokenizerMustNotRun:
        @staticmethod
        def __call__(_text: str, **_kwargs: object) -> object:
            raise AssertionError("pretokenized validation requests must not invoke the tokenizer")

    class _PretokenizedPacker:
        tokenizer = _TokenizerMustNotRun()

        @staticmethod
        def adds_special_tokens() -> bool:
            return False

        @staticmethod
        def get_rollout_input(_obs: ObsBundle) -> dict[str, object]:
            return {
                "prompt_token_ids": (41, 42),
                "multi_modal_data": {"image": [image]},
                "multi_modal_uuids": {"image": ["image-content-hash"]},
            }

    obs = ObsBundle(representations={"prompt": "unused", "image": image}, id="pretokenized-image")
    rollout = VLLMRollout(object(), packer=_PretokenizedPacker())

    assert rollout._prepare_validation_request(obs) == {
        "prompt_token_ids": [41, 42],
        "multi_modal_data": {"image": [image]},
        "multi_modal_uuids": {"image": ["image-content-hash"]},
    }

    class _AmbiguousPacker(_PretokenizedPacker):
        @staticmethod
        def get_rollout_input(_obs: ObsBundle) -> dict[str, object]:
            return {"prompt": "question", "prompt_token_ids": [41, 42]}

    ambiguous = VLLMRollout(object(), packer=_AmbiguousPacker())
    with pytest.raises(RuntimeError, match="exactly one of prompt or prompt_token_ids"):
        ambiguous._prepare_validation_request(obs)


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_validation_identity_binds_resolved_model_and_live_tokenizer_semantics() -> None:
    assert not json_identity_equal({"value": True}, {"value": 1}, context="identity")
    assert not json_identity_equal({"value": 1}, {"value": 1.0}, context="identity")
    with pytest.raises(TypeError, match="non-string mapping key"):
        normalize_json_dict({"nested": {1: "value"}}, context="identity")

    added_token = SimpleNamespace(
        content="<extra>", single_word=False, lstrip=False, rstrip=False, normalized=False, special=True
    )
    typed_keys = [project_typed_json_value({key: added_token}, context="metadata") for key in ("1", 1, True)]
    assert len({str(value) for value in typed_keys}) == 3
    for key in (float("nan"), float("inf"), object()):
        with pytest.raises(TypeError, match="mapping key"):
            project_typed_json_value({key: "value"}, context="metadata")

    common = {
        "packer": _Packer(),
        "sampling": {"temperature": 0.0, "max_tokens": 8},
        "model_identifier": "/models/current",
        "engine_config": {"model": "/models/current", "dtype": "bfloat16"},
    }
    first = VLLMRollout(
        object(),
        resolved_model_config={"model_type": "qwen3", "rope_scaling": {"factor": 1.0}},
        **common,
    ).validation_semantics_identity()
    changed = VLLMRollout(
        object(),
        resolved_model_config={"model_type": "qwen3", "rope_scaling": {"factor": 2.0}},
        **common,
    ).validation_semantics_identity()

    assert first != changed
    assert first["model_identifier"] == "/models/current"
    assert first["tokenizer"]["vocab_sha256"]
    assert first["tokenizer"]["settings"]["chat_template"] == "{{ messages }}"
    assert first["packer"]["source_sha256"]


class _Encoding(dict[str, torch.Tensor]):
    def to(self, _device: torch.device) -> _Encoding:
        return self


class _Tokenizer:
    name_or_path = "fake-hf-tokenizer"
    model_max_length = 128
    padding_side = "right"
    truncation_side = "right"
    chat_template = "{{ messages }}"
    pad_token_id = 99
    eos_token_id = 99

    @staticmethod
    def get_vocab() -> dict[str, int]:
        return {"<eos>": 99, "token": 1}

    @staticmethod
    def __call__(text: str, **kwargs: object) -> object:
        token_id = len(text) + 1
        if kwargs.get("return_tensors") == "pt":
            return _Encoding(input_ids=torch.tensor([[token_id]], dtype=torch.long))  # [1,1]
        return SimpleNamespace(input_ids=[token_id])

    @staticmethod
    def decode(token_ids: torch.Tensor, **_kwargs: object) -> str:  # token_ids: [T]
        return ",".join(str(int(token)) for token in token_ids)


class _HFPacker:
    tokenizer = _Tokenizer()

    @staticmethod
    def get_rollout_input(obs: ObsBundle) -> str:
        return str(obs.representations["prompt"])


class _RandomPacker:
    tokenizer = _Tokenizer()

    def __init__(self) -> None:
        self.rendered: list[tuple[str, str]] = []
        self.raise_next = False

    @staticmethod
    def adds_special_tokens() -> bool:
        return False

    def get_rollout_input(self, obs: ObsBundle) -> str:
        rendered = (
            f"{obs.representations['prompt']}|{random.random():.12f}|"
            f"{float(np.random.random()):.12f}|{float(torch.rand(())):.12f}"
        )
        if self.raise_next:
            self.raise_next = False
            raise RuntimeError("packer failed")
        self.rendered.append((obs.id, rendered))
        return rendered


def _rng_snapshot() -> tuple[object, tuple[object, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.random.get_rng_state()  # [RNG]


def _assert_rng_snapshot(
    actual: tuple[object, tuple[object, ...], torch.Tensor], expected: tuple[object, tuple[object, ...], torch.Tensor]
) -> None:
    actual_python, actual_numpy, actual_torch = actual
    expected_python, expected_numpy, expected_torch = expected
    assert actual_python == expected_python
    assert actual_numpy[0] == expected_numpy[0]
    assert np.array_equal(actual_numpy[1], expected_numpy[1])
    assert actual_numpy[2:] == expected_numpy[2:]
    assert torch.equal(actual_torch, expected_torch)


class _RandomHFModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))  # []
        self.frozen_child = torch.nn.Linear(1, 1)
        self.raise_next = False
        self.generate_calls = 0

    def generate(self, input_ids: torch.Tensor, **_kwargs: object) -> object:  # input_ids: [1,P]
        self.generate_calls += 1
        sampled = torch.randint(0, 50, (1, 1), device=input_ids.device)  # [1,1]
        if self.raise_next:
            self.raise_next = False
            raise RuntimeError("generation failed")
        sequences = torch.cat((input_ids, sampled), dim=1)  # [1,P+1]
        return SimpleNamespace(sequences=sequences, scores=())


def _actions_by_id(trajectories: list[object]) -> dict[str, list[int]]:
    return {trajectory.id: trajectory.transitions[0].action.tolist() for trajectory in trajectories}


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_validation_rng_is_sample_local_and_restores_rng_and_mode_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _RandomHFModel()
    model.train()
    model.frozen_child.eval()
    rollout = HFRollout(
        model,
        packer=_HFPacker(),
        sampling={"do_sample": True, "temperature": 1.0, "max_new_tokens": 1},
    )
    context = ValidationRolloutContext(model_step=0, temperature=1.0, n_generation=2, seed=17)
    prompts = [
        ObsBundle(representations={"prompt": "alpha"}, id="sample-a"),
        ObsBundle(representations={"prompt": "beta"}, id="sample-b"),
    ]

    torch.manual_seed(1234)
    rng_before = torch.random.get_rng_state()  # [RNG]
    batched = _actions_by_id(rollout.generate(prompts, validation=context))
    rng_after = torch.random.get_rng_state()  # [RNG]
    split = {
        **_actions_by_id(rollout.generate([prompts[1]], validation=context)),
        **_actions_by_id(rollout.generate([prompts[0]], validation=context)),
    }

    assert batched == split
    assert torch.equal(rng_before, rng_after)
    assert model.training is True
    assert model.frozen_child.training is False

    model.raise_next = True
    failure_rng_before = torch.random.get_rng_state()  # [RNG]
    with pytest.raises(RuntimeError, match="generation failed"):
        rollout.generate([prompts[0]], validation=context)
    assert torch.equal(failure_rng_before, torch.random.get_rng_state())
    assert model.training is True
    assert model.frozen_child.training is False

    def _forbid_global_seed(_seed: int) -> torch.Generator:
        raise AssertionError("validation must not seed every visible CUDA device")

    monkeypatch.setattr(torch, "manual_seed", _forbid_global_seed)
    rollout.generate([prompts[0]], validation=context)


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_validation_random_packer_is_sample_local_and_restores_all_rngs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_vllm = ModuleType("vllm")
    fake_vllm.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    output = SimpleNamespace(token_ids=[7], text="ok", logprobs=None)

    class _Engine:
        @staticmethod
        def generate(prompts: list[dict[str, list[int]]], params: object) -> list[object]:
            assert isinstance(params, list)
            return [
                SimpleNamespace(prompt_token_ids=prompt["prompt_token_ids"], outputs=[output]) for prompt in prompts
            ]

    packer = _RandomPacker()
    rollout = VLLMRollout(_Engine(), packer=packer, sampling={"temperature": 0.9, "max_tokens": 4})
    context = ValidationRolloutContext(model_step=0, temperature=0.0, n_generation=2, seed=31)
    prompts = [
        ObsBundle(representations={"prompt": "alpha"}, id="sample-a"),
        ObsBundle(representations={"prompt": "beta"}, id="sample-b"),
    ]

    random.seed(201)
    np.random.seed(202)
    torch.random.default_generator.manual_seed(203)
    before = _rng_snapshot()
    rollout.generate(prompts, validation=context)
    _assert_rng_snapshot(_rng_snapshot(), before)
    batched = dict(packer.rendered)

    packer.rendered.clear()
    rollout.generate(list(reversed(prompts)), validation=context)
    _assert_rng_snapshot(_rng_snapshot(), before)
    assert dict(packer.rendered) == batched

    packer.raise_next = True
    with pytest.raises(RuntimeError, match="packer failed"):
        rollout.generate([prompts[0]], validation=context)
    _assert_rng_snapshot(_rng_snapshot(), before)


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_validation_rejects_different_fsdp_rank_inputs_before_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _different_peer_input(state: object, *, context: str) -> dict[int, object]:
        assert context == "HF validation request identity preparation"
        peer = copy.deepcopy(state)
        assert isinstance(peer, dict)
        peer["model_inputs"]["input_ids"]["values"] = [[999]]
        return {0: state, 1: peer}

    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.rank_state.gather_rank_states",
        _different_peer_input,
    )
    model = _RandomHFModel()
    rollout = HFRollout(
        model,
        packer=_HFPacker(),
        sampling={"do_sample": True, "temperature": 1.0, "max_new_tokens": 1},
    )
    context = ValidationRolloutContext(model_step=0, temperature=1.0, n_generation=1, seed=39)

    with pytest.raises(RuntimeError, match="request differs across FSDP ranks"):
        rollout.generate(
            [ObsBundle(representations={"prompt": "alpha"}, id="sample-a")],
            validation=context,
        )
    assert model.generate_calls == 0


@pytest.mark.L0
@pytest.mark.CPU
def test_vllm_validation_rejects_different_tp_rank_requests_before_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    fake_vllm = ModuleType("vllm")
    fake_vllm.SamplingParams = _SamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    class _Engine:
        def __init__(self) -> None:
            self.generate_calls: int = 0

        def generate(self, _prompts: list[dict[str, list[int]]], _params: object) -> list[object]:
            self.generate_calls += 1
            return []

    def _different_peer_request(value: object, context: str) -> object:
        assert context == "vLLM validation request identity"
        assert isinstance(value, str) and len(value) == 64
        raise RuntimeError(f"{context} differs across ranks")

    monkeypatch.setattr(
        "cosmos_framework.posttrain.utils.dist.rank_identical",
        _different_peer_request,
    )
    engine = _Engine()
    rollout = VLLMRollout(engine, packer=_Packer(), sampling={"temperature": 0.9, "max_tokens": 4})
    context = ValidationRolloutContext(model_step=0, temperature=0.0, n_generation=1, seed=43)

    with pytest.raises(RuntimeError, match="request identity differs across ranks"):
        rollout.generate(
            [ObsBundle(representations={"prompt": "question"}, id="sample")],
            validation=context,
        )
    assert engine.generate_calls == 0


class _RankFailPacker:
    tokenizer: _Tokenizer = _Tokenizer()

    @staticmethod
    def get_rollout_input(obs: ObsBundle) -> str:
        import torch.distributed as dist

        if dist.get_rank() == 0:
            raise RuntimeError("rank-local packer failure")
        return str(obs.representations["prompt"])


class _RankFailDecodeTokenizer(_Tokenizer):
    @staticmethod
    def decode(token_ids: torch.Tensor, **_kwargs: object) -> str:  # token_ids: [T]
        import torch.distributed as dist

        if dist.get_rank() == 0:
            raise RuntimeError("rank-local output parsing failure")
        return ",".join(str(int(token)) for token in token_ids)


class _RankFailParsePacker:
    tokenizer: _RankFailDecodeTokenizer = _RankFailDecodeTokenizer()

    @staticmethod
    def get_rollout_input(obs: ObsBundle) -> str:
        return str(obs.representations["prompt"])


class _CollectiveHFModel(_RandomHFModel):
    def generate(self, input_ids: torch.Tensor, **_kwargs: object) -> object:  # input_ids: [1,P]
        import torch.distributed as dist

        self.generate_calls += 1
        collective_marker = torch.ones((), dtype=torch.int32)  # []
        dist.all_reduce(collective_marker)  # []
        sampled = torch.ones((1, 1), dtype=torch.long)  # [1,1]
        sequences = torch.cat((input_ids, sampled), dim=1)  # [1,P+1]
        return SimpleNamespace(sequences=sequences, scores=())


def _hf_validation_failure_worker(rank: int, world_size: int, rendezvous: str, failure_stage: str) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=15),
    )
    model = _CollectiveHFModel()
    packer = _RankFailPacker() if failure_stage == "preflight" else _RankFailParsePacker()
    rollout = HFRollout(model, packer=packer, sampling={"do_sample": False, "max_new_tokens": 1})
    generation_count = 1 if failure_stage == "preflight" else 2
    context = ValidationRolloutContext(model_step=0, n_generation=generation_count, seed=43)
    try:
        expected_phase = (
            "HF validation prompt preparation 0" if failure_stage == "preflight" else "HF validation output parsing 0:0"
        )
        with pytest.raises(RuntimeError, match=f"{expected_phase} failed on one or more ranks"):
            rollout.generate(
                [ObsBundle(representations={"prompt": "question"}, id="sample")],
                validation=context,
            )
        assert model.generate_calls == (0 if failure_stage == "preflight" else 1)
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_hf_validation_rank_local_failures_converge_around_collective_generate(tmp_path: Path) -> None:
    for failure_stage in ("preflight", "parsing"):
        rendezvous = str(tmp_path / f"hf-validation-{failure_stage}")
        torch.multiprocessing.spawn(
            _hf_validation_failure_worker,
            args=(2, rendezvous, failure_stage),
            nprocs=2,
            join=True,
        )


@pytest.mark.L0
@pytest.mark.CPU
def test_validation_rollout_refuses_a_stale_weight_version() -> None:
    rollout = VLLMRollout(object(), packer=_Packer(), sampling={"max_tokens": 4})
    rollout._weight_version = 3
    context = ValidationRolloutContext(model_step=2)
    with pytest.raises(RuntimeError, match="pinned model_step=2.*weight version 3"):
        rollout.generate([], validation=context)


@pytest.mark.L0
@pytest.mark.CPU
def test_shared_rollout_weight_confirmation_pins_periodic_validation_version() -> None:
    shared_model = SimpleNamespace()
    rollout = VLLMRollout(shared_model, packer=_Packer(), sampling={"max_tokens": 4})

    # No backend is the shared/static weight edge. Exercise the same prepare/commit phases the
    # controller runs after the policy sync barrier drains, preserving the strict validator.
    producer_identity = rollout.shared_weight_identity()
    rollout.prepare_weight_version(2, producer_identity)
    rollout.commit_weight_version(2)

    rollout.require_validation_weight_version(ValidationRolloutContext(model_step=2))
    assert rollout._weight_version == 2

    other_rollout = VLLMRollout(SimpleNamespace(), packer=_Packer(), sampling={"max_tokens": 4})
    with pytest.raises(RuntimeError, match="share the exact model object"):
        other_rollout.prepare_weight_version(2, producer_identity)
