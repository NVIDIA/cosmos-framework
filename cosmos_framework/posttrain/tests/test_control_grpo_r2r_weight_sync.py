# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""R2R rank-isolation and GRPO P2R+R2R value proofs for HF and vLLM rollout engines."""

from __future__ import annotations

import os
from typing import Any

import pytest

torch = pytest.importorskip("torch")

ROLLOUT_REPLICAS = 3
MULTIRANK_REPLICAS = 2
MULTIRANK_RANKS = 2
REQUIRED_GPUS = max(ROLLOUT_REPLICAS + 1, MULTIRANK_REPLICAS * MULTIRANK_RANKS)
MODEL_VERSION = 17
VLLM_MODEL = os.environ.get("VLLM_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")
VLLM_WITNESS = "model.layers.0.input_layernorm.weight"


def _tiny_gpt2_model() -> Any:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(n_layer=1, n_head=2, n_embd=32, vocab_size=50257, n_positions=64)
    return GPT2LMHeadModel(config).cuda()


def _setup_hf_rollout() -> dict[str, Any]:
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.rollout import HFRollout

    model = _tiny_gpt2_model()
    return {
        "rollout": HFRollout(
            model,
            packer=init_data_packer("gpt2"),
            sampling={"do_sample": False, "max_new_tokens": 1},
        )
    }


def _setup_hf_policy() -> dict[str, Any]:
    from cosmos_framework.posttrain.algorithm.grpo import make_objective
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    model = _tiny_gpt2_model()
    packer = init_data_packer("gpt2")
    return {"policy": LLMTrainer(model, make_objective(packer), torch.optim.SGD(model.parameters(), lr=1e-3))}


def _state_cpu(model: Any) -> dict[str, torch.Tensor]:
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}  # [*state]
    return state


def _stamp_hf_policy(host: Any) -> dict[str, torch.Tensor]:
    body = host._bodies["policy"]
    with torch.no_grad():
        for parameter in body._net().parameters():
            parameter.add_(5.0)  # [*parameter]
    body._iter = MODEL_VERSION
    return _state_cpu(body._net())


def _stamp_hf_rollout(host: Any, value: float) -> dict[str, torch.Tensor]:
    body = host._bodies["rollout"]
    with torch.no_grad():
        for parameter in body.model.parameters():
            parameter.fill_(value)  # [*parameter]
    return _state_cpu(body.model)


def _read_hf_rollout_state(host: Any) -> dict[str, torch.Tensor]:
    return _state_cpu(host._bodies["rollout"].model)


def _setup_vllm_rollout() -> dict[str, Any]:
    # TP=1 follows the supported in-process model-owner contract used by R2R callbacks.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM

    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.rollout import VLLMRollout

    model = LLM(
        model=VLLM_MODEL,
        tensor_parallel_size=1,
        kv_cache_memory_bytes=64 * 1024 * 1024,
        max_model_len=128,
        max_num_seqs=1,
        dtype="float32",
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    return {"rollout": VLLMRollout(model, packer=init_data_packer(VLLM_MODEL))}


def _setup_vllm_policy() -> dict[str, Any]:
    from transformers import AutoModelForCausalLM

    from cosmos_framework.posttrain.algorithm.grpo import make_objective
    from cosmos_framework.posttrain.data.packer import init_data_packer
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    model = AutoModelForCausalLM.from_pretrained(VLLM_MODEL, torch_dtype=torch.float32).cuda()
    packer = init_data_packer(VLLM_MODEL)
    return {"policy": LLMTrainer(model, make_objective(packer), torch.optim.SGD(model.parameters(), lr=1e-3))}


def _stamp_vllm_policy(host: Any) -> torch.Tensor:  # returns [H]
    body = host._bodies["policy"]
    parameter = dict(body._net().named_parameters())[VLLM_WITNESS]  # [H]
    with torch.no_grad():
        parameter.add_(5.0)  # [H]
    body._iter = MODEL_VERSION
    return parameter.detach().float().cpu()  # [H]


def _read_vllm_model_witness(model: Any) -> torch.Tensor:  # returns [H]
    parameter = dict(model.named_parameters())[VLLM_WITNESS]  # [H]
    return parameter.detach().float().cpu()  # [H]


def _read_vllm_rollout_witness(host: Any) -> torch.Tensor:  # returns [H]
    from cosmos_framework.posttrain.roles.rollout import vllm_weights

    rollout = host._bodies["rollout"]
    results = vllm_weights._apply_to_models(rollout.model, _read_vllm_model_witness)
    if len(results) != 1 or not isinstance(results[0], torch.Tensor):
        raise RuntimeError(f"expected one vLLM tensor witness, got {results!r}")
    return results[0]  # [H]


def _read_weight_backend_kind(host: Any) -> str:
    backend = getattr(host._bodies["rollout"], "_weight_backend", None)
    return type(getattr(backend, "mover", backend)).__name__


def _assert_relay_binding(ray: Any, rollout: Any) -> None:
    kinds = {
        cell: ray.get(actor.run.remote(_read_weight_backend_kind)) for cell, actor in rollout._backend.actors.items()
    }
    assert kinds == {(0, 0): "NixlWeightMover", (1, 0): "NoneType", (2, 0): "NoneType"}


def _states_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _training_batch(step: int) -> list[Any]:
    from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
    from cosmos_framework.posttrain.schema import ObsBundle, Status, Trajectory, Transition

    trajectories: list[Trajectory] = []
    for sample, reward in enumerate((1.0, 0.0)):
        prompt_ids = torch.tensor([1, 2], dtype=torch.long)  # [P]
        completion_ids = torch.tensor([3 + step + sample], dtype=torch.long)  # [C]
        transition = Transition(
            obs=ObsBundle(representations={"prompt": "synthetic training prompt"}),
            action=completion_ids,
            policy_info={"prompt_ids": prompt_ids},
            reward={"final": reward},
            done=True,
        )
        trajectories.append(
            Trajectory(
                transitions=[transition],
                id=f"train-{step}-{sample}",
                group_id=f"train-{step}",
                status=Status.COMPLETED,
            )
        )
    group_relative(group_size=2)(trajectories)
    return trajectories


def _train_and_sync_twice(policy: Any, rollout: Any) -> None:
    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync

    for offset in range(1, 3):
        expected_version = MODEL_VERSION + offset
        metrics = policy.train(_training_batch(offset)).result()
        assert metrics["optim/update_outcome"] == "applied", metrics
        assert metrics["optim/optimizer_step"] == expected_version
        assert float(metrics["train/loss_avg"]) == float(metrics["train/loss_avg"])
        # Steady-state sync intentionally leaves exact-value verification at its default False.
        weight_sync(rollout, policy, validation_step=expected_version)


@pytest.mark.L1
@pytest.mark.GPU
def test_grpo_initial_p2r_r2r_syncs_three_hf_rollout_replicas_exactly() -> None:
    ray = pytest.importorskip("ray")
    pytest.importorskip("transformers")
    if torch.cuda.device_count() < REQUIRED_GPUS:
        pytest.skip(f"needs {REQUIRED_GPUS} GPUs")

    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    try:
        rollout = roles.rollout("rollout", replicas=ROLLOUT_REPLICAS, ranks=1)
        policy = roles.trainer("policy", replicas=1, ranks=1)
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(ActorGroup(members=(rollout,), setup=_setup_hf_rollout),),
                    device="gpu",
                ),
                DeviceGroup(
                    actor_groups=(ActorGroup(members=(policy,), setup=_setup_hf_policy),),
                    device="gpu",
                ),
            ],
            backend="ray",
            threshold=0,
            weight_sync=True,
            weight_sync_transport="nixl",
            weight_sync_bucket_bytes=16 * 1024 * 1024,
            weight_sync_probe_bytes=1024 * 1024,
            weight_sync_min_gbps=0.0,
            weight_sync_r2r_broadcast=True,
        )
        _assert_relay_binding(ray, rollout)
        before = {
            cell: ray.get(actor.run.remote(_read_hf_rollout_state)) for cell, actor in rollout._backend.actors.items()
        }
        policy_actor = policy._backend.actors[(0, 0)]
        expected = ray.get(policy_actor.run.remote(_stamp_hf_policy))
        assert all(not _states_equal(state, expected) for state in before.values())

        weight_sync(rollout, policy, validation_step=MODEL_VERSION, verify_replica_values=True)

        after = {
            cell: ray.get(actor.run.remote(_read_hf_rollout_state)) for cell, actor in rollout._backend.actors.items()
        }
        assert all(_states_equal(state, expected) for state in after.values())
    finally:
        ray.shutdown()


@pytest.mark.L1
@pytest.mark.GPU
def test_r2r_broadcast_keeps_two_hf_ranks_isolated_across_two_replicas() -> None:
    ray = pytest.importorskip("ray")
    pytest.importorskip("transformers")
    if torch.cuda.device_count() < REQUIRED_GPUS:
        pytest.skip(f"needs {REQUIRED_GPUS} GPUs")

    from cosmos_framework.posttrain.control.loop.weight_sync import r2r_weight_sync
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    try:
        rollout = roles.rollout("rollout", replicas=MULTIRANK_REPLICAS, ranks=MULTIRANK_RANKS)
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(ActorGroup(members=(rollout,), setup=_setup_hf_rollout),),
                    device="gpu",
                ),
            ],
            backend="ray",
            threshold=0,
            weight_sync_bucket_bytes=16 * 1024 * 1024,
            weight_sync_r2r_broadcast=True,
        )
        actors = rollout._backend.actors
        expected = {
            rank: ray.get(actors[(0, rank)].run.remote(_stamp_hf_rollout, value=float(rank + 1)))
            for rank in range(MULTIRANK_RANKS)
        }
        before = {
            rank: ray.get(actors[(1, rank)].run.remote(_read_hf_rollout_state)) for rank in range(MULTIRANK_RANKS)
        }
        assert all(not _states_equal(before[rank], expected[rank]) for rank in range(MULTIRANK_RANKS))

        r2r_weight_sync(rollout, model_step=MODEL_VERSION, verify_replica_values=True)

        after = {cell: ray.get(actor.run.remote(_read_hf_rollout_state)) for cell, actor in actors.items()}
        for rank in range(MULTIRANK_RANKS):
            assert _states_equal(after[(0, rank)], expected[rank])
            assert _states_equal(after[(1, rank)], expected[rank])
        assert not _states_equal(after[(1, 0)], after[(0, 1)])
    finally:
        ray.shutdown()


@pytest.mark.L1
@pytest.mark.GPU
def test_grpo_initial_p2r_r2r_syncs_three_vllm_rollout_replicas_exactly() -> None:
    ray = pytest.importorskip("ray")
    pytest.importorskip("transformers")
    pytest.importorskip("vllm")
    if torch.cuda.device_count() < REQUIRED_GPUS:
        pytest.skip(f"needs {REQUIRED_GPUS} GPUs")

    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles

    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    try:
        rollout = roles.rollout("rollout", replicas=ROLLOUT_REPLICAS, ranks=1)
        policy = roles.trainer("policy", replicas=1, ranks=1)
        bind_device_groups(
            [
                DeviceGroup(
                    actor_groups=(ActorGroup(members=(rollout,), setup=_setup_vllm_rollout),),
                    device="gpu",
                ),
                DeviceGroup(
                    actor_groups=(ActorGroup(members=(policy,), setup=_setup_vllm_policy),),
                    device="gpu",
                ),
            ],
            backend="ray",
            threshold=0,
            weight_sync=True,
            weight_sync_transport="nixl",
            weight_sync_bucket_bytes=16 * 1024 * 1024,
            weight_sync_probe_bytes=1024 * 1024,
            weight_sync_min_gbps=0.0,
            weight_sync_r2r_broadcast=True,
        )
        _assert_relay_binding(ray, rollout)
        before = {
            cell: ray.get(actor.run.remote(_read_vllm_rollout_witness))
            for cell, actor in rollout._backend.actors.items()
        }
        policy_actor = policy._backend.actors[(0, 0)]
        expected = ray.get(policy_actor.run.remote(_stamp_vllm_policy))  # [H]
        assert all(not torch.equal(value, expected) for value in before.values())

        weight_sync(rollout, policy, validation_step=MODEL_VERSION, verify_replica_values=True)

        after = {
            cell: ray.get(actor.run.remote(_read_vllm_rollout_witness))
            for cell, actor in rollout._backend.actors.items()
        }
        assert all(torch.equal(value, expected) for value in after.values())
        _train_and_sync_twice(policy, rollout)
    finally:
        ray.shutdown()
