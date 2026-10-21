# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Colocated shared-model GRPO over an FSDP2-sharded ``HFModel`` — test fixture.

Builds the tier-3 shape (DESIGN §4.2): ONE ``ActorGroup`` whose ``setup`` constructs a single FSDP
``HFModel`` that the rollout and the trainer SHARE, so ``push_weights`` is the SharedRef no-op and
the whole loop runs on a sharded model with no cross-mesh weight transfer. The same wiring binds on
both backends — ``local`` (bodies in this process, mesh from the torchrun world) and ``ray`` (bodies
in a Ray actor grid, mesh injected via ``workers.ray.set_mesh`` and read by ``resolve_pdims``).

No production entry emits this shape: ``entrypoints/grpo.py`` always builds two ActorGroups and only
ever constructs a vLLM rollout. This fixture is therefore the only executable coverage of the shared
model path, and it lives in ``tests/`` rather than in user-facing integration code for exactly that reason.

PROMPTS are mirrored; the TRAINING BATCH is not. data/rm/replay are ``ranks=1``, so the registry
rank-mirrors one ref to all K lanes and every lane rolls out identical prompts under an identical
per-step seed. The ``train`` edge then applies ``workers/dispatch.py::select_chunk``, which takes
this rank's ``index % dp_world == dp_rank`` share — so the optimizer step IS data-parallel over the
injected mesh. Unequal per-rank counts do not deadlock: the objective agrees the forward COUNT
across ranks (``utils/dist.py::rank_max_count``) and pads the short ranks with zero-weight forwards.
"""

from __future__ import annotations

import torch

from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
from cosmos_framework.posttrain.algorithm.grpo import make_objective
from cosmos_framework.posttrain.algorithm.reward import (
    RewardCandidate,
    RewardCombiner,
    RewardContext,
    RewardProviderResult,
    bind_candidate_provider,
)
from cosmos_framework.posttrain.config.build import resolve_pdims
from cosmos_framework.posttrain.config.model import HFModelConfig
from cosmos_framework.posttrain.control import ExperienceBuffer
from cosmos_framework.posttrain.model.hf import build_hf_model
from cosmos_framework.posttrain.roles.env import ArithmeticEnv
from cosmos_framework.posttrain.roles.reward import RewardWorker
from cosmos_framework.posttrain.roles.rollout import HFRollout
from cosmos_framework.posttrain.roles.trainer import LLMTrainer
from cosmos_framework.posttrain.utils.entry import resolve, step_resolver

# role name -> roles.* factory kind (the colocation group; one setup builds all five)
ROLE_SPECS = [("data", "env"), ("rollout", "rollout"), ("rm", "reward"), ("replay", "buffer"), ("policy", "trainer")]
GROUP_SIZE = 3


def _variance_reward(candidate: RewardCandidate, context: RewardContext) -> RewardProviderResult:
    """Completion-length parity -> a sampled group has non-zero advantage (real optimizer step)."""
    del context
    return RewardProviderResult(terms={"task": float(len(candidate.completion) % 2)})


def param_signature(net: torch.nn.Module) -> float:
    """One deterministic scalar over the FULL (gathered) params.

    Detects that a step changed the weights or that a DCP resume restored them. Collective: every
    rank calls ``full_tensor`` in lockstep.

    Args:
        net: The (possibly FSDP-sharded) module to summarize.

    Returns:
        The sum over every gathered parameter.
    """
    from torch.distributed.tensor import DTensor

    total = 0.0
    for p in net.parameters():
        t = p.full_tensor() if isinstance(p, DTensor) else p
        total += float(t.detach().double().sum().item())
    return total


def build_reward_worker(group_size: int) -> RewardWorker:
    """Build the reward stage with the same group size used by rollout and replay.

    Args:
        group_size: Completions per prompt; the advantage estimator checks group completeness.

    Returns:
        The configured reward worker.
    """
    return RewardWorker(
        [bind_candidate_provider("variance", _variance_reward)],
        RewardCombiner(),
        annotators=(group_relative(group_size=group_size),),
        training_group_size=group_size,
    )


def build_shared_bodies(
    model_dir: str,
    *,
    group_size: int = GROUP_SIZE,
    checkpoint_format: str = "dcp",
    save_async: bool = False,
) -> dict:
    """The colocation ``setup``: build ONE FSDP ``HFModel`` and the five roles that SHARE it.

    Runs in-process (local) or inside each Ray actor (ray) — in the Ray case the injected mesh
    arrives via ``current_pdims()`` inside ``config.build.resolve_pdims``. Rollout wraps ``hf.net``
    (the raw FSDP module with ``.generate``); policy wraps the ``HFModel`` body (uses ``.net``).

    Args:
        model_dir: HF model dir (config + safetensors) shared by all five roles.
        group_size: Completions generated per prompt and expected by the reward stage.
        checkpoint_format: Trainer checkpoint format (``consolidated`` or ``dcp``).
        save_async: Allow immutable consolidated snapshots to use the background slot.

    Returns:
        The ``{name: body}`` setup dict for the single ActorGroup.
    """
    from cosmos_framework.posttrain.data.packer import init_data_packer

    hf = build_hf_model(
        HFModelConfig(
            name_or_path=model_dir,
            dtype="float32",
            attn_impl="eager",
            trust_remote_code=False,
            auto_class="AutoModelForCausalLM",
        ),
        pdims=resolve_pdims(),
    )
    packer = init_data_packer(model_dir)  # rollout + objective shaping seam; owns the tokenizer
    # Momentum gives the checkpoint gate real optimizer tensor state instead of an empty SGD state.
    opt = torch.optim.SGD(hf.net.parameters(), lr=1.0, momentum=0.9)
    return {
        "data": ArithmeticEnv(seed=0, max_operand=9),
        "rollout": HFRollout(
            hf.net, packer=packer, sampling={"do_sample": True, "temperature": 1.0, "max_new_tokens": 4}
        ),
        "rm": build_reward_worker(group_size),
        "replay": ExperienceBuffer(),
        "policy": LLMTrainer(
            hf,
            make_objective(packer),
            opt,
            checkpoint_format=checkpoint_format,
            save_async=save_async,
        ),
    }


def grpo_loop(wgs: dict, steps: int = 2, P: int = 2, base_seed: int = 1234, backend: str = "ray") -> list[float]:
    """Drive the colocation-group GRPO loop over the WorkerGroups; returns the per-step losses.

    ``seed=base_seed+step`` is broadcast to every cell so all FSDP ranks sample identically (aligned
    collectives) while a group still varies internally (non-zero GRPO advantage). The loop reads
    group size from the bound reward body, so rollout, replay, reward validation, and advantage
    annotation cannot receive independent values.

    Handle flow differs by backend and is load-bearing, NOT cosmetic:
      ray   : PASS handles through so the controller forwards only refs, never payloads. data/rm/
              replay are ranks=1, so the registry forwards ONE data ref to all K FSDP lanes
              (op="one", readers=K) and the REFCOUNTED bulk store lets every lane materialize it.
              Resolving and broadcasting cell-0's batch instead would make a peer FSDP actor
              ``ray.get()`` a payload owned by an actor already blocked in the ``train()``
              collective — a mutual wait.
      local : bodies run in-process, so resolve each step to a plain value (there is no dp-align
              step and a Handle passed as an arg would not auto-resolve).
    Only the final loss is resolved in both cases.

    Args:
        wgs: The ``{name: WorkerGroup}`` colocation dict (data/rollout/rm/replay/policy).
        steps: Number of GRPO steps to run.
        P: Prompts sampled per step.
        base_seed: Seed base; step ``i`` broadcasts ``base_seed + i`` to every cell.
        backend: ``"ray"`` (pass handles through) or ``"local"`` (resolve to eager values).

    Returns:
        The per-step ``train/loss_avg`` values (one float per step).

    Raises:
        RuntimeError: A step produced a non-finite loss.
    """
    data, rollout, rm, replay, policy = (wgs[n] for n in ("data", "rollout", "rm", "replay", "policy"))
    group_size = int(resolve(rm.configured_group_size()))
    step_res = step_resolver(backend)  # local: resolve eagerly; ray: pass handles through
    losses: list[float] = []
    for step in range(steps):
        prompts = step_res(data.get_samples(P))
        trajs = step_res(rollout.generate(prompts, group=group_size, seed=base_seed + step))
        scored = step_res(rm.score(trajs))
        replay.put_samples(scored)
        batch = step_res(replay.get_samples(P * group_size))
        # The select control kwargs are what ARM `select_chunk` (selection is opt-in on their
        # presence — roles/core/dispatch.py). n_mini/n_micro=1 means no mini/micro chunking; the
        # DP interleave still runs, giving each rank its own share of the mirrored batch ref.
        loss = float(
            resolve(policy.train(batch, mini=0, micro=0, n_mini=1, n_micro=1))["train/loss_avg"]
        )  # always resolve the reduced loss
        policy.push_weights()  # SharedRef no-op: trainer + rollout share the model
        if not (loss == loss and abs(loss) != float("inf")):
            raise RuntimeError(f"non-finite train/loss_avg at step {step}: {loss}")
        losses.append(loss)
    return losses


def bind(
    model_dir: str,
    backend: str,
    ranks: int = 2,
    *,
    group_size: int = GROUP_SIZE,
    checkpoint_format: str = "dcp",
    save_async: bool = False,
) -> dict:
    """Create the five role WGs and bind the shared-model ActorGroup on ``backend``.

    Args:
        model_dir: HF model dir (config + safetensors) shared by all five roles.
        backend: ``"local"`` (in-process, torchrun world) or ``"ray"`` (Ray actor grid).
        ranks: ray only — FSDP shard degree (number of GPU actors).
        group_size: Completions generated per prompt and expected by the reward stage.
        checkpoint_format: Trainer checkpoint format (``consolidated`` or ``dcp``).
        save_async: Allow immutable consolidated snapshots to use the background slot.

    Returns:
        The ``{name: WorkerGroup}`` dict for the bound ActorGroup.

    Raises:
        ValueError: ``backend`` is neither ``"local"`` nor ``"ray"``.
    """
    from cosmos_framework.posttrain.control.placement import ActorGroup, DeviceGroup, bind_device_groups
    from cosmos_framework.posttrain.roles import roles as role_factory

    def _setup() -> dict:
        return build_shared_bodies(
            model_dir,
            group_size=group_size,
            checkpoint_format=checkpoint_format,
            save_async=save_async,
        )

    if backend == "local":
        wgs = {name: getattr(role_factory, kind)(name) for name, kind in ROLE_SPECS}
        device, bind_kwargs = "cpu", {}
    elif backend == "ray":
        # replicas=1, ranks=K: rollout + policy form one pure-FSDP group of K GPU actors; the
        # placement layer builds + injects ParallelDims(world=K, dp_shard=K) which resolve_pdims
        # reads via current_pdims(). data/rm/replay are ranks=1 SPMD helpers: the dispatch registry
        # then routes generate/train op="one" (P == R_c == 1) — one producer ref rank-mirrored to
        # all K FSDP lanes via REFCOUNTED bulk, so every lane gets identical data and the FSDP
        # forward all-gathers stay aligned. Declaring the helpers at ranks=K would make the registry
        # see K units into 1 replica and try to reduce/split them (raise).
        helper_kinds = {"env", "reward", "buffer"}
        wgs = {
            name: getattr(role_factory, kind)(name, replicas=1, ranks=(1 if kind in helper_kinds else ranks))
            for name, kind in ROLE_SPECS
        }
        device, bind_kwargs = "gpu", {"threshold": 0}
    else:
        raise ValueError(f"backend={backend!r}: expected 'local' or 'ray'")

    bind_device_groups(
        [DeviceGroup(actor_groups=(ActorGroup(members=tuple(wgs.values()), setup=_setup),), device=device)],
        backend=backend,
        **bind_kwargs,
    )
    return wgs
