# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""AlpaGym closed-loop RL: the replacement for `cosmos_rl.launcher.launch_all`.

Scope is bounded to what that launcher did — stand up the trainer and rollout processes and run
the RL loop. AlpaGym keeps everything around it: its host CLI, the AlpaSim wizard bring-up, the
endpoint registry, and the Slurm `srun` wrapping. Only the command that step launches changes.

TOPOLOGY (design §3). The two meshes are TRANSPOSED — policy `(1, gpus)` FSDP, rollout
`(gpus, 1)` independent engines — because AlpaGym's rollout is an HF model behind an
InferenceEngine and does not tensor-parallel shard, while its trainer needs FSDP.

`entrypoints/grpo.py::declare_wgs` now takes `shapes` (`grid.shapes`, §8.2) and would express
exactly this pair, so the shapes are no longer the reason to keep a separate declaration. What
still differs is `data`: that entry derives every experience role at `(R_rollout, 1)`, while this
recipe's scene source is smaller than `gpus` and `DatasetEnv` rejects sharding it, so `data` stays
single-cell and rides the registered `("rollout","generate","prompts")` split instead. Reusing that
function is a real option once the scene source is large enough to shard.

Sharding the policy is a REQUIREMENT, not a tuning choice: measured on cw-dfw, an unsharded
trainer is 54.62 GiB and does not fit beside the rollout's 20.64 GiB on one 80 GiB card.

CONFIG. This entry reads AlpaGym's already-resolved run config (`alpagym_host.config.
load_run_config`), the same file `cosmos/entrypoint.py` reads today — it does NOT go through i4's
config root. That root needs `multi-storage-client`, which moves protobuf 4.25.9 → 7.35.1 in
AlpaGym's venv, under a link where AlpaGym exchanges protobuf messages with the simulator on every
one of an episode's 22 ticks. Migrating the config surface is milestone 2 (design §12).
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cosmos_framework.posttrain.config.placement import GpuPlacement, normalize_gpu_placement
from cosmos_framework.posttrain.roles.reward import reward_state_transaction
from cosmos_framework.posttrain.workers.worker_group import WorkerGroup

if TYPE_CHECKING:
    from cosmos_framework.posttrain.schema import Trajectory

# One list drives the WorkerGroup shapes and the setup closure, so the roles built cannot drift
# from the roles declared.
#
# NOT VALIDATED AOT, deliberately: this entry binds and loops directly (`run`) with no `Manifest`
# and no `bind_check`, unlike every other entry, and `"alpagym_rollout"` has no
# `CapabilityDescriptor`. `docs/howto/preflight.md` calls that the entry contract, so this is a knowing
# departure rather than an oversight -- the recipe is one fixed internal topology whose wiring is
# exercised end-to-end on every run, and the pre-flight check would restate `ROLE_SPECS` and the
# five edges below without a second source to disagree with. Revisit if the topology becomes
# configurable, which is when a manifest starts catching something the run does not.
ROLE_SPECS: list[tuple[str, str]] = [
    ("data", "env"),
    ("rollout", "rollout"),
    ("rm", "reward"),
    ("replay", "buffer"),
    ("policy", "trainer"),
]

# The experience side shares one ActorGroup; the learner gets its own. One GPU DeviceGroup places
# them on the same physical GPUs by flat cell index — GPU-colocated, process-separated.
_EXPERIENCE = ("data", "rollout", "rm", "replay")
_LEARNER = ("policy",)


@dataclass(frozen=True)
class AlpagymSection:
    """The AlpaGym-side inputs this entry needs, all resolved by AlpaGym's host CLI."""

    resolved_config_path: str
    # The AlpaGym project root that `artifact_paths`' relative paths hang off. See `make_setup`.
    project_root: str


@dataclass(frozen=True)
class EntryConfig:
    """This entry's own config. Deliberately small: everything about the policy, the simulator and
    the run layout already lives in AlpaGym's resolved config, reached through `alpagym`."""

    alpagym: AlpagymSection
    gpus: int
    steps: int
    # Derived, never passed: `prompts x group == batch_per_replica` is the conservation law
    # upstream states for `grid.shapes` topologies, degenerate here at one data cell and one policy
    # replica. AlpaGym's launcher used to compute `train_batch // group_size` and hand it over,
    # which is a second place for the same number to be wrong.
    prompts_per_step: int
    group_size: int
    # Drive ticks per episode. Each becomes one micro-batch, so this sets both the loop's
    # iteration count and the trainer's grad_accum -- they MUST agree (see `loop`).
    ticks_per_episode: int
    # "colocated" (default) or "disaggregated" — see `declare_wgs` and `run`. The word is the
    # framework's, normative on the GPU axis (DESIGN §4.0): it says whether policy and rollout
    # occupy the SAME physical GPUs, not how the processes are packed.
    placement: GpuPlacement = "colocated"


def declare_wgs(gpus: int, placement: str = "colocated") -> dict[str, WorkerGroup]:
    """Declare this recipe's WorkerGroups as a transposed mesh.

    The mesh is TRANSPOSED either way -- policy `(1, width)` FSDP against rollout `(width, 1)`
    independent engines. What `placement` changes is `width`, and with it which physical GPUs the
    two sides land on:

    - "colocated" (default): `width = gpus`. Every GPU carries one FSDP shard AND one engine, so
      the weight edge is same-GPU CUDA IPC. Smaller shards (13.7 GiB at gpus=4) are what make the
      two fit on one 80 GiB card beside the rollout's 20.6 GiB.
    - "disaggregated": `width = gpus // 2`. The trainer takes half the GPUs and the rollout the
      other half -- the layout cosmos-rl ran (`rollout_replicas: 2`, `dp_shard_size: 2`) -- so the
      weight edge is cross-device and needs the NIXL transport (`run`). Shards double (~27 GiB at
      gpus=4) but each side has its card to itself.

    `full_tensor_actor_targets`' bijection needs `R_policy x K_policy == R_rollout x K_rollout`, which
    `(1, width)` against `(width, 1)` satisfies at either setting.

    Args:
        gpus: GPUs available to the cosmos half of the node — ALL of them, both placements.
        placement: "colocated" or "disaggregated".

    Returns:
        `{name: WorkerGroup}` for every entry in `ROLE_SPECS`.
    """
    from cosmos_framework.posttrain.roles import roles as role_factory

    placement = normalize_gpu_placement(placement, field_name="placement")
    if gpus < 1:
        raise ValueError(
            f"gpus={gpus}: the transposed mesh needs at least one GPU. A zero would declare "
            "policy (1,0) and rollout (0,1), which fails deep inside bind with no hint that the "
            "GPU count was the problem -- the local_process topology reports 0 cosmos GPUs."
        )
    if placement == "disaggregated" and gpus % 2:
        raise ValueError(
            f"gpus={gpus} is odd: disaggregated splits the cosmos GPUs evenly between the trainer "
            "and the rollout, so an odd count would leave one side a GPU short of its own mesh"
        )
    width = gpus if placement == "colocated" else gpus // 2

    def _shape(name: str) -> dict[str, int]:
        if name == "policy":
            return {"replicas": 1, "ranks": width}  # one FSDP replica across the trainer's GPUs
        if name == "rollout":
            return {"replicas": width, "ranks": 1}  # one independent engine per rollout GPU, no TP
        if name == "data":
            # The scene source is smaller than `gpus`, which `DatasetEnv` rejects outright when asked
            # to shard. Its edge into the rollout needs no marker anyway: `roles/core/dispatch.py`
            # registers `("rollout", "generate", "prompts")` as a split, which shares the one
            # producer ref across the rollout's replicas, each keeping its own prompts.
            return {"replicas": 1, "ranks": 1}
        # `rm` and `replay` MATCH the rollout so every edge downstream of it is either replica-aligned
        # or already in that same registry, and the loop needs no dispatch marker at all:
        #   rollout (width,1) -> rm (width,1)      aligned
        #   rm      (width,1) -> replay (width,1)  aligned
        #   replay  (gpus,1) -> policy (1,gpus)  the registered ("trainer","train","batch") reduce
        # The fan-in to the single FSDP replica does not disappear — the meshes are transposed, so it
        # cannot — but it lands on the ONE edge the registry already covers. Ranks stay 1: a
        # multi-rank producer behind a Sharded edge is rejected, its mirrored lanes double-counting
        # as producer units (spec 2026-07-22 §2.2).
        return {"replicas": width, "ranks": 1}

    return {name: getattr(role_factory, kind)(name, **_shape(name)) for name, kind in ROLE_SPECS}


class EpisodeScorer:
    """The `rm` role: score whole episodes, then emit one trajectory per drive tick.

    AlpaSim already writes the episode reward onto the trajectory, so this is the env-native
    placement contract from DESIGN 1.5 rather than an online `RewardProvider`. It runs the configured
    whole-group annotators transactionally, preserving the simulator's reward terms and final value,
    then splits each episode for replay.

    SPLITTING HAPPENS AFTER, AND THAT ORDER IS WHAT MAKES IT SAFE. `group_relative` computes each
    group's mean/std over the WHOLE episodes and writes the result onto every transition, so
    splitting afterwards carries the advantage along unchanged. Splitting BEFORE would hand the
    estimator 22x more, much smaller "episodes" and silently redefine the group statistics. It
    cannot live in `annotators` either: those return None and mutate in place, so they cannot
    change the list's length.

    One episode is 22 drive ticks, and a tick is a multi-image VLM forward over 4 camera streams --
    together they do not fit beside the co-located rollout (measured: ~41 GiB, dying in the KV
    cache). Splitting is also what makes the counting downstream mean anything: `ExperienceBuffer`
    counts trajectories, so `select(1)` is one training sample only if a trajectory IS one.
    """

    _annotators: tuple[Callable[[list[Trajectory]], None], ...]

    def __init__(self, *, annotators: Sequence[Callable[[list[Trajectory]], None]]) -> None:
        self._annotators = tuple(annotators)

    def score(self, trajectories: list[Trajectory]) -> list[Trajectory]:
        """Score whole episodes, then return one trajectory per drive tick."""
        from cosmos_framework.posttrain.schema import FINAL_REWARD_KEY, Status, Trajectory

        with reward_state_transaction(trajectories):
            for annotate in self._annotators:
                annotate(trajectories)
            ticks: list[Trajectory] = []
            for trajectory in trajectories:
                for i, transition in enumerate(trajectory.transitions):
                    ticks.append(
                        Trajectory(
                            transitions=[transition],
                            id=f"{trajectory.id}_t{i}",
                            session=trajectory.session,
                            status=trajectory.status,
                            group_id=trajectory.group_id,
                        )
                    )
            # The loop draws a fixed `prompts x group x ticks` micro-batches per step, but
            # `ExperienceBuffer.insert` drops ABORTED trajectories -- so a short or aborted episode
            # leaves the tail of those draws empty and the step trains on less than it reports. This
            # is the last place the count is known, and the analogue of the cosmos-rl packer's
            # `valid_steps=22 padded_steps=0`.
            # Reward and advantage are summarized HERE, the one place both are in hand: the loop
            # forwards `score`'s result as a handle, and resolving it driver-side to read a scalar
            # would materialize every episode's camera frames -- what the metadata/bulk split exists
            # to avoid. Distributions, not means alone: a group-relative advantage is centred by
            # construction, so its mean says nothing and its spread says everything.
            # Read the SAME key the advantage estimator reads (`reward['final']`, via
            # `algorithm/reward/contract.py::_terminal_reward`), not a sum over the dict: the dict also
            # carries the per-term breakdown, and summing it would double-count the total.
            rewards = [
                float(t.transitions[-1].reward[FINAL_REWARD_KEY])
                for t in trajectories
                if t.transitions and t.transitions[-1].reward
            ]
            terms: dict[str, list[float]] = {}
            for trajectory in trajectories:
                row = trajectory.transitions[-1].reward if trajectory.transitions else None
                for key, value in (row or {}).items():
                    if key.startswith("term/"):
                        terms.setdefault(key, []).append(float(value))
            advantages = [
                float(transition.advantage)
                for trajectory in trajectories
                for transition in trajectory.transitions
                if transition.advantage is not None
            ]

            def _spread(values: list[float]) -> str:
                if not values:
                    return "n=0"
                mean = sum(values) / len(values)
                var = sum((v - mean) ** 2 for v in values) / len(values)
                return (
                    f"n={len(values)} mean={mean:+.6f} std={var**0.5:.6f} min={min(values):+.6f} max={max(values):+.6f}"
                )

            logging.getLogger(__name__).info(
                "[alpagym] scored episodes=%d ticks=%d aborted=%d ticks_per_episode=%s",
                len(trajectories),
                len(ticks),
                sum(1 for t in trajectories if t.status == Status.ABORTED),
                sorted({len(t.transitions) for t in trajectories}),
            )
            logging.getLogger(__name__).info("[alpagym] reward %s", _spread(rewards))
            for key in sorted(terms):
                logging.getLogger(__name__).info("[alpagym] %s %s", key, _spread(terms[key]))
            logging.getLogger(__name__).info("[alpagym] advantage %s", _spread(advantages))
        return ticks

    def score_with_report(self, *args: Any, **kwargs: Any) -> list[Trajectory]:
        """Refused: the report and this body disagree about what a trajectory is.

        The standard report contract describes the batch it scores, while this body emits split
        ticks. Returning a report over whole episodes would therefore describe a batch the caller
        never receives.
        """
        raise NotImplementedError(
            "EpisodeScorer emits one trajectory per drive tick; a report over unsplit episodes "
            "would disagree with score(). Use score() and read the counts this body logs."
        )


def make_setup(
    resolved_config_path: str, project_root: str, roles: tuple[str, ...], updates_per_step: int = 1
) -> Callable[[], dict]:
    """Build a closure that constructs ONLY `roles`, inside whichever actor it lands in.

    Captures nothing but strings so it can be shipped to a Ray actor, matching
    `entrypoints/grpo.py::make_setup`.

    `project_root` exists because EVERY path under AlpaGym's `artifact_paths` is RELATIVE
    (`tmp/alpagym-runs/<ts>/...`) -- the run config, the scene-id list, the topology registry, the
    log dir. Under cosmos-rl that was invisible: its workers inherited the container workdir, which
    is the AlpaGym project root those paths are relative to. A Ray actor does not. Ray stages a
    copy of the working directory under `/tmp/.../runtime_resources/working_dir_files/_ray_pkg_*`
    and runs there, so every one of those paths silently resolves inside that staging area and the
    files are simply absent.

    Restoring the cwd inside the actor is the smallest faithful fix: it re-establishes the very
    assumption the config was written under, and covers all of the paths at once rather than the
    one that happened to fail first.
    """

    def _setup() -> dict:
        import logging as _logging
        import os
        from pathlib import Path

        # Each Ray actor is its own process, so the driver's basicConfig does not apply here. The
        # file handler is the load-bearing part: Ray's own worker-*.out sit under its temp dir,
        # which lives in the ephemeral Slurm step container and is gone by the time anyone reads
        # it -- so an actor-side failure would otherwise leave only a driver-side traceback.
        log_dir = Path(resolved_config_path).parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        _logging.basicConfig(
            level=_logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            force=True,
            handlers=[
                _logging.StreamHandler(),
                _logging.FileHandler(log_dir / f"actor_{os.getpid()}.log"),
            ],
        )

        from alpagym_host.config import load_run_config

        os.chdir(project_root)
        from cosmos_framework.posttrain.algorithm.annotators.advantage import group_relative
        from cosmos_framework.posttrain.control import ExperienceBuffer
        from cosmos_framework.posttrain.roles.core.registry import build_rollout
        from cosmos_framework.posttrain.roles.env import DatasetEnv

        run_config = load_run_config(Path(resolved_config_path))
        # The rollout builder gets the AlpaGym section alone, not a whole `EntryConfig`: the loop's
        # own fields (gpus, steps, batch shape) mean nothing inside a role body, and filling them
        # with zeros to satisfy a signature would hand the builder the very values `declare_wgs`
        # rejects a few lines above.
        alpagym = AlpagymSection(resolved_config_path=resolved_config_path, project_root=project_root)

        out: dict[str, Any] = {}
        if "data" in roles:
            import yaml

            scene_ids = yaml.safe_load(run_config.artifact_paths.alpasim_scene_ids_path.read_text(encoding="utf-8"))[
                "scene_ids"
            ]
            out["data"] = DatasetEnv(
                dataset=[{"scene_id": str(scene_id)} for scene_id in scene_ids],
                repeat=True,
                modality_tags=("state",),
            )
        if "rollout" in roles:
            # Registration is an import side effect, so the module holding the @register_rollout
            # decorator has to be imported HERE -- inside the actor, before the lookup. At this
            # module's top level it would also make posttrain import alpagym, which is backwards:
            # AlpaGym depends on posttrain, not the reverse.
            import alpagym_runtime.posttrain.rollout  # noqa: F401

            out["rollout"] = build_rollout("alpagym_rollout", alpagym)
        if "rm" in roles:
            # `group_size` is the completeness guard, not decoration: without it the estimator
            # silently normalizes over whatever members arrived, and a short group is much more
            # likely to tie -- which in GRPO means an advantage of exactly 0 for every member,
            # a zero gradient, and a skipped update. Measured against the cosmos-rl baseline:
            # 27% of updates skipped here against its 6%.
            out["rm"] = EpisodeScorer(
                # The estimator is cosmos-rl's default: `(r - mean) / (std + eps)`, std
                # division included. `group_relative` also offers `unbiased=True` (mean-centred
                # only), and it was run here for a while because the division has a real failure
                # mode on this env: a group whose six members all reached the SAME outcome still
                # gets normalized to unit variance, so a reward spread of 0.015 becomes advantages
                # of +/-2, indistinguishable from a group that genuinely disagreed. Measured,
                # `advantage std` reads 1.000 on every step whether the reward std behind it was
                # 3.7 or 0.0076, and both collapses this loop produced began on the step after a
                # tied SUCCESS group. Mean-centring removed them.
                #
                # It is off because the baseline runs it off: cosmos-rl spells the same choice
                # `train.train_policy.unbiased_advantage`, that run logged it `False`, and parity
                # is what this entry is for. Note if you go to set it there -- its config
                # DESCRIPTION is inverted, reading "whether to divide the advantage by the
                # standard deviation" while `dispatcher/algo/grpo.py` divides when the flag is
                # FALSE.
                annotators=(group_relative(group_size=int(run_config.cosmos.rollout.n_generation)),),
            )
        if "replay" in roles:
            out["replay"] = ExperienceBuffer()
        if "policy" in roles:
            out["policy"] = _build_trainer(run_config, updates_per_step)
        return out

    return _setup


def _build_trainer(run_config: Any, updates_per_step: int) -> Any:
    """Assemble the learner: AlpaGym's policy model + RL optimizer + AlpaGym's replay objective,
    driven by posttrain's `LLMTrainer` engine.

    `LLMTrainer` is model-agnostic — it calls `objective(model, batch)` and never inspects the
    result — which is exactly why AlpaGym's forward contract can live in the objective and the
    ~450 lines of cosmos-rl trainer plumbing are replaced rather than ported.
    """
    from types import SimpleNamespace

    import torch
    from alpagym_runtime.policies.registry import get_policy_bundle
    from alpagym_runtime.posttrain.model import AlpamayoTrainModel
    from alpagym_runtime.posttrain.objective import make_alpagym_objective

    from cosmos_framework.posttrain.config.build import build_lr_scheduler
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    grpo = run_config.cosmos.train.train_policy
    bundle = get_policy_bundle(run_config.policy.model.kind)
    # MUST precede construction: the bridge registers the HF model type AND patches
    # `ExpertModelCosmos.forward` into AlpaGym's replay contract. Without it the model builds but
    # forwards the wrong signature -- which is exactly how the first end-to-end attempt failed.
    bundle.install_runtime_bridge()
    # The TRAINING model, not the rollout's: `build_inference_engine` yields ExpertModelRL
    # (generation), while training needs the patched ExpertModelCosmos.
    model = AlpamayoTrainModel(str(run_config.policy.model.path), dtype=torch.bfloat16)
    # FSDP width comes from the RANK mesh, not the replica count. This policy is declared
    # (replicas=1, ranks=gpus): `replica_coords()` would report (0, 1) and silently skip the FSDP
    # wrap entirely, loading the whole 54.62 GiB model on one card. The per-replica rendezvous
    # group the grid stands up is the right source -- the same one `grpo.py::build_rollout` reads
    # for its TP width.
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    model.setup(world_size=world_size)
    optimizer = torch.optim.AdamW(
        [p for p in model.net.parameters() if p.requires_grad],
        lr=float(run_config.cosmos.train.optm_lr),
        # AlpaGym's config exposes none of these three; each carries the value the cosmos-rl
        # baseline ran with (its `optm_weight_decay`, `optm_betas` and `epsilon` defaults), stated
        # explicitly rather than inherited from whatever torch defaults to.
        #
        # `eps` is the one that matters and the one torch disagrees on: AdamW's step is
        # `lr * m / (sqrt(v) + eps)`, so where gradients are small sqrt(v) is small too and eps
        # sets the scale. Torch's 1e-8 against cosmos-rl's 1e-6 is 100x on exactly those
        # parameters -- measured as the policy moving FURTHER per step (ratio saturating the
        # surrogate's exp(-5) clamp) while reporting a SMALLER gradient norm than the baseline.
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.01,
    )
    # `build_lr_scheduler` is cosmos-rl's `optm_warmup_steps` — linear ramp, then constant, on the
    # 1-based convention its docstring pins down (step 0 is lr/warmup, not zero). Reached through a
    # SimpleNamespace because it reads `warmup_steps` while AlpaGym spells the field
    # `optm_warmup_steps`: handed `run_config.cosmos.train` directly, its `getattr(..., 0)` finds no
    # such attribute, reads 0, and returns None -- no warmup, no error. That silence is the bug this
    # replaces, so the value is renamed explicitly rather than left to a lookup that cannot fail.
    train_cfg = run_config.cosmos.train
    # NOT scaled to the optimizer clock, though `LLMTrainer` advances the scheduler with it while
    # cosmos-rl steps once per closed-loop step. The two agree anyway at this setting: the ramp is
    # `lr * (step + 1) / warmup`, so `warmup_steps=1` is full LR from the very first update -- and
    # the baseline measurably runs at full LR from its first minibatch too (`lr=0.0001` on every
    # one of them). Scaling it by `updates_per_step` was tried and is wrong in the other direction:
    # it invents a 66-update ramp across the first closed-loop step that the baseline never has.
    lr_scheduler = build_lr_scheduler(optimizer, SimpleNamespace(warmup_steps=int(train_cfg.optm_warmup_steps)))
    logging.getLogger(__name__).info(
        "[alpagym] lr=%g, %d-step linear warmup", float(train_cfg.optm_lr), int(train_cfg.optm_warmup_steps)
    )
    objective = make_alpagym_objective(
        bundle.build_model_inputs(run_config),
        ratio_clip_low=float(grpo.grpo_ratio_clip_low),
        ratio_clip_high=float(grpo.grpo_ratio_clip_high),
        kl_beta=float(grpo.kl_beta),
        reference_reset_interval=int(grpo.reference_reset_interval),
    )
    return LLMTrainer(
        model=model,
        objective=objective,
        optimizer=optimizer,
        # AlpaGym's config exposes neither of these; both carry the value the cosmos-rl baseline
        # actually ran with (its `optm_grad_norm_clip` default of 1.0, visible as grad_norm=1.04 in
        # that run's log). LLMTrainer defaults max_grad_norm to 0.0, i.e. NO clipping -- inheriting
        # that default would silently change the update rule rather than preserve it.
        max_grad_norm=1.0,
        # cosmos-rl treats an all-zero GRPO gradient as no update. This is intentionally not an
        # LLMTrainer default: AdamW normally applies weight decay and advances moments when a zero
        # gradient tensor is present, so changing it for existing entries would change training.
        skip_zero_grad=True,
        # float32, which is what the cosmos-rl baseline sends: its `transfer_dtype` defaults to
        # `master_dtype`, and that run logged `'transfer_dtype': 'float32'`.
        #
        # bf16 was set here first, on the belief that both endpoints hold bf16. They do not. The
        # model's state dict is MIXED -- 356 float32 tensors against 810 bfloat16 (measured on
        # `ExpertModelCosmos`, the class BOTH the trainer and the rollout engine build) -- so a
        # bf16 wire rounds every one of those 356 on every sync, at bf16's ~2**-9 relative error,
        # while the baseline's fp32 wire delivers them intact.
        #
        # Why that went unnoticed: the checkpoint on disk is entirely BF16, so at startup even the
        # fp32 slots hold bf16-representable values and rounding them is exact. The startup
        # identity self-check therefore passes, and the divergence only begins once training gives
        # those parameters fp32-precision updates -- by which time nothing compares the two sides
        # again.
        #
        # The cost is the wire, not the arithmetic: buckets are sized from the declared dtype
        # (`apply_wire_dtype`), so this doubles the bytes per sync.
        transfer_dtype="float32",
        lr_scheduler=lr_scheduler,
        # ONE micro-batch per optimizer step, as the cosmos-rl baseline runs it: that trainer does
        # `zero_grad -> backward -> clip -> step` per minibatch. With the dispatch edge's DP split
        # armed (see `policy.train(..., n_mini=1, n_micro=1)` in the loop) each FSDP rank draws its
        # own half of the step's 132 ticks, so this is 66 updates per rank -- the baseline's count.
        #
        # Accumulating the whole batch instead was the previous behaviour and its cost was
        # measurable: Reduction.MEAN divides every loss by the window, giving grad_norm 0.035
        # against the baseline's 1.0, clip_fraction pinned at 0, and roughly 3x worse reward.
        grad_accum=1,
        # Optimizer steps per COLLECTED BATCH, and it must equal what the loop actually runs: the
        # trainer advances the LR schedule once per `n_mini` steps (the collected-batch clock) and
        # sizes the pi_old replay window by it. Left at 1 while the loop runs 66, the schedule
        # advances 66x too fast and the window closes after one chunk.
        n_mini=updates_per_step,
    )


def _updates_per_step(config: EntryConfig) -> int:
    """Return the exact number of optimizer steps in one collected batch."""
    batch_size = config.prompts_per_step * config.group_size * config.ticks_per_episode
    width = config.gpus if config.placement == "colocated" else config.gpus // 2
    if batch_size % width:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by policy width={width}; "
            "integer division would silently drop trajectories"
        )
    return batch_size // width


def loop(units: dict, config: EntryConfig) -> dict:
    """The closed-loop RL step: scenes → episodes → advantage → buffer → train → weight sync."""
    from pathlib import Path

    from cosmos_framework.posttrain.control.loop.weight_sync import weight_sync
    from cosmos_framework.posttrain.utils.entry import resolve, resolve_all, resolve_identical
    from cosmos_framework.posttrain.utils.report import (
        ConsoleReporter,
        emit_metric_frame,
        finish_best_effort,
        reportable_metrics,
        start_best_effort,
    )

    data, rollout, rm, replay, policy = (units[n] for n in ("data", "rollout", "rm", "replay", "policy"))
    # Micro-batches per optimizer step == the trainer's grad_accum. The reward role splits each
    # episode into single ticks, so this is prompts x group x ticks, not prompts x group.
    batch_size = config.prompts_per_step * config.group_size * config.ticks_per_episode
    # The policy's FSDP width, by the same rule `declare_wgs` shapes the mesh with, and the
    # optimizer steps one closed-loop step performs: the step's trajectories, one per rank per
    # step. `make_setup` is handed the same expression so the LR schedule counts in the same unit.
    updates_per_step = _updates_per_step(config)
    # Reporting is a control-plane concern and the loop is its single writer, fed the metrics the
    # trainer already reduced. `ConsoleReporter` is constructed directly rather than through
    # `make_reporters`, which reads `config.job.wandb_mode` off an i4 config this entry does not
    # have; the W&B channel comes with the config migration (milestone 2).
    reporters = [
        ConsoleReporter(
            total_steps=config.steps,
            # `train/learning_rate` is deliberately absent: ConsoleReporter renders at +.4f and this
            # run's LR is 1e-6, so it would read +0.0000 on every step. The schedule is logged once
            # at construction instead (`_build_trainer`).
            headline_keys=(
                "train/weight_version",
                "train/loss_avg",
                "train/ratio_min",
                "train/ratio_max",
                "train/clip_fraction",
                "train/grad_norm",
                # How many micro-batches this rank actually walked. Headline, not buried: it is the
                # one number that shows whether the dispatch edge's DP split is armed -- unarmed,
                # every rank walks the whole draw and this reads double the baseline's `batches=`.
                "train/batches",
                "train/rows",
            ),
        )
    ]
    # `make_reporters` reads `config.job.wandb_mode` off an i4 config this entry does not have, so
    # the list is built here; the W&B channel comes with the config migration.
    failed_reporters: set[int] = set()
    start_best_effort(reporters, failed_reporters=failed_reporters)

    # Sync once before training as a self-check. Both sides still hold the same checkpoint here, so
    # this transfer must be numerically identity, and the rollout body verifies it bucket by bucket
    # (`AlpagymRollout.apply_bucket`) on this first stream only. It costs one sync at startup and
    # turns a corrupt weight plane into an immediate, named failure; without it the symptom is the
    # ratio wandering off over several steps, which reads as a training problem.
    weight_sync(rollout, policy)

    for step in range(config.steps):
        # Handles are FORWARDED, not resolved. `resolve` on a CollectiveHandle returns replica-0
        # rank-0 only (its own docstring), so resolving mid-loop would keep one lane of a mesh whose
        # roles all run at `gpus` replicas. `entrypoints/grpo.py` resolves mid-loop only because it
        # computes driver-side batch statistics, which this loop does not.
        prompts = data.get_samples(config.prompts_per_step)
        _t_rollout = time.perf_counter()
        trajectories = rollout.generate(prompts, group=config.group_size)
        scored = rm.score(trajectories)
        # Await the write: a dropped handle hides a failed one (`utils/entry.py::resolve_all`).
        resolve_all(replay.put_samples(scored))
        # AFTER the resolve, not before: `generate` and `score` return handles and come back
        # immediately, so a boundary above this line credits the whole simulation to the training
        # phase and reports rollout=0.0s.
        _t_train = time.perf_counter()
        # ONE trajectory per train() call, gradient-accumulated -- never one forward over the whole
        # batch. A step's 6 trajectories are 132 transitions, and forwarding them together asks for
        # ~11.6 GiB on the embedding alone; the cosmos-rl baseline ran `mini_batch=1` for the same
        # reason (its log reads `steps=66 batches=66` for 3 rollouts x 22 ticks).
        #
        # ONE draw for the whole step; `select_chunk` does every split. posttrain's units:
        # a MICRO-batch is one forward, a MINI-batch is one optimizer step. So `n_mini` is the
        # optimizer steps this closed-loop step performs and `n_micro` the forwards each
        # accumulates -- here 66 and 1, matching the cosmos-rl baseline's 66 updates per rank of
        # one minibatch each.
        #
        # Passing the batch WHOLE is what makes the slicing deterministic: `select_chunk` sorts by
        # `(group_id, id)` before chunking, so mini-batch i is the same trajectories on every rank
        # and the DP interleave hands each rank a disjoint share. Draining the buffer one draw at
        # a time instead would leave the split to FIFO order.
        #
        # `mini/micro/n_mini/n_micro` also ARM that split: it is opt-in on exactly these kwargs
        # (`workers/dispatch.py::select_chunk`). Without them every FSDP rank receives the whole
        # batch and trains on the same trajectories -- measured as both trainer actors reporting an
        # identical first-batch check, which halves the effective batch and doubles the work.
        batch = replay.get_samples(batch_size)

        # NO pi_old ahead pass here, though the loop runs `n_mini` optimizer steps over one
        # collected batch and that is exactly the shape the pass exists for. The gate
        # (`LLMTrainer.needs_old_logprobs`) names the condition it is FOR: "the loss is not reading
        # rollout logprobs". This objective does read them -- `build_model_inputs` returns the
        # logprob vLLM recorded when it sampled the action, so pi_old is already the policy that
        # collected the batch and is already frozen for the whole window. Recording a second one
        # under eval() would REPLACE that with the current policy's score of the same tokens, which
        # is a different quantity and drifts as the window progresses.
        #
        # Wiring it in anyway was tried and its diagnostics are worth knowing: ratio_min went
        # steady near 0.5, never approaching the surrogate's clamp -- i.e. the pass makes the
        # RATIO look healthier while changing what the ratio measures. Reward on that run did not
        # improve, but a run without the pass reached the same reward, so these 20-step runs do not
        # separate the two; the gate's own stated condition is the reason to leave it off.
        per_batch: list[dict] = []
        for mini in range(updates_per_step):
            # Safe to resolve: `train` returns metrics already reduced across replicas. Resolving
            # each call also keeps the accumulation ordered, which grad_accum requires.
            per_batch.append(
                resolve(
                    policy.train(
                        batch,
                        mini=mini,
                        micro=0,
                        n_mini=updates_per_step,
                        n_micro=1,
                        # This loop's advantage is ONE value broadcast across an episode's 22
                        # steps, so the canonical sort would hand the optimizer eleven consecutive
                        # updates that all push it the same way. cosmos-rl reshuffles its pool
                        # before every pass; `select_chunk` does it here on request. Off by
                        # default there because an entry whose items each carry their own
                        # advantage has no such run to break up.
                        shuffle=True,
                    )
                )
            )

        def _reduce(key: str, how: str) -> float:
            values = [float(m[key]) for m in per_batch if key in m]
            if not values:
                return 0.0
            return {"min": min, "max": max}[how](values) if how in ("min", "max") else sum(values) / len(values)

        metrics: dict = {
            "train/loss_avg": _reduce("train/policy_loss", "mean") + _reduce("train/kl_loss", "mean"),
            "train/policy_loss": _reduce("train/policy_loss", "mean"),
            "train/kl_loss": _reduce("train/kl_loss", "mean"),
            "train/ratio_min": _reduce("train/ratio_min", "min"),
            "train/ratio_max": _reduce("train/ratio_max", "max"),
            "train/clip_fraction": _reduce("train/clip_fraction", "mean"),
            "train/advantage_mean": _reduce("train/advantage_mean", "mean"),
            "train/grad_norm": _reduce("train/grad_norm", "mean"),
            "train/batches": float(len(per_batch)),
            # Rows per micro-batch: what ONE optimizer step actually sees. The baseline's
            # `minibatch rows=1` is one TICK; a trajectory here carries the whole 22-tick episode,
            # so the same "one item per rank" can mean 22x the data per update.
            "train/rows": _reduce("train/rows", "mean"),
        }
        # Rows per micro-batch, logged directly rather than through the reporter: it is the size
        # of ONE optimizer step's view, and the baseline's own `minibatch rows=1` is the number to
        # compare against. A trajectory here is one drive tick, so this should read 1 per rank.
        logging.getLogger(__name__).info(
            "[alpagym] rows/micro-batch: %s",
            sorted({int(m.get("train/rows", -1)) for m in per_batch}),
        )
        _t_sync = time.perf_counter()
        weight_sync(rollout, policy)
        # Phase wall-clock, the breakdown a total step time cannot give: rollout is
        # simulation-bound, train scales with the micro-batch count, sync is the weight
        # plane. Comparing totals against the cosmos-rl baseline says only 'slower'.
        logging.getLogger(__name__).info(
            "[alpagym] step timing rollout=%.1fs train=%.1fs sync=%.1fs",
            _t_train - _t_rollout,
            _t_sync - _t_train,
            time.perf_counter() - _t_sync,
        )
        # The weight version IS the trainer iteration, so after each train + sync the two must agree
        # (`tests/test_comm_weight_sync_asym.py` asserts the same equality). `stamp_weight_version` already
        # refuses to stamp when no backend is bound, and a stream that dies partway raises before the
        # stamp; this catches what neither covers -- a sync that completed but left the rollout on an
        # older iterate. Nothing else in this loop would show it: the metrics, the loss and the
        # checkpoint all look identical while the rollout samples from a policy that no longer exists.
        # `resolve_identical`, not `resolve`: the latter reads replica 0 only, so three of the four
        # rollout lanes went unchecked and a lane left behind by a partial sync would sample stale
        # weights unnoticed. This resolves EVERY cell and rejects rank-divergent values too.
        trainer_iteration = resolve_identical(policy.iteration(), operation="trainer iteration")
        rollout_version = resolve_identical(rollout.weight_version(), operation="rollout weight version")
        if rollout_version != trainer_iteration:
            raise RuntimeError(
                f"step {step}: rollout weight_version {rollout_version} != trainer iteration "
                f"{trainer_iteration} -- the rollout would sample from stale weights"
            )
        # The version rides in the metrics dict, not a separate line: it is the staleness signal
        # (DESIGN 1.4), so the channel that carries metrics to the console -- and to W&B once that
        # is wired -- is the one that should carry it.
        # `emit_metric_frame`, not a bare `report`: it converges a schema rejection across ranks
        # (one rank raising while the others carry on hangs the next collective) and isolates a
        # broken sink, which is what `entrypoints/grpo.py` does at the same seam.
        emit_metric_frame(
            reporters,
            reportable_metrics({**metrics, "train/weight_version": rollout_version}),
            step,
            failed_reporters=failed_reporters,
        )

    # Save once, at the end. `LLMTrainer.save` owns the layout
    # (`iter_*/{model,optim,trainer}.pt` + an atomic `latest_checkpoint.txt`); the run's acceptance
    # check looks for it, and a training run that leaves nothing behind has produced nothing.
    checkpoint_dir = str(Path(config.alpagym.resolved_config_path).parent / "checkpoints")
    # `resolve_all`, not `resolve`: the policy is (1, gpus), so `save` runs on every rank and
    # `resolve` would wait on rank 0 alone -- discarding whatever a peer rank raised while writing.
    resolve_all(policy.save(checkpoint_dir))
    finish_best_effort(reporters, failed_reporters=failed_reporters)
    logging.getLogger(__name__).info("[alpagym] saved checkpoint to %s", checkpoint_dir)

    return {"steps": config.steps, "checkpoint_dir": checkpoint_dir}


def run(config: EntryConfig) -> dict:
    """Bind the two groups and run the loop.

    ONE knob selects explicit device grouping. Colocated puts the two ActorGroups in one GPU
    DeviceGroup (`same_gpu` -> `IpcMover`); disaggregated gives each ActorGroup its own GPU
    DeviceGroup (`cross_device`). The second edge is unservable without a transport -- the NCCL
    weight plane is never wired at bind -- so disaggregated also asks for NIXL.
    """

    from cosmos_framework.posttrain.control.placement import (
        ActorGroup,
        bind_device_groups,
        trainer_experience_device_groups,
    )

    gpu_placement = normalize_gpu_placement(config.placement, field_name="placement")
    wgs = declare_wgs(config.gpus, gpu_placement)
    disaggregated = gpu_placement == "disaggregated"
    updates_per_step = _updates_per_step(config)
    experience_actor = ActorGroup(
        members=tuple(wgs[name] for name in _EXPERIENCE),
        setup=make_setup(config.alpagym.resolved_config_path, config.alpagym.project_root, _EXPERIENCE),
        name="alpagym_experience",
    )
    learner_actor = ActorGroup(
        members=tuple(wgs[name] for name in _LEARNER),
        setup=make_setup(
            config.alpagym.resolved_config_path,
            config.alpagym.project_root,
            _LEARNER,
            # The SAME expression the loop draws with, so the trainer's collected-batch
            # clock and the loop's chunking cannot disagree about how long a batch is.
            updates_per_step=updates_per_step,
        ),
        name="alpagym_learner",
    )
    device_groups = trainer_experience_device_groups(
        experience_actor,
        learner_actor,
        gpu_placement=gpu_placement,
        name="alpagym",
        trainer_name="learner",
    )
    bind_device_groups(
        device_groups,
        backend="ray",
        weight_sync=True,
        # None keeps the relationship table's default, which for `same_gpu` is IPC. `"nixl"` is
        # only meaningful on the cross_device edge, and asking for it there fails loud at bind if
        # the stack is missing rather than at the first sync.
        weight_sync_transport="nixl" if disaggregated else None,
        # temp_dir stays on /tmp: Ray puts its plasma-store UNIX socket under it, and AF_UNIX paths
        # cap at 107 bytes -- the Lustre run directory alone spends most of that. Actor visibility
        # is solved by logging to a file instead (see `make_setup`), not by relocating the session.
    )
    return loop(wgs, config)


def main() -> None:
    """CLI. AlpaGym's host CLI invokes this in place of `cosmos_rl.launcher.launch_all`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", required=True, help="AlpaGym's resolved_config.yaml")
    parser.add_argument("--gpus", type=int, required=True, help="GPUs for the cosmos half")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument(
        "--placement",
        choices=("colocated", "disaggregated"),
        default="colocated",
        help="colocated: every GPU carries an FSDP shard AND an engine (CUDA IPC weight edge). "
        "disaggregated: trainer and rollout split the GPUs in half (NIXL weight edge), the layout "
        "cosmos-rl ran.",
    )
    args = parser.parse_args()

    # Configure logging FIRST. Without a handler every `logger.info` in this entry, in the AlpaGym
    # bodies and in the model wrapper is silently dropped, leaving stack traces as the only
    # evidence of what a run did -- which is how several rounds of this bring-up were debugged
    # blind. `cosmos/entrypoint.py` did this too; it is not optional plumbing.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    # Absolutise BEFORE anything reaches a Ray actor: `main` runs in the container workdir, which
    # IS the AlpaGym project root, so this is the only place where the relative form is meaningful.
    from pathlib import Path

    project_root = str(Path.cwd())
    resolved_config = str(Path(args.resolved_config).resolve())

    # Read the tick count from the run config rather than adding a CLI flag: the loop's iteration
    # count and the trainer's grad_accum both derive from it, and a third hand-passed copy is a
    # third thing that can disagree. `expected_valid_steps` is the same field the run's acceptance
    # check uses.
    from alpagym_host.config import load_run_config

    run_config = load_run_config(Path(resolved_config))
    ticks_per_episode = int(run_config.expected_valid_steps)
    # The loop draws `prompts x group x ticks` micro-batches while the trainer accumulates
    # `train_batch_per_replica x expected_valid_steps`; if those disagree the optimizer steps
    # mid-batch or never. They agree because AlpaGym's launcher derives both CLI values from the
    # same two config fields — check it here, where both sides are visible, rather than leaving a
    # hand-passed value free to break the step silently.
    # `build_lr_scheduler` implements warmup and nothing else. AlpaGym's decay fields default to
    # off ("none"/0.0); a run that turns them on would otherwise train on a schedule nobody wired.
    train_cfg = run_config.cosmos.train
    if str(train_cfg.optm_decay_type) != "none" or float(train_cfg.optm_decay_ratio) != 0.0:
        raise ValueError(
            f"optm_decay_type={train_cfg.optm_decay_type!r} / optm_decay_ratio="
            f"{train_cfg.optm_decay_ratio}: this entry wires warmup only "
            "(config/build.py::build_lr_scheduler); LR decay is not implemented here"
        )
    # Prompts are DERIVED from the batch, not passed. `prompts x group == batch_per_replica` is the
    # conservation law; computing it HERE is what keeps the loop's draw and the trainer's
    # accumulation from disagreeing about when the optimizer steps, instead of trusting a value the
    # launcher computed from the same two fields on its side.
    declared = int(run_config.cosmos.train.train_batch_per_replica)
    if declared % args.group_size:
        raise ValueError(
            f"train_batch_per_replica={declared} is not a multiple of --group-size "
            f"({args.group_size}): a prompt's group cannot be split across optimizer steps, so the "
            "batch must be a whole number of groups"
        )
    prompts_per_step = declared // args.group_size

    run(
        EntryConfig(
            alpagym=AlpagymSection(resolved_config_path=resolved_config, project_root=project_root),
            gpus=args.gpus,
            steps=args.steps,
            prompts_per_step=prompts_per_step,
            group_size=args.group_size,
            ticks_per_episode=ticks_per_episode,
            placement=normalize_gpu_placement(args.placement, field_name="--placement"),
        )
    )


if __name__ == "__main__":
    main()
