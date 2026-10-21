# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Thin launcher for the parity harness. Three modes: capture a loss reference (`--print-losses`,
printed for pasting into the owning test), run the reference-free overfit canary (`--overfit`, any
model with a fixed-pair env), or the toy trainer-side resume self-check (default). `--model toy` is
the CPU zero-dep tier (the L0 per-commit suite drives the harness directly); `--model cosmos3` builds
the real 8B stack for the @L1 @GPU nightly tiers (loss golden + overfit canary) and the one-off golden
capture on GB200.

Toy resume self-check / overfit canary:
    PYTHONPATH=. python -m cosmos_framework.posttrain.tests.parity_cli --model toy --steps 6 --break-at 3
    PYTHONPATH=. python -m cosmos_framework.posttrain.tests.parity_cli --model toy --overfit
Capture the toy loss reference (paste the printed list into test_parity_harness._TOY_REFERENCE):
    PYTHONPATH=. python -m cosmos_framework.posttrain.tests.parity_cli --model toy --steps 8 --print-losses
8B on GB200 — capture the golden (paste into test_parity_cosmos3._COSMOS3_REFERENCE) or run the canary.
Both run under the determinism stack (see `_enable_determinism`), so results are bit-exact on the same
GB200 + torch + world size; use --nproc_per_node=4 (the golden is N=4-pinned) and prepend PYTHONHASHSEED:
    PYTHONHASHSEED=42 torchrun --nproc_per_node=4 --standalone -m cosmos_framework.posttrain.tests.parity_cli --model cosmos3 --steps 20 --print-losses
    PYTHONHASHSEED=42 torchrun --nproc_per_node=4 --standalone -m cosmos_framework.posttrain.tests.parity_cli --model cosmos3 --overfit --overfit-steps 30 --overfit-floor 0.2
"""

from __future__ import annotations

import argparse
import os
import tempfile

# Toy-stack config (CPU, zero-dep): the toy model/env fixtures reused here to exercise the real
# trainer/objective without a GPU or any data wiring. _LR is deliberately small: a larger step makes
# the toy loss oscillate by orders of magnitude per step (a chaotic map), which amplifies harmless
# floating-point differences across platforms/threads until the pinned golden no longer matches within
# tolerance. A small _LR keeps the trajectory smooth so the golden is reproducible across boxes.
_DIM, _BETA, _LR, _TRAIN_SEED, _BASE_SEED = 8, 100.0, 1e-3, 1, 2

# Pinned config for the 8B parity tier: frozen ref, fusedadam, consolidated (same world size for run
# + resume), fixed-seed synthetic pairs (no external data assets).
_COSMOS3_EXPERIMENT = "cosmos3_ga_16bm8b_dpo_v1"
_COSMOS3_OVERRIDES = [
    "rl.optimizer.name=fusedadam",
    "rl.optimizer.lr=5e-6",
    "rl.optimizer.betas=[0.9,0.99]",
    "rl.optimizer.train_keys=[]",
    "rl.ref.update_every=0",
    "rl.checkpoint_format=consolidated",
    "rl.data.synthetic_pairs=8",
    "rl.data.seed=0",
]


def _toy_build_trainer(seed: int = _TRAIN_SEED, grad_accum: int = 1):
    import torch

    from cosmos_framework.posttrain.algorithm.dpo import make_objective
    from cosmos_framework.posttrain.roles.trainer.diffusion import DiffusionTrainer
    from cosmos_framework.posttrain.tests.helpers.toy_model import ToyFlowModel

    model = ToyFlowModel(dim=_DIM, seed=seed)
    opt = torch.optim.AdamW(model.net.parameters(), lr=_LR)
    return DiffusionTrainer(model, make_objective(beta=_BETA), opt, grad_accum=grad_accum)


def _toy_build_env():
    """Varying, seeded source for loss parity (two fresh instances yield identical streams)."""
    from cosmos_framework.posttrain.tests.helpers.preference_env import SyntheticPreferenceEnv

    return SyntheticPreferenceEnv(dim=_DIM, seed=0)


def _toy_build_fixed_env():
    """Cursor-free source for the overfit canary and the trainer-side resume check."""
    from cosmos_framework.posttrain.tests.helpers.preference_env import FixedPairEnv

    return FixedPairEnv(dim=_DIM, seed=0, n_pairs=2)


def _cosmos3_config(extra_overrides: list[str] | None = None):
    from cosmos_framework.utils.config import load_config
    from cosmos_framework.posttrain.utils.dist import world_size_or_1

    # FSDP-shard the 8B across the torchrun world (pure data-parallel shard, like the DPO entry's
    # local path); world size is read AFTER init_local_process_group() has joined the NCCL group.
    shard = [f"model.config.parallelism.data_parallel_shard_degree={world_size_or_1()}"]
    config = load_config(
        "cosmos_framework/posttrain/config/config_dpo.py",
        ["--", f"experiment={_COSMOS3_EXPERIMENT}", *_COSMOS3_OVERRIDES, *(extra_overrides or []), *shard],
    )
    # Determinism needs the eager path (Blackwell FMHA has no deterministic kernel under torch.compile)
    # and the parity tier doesn't need compile, so force it off.
    mcfg = config.model.config
    if "compile" in mcfg and "enabled" in mcfg.compile:
        mcfg.compile.enabled = False
    return config


def _cosmos3_build_trainer(grad_accum: int = 1):
    """The real 8B stack (build_backbone -> Cosmos3FlowPolicy -> fusedadam) via the DPO entry's own
    assembly under the pinned config. GPU-only; exercised by the @L1 tier and the golden capture."""
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.entrypoints.dpo import build_trainer
    from cosmos_framework.posttrain.utils.dist import world_size_or_1

    config = _cosmos3_config([f"rl.grad_accum={grad_accum}"])
    return build_trainer(config, instantiate(config.rl), world_size_or_1())


def _cosmos3_build_env():
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.entrypoints.dpo import build_env

    return build_env(instantiate(_cosmos3_config().rl))


def _cosmos3_build_fixed_env():
    """Cursor-free fixed-pair source for the 8B overfit canary: one fixed synthetic pair PER RANK
    (`synthetic_pairs == world`, so the DistributedSampler-style shard gives each rank exactly one, and
    `shuffle=false` keeps it fixed across steps). The loop memorizes that tiny set, so the loss must
    collapse — a reference-free "does it learn" check (no golden; N- and torch-agnostic)."""
    from cosmos_framework.utils.lazy_config import instantiate
    from cosmos_framework.posttrain.entrypoints.dpo import build_env
    from cosmos_framework.posttrain.utils.dist import world_size_or_1

    extra = [f"rl.data.synthetic_pairs={world_size_or_1()}", "rl.data.shuffle=false"]
    return build_env(instantiate(_cosmos3_config(extra).rl))


def _enable_determinism() -> None:
    """Set PyTorch's determinism knobs so the 8B loss is bit-exact across reruns on the same GPU +
    torch build. Must run before any CUDA / flash-attn init; compile is forced off separately (in
    `_cosmos3_config`), since Blackwell FMHA has no deterministic kernel under torch.compile.

    Delegates to `utils.determinism.enable_determinism` — the single source of truth for this
    recipe (hoisted from here so `RunConfig.deterministic` and the GRPO entry share it too); this
    thin wrapper stays so existing call sites / docs in this module keep working unchanged."""
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enable_determinism()


def _is_rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _builders(model: str, grad_accum: int = 1):
    """(build_trainer, build_env, build_fixed_env) for a model. `grad_accum` is baked into the trainer
    (the harness reads it back off the built trainer — one knob). `build_fixed_env` is the cursor-free
    source for the overfit canary (both tiers) and the toy resume check; cosmos3's fixed env feeds only
    the overfit canary for now (its resume leg is not wired yet)."""
    if model == "toy":
        return (lambda: _toy_build_trainer(grad_accum=grad_accum)), _toy_build_env, _toy_build_fixed_env
    if model == "cosmos3":
        return (lambda: _cosmos3_build_trainer(grad_accum=grad_accum)), _cosmos3_build_env, _cosmos3_build_fixed_env
    raise SystemExit(f"unknown model {model!r}")


def _cli(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Loss + checkpoint-resume parity launcher")
    p.add_argument("--model", choices=["toy", "cosmos3"], default="toy")
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--break-at", type=int, default=0, help="resume-parity break step (0 => steps // 2)")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--pairs-per-step", type=int, default=1)
    p.add_argument("--overfit", action="store_true", help="run the reference-free overfit canary (toy or cosmos3)")
    p.add_argument("--overfit-steps", type=int, default=60)
    p.add_argument("--overfit-floor", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--print-losses",
        action="store_true",
        help="run the trajectory and print the per-step losses to paste into a test's reference constant",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _cli(argv)
    # The 8B FSDP-shards across the torchrun world — join the NCCL group before building, and destroy it
    # on exit (mirroring dpo.py:run()). The toy tier is single-process CPU and needs none of this.
    if args.model != "cosmos3":
        return _dispatch(args)
    from cosmos_framework.posttrain.utils.dist import destroy_process_group_if_initialized, init_local_process_group

    # Determinism must be set before NCCL/CUDA init so capture and the nightly test agree bit-for-bit.
    _enable_determinism()
    init_local_process_group()
    try:
        return _dispatch(args)
    finally:
        destroy_process_group_if_initialized()


def _dispatch(args) -> int:
    from cosmos_framework.posttrain.tests.helpers.parity import assert_overfit, assert_resume_parity, run_trajectory

    build_trainer, build_env, build_fixed_env = _builders(args.model, args.grad_accum)

    # Capture: run the trajectory and print the per-step losses to paste into the owning test's inline
    # reference constant (the toy L0 golden, or the 8B golden captured once on GB200).
    if args.print_losses:
        losses = run_trajectory(
            build_trainer,
            build_env,
            args.steps,
            pairs_per_step=args.pairs_per_step,
            seed=args.seed,
        )
        if _is_rank_zero():  # FSDP reduces the loss, so every rank has the same list — print it once
            import torch

            from cosmos_framework.posttrain.utils.dist import world_size_or_1

            # World size + torch pin the golden (FSDP reduction order + kernel selection) — record both.
            print(
                f"# {args.model} reference: {args.steps} steps, seed {args.seed}, "
                f"world_size {world_size_or_1()}, torch {torch.__version__} (paste into the test)"
            )
            print(repr(losses))
        return 0

    # Overfit canary: reference-free "does the assembled loop learn" on a fixed pair set. Model-agnostic
    # (toy + cosmos3) — the signal that survives a torch/world change with no re-capture, catching
    # signal-destroying bugs (dropped prefactor, wrong sign, bad cast) a bit-match golden can hide.
    if args.overfit:
        try:
            ov = assert_overfit(
                build_trainer, build_fixed_env, steps=args.overfit_steps, floor=args.overfit_floor, seed=args.seed
            )
            ok, detail = True, f"loss {ov[0]:.4f} -> {ov[-1]:.6f} (floor {args.overfit_floor})"
        except AssertionError as e:
            ok, detail = False, str(e).splitlines()[0]
        if _is_rank_zero():
            print(f"[{'PASS' if ok else 'FAIL'}] overfit ({args.model}): {detail}")
            print("PARITY OK" if ok else "PARITY FAILED")
        return 0 if ok else 1

    # Default self-check: trainer-side resume parity (net/opt/iter/RNG round-trip via save→rebuild→
    # restore). Toy only — the base anchor is a toy checkpoint; the 8B tier is --print-losses (golden)
    # + --overfit (canary).
    if args.model != "toy":
        raise SystemExit(f"{args.model!r}: use --print-losses (golden) or --overfit (canary)")
    break_at = args.break_at or max(1, args.steps // 2)
    with tempfile.TemporaryDirectory() as tmp:
        base_ckpt = os.path.join(tmp, "base", "iter_000000")
        _toy_build_trainer(seed=_BASE_SEED).save(base_ckpt)
        try:
            assert_resume_parity(
                build_trainer,
                build_fixed_env,
                total_steps=args.steps,
                break_at=break_at,
                tmp_dir=tmp,
                base_ckpt=base_ckpt,
                pairs_per_step=args.pairs_per_step,
                seed=args.seed,
            )
            ok, detail = True, "bit-equal resume tail"
        except AssertionError as e:
            ok, detail = False, str(e).splitlines()[0]
    print(f"[{'PASS' if ok else 'FAIL'}] resume_parity: {detail}")
    print("PARITY OK" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
