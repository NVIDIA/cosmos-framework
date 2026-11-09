# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real-model GPU qualification using SFT's builder, packer, trainer and checkpoints.

Run with torchrun from an installed, byte-attested wheel. This is a test harness,
not a training entrypoint. Production training uses entrypoints.sft.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import zipfile
from pathlib import Path

import torch
import torch.distributed as dist

from cosmos_framework.posttrain.checkpoint.adapters import save_adapter_artifact
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json
from cosmos_framework.posttrain.config.adapters import AdapterExtraModuleConfig
from cosmos_framework.posttrain.control.checkpointing import CheckpointLifecycle
from cosmos_framework.posttrain.control.validation.lifecycle import ValidationTrigger
from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
from cosmos_framework.posttrain.entrypoints.sft import build_trainer, evaluate, train_step, validate_config
from cosmos_framework.posttrain.model.adapters import PosttrainAdapterLinear, assert_trainability, canonical_name
from cosmos_framework.posttrain.roles.env.sft import build_sft_env
from cosmos_framework.posttrain.roles.trainer.validation.runtime import reshard_fsdp_modules
from cosmos_framework.posttrain.tests.adapter_numerics import configure_qualification_numerics
from cosmos_framework.posttrain.utils.dist import init_local_process_group


def installed_environment(wheel: str) -> dict:
    """Fingerprint an isolated host runtime without calling it a container image."""
    return {
        "format": "cosmos-adapter-qualification-environment-v1",
        "python": sys.version,
        "platform": platform.platform(),
        "cuda_runtime": torch.version.cuda,
        "distributions": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
        "wheel_sha256": hashlib.sha256(Path(wheel).read_bytes()).hexdigest(),
    }


def environment_identity(wheel: str) -> dict | None:
    path = os.environ.get("ADAPTER_ENVIRONMENT_MANIFEST")
    if path is None:
        return None
    contents = Path(path).read_bytes()
    expected = json.loads(contents)
    if expected != installed_environment(wheel):
        raise ValueError(
            "qualification environment differs from its sealed manifest; rebuild and requalify the runtime"
        )
    return {"sha256": hashlib.sha256(contents).hexdigest(), "manifest": expected}


def attest(wheel: str, record: str) -> dict:
    build = json.loads(Path(record).read_text())
    path = Path(wheel)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == build["wheel_sha256"]
    distribution = importlib.metadata.distribution("cosmos-framework")
    import cosmos_framework.posttrain.model.adapters as module

    assert (
        Path(module.__file__).resolve()
        == Path(distribution.locate_file("cosmos_framework/posttrain/model/adapters.py")).resolve()
    )
    checked = 0
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.startswith("cosmos_framework/") and not name.endswith("/"):
                assert Path(distribution.locate_file(name)).read_bytes() == archive.read(name), name
                checked += 1
    return {
        **build,
        "container_sha256": os.environ.get("ADAPTER_CONTAINER_SHA256"),
        "environment": environment_identity(wheel),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "verified_package_files": checked,
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "torchcodec")
        },
    }


def local_value(tensor):
    return (tensor.to_local() if hasattr(tensor, "to_local") else tensor).detach().cpu().contiguous()


def observe_optimizer_gradients(net, optimizer):
    """Observe final reduced parameters at the step boundary, including FSDP2.

    FSDP2 differentiates unsharded parameter objects and then assigns .grad to
    the optimizer's DTensors. Autograd hooks on those DTensors do not run.
    """
    names = {id(p): canonical_name(name) for name, p in net.named_parameters() if p.requires_grad}
    records = {name: {"observed": False, "finite": True, "nonzero_local": False} for name in names.values()}

    def observe(opt, args, kwargs):
        for group in opt.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                local = local_value(parameter.grad)
                row = records[names[id(parameter)]]
                row["observed"] = True
                row["finite"] &= bool(torch.isfinite(local).all())
                row["nonzero_local"] |= bool(local.count_nonzero())

    return records, optimizer.register_step_pre_hook(observe)


def frozen_digest(net):
    reshard_fsdp_modules({"net": net})
    digest = hashlib.sha256()
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            value = local_value(parameter)
            digest.update(canonical_name(name).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    for name, buffer in net.named_buffers():
        if name.endswith(("initial_A", "initial_B")):
            digest.update(canonical_name(name).encode())
            digest.update(local_value(buffer).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainable_state(net):
    reshard_fsdp_modules({"net": net})
    return {canonical_name(n): local_value(p).clone() for n, p in net.named_parameters() if p.requires_grad}


def verify_hybrid_replicas(net):
    """Confirm updated HSDP shards agree across replicas, not just shard ranks."""
    reshard_fsdp_modules({"net": net})
    digest = hashlib.sha256()
    group, count = None, 0
    for name, parameter in net.named_parameters():
        mesh = getattr(parameter, "device_mesh", None)
        if not parameter.requires_grad or mesh is None or mesh.ndim != 2:
            continue
        axes = [i for i, placement in enumerate(parameter.placements) if placement.is_replicate()]
        if len(axes) != 1:
            raise ValueError("qualification expected one HSDP replica axis")
        group = mesh.get_group(axes[0])
        digest.update(canonical_name(name).encode())
        digest.update(local_value(parameter).view(torch.uint8).numpy().tobytes())
        count += 1
    if group is None:
        return {"hybrid": False}
    digests = [None] * dist.get_world_size(group)
    dist.all_gather_object(digests, digest.hexdigest(), group=group)
    if len(set(digests)) != 1:
        raise AssertionError("HSDP trainable shards diverged across replica ranks")
    return {"hybrid": True, "trainable_shards": count, "replicas": len(digests), "sha256": digests[0]}


def verify_trainable_updates(before, after):
    """Require each logical tensor to update, including partially supervised shards.

    A physical-action output can contain padded channels. An FSDP shard covering
    only those channels correctly stays unchanged while another shard updates.
    Preserve local evidence and require a change somewhere in each whole tensor.
    """
    from cosmos_framework.posttrain.utils.dist import collective_device, rank_identical

    names = rank_identical(sorted(before), "qualification trainable parameter names")
    if set(before) != set(after):
        raise ValueError("trainable parameter inventory changed during qualification")
    local = {name: not torch.equal(after[name], before[name]) for name in names}
    flags = torch.tensor(
        list(local.values()), dtype=torch.int32, device=collective_device() if dist.is_initialized() else "cpu"
    )
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    global_changes = dict(zip(names, flags.bool().cpu().tolist()))
    missing = [name for name, changed in global_changes.items() if not changed]
    if missing:
        raise AssertionError(f"trainable tensors did not update on any rank: {missing}")
    return local, global_changes


def runtime(recipe, opts, output):
    from cosmos_framework.utils.config import load_config

    config = load_config(recipe, ["--", *opts])
    component = os.environ.get("SFT_QUALIFY_COMPONENT", "all")
    if component != "all":
        config.rl.adapters.components = [c for c in config.rl.adapters.components if c.component == component]
    if os.environ.get("SFT_QUALIFY_EXTRA_NORM") == "1":
        config.rl.adapters.extra_modules = [
            AdapterExtraModuleConfig(component="language", targets=["model.language_model.norm"])
        ]
    rl = validate_config(config)
    coordinates = {"rank": dist.get_rank(), "world_size": dist.get_world_size()}
    env_builder = build_sft_env
    if rl.task == "flow_bc":
        from cosmos_framework.posttrain.roles.env.flow_sft import build_flow_sft_env

        env_builder = build_flow_sft_env
    data = env_builder(rl.data, rl.seed, **coordinates)
    validation = env_builder(rl.validation.data, rl.validation.seed, training=False, **coordinates)
    lifecycle = CheckpointLifecycle.from_config(config, rl, run_out_dir=output)
    trainer = build_trainer(config, rl, data, validation, lifecycle)
    return config, rl, data, validation, trainer


def probe_batches(config, data):
    from cosmos_framework.posttrain.entrypoints.sft import sft_policy_source

    if config.rl.task == "flow_bc":
        from cosmos_framework.posttrain.config.build import select_policy_backend
        from cosmos_framework.posttrain.model.sft_source import build_flow_sft_packer

        packer = build_flow_sft_packer(select_policy_backend(config).native_spec, config.rl)
        saved = copy.deepcopy(data.state_dict())
        batches = []
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(321)
            for _ in range(4):
                batch = packer(data.get_samples(1))
                if hasattr(packer, "controlled_probe"):
                    batch = packer.controlled_probe(batch, seed=321 + len(batches))
                else:
                    batch["noise"] = torch.randn_like(batch["actions"])
                    batch["time"] = torch.full((batch["actions"].shape[0],), 0.3)
                batches.append(batch)
        data.load_state_dict(saved)
        return batches
    source, trust = sft_policy_source(config)
    packer = build_sft_packer(
        source, max_length=config.rl.max_length, processor_kwargs=config.rl.processor_kwargs, trust_remote_code=trust
    )
    saved = copy.deepcopy(data.state_dict())
    batches = []
    for _ in range(4):
        rows = data.get_samples(1)
        samples = [packer.sft_process_sample(row.transitions[0].targets) for row in rows]
        batches.append(dict(packer.sft_collate_fn(samples, packer.sft_compute_max_len(samples))))
    data.load_state_dict(saved)
    return batches


def logits(policy, batches, *, base_only=False):
    net = policy.net
    modes = [(module, module.training) for module in net.modules()]
    adapters = [module for module in net.modules() if isinstance(module, PosttrainAdapterLinear)]
    net.eval()
    if base_only:
        from types import MethodType

        for module in adapters:
            module.forward = MethodType(lambda self, x: torch.nn.functional.linear(x, self.weight, self.bias), module)
    try:
        with torch.no_grad():
            if hasattr(policy, "flow_matching_loss"):
                result = []
                for batch in batches:
                    output = policy.flow_matching_loss(batch)
                    if output.prediction is None:
                        raise ValueError(
                            "flow adapter qualification requires controlled predictions, not only a scalar loss"
                        )
                    result.append(
                        torch.cat((output.prediction.float().flatten(), output.loss.float().reshape(1))).cpu()
                    )
            else:
                result = [
                    policy.forward_policy_batch(policy.prepare_policy_batch(batch)).logits[:, -8:].float().cpu()
                    for batch in batches
                ]
    finally:
        if base_only:
            for module in adapters:
                del module.forward
        for module, mode in modes:
            module.training = mode
        reshard_fsdp_modules({"net": net})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--build-record", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method", choices=["lora", "rslora", "dora", "pissa", "mixed", "configured"], required=True)
    parser.add_argument("--activation-checkpointing", choices=["none", "full", "vision", "selective"], default="full")
    parser.add_argument("--updates", type=int, help="training updates before checkpoint; default consumes eight rows")
    parser.add_argument(
        "--warm-start-check", action="store_true", help="also train a fresh run from its portable artifact"
    )
    parser.add_argument("--component", choices=["all", "language", "projector", "vision", "action"], default="all")
    parser.add_argument("--override", action="append", default=[], help="existing recipe key=value override")
    parser.add_argument("--recipe", default="cosmos_framework/posttrain/configs/sft_cosmos3_nano_adapters.py")
    parser.add_argument("--backend", choices=["i4", "hf"], default="i4")
    args = parser.parse_args()
    if args.backend == "hf" and args.activation_checkpointing not in ("none", "full"):
        parser.error("HF qualification supports only activation checkpointing none/full")
    if args.updates is not None and args.updates < 2:
        parser.error("--updates must be at least two to exercise initially zero LoRA B")
    if args.method == "mixed" and args.component != "all":
        parser.error("the mixed-method qualification requires --component=all")
    evidence = attest(args.wheel, args.build_record)
    assert evidence.get("environment") or (evidence["container_sha256"] and len(evidence["container_sha256"]) == 64), (
        "qualification requires a sealed container or isolated-environment identity"
    )
    evidence["numerical_profile"] = configure_qualification_numerics()
    init_local_process_group()
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    recipe = args.recipe
    os.environ["SFT_ADAPTER_METHOD"] = "lora" if args.method in ("rslora", "mixed", "configured") else args.method
    os.environ["SFT_ADAPTER_RSLORA"] = "1" if args.method == "rslora" else "0"
    os.environ["SFT_QUALIFY_EXTRA_NORM"] = "1" if args.method == "mixed" else "0"
    os.environ["SFT_QUALIFY_COMPONENT"] = args.component
    opts = [
        "rl.resume=off",
        "rl.max_steps=8",
        "rl.data.shuffle=False",
        "rl.keep_optim_last=0",
        "rl.optimizer.lr=0.0001",
        "rl.validation.every_steps=0",
        "rl.validation.before_training=False",
        (
            f"model.model_config.activation_checkpointing={args.activation_checkpointing}"
            if args.backend == "i4"
            else f"model_hf.grad_ckpt={str(args.activation_checkpointing == 'full').lower()}"
        ),
    ]
    if args.method == "mixed":
        opts += [
            "rl.adapters.components.0.dropout=0.1",
            "rl.adapters.components.1.method=dora",
            "rl.adapters.components.1.rank=4",
            "rl.adapters.components.2.method=pissa",
            "rl.adapters.components.2.rank=4",
        ]
    opts += args.override
    started = time.monotonic()
    config, rl, data, validation, trainer = runtime(recipe, opts, str(root))
    initial_cursor = copy.deepcopy(data.state_dict())
    probes = probe_batches(config, data)
    initial = logits(trainer.model, probes)
    base = logits(trainer.model, probes, base_only=True)
    init_error = max((a - b).abs().max().item() for a, b in zip(initial, base))
    for a, b in zip(initial, base):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    before = trainable_state(trainer.model.net)
    frozen = frozen_digest(trainer.model.net)
    assert_trainability(trainer.model.net, trainer.opt)
    placement = {
        canonical_name(name)
        for name, module in trainer.model.net.named_modules(remove_duplicate=False)
        if isinstance(module, PosttrainAdapterLinear)
    }
    assert placement == {row["name"] for row in trainer.model.net._posttrain_adapter_manifest["targets"]}
    parameter_names = {id(p): canonical_name(n) for n, p in trainer.model.net.named_parameters()}
    optimizer_inventory = [
        {
            "names": [parameter_names[id(p)] for p in group["params"]],
            "lr": group["lr"],
            "weight_decay": group["weight_decay"],
        }
        for group in trainer.opt.param_groups
    ]
    gradients, gradient_hook = observe_optimizer_gradients(trainer.model.net, trainer.opt)
    validation_before = evaluate(
        {"validation": validation, "policy": trainer}, rl, 0, frozenset({ValidationTrigger.BEFORE_TRAINING})
    ).metrics
    reports = []
    update_resources = []
    measure_cuda = next(trainer.model.net.parameters()).device.type == "cuda"
    # Eight rows include real text, PNG, MP4 and WTS videos. Rank 0 and rank 1
    # encounter different modality combinations on the same FSDP collective path.
    for _ in range(args.updates or 8 // world):
        if measure_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        update_started = time.monotonic()
        reports.append(train_step(data, trainer, rl))
        if measure_cuda:
            torch.cuda.synchronize()
        update_resources.append(
            {
                "seconds": time.monotonic() - update_started,
                "peak_cuda_allocated": torch.cuda.max_memory_allocated() if measure_cuda else None,
                "peak_cuda_reserved": torch.cuda.max_memory_reserved() if measure_cuda else None,
            }
        )
    replicas_after_updates = verify_hybrid_replicas(trainer.model.net)
    gradient_hook.remove()
    assert all(row["observed"] and row["finite"] for row in gradients.values()), gradients
    after = trainable_state(trainer.model.net)
    changed_locally, changed = verify_trainable_updates(before, after)
    assert frozen_digest(trainer.model.net) == frozen
    assert all(torch.isfinite(value).all() for value in after.values())
    checkpoint = str(root / "checkpoints" / f"iter_{len(reports):06d}")
    trainer.save(checkpoint, data_state=data.state_dict())
    trainer.finalize_checkpoints()
    reports.append(train_step(data, trainer, rl))
    expected = trainable_state(trainer.model.net)
    expected_cursor = copy.deepcopy(data.state_dict())
    expected_scheduler = copy.deepcopy(trainer.lr_scheduler.state_dict()) if trainer.lr_scheduler else None
    expected_optimizer = {
        n: {k: local_value(v).clone() if isinstance(v, torch.Tensor) else v for k, v in trainer.opt.state[p].items()}
        for n, p in trainer.model.net.named_parameters()
        if p.requires_grad
    }
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    resume_opts = [x for x in opts if not x.startswith("rl.resume=")] + [f"rl.resume={checkpoint}"]
    _, _, data, validation, trainer = runtime(recipe, resume_opts, str(root))
    resumed_report = train_step(data, trainer, rl)
    resumed = trainable_state(trainer.model.net)
    resume_error = max((resumed[n] - expected[n]).abs().max().item() if resumed[n].numel() else 0 for n in expected)
    for name in expected:
        torch.testing.assert_close(resumed[name], expected[name], atol=1e-7, rtol=1e-5)
    assert data.state_dict() == expected_cursor
    if trainer.lr_scheduler:
        assert trainer.lr_scheduler.state_dict() == expected_scheduler
    for name, parameter in trainer.model.net.named_parameters():
        if parameter.requires_grad:
            for key, value in trainer.opt.state[parameter].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(local_value(value), expected_optimizer[name][key], atol=1e-7, rtol=1e-5)
    assert_trainability(trainer.model.net, trainer.opt)
    assert frozen_digest(trainer.model.net) == frozen
    validation_after = evaluate(
        {"validation": validation, "policy": trainer}, rl, len(reports), frozenset({ValidationTrigger.FINAL})
    ).metrics
    record = {
        **evidence,
        "method": args.method,
        "component": args.component,
        "overrides": args.override,
        "activation_checkpointing": args.activation_checkpointing,
        "world_size": world,
        "rank": rank,
        "resolved": trainer.model.net._posttrain_adapter_manifest,
        "optimizer_inventory": optimizer_inventory,
        "observed_gradients": gradients,
        "configured_execution": {
            "backend": args.backend,
            "attention": config.model_hf.attn_impl
            if args.backend == "hf"
            else getattr(config.model.model_config, "attn_impl", "model_owned"),
            "dp_replicate": config.model_hf.parallelism.dp_replicate
            if args.backend == "hf"
            else config.model.parallelism.dp_replicate,
            "dp_shard": config.model_hf.parallelism.dp_shard
            if args.backend == "hf"
            else config.model.parallelism.dp_shard,
        },
        "trainable_tensors": len(after),
        "changed_tensors": sum(changed.values()),
        "changed_local_shards": sum(changed_locally.values()),
        "unchanged_local_shards": [name for name, value in changed_locally.items() if not value],
        "initial_max_abs_logit_error": init_error,
        "controlled_output_kind": "flow_velocity_and_loss" if rl.task == "flow_bc" else "language_logits",
        "resume_max_abs_parameter_error": resume_error,
        "frozen_sha256": frozen,
        "train": reports,
        "update_resources": update_resources,
        "replicas_after_updates": replicas_after_updates,
        "replicas_after_resume": verify_hybrid_replicas(trainer.model.net),
        "resumed_update": resumed_report,
        "validation_before": validation_before,
        "validation_after": validation_after,
        "peak_cuda_allocated": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": time.monotonic() - started,
        "status": "training_and_resume_passed_export_pending",
    }
    atomic_write_json(str(root / f"training-rank{rank}.json"), record)
    snapshot = trainable_state(trainer.model.net)
    save_adapter_artifact(trainer.model.net, str(root / "portable"), provenance=evidence)
    exported_state = trainable_state(trainer.model.net)
    assert all(torch.equal(v, exported_state[n]) for n, v in snapshot.items())
    # Fresh-process reload and merged HF inference consume these exact media
    # tensors and outputs after this process exits, so no live module can leak in.
    expected_logits = logits(trainer.model, probes)
    torch.save(
        {
            "batches": probes,
            "logits": expected_logits,
            "identity": trainer.model.net._posttrain_adapter_identity,
            "numerical_profile": evidence["numerical_profile"],
        },
        root / f"probes-rank{rank}.pt",
    )
    record.update(
        status="training_and_resume_passed_reload_pending",
        elapsed_seconds=time.monotonic() - started,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved=torch.cuda.max_memory_reserved(),
    )
    atomic_write_json(str(root / f"training-rank{rank}.json"), record)
    if args.warm_start_check:
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        warm_opts = [x for x in opts if not x.startswith(("rl.resume=", "rl.adapters.warm_start="))]
        warm_opts += ["rl.resume=off", f"rl.adapters.warm_start={root / 'portable'}"]
        _, _, warm_data, _, warm = runtime(recipe, warm_opts, str(root / "warm-start"))
        warm_state = trainable_state(warm.model.net)
        assert all(torch.equal(warm_state[name], value) for name, value in snapshot.items())
        assert frozen_digest(warm.model.net) == frozen
        assert warm.optimizer_is_pristine() and warm._update_state_dict()["optimizer_step"] == 0
        assert warm_data.state_dict() == initial_cursor
        assert_trainability(warm.model.net, warm.opt)
        warm_logits = logits(warm.model, probes)
        for value, reference in zip(warm_logits, expected_logits, strict=True):
            torch.testing.assert_close(value, reference, atol=2e-5, rtol=2e-5)
        warm_reports = [train_step(warm_data, warm, rl) for _ in range(args.updates or 8 // world)]
        warm_after = trainable_state(warm.model.net)
        _, warm_changed = verify_trainable_updates(warm_state, warm_after)
        assert frozen_digest(warm.model.net) == frozen
        assert_trainability(warm.model.net, warm.opt)
        record["warm_start"] = {
            "exact_initial_state": True,
            "fresh_optimizer_and_clock": True,
            "fresh_data_cursor": True,
            "controlled_maximum_error": max(
                (a - b).abs().max().item() for a, b in zip(warm_logits, expected_logits, strict=True)
            ),
            "updates": warm_reports,
            "changed_tensors": sum(warm_changed.values()),
            "final_clock": warm._update_state_dict(),
        }
        atomic_write_json(str(root / f"training-rank{rank}.json"), record)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
