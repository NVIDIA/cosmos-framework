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
from cosmos_framework.posttrain.utils.dist import init_local_process_group
from cosmos_framework.utils.config import load_config


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
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "verified_package_files": checked,
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "safetensors", "torchcodec")
        },
    }


def local_value(tensor):
    return (tensor.to_local() if hasattr(tensor, "to_local") else tensor).detach().cpu().contiguous()


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


def runtime(recipe, opts, output):
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
    data = build_sft_env(rl.data, rl.seed, **coordinates)
    validation = build_sft_env(rl.validation.data, rl.validation.seed, training=False, **coordinates)
    lifecycle = CheckpointLifecycle.from_config(config, rl, run_out_dir=output)
    trainer = build_trainer(config, rl, data, validation, lifecycle)
    return config, rl, data, validation, trainer


def probe_batches(config, data):
    packer = build_sft_packer(
        config.model.model_id, max_length=config.rl.max_length, processor_kwargs=config.rl.processor_kwargs
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
    parser.add_argument("--method", choices=["lora", "rslora", "dora", "pissa", "mixed"], required=True)
    parser.add_argument("--activation-checkpointing", choices=["none", "full"], default="full")
    parser.add_argument("--component", choices=["all", "language", "projector", "vision"], default="all")
    parser.add_argument("--override", action="append", default=[], help="existing recipe key=value override")
    args = parser.parse_args()
    if args.method == "mixed" and args.component != "all":
        parser.error("the mixed-method qualification requires --component=all")
    evidence = attest(args.wheel, args.build_record)
    assert evidence["container_sha256"] and len(evidence["container_sha256"]) == 64
    init_local_process_group()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    rank, world = dist.get_rank(), dist.get_world_size()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    recipe = "cosmos_framework/posttrain/configs/sft_cosmos3_nano_adapters.py"
    os.environ["SFT_ADAPTER_METHOD"] = "lora" if args.method in ("rslora", "mixed") else args.method
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
        f"model.model_config.activation_checkpointing={args.activation_checkpointing}",
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
    probes = probe_batches(config, data)
    initial = logits(trainer.model, probes)
    base = logits(trainer.model, probes, base_only=True)
    init_error = max((a - b).abs().max().item() for a, b in zip(initial, base))
    for a, b in zip(initial, base):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    before = trainable_state(trainer.model.net)
    frozen = frozen_digest(trainer.model.net)
    assert_trainability(trainer.model.net, trainer.opt)
    validation_before = evaluate(
        {"validation": validation, "policy": trainer}, rl, 0, frozenset({ValidationTrigger.BEFORE_TRAINING})
    ).metrics
    reports = []
    # Eight rows include real text, PNG, MP4 and WTS videos. Rank 0 and rank 1
    # encounter different modality combinations on the same FSDP collective path.
    for _ in range(8 // world):
        reports.append(train_step(data, trainer, rl))
    after = trainable_state(trainer.model.net)
    changed = {name: not torch.equal(after[name], before[name]) for name in before}
    assert all(changed.values()), [n for n, changed in changed.items() if not changed]
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
        "trainable_tensors": len(after),
        "changed_tensors": sum(changed.values()),
        "initial_max_abs_logit_error": init_error,
        "resume_max_abs_parameter_error": resume_error,
        "frozen_sha256": frozen,
        "train": reports,
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
        {"batches": probes, "logits": expected_logits, "identity": trainer.model.net._posttrain_adapter_identity},
        root / f"probes-rank{rank}.pt",
    )
    record.update(
        status="training_and_resume_passed_reload_pending",
        elapsed_seconds=time.monotonic() - started,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved=torch.cuda.max_memory_reserved(),
    )
    atomic_write_json(str(root / f"training-rank{rank}.json"), record)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
