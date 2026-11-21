# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Opt-in CUDA mechanism diagnostic; no native feature or end-to-end speed claim.

The seven real Qwen3-8B linear shapes are wired into a small residual graph,
not a transformer/attention model. Two blocks expose cross-block saved-storage
lifetime. A failed backward aborts this process: in-process retry is unsupported.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import json
import os
import statistics
import time
import zipfile
from collections import Counter
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils import checkpoint as cp

SHAPES = ((4096, 4096), (1024, 4096), (1024, 4096), (4096, 4096), (12288, 4096), (12288, 4096), (4096, 12288))
BLOCK_BYTES = sum(out * inp * 2 for out, inp in SHAPES)
BUILD_RECEIPT_SHA = "32b2b8486c8a51f7070da3d13cd3168fa78d1210eec61c9527e29f126a3213a6"
WHEEL_SHA = "55173a4c95d63a318155a7f5568b34ffa1b0ee1dca8f875ab8f098841893f85c"
PINNED_SOURCES = {
    "bitsandbytes.autograd._functions": "8c898f91e97229432b30f884a11d3a6abfe33267d7f389e9cc96a14a94a942d1",
    "bitsandbytes.nn.modules": "6190e06473afd89bbf9d3520641c31ea51bd7eef9155e3c220916d700aeb17ae",
    "bitsandbytes.functional": "42f5364b350ca9f9de50f4a818070a663db60aefc3daa070c83f896cd4ca97ed",
    "torch.utils.checkpoint": "bce92715ec3c0d3941dd74c939396686632457a2c7cf98247f3f67b351e97309",
    "torch.distributed.algorithms._checkpoint.checkpoint_wrapper": "d0211372563f320df88630dd553aba97fecf3e8728f42c6afc4db41b6467f1d1",
}
NATIVE_MODULES = (
    "cosmos_framework.posttrain.model.quantization.bnb",
    "cosmos_framework.model.generator.parallelize_vlm",
    "cosmos_framework.utils.generator.lora",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def require_policy(mode, compiled=False):
    if mode != "full" or compiled:
        raise ValueError("saved-dequant fixture admits only eager full non-reentrant checkpointing")


def require_device(name, memory_bytes, capability, bf16):
    if capability != (8, 0) or "A100" not in name or memory_bytes < 79_000_000_000 or not bf16:
        raise ValueError("saved-dequant diagnostic requires the qualified A100 80GB with BF16 kernels")


def implementation_pins():
    mode = os.environ.get("COSMOS_SAVED_DEQUANT_IMPLEMENTATION", "fixture")
    if mode == "fixture":
        return False, BUILD_RECEIPT_SHA, WHEEL_SHA
    if mode != "native":
        raise ValueError("saved-dequant implementation must be fixture or native")
    pins = tuple(
        os.environ[name] for name in ("COSMOS_SAVED_DEQUANT_BUILD_RECEIPT_SHA", "COSMOS_SAVED_DEQUANT_WHEEL_SHA")
    )
    if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in pins):
        raise ValueError("native fixture requires externally pinned build and wheel SHA256")
    return True, *pins


def verify_sources(build, *, build_receipt_sha=BUILD_RECEIPT_SHA, wheel_sha=WHEEL_SHA, production=False):
    """Parent verifies the whole build; independently bind the touched source here."""
    build = Path(build).resolve()
    receipt = build / "receipt.json"
    assert digest(receipt) == build_receipt_sha
    wheels = list(build.glob("*.whl"))
    assert len(wheels) == 1 and digest(wheels[0]) == wheel_sha
    paths = {str(receipt): build_receipt_sha, str(wheels[0]): wheel_sha}
    native_modules = NATIVE_MODULES + (
        ("cosmos_framework.posttrain.config.model", "cosmos_framework.posttrain.model.quantization.loader")
        if production
        else ()
    )
    with zipfile.ZipFile(wheels[0]) as archive:
        for name in native_modules:
            relative = name.replace(".", "/") + ".py"
            path = build / "site" / relative
            assert path.read_bytes() == archive.read(relative), name
            paths[str(path)] = digest(path)
    for name in native_modules:
        module = importlib.import_module(name)
        expected = build / "site" / (name.replace(".", "/") + ".py")
        assert Path(module.__file__).resolve() == expected.resolve(), name
    for name, expected in PINNED_SOURCES.items():
        path = Path(importlib.import_module(name).__file__).resolve()
        assert digest(path) == expected, name
        paths[str(path)] = expected
    return paths


class DenseTracker:
    """StorageImpl weak handles do not keep either Python tensors or CUDA storage alive."""

    def __init__(self, states):
        self.states = states
        self.refs = []
        self.counts = Counter()
        self.phase = "forward"
        self.peak_bytes = 0
        self.peak_blocks = []
        self.max_simultaneous_blocks = 0

    def live(self):
        alive = [(ref.cdata, size, block) for ref, size, block in self.refs if not ref.expired()]
        unique = {handle: (size, block) for handle, size, block in alive}
        size = sum(size for size, _ in unique.values())
        self.max_simultaneous_blocks = max(self.max_simultaneous_blocks, len({block for _, block in unique.values()}))
        if size > self.peak_bytes:
            self.peak_bytes = size
            self.peak_blocks = sorted({block for _, block in unique.values()})
        return size

    @contextlib.contextmanager
    def instrument(self, functional):
        original = functional.dequantize_4bit

        def observed(*args, **kwargs):
            state = kwargs.get("quant_state", args[1] if len(args) > 1 else None)
            assert id(state) in self.states, "unexpected matrix/QuantState in diagnostic"
            value = original(*args, **kwargs)
            block, name = self.states[id(state)]
            self.refs.append((StorageWeakRef(value.untyped_storage()), value.untyped_storage().nbytes(), block))
            self.counts[(self.phase, block, name)] += 1
            self.live()
            return value

        functional.dequantize_4bit = observed
        try:
            yield self
        finally:
            functional.dequantize_4bit = original

    def report(self):
        return {
            "peak_live_dense_storage_bytes": self.peak_bytes,
            "blocks_at_peak": self.peak_blocks,
            "max_simultaneous_blocks": self.max_simultaneous_blocks,
            "counts": {"/".join(map(str, key)): value for key, value in sorted(self.counts.items())},
            "total_dequantizations": sum(self.counts.values()),
        }


def saved_forward(module, x, native_forward, dequantize):
    # Preserve upstream inference, including its no-grad single-vector GEMV.
    if not module.training or not torch.is_grad_enabled():
        return native_forward(x)
    assert module.fixture_full_checkpoint, "unwrapped candidate is not admitted"
    if x.dtype != torch.bfloat16 or x.device != module.weight.device:
        raise ValueError("native BF16/device contract")
    # Only dequantization is detached. F.linear must participate in autograd
    # so the actual BF16 weight uses checkpoint's saved-tensor hooks.
    with torch.no_grad():
        weight = dequantize(module.weight.data, module.weight.quant_state)
    assert tuple(weight.shape) == module.logical_shape and weight.dtype == x.dtype
    base = F.linear(x, weight)
    return base + module.adapter_delta(x, output_dtype=base.dtype)


def make_pair(device, *, production=False):
    import bitsandbytes as bnb

    from cosmos_framework.configs.base.defaults.activation_checkpointing import ActivationCheckpointingConfig
    from cosmos_framework.model.generator.parallelize_vlm import _apply_full_ac
    from cosmos_framework.posttrain.config.model import NativeLoRAConfig, QuantizedBaseTrainingConfig
    from cosmos_framework.posttrain.model.quantization.bnb import NativeLoraLinear4bit

    class SavedWeight(NativeLoraLinear4bit):
        def forward(self, x):
            return saved_forward(self, x, super().forward, bnb.functional.dequantize_4bit)

    class Block(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = torch.nn.ModuleList(layers)

        def forward(self, x):
            q, k, v, o, gate, up, down = self.layers
            # Shape-correct synthetic graph; no attention/transformer equivalence claim.
            h = (q(x) + k(x).repeat_interleave(4, -1) + v(x).repeat_interleave(4, -1)) / 3
            x = x + o(h) * 0.125
            return x + down(F.silu(gate(x)) * up(x)) * 0.125

    generator = torch.Generator().manual_seed(20261103)
    pairs = [[], []]
    for _ in range(2):
        layers = [[], []]
        for out_features, in_features in SHAPES:
            source = (
                torch.randn(out_features, in_features, generator=generator, dtype=torch.bfloat16) / in_features**0.5
            )
            a = torch.randn(32, in_features, generator=generator) * 0.01
            b = torch.randn(out_features, 32, generator=generator) * 0.01
            candidate = NativeLoraLinear4bit if production else SavedWeight
            for index, cls in enumerate((NativeLoraLinear4bit, candidate)):
                options = {"quantized_linear_execution": "checkpointed_dequant"} if production and index else {}
                module = cls(
                    torch.nn.Linear(in_features, out_features, bias=False, device="meta"),
                    NativeLoRAConfig(rank=32, alpha=64),
                    QuantizedBaseTrainingConfig(),
                    **options,
                )
                module.load_dense_weight_once(source.clone(), device)
                module.lora_A.weight = torch.nn.Parameter(a.to(device).clone())
                module.lora_B.weight = torch.nn.Parameter(b.to(device).clone())
                if not production:
                    module.fixture_full_checkpoint = False
                layers[index].append(module)
            del source
        for index in range(2):
            pairs[index].append(Block(layers[index]))
    stacks = []
    for arm, blocks in enumerate(pairs):
        require_policy("full")
        wrapped = [_apply_full_ac(block, ActivationCheckpointingConfig(mode="full")) for block in blocks]
        assert_wrappers(wrapped)
        if production and arm:
            from cosmos_framework.posttrain.model.quantization.bnb import checkpointed_dequant_contexts

            for wrapper, block in zip(wrapped, blocks, strict=True):
                wrapper.checkpoint_fn = partial(
                    cp.checkpoint,
                    **wrapper.checkpoint_fn.keywords,
                    context_fn=partial(checkpointed_dequant_contexts, tuple(block.layers)),
                )
        elif not production:
            for block in blocks:
                for module in block.layers:
                    module.fixture_full_checkpoint = True
        assert_wrappers(wrapped)
        stacks.append(torch.nn.Sequential(*wrapped))
    return stacks, pairs


def assert_wrappers(wrapped):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, CheckpointWrapper

    for block in wrapped:
        assert type(block) is CheckpointWrapper
        assert block.checkpoint_impl == CheckpointImpl.NO_REENTRANT
        fn = block.checkpoint_fn
        assert fn.func is cp.checkpoint and not fn.args
        expected = {"use_reentrant": False, "preserve_rng_state": True, "determinism_check": "default"}
        if "context_fn" in fn.keywords:
            context = fn.keywords["context_fn"]
            assert type(context) is partial
            from cosmos_framework.posttrain.model.quantization.bnb import checkpointed_dequant_contexts

            members = tuple(block._checkpoint_wrapped_module.layers)
            assert context.func is checkpointed_dequant_contexts
            assert context.args == (members,) and not context.keywords
            assert len(members) == 7 and all(m.quantized_linear_execution == "checkpointed_dequant" for m in members)
            expected["context_fn"] = context
        assert fn.keywords == expected


def production_guards(blocks, x):
    """Exercise the actual CUDA class, including inference and rejected raw calls."""
    from cosmos_framework.posttrain.config.model import NativeLoRAConfig, QuantizedBaseTrainingConfig
    from cosmos_framework.posttrain.model.quantization.bnb import (
        NativeLoraLinear4bit,
        checkpointed_dequant_contexts,
        kernel_preflight,
    )

    left, right = blocks[0][0].layers[0], blocks[1][0].layers[0]
    assert type(left) is type(right) is NativeLoraLinear4bit
    assert right.quantized_linear_execution == "checkpointed_dequant"
    for grad in (False, True):
        with pytest.raises(RuntimeError, match="block scope"):
            right(x.detach().requires_grad_(grad))
    with checkpointed_dequant_contexts(tuple(blocks[1][1].layers))[0]:
        with pytest.raises(RuntimeError, match="block scope"):
            right(x)
    with torch.no_grad():
        torch.testing.assert_close(right(x), left(x), rtol=0, atol=0)
    left.eval()
    right.eval()
    try:
        with torch.no_grad():
            torch.testing.assert_close(right(x[:, :1]), left(x[:, :1]), rtol=0, atol=0)
    finally:
        left.train()
        right.train()
    base = torch.nn.Linear(128, 128, bias=False, device="meta")
    for options in (
        {"quantized_linear_execution": "unknown"},
        {"quantized_linear_execution": "checkpointed_dequant", "adapter_compute_dtype": torch.bfloat16},
    ):
        with pytest.raises(ValueError):
            NativeLoraLinear4bit(base, NativeLoRAConfig(rank=32, alpha=64), QuantizedBaseTrainingConfig(), **options)
    with pytest.raises(ValueError, match="eager full"):
        kernel_preflight(
            x.device,
            NativeLoRAConfig(rank=32, alpha=64),
            QuantizedBaseTrainingConfig(),
            quantized_linear_execution="checkpointed_dequant",
            compile_model=True,
        )
    return {
        "actual_native_class": True,
        "raw_and_foreign_scope_rejected": True,
        "no_grad_batch_and_eval_single_vector_exact": True,
        "invalid_execution_bf16_adapter_and_compile_rejected": True,
    }


def adapters(net):
    return [parameter for parameter in net.parameters() if parameter.requires_grad]


def frozen_state(blocks):
    from cosmos_framework.posttrain.model.quantization.loader import quantized_state_record

    records, aliases = [], []
    for block in blocks:
        for module in block.layers:
            state = module.weight.quant_state
            records.append(quantized_state_record(module))
            tensors = [module.weight, state.absmax, state.code, state.offset, state.state2.absmax, state.state2.code]
            aliases.append([id(module.weight), id(state), id(state.state2), *[(id(t), t.data_ptr()) for t in tensors]])
            assert module.quant_state is state and module.weight.grad is None
    return records, aliases


def difference(actual, expected, *, rtol, atol):
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    result = {
        "exact": torch.equal(actual, expected),
        "max_abs": delta.abs().max().item(),
        "relative_l2": (delta.norm() / expected.norm().clamp_min(1e-30)).item(),
        "rtol": rtol,
        "atol": atol,
    }
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    assert result["relative_l2"] <= rtol, result
    return result


def step(net, x, optimizer):
    optimizer.zero_grad(set_to_none=True)
    x.grad = None
    output = net(x)
    loss = output.float().square().mean()
    loss.backward()
    optimizer.step()
    return output


def correct_pair(stacks, blocks, source, *, input_grad, early_stop):
    import bitsandbytes as bnb

    initial = [p.detach().clone() for p in adapters(stacks[0])]
    for net in stacks:
        for parameter, value in zip(adapters(net), initial, strict=True):
            parameter.data.copy_(value)
    optimizers = [torch.optim.AdamW(adapters(net), lr=1e-4, fused=True) for net in stacks]
    states = [
        {id(m.weight.quant_state): (i, j) for i, b in enumerate(arm) for j, m in enumerate(b.layers)} for arm in blocks
    ]
    report = []
    for iteration in range(3):
        observed = []
        for net, optimizer, owned in zip(stacks, optimizers, states, strict=True):
            assert_wrappers(net)
            x = source.detach().clone().requires_grad_(input_grad)
            optimizer.zero_grad(set_to_none=True)
            tracker = DenseTracker(owned)
            with cp.set_checkpoint_early_stop(early_stop), tracker.instrument(bnb.functional):
                output = net(x)
                after_forward = tracker.live()
                assert after_forward == 0, "dense storage survived checkpoint original forward"
                tracker.phase = "backward"
                output.float().square().mean().backward()
                torch.cuda.synchronize()
                assert tracker.live() == 0, "dense storage survived successful backward"
            grads = [p.grad.detach().cpu().clone() for p in adapters(net)]
            assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in grads)
            before = [p.detach().clone() for p in adapters(net)]
            optimizer.step()
            changed = sum(torch.count_nonzero(a != b).item() for a, b in zip(adapters(net), before, strict=True))
            assert changed > 0
            observed.append(
                {
                    "output": output.detach().cpu(),
                    "input": None if x.grad is None else x.grad.detach().cpu(),
                    "grads": grads,
                    "parameters": [p.detach().cpu().clone() for p in adapters(net)],
                    "tracker": tracker.report(),
                    "changed_adapter_elements": changed,
                }
            )
            del output, x, before
            gc.collect()
            assert tracker.live() == 0
        baseline, candidate = observed
        assert candidate["tracker"]["peak_live_dense_storage_bytes"] <= BLOCK_BYTES
        assert candidate["tracker"]["max_simultaneous_blocks"] == 1
        assert candidate["tracker"]["total_dequantizations"] < baseline["tracker"]["total_dequantizations"]
        comparisons = {
            "output": difference(candidate["output"], baseline["output"], rtol=0.02, atol=0.02),
            "adapter_gradients": [
                difference(a, b, rtol=0.03, atol=0.00002)
                for a, b in zip(candidate["grads"], baseline["grads"], strict=True)
            ],
            "parameters": [
                difference(a, b, rtol=0.003, atol=0.000002)
                for a, b in zip(candidate["parameters"], baseline["parameters"], strict=True)
            ],
        }
        if input_grad:
            comparisons["input_gradient"] = difference(candidate["input"], baseline["input"], rtol=0.03, atol=0.0002)
        else:
            assert baseline["input"] is candidate["input"] is None
        report.append(
            {
                "iteration": iteration + 1,
                "comparisons": comparisons,
                "arms": [{k: arm[k] for k in ("tracker", "changed_adapter_elements")} for arm in observed],
            }
        )
    del initial, optimizers
    return {"input_requires_grad": input_grad, "early_stop": early_stop, "steps": report}


def snapshot():
    values = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()}
    try:
        values["host_status"] = {
            line.split(":")[0]: line.split(":", 1)[1].strip()
            for line in Path("/proc/self/smaps_rollup").read_text().splitlines()
            if ":" in line
        }
    except OSError as error:
        values["host_status"] = {"unavailable": repr(error)}
    return values


def timings(stacks, source):
    """Uninstrumented ABBA warm block timings; distinct from StorageWeakRef runs."""
    initial = [p.detach().clone() for p in adapters(stacks[0])]
    rounds = []
    for arm in (0, 1, 1, 0):
        net = stacks[arm]
        for parameter, value in zip(adapters(net), initial, strict=True):
            parameter.data.copy_(value)
        optimizer = torch.optim.AdamW(adapters(net), lr=1e-4, fused=True)
        x = source.detach().clone().requires_grad_(True)
        for _ in range(2):
            del_output = step(net, x, optimizer)
            del del_output
        torch.cuda.synchronize()
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        before = snapshot()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(5):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            wall = time.perf_counter()
            start.record()
            output = step(net, x, optimizer)
            end.record()
            end.synchronize()
            samples.append(
                {"wall_seconds": time.perf_counter() - wall, "cuda_stream_seconds": start.elapsed_time(end) / 1000}
            )
            del output
        rounds.append(
            {
                "arm": ("bnb", "saved_weight")[arm],
                "warmup_updates": 2,
                "measured_updates": 5,
                "before": before,
                "after": snapshot(),
                "samples": samples,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        )
        del optimizer, x
    return {
        "order": "ABBA",
        "instrumented": False,
        "rounds": rounds,
        "median_wall_seconds": {
            name: statistics.median(s["wall_seconds"] for r in rounds if r["arm"] == name for s in r["samples"])
            for name in ("bnb", "saved_weight")
        },
        "scope": "Two synthetic seven-linear blocks; both packed models resident. No attention/media/loading/head or full-model/end-to-end claim. Host values are boundary observations, not peaks; process NVML comes from the outer stage sampler.",
    }


@pytest.mark.GPU
def test_actual_saved_dequant_full_checkpoint_cuda(record_property):
    if os.environ.get("COSMOS_RUN_SAVED_DEQUANT_CUDA") != "1":
        pytest.skip("explicit allocated saved-dequant mechanism diagnostic only")
    assert torch.cuda.is_available(), "opt-in qualification requires real CUDA; never skip or fall back"
    assert torch.__version__ == "2.11.0+cu128"
    production, build_sha, wheel_sha = implementation_pins()
    sources = verify_sources(
        os.environ["COSMOS_SAVED_DEQUANT_BUILD"],
        build_receipt_sha=build_sha,
        wheel_sha=wheel_sha,
        production=production,
    )
    from cosmos_framework.posttrain.utils.determinism import enable_determinism

    enable_determinism(strict=True)
    assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    require_device(properties.name, properties.total_memory, capability, torch.cuda.is_bf16_supported())
    report = {
        "protocol": "cosmos.saved-dequant-cuda-mechanism.v1",
        "complete": False,
        "scope": "Test-only mechanism, not native feature/end-to-end speed or task quality qualification.",
        "exception_policy": "Abort this process on backward/error; no in-process recovery or retry claim.",
        "shapes": SHAPES,
        "blocks": 2,
        "device": {"name": properties.name, "total_memory_bytes": properties.total_memory, "capability": capability},
        "bf16_dense_bytes_per_block": BLOCK_BYTES,
        "tokens": 128,
        "source_sha256": sources,
        "self_sha256": digest(__file__),
        "correctness": [],
    }
    if production:
        report.update(
            protocol="cosmos.native-checkpointed-dequant-cuda.v1",
            scope="Production native class/context on two synthetic blocks; not full-model lifecycle, task quality, or end-to-end speed qualification.",
            quantized_linear_execution="checkpointed_dequant",
        )
    destination = Path(os.environ["COSMOS_SAVED_DEQUANT_REPORT"])
    assert not destination.exists()
    try:
        with torch.random.fork_rng(devices=[device.index]):
            stacks, blocks = make_pair(device, production=production)
            frozen = [frozen_state(arm) for arm in blocks]
            assert frozen[0][0] == frozen[1][0]
            x = torch.randn(1, 128, 4096, generator=torch.Generator().manual_seed(42), dtype=torch.bfloat16).to(device)
            # Direct no-grad inference is deliberately native, before parameters diverge.
            left, right = blocks[0][0].layers[0], blocks[1][0].layers[0]
            with torch.no_grad():
                torch.testing.assert_close(right(x[:, :1]), left(x[:, :1]), rtol=0, atol=0)
            report["native_single_vector_inference_exact"] = True
            if production:
                report["production_guards"] = production_guards(blocks, x)
            for grad, early in ((True, True), (False, True), (True, False)):
                report["correctness"].append(correct_pair(stacks, blocks, x, input_grad=grad, early_stop=early))
            if production:
                comparisons = [
                    item
                    for variant in report["correctness"]
                    for step_report in variant["steps"]
                    for value in step_report["comparisons"].values()
                    for item in (value if isinstance(value, list) else [value])
                ]
                assert len(comparisons) == 519 and all(item["exact"] for item in comparisons)
                report["exact_comparison_count"] = len(comparisons)
            report["timings"] = timings(stacks, x)
            for arm, original in zip(blocks, frozen, strict=True):
                assert frozen_state(arm) == original, "packed bytes/QuantState/nested aliases changed"
            report["packed_state_and_aliases_unchanged"] = True
            assert all(digest(path) == value for path, value in sources.items())
            report["complete"] = True
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error), "requires_process_exit": True}
        raise
    finally:
        with destination.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
        record_property(
            "saved_dequant_mechanism_report",
            json.dumps(
                {"path": str(destination), "sha256": digest(destination), "complete": report["complete"]},
                sort_keys=True,
            ),
        )
