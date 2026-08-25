# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Mixed-precision diffusion steps for ModelOpt static-FP8 checkpoints.

The first/last N denoising steps run generation-path FP8 linears with 16-bit
activations (W8A16: the checkpoint's E4M3 weight dequantized to the compute
dtype + dense GEMM); the middle steps keep the TorchAO FP8-activation path
(W8A8). Weights are always the checkpoint's FP8 tensors — only the activation
treatment changes per step. Port of vllm-omni's Cosmos3 mixed-precision
feature (branch ``mixed-precision-diffusion-steps``) onto the framework's
TorchAO FP8 path.
"""

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from cosmos_framework.utils import log
from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig

PRECISION_PATHS = ("reasoner", "generation")
W8A16_CACHE_MODES = ("none", "generation", "all", "cpu_block", "gpu_block")


def use_w8a16_step(first_steps: int, last_steps: int, step_index: int, num_steps: int) -> bool:
    """Return whether one denoising step runs the W8A16 path.

    ``num_steps == 1`` always selects W8A8: FSDP collective alignment pads
    slow ranks with dummy single-step samples, and those must stay on the
    cheap path (a genuine 1-step request also gets W8A8 — same open question
    as the vllm-omni reference).
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if step_index < 0 or step_index >= num_steps:
        raise IndexError(f"step_index must be in [0, {num_steps}), got {step_index}")
    if num_steps == 1:
        return False
    return step_index < first_steps or step_index >= num_steps - last_steps


def _unwrap_float8(weight: torch.Tensor) -> torch.Tensor:
    """Return the local PrototypeFloat8Tensor behind an (optionally DTensor) weight."""
    return weight.to_local() if isinstance(weight, DTensor) else weight


def _dequantize_w8a16_weight(weight: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a full (gathered) PrototypeFloat8Tensor to a dense (N, K) matrix.

    ``dtype`` must be the activation compute dtype, not ``weight.dtype`` --
    the PrototypeFloat8Tensor subclass's own ``.dtype`` reflects how the
    checkpoint was exported (e.g. float32 on the real Cosmos3-Nano FP8
    checkpoint) and need not match the activations the dense GEMM runs
    against. Mirrors vllm-omni, which always resolves dense W8A16 weights in
    the activation dtype.
    """
    return weight.qdata.to(dtype) * weight.scale.to(dtype)


def _validate_fp8_linear(module: nn.Module, fqn: str) -> None:
    """Bind-time validation, parity with vllm-omni's Fp8W8A8W8A16Strategy._validate_layer."""
    if hasattr(module, "pre_quant_scale"):
        raise ValueError(
            f"{fqn} carries pre_quant_scale; SmoothQuant checkpoints are not supported "
            "by mixed-precision diffusion steps."
        )
    local = _unwrap_float8(module.weight)
    qdata = local.qdata
    if qdata.dtype != torch.float8_e4m3fn:
        raise TypeError(f"{fqn} must hold a canonical float8_e4m3fn weight; got {qdata.dtype}")
    if qdata.ndim != 2:
        raise ValueError(f"{fqn} must hold a rank-2 FP8 weight; got shape {tuple(qdata.shape)}")
    scale = local.scale
    if scale.numel() != 1:
        raise ValueError(f"{fqn} requires one per-tensor FP8 weight scale, got {scale.numel()}")
    scale_value = scale.detach().float()
    if not torch.isfinite(scale_value).all() or not (scale_value > 0).all():
        raise ValueError(f"{fqn} has a non-finite or non-positive FP8 weight scale")


class MixedPrecisionRuntime:
    """Owns per-step precision state and the tagged FP8 linear inventory."""

    def __init__(self, config: QuantizationConfig, activation_dtype: torch.dtype = torch.bfloat16) -> None:
        if activation_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"activation_dtype must be torch.bfloat16 or torch.float16, got {activation_dtype}")
        self.config = config
        self.activation_dtype = activation_dtype
        self.installed_counts: dict[str, int] = {path: 0 for path in PRECISION_PATHS}
        self.last_trace: tuple[str, ...] = ()
        self.is_sharded = False
        self._modules_by_path: dict[str, list[nn.Module]] = {path: [] for path in PRECISION_PATHS}
        self._generation_high_precision = False
        self._trace: list[str] = []
        self._block_provider = None  # set by Task 4
        self.cached_bytes = 0  # set by Task 3

    def install(self, net: nn.Module, blocks: list[nn.Module] | None = None) -> None:
        """Discover, validate, and tag every ModelOpt FP8 linear under ``net``."""
        # Import here: quantization.py imports are lazy-torchao friendly and
        # mixed_precision must not become a hard torchao dependency at import.
        from cosmos_framework.utils.generator.quantization import _ModelOptFloat8Linear

        for fqn, module in net.named_modules():
            if not isinstance(module, _ModelOptFloat8Linear):
                continue
            _validate_fp8_linear(module, fqn)
            path = "generation" if "_moe_gen" in fqn else "reasoner"
            module._mixed_precision_path = path
            module._mixed_precision_runtime = self
            self._modules_by_path[path].append(module)
            self.installed_counts[path] += 1
            self.is_sharded = self.is_sharded or isinstance(module.weight, DTensor)

        missing = [path for path, count in self.installed_counts.items() if count == 0]
        if missing:
            raise ValueError(
                f"Mixed precision found no ModelOpt FP8 linears on path(s) {missing}; "
                f"discovered counts={self.installed_counts}"
            )
        if self.is_sharded and self.config.mixed_precision_w8a16_cache != "none":
            raise ValueError(
                "FSDP-sharded FP8 weights support only mixed_precision_w8a16_cache='none' "
                f"(the default); got {self.config.mixed_precision_w8a16_cache!r}. Drop the "
                "flag or pass --mixed-precision-w8a16-cache none for sharded runs."
            )
        net._mixed_precision_runtime = self
        cache_mode = self.config.mixed_precision_w8a16_cache
        if cache_mode in ("generation", "all"):
            cached_paths = PRECISION_PATHS if cache_mode == "all" else ("generation",)
            for path in cached_paths:
                for module in self._modules_by_path[path]:
                    cache = _dequantize_w8a16_weight(_unwrap_float8(module.weight), self.activation_dtype).contiguous()
                    module.register_buffer("_w8a16_weight_cache", cache, persistent=False)
                    self.cached_bytes += cache.numel() * cache.element_size()
        if cache_mode in ("gpu_block", "cpu_block"):
            if blocks is None:
                blocks = list(net.language_model.model.layers)
            provider_cls = _GpuBlockWeightProvider if cache_mode == "gpu_block" else _CpuBlockWeightProvider
            self._block_provider = provider_cls(self, blocks, self.activation_dtype)
            self._block_provider.initialize()
            self.cached_bytes = self._block_provider.device_bytes
        reasoner_label = "W8A16" if self.config.mixed_precision_reasoner_policy == "high_precision" else "W8A8"
        log.info(
            f"Mixed precision installed: reasoner={reasoner_label}, "
            f"generation=first {self.config.mixed_precision_first_steps} + "
            f"last {self.config.mixed_precision_last_steps} W8A16 / middle W8A8, "
            f"cache={self.config.mixed_precision_w8a16_cache}, linears={self.installed_counts}"
        )
        host_bytes = self._block_provider.host_bytes if self._block_provider is not None else 0
        log.info(
            f"Mixed precision W8A16 cache ready: mode={cache_mode}, "
            f"resident_bytes={self.cached_bytes}, host_bytes={host_bytes}"
        )

    def use_high_precision(self, path: str) -> bool:
        """Resolve the static reasoner policy or the current generation step selection."""
        if path == "reasoner":
            return self.config.mixed_precision_reasoner_policy == "high_precision"
        return self._generation_high_precision

    def set_step(self, step_index: int, num_steps: int) -> None:
        """Select one precision at the sampler-step boundary and record the trace."""
        self._generation_high_precision = use_w8a16_step(
            self.config.mixed_precision_first_steps,
            self.config.mixed_precision_last_steps,
            step_index,
            num_steps,
        )
        if self._block_provider is not None and self._generation_high_precision:
            self._block_provider.preload_first()
        self._trace.append("W8A16" if self._generation_high_precision else "W8A8")

    def set_base_precision(self) -> None:
        """Force W8A8 (used before FSDP-alignment dummy padding calls)."""
        self._generation_high_precision = False

    def resolve_w8a16_weight(self, module: nn.Module) -> torch.Tensor:
        """Resolve a dense (N, K) weight: staged slot -> full cache -> on-the-fly."""
        staged = module._mixed_precision_staged_weight
        if staged is not None:
            if staged.dtype != self.activation_dtype:
                raise TypeError(
                    f"staged W8A16 weight dtype {staged.dtype} does not match "
                    f"activation_dtype {self.activation_dtype}"
                )
            return staged
        cached = getattr(module, "_w8a16_weight_cache", None)
        if cached is not None:
            if cached.dtype != self.activation_dtype:
                raise TypeError(
                    f"cached W8A16 weight dtype {cached.dtype} does not match "
                    f"activation_dtype {self.activation_dtype}"
                )
            return cached
        return _dequantize_w8a16_weight(_unwrap_float8(module.weight), self.activation_dtype)

    def reset(self) -> None:
        """Finish the request: log the trace and return to idle W8A8 state."""
        if self._trace:
            self.last_trace = tuple(self._trace)
            log.info(f"MIXED_PRECISION_TRACE steps={','.join(self.last_trace)}")
        self._trace.clear()
        self._generation_high_precision = False
        if self._block_provider is not None:
            self._block_provider.reset()


def install_mixed_precision_runtime(
    net: nn.Module,
    quantization_config: QuantizationConfig,
    blocks: list[nn.Module] | None = None,
    activation_dtype: torch.dtype = torch.bfloat16,
) -> MixedPrecisionRuntime | None:
    """Install mixed precision on ``net`` when enabled; return the runtime or None."""
    if not quantization_config.mixed_precision_enabled:
        return None
    runtime = MixedPrecisionRuntime(quantization_config, activation_dtype=activation_dtype)
    runtime.install(net, blocks=blocks)
    return runtime


class _BlockWeightProvider:
    """Stage per-decoder-layer W8A16 generation weights through two device slots.

    Layer hooks drive a double buffer: while layer N computes against its
    ready slot, the staging stream fills the other slot with layer N+1's
    dense weights. Ready/free CUDA events order the two streams. Port of
    vllm-omni's W8A16BlockWeightProvider.
    """

    def __init__(self, runtime: "MixedPrecisionRuntime", blocks: list[nn.Module], dtype: torch.dtype) -> None:
        self._runtime = runtime
        self._blocks = blocks
        self._dtype = dtype
        # entries[b] = list of (module, offset, out_features, in_features)
        self._entries: list[list[tuple[nn.Module, int, int, int]]] = []
        self._block_numels: list[int] = []
        self._slots: tuple[torch.Tensor, torch.Tensor] | None = None
        self._stage_stream: torch.cuda.Stream | None = None
        self._ready_events: list[torch.cuda.Event] = []
        self._free_events: list[torch.cuda.Event] = []
        self._loaded: list[int | None] = [None, None]
        self._hook_handles: list = []
        self.device_bytes = 0
        self.host_bytes = 0

    def initialize(self) -> None:
        """Build per-block inventories, allocate the two slots, install hooks."""
        from cosmos_framework.utils.generator.quantization import _ModelOptFloat8Linear

        for block in self._blocks:
            entries: list[tuple[nn.Module, int, int, int]] = []
            offset = 0
            for _, module in block.named_modules():
                if not isinstance(module, _ModelOptFloat8Linear):
                    continue
                if module._mixed_precision_path != "generation":
                    continue
                out_features, in_features = module.weight.qdata.shape
                entries.append((module, offset, out_features, in_features))
                offset += out_features * in_features
            self._entries.append(entries)
            self._block_numels.append(offset)

        if not any(self._entries):
            raise ValueError(
                "gpu_block/cpu_block cache found no generation-path FP8 linears in the provided blocks"
            )

        max_numel = max(self._block_numels)
        # Blocks without generation-path linears are legitimate (e.g. a leading
        # reasoner-only layer); take the device from the first block that has any.
        first_nonempty = next(entries for entries in self._entries if entries)
        device = first_nonempty[0][0].weight.device
        self._slots = (
            torch.empty(max_numel, dtype=self._dtype, device=device),
            torch.empty(max_numel, dtype=self._dtype, device=device),
        )
        self.device_bytes = 2 * max_numel * self._slots[0].element_size()
        self._stage_stream = torch.cuda.Stream(device=device)
        self._ready_events = [torch.cuda.Event(), torch.cuda.Event()]
        self._free_events = [torch.cuda.Event(), torch.cuda.Event()]
        for event in self._free_events:
            event.record()  # both slots start free

        for block_index, block in enumerate(self._blocks):
            self._hook_handles.append(
                block.register_forward_pre_hook(self._make_pre_hook(block_index), prepend=True)
            )
            self._hook_handles.append(
                block.register_forward_hook(self._make_post_hook(block_index), always_call=True)
            )
        self._materialize_sources()

    def _materialize_sources(self) -> None:
        """Subclass hook: prepare per-block weight sources (host copies for CPU mode)."""

    def _fill_slot(self, slot: int, block_index: int) -> None:
        raise NotImplementedError

    def _stage(self, block_index: int) -> None:
        """Queue block ``block_index`` into its slot on the staging stream."""
        slot = block_index % 2
        if self._loaded[slot] == block_index:
            return
        stream = self._stage_stream
        stream.wait_event(self._free_events[slot])
        with torch.cuda.stream(stream):
            self._fill_slot(slot, block_index)
        self._ready_events[slot].record(stream)
        self._loaded[slot] = block_index

    def preload_first(self) -> None:
        self._stage(0)

    def _make_pre_hook(self, block_index: int):
        def _pre_hook(module: nn.Module, args) -> None:
            if not self._runtime.use_high_precision("generation"):
                return
            self._stage(block_index)  # no-op when already queued by the previous post-hook
            slot = block_index % 2
            torch.cuda.current_stream().wait_event(self._ready_events[slot])
            slot_buffer = self._slots[slot]
            for linear, offset, out_features, in_features in self._entries[block_index]:
                linear._mixed_precision_staged_weight = slot_buffer[
                    offset : offset + out_features * in_features
                ].view(out_features, in_features)
            if block_index + 1 < len(self._blocks):
                self._stage(block_index + 1)

        return _pre_hook

    def _make_post_hook(self, block_index: int):
        def _post_hook(module: nn.Module, args, output) -> None:
            slot = block_index % 2
            staged_any = False
            for linear, _, _, _ in self._entries[block_index]:
                staged_any = staged_any or linear._mixed_precision_staged_weight is not None
                linear._mixed_precision_staged_weight = None
            if staged_any:
                # The slot may be refilled only after this block's compute is done.
                self._free_events[slot].record(torch.cuda.current_stream())
                self._loaded[slot] = None

        return _post_hook

    def reset(self) -> None:
        """Synchronize staging work and clear request-scoped state."""
        if self._stage_stream is not None:
            self._stage_stream.synchronize()
        self._loaded = [None, None]
        for entries in self._entries:
            for linear, _, _, _ in entries:
                linear._mixed_precision_staged_weight = None
        # Re-recording free events on the *current* stream (no explicit stream
        # arg) is only correct because reset() always runs on the same stream
        # as the forward passes it follows, at all current call sites.
        for event in self._free_events:
            event.record()


class _GpuBlockWeightProvider(_BlockWeightProvider):
    """Fill slots by dequantizing the resident FP8 weights on the staging stream."""

    def _fill_slot(self, slot: int, block_index: int) -> None:
        slot_buffer = self._slots[slot]
        for linear, offset, out_features, in_features in self._entries[block_index]:
            view = slot_buffer[offset : offset + out_features * in_features].view(out_features, in_features)
            weight = linear.weight
            view.copy_(weight.qdata.to(self._dtype))
            view.mul_(weight.scale.to(device=view.device, dtype=self._dtype))


class _CpuBlockWeightProvider(_BlockWeightProvider):
    """Fill slots by H2D-copying pre-dequantized pinned-host BF16 blocks."""

    def _materialize_sources(self) -> None:
        self._host_blocks: list[torch.Tensor] = []
        for block_index, entries in enumerate(self._entries):
            host = torch.empty(self._block_numels[block_index], dtype=self._dtype, pin_memory=True)
            for linear, offset, out_features, in_features in entries:
                dense = _dequantize_w8a16_weight(linear.weight, self._dtype)
                host[offset : offset + out_features * in_features].copy_(dense.reshape(-1))
            self._host_blocks.append(host)
            self.host_bytes += host.numel() * host.element_size()

    def _fill_slot(self, slot: int, block_index: int) -> None:
        numel = self._block_numels[block_index]
        self._slots[slot][:numel].copy_(self._host_blocks[block_index], non_blocking=True)
