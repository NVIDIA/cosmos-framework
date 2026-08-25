# FP8 Mixed-Precision Diffusion Steps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For ModelOpt static-FP8 checkpoints, run the first/last N diffusion steps with 16-bit activations (W8A16) and the middle steps on the existing FP8-activation path (W8A8), with five W8A16 weight-cache modes.

**Architecture:** A `MixedPrecisionRuntime` installed on the VFM net after FP8 weight install tags every `_ModelOptFloat8Linear` as `reasoner` or `generation` path and holds the per-step precision flag. `_ModelOptFloat8Linear.forward` branches on that flag: W8A16 resolves a dense BF16 weight (staged double-buffer slot → full cache buffer → on-the-fly dequantization) and runs `F.linear`; otherwise the existing TorchAO W8A8 path runs. Samplers gain a `step_callback(step_index, num_steps)` invoked at the top of each denoising step.

**Tech Stack:** PyTorch, TorchAO `PrototypeFloat8Tensor` (attrs: `.qdata` E4M3 `(out,in)`, `.scale` fp32 `(1,1)`), attrs configs, pydantic CLI args, pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-fp8-mixed-precision-diffusion-steps-design.md`

## Global Constraints

- Defaults: `mixed_precision_first_steps=0`, `mixed_precision_last_steps=0`, `mixed_precision_reasoner_policy="high_precision"`, `mixed_precision_w8a16_cache="gpu_block"`. Feature is enabled iff `first_steps + last_steps > 0` AND the checkpoint is ModelOpt FP8; with defaults nothing installs and behavior is byte-identical to today.
- Step schedule: `step_index < first_steps or step_index >= num_steps - last_steps`; `num_steps == 1` → always W8A8; `num_steps <= 0` → `ValueError`; out-of-range index → `IndexError`.
- Path classification: FQN contains `_moe_gen` → `generation`, else `reasoner`.
- FSDP-sharded weights (any tagged `weight` is a `DTensor`) → only `w8a16_cache="none"` allowed, else `ValueError`.
- `use_cuda_graphs=True` + mixed precision enabled → `ValueError` at load.
- Mixed-precision steps configured + non-ModelOpt-FP8 checkpoint → `ValueError` at load.
- Copyright header on every new file: `# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.` + `# SPDX-License-Identifier: OpenMDW-1.1`.
- Performance numbers go in the PR description, never in committed docs.
- Repo test command: `pytest <file> -v` (GPU tests follow existing convention in `cosmos_framework/utils/generator/quantization_test.py`: plain `cuda` usage, `pytest.skip` on insufficient capability).
- Commits: end message with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. Branch: `pzeren/fp8-mixed-precision-steps`.

---

### Task 1: Config fields on `QuantizationConfig` + step-schedule function

**Files:**

- Modify: `cosmos_framework/configs/base/defaults/quantization.py`
- Create: `cosmos_framework/utils/generator/mixed_precision.py`
- Test: `cosmos_framework/utils/generator/mixed_precision_test.py`

**Interfaces:**

- Consumes: nothing new.

- Produces:
  - `QuantizationConfig.mixed_precision_first_steps: int` (default 0), `.mixed_precision_last_steps: int` (default 0), `.mixed_precision_reasoner_policy: str` (`"high_precision"`/`"base_precision"`, default `"high_precision"`), `.mixed_precision_w8a16_cache: str` (`"none"`/`"generation"`/`"all"`/`"cpu_block"`/`"gpu_block"`, default `"gpu_block"`), property `.mixed_precision_enabled -> bool`.
  - `mixed_precision.use_w8a16_step(first_steps: int, last_steps: int, step_index: int, num_steps: int) -> bool`
  - `mixed_precision.PRECISION_PATHS = ("reasoner", "generation")`, `mixed_precision.W8A16_CACHE_MODES = ("none", "generation", "all", "cpu_block", "gpu_block")`

- [ ] **Step 1: Write the failing tests**

Create `cosmos_framework/utils/generator/mixed_precision_test.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import pytest

from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig
from cosmos_framework.utils.generator.mixed_precision import use_w8a16_step


def test_config_defaults_disable_mixed_precision() -> None:
    config = QuantizationConfig()
    assert config.mixed_precision_first_steps == 0
    assert config.mixed_precision_last_steps == 0
    assert config.mixed_precision_reasoner_policy == "high_precision"
    assert config.mixed_precision_w8a16_cache == "gpu_block"
    assert not config.mixed_precision_enabled


def test_config_enabled_when_any_width_positive() -> None:
    assert QuantizationConfig(mixed_precision_first_steps=1).mixed_precision_enabled
    assert QuantizationConfig(mixed_precision_last_steps=2).mixed_precision_enabled


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        QuantizationConfig(mixed_precision_first_steps=-1)
    with pytest.raises(ValueError):
        QuantizationConfig(mixed_precision_reasoner_policy="fast")
    with pytest.raises(ValueError):
        QuantizationConfig(mixed_precision_w8a16_cache="huge")


def test_schedule_selects_first_and_last_steps() -> None:
    selected = [i for i in range(10) if use_w8a16_step(2, 4, i, 10)]
    assert selected == [0, 1, 6, 7, 8, 9]


def test_schedule_overlap_selects_every_step() -> None:
    assert all(use_w8a16_step(3, 3, i, 4) for i in range(4))


def test_schedule_single_step_is_base_precision() -> None:
    # FSDP alignment pads slow ranks with dummy num_steps=1 samples; keep them cheap.
    assert not use_w8a16_step(3, 3, 0, 1)


def test_schedule_validates_bounds() -> None:
    with pytest.raises(ValueError):
        use_w8a16_step(1, 1, 0, 0)
    with pytest.raises(IndexError):
        use_w8a16_step(1, 1, 5, 5)
    with pytest.raises(IndexError):
        use_w8a16_step(1, 1, -1, 5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest cosmos_framework/utils/generator/mixed_precision_test.py -v`
Expected: FAIL (`ModuleNotFoundError: ... mixed_precision` / `AttributeError: mixed_precision_first_steps`)

- [ ] **Step 3: Implement**

Append to `QuantizationConfig` in `cosmos_framework/configs/base/defaults/quantization.py` (after `modelopt_fp8_target_fqns`):

```python
    # Mixed-precision diffusion steps for ModelOpt FP8 checkpoints: the first
    # ``mixed_precision_first_steps`` and last ``mixed_precision_last_steps``
    # denoising steps run generation-path linears with 16-bit activations
    # (W8A16: dequantized FP8 weight + dense GEMM); the middle steps keep the
    # FP8-activation path (W8A8). Both 0 (default) disables the feature.
    mixed_precision_first_steps: int = attrs.field(
        default=0, validator=[attrs.validators.instance_of(int), attrs.validators.ge(0)]
    )
    mixed_precision_last_steps: int = attrs.field(
        default=0, validator=[attrs.validators.instance_of(int), attrs.validators.ge(0)]
    )

    # Reasoner-path (understanding pathway) precision, independent of the step
    # schedule: "high_precision" keeps those linears on W8A16 for every step;
    # "base_precision" keeps them on W8A8.
    mixed_precision_reasoner_policy: str = attrs.field(
        default="high_precision",
        validator=attrs.validators.in_({"high_precision", "base_precision"}),
    )

    # Where W8A16 dense weights come from: "none" dequantizes per call,
    # "generation"/"all" hold resident BF16 caches, "gpu_block"/"cpu_block"
    # stage per-decoder-layer slots through a double buffer. Only "none" is
    # supported when the model is FSDP-sharded.
    mixed_precision_w8a16_cache: str = attrs.field(
        default="gpu_block",
        validator=attrs.validators.in_({"none", "generation", "all", "cpu_block", "gpu_block"}),
    )

    @property
    def mixed_precision_enabled(self) -> bool:
        """Whether mixed-precision diffusion steps are requested."""
        return self.mixed_precision_first_steps + self.mixed_precision_last_steps > 0
```

Create `cosmos_framework/utils/generator/mixed_precision.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest cosmos_framework/utils/generator/mixed_precision_test.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/configs/base/defaults/quantization.py cosmos_framework/utils/generator/mixed_precision.py cosmos_framework/utils/generator/mixed_precision_test.py
git commit -m "feat(quantization): mixed-precision step schedule and config fields"
```

---

### Task 2: `MixedPrecisionRuntime` (install/tagging/validation/state/trace) + on-the-fly W8A16 dispatch in `_ModelOptFloat8Linear`

**Files:**

- Modify: `cosmos_framework/utils/generator/mixed_precision.py`
- Modify: `cosmos_framework/utils/generator/quantization.py` (`_ModelOptFloat8Linear`, ~line 43-65)
- Test: `cosmos_framework/utils/generator/mixed_precision_test.py`

**Interfaces:**

- Consumes: `use_w8a16_step`, `QuantizationConfig` (Task 1); `_ModelOptFloat8Linear` and TorchAO weight attrs `.qdata`/`.scale` (existing).

- Produces:
  - `class MixedPrecisionRuntime:` with `__init__(self, config: QuantizationConfig)`, `install(self, net: torch.nn.Module) -> None`, `set_step(self, step_index: int, num_steps: int) -> None`, `set_base_precision(self) -> None`, `use_high_precision(self, path: str) -> bool`, `resolve_w8a16_weight(self, module: torch.nn.Module) -> torch.Tensor`, `reset(self) -> None`; attributes `installed_counts: dict[str, int]`, `last_trace: tuple[str, ...]`, `is_sharded: bool`, `config`.
  - `install_mixed_precision_runtime(net: torch.nn.Module, quantization_config: QuantizationConfig) -> MixedPrecisionRuntime | None` — returns None when `not config.mixed_precision_enabled`; otherwise installs and sets `net._mixed_precision_runtime`.
  - `_dequantize_w8a16_weight(weight) -> torch.Tensor` (module-private helper).
  - On `_ModelOptFloat8Linear`: class attributes `_mixed_precision_runtime = None`, `_mixed_precision_path: str = "generation"`, `_mixed_precision_staged_weight: torch.Tensor | None = None` (Task 4 sets/clears the staged view; this task only reads it).

- [ ] **Step 1: Write the failing tests**

Append to `cosmos_framework/utils/generator/mixed_precision_test.py`:

```python
import torch
from torch import nn

from cosmos_framework.utils.generator.mixed_precision import (
    MixedPrecisionRuntime,
    install_mixed_precision_runtime,
)
from cosmos_framework.utils.generator.quantization import _ModelOptFloat8Linear


def _make_fp8_linear(
    out_features: int = 8, in_features: int = 4, device: str = "cpu"
) -> _ModelOptFloat8Linear:
    """Build a _ModelOptFloat8Linear with a real PrototypeFloat8Tensor weight on ``device``.

    Built directly on the target device: moving a constructed PrototypeFloat8Tensor
    with ``.to()`` is not guaranteed to be supported by the subclass.
    """
    from torchao.float8.inference import Float8MMConfig
    from torchao.prototype.quantization.float8_static_quant.prototype_float8_tensor import (
        PrototypeFloat8Tensor,
    )
    from torchao.quantization import PerTensor
    from torchao.quantization.quantize_.workflows import QuantizeTensorToFloat8Kwargs

    module = _ModelOptFloat8Linear(in_features, out_features, bias=False, dtype=torch.bfloat16, device=device)
    module._modelopt_high_precision_dtype = torch.bfloat16
    qdata = torch.randn(out_features, in_features, device=device).clamp(-1, 1).to(torch.float8_e4m3fn)
    mm_config = Float8MMConfig(use_fast_accum=True)
    module.weight = nn.Parameter(
        PrototypeFloat8Tensor(
            qdata,
            torch.full((1, 1), 0.5, dtype=torch.float32, device=device),
            act_quant_scale=torch.full((1, 1), 1.0, dtype=torch.float32, device=device),
            block_size=[out_features, in_features],
            mm_config=mm_config,
            act_quant_kwargs=QuantizeTensorToFloat8Kwargs(
                float8_dtype=torch.float8_e4m3fn, granularity=PerTensor(), mm_config=mm_config
            ),
            dtype=torch.bfloat16,
        ),
        requires_grad=False,
    )
    return module


class _TinyMoTNet(nn.Module):
    """Minimal net shape: one reasoner-path and one generation-path FP8 linear."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = _make_fp8_linear()
        self.mlp_moe_gen = nn.Module()
        self.mlp_moe_gen.up_proj = _make_fp8_linear()


def _enabled_config(**overrides) -> QuantizationConfig:
    values = dict(mixed_precision_first_steps=1, mixed_precision_last_steps=1, mixed_precision_w8a16_cache="none")
    values.update(overrides)
    return QuantizationConfig(**values)


def test_install_returns_none_when_disabled() -> None:
    net = _TinyMoTNet()
    assert install_mixed_precision_runtime(net, QuantizationConfig()) is None
    assert getattr(net, "_mixed_precision_runtime", None) is None


def test_install_classifies_paths_and_tags_modules() -> None:
    net = _TinyMoTNet()
    runtime = install_mixed_precision_runtime(net, _enabled_config())
    assert runtime is not None
    assert net._mixed_precision_runtime is runtime
    assert runtime.installed_counts == {"reasoner": 1, "generation": 1}
    assert net.mlp.up_proj._mixed_precision_path == "reasoner"
    assert net.mlp_moe_gen.up_proj._mixed_precision_path == "generation"
    assert net.mlp.up_proj._mixed_precision_runtime is runtime


def test_install_requires_both_paths() -> None:
    net = _TinyMoTNet()
    net.mlp.up_proj = nn.Linear(4, 8)  # no reasoner-path FP8 linear left
    with pytest.raises(ValueError, match="reasoner"):
        install_mixed_precision_runtime(net, _enabled_config())


def test_step_state_and_trace() -> None:
    net = _TinyMoTNet()
    runtime = install_mixed_precision_runtime(net, _enabled_config())
    runtime.set_step(0, 4)
    assert runtime.use_high_precision("generation")
    runtime.set_step(1, 4)
    assert not runtime.use_high_precision("generation")
    # reasoner policy is static, independent of the step
    assert runtime.use_high_precision("reasoner")
    runtime.set_base_precision()
    assert not runtime.use_high_precision("generation")
    runtime.reset()
    assert runtime.last_trace == ("W8A16", "W8A8")
    assert not runtime.use_high_precision("generation")


def test_reasoner_base_policy() -> None:
    net = _TinyMoTNet()
    runtime = install_mixed_precision_runtime(
        net, _enabled_config(mixed_precision_reasoner_policy="base_precision")
    )
    assert not runtime.use_high_precision("reasoner")


def test_w8a16_forward_matches_manual_dequant() -> None:
    net = _TinyMoTNet()
    runtime = install_mixed_precision_runtime(net, _enabled_config())
    module = net.mlp_moe_gen.up_proj
    inputs = torch.randn(3, 5, 4, dtype=torch.bfloat16)
    runtime.set_step(0, 4)  # W8A16
    output = module(inputs)
    weight = module.weight
    expected = torch.nn.functional.linear(
        inputs.reshape(-1, 4), weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16)
    ).reshape(3, 5, 8)
    assert output.shape == (3, 5, 8)
    torch.testing.assert_close(output, expected)


def test_w8a16_forward_zero_rows() -> None:
    net = _TinyMoTNet()
    runtime = install_mixed_precision_runtime(net, _enabled_config())
    runtime.set_step(0, 4)
    output = net.mlp_moe_gen.up_proj(torch.empty(0, 4, dtype=torch.bfloat16))
    assert output.shape == (0, 8)


def test_install_rejects_smoothquant_layer() -> None:
    net = _TinyMoTNet()
    net.mlp_moe_gen.up_proj.pre_quant_scale = torch.ones(4)
    with pytest.raises(ValueError, match="pre_quant_scale"):
        install_mixed_precision_runtime(net, _enabled_config())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest cosmos_framework/utils/generator/mixed_precision_test.py -v`
Expected: new tests FAIL (`ImportError: MixedPrecisionRuntime`), Task 1 tests still PASS.

- [ ] **Step 3: Implement the runtime**

Append to `cosmos_framework/utils/generator/mixed_precision.py`:

```python
import torch
from torch import nn
from torch.distributed.tensor import DTensor

from cosmos_framework.utils import log
from cosmos_framework.configs.base.defaults.quantization import QuantizationConfig


def _unwrap_float8(weight: torch.Tensor) -> torch.Tensor:
    """Return the local PrototypeFloat8Tensor behind an (optionally DTensor) weight."""
    return weight.to_local() if isinstance(weight, DTensor) else weight


def _dequantize_w8a16_weight(weight: torch.Tensor) -> torch.Tensor:
    """Dequantize a full (gathered) PrototypeFloat8Tensor to a dense (N, K) matrix."""
    return weight.qdata.to(weight.dtype) * weight.scale.to(weight.dtype)


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

    def __init__(self, config: QuantizationConfig) -> None:
        self.config = config
        self.installed_counts: dict[str, int] = {path: 0 for path in PRECISION_PATHS}
        self.last_trace: tuple[str, ...] = ()
        self.is_sharded = False
        self._modules_by_path: dict[str, list[nn.Module]] = {path: [] for path in PRECISION_PATHS}
        self._generation_high_precision = False
        self._trace: list[str] = []
        self._block_provider = None  # set by Task 4
        self._cached_bytes = 0  # set by Task 3

    def install(self, net: nn.Module) -> None:
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
                "FSDP-sharded FP8 weights support only mixed_precision_w8a16_cache='none'; "
                f"got {self.config.mixed_precision_w8a16_cache!r}"
            )
        net._mixed_precision_runtime = self
        reasoner_label = "W8A16" if self.config.mixed_precision_reasoner_policy == "high_precision" else "W8A8"
        log.info(
            "Mixed precision installed: reasoner=%s, generation=first %d + last %d W8A16 / middle W8A8, "
            "cache=%s, linears=%s"
            % (
                reasoner_label,
                self.config.mixed_precision_first_steps,
                self.config.mixed_precision_last_steps,
                self.config.mixed_precision_w8a16_cache,
                self.installed_counts,
            )
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
            return staged
        cached = getattr(module, "_w8a16_weight_cache", None)
        if cached is not None:
            return cached
        return _dequantize_w8a16_weight(module.weight)

    def reset(self) -> None:
        """Finish the request: log the trace and return to idle W8A8 state."""
        if self._trace:
            self.last_trace = tuple(self._trace)
            log.info("MIXED_PRECISION_TRACE steps=%s" % ",".join(self.last_trace))
        self._trace.clear()
        self._generation_high_precision = False
        if self._block_provider is not None:
            self._block_provider.reset()


def install_mixed_precision_runtime(
    net: nn.Module, quantization_config: QuantizationConfig
) -> MixedPrecisionRuntime | None:
    """Install mixed precision on ``net`` when enabled; return the runtime or None."""
    if not quantization_config.mixed_precision_enabled:
        return None
    runtime = MixedPrecisionRuntime(quantization_config)
    runtime.install(net)
    return runtime
```

Note: `log.info` in this codebase takes a single message string (match surrounding usage in `quantization.py`, e.g. f-strings); adjust the two calls to f-strings if `%`-interpolation reads awkwardly next to neighbors.

- [ ] **Step 4: Add the dispatch branch to `_ModelOptFloat8Linear`**

In `cosmos_framework/utils/generator/quantization.py`, extend the class (current body at ~line 43-65). Add class attributes and the branch:

```python
class _ModelOptFloat8Linear(nn.Linear):
    """Work around TorchAO 0.16 static-FP8 limitations for linear inputs."""

    _modelopt_high_precision_dtype: torch.dtype | None = None
    _modelopt_weight_loaded: bool = False
    # Mixed-precision diffusion steps (see utils/generator/mixed_precision.py).
    # None until install_mixed_precision_runtime tags this module.
    _mixed_precision_runtime = None
    _mixed_precision_path: str = "generation"
    # A (N, K) dense view staged by the block provider for the current decoder
    # layer, or None. Read here, written only by the provider hooks.
    _mixed_precision_staged_weight: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output_shape = (*inputs.shape[:-1], self.out_features)
        if inputs.numel() == 0:
            return inputs.new_empty(output_shape)
        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        runtime = self._mixed_precision_runtime
        if runtime is not None and runtime.use_high_precision(self._mixed_precision_path):
            # W8A16: dense GEMM against the dequantized FP8 weight; the
            # activation stays in the compute dtype.
            weight = runtime.resolve_w8a16_weight(self)
            return F.linear(flat_inputs, weight, self.bias).reshape(output_shape)
        flat_outputs = F.linear(flat_inputs, self.weight, self.bias)
        return flat_outputs.reshape(output_shape)
```

(Keep the existing comments about PrototypeFloat8Tensor rank/zero-row behavior on the base path.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest cosmos_framework/utils/generator/mixed_precision_test.py cosmos_framework/utils/generator/quantization_test.py -v`
Expected: all mixed_precision tests PASS; quantization tests unchanged (GPU ones may skip off-GPU).

- [ ] **Step 6: Commit**

```bash
git add cosmos_framework/utils/generator/mixed_precision.py cosmos_framework/utils/generator/mixed_precision_test.py cosmos_framework/utils/generator/quantization.py
git commit -m "feat(quantization): mixed-precision runtime and W8A16 dispatch in FP8 linears"
```

---

### Task 3: Full W8A16 caches (`generation` / `all` modes)

**Files:**

- Modify: `cosmos_framework/utils/generator/mixed_precision.py` (`MixedPrecisionRuntime.install`)
- Test: `cosmos_framework/utils/generator/mixed_precision_test.py`

**Interfaces:**

- Consumes: `MixedPrecisionRuntime`, `_dequantize_w8a16_weight` (Task 2).

- Produces: modules on covered paths gain a non-persistent buffer `_w8a16_weight_cache` (dense `(N, K)`, weight dtype); `runtime.cached_bytes: int` (rename of `_cached_bytes`, public); a ready log line.

- [ ] **Step 1: Write the failing tests**

Append to `cosmos_framework/utils/generator/mixed_precision_test.py`:

```python
def test_generation_cache_covers_generation_path_only() -> None:
    net = _TinyMoTNet()
    runtime = install_mixed_precision_runtime(
        net, _enabled_config(mixed_precision_w8a16_cache="generation")
    )
    gen = net.mlp_moe_gen.up_proj
    reasoner = net.mlp.up_proj
    assert isinstance(gen._w8a16_weight_cache, torch.Tensor)
    assert gen._w8a16_weight_cache.dtype == torch.bfloat16
    assert getattr(reasoner, "_w8a16_weight_cache", None) is None
    assert runtime.cached_bytes == gen._w8a16_weight_cache.numel() * 2
    # cache must not leak into state_dict
    assert not any("_w8a16_weight_cache" in key for key in net.state_dict())


def test_all_cache_covers_both_paths_and_matches_dequant() -> None:
    net = _TinyMoTNet()
    install_mixed_precision_runtime(net, _enabled_config(mixed_precision_w8a16_cache="all"))
    for module in (net.mlp.up_proj, net.mlp_moe_gen.up_proj):
        weight = module.weight
        torch.testing.assert_close(
            module._w8a16_weight_cache,
            weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16),
        )


def test_cached_forward_matches_uncached_forward() -> None:
    torch.manual_seed(0)
    net_cached = _TinyMoTNet()
    torch.manual_seed(0)
    net_plain = _TinyMoTNet()
    runtime_cached = install_mixed_precision_runtime(
        net_cached, _enabled_config(mixed_precision_w8a16_cache="all")
    )
    runtime_plain = install_mixed_precision_runtime(net_plain, _enabled_config())
    runtime_cached.set_step(0, 4)
    runtime_plain.set_step(0, 4)
    inputs = torch.randn(2, 4, dtype=torch.bfloat16)
    torch.testing.assert_close(net_cached.mlp_moe_gen.up_proj(inputs), net_plain.mlp_moe_gen.up_proj(inputs))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest cosmos_framework/utils/generator/mixed_precision_test.py -v -k cache`
Expected: FAIL (`AttributeError: _w8a16_weight_cache` / `cached_bytes`)

- [ ] **Step 3: Implement**

In `MixedPrecisionRuntime.__init__` replace `self._cached_bytes = 0` with `self.cached_bytes = 0`. At the end of `install` (after the sharded-cache check, before the install log), add:

```python
        cache_mode = self.config.mixed_precision_w8a16_cache
        if cache_mode in ("generation", "all"):
            cached_paths = PRECISION_PATHS if cache_mode == "all" else ("generation",)
            for path in cached_paths:
                for module in self._modules_by_path[path]:
                    cache = _dequantize_w8a16_weight(module.weight).contiguous()
                    module.register_buffer("_w8a16_weight_cache", cache, persistent=False)
                    self.cached_bytes += cache.numel() * cache.element_size()
        log.info(
            f"Mixed precision W8A16 cache ready: mode={cache_mode}, "
            f"resident_bytes={self.cached_bytes}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest cosmos_framework/utils/generator/mixed_precision_test.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/utils/generator/mixed_precision.py cosmos_framework/utils/generator/mixed_precision_test.py
git commit -m "feat(quantization): resident W8A16 full-cache modes (generation/all)"
```

---

### Task 4: Double-buffered block providers (`gpu_block` / `cpu_block`)

**Files:**

- Modify: `cosmos_framework/utils/generator/mixed_precision.py`
- Test: `cosmos_framework/utils/generator/mixed_precision_test.py`

**Interfaces:**

- Consumes: `MixedPrecisionRuntime` (`_block_provider` slot, `use_high_precision`), `_dequantize_w8a16_weight`, `module._mixed_precision_staged_weight` (Task 2).

- Produces:
  - `class _BlockWeightProvider:` (base) with `__init__(self, runtime, blocks: list[nn.Module], dtype: torch.dtype)`, `initialize() -> None`, `preload_first() -> None`, `reset() -> None`, `device_bytes: int`, `host_bytes: int`. Subclasses `_GpuBlockWeightProvider` / `_CpuBlockWeightProvider` override `_fill_slot(self, slot: int, block_index: int) -> None`.
  - `MixedPrecisionRuntime.install` creates the provider for `gpu_block`/`cpu_block` given `blocks` — a new optional `install(net, blocks=None)` parameter: `blocks: list[nn.Module] | None`, the ordered decoder layers to hook. When `blocks is None` and a block cache mode is configured, `install` resolves `list(net.language_model.model.layers)`; tests pass an explicit list.

- [ ] **Step 1: Write the failing tests**

Append to `cosmos_framework/utils/generator/mixed_precision_test.py`:

```python
class _TinyLayeredNet(nn.Module):
    """Two decoder-layer-like blocks, each with reasoner and generation linears."""

    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyMoTLayer(device), _TinyMoTLayer(device)])


class _TinyMoTLayer(nn.Module):
    def __init__(self, device: str = "cpu") -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.up_proj = _make_fp8_linear(device=device)
        self.mlp_moe_gen = nn.Module()
        self.mlp_moe_gen.up_proj = _make_fp8_linear(device=device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp_moe_gen.up_proj(inputs)


def _cuda_layered_net() -> "_TinyLayeredNet":
    # Built directly on cuda — no .to() move of the FP8 tensor subclass.
    return _TinyLayeredNet(device="cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block providers need CUDA streams")
@pytest.mark.parametrize("cache_mode", ["gpu_block", "cpu_block"])
def test_block_provider_stages_and_matches_dequant(cache_mode: str) -> None:
    net = _cuda_layered_net()
    runtime = install_mixed_precision_runtime(
        net,
        _enabled_config(mixed_precision_w8a16_cache=cache_mode),
        blocks=list(net.layers),
    )
    runtime.set_step(0, 4)  # W8A16 -> preload_first
    inputs = torch.randn(2, 4, dtype=torch.bfloat16, device="cuda")
    for layer in net.layers:
        output = layer(inputs)
        weight = layer.mlp_moe_gen.up_proj.weight
        expected = torch.nn.functional.linear(
            inputs, weight.qdata.to(torch.bfloat16) * weight.scale.to(torch.bfloat16)
        )
        torch.testing.assert_close(output, expected)
    runtime.reset()
    assert all(layer.mlp_moe_gen.up_proj._mixed_precision_staged_weight is None for layer in net.layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block providers need CUDA streams")
def test_block_provider_idle_on_w8a8_steps() -> None:
    net = _cuda_layered_net()
    runtime = install_mixed_precision_runtime(
        net,
        _enabled_config(mixed_precision_w8a16_cache="gpu_block"),
        blocks=list(net.layers),
    )
    runtime.set_step(1, 4)  # middle step -> W8A8
    _ = net.layers[0](torch.randn(2, 4, dtype=torch.bfloat16, device="cuda"))
    assert net.layers[0].mlp_moe_gen.up_proj._mixed_precision_staged_weight is None
    runtime.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="block providers need CUDA streams")
def test_block_provider_reusable_across_requests() -> None:
    net = _cuda_layered_net()
    runtime = install_mixed_precision_runtime(
        net,
        _enabled_config(mixed_precision_w8a16_cache="gpu_block"),
        blocks=list(net.layers),
    )
    inputs = torch.randn(2, 4, dtype=torch.bfloat16, device="cuda")
    for _ in range(2):  # two full "requests" with mixed schedules
        for step in range(4):
            runtime.set_step(step, 4)
            for layer in net.layers:
                _ = layer(inputs)
        runtime.reset()
    assert runtime.last_trace == ("W8A16", "W8A8", "W8A8", "W8A16")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest cosmos_framework/utils/generator/mixed_precision_test.py -v -k block`
Expected: FAIL (`install_mixed_precision_runtime() got an unexpected keyword argument 'blocks'`) — or SKIP entirely off-GPU (then continue and validate on a GPU node before commit).

- [ ] **Step 3: Implement the providers**

Append to `cosmos_framework/utils/generator/mixed_precision.py`:

```python
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

        max_numel = max(self._block_numels)
        device = self._entries[0][0][0].weight.device
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
                dense = _dequantize_w8a16_weight(linear.weight)
                host[offset : offset + out_features * in_features].copy_(dense.reshape(-1))
            self._host_blocks.append(host)
            self.host_bytes += host.numel() * host.element_size()

    def _fill_slot(self, slot: int, block_index: int) -> None:
        numel = self._block_numels[block_index]
        self._slots[slot][:numel].copy_(self._host_blocks[block_index], non_blocking=True)
```

Wire into the runtime. Change `install` signature to
`def install(self, net: nn.Module, blocks: list[nn.Module] | None = None) -> None:` and, after the full-cache block from Task 3, add:

```python
        if cache_mode in ("gpu_block", "cpu_block"):
            if blocks is None:
                blocks = list(net.language_model.model.layers)
            provider_cls = _GpuBlockWeightProvider if cache_mode == "gpu_block" else _CpuBlockWeightProvider
            dtype = self._modules_by_path["generation"][0].weight.dtype
            self._block_provider = provider_cls(self, blocks, dtype)
            self._block_provider.initialize()
            self.cached_bytes = self._block_provider.device_bytes
```

Extend the ready log to include `host_bytes` when a provider exists. Change
`install_mixed_precision_runtime` to
`def install_mixed_precision_runtime(net, quantization_config, blocks=None)` and pass `blocks` through.

- [ ] **Step 4: Run tests to verify they pass (needs a GPU node)**

Run: `pytest cosmos_framework/utils/generator/mixed_precision_test.py -v`
Expected: PASS (block tests exercise both providers, W8A8 idleness, and cross-request reuse).

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/utils/generator/mixed_precision.py cosmos_framework/utils/generator/mixed_precision_test.py
git commit -m "feat(quantization): double-buffered gpu_block/cpu_block W8A16 staging"
```

---

### Task 5: `step_callback` in the three samplers

**Files:**

- Modify: `cosmos_framework/model/generator/diffusion/samplers/fixed_step.py` (`FixedStepSampler.__call__`)
- Modify: `cosmos_framework/model/generator/diffusion/samplers/unipc.py` (`UniPCSampler.forward`)
- Modify: `cosmos_framework/model/generator/diffusion/samplers/edm.py` (`EDMSampler.forward`, `EDMSampler._forward_impl`, `differential_equation_solver`)
- Test: `cosmos_framework/model/generator/diffusion/samplers/fixed_step_test.py` (append) and inline tests below in a new `cosmos_framework/model/generator/diffusion/samplers/step_callback_test.py`

**Interfaces:**

- Consumes: nothing from earlier tasks (pure sampler change).

- Produces: every sampler accepts `step_callback: Callable[[int, int], None] | None = None` and calls `step_callback(step_index, num_steps)` exactly once at the top of each denoising step, before that step's model evaluation(s). EDM's optional final `sample_clean` forward runs after the last step under the last step's selection (no extra callback).

- [ ] **Step 1: Write the failing tests**

Create `cosmos_framework/model/generator/diffusion/samplers/step_callback_test.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import torch

from cosmos_framework.model.generator.diffusion.samplers.edm import EDMSampler
from cosmos_framework.model.generator.diffusion.samplers.fixed_step import FixedStepSampler
from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler


def test_fixed_step_invokes_callback_per_step() -> None:
    calls: list[tuple[int, int]] = []
    sampler = FixedStepSampler(t_list=[1.0, 0.6, 0.3, 0.0])
    noise = torch.randn(8)
    sampler(
        lambda latent, timestep: torch.zeros_like(latent),
        noise,
        seed=0,
        step_callback=lambda i, n: calls.append((i, n)),
    )
    assert calls == [(0, 3), (1, 3), (2, 3)]


def test_unipc_invokes_callback_per_step() -> None:
    calls: list[tuple[int, int]] = []
    sampler = UniPCSampler(tensor_kwargs={"device": torch.device("cpu")})
    noise = torch.randn(4, 1, 2, 2)
    sampler(
        lambda latent, timestep: torch.zeros_like(latent),
        noise,
        num_steps=5,
        seed=0,
        step_callback=lambda i, n: calls.append((i, n)),
    )
    assert calls == [(i, 5) for i in range(5)]


def test_edm_invokes_callback_per_step_before_evaluations() -> None:
    calls: list[tuple[int, int]] = []
    evaluations: list[int] = []

    def x0_fn(noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        evaluations.append(len(calls))
        return torch.zeros_like(noisy)

    sampler = EDMSampler()
    sampler(
        x0_fn,
        torch.randn(1, 4),
        num_steps=6,
        step_callback=lambda i, n: calls.append((i, n)),
    )
    num_steps = calls[-1][1]
    assert [c[0] for c in calls] == list(range(num_steps))
    # every model evaluation happened after at least one callback
    assert min(evaluations) >= 1


def test_callback_default_none_keeps_behavior() -> None:
    sampler = FixedStepSampler(t_list=[1.0, 0.5, 0.0])
    noise = torch.randn(8)
    out_a = sampler(lambda latent, timestep: torch.zeros_like(latent), noise, seed=1)
    out_b = sampler(
        lambda latent, timestep: torch.zeros_like(latent), noise, seed=1, step_callback=lambda i, n: None
    )
    torch.testing.assert_close(out_a, out_b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest cosmos_framework/model/generator/diffusion/samplers/step_callback_test.py -v`
Expected: FAIL (`unexpected keyword argument 'step_callback'`)

- [ ] **Step 3: Implement**

`fixed_step.py` — add the parameter to `__call__` after `condition_mask`:

```python
        step_callback: Callable[[int, int], None] | None = None,
```

(import `Callable` from `collections.abc` at top). In the loop, the existing enumerate already yields `step_idx`; add as the first statement of the loop body:

```python
        num_steps_total = len(t_list) - 1
        for step_idx, (sigma_cur, sigma_next) in enumerate(
            zip(t_list[:-1], t_list[1:]),
        ):
            if step_callback is not None:
                step_callback(step_idx, num_steps_total)
```

`unipc.py` — add `step_callback: Callable[[int, int], None] | None = None` to `forward` after `seed` (import `Callable` from `collections.abc`). Change the loop to:

```python
        for step_index, timestep in enumerate(progress_bar(timesteps, desc="Sampling", total=len(timesteps))):
            if step_callback is not None:
                step_callback(step_index, len(timesteps))
            velocity_pred = velocity_fn(latent, timestep.reshape(1, 1))
```

`edm.py` — thread through three layers:

1. `EDMSampler.forward(..., step_callback: Optional[Callable[[int, int], None]] = None)`; pass to `self._forward_impl(float64_x0_fn, x_sigma_max, sampler_cfg, step_callback=step_callback)`.
2. `_forward_impl(..., step_callback: Optional[Callable[[int, int], None]] = None)`; pass to `differential_equation_solver(denoiser_fn, sigmas_L, sampler_cfg.solver, callback_fns=callback_fns, step_callback=step_callback)`. Leave the trailing `sample_clean` forward untouched — it runs under the last step's selection by design.
3. `differential_equation_solver(..., step_callback: Optional[Callable[[int, int], None]] = None)`; inside `step_fn`, as the first statement:

```python
        def step_fn(i: int, x0_preds_pair):
            if step_callback is not None:
                step_callback(i, num_step)
```

(`step_fn`'s existing signature/body continues unchanged after this guard; `num_step = len(sigmas_L) - 1` is already in scope.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest cosmos_framework/model/generator/diffusion/samplers/step_callback_test.py cosmos_framework/model/generator/diffusion/samplers/fixed_step_test.py -v`
Expected: PASS, existing fixed_step tests unaffected.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/model/generator/diffusion/samplers/fixed_step.py cosmos_framework/model/generator/diffusion/samplers/unipc.py cosmos_framework/model/generator/diffusion/samplers/edm.py cosmos_framework/model/generator/diffusion/samplers/step_callback_test.py
git commit -m "feat(samplers): optional per-step callback in FixedStep/UniPC/EDM"
```

---

### Task 6: Wire the runtime into `generate_samples_from_batch`

**Files:**

- Modify: `cosmos_framework/model/generator/omni_mot_model.py` (sampler invocation region, ~line 3382-3500)

**Interfaces:**

- Consumes: `net._mixed_precision_runtime` (`set_step`, `set_base_precision`, `reset` — Task 2), sampler `step_callback` (Task 5).

- Produces: per-request precision lifecycle around every sampler branch. No new public API.

- [ ] **Step 1: Implement (no isolated unit test — covered by Task 8 e2e; the change is control-flow glue)**

In `generate_samples_from_batch`, just before the `if isinstance(sampler, FixedStepSampler) or scheduler_type == "unipc":` branch (after `fixed_step_sampler_kwargs` is built), insert:

```python
            # Mixed-precision diffusion steps: select W8A16/W8A8 once per
            # sampler step; reset (trace + staging cleanup) when the request
            # ends, including on error.
            _mixed_precision_runtime = getattr(self.net, "_mixed_precision_runtime", None)
            _step_callback = _mixed_precision_runtime.set_step if _mixed_precision_runtime is not None else None
```

Wrap the entire sampler section (both the UniPC/FixedStep branch including its `_extra_num_steps` dummy call, and the EDM branch including its dummy `x0_fn` padding) in:

```python
            try:
                ...existing branches...
            finally:
                if _mixed_precision_runtime is not None:
                    _mixed_precision_runtime.reset()
```

Inside the UniPC/FixedStep branch, pass the callback to the real call only:

```python
                latents = sampler(
                    velocity_fn,
                    initial_noise,
                    num_steps=num_steps,
                    shift=shift,
                    seed=seed,
                    step_callback=_step_callback,
                    **fixed_step_sampler_kwargs,
                )
                if _extra_num_steps > 0:
                    if _mixed_precision_runtime is not None:
                        # Deterministic cheap padding: the FSDP-alignment dummy
                        # call must not inherit the last real step's W8A16.
                        _mixed_precision_runtime.set_base_precision()
                    _ = sampler(
                        velocity_fn,
                        latents,
                        num_steps=_extra_num_steps,
                        shift=shift,
                        seed=seed,
                        **fixed_step_sampler_kwargs,
                    )
```

Inside the EDM branch, pass `step_callback=_step_callback` to the `sampler(x0_fn, initial_noise, ...)` call, and before the dummy `x0_fn` padding loop add the same `set_base_precision()` guard:

```python
                if _extra_num_steps > 0:
                    if _mixed_precision_runtime is not None:
                        _mixed_precision_runtime.set_base_precision()
                    ...existing dummy x0_fn padding loop...
```

- [ ] **Step 2: Sanity-check imports and syntax**

Run: `python -c "import cosmos_framework.model.generator.omni_mot_model"`
Expected: imports cleanly.

Run: `pytest cosmos_framework/model/generator/diffusion/samplers/ cosmos_framework/utils/generator/mixed_precision_test.py -v`
Expected: PASS (no regression).

- [ ] **Step 3: Commit**

```bash
git add cosmos_framework/model/generator/omni_mot_model.py
git commit -m "feat(inference): drive mixed-precision steps from the sampler loop"
```

---

### Task 7: CLI plumbing + load-time validation + runtime install in `load_model`

**Files:**

- Modify: `cosmos_framework/inference/common/args.py` (`QuantizationArgs` ~line 733, `QuantizationOverrides` ~line 741)
- Modify: `cosmos_framework/inference/inference.py` (`OmniInference._get_quantization_config`, ~line 1166)
- Modify: `cosmos_framework/inference/model.py` (`load_model`, ~line 652-736)
- Test: `cosmos_framework/utils/generator/mixed_precision_test.py` (validation), plus a CLI-args assertion in `cosmos_framework/inference/common/args.py`'s existing test file if present — otherwise the args round-trip test below lives in `mixed_precision_test.py`.

**Interfaces:**

- Consumes: `QuantizationConfig` mixed-precision fields (Task 1), `install_mixed_precision_runtime` (Tasks 2-4).

- Produces: CLI flags `--mixed-precision-first-steps`, `--mixed-precision-last-steps`, `--mixed-precision-reasoner-policy`, `--mixed-precision-w8a16-cache`; load-time errors per Global Constraints; runtime installed right after `apply_modelopt_fp8_checkpoint_inplace`.

- [ ] **Step 1: Write the failing tests**

Append to `cosmos_framework/utils/generator/mixed_precision_test.py`:

```python
def test_quantization_args_round_trip_mixed_precision_fields() -> None:
    from cosmos_framework.inference.common.args import QuantizationOverrides

    overrides = QuantizationOverrides(
        mixed_precision_first_steps=2,
        mixed_precision_last_steps=3,
        mixed_precision_reasoner_policy="base_precision",
        mixed_precision_w8a16_cache="none",
    )
    args = overrides.build_quantization()
    assert args.mixed_precision_first_steps == 2
    assert args.mixed_precision_last_steps == 3
    assert args.mixed_precision_reasoner_policy == "base_precision"
    assert args.mixed_precision_w8a16_cache == "none"


def test_validate_mixed_precision_load_rejects_non_fp8_checkpoint() -> None:
    from cosmos_framework.inference.model import _validate_mixed_precision_load

    with pytest.raises(ValueError, match="ModelOpt FP8"):
        _validate_mixed_precision_load(
            _enabled_config(), modelopt_checkpoint=False, use_cuda_graphs=False
        )


def test_validate_mixed_precision_load_rejects_cuda_graphs() -> None:
    from cosmos_framework.inference.model import _validate_mixed_precision_load

    with pytest.raises(ValueError, match="cuda_graphs"):
        _validate_mixed_precision_load(
            _enabled_config(), modelopt_checkpoint=True, use_cuda_graphs=True
        )


def test_validate_mixed_precision_load_noop_when_disabled() -> None:
    from cosmos_framework.inference.model import _validate_mixed_precision_load

    _validate_mixed_precision_load(QuantizationConfig(), modelopt_checkpoint=False, use_cuda_graphs=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest cosmos_framework/utils/generator/mixed_precision_test.py -v -k "round_trip or validate_mixed"`
Expected: FAIL (missing fields / missing `_validate_mixed_precision_load`)

- [ ] **Step 3: Implement**

`args.py` — extend `QuantizationArgs`:

```python
class QuantizationArgs(ArgsBase):
    """Low-precision quantization arguments applied to the model at load time."""

    quantization_method: QuantizationMethod | None
    quantization_include_regex: list[str]
    quantization_exclude_regex: list[str]
    mixed_precision_first_steps: int
    mixed_precision_last_steps: int
    mixed_precision_reasoner_policy: Literal["high_precision", "base_precision"]
    mixed_precision_w8a16_cache: Literal["none", "generation", "all", "cpu_block", "gpu_block"]
```

and `QuantizationOverrides` (after `quantization_exclude_regex`):

```python
    mixed_precision_first_steps: int = pydantic.Field(default=0, ge=0)
    """ModelOpt FP8 checkpoints only: run the first N diffusion steps with 16-bit
    activations (W8A16) instead of FP8 activations (W8A8). 0 disables."""
    mixed_precision_last_steps: int = pydantic.Field(default=0, ge=0)
    """ModelOpt FP8 checkpoints only: run the last N diffusion steps with W8A16."""
    mixed_precision_reasoner_policy: Literal["high_precision", "base_precision"] = "high_precision"
    """Understanding-pathway precision when mixed precision is enabled:
    high_precision keeps reasoner linears on W8A16 for every step."""
    mixed_precision_w8a16_cache: Literal["none", "generation", "all", "cpu_block", "gpu_block"] = "gpu_block"
    """W8A16 dense-weight source: none dequantizes per call; generation/all hold
    resident BF16 caches; gpu_block/cpu_block stream per-layer double buffers.
    FSDP-sharded runs support only none."""
```

`inference.py` — extend `_get_quantization_config`:

```python
    @classmethod
    def _get_quantization_config(cls, setup_args: SetupArgs) -> QuantizationConfig:
        return QuantizationConfig(
            method=setup_args.quantization_method,
            include_regex=list(setup_args.quantization_include_regex),
            exclude_regex=list(setup_args.quantization_exclude_regex),
            mixed_precision_first_steps=setup_args.mixed_precision_first_steps,
            mixed_precision_last_steps=setup_args.mixed_precision_last_steps,
            mixed_precision_reasoner_policy=setup_args.mixed_precision_reasoner_policy,
            mixed_precision_w8a16_cache=setup_args.mixed_precision_w8a16_cache,
        )
```

`inference/model.py` — module-level helper next to the other private helpers:

```python
def _validate_mixed_precision_load(
    quantization_config: "QuantizationConfig", *, modelopt_checkpoint: bool, use_cuda_graphs: bool
) -> None:
    """Fail fast on unsupported mixed-precision-step combinations."""
    if not quantization_config.mixed_precision_enabled:
        return
    if not modelopt_checkpoint:
        raise ValueError(
            "mixed_precision_first_steps/last_steps require a ModelOpt FP8 checkpoint "
            "(hf_quant_config.json with quant_method='modelopt', quant_algo='FP8')."
        )
    if use_cuda_graphs:
        raise ValueError(
            "Mixed-precision diffusion steps are incompatible with use_cuda_graphs: "
            "per-step precision switching cannot be captured/replayed."
        )
```

In `load_model`, right after `modelopt_checkpoint = is_modelopt_fp8_checkpoint(checkpoint_path)` and the existing method-conflict check, add:

```python
        _validate_mixed_precision_load(
            quantization_config,
            modelopt_checkpoint=modelopt_checkpoint,
            use_cuda_graphs=compile_config.use_cuda_graphs,
        )
```

And right after the `apply_modelopt_fp8_checkpoint_inplace(...)` call (before `return model`), add:

```python
                        # Mixed precision installs after FP8 weights are in
                        # place and the module graph is final (parallelize +
                        # materialize already ran inside build_net).
                        from cosmos_framework.utils.generator.mixed_precision import (
                            install_mixed_precision_runtime,
                        )

                        install_mixed_precision_runtime(model.model.net, quantization_config)
```

(The runtime resolves decoder blocks itself via `net.language_model.model.layers` when a block cache mode is configured — Task 4.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest cosmos_framework/utils/generator/mixed_precision_test.py -v`
Expected: PASS. Also run `python -c "import cosmos_framework.inference.inference"` to confirm imports.

- [ ] **Step 5: Commit**

```bash
git add cosmos_framework/inference/common/args.py cosmos_framework/inference/inference.py cosmos_framework/inference/model.py cosmos_framework/utils/generator/mixed_precision_test.py
git commit -m "feat(inference): CLI flags, load-time validation, and runtime install for mixed-precision steps"
```

---

### Task 8: End-to-end verification on `cosmos3-nano-fp8-14072026` (single GPU)

**Files:**

- No committed changes (results go in the PR description).

**Interfaces:**

- Consumes: everything above; the FP8 checkpoint from `nvidia/Cosmos3-Experimental@f0cdb8ea37360e8510e2c0caf84c0f9f3e8751c8`, subfolder `cosmos3-nano-fp8-14072026`.

- [ ] **Step 1: Download the checkpoint**

```bash
export HF_HOME=/lustre/fsw/portfolios/cosmos/projects/cosmos_base_training/users/pzeren/hf-cache
python - <<'EOF'
from huggingface_hub import snapshot_download
path = snapshot_download(
    "nvidia/Cosmos3-Experimental",
    revision="f0cdb8ea37360e8510e2c0caf84c0f9f3e8751c8",
    allow_patterns=["cosmos3-nano-fp8-14072026/*"],
)
print(path + "/cosmos3-nano-fp8-14072026")
EOF
```

- [ ] **Step 2: Baseline W8A8 run (flags off — must behave exactly as before this change)**

```bash
CKPT=<path printed above>
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/omni/t2i.json" \
    -o /tmp/mp_e2e/w8a8 \
    --checkpoint-path "$CKPT" \
    --seed=0
```

(If `inputs/omni/t2i.json` does not exist in the repo, use `inputs/omni/t2v.json`
as in docs/inference.md's quick start — any single sample works; keep it
identical across all runs.)

Confirm: no `Mixed precision installed` log line; output generated.

- [ ] **Step 3: Mixed run (3/3, default gpu_block) + trace check**

```bash
python -m cosmos_framework.scripts.inference \
    --parallelism-preset=latency \
    -i "inputs/omni/t2i.json" \
    -o /tmp/mp_e2e/mixed_gpu_block \
    --checkpoint-path "$CKPT" \
    --seed=0 \
    --mixed-precision-first-steps 3 --mixed-precision-last-steps 3
```

Confirm the logs show `Mixed precision installed: ... linears={'reasoner': N, 'generation': M}` with both counts > 0, and `MIXED_PRECISION_TRACE steps=W8A16,W8A16,W8A16,W8A8,...,W8A8,W8A16,W8A16,W8A16` matching the sampler's step count.

- [ ] **Step 4: Cache-mode equivalence + full-W8A16 sweep**

Repeat Step 3 with `--mixed-precision-w8a16-cache none`, `generation`, `all`, `cpu_block`, and once with `--mixed-precision-first-steps 999` (all steps W8A16). Same seed for all runs. Confirm:

- all five cache modes produce identical (or bitwise-near, allclose) outputs to each other;

- the trace for the 999 run is all `W8A16`;

- record per-mode peak memory (`torch.cuda.max_memory_allocated` is logged by the framework; otherwise `nvidia-smi` sampling) and wall time for the PR description.

- [ ] **Step 5: Error-path spot checks**

```bash
# non-FP8 checkpoint + flags -> must fail fast with the ModelOpt FP8 message
python -m cosmos_framework.scripts.inference --checkpoint-path Cosmos3-Nano \
    --mixed-precision-first-steps 3 ... ; echo "exit=$?"
```

Confirm the `ValueError` message names ModelOpt FP8.

- [ ] **Step 6: Record results in the PR description (not in committed docs)**

---

## Self-review notes

- Spec coverage: config surface (Task 1, 7), schedule + `num_steps==1` (Task 1), runtime/tagging/validation/trace/logs (Task 2), dispatch (Task 2), full caches (Task 3), block providers + preload_first + hooks-only-when-W8A16 (Task 4), sampler hook incl. EDM multi-eval and `sample_clean` (Task 5), lifecycle + FSDP padding base-precision + try/finally (Task 6), load-time errors (FSDP×cache inside runtime install in Task 2/4; non-FP8 + cuda_graphs in Task 7), install ordering after weight install (Task 7), e2e (Task 8).
- FSDP×cache validation lives in `MixedPrecisionRuntime.install` (detects `DTensor` weights directly — runtime truth) rather than in `load_model`, deliberately.
- Type consistency: `install_mixed_precision_runtime(net, config, blocks=None)` gains `blocks` in Task 4; Tasks 2-3 call it without `blocks` (compatible). `_mixed_precision_staged_weight` defined in Task 2, written by Task 4 hooks, read by `resolve_w8a16_weight`.
