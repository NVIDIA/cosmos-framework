# FP8 Mixed-Precision Diffusion Steps (W8A16 edge steps, W8A8 middle)

**Date:** 2026-08-23
**Status:** Approved design, pending implementation plan
**Reference implementation:** vllm-omni branch `mixed-precision-diffusion-steps`
(`vllm_omni/diffusion/models/cosmos3/mixed_precision/`), by wkutak.

## Goal

For ModelOpt static-FP8 checkpoints (Cosmos3 Nano / Super / Super-I2V /
Super-T2I, e.g. `nvidia/Cosmos3-Experimental` folders
`cosmos3-*-fp8-14072026`), run a configurable number of diffusion steps at the
start and end of denoising with 16-bit activations (W8A16) while the middle
steps keep the existing FP8-activation path (W8A8). The first and last steps
set global structure and final detail and are most sensitive to activation
quantization error; the middle steps tolerate it and keep the W8A8 speedup.

Weights are always the checkpoint's FP8 E4M3 tensors. W8A16 means: dequantize
the same FP8 weight to the model compute dtype (BF16) and run a dense
`F.linear`, skipping activation quantization. No checkpoint change is needed.

Functional parity target: all four vllm-omni knobs (first/last step widths,
reasoner policy, W8A16 weight-cache mode) and all five cache modes
(`none` / `generation` / `all` / `cpu_block` / `gpu_block`).

## Current framework state (baseline)

- `is_modelopt_fp8_checkpoint` auto-detects `hf_quant_config.json`
  (quant_method `modelopt`, quant_algo `FP8`); `plan_modelopt_fp8_targets`
  derives target FQNs from E4M3 tensors in the diffusers weight map;
  `swap_modelopt_fp8_linears_on_meta` swaps targets to `_ModelOptFloat8Linear`
  before FSDP wrap; `apply_modelopt_fp8_checkpoint_inplace` installs TorchAO
  `PrototypeFloat8Tensor` weights with static per-tensor weight and activation
  scales. `F.linear` on that subclass quantizes the activation and runs the
  FP8 GEMM — this is the W8A8 base path.
- Quantized inventory (verified on `cosmos3-nano-fp8-14072026`): 36 MoT
  decoder layers × 14 linears, both pathways of each layer —
  understanding/reasoner (`mlp.{gate,up,down}_proj`,
  `self_attn.{q,k,v,o}_proj`) and generation (`mlp_moe_gen.*`,
  `self_attn.{q,k,v,o}_proj_moe_gen`). `proj_in`/`proj_out`/`lm_head`/vision
  tower stay high precision in the export.
- Samplers (`fixed_step.py`, `unipc.py`, `edm.py`) call
  `velocity_fn(latent, timestep)` once per denoising step; CFG cond/uncond
  forwards happen inside one `velocity_fn` call. There is no per-step hook.

## Configuration

`QuantizationConfig` (attrs) and `QuantizationOverrides` (pydantic CLI) gain
four fields:

| CLI flag | Values | Default |
|---|---|---|
| `--mixed-precision-first-steps` | int ≥ 0 | 0 |
| `--mixed-precision-last-steps` | int ≥ 0 | 0 |
| `--mixed-precision-reasoner-policy` | `high_precision` / `base_precision` | `high_precision` |
| `--mixed-precision-w8a16-cache` | `none` / `generation` / `all` / `cpu_block` / `gpu_block` | `gpu_block` |

- **Enablement:** active iff `first_steps + last_steps > 0` AND the loaded
  checkpoint is ModelOpt FP8. Defaults (0/0) leave behavior byte-identical to
  today's W8A8 path (nothing is installed).
- Mixed-precision flags set on a non-ModelOpt-FP8 checkpoint → `ValueError`
  at load (fail fast, mirroring vllm's `validate_quant_config`).
- `dp_shard_size > 1` (FSDP) → only `w8a16_cache="none"` is allowed; other
  modes raise with an explanation. Resident BF16 caches and block prefetch
  are incompatible with per-forward all-gathered DTensor shards; the
  dequantize-on-the-fly path works unchanged because the forward sees the
  gathered full `PrototypeFloat8Tensor`.
- No `format` knob (vllm has one): the framework auto-detects the checkpoint
  format.

**Step schedule** (identical to vllm):
`high_precision = step_index < first_steps or step_index >= num_steps - last_steps`.
Overlap (`first+last >= num_steps`) → all steps W8A16. `num_steps == 1` is
special-cased to W8A8, matching vllm.

## New module: `cosmos_framework/utils/generator/mixed_precision.py`

Native implementation preserving vllm's concept structure (approach B —
chosen over porting the LinearMethod-wrapper architecture, which has no
framework counterpart).

### `MixedPrecisionRuntime`

Attached to the VFM network as `net._mixed_precision_runtime` when enabled.

- **Install:** scans for `_ModelOptFloat8Linear` modules and classifies each
  by FQN: contains `_moe_gen` → `generation` path, else → `reasoner` path.
  Tags each module with its path and a back-reference to the runtime.
  Validates both inventories are non-empty.
- **State:** the config, the current step's `generation_high_precision`
  flag, `use_high_precision(path)` (reasoner → static policy; generation →
  current step flag), and a per-request precision trace logged on reset
  (`MIXED_PRECISION_TRACE steps=W8A16,W8A8,...`).
- **Lifecycle:** `set_step(step_index, num_steps)` at each sampler step
  (also triggers block-provider `preload_first` when the step selects
  W8A16); `reset()` in a `finally` around the sampler call (log trace, sync
  the staging stream, clear staged state, clear the flag).

### W8A16 weight sources (cache modes)

| Mode | Behavior | Memory |
|---|---|---|
| `none` | Dequantize per call: `weight.qdata.to(bf16) * weight.scale` | zero extra |
| `generation` | Non-persistent BF16 buffers for all generation-path linears, filled once after load | ~2× FP8 gen weights, resident |
| `all` | Same, for both paths | ~2× all FP8 weights, resident |
| `cpu_block` | Two max-block device slots; next layer's BF16 streamed H2D from pinned host copies | 2 block slots device + full BF16 pinned host |
| `gpu_block` (default) | Two max-block device slots; next layer's BF16 dequantized from resident FP8 on a side stream | 2 block slots device |

`W8A16BlockWeightProvider` (gpu_block/cpu_block) is a port of vllm's
`block_cache.py`: per-`MoTDecoderLayer` `forward_pre_hook`/`forward_hook`
(hooks only act when the current step is W8A16), a side CUDA stream, and
ready/free CUDA events rotating the two slots so dequantize/H2D of layer N+1
overlaps compute of layer N. Block providers cover **generation-path linears
only**; the reasoner pathway is interleaved with the gen pathway inside each
decoder layer and is not block-prefetchable. When
`reasoner_policy=high_precision`, reasoner linears use the `all` full cache if
selected, else dequantize on the fly.

### Dispatch: extend `_ModelOptFloat8Linear.forward`

```python
def forward(self, inputs):
    runtime = self._mixed_precision_runtime   # None when disabled
    if runtime is not None and runtime.use_high_precision(self._mixed_precision_path):
        weight = <staged slot | cached buffer | on-the-fly dequantized>
        return F.linear(flat_inputs, weight, self.bias).reshape(output_shape)
    return <existing W8A8 subclass path>
```

The existing zero-row and rank-flattening workarounds stay shared. Weight
resolution priority mirrors vllm: staged slot view → full-cache buffer →
on-the-fly dequantization.

`torch.compile`: the precision flag is a Python bool read in forward, so each
precision compiles its own variant (2 total); first W8A16 step pays a
compile. Accepted; documented.

## Per-step hook

All three samplers gain an optional
`step_callback: Callable[[int, int], None] | None` parameter, invoked as
`step_callback(step_index, num_steps)` before each step's `velocity_fn`.
`generate_samples_from_batch` (omni_mot_model) passes a callback when the
runtime is installed, and wraps the sampler call in `try/finally` with
`runtime.reset()` in `finally`.

- CFG cond/uncond forwards share one step → one precision selection, matching
  vllm semantics.
- FSDP `_extra_num_steps` dummy sampler padding calls run after the real
  call; their precision is whatever state remains. Output is discarded and
  mixed precision changes GEMM kernels, not the collective sequence, so
  alignment is unaffected. (To keep kernels cheap, `reset()` runs after the
  padding calls; padding thus executes W8A8 unless the trailing state was
  W8A16 — harmless either way.)

## Error handling

- Mixed-precision flags + non-FP8 checkpoint → `ValueError` at load.
- FSDP + cache mode other than `none` → `ValueError` at load.
- Install finds an empty reasoner or generation inventory → `ValueError`
  (unexpected checkpoint shape).
- `set_step` with out-of-range index → `IndexError` (ported validation).
- Negative step widths rejected by pydantic field validation.

## Testing

Unit (`cosmos_framework/utils/generator/mixed_precision_test.py`):

- Step schedule: first/last widths, overlap, `num_steps==1`, index
  validation.
- FQN classification (`_moe_gen` → generation, else reasoner).
- Numerics: W8A16 on-the-fly path ≡ manual `(qdata.to(bf16) * scale) @ x`;
  full-cache and staged-slot outputs ≡ on-the-fly outputs.
- Block provider: slot rotation, ready/free event ordering, reset mid-request
  (single GPU).
- Config validation: FSDP × cache-mode rejection, non-FP8 checkpoint
  rejection, disabled-by-default no-op.

End-to-end (manual, on one GPU with `cosmos3-nano-fp8-14072026`):

- t2i with first/last = 0/0 (pure W8A8), 35/35-equivalent (all W8A16), 3/3;
  verify trace log step sequence; compare outputs across all five cache
  modes for bitwise/near equality; record memory and latency per mode.
  Performance numbers go in the PR description, not committed docs.
