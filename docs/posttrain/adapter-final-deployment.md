# Final trained-artifact deployment

This continuation starts at `bf622ea` and evaluates the final update-1111
Qwen3-VL-2B-Instruct mixed-component WTS artifacts. It does not repeat training
or change the historical 12-checkpoint / 48-profile / 47-pass qualification.
The reused scene-disjoint 576-question benchmark is part of validation, not an
untouched final test. Physical-model correctness is not task-performance evidence.

## Portable inference weight reuse

After strict portable loading, the explicit canonical inference installer accepts
`cache_effective_weights=True`. It reuses the existing rounded effective-weight
construction for LoRA, rsLoRA, DoRA and PiSSA, without changing factors, initial
factors, magnitude, base parameters, optimizer membership or checkpoint state.
The default remains uncached. This cache is separate from the KV/prefix cache.

```python
from cosmos_framework.posttrain.model.adapter_bf16_execution import install_adapter_inference_profile
from cosmos_framework.posttrain.model.adapter_inference_cache import (
    adapter_inference_cache_info,
    bind_adapter_inference_cache_optimizer,
    clear_adapter_inference_cache,
)

# net is the strictly loaded portable model, on the supported GPU, in eval mode.
install_adapter_inference_profile(
    net, "qwen3_vl_bf16_canonical_cache_v1", cache_effective_weights=True
)
print(adapter_inference_cache_info(net))
# Optional: bind every optimizer that may update parameters between inference calls.
handle = bind_adapter_inference_cache_optimizer(net, optimizer)
# Use no_grad/inference_mode for inference; the canonical profile cannot train.
# Unsafe .data/custom writes that bypass tensor versions require explicit clearing:
clear_adapter_inference_cache(net)
# Remove the binding when no longer needed; reinstalling with caching disabled
# also removes bindings and releases all derived tensors.
handle.remove()
```

Version-tracked in-place writes, tensor replacement, dtype/device conversion,
state loading and entering training invalidate derived weights. Fused optimizers
can bypass tensor versions: bind them explicitly as above or clear after each
update. Bound optimizers clear before and after stepping, including partial
updates followed by an exception. Unbound fused/custom writes are not detected
automatically. Do not mutate weights concurrently with inference. Derived tensors
are neither Parameters nor buffers and are never exported or checkpointed.

## Canonical attention optimization

The original fixed-tile kernel remains the correctness reference. Long Qwen3-VL
prefill with head dimension 128 and at least 1024 query positions uses a one-warp
launch to reduce cross-warp exchanges. It retains the same 64-key FP64 reductions,
online recurrence and BF16 output. Other architectures, head dimensions, short
inputs and decoding retain the original four-warp launch. This does not introduce
whole-model FP32 execution or change the execution-profile identity.

The deployment matrix was fixed before candidate evaluation, including a
two-second 8K prefill budget and comparisons against stock BF16 and FP32. Passing
numerical checks does not imply that the additional performance criteria pass.
Final installed-package measurements and task results are recorded separately.

## Evidence location

Local evidence and reproduction scripts:
`/localhome/local-rarunachalam/cosmos-adapter-final-deployment-20261105`.
The original final factorized portable model and its strict merge failure remain
unchanged. Higher-precision factorized-branch diagnostics are separate experiments,
not a silent reinterpretation of `factorized_v1` artifacts.
