# Transfer numerical preservation

Compiled causal single-view RGB Transfer preserves native RMSNorm reductions,
low-precision cast boundaries, and chronological KV reads. Validated
edge/blur/depth/seg requests enable these safeguards automatically. There is
no additional CLI option; compilation and CUDA Graphs use the ordinary settings.

## Implementation and scope

- The shared `Qwen3VLTextRMSNorm.forward` retains its original implementation.
  Internal `preserve_transfer_numerics` metadata travels with the Transfer
  request through its AR memory state and each `ARMemoryValue`. Only marked
  decoder calls select native ATen RMSNorm during compiled evaluation without
  gradients. Eager execution and training retain their differentiable path.
- Transfer has a separate, lazily compiled callable sharing the original block
  and weights. With CUDA Graphs, its compiler options enable
  `emulate_precision_casts`; block scope retains `reduce-overhead`, and
  whole-forward scope retains external capture. The ordinary callable keeps
  its original compiler options. No shared module enable flag is changed.
- The same request metadata selects static KV order
  `[text | history | current | padding]`. Other AR requests retain
  `[text | current | history | padding]` and their existing fixed-offset copy.
  Current K/V are scratch values; clean cache commits remain authoritative.
- The inference loader keeps its existing recompile limit of 8.
- Both AR generators release their request's KV cache ownership in `finally`,
  including early close and exceptions, while retaining pooled tensor storage.
  Transfer consumers close their iterators if output processing fails. This
  shared lifecycle fix is independent of the Transfer numerical policy.

Ordinary Bimanual forward dynamics, Nano/Super text-to-video, Auto WSM and
joint camera/LiDAR, and Edge requests retain their original numerical paths. Compilation
and evaluation mode alone cannot activate Transfer preservation, and a later
ordinary request on the same model does not inherit it. Selection follows the
current request rather than a model-name allowlist. Edge's Nemotron norm is
unchanged; this implementation preserves the dense Qwen RMSNorm used by the
validated Transfer checkpoint.

Cast preservation covers the normal `ar_post_saturation_mode=default`
compilation path. The optional nondefault post-saturation compiler is separate;
its cast-preservation gap predates this scope change and is not covered here.
Noncausal or multiview Transfer paths are also outside this change's scope.

## Validation

Run the focused numerical and cache tests from the repository root:

```bash
python -m cosmos_framework.model.generator.utils.numerics_test
python -m pytest cosmos_framework/model/generator/utils/kv_cache_test.py -m CPU --num-gpus=0
```

These checks exercise numerical policy selection and cache behavior. Validate
GPU output preservation separately with a publicly available checkpoint and
identical inputs, seed, runtime, and hardware in both runs. Preserve earlier
results by using a fresh output directory for each run.

## Accuracy limits

CPU checks do not establish GPU output equivalence, task quality, all-model
performance, or arbitrary-workload stability. Compare decoded outputs and
per-request quality metrics for the intended workload before drawing those
conclusions.
