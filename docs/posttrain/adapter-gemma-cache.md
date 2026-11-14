# Gemma cache-consistency candidate

`gemma_fp32_fixed_mma_sdpa64_v1` is an experimental inference profile for the
recorded `unsloth/gemma-2b-it` checkpoint at
`5ae754dee0b5cbf7f7fb39a7731aefa6b7987c2d`. CPU correctness and public artifact
composition tests pass. Real-model numerical, task and resource qualification
is pending. This is not Gemma2 support or a new training execution mode.

The preserved fixed-MMA traces first differ at attention reduction after equal
projection, normalization, RoPE, mask and cache operands. The new profile keeps
the fixed-MMA projections, Gemma's offset RMSNorm and separate adapter branches.
It converts attention Q/K/V and floating masks to FP64 after the native FP32
RoPE/cache update, uses the same scale, causal masking and grouped-head semantics,
and rounds the attention output to FP32 before its output projection. Persistent
weights, serialized adapters, tied embeddings and KV caches remain FP32.

FP64 attention can increase transient memory and latency substantially, especially
at 8K context. Whole-process peak memory, cache and weight bytes, first-use and
warm prefill/decode/prefix timings must be measured. CPU tests do not establish
the real-model cache or merged-output gates. No efficiency improvement is claimed.

Use an explicit public portable profile only in the pinned reasoner environment:

```toml
[model.portable_adapter]
directory = "/path/to/immutable/portable-adapter"
execution_profile = "gemma_fp32_fixed_mma_sdpa64_v1"
cache_effective_weights = false
```

The existing generation/evaluation configuration must point `model.model_name`
at that artifact's immutable base and request FP32, SDPA, batch one and a single
A100. The ordinary portable loader validates the base, processor, method and
execution identity before installing the profile. A merged export must record
the same explicit profile through the existing adapter export command; the
Cosmos public merged loader validates and activates that tag. Plain stock HF
loading does not activate this arithmetic.

The profile rejects training/autograd, autocast, offload, compilation, CUDA
graphs, effective-weight caching and foreign-profile prefixes. Changing tensors,
adapter settings, attention configuration or installed forwards invalidates
reuse. Restoring and reinstalling creates a new prefix owner.

`gemma_fp32_sdpa_contiguous_kv_v1` and `gemma_fp32_fixed_mma_v1` retain their
original arithmetic and failed numerical evidence. The new candidate uses the
same 34 original and four separately frozen inputs, unchanged cache and merge
thresholds, common-history and split-prefix comparisons, forced 32-token
generation, and 2K/8K contexts. Complete paired GSM8K evaluation is required
after numerical correctness passes; no training is repeated for this candidate.
