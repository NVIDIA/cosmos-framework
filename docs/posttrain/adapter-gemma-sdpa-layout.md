# Experimental Gemma1 contiguous K/V inference

`gemma_fp32_sdpa_contiguous_kv_v1` is an opt-in candidate for the existing
`unsloth/gemma-2b-it` binding at
`5ae754dee0b5cbf7f7fb39a7731aefa6b7987c2d`. It is not a Gemma2 profile and does
not change training execution or previously recorded qualification outcomes.
Real GPU qualification is required before claiming deployment support.

The preserved 66-token Gemma1 split-prefix fixture exposes a large SDPA error
at the last query when masked attention expands its single K/V head into a
zero-head-stride view. The candidate materializes those expanded K/V operands
as contiguous tensors. It reuses the pinned native SDPA wrapper and keeps
unmasked native GQA, attention masks, positions, RoPE, scaling, FP32 reductions,
low-rank branches and persistent single-head KV cache unchanged. Parameters,
tying, processor assets and merged tensor values remain unchanged. Plain stock
loading does not install this inference behavior.

Public text preparation uses the tokenizer's native chat template, including its
special tokens, exactly once. In particular, the rendered Gemma template already
contains BOS; the subsequent padded tokenization disables automatic special-token
insertion. This matches direct `apply_chat_template(tokenize=True)` inputs without
changing the prompt, model weights or attention profile. Historical results keep
their recorded token IDs and implementation revision.

Admission requires the exact recorded base and processor, factorized adapter
identity, Torch 2.10.0, Transformers 5.8.1, FP32, SDPA, one A100, batch one and
eager evaluation without autograd, autocast, TF32, graphs, offload or alternate
cache implementations. The input bound is an 8192-token prefix plus 32 decoding
tokens. Prefixes belong to one installation and can be deep-copied for reuse;
foreign prefixes and mutations of owned tensors/configuration are rejected.
No effective-weight cache is supported. K/V expansion adds temporary memory
and copy cost; measurements are part of candidate qualification.

For public portable loading, supply the immutable base as `model_path` and:

```json
{
  "dtype": "float32",
  "attn_implementation": "sdpa",
  "portable_adapter": {
    "directory": "/absolute/path/to/portable",
    "execution_profile": "gemma_fp32_sdpa_contiguous_kv_v1"
  }
}
```

For a separate, explicitly tagged FP32 merged artifact, use the existing
`cosmos_framework.posttrain.checkpoint.tools.adapters` export entry point with
`--execution-profile=gemma_fp32_sdpa_contiguous_kv_v1` and
`--output-dtype=float32`. The existing transaction binds this profile in both
the config and completion marker. `CosmosFrameworkRuntime` validates that
binding and installs it automatically. Untagged historical exports retain
their original execution.

This candidate addresses a demonstrated operand-layout failure. It does not
promise shape-independent FP32 projection/reduction arithmetic or satisfy the
strict full-versus-cached and merge gates merely because that large error is
removed. Keep original failures, candidate numerical outcomes, task quality
and resource measurements separate. No full task or GPU pass is claimed by
the CPU contract tests.

## Experimental fixed-MMA projection and normalization profile

`gemma_fp32_fixed_mma_v1` is a separate, unqualified inference candidate for
the same exact Gemma1 binding. The recorded full-versus-cached traces first
diverge in layer-0 low-rank/base projections or RMSNorm. Identical retained
projection operands produce different FP32 outputs as the sequence shape
changes. Correcting K/V layout alone leaves those differences unchanged.

The new profile reuses the existing A100 fixed FP64-MMA projection kernel.
Every token row uses the same reduction tiles; native linear outputs and the
separate factorized adapter result round to FP32. LoRA/rsLoRA, DoRA and PiSSA
retain their low-rank branches, magnitude parameters and saved initial factors.
DoRA's dense direction only computes its normalization magnitude; it is not
used as an effective-weight forward. Parameters and persistent caches remain
FP32, with no derived-weight cache or parameter replacement.

Gemma RMSNorm uses FP64 transients for mean-square and inverse-square-root
reduction, then rounds the normalized activation to FP32 before its original
FP32 `1 + weight` multiplication. The canonical FP32 embedding scale is
unchanged. Attention continues through the existing contiguous-K/V FP32 SDPA
implementation: masks, native RoPE, cache storage and attention arithmetic are
not switched. The additional fixed-MMA dependency is Triton 3.6.0.

Select this candidate explicitly in `portable_adapter.execution_profile`, or
export a separate artifact with
`--execution-profile=gemma_fp32_fixed_mma_v1 --output-dtype=float32`.
The public loader validates the distinct profile/transaction identity and its
single-A100, batch-one, eager inference contract. Plain Hugging Face loading
does not install these operators. Training, autograd, autocast, offload,
effective-weight caching and foreign/alternate cache implementations are
rejected. Restore the current installation before changing profiles; rebuild
its prefixes afterward.

This is a new execution oracle. It does not reinterpret `factorized_v1` or the
earlier contiguous-K/V profile. Their original numerical failures and task
scores remain attached to their own revisions. CPU algebra and lifecycle tests
do not qualify real-model deployment. Qualification must retain the original
34 inputs, independently fixed operator/input controls, numerical thresholds,
paired task comparisons and measurements of first-use cost, latency, transient
memory and KV storage. No GPU, task-quality or performance pass is claimed for
this candidate before those checks execute.
