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
