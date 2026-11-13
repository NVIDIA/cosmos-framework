# SmolLM consistent FP32 inference

`smollm_fp32_consistent_v1` is an opt-in inference profile for the pinned
`HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac`
base and its v1 factorized adapter artifacts. It keeps the original FP32
parameters, ties, low-rank branches and saved PiSSA factors. Linear, RMSNorm and
SDPA reductions use FP64 temporaries and round their results back to FP32.
RoPE, cache updates, masks, activations and residual additions remain native.

This creates a **new inference oracle**. It does not change training execution,
overwrite artifacts, or erase the original full-versus-cached numerical failures.
The campaign's source, installed-package, numerical, task and resource receipts
remain necessary before any final checkpoint is described as qualified. The
initial three-case mechanism diagnostic is not full qualification.

The implementation admits FP32 materialization on one A100, batch one, eager
inference, Torch 2.10.0 and Transformers 5.8.1. Use the separately resolved
reasoner runtime described in [public adapter inference](adapter-public-inference.md).
Do not combine the Transformers-5 inference environment with incompatible native
training extras. Portable assembly and export admission check the artifact
dependency and processor identities; merged loading validates the committed
profile/identity and strictly loads tensors. It does not perform an additional
processor-file hash audit.
This profile does not admit arbitrary Llama checkpoints, `pissa_dense_v1`,
rounded BF16 training artifacts, autocast, training/autograd, graphs/compilation,
offload, static caches, or effective-weight caching. Component dropout is
disabled by evaluation, including independently configured LoRA/DoRA dropout.

Load a portable adapter through the public runtime outside `inference_mode`:

```python
import torch
from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime

torch.set_float32_matmul_precision("highest")
runtime = CosmosFrameworkRuntime(
    "/models/SmolLM2-135M-Instruct/12fd25f77366fa6b3b4b768ec3050bf629380bac",
    dtype="float32",
    portable_adapter={
        "directory": "/artifacts/portable",
        "execution_profile": "smollm_fp32_consistent_v1",
    },
)
inputs = runtime.processor("A short text prompt", return_tensors="pt").to(runtime.device)
with torch.inference_mode():
    generated = runtime.model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(runtime.processor.batch_decode(generated, skip_special_tokens=True))
```

Export a separate tagged merge using the existing transactional exporter:

```bash
python -m cosmos_framework.posttrain.checkpoint.tools.adapters \
  --adapter /artifacts/portable --base /models/SmolLM2-135M-Instruct/12fd25f77366fa6b3b4b768ec3050bf629380bac \
  --merged --output-dtype float32 --execution-profile smollm_fp32_consistent_v1 \
  --out /artifacts/merged-smollm-consistent-v1
```

Load that directory with `CosmosFrameworkRuntime(path, dtype="float32")`.
Its config and commit marker must agree before loading. Plain Transformers
`from_pretrained` alone does **not** activate the required arithmetic. The
merged tensors retain the existing FP32 merge values; no adapter branches are
retained in the merged model.

Native `DynamicCache` supports incremental decoding and cloned prefix reuse.
Populated caches must originate from the same model/profile installation;
foreign caches and prefixes from an earlier installation are rejected. Inputs
are bounded to 8K plus 32 continuation tokens. The campaign's preserved long
inputs use their original, stricter context lengths. Weight/device/dtype or
adapter-setting changes require restoration and a fresh load. No derived dense
weight cache is used.

`runtime.restore_inference_execution()` restores the original FP32 forwards,
clears runtime-owned caches and updates deployment provenance. Discard external
prefixes when restoring. Reinstalling the profile creates a new cache owner;
old prefixes cannot be silently reused. Restoration is an explicit return to
the old oracle and its recorded limitations.

Resource and task measurements must be reported separately from consistency.
FP64 attention can allocate a substantially larger working set than fused FP32
attention. The full 1,319-row GSM8K comparison uses the same reused held-out
questions and capped decoding protocol as the original final checkpoint; it is
not a new untouched test or a statistically robust method ranking.

## Fixed projection reduction, version 2

`smollm_fp32_fixed_mma_v2` is a separately named inference execution profile for
the same pinned base, factorized artifacts, public loaders and FP32 boundaries.
It replaces projection GEMMs with one A100 FP64 MMA schedule: 16 rows, 32 output
columns and 32 reduction elements per tile, four warps, one stage, and disabled
floating-point fusion. Every row follows the same reduction tiles regardless of
prefill length or a split prefix. RMSNorm, attention, RoPE, masks and cache updates
retain the version-1 implementation. It requires Triton 3.6.0 and retains separate
base, low-rank and initial PiSSA branches; it creates no derived weight cache.

Use `smollm_fp32_fixed_mma_v2` in the portable `execution_profile` or the export
command above, with a new output directory. Never edit an existing export's
profile tag. Config and transaction declarations must agree, and a populated
version-1 prefix cannot be reused by version 2. Restoration and the other
precision, checkpoint, hardware and inference-only restrictions above apply.

The original version-1 final LoRA/FSDP checkpoint fails the preserved split-prefix
mean-error gate at the 2K case (`4.40152e-6 > 2e-6`). The fixed-tile diagnostic
passes all 34 frozen cases in portable, fresh portable reload and merged runs
(job `38480714`), with unchanged gates and tensors. That run used the original
installed package plus explicitly hashed experimental code; it is **not**
qualification of this production integration. Installed version-2 execution and
its complete matched 1,319-question evaluation remain required. The diagnostic
peaked at about 13.56 GB CUDA allocation; weights occupied 539,903,488 bytes for
portable and 538,060,288 bytes for merged. These resource results are separate
from task quality and do not establish a speedup.
