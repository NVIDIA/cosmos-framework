# Adapter deployment continuation

This continues `feat/posttrain-family-adapters` from clean `afbd4ab`. Production
commit `263dac1` adds the explicit inference profile
`qwen3_vl_bf16_residual_fp32_v1`, its transactional export binding and two fixes to
the optional video-prefix cache. Training mathematics, optimizer policy, portable
adapters, full checkpoints and their execution identities are unchanged.
Follow-up `217cc5f` rejects incomplete tagged deployments after independent review
demonstrated that HF could silently initialize an omitted projection. Both
production commits were built from clean archives and independently reviewed.

The previous [extension record](adapter-extension-progress.md) retains its source
revisions and results: **12 named checkpoints, 48 fixed profiles, 47 fully passing
profiles**, five additional mixed-component execution profiles and three BF16
deployment profiles. This continuation adds one qualified BF16 deployment profile;
it adds no checkpoint, training profile, training run or HPO trial.

## Requirement-to-code-and-test status

| Requirement | Code / evidence | Current status |
| --- | --- | --- |
| Ordinary LoRA BF16 deployment | `inference/reasoner/precision.py::{deployment_profile,install_deployment_profile}` and `CosmosFrameworkRuntime._load`; actual installed export, four saved probe batches and paired WTS evaluation | Implemented and validated through the explicit Framework profile below. Export for default stock LoRA BF16 execution remains refused/unqualified |
| Artifact identity and precision | `posttrain/checkpoint/tools/adapters.py::export_merged_adapter`, existing materializer transaction, `conversion/hf.py::{config_with_export_dtype,verify_export_dtype}` | Config and transaction bind the same execution profile; all 625 serialized tensors exactly equal BF16 rounding of the existing FP32 merge |
| Incomplete deployment rejection | `CosmosFrameworkRuntime._load`; installed real tied-export fixtures with keys removed/added in both shard and index; independent reproduction and fresh real GPU reload | Validated. Tagged loading checks HF missing/unexpected/mismatched/error diagnostics before assigning the model or installing its profile. Untagged loading is unchanged |
| Ownership, ties and serialization | `test_adapter_deployment.py`, existing native export tests; actual installed model before/after controlled generation | Parameter IDs, keys, values and tying are preserved; no adapter branches or trained-state mutation |
| Cached multimodal inference | `CosmosFrameworkRuntime._generate_with_video_prefix_cache`; GPU text/image/video fixture and real saved WTS probes | Ordinary cached/recomputed generation and first/reused video prefixes pass; processor modality IDs and consumed media are handled correctly |
| Existing training, independent settings, warm start, resume, LoRA/rsLoRA/DoRA/PiSSA | Previous five mixed-profile runs remain attached to `f6baa78`/`2d12fd5`; affected installed native SFT/export tests rerun here | Existing bounded qualification preserved; no training result relabeled as a new run |
| MoE PiSSA stock FP32 merge | Four new dense-GEMM FP64 accumulation probes, on both preserved stock FP32 artifacts, with all eight original inputs | **Incomplete**. All four fail, bringing the tested failed candidates to eleven. Portable support and the merge rejection are preserved |
| Adapter-disabled/default exports | Untagged loader regression tests, existing materializer/SFT tests and strict default metadata compatibility | Existing default behavior preserved; new arithmetic is activated only by a matching explicit profile |

Paths above are beneath `cosmos_framework/`. The
[compact evidence](assets/adapter-deployment-continuation.json) records source,
wheel, artifact, data and report hashes; the full reports are in the local review
package. Counts from overlapping test suites are not summed.

## Exact qualified profile

The checkpoint is `Qwen/Qwen3-VL-2B-Instruct`, revision
`89644892e4d85e24eaac8bacfd4f463576704203`. The existing WTS-trained ordinary LoRA
artifact comes from `076cfbb2b92994a9939cce4211f27fe3c39e79c1` and was not retrained.
It has rank 8, alpha 16, ordinary `alpha/r` scaling, dropout zero and FP32 adapter
storage: 56 language Q/V projections, eight final/deepstack merger projections and
vision block 23 QKV. LR multipliers were 1, .5 and .1 respectively; there are no
extra trained modules. The original base/processor/template/media identities
remain in the export marker.

The measured deployment uses one A100 80 GB, Python 3.12, Torch `2.10.0+cu128`,
Transformers `5.8.1`, SDPA and eager module execution. All stored parameters are
BF16. Dense linear GEMMs use BF16 inputs/weights with FP32 results; embedding
outputs, normalization, attention and residual arithmetic retain FP32. The vision
patch convolution computes in FP32 from BF16-stored weights. The final LM head
and public logits remain BF16. This changes execution, not merged weight values.

The profile keeps ordinary stock model classes and parameters, but **plain
`AutoModelForImageTextToText.from_pretrained` alone does not install it**. Use the
Framework loader. A manual stock-load/install integration must also check
`output_loading_info=True` diagnostics: installing the profile alone cannot detect
a missing tensor that HF has already initialized.
Training, autocast, CPU/offload, multiple inference devices and other attention
backends are refused. Other checkpoints, methods under this new profile, devices
and library versions are not additional qualifications. The three previously
qualified rsLoRA/DoRA/PiSSA BF16 artifacts keep their default stock execution.

The config field `cosmos_inference_profile` and
`.materialized.identity.export_metadata.deployment_precision.execution_profile`
must agree. Missing, conflicting or unknown declarations fail before model
loading. Nonempty weight-loading diagnostics fail before profile installation;
an otherwise intact tied embedding/head export loads without a special exception.
Profile settings participate in the existing materializer identity,
staging, verification and idempotent-reuse checks. FP32 remains the default export;
the unchanged FP32 merge contract is separate from this approximate deployment.

## Measured acceptance

All original BF16 gates were preserved, including maximum confident-decision
regret .02, maximum logit error 2, the distribution/argmax bounds, paired WTS
quality limits and resource budgets. Every saved probe was used. Compared with
the independent saved BF16 portable reference, maximum logit error is **2.0**,
mean TV **.003418**, regret **0**; compared with the FP32 merged reference the
values are **.350277**, **.001795** and **0**. The installed Framework outputs are
bitwise equal to the successful diagnostic candidate's saved outputs.

| Measurement | FP32 reference | BF16 residual profile |
| --- | ---: | ---: |
| Floating weight storage | 7.926 GiB | 3.963 GiB |
| Controlled peak allocated CUDA memory | 8.172 GiB | 4.261 GiB |
| Median controlled inference time, 3 warm-ups / 20 repetitions | .3670 s | .3852 s |
| WTS correct / 576 | 528 | 528 |

Memory ratio is **.5214** and latency ratio **1.0496**, within the predeclared .75
and 1.1 limits. This is a memory reduction with about 5% higher controlled latency,
not a serving-throughput speedup. KV tensors are FP32; prefix-cache capacity has
its own memory cost. Cached-generation peak allocation was also recorded.

The WTS rows use the same corrected four-frame preprocessing, 50,176–81,920
pixels/frame, TorchCodec CPU decoding, SDPA, batch one, greedy 32-token decoding,
seed 42, IDs and gold answers as the paired FP32 reference. One answer changes
from wrong to right and one from right to wrong; there are no invalid answers.
Accuracy delta is **0 percentage points**, with a paired-row 95% bootstrap interval
**[−.520833, .520833]** (10,000 resamples, seed 42). This is the **reused
scene-disjoint benchmark**, not an untouched final test or evidence from multiple
independent scenes/training seeds. No new accuracy-improvement claim is made.

The real cache comparison covers all four saved batches. Cached versus recomputed
maximum logit differences are .1875–.21875; first/reused video-prefix differences
are at most .25, with identical generated IDs and zero decision regret. The
unchanged distribution gates also pass, so this is more than a token-only check.
The tiny GPU fixture separately covers text, image, video, tied parameters and
serialization. Physical-model task quality remains outside these correctness
claims; no physical experiment was repeated.

## Remaining MoE constraint

The additional candidates compute either every formerly adapted linear or every
plain linear GEMM in FP64 and return FP32, using both the original rank-2r stock
artifact and the earlier FP64-accumulated/FP32-rounded stock artifact. Neither
artifact retains adapter branches. Stored weights, router/top-k semantics and
the independent portable FP32 oracle are unchanged.

| Candidate | Maximum / mean final-logit error |
| --- | --- |
| Original weights, adapted linears accumulated in FP64 | .0295105 / .000388774 |
| Original weights, all plain linears accumulated in FP64 | .0294962 / .000388829 |
| FP64-accumulated weights, adapted linears accumulated in FP64 | .0294943 / .000388682 |
| FP64-accumulated weights, all plain linears accumulated in FP64 | .0294819 / .000388725 |

All fail the existing FP32 gate (`atol=5e-4`, `rtol=2e-4`, mean ≤ `5e-5`) despite
64/64 greedy agreement. Tracing the preserved failing probe again finds layer
37/token 7 replacing expert 5 with 68; the portable margin is `7.45e-9`, and the
downstream block error is about .097. Output gates cover all eight probes. Along
with the earlier seven candidates, this supports arithmetic sensitivity at a
hard routing boundary, not an observed PiSSA residual/conversion defect. It does
not prove every imaginable implementation must fail.

No production numerical correction has satisfied this checkpoint's contract.
The guard stays. A new training execution that consistently uses dense effective
weights would require new training/qualification; an approximate routing/task
contract would require separately agreed criteria. Neither would retroactively
pass the preserved oracle. Portable PiSSA remains supported; stock merge remains
an incomplete acceptance criterion.

## Review, provenance and reproduction

The clean `263dac1a9269a154ebd3ab6669c830647e261eeb` wheel has SHA-256
`a5adbfc3c2f041cbc7a855231eb64ab7d2155f55b0cb4c729f8465873845b6e1`; all 1,886
package files were verified against its source archive. The production export,
actual Framework reload, resource measurement, cache checks and WTS evaluation
use that installed revision. Earlier arithmetic diagnostics retain the `2d12fd5`
wheel identity; they are not relabeled. New training was unnecessary because
training mathematics/state semantics did not change.

Affected source and installed CPU suites at `263dac1` each pass **176 tests**, with
one CUDA-only skip. The installed CUDA fixture passes separately. Independent
review rechecked two prefix-cache fixes: named RoPE arguments with processor
modality IDs, and omission of pixels already represented in a cached prefix.
Its later incomplete-weight reproduction found that HF could accept a projection
removed from both shard and index. `217cc5f` fixes that separate finding;
its targeted source suite passes 11 tests with one CUDA-only skip. The independent
CPU graph probe is explicitly not CUDA arithmetic qualification. Initial fixture
and pytest-invocation errors, and every failed numerical candidate, remain in
the evidence directory.

The final production wheel at
`217cc5f5a92c2d8d6b62ef9727ab9576739640cd` has SHA-256
`408ff6a7b532770b52f9ed8738857f551dcc00db5006f7c9e1bf7977cd1bef13`; all 1,886
package files match its clean source archive. Its installed targeted CPU suite
also passes 11 tests with one CUDA-only skip. A fresh process through the actual
Framework loader reproduces **every saved controlled output bitwise** (all four
batches / 32 positions) from `263dac1`. The only production delta is checking load
diagnostics. No training, arithmetic, weights, preprocessing, decoding or evaluation
code changed, so WTS was not repeated or relabeled to the final loader revision.
Independent follow-up verifies intact tied loading, rejection before installation
for missing and unexpected weights, and unchanged untagged loading. There are no
unresolved review findings in these two production commits.

```bash
"$SFT_PYTHON" -m cosmos_framework.posttrain.checkpoint.tools.adapters \
  --adapter /exports/ordinary-lora --base "$SFT_MODEL_PATH" \
  --merged --output-dtype bfloat16 \
  --execution-profile qwen3_vl_bf16_residual_fp32_v1 \
  --out /exports/lora-bf16-residual
```

```python
from cosmos_framework.inference.reasoner.runtime import CosmosFrameworkRuntime

runtime = CosmosFrameworkRuntime(
    "/exports/lora-bf16-residual", dtype="auto", attn_implementation="sdpa"
)
# Use the existing runtime generation/evaluation API. The export selects its profile.
```

Exact scripts, commands, environment manifests, failed probes, paired predictions
and local patches are retained in
`/localhome/local-rarunachalam/cosmos-adapter-continuation-20261104`.
The previous sealed review package is unchanged. Four-bit training, HPO and
additional model families remain outside scope. PR #342 stays closed; nothing
is pushed or published.
