# Corrected canonical BF16 and fast-prefill deployment

Production revision `f831b992b7d2506e6613778a3d02af58f9c9a15c` adds a clean
dependency route, repairs the expanded-input cache inconsistency under a new
profile, and integrates the previously measured fast candidate. Both public
portable and merged loaders pass the corrected numerical contract. The fast
profile also passes every existing latency/memory gate, including 8K prefill
below two seconds. No model training, HPO or four-bit work was repeated.

Qualification is limited to **Qwen/Qwen3-VL-2B-Instruct**, immutable snapshot
`89644892e4d85e24eaac8bacfd4f463576704203`, and the final full-WTS update-1111
`merged_weight_bf16_v1` portable artifact. Its mixed recipe contains 56 language
q/v LoRA targets (rank 8, alpha 12), eight merger/deepstack rsDoRA targets
(rank 4, alpha 9), vision block 23 qkv rsPiSSA (rank 3, alpha 6), and separately
trained final language normalization. Dropout is zero. Structural acceptance of
other checkpoints/settings is not their numerical or performance qualification.

## Requirement-to-code-and-test matrix

| Requirement | Production path / evidence | Result |
| --- | --- | --- |
| Resolver-consistent installation | `pyproject.toml`, `packages/reasoner-runtime/`; paired wheels, empty environments, `pip check`, fresh public generation | Implemented and validated. Native/train retain Transformers 4; the separate dependency-only distribution pins 5.8.1. |
| Diagnose original cache failure | Forced common-history public portable/merged traces at `df88223`; pre-public `dc11a03` reproduction | Completed. Shape-dependent FP32 RMSNorm reduction is the first differing operation, before the third decision. |
| Correct BF16 cache computation | `utils/generator/bf16_cache.py::_rms_norm_forward`, `bf16_cache_kernels.py::_rms_norm_kernel` | Implemented and validated as `qwen3_vl_bf16_canonical_cache_v2`; original v1 unchanged. |
| Fast public inference | `bf16_prefill_kernels.py`, `bf16_cache.py::_tensorcore_attention_forward`, `portable_adapters.py`, `inference/reasoner/precision.py`, public exporter | Implemented and validated as `qwen3_vl_bf16_tensorcore_prefill_v1`, through both public loaders. |
| Identity, ownership and cache lifecycle | Existing strict portable state loading/derived-cache invalidation; config/transaction profile binding; new live attention signature and hardware/device preflight | Passed installed tests, parameter identity/version/trainability checks, prefix immutability and repeated-use memory checks. |
| Numerical populations and horizons | 108 historical cases + 16 separately identified additions; four profile/loader executions; 12 forced 32-token and two long-context cases per execution | All pass. Controlled outputs are bitwise equal across corrected reference, fast, portable and merged. Counts are qualification executions, not new training profiles. |
| Deployment performance | Existing absolute/relative latency and memory matrix; first use measured separately | Fast portable and merged pass all gates. Corrected reference still fails four long-prefill gates. |
| Task drift | Public evaluator, same 576 questions and this checkpoint's own FP32 reference | Both fast loaders: 528/576; reference: 527/576; one changed answer; task-drift gates pass. Reused benchmark only. |
| Independent review | Dependency, public export, live backend, legacy hook, device placement and kernel provenance review | Four confirmed production findings fixed; no open confirmed findings. |

## Installation and public use

Follow the paired-wheel commands in
[adapter-public-inference.md](adapter-public-inference.md). From the checkout,
use uv **0.11.29**. The mandatory shared wheel no longer selects Transformers;
`[native]`, `[train]` and default `uv sync` retain `>=4.57.1,<5.0.0`.
`cosmos-framework-reasoner-runtime==1.2.2` owns only distribution metadata and
selects the pinned inference stack. It is not an extra attempting to override an
incompatible mandatory dependency. Build/install both wheels from the same
revision: older framework wheels also named 1.2.2 retain the conflict.

The declared route was resolved into an empty CPython 3.12.13 environment with
no `--no-deps` or system-site borrowing. `pip check` passes. Versions include
Torch 2.10.0+cu128, Transformers 5.8.1, Triton 3.6.0, Torchvision 0.25.0+cu128,
TorchCodec 0.10.0 (hash-pinned CPU wheel), AV 18.0.0, Pillow 13.0.0,
qwen-vl-utils 0.0.14, Accelerate 1.15.0, Diffusers 0.39.0 and Safetensors 0.8.0.
CUDA drivers and FFmpeg shared libraries are host prerequisites.

Two clean resolver runs have separate identities. The documented checkout
route used uv 0.11.29/root constraints and resolved NumPy 2.2.6; both original
public CLI workflows and 15 media/preprocessing tests pass there. The numerical,
task and resource runs used a clean installation resolved from `/tmp` with uv
0.12.21, NumPy 2.5.4 and installed uv 0.13.7. Those are the only distribution
version differences; full freezes are preserved. Both environments use the same
final wheels and critical model/media versions. Their measurements are not
relabelled as one environment.

A separate native installation resolves Transformers 4.57.6 and passes actual
stock public generation. The packaging-only wheel has the same 1,918 framework
source files as `df88223`. A targeted generic entry-point suite has 19 passes,
one skip and six inapplicable 5.x-only synthetic model/processor/profile failures
under 4.x; all 26 pass in the pinned 5.8.1 environment. This is not a broad native
training/backend qualification. The native root lock remains valid.

Stage the base and unchanged portable artifact, edit their paths in
[portable_adapter_fast.toml](../../examples/reasoner/portable_adapter_fast.toml),
then run:

```bash
CUDA_VISIBLE_DEVICES=<A100-UUID> python -m cosmos_framework.inference.reasoner.generate \
  --config examples/reasoner/portable_adapter_fast.toml --output fast-answer.json
```

The same `[model.portable_adapter]` table works in the existing evaluator TOML.
For a merged artifact, export explicitly with
`--output-dtype bfloat16 --execution-profile qwen3_vl_bf16_tensorcore_prefill_v1`,
point `model.model_name` at the completed export, and omit `portable_adapter`.
The public Cosmos loader validates/activates the profile declared in
`config.json` and `.materialized`. Plain Transformers loading does **not**
activate this arithmetic and is not the supported BF16 workflow. All 625 tensors
in each new export equal the earlier qualified BF16 artifact exactly; profile
metadata changes, not trained weights. The FP32 stock export remains a separate
precision reference.

Support is single A100 SM80, batch one, BF16 weights/activations/KV, pinned
versions, SDPA-configured stock dense Qwen3-VL, eager `eval()` and
`no_grad()`/`inference_mode()`, and ordinary non-offloaded `DynamicCache`.
Autograd, autocast, compilation, CUDA graphs, static/custom caches, offload,
alternate device maps and changing root/text/vision attention after installation
are rejected. Short/decode attention uses the reference implementation. No
whole-model FP32 transient execution is substituted. Portable effective-weight
caching retains the existing explicit clearing and mutation/version rules.

## Root cause and execution semantics

The fixed `video-961109` fixture remains additional index 8 / combined index 40,
with its original frame selection, preprocessing and token IDs. Its containing
file SHA256 is
`01dfdde5738a10ecaa9b73a271d775ef536469244653a50ea488136a837b645c`.
Both public loaders reproduce the failure, as does the pre-public `dc11a03`
merged implementation. Forced history `[3862,525,220,19]` separates the original
error from later token divergence: the third-position maximum logit error is
0.1875; the fourth is 0.125 under the same history. The historical 16.875 fourth
position arose after different generated histories and is not the initial error.

The first differing tensor is the output of
`model.language_model.layers.14.post_attention_layernorm`: identical input,
nine different BF16 values, maximum difference 0.0078125. Up to that point,
attention masks, multimodal positions/RoPE, vision/deepstack recomputation, Q/K/V
and attention outputs match. K/V differences begin downstream at layer 15.
For the same captured row, PyTorch FP32 variance is 7.089112281799316 at one row
and 7.089113235473633 at longer shapes; the independent FP64 value is
7.089112836845743. The scale difference crosses a BF16 rounding boundary.

Version 2 uses one fixed-row Triton RMSNorm program, FP32 variance/scaling,
BF16 rounding before affine multiplication, and the effective BF16 norm weight.
It preserves FP32 ownership of the separately trained portable normalization.
The captured diagnostic trace becomes bitwise equal through all four decisions.
The regression fixture contains only that small row and norm weight. Existing
v1 linear/attention kernels are unchanged.

The fast profile adds the preserved candidate2 FP64 tensor-core attention when
query length is at least 1024 and head dimension is 128, with fixed 16-query /
64-key tiles and FP64 online softmax. Other admitted shapes and decoding use
reference attention. Its reduction semantics have their own version name;
neither legacy profile is silently reinterpreted. No further broad kernel search
was performed.

## Numerical and task results

The four executions separately retain the original 24, 32 and independent 32
populations, 16 older distributed-training probes, four matched-training probes,
and the fixed additional 16: **124 cases per profile/loader**, not 496 unique
inputs. Each covers full forward, cached/full decoding and forced common-history
logits. Video cases exercise three prefix reuses and ownership. Each execution
also covers 12 fixed forced 32-token cases (including the failure) and 2K/8K
contexts. Strict comparison remains `atol=5e-4, rtol=2e-4`, mean absolute logit
error at most `5e-5`, and equal greedy positions. Existing cache-distribution
gates and the 16 MiB prefix-growth limit are unchanged. All pass; compared
reference/fast and portable/merged outputs are bitwise equal on these cases.

The 576-row benchmark is **reused and scene-disjoint**, previously used in
validation/HPO, not an untouched final test. Both fast loaders score 528/576
(91.667%) versus 527/576 for this artifact's own FP32 reference. They have
identical answers; one reference answer changes from wrong to right. Paired
accuracy difference is +0.174 percentage points, row-bootstrap 95% interval
[0, 0.521] points. The unchanged task-drift gates (point >= -1 point, lower
interval >= -2 points) pass. This small reused-benchmark difference is not a
general accuracy-improvement claim.

Preprocessing, four CPU-decoded frames, rows and greedy seed-42 decoding match
the reference. The evaluator's effective answer-letter cap is 10 tokens despite
the historical TOML request of 32; separate numerical probes actually force 32.
Attested worker time including loading/evaluation is 271.21 s merged / 288.47 s
portable, peak CUDA allocation 4,336,186,368 / 5,014,534,656 bytes. Physical-model
correctness and historical training/validation losses retain their original
identities; neither is inferred from this task run.

## Deployment costs and unchanged performance gates

Warm synchronized A100-SXM4-80GB medians, seconds. All four new executions use
the same unshared GPU UUID, fresh processes and separate empty Triton disk caches.
Other-GPU work overlapped some runs; host CPU/I/O were shared. Historical stock
controls remain attributed to their original revisions.

| Case | Corrected merged | Corrected portable | Fast merged | Fast portable |
| --- | ---: | ---: | ---: | ---: |
| Short text/image/video triplet | .1900 | .1987 | .1905 | .1961 |
| 2K prefill | .3403 | .3406 | .1699 | .1706 |
| 8K prefill | 4.2490 | 4.2467 | 1.7014 | 1.7005 |
| Decode after 2K | .0361 | .0372 | .0354 | .0375 |
| Decode after 8K | .0423 | .0427 | .0424 | .0426 |
| Four-token reused video prefix | .1550 | .1681 | .1549 | .1668 |

Fast meets the entire existing matrix: 2K <= .35 s and <= 4x stock BF16;
8K <= 2 s, <= 8x stock BF16 and <= .75x stock FP32; short <= 1.1x old canonical
and <= 1.25x stock; decode <= .075 s and prefix <= .5 s, both <= 1.1x old
canonical / <= 2x stock. Fast 8K is about 7.02x stock BF16, .558x stock FP32
and 2.50x faster than corrected reference. Portable short/8K overhead is within
1.25x/1.1x merged. Reference v2 still fails the 2K stock ratio and all three
8K latency/ratio gates; it remains the slower correctness reference.

| Resource, bytes | Fast merged | Fast portable |
| --- | ---: | ---: |
| Parameter storage | 4,255,064,064 | 4,262,555,648 |
| Separate derived-weight cache | 0 | 559,939,584 |
| 2K KV | 234,881,024 | 234,881,024 |
| 8K KV | 939,524,096 | 939,524,096 |
| 8K peak allocated | 5,670,855,168 | 6,238,335,488 |
| Whole-process NVML peak (10 ms samples) | 6,981,419,008 | 7,558,135,808 |

Merged weights equal stock BF16 bytes and stay below .51x stock FP32; KV bytes
equal stock. Allocated peaks pass <= 1.1x historical reference, portable peaks
<= 1.2x corresponding merged, derived cache <= .75 GiB. NVML samples cover
construction, first/warm short calls, long prefill, decode and prefix reuse;
this broader scope is not directly comparable to older prefill-only NVML peaks.

First use is separate: fast load 1.673 s merged / 11.552 s portable (including
artifact validation); first short triplet 3.435 / 3.984 s with cold compilation;
first 2K 1.885 / 1.909 s; first 8K 1.711 / 1.713 s after 2K compilation. Warm
statistics use 20 short, five prefill, 20 decode and ten prefix repetitions.

## Validation provenance, review and remaining boundaries

Final framework wheel SHA256:
`dd787ba2b95718dfab07c731fbaec74df249145145f00a3146921e4dcbf31945`.
Dependency-only wheel SHA256:
`b395273c70e310cb00263896352112827d969c293d4b34d3b5317fc8afbc650b`.
All 1,921 framework files match clean archived source. Export creation and 51
focused tests used `d94b032`; numerical sweeps used `2e678ad`; performance,
task runs, final public CLI checks and affected guards used `f831b99`. Independent
review verifies that intervening changes only correct legacy guard scope and
pre-allocation host/version/device validation: kernels and admitted-path
arithmetic are unchanged. Historical reports are not relabelled.

Installed validation covers **80 unique tests** across these revisions (51
focused, 26 existing entry-point and three additional compatibility/preflight
cases), plus repeated affected checks. Independent source review ran 20 CPU
tests with 28 GPU skips; it is separate evidence. Confirmed findings fixed the
public exporter allowlist, vision-only attention mutation, shared Qwen3.5 hook
regression, and mismatched device-map preflight. The initial AV 18.1 artifact
refusal led to the exact AV 18.0 pin. Original logs, intermediate failures and
the accidental unsupported-display-GPU launch are preserved.

Legacy `qwen3_vl_bf16_canonical_cache_v1` retains its failure and slow-prefill
limits. Existing `qwen3_vl_factorized_fp64_branch_v1` remains separately
qualified and unchanged; original `factorized_v1` strict merge failure remains.
This continuation adds no Qwen3.5, MoE, physical-model, other-device or serving
backend qualification. Historical 12 named checkpoints / 48 fixed training
profiles / 47 fully passing legacy profiles are unchanged.

The local evidence and independently reviewable patches are in
`/localhome/local-rarunachalam/cosmos-adapter-cache-production-20261109/review-package`.
`REPRODUCE.md`, `validation-provenance.json`, `artifact-catalog.json`, per-row
task CSVs, immutable input catalogs and independent review accompany the patch.
Large weights, raw traces, runtimes and datasets stay at their catalogued paths.
PR #342 remains closed; nothing was pushed or published.
