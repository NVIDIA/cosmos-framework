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

## Current validation

The following checks use the branch rebased onto main `00498c7`, with the
cache-lifecycle fix at `8a7f178`. They are separate from the older checks below.

- **Transfer output preservation:** before the lifecycle fix, `26d0c48` with
  HF revision `79b47f7c5ba3dc66af97ef9de7973ac6959cdcfa` produced pixel-identical
  eager/compiled native videos for all 32 pairs: 16 depth inputs in each of
  full-history and sink-plus-recent configurations, 597 frames per video
  (19,104 compared decoded frames). Per-video DOVER scores also matched.
  These BF16 tests used compile plus block CUDA Graphs on GB200, 640x640,
  seed 42. The HF revision has the same weight hashes and model/tokenizer
  configurations as `55df9a79`; this is not evidence of a weight update.
- **Sequential requests after the lifecycle fix:** each compiled history
  configuration completed eight consecutive 597-frame requests in one
  process, using two inputs repeated four times. All 16 requests passed
  without forced garbage collection or a process restart. KV owners were
  released after each request, pooled ring addresses stayed stable, and
  repeated-input decoded FP32 hashes matched. Four saved videos matched the
  frozen eager references across all 2,388 decoded frames. The related CPU
  regression suite passed 455 tests, including 34 new lifecycle cases;
  modified-code Ruff and formatting checks passed.
- **Non-Transfer Nano regression:** main `00498c7` versus `8a7f178`, same GB200,
  PyTorch `2.13.0a0+9186a08b2c.nv26.07`, explicit evaluation mode, BF16, compile
  plus block CUDA Graphs, and diffusion caching off. HF revision
  `e59a53c25979a090fa8706c9acc0c254a6e89b92`, one text-to-video input, seed 42,
  35 steps, 33 frames at 832x480. After two warmups, all five generation-plus-
  decode measurements were retained: mean 4.451326 s on main versus
  4.444163 s on the branch (-0.16%; median change -0.37%). All 33 decoded
  frames and the MP4 bytes matched. Both runs recorded CUDA Graph activity
  without graph breaks; neither enabled `emulate_precision_casts` or emitted
  native RMSNorm calls. All 86 shared compiler source/tuning artifacts were
  unchanged. Loading, initial compilation, input preparation, and output
  encoding are excluded from these timings.

The Nano case does not exercise causal AR KV ordering. A Bimanual preflight
with its multi-frame AR chunk checkpoint was rejected by unchanged main's
compile/CUDA-Graph guard, so it supplies no new Bimanual regression result.
The checks do not establish all-model performance, arbitrary-workload
stability, segmentation mIoU, Persistence/control, or intermediate-tensor
equivalence to eager. The default Dynamo recompile limit remains unchanged.

## Historical validation

Framework GPU checks compare main `33a7341b` with the request-isolated working
revision on the same checkpoint, input, seed, runtime, and GB200 GPU. Both arms
explicitly use evaluation mode and batch one. Two warmups precede ten measured
`OmniInference.generate_batch` passes, including input preparation; loading,
initial compilation, output encoding,
and profiler overhead are excluded. Bimanual and Nano use compilation;
CUDA Graphs and approximate diffusion caching are disabled in these older
measurements.

| Case                 | Main (s) | Request-isolated revision (s) | Latency change |
| -------------------- | -------: | ----------------------------: | -------------: |
| Bimanual, 121 frames | 3.506552 |                      3.510271 |         +0.11% |
| Nano, 33 frames      | 4.139353 |                      4.132361 |         −0.17% |

A separate Transfer canary compares the prior broad PR `6d4a13e` with this
request-isolated revision using the same iteration-1250 weights: one AgiBot
source with blur/depth/edge/seg controls, batch one, 121 frames, seed 2025,
four-step SDE, full history, and compilation with block CUDA Graphs enabled.
All four native MP4 outputs are byte-identical between the two versions.
Both versions retain 16 native RMSNorm call sites in their generated code;
all 17 generated Python files match after normalizing comments and paths.
Neither log reports a recompile-limit, eager-fallback, or skipped-CUDA-Graph
warning. One warmup and two measured passes per control give mean request
times of 7.144191 and 7.155622 seconds (+0.16%), including control preparation,
generation, and decode, while excluding output encoding. This small canary
checks preservation of the prior PR's behavior, not equivalence to an eager
baseline or the quality of the full 96-video population.

Both canary arms use the same compatibility overlay for the older checkpoint:
17 type-metadata keys allow the current loader to fill schema defaults; all
original configuration values and the seven weight shards remain unchanged.

Bimanual and Nano contain no native RMSNorm calls in either arm's generated
compiler code. Sample arguments match except for output directories, and video
counts, frame counts, geometry, and FPS pass validation. Decoded pixels differ
in both cases. After normalizing comments and absolute paths, all 69 Bimanual
and 47 Nano generated Python code files match; 15 and 12 autotuning selections
respectively differ. Those differences are confounders, not proven causes of
pixel drift. These runtime checks do not establish task accuracy or performance
for every model. They measure Framework, not a separate upstream GPU run.

CPU coverage includes request routing, gradient preservation, exact KV ordering
at the attention boundary, CFG branches, streaming control/target/refresh
propagation, growing histories, and independent row resets. Each repository's
norm/compiler suite passed 30 tests plus nine subtests. Framework's selected
frontend/KV suite passed 155 tests with four skips, and its AR dispatch suite
passed 62 tests. The corresponding focused upstream frontend/KV selection
passed 28 tests and AR dispatch passed 62. These suites are reported separately;
the counts must not be added as distinct coverage. Both repositories passed
modified-file Ruff, formatting, and code pre-commit checks. A broader
upstream frontend run stopped after 85 passes on a radar error-message
assertion also reproduced on unchanged HEAD.

Focused Framework norm/compiler type checks report zero errors; its nine-file
cache/frontend check reports zero errors and one existing warning. In
the upstream codebase, the five-file core cache check reports zero errors and one existing
warning; norm/compiler checks retain the same 42 diagnostics before and after,
and the two-file package/frontend check retains 250 errors and one warning on
both revisions, with no new diagnostics. These scoped checks do not establish
a clean repository-wide type check.

```bash
python -m cosmos_framework.model.generator.utils.numerics_test
python -m pytest cosmos_framework/model/generator/utils/kv_cache_test.py -m CPU --num-gpus=0
```

## Why preservation is scoped to Transfer

A historical same-source diagnostic at Framework `6d4a13e` changed only RMSNorm
between the compiler-visible formula and an opaque native implementation. With
evaluation mode, batch one, compilation, CUDA Graphs off, and ten unprofiled
passes after two warmups, the broad implementation produced:

| Model                | Fused formula (s) | Native implementation (s) | Latency change |
| -------------------- | ----------------: | ------------------------: | -------------: |
| Bimanual, 121 frames |          3.557431 |                  5.168490 |        +45.29% |
| Nano, 33 frames      |          4.336410 |                  5.415985 |        +24.90% |

The native BF16 implementation used eight GPU kernels per normalization:
conversion, square, mean, epsilon addition, reciprocal square root, normalize,
conversion back, and weight multiplication. Bimanual issued 8,784 native calls
and Nano 10,368; total kernel counts rose from 55,557 to 123,633 and from 51,387
to 131,739 respectively. Attention, matrix-multiply, and convolution kernel
counts were unchanged, and their GPU durations remained close. Matrix multiply
includes attention projections as well as other linear layers, not only FFN.

Small FLOP counts do not capture the cost of lost fusion, extra dispatch,
launches, and intermediate memory traffic. Native RMSNorm accounted for about
0.313 s of GPU work in Bimanual and 1.266 s in Nano. Bimanual showed substantial
extra launch/dispatch overhead; Nano also added substantial GPU work. Neither
arm recorded graph breaks or timed-iteration recompilation. Profiler overhead
changes scheduling, so these trace durations cannot precisely divide the
unprofiled latency difference. This diagnostic explains the previous broad
implementation; it does not validate the current request-isolated revision.

## Accuracy limits

The October 4 PAIBench-C experiment used batch one, 24 sources × four controls,
iteration-1250 student EMA, and seed 2025. It has not been rerun on this revision
and does not establish current numerical equivalence or regression-free
quality. A paired 96-video seven-metric evaluation remains outstanding. Retain
per-video mIoU as well as means: the historical mean was 0.50% below baseline
and the worst clip 14.58% below. Gemini judging was not part of that report.
