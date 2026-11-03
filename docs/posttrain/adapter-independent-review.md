# Independent review of the adapter contribution

Review requested at clean `feat/posttrain-family-adapters` commit
`e2db8167fef08b186c838be86b45da4afb7d8240`, relative to audited main
`86e1ee52f276c11d6cc66dfd17da2795dfbe937b`. The existing package contained
qualification evidence but no independent review covering the last production
correction. Three separate reviewers therefore inspected targets/mathematics,
lifecycle/integration, and artifacts/export, with fresh contexts and read-only
repository access. The implementation owner made the corrections below; the
lifecycle and artifact reviewers independently rechecked them.

The [qualification baseline](adapter-final-review.md) remains the authoritative
requirement matrix and exact checkpoint/component/precision/distributed table:
**12 named checkpoints, 48 fixed profiles, 47 fully passing profiles**. Repeated
executions and the checks below do not add profiles. Native
Qwen3-VL-30B-A3B PiSSA portable support remains available; its merged-export
acceptance criterion remains **incomplete** and public merged export is refused.
No numerical threshold, routing operation, method formula or artifact schema was
changed in this review.

## Confirmed findings and corrections

| Finding at `e2db816` | Reachable failure and correction | Regression and independent recheck |
| --- | --- | --- |
| L1, P2: disconnected Alpamayo normalization extras | `model/alpamayo/policy.py::validate_targets` admitted final-prefix `self_attn.q_norm` and `post_attention_layernorm`. They entered the optimizer but received no action-loss gradient because the action expert consumes only cached K/V. Commit `b21afa4` refuses them before injection, including wildcard selections, while retaining final-layer `input_layernorm`/`self_attn.k_norm` and live earlier-layer norms. | Both negative regressions failed before the fix. `test_alpamayo_adapters.py::test_r1_rejects_dead_final_prefix_norms_before_injection` and eight `test_r1_live_prefix_norm_updates_and_optimizer_membership` cases pass. The latter use the actual tiny R1 graph in training mode, optimizer membership checks and three updates, with/without vision AC. Independent graph probes verify the dead and live paths. |
| A1, P2: portable payload/provenance could diverge after external replacement | `checkpoint/adapters.py::read_adapter_artifact` hashed a path and then reopened it to deserialize. An external replacement could supply different tensors under the original digest. HF export also reopened the manifest, allowing the first adapter's weights to be labeled with another same-configuration adapter's digest. Commit `9f90524` adds `read_adapter_snapshot`: hash and deserialize the same captured bytes, retain that record through `checkpoint/tools/adapters.py::export_merged_adapter`. Existing reader API and schema stay compatible. | Both regressions failed before the fix. `test_adapters.py::test_portable_read_deserializes_the_verified_payload` and `test_adapters_native.py::test_merged_export_metadata_binds_the_consumed_adapter_snapshot` pass. Independent replacement probes now return the verified state and publish its exact digest. |

L1 affects an advertised optional extra-module configuration, not the four fixed
Alpamayo profiles: their saved manifests have zero extra normalization parameters.
The normal recipe's K/V/vision/merger/action targets are unchanged. The trainer's
generic finite-gradient check is not a substitute for validating graph connectivity.

A1 requires external replacement/copy of an artifact during reading; the existing
commit-last, no-replace publication helper does not itself overwrite artifacts.
The physical exporters inherit the corrected byte reader on their final,
identity-checked load; their merged markers do not claim the earlier payload's
weight digest, so their format and reconstruction code need no change for this
finding. Reading bytes uses temporary memory proportional to the portable
payload. The largest fixed-profile tensor payload is 45,289,248 bytes (Alpamayo
PiSSA); base weights are not copied by this reader.

## Other reviewed requirements

| Area | Independent conclusion and evidence |
| --- | --- |
| Component resolution, ties, optimizer membership | No additional confirmed defect. Review covered per-selector matches, component/type/layer restrictions, conflicting overlaps, shared module/parameter identities, alias setting agreement and deduplicated final optimizer membership. Twenty-four independent CPU cases exercise math/ownership/extra-alias order; these are fixture coverage. |
| LoRA/rsLoRA/DoRA | Existing native LoRA scaling is reused. DoRA magnitudes participate in optimization, state and pure merge. The magnitude denominator's merged-weight helper already runs under `no_grad`; no missing-detach correction is needed. Independent forward/backward references cover both scaling modes, dropout and bias. Zero first-step A gradients remain permitted. |
| PiSSA | Original W0 stays intact; computation and merge use W0 + sBA − sB0A0. Initial factors are mandatory immutable artifact/checkpoint state, not recomputed at resume or warm start. No additional residual/decomposition defect was found. The known MoE hard-routing sensitivity is not reclassified as a pass. |
| Initialization, warm start and resume | Loaded/materialized/tied base precedes initialization; adapters precede AC/FSDP registration and optimizer construction uses final parameters. Portable warm start starts a new optimizer/data clock; exact checkpoint resume restores state without decomposition or missing-key tolerance. Three independent native lifecycle tests pass, including adapter-disabled hook order. |
| Portable/merged reload and live-state preservation | Eighteen independent normalization-alias permutations across LoRA/DoRA/PiSSA pass exact portable reload and tying checks. Missing magnitudes/initial factors fail. Pure merge leaves live tensors and CPU RNG unchanged; maximum tiny FP32 output error is 2.384e-7. Custom tied-target stock-HF merge remains refused. |
| Existing abstractions and adapter-disabled behavior | No new loop, generic causal-LM wrapper, optimizer implementation or checkpoint transaction was added. The new target guard is adapter-only; existing reader API callers retain their return contract. Affected suites include adapter-disabled ordering and existing export/model integrations. Historical distributed/training validation is retained at its actual revisions. |

## Validation and provenance

The final production/test revision is
`9f9052483da96adc1c9ac1033f14b6d7ab4b8cfc`. Its wheel SHA-256 is
`5a7ab6d1cb41d71c926dbc7a68eba0b67b9875e9e5adf9eede8870dfc27b47ee`.
All 1,880 packaged files equal its clean Git archive and the installed copies in
two isolated runtimes. Later commits change only review documentation/evidence.
The previous sealed package and runtimes remain intact.

| Affected validation | Result and scope |
| --- | --- |
| Source adapter/export suites | 99 passed. Covers portable state, ownership, native integration, DoRA/PiSSA, Qwen3.5 and the unchanged MoE refusal. |
| Source physical suites | 60 passed, one CUDA-only skip. This initial run preceded an explicit `net.train()` addition to the new positive normalization test; final installed results below cover that last test edit. |
| Final installed adapter/export suites | 99 passed in the pinned HF runtime: Torch 2.10.0+cu128, Transformers 5.8.1, Safetensors 0.8.0. |
| Final installed physical suites | 60 passed, one CUDA-only skip, in the pinned physical runtime with Transformers 4.57.6. Includes the final training-mode normalization tests and unchanged Pi05/GR00T suites. |
| Independent small checks | Separate reports preserve math/ownership, alias/artifact and lifecycle probes, original failures and post-fix rechecks. Counts overlap repository tests and are not summed. |

Fresh-process GPU results and their actual source/wheel/environment/input hashes
are recorded in [installed validation](assets/independent-installed-validation.json)
and the local review package. Four new exports and all eight reload checks pass:

| Method | Portable maximum error | New FP32 merged maximum / mean error | Greedy positions equal |
| --- | ---: | ---: | ---: |
| LoRA | 0 | 8.965e-5 / 6.913e-6 | 32/32 |
| rsLoRA | 0 | 7.629e-5 / 6.797e-6 | 32/32 |
| DoRA | 0 | 7.534e-5 / 6.926e-6 | 32/32 |
| PiSSA | 0 | 1.097e-4 / 6.835e-6 | 32/32 |

They reuse the four completed host WTS checkpoints; no new training, evaluation
accuracy experiment or profile coverage is implied. Existing full training,
resume, distributed and downstream evaluation results remain mapped to their
original source, wheel/container, base and data identities in the baseline assets.

Portable comparisons retain recorded training precision. Merged comparisons use
FP32 portable versus FP32 merged with the already-rounded training-base values.
They preserve `atol=5e-4`, `rtol=2e-4`, mean error at most `5e-5` and greedy
agreement; cross-precision BF16-to-FP32 drift remains a separate measurement.
Portable reload retains its tighter original bounds. No greedy-only pass is used.

## Remaining limits and readiness

The two confirmed review findings are corrected and re-reviewed. Within the
documented scope, the patch is ready for maintainer review. This is not a claim
that every original acceptance criterion is complete: native MoE PiSSA merged
export is still unsupported and fails its unchanged numerical contract.

The 576-row WTS subset remains a **reused scene-disjoint benchmark**, not an
untouched final test. Paired unadapted/adapted results from corrected preprocessing
are unchanged. Physical-model runs establish training/artifact correctness on
small real fixtures, not downstream robotics/driving performance or safety.
Untested checkpoints/backends/distributed modes, four-bit training, additional
families and HPO are outside this review. No broad retraining was performed.

The local review package contains the cumulative framework patch, the incremental
review fixes, independent reviewer reports and probes, reproduction drivers,
installed/test/GPU provenance and checksums. PR #342 remains closed. Nothing is
pushed, reopened or published.
