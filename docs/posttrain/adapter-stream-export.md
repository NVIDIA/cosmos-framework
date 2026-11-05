# Adapter export memory and BF16 execution

This continuation starts at `338971dcf8f0b02281f81df2546c9318a9053d6b`.
The qualified versioned MoE PiSSA execution, physical 2×2 HSDP and full
Alpamayo replay remain unchanged. Historical qualification remains 12 named
checkpoints, 48 fixed profiles and 47 fully passing legacy profiles; the separate
new-mode results are recorded in [adapter-effective-execution.md](adapter-effective-execution.md).

## Requirement and evidence

| Requirement | Production path | Validation / status |
| --- | --- | --- |
| Bounded merged HF construction | `checkpoint/conversion/adapter_stream.py:AdapterTensorSource` | Validated: real 30B fresh export in a 16 GiB cgroup; 4.02 GiB process-tree RSS, 603.18 seconds; installed-package exact-tensor and recovery checks. |
| Preserve LoRA/rsLoRA/DoRA/PiSSA arithmetic | Existing `PosttrainAdapterLinear.merged_state_weight`, whole target tensors | Exact equality to the former full-state transformation for FP32 and BF16, including mixed components and trained norms. No new adapter math. |
| Inventory, expert layout and ties | Meta architecture target resolution; source shapes; existing alias rejection | Exact key coverage, original fused expert layout, tied head ownership; real stock-loader fixture tests pass. |
| Transaction, identity and reuse | Existing `materialize_hf.run_request` lock/staging/verification/publication | Interrupted shard write, completed serialization and marked staging recovery pass. Legacy completed-output reuse leaves its marker unchanged; legacy staging remains version-strict. |
| Source integrity and incomplete/corrupt export rejection | Held/hash-verified shard inodes; captured portable state; full tensor/processor verification | Missing/shape/wrong-base, in-place mutation, processor mutation and same-size output corruption tests pass. |
| Public fresh 30B export, every tensor, established input populations | New installed wheel and original qualified stock artifact | Passed: all 882 tensors bitwise equal, 64 established stock-loader inputs pass, new/legacy completed exports reuse without marker mutation; no retraining. |
| Broader ordinary-LoRA BF16 qualification | Existing deployment profile unchanged | New body-FP32/head-BF16 diagnostic fails old gates; no expanded support claim. |
| Mixed/Qwen3.5 legacy BF16 references | Existing references and gates unchanged | Preserved incompatible-oracle failures; no candidate search against an impossible intersection. |
| Prospective BF16 training/deployment semantics | Explicit `merged_weight_bf16_v1`, portable format v4, approved 2026-11-05 | Three real profiles pass training/resume and precision-matched stock exports; broad BF16 cache gates fail, so deployment qualification is incomplete. See [new execution contract](adapter-merged-weight-bf16.md). |

## Why the previous export used 329.6 GiB

The preserved `candidate-export.json` records 329.597 GiB process peak RSS,
1535.67 seconds and 115.748 GiB of FP32 weights. Code inspection identifies
the allocation sequence; these are not independent measured phase peaks:

1. `tools/adapters.export_merged_adapter` assembled every rounded base tensor
   and merged adapter weight into one FP32 state and serialized `model.pt`.
2. `conversion/hf.overlay_consolidated_onto_hf` mapped that state, loaded a
   full BF16 HF base, and retained a parameter dictionary through assignment.
3. Native fused experts were transposed into contiguous temporary HF tensors.
4. `save_pretrained` serialized the HF model; `rewrite_expert_shards` then
   loaded each output shard and rewrote expert tensors into their original
   checkpoint layout. The large output shards amplified temporary storage.

The new adapter source keeps portable state, metadata, one whole source tensor
and its conversion/merge workspace, plus at most a 256 MiB output tensor group.
An individual larger tensor gets its own shard. It writes native expert layout
directly, without `model.pt`, an instantiated HF model or an expert rewrite.
The bound therefore depends on the largest tensor and method workspace; it is
not a universal fixed-memory guarantee for arbitrary models.

Verification reopens mappings per tensor, including when comparing an older
completed export with large shards. It never retains all touched pages of a
complete old shard through the comparison. Config, processor, tokenizer,
template, numerical values and adapter semantic identity stay unchanged.

## Source identity and recovery

The original `model.pt` route remains `materialize_model_pt_to_hf/6`.
The new source is `materialize_adapter_shards_to_hf/1`: its source SHA hashes
a canonical descriptor containing the full adapter/base identity, captured
adapter-weights digest, output dtype and shard-size policy. `source_bytes`
is the descriptor length, not a fictitious `model.pt` size. The unchanged
materialization marker also records output tensor hashes and shard inventory.

Base shards are hashed and consumed through the same held file descriptors
under the output lock. Changes to held inodes or config/index/processor assets
refuse publication. The existing staging transaction publishes only after
exact key/value/dtype, sidecar, processor and config verification.

A completed v6 export can be reused only when its recorded semantic metadata,
base, dtype and sidecars match and every tensor matches the streaming source.
Its original marker and source digest are preserved. A v6 staging tree is not
reinterpreted as a new-format staging tree; protection of the sole complete
marked copy remains in force.

This change bounds the portable-adapter-to-HF materialization path. Extraction
from a full consolidated/DCP training checkpoint and physical model-owned
exporters retain their existing memory behavior.

## Precision limits

The lossless exporter retains the FP32 equivalence contract: comparison is
against the portable model promoted to FP32 after its training base was rounded
to the recorded dtype. This does not promise BF16-to-FP32 behavioral equality.

The new ordinary-LoRA diagnostic preserves BF16 stored weights and the BF16
head, but uses transient FP32 body projections to remove projection-input
rounding. It fails the original population's logit limit (2.125 > 2.0) and
expanded population's TV limit (0.1013375 > 0.1). It is diagnostic code only.
Previous failed global-FP32 and head-only candidates, and the mixed/Qwen3.5
incompatibility proofs, remain attached to their original revisions.

The new-artifact BF16 contract was approved before its implementation qualification.
Its real-model status is tracked separately; approval alone is not qualification. No
legacy artifact, gate, routing semantic or historical result has been changed.
Correctness, downstream WTS quality and resource measurements remain distinct.
The 576-row scene-disjoint benchmark is reused, not an untouched final test.

## Installed exporter provenance

Fresh export used `b57792030527f0f680a9f0055a0653b76abe1bcf` (wheel SHA256
`e08fd009e16ed277b71a5576274c39b01fe15e221a7efdb1402a9638ed228a8c`).
The reviewed path-collision fix and unchanged metadata-helper relocation are
`da387cc5fc3f6be49145852d2f0358ba8a989b35` (wheel
`e368be022d9434b805506e4db00af88a3593e6c25f5003b14de27bc6cce0411b`).
The latter passed 228 installed CPU tests, one CUDA deployment test, completed
export reuse, all-tensor comparison, real 30B stock reload and dense/Qwen3.5
fresh-export regressions. Source inputs, commands, timing and process/cgroup
samples are preserved in `cosmos-adapter-export-precision-20261105`.

The process-tree RSS measurement includes conversion and serialization; no
allocator worker was moved outside the measured process. Page-cache accounting
reached the 16 GiB cgroup limit and reclaimed cache; there was no OOM, swap or GPU
allocation. Temporary disk peaked at 124,350,502,165 bytes, essentially the
staged output, with no full `model.pt`. The 329.597 GiB / 1535.67-second historic
run had different concurrent activity, so timing is observational, not a paired
exclusive-host benchmark. New completed-export reuse took 353.36 seconds and
legacy reuse 281.75 seconds, both below 3.94 GiB process-tree RSS.

The final installed `df334fc` package also passed an actual process interruption
check: SIGTERM after the first public-export shard left an uncommitted staging
tree; the normal public CLI rejected its missing index, rebuilt it, and committed
the export. All 625 recovered Qwen3-VL tensors match the independently completed
reference exactly. The first comparator invocation correctly refused a stale
wheel attestation; only the comparison was rerun with the correct installed
revision. The completed recovery was not repeated. See local
`public-recovery/final-report.json` and preserved failed invocation logs.
