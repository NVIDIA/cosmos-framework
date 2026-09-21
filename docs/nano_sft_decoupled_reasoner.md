# Decoupling the frozen Reasoner from Nano generator SFT

Status: Phase 1 and the Phase 2 offline extraction MVP are implemented; an
eight-H20 short-run FSDP/EMA A/B is complete, while full-corpus and
compile-enabled production validation remain pending

Target recipe: `vision_sft_nano` and later Nano multiview SFT variants

## Current implementation status

Implemented in this branch:

- a provider-neutral per-layer Reasoner K/V contract, exact packed-sample
  isolation, inline capture, and static generator-only replay;
- an UND-only extraction primitive, `extract_reasoner_feature_batch`, for a
  supplied set of finalized `ReasonerFeatureRequest` objects;
- structural `prune_und_pathway_` before materialization/FSDP, including the
  regular and EMA model construction paths;
- a generator-only VFM text-layout path that does not require `embed_tokens`;
- an immutable, layer-major safetensors cache writer, manifest, checksums,
  per-record fingerprints, and `OfflineReasonerFeatureProvider`;
- a bounded-memory, resumable rank-local shard writer plus shared-POSIX
  distributed finalizer for production cache extraction;
- a finite deterministic SFT document enumerator that reuses the training
  caption/window framing and expands every reachable caption/CFG variant;
- a strict Reasoner-only DCP loader for either `net.language_model.*` or
  `net_ema.language_model.*`, without constructing Generator, VAE, or EMA;
- a shared-POSIX `torchrun` extraction CLI with independent rank-local workers,
  bounded file-based coordination, shared CFG-null deduplication, and atomic
  publication;
- strict external-backend fingerprints plus cache/model layer, KV-head, and
  head-dimension validation before FSDP materialization;
- offline-provider submission before noising and synchronized failure handling
  before the FSDP forward; and
- CPU contract/storage tests plus a tiny H20 CUDA test comparing all generator
  outputs and gradients between the full cached model and the structurally
  pruned model.

Not implemented yet:

- remote and read-through clients/services (the backend names and `endpoint`
  field are reserved, but selecting either backend currently fails explicitly);
- layerwise H2D staging (`layerwise_h2d=true` is rejected);
- automatic content-digest derivation for Reasoner/tokenizer/dataset artifacts;
  the CLI currently requires the three pinned fingerprints explicitly; and
- a real full-corpus extraction plus a compile-enabled, representative-duration
  Cosmos3-Nano memory and throughput benchmark. A short eager-mode eight-H20 A/B
  on the official eight-video sample is complete.

## Decision

Use a single external-conditioning boundary at the per-layer Reasoner
cross-attention K/V tensors. The configuration contract names these backends:

1. `joint` preserves the original one-pass dual-pathway forward and is the default.
2. `inline` captures Reasoner K/V locally, then runs a second cached GEN-only pass;
   it is the numerical-reference path.
3. `offline` reads precomputed immutable Reasoner K/V shards and is implemented.
4. `remote` will obtain the same tensors asynchronously from dedicated Reasoner workers.
5. `read_through` will check the offline cache first and send misses to remote workers.

Only `joint`, `inline`, and `offline` are executable today. `remote` and
`read_through` intentionally raise `NotImplementedError` during provider creation.

For a fixed SFT corpus, `offline` should be the default. It removes the Reasoner from
every training rank, is deterministic, and turns Reasoner work into a one-time dataset
preparation cost. `remote` is useful for very large corpora, changing prompts, or
augmentations that make an exhaustive cache too large. The hybrid backend is the
long-term operational recommendation, but it should be built only after the offline
path establishes numerical parity.

Do **not** cache QKV or every layer's hidden state. The generator computes Q from its
own trainable, noised tokens. It only consumes the frozen Reasoner's K and V at each
layer. For Nano's grouped-query attention, Q is also four times wider than K or V, so
caching QKV would triple the storage of the minimal K/V boundary.

## Why this is the correct boundary

Nano uses a dual-pathway MoT decoder. At every layer:

- UND/Reasoner tokens use `q_proj`, `k_proj`, `v_proj`, `o_proj`, and `mlp`.
- GEN tokens use the corresponding `*_moe_gen` modules.
- GEN queries attend to both UND K/V and live GEN K/V.
- UND attention is causal and independent of GEN tokens, so all UND K/V can be
  produced without running the generator.

The current `vision_sft_nano` optimizer selects only `moe_gen`, `time_embedder`,
`vae2llm`, and `llm2vae`. The optimizer factory marks every parameter outside that
allowlist as `requires_grad=False`. The Reasoner is therefore frozen, but the current
joint forward still constructs, shards, all-gathers, and executes it.

There is already a close precedent in `inference_text_kv_memory.py`: it stores
RoPE-applied UND K/V per layer, reports `is_gen_only()` after all layers are populated,
and then executes only the GEN pathway. The implemented training contract generalizes
that pattern without introducing a second attention abstraction.

### Canonical tensor contract

For each causal text segment and each decoder layer `l`, persist:

```text
cross_k[l]: [S_und, num_kv_heads, head_dim]
cross_v[l]: [S_und, num_kv_heads, head_dim]
```

`cross_k` is the exact key seen by GEN cross-attention: after the Reasoner K projection,
text K normalization, any generator-facing UND K normalization, and RoPE. `cross_v` is
the V projection output (V does not receive RoPE). For Nano, `cross_k` is the same K
used by UND self-attention. The name deliberately does not promise that equality for
other tiers; a future model can materialize a generator-specific normalized K.

The batch-level feature object also carries the per-sample/per-caption offsets needed
to isolate packed samples. The existing `PackedSequence` remains the source of truth
for GEN layout, attention masks, view IDs, and current noisy/conditioning tokens.
The persistent representation is canonical and unpadded, with all K/V heads present;
batch padding and any context-parallel sharding are runtime concerns and must never be
written into a cache shard.

The target feature identity must cover:

- the ordered, fully framed token IDs, including EOS and start-of-generation;
- exact mRoPE position IDs and causal-document boundaries;
- Reasoner checkpoint content hash and model/config hash;
- tokenizer and special-token hash;
- K/V dtype or quantization recipe; and
- cache schema and producer-code versions.

Today, `compute_reasoner_feature_fingerprint` hashes the exact token IDs, position IDs,
causal offsets, cache schema, and the configured Reasoner/tokenizer/framing identity.
The manifest separately pins dtype and tensor geometry. Until an explicit producer
revision field is added, deployments should include it in one of the three configured
identity strings. A mismatch is an error, never a warning or an implicit fallback.

## Size and memory model

Cosmos3-Nano has 36 layers, 8 K/V heads, and head dimension 128. In BF16:

```text
bytes per UND token
  = layers * (K + V) * kv_heads * head_dim * bytes_per_element
  = 36 * 2 * 8 * 128 * 2
  = 147,456 bytes
  = 144 KiB
```

| Framed UND tokens | BF16 K/V per example |
|---:|---:|
| 256 | 36 MiB |
| 512 | 72 MiB |
| 1,024 | 144 MiB |
| 1,790 | 251.7 MiB |
| 2,048 | 288 MiB |

The local full BridgeData manifest has 1,222 examples. With the recipe's real Qwen
tokenization, it has 1,082 framed tokens on average (p50 1,004, p95 1,586, maximum
1,797), which gives about **181.6 GiB** of BF16 K/V. This is a reasonable offline-cache
MVP. At one million 1,024-token examples, however, the cache is about **137 TiB**, so a
remote or read-through backend becomes attractive.

The Nano language model contains approximately:

| Component | Parameters | BF16 logical size |
|---|---:|---:|
| GEN layer pathway | 6.946B | 12.94 GiB |
| UND layer pathway | 6.946B | 12.94 GiB |
| UND embeddings + LM head + final norm | 1.245B | 2.32 GiB |
| Total removable Reasoner | 8.191B | 15.26 GiB |

A real Nano meta-model audit measured 15,136,811,008 parameters in the full
language model and 6,946,075,648 after pruning: 8,190,735,360 parameters
(54.111%) removed. The 397 remaining parameter tensors are exactly the original
`moe_gen` FQNs, and the pruned model completed meta initialization, activation
checkpoint wrapping, and block/root FSDP2 wrapping without a missing attribute.

The standard recipe stores FSDP master parameters in FP32 and enables a second FP32
EMA network. With eight-way FSDP, removing the 8.191B Reasoner parameters from both
networks saves about **7.63 GiB of resident parameter shards per GPU**, before counting
smaller layer all-gathers, activations, and allocator fragmentation. The short
eight-H20 run measured an average peak-allocated reduction of 15.630 GiB/GPU and an
average peak-reserved reduction of 19.419 GiB/GPU; representative full-corpus and
compile-enabled measurements remain required.

## Common provider API

The provider-neutral request and result live in
`cosmos_framework/model/generator/reasoner_features.py`:

```python
@dataclass(frozen=True)
class ReasonerFeatureRequest:
    sample_key: str
    token_ids: torch.Tensor
    position_ids: torch.Tensor
    causal_offsets: torch.Tensor
    fingerprint: str

@dataclass(frozen=True)
class ReasonerFeatureBatch:
    cross_k: tuple[torch.Tensor, ...]
    cross_v: tuple[torch.Tensor, ...]
    causal_offsets: torch.Tensor
    fingerprints: tuple[str, ...] = ()

class ReasonerFeatureProvider(Protocol):
    def submit(
        self, requests: Sequence[ReasonerFeatureRequest]
    ) -> Future[ReasonerFeatureBatch]: ...
```

`submit` is a future-shaped contract for every provider. The current offline provider
performs its local read synchronously and returns an already-completed future. The
training path submits after final sequence packing and before noising, then resolves
the future before entering the FSDP forward. A future asynchronous provider can use
the same seam to overlap provider work with noising or other preparation.

`extract_reasoner_feature_batch(causal_lm, requests)` is the current UND-only
reference extractor. The immutable cache APIs are in
`cosmos_framework/model/generator/reasoner_feature_cache.py`:

- `ReasonerFeatureCacheIdentity(reasoner, tokenizer, framing)`;
- `build_reasoner_feature_requests(packed_sequence, sample_keys, identity)`, the
  shared training/extraction framing boundary;
- `build_reasoner_feature_request_from_text_tokens(...)`, which uses
  `PackedSequenceBuilder.pack_text_tokens` for the supported single-caption
  offline producer path;
- `ReasonerFeatureCacheEntry(sample_key, features)`;
- `write_reasoner_feature_cache(cache_root, entries, identity=...)`; and
- `IncrementalReasonerFeatureCacheWriter(...).append/flush/finalize()` plus
  `finalize_incremental_reasoner_feature_cache(...)` for distributed extraction; and
- `OfflineReasonerFeatureProvider(cache_root, expected_identity=...,
  expected_dtype=..., strict_fingerprint=True)`.

The incremental writer keeps only one target-sized shard payload in CPU memory. Each
rank writes uniquely named safetensors into a hidden sibling staging directory. A
shard is committed by publishing its checksum-bearing JSON sidecar only after the
safetensors file has been fsynced and atomically renamed. Reopening the same rank with
`IncrementalReasonerFeatureCacheWriter.resume(...)` validates those commits and skips
already-seen `(sample_key, fingerprint)` pairs. Extraction loops can call
`contains(sample_key, fingerprint)` before running the Reasoner and `append(...)`
also returns `False` for a repeated pair. `finalize()` writes a rank-complete marker
binding all sidecar checksums.

After every rank is complete, one coordinator calls
`finalize_incremental_reasoner_feature_cache(cache_root, identity=..., world_size=...)`.
The finalizer requires all expected rank markers, validates cache identity, tensor
geometry, dtype, checksums, and cross-rank duplicate identities, then atomically
publishes the ordinary immutable `manifest.json` layout consumed by
`OfflineReasonerFeatureProvider`. The current implementation assumes a shared POSIX
filesystem. The low-level writer deliberately does not own dataset enumeration,
process launch, or a remote/object-store writer; the implemented extraction CLI
supplies enumeration and process coordination for a shared-POSIX deployment. Staging
data is retained after publication so finalization is idempotent and can be audited
or retried.

The actual offline TOML configuration is a single flat table:

```toml
[model.reasoner_conditioning]
backend = "offline" # joint | inline | offline | remote | read_through
cache_root = "/path/to/reasoner-kv-cache"
reasoner_fingerprint = "<pinned-reasoner-checkpoint-digest>"
tokenizer_fingerprint = "<pinned-tokenizer-and-special-token-digest>"
framing_fingerprint = "<pinned-prompt-framing-schema-digest>"
strict_fingerprint = true
request_timeout_s = 300.0
```

External backends deliberately require `strict_fingerprint = true`. A sample
key is only a diagnostic alias: the content fingerprint over exact token IDs,
positions, offsets, and cache identity is the authoritative lookup key. This
prevents a same-key, same-length caption variant from silently reusing the
wrong K/V record.

With `strict_fingerprint=true` (the default), all three fingerprint fields are
required for an external backend. `request_timeout_s` bounds the
`Future.result()` wait; the current offline provider performs its read inside
`submit()`, so this deadline becomes operationally useful once submission is truly
asynchronous. The schema also currently accepts
`prefetch_batches`, `endpoint`, and `layerwise_h2d`; prefetch-depth scheduling is
not wired yet, `endpoint` is reserved for `remote`/`read_through`, and
`layerwise_h2d` must remain `false`.

External modes initially require:

- frozen UND weights and `predict_text_tokens=False`;
- Qwen3-VL-8B Nano dense layers;
- the base `OmniMoTModel` (`OmniMoTCausalModel` fails closed until its
  AR/teacher-forcing memory dispatcher can compose with external K/V);
- `joint_attn_implementation="two_way"`;
- no context parallelism; and
- an initialized generator checkpoint (fresh initialization cannot copy GEN weights
  from an UND tower that is absent).

Multiview attention, context parallelism, teacher forcing, and other model tiers should
be enabled only after their exact cached-attention parity tests exist.

## Generator-only structural model

Skipping UND execution is not enough: the user's goal requires that training ranks do
not materialize UND parameters at all.

The implemented external-backend path instantiates the existing model on the meta
device, calls `prune_und_pathway_` while the removed modules still occupy no storage,
and only then applies compile/FSDP and materializes the model. Every GEN module keeps
its existing fully-qualified name. The pruning operation removes:

- `embed_tokens`, `lm_head`, and the UND final norm;
- the optional Reasoner-side `visual` encoder when configured;
- per-layer UND Q/K/V/O projections and Q/K norms;
- per-layer UND MLP and UND pre-attention/post-attention norms.

It keeps `rotary_emb`, every `*_moe_gen` module, and VFM-level encoders/decoders. The
VFM input construction allocates the packed hidden-state buffer without calling
`embed_tokens`; the text rows are layout placeholders while GEN rows are filled by
`vae2llm`, the timestep embedder, and other live modality projectors.

The transformer loop now has an explicit generator-only branch:

1. `StaticReasonerKVMemoryState` is initialized from a `ReasonerFeatureBatch`.
2. `is_gen_only()` is true before layer 0.
3. Each decoder layer reads its external UND K/V and executes only GEN norms,
   projections, attention, MLP, and residuals.
4. The final output applies only `norm_moe_gen`; it never touches an UND tensor or
   parameter.

Both regular and EMA networks go through the same `build_net` path and therefore have
the identical pruned structure. This prevents the frozen Reasoner from being
materialized, FSDP-sharded, all-gathered, or duplicated in EMA for external backends.
The tiny CUDA parity test additionally asserts that the pruned state dict exposes only
GEN pathway parameters and that all generator gradients match the full cached model.

### Checkpoint behavior

Implemented startup behavior skips pretrained Reasoner loading for an external
backend. If `load_weights_from_pretrained=true`, an external run must supply a resume
or warm-start checkpoint because the deleted UND tower cannot seed the GEN weights;
explicit random GEN initialization remains possible by disabling that option.

For a model-only warm start whose DCP configuration skips the complete `net_ema.*`
subtree, the callback explicitly reports that fact and initializes the pruned EMA
model from the newly loaded regular generator. If EMA was loaded in full, it is
preserved. A partial EMA skip is rejected because it would mix restored and random
EMA leaves. Same-job resume always preserves its restored EMA trajectory. A
non-strict warm start with EMA and no complete EMA skip is also rejected because it
cannot prove whether EMA was loaded or left randomly initialized.

The shipped full Nano DCP was audited against the pruned vision-SFT target: all
405 target tensors (397 language-model GEN tensors plus 8 VFM tensors) exist with
matching shapes, and strict DCP subset loading succeeds while ignoring source-only
UND tensors. This validates the model-only warm-start shape contract; a distributed
generator-only save/optimizer/resume integration test is still pending.

Still to validate before production use:

- full-checkpoint to generator-only warm start for both DCP and safetensors;
- generator-only save/resume while preserving immutable cache identity;
- reconstruction of a full inference model from the generator checkpoint and pinned
  Reasoner; and
- mismatched generator-checkpoint/cache provenance rejection on resume.

## Offline backend

### Extraction

The in-process `extract_reasoner_feature_batch` primitive is implemented and runs
each finalized causal document independently under `torch.inference_mode()`. It uses
the existing `ReasonerKVCache` to return normalized, RoPE-applied K/V. It currently
rejects architectures that require a second generator-specific UND K normalization.

A resumable distributed CLI is implemented. Its current interface is:

```shell
torchrun --nproc-per-node=8 -m cosmos_framework.scripts.extract_reasoner_features \
  --sft-toml examples/toml/sft_config/vision_sft_nano.toml \
  --checkpoint /shared/checkpoints/iter_000000100 \
  --checkpoint-source regular \
  --output /shared/caches/nano-sft-reasoner-kv \
  --reasoner-fingerprint <reasoner-subtree-content-digest> \
  --tokenizer-fingerprint <tokenizer-content-digest> \
  --framing-fingerprint <dataset-and-framing-content-digest>
```

Each rank constructs only the Reasoner (`include_gen_pathway=false`,
`include_visual=false`), loads the explicitly selected regular or EMA subtree from a
local DCP, and processes a deterministic video-level metadata slice. The finite
enumerator probes and decodes each assigned video through the same FFmpeg geometry as
training, then expands all retained windows, reachable positive-weight captions, and
CFG variants. It never enters the infinite/random `SFTDataset.__iter__` path. Positive
FPS noise and random fixed-length frame selection create unbounded or non-canonical
variants and therefore fail closed.

The shipped Nano DCP keeps the Generator and optional visual encoder below the same
language-model root. The Reasoner-only loader requires every one of the 399 text
Reasoner target leaves, while safely omitting the 397 source-only Generator leaves
and 351 source-only visual leaves. Any other source extra remains an error. Every
worker performs a complete independent `no_dist=True` DCP read; the extraction CLI
does not initialize a process group or issue GPU collectives.

The CLI frames each tokenizer output through the same
`PackedSequenceBuilder.pack_text_tokens` boundary as training, including EOS,
start-of-generation, and float mRoPE positions when FPS modulation is active. It
checks `writer.contains(...)` before Reasoner execution, reports construction/load/
extraction time and peak allocated/reserved GPU memory per rank, and lets rank zero
publish only after every rank's current-run stats and completion marker exist. Rank
zero polls shared-POSIX status files with a bounded timeout, then invokes the strict
finalizer to validate every marker, sidecar, checksum, identity, tensor signature,
and cross-rank duplicate before publication. Worker exceptions are recorded and
allowed to escape so `torchrun` can terminate peers immediately; there is no
long-running collective in which a failed rank can strand the others.

`--max-documents-per-rank` is a resumable smoke mode: it flushes committed shards but
never writes a rank-complete marker or publishes `manifest.json`, and its output is
therefore not valid for training. A later unlimited run resumes those shards. The
first version intentionally runs the single-document reference extractor;
token-budget dynamic batching is pending. One live extraction job must own a given
output root; concurrent jobs targeting the same root are not supported. Single-node
`torchrun` derives a per-launch coordination ID automatically; multi-node launches
must pass the same unique `--coordination-run-id` on every node. The CLI appends
TorchElastic's restart count so a restarted attempt ignores stale failure/status
records while still resuming durable shards.

The unit of caching should be a causal document, not necessarily an entire video:

- normal SFT has one document per sample;
- per-view caption SFT has independent documents that can be concatenated using the
  recorded offsets; and
- unconditional CFG uses one shared empty-caption entry instead of duplicating it for
  every sample.

### On-disk format

The implemented writer does not create one filesystem object per sample. It groups
records into immutable safetensors shards plus a JSON manifest. The low-level writer
defaults to a 4 GiB target; the extraction CLI uses 512 MiB per rank by default to
bound aggregate host memory on an eight-rank job. The current safetensors keys are:

```text
cross_k.000: [total_tokens_in_shard, H_kv, D]
cross_v.000: [total_tokens_in_shard, H_kv, D]
...
cross_k.035
cross_v.035
record_offsets: [num_records + 1]
```

The cache root must not already exist. The writer builds a sibling temporary
directory, writes every shard and the manifest, and publishes the complete root with
one atomic rename. The manifest records cache identity, tensor geometry, dtype,
per-record `(sample_key, fingerprint, token range)`, and a SHA-256 per shard. The
offline provider validates the manifest eagerly, verifies each shard checksum before
its first read, and slices only requested records with `safe_open`. The extraction
CLI additionally performs a streaming existence/checksum/metadata/shape audit of all
shards on rank zero when it encounters an already-published output.

Layer-major storage supports future double-buffered, layerwise H2D staging. The
current implementation stages the whole requested `ReasonerFeatureBatch`; even a
1,024-token example is 144 MiB, far smaller than the removed Reasoner weights.

For multiview/per-view text, each record must additionally preserve caption offsets,
stable camera/view IDs, and the shared mRoPE temporal origin. Per-view captions are
independent causal documents, while the GEN temporal origin follows the longest
caption rather than the sum of all caption lengths. The existing inference cache does
not encode this geometry, so it must not be advertised as multiview-training support.

Start with BF16 for parity. FP8/INT8 K/V with per-layer or per-head scales can halve
storage and bandwidth, but is an explicit approximation mode with its own quality and
gradient-parity gate.

### Dataset nondeterminism

Cache the exact text feature, not only a video UUID. For the current SFT dataset:

- the random window index is already represented by `uuid_w<window>`;
- all possible finite windows must be enumerated during extraction;
- CFG dropout selects either the normal feature or the single null-caption feature;
- random T2V/I2V/V2V conditioning does not change Reasoner text K/V; and
- any stochastic caption/FPS/resolution transform that changes token IDs must either
  be frozen, represented in the cache key, or handled by `remote/read_through`.

## Remote backend

This section is a design target. No remote transport, client, service, or
read-through write path is implemented in the current branch.

Run Reasoner workers as a separate service allocation, not as ranks in the generator's
FSDP/DDP process group. Mixing service ranks into the training world size would make
generator collectives hang and would waste Reasoner ranks in every all-reduce.

Recommended data flow:

```text
generator rank -- submit(token IDs, positions, fingerprint) --> request queue
       |                                                    reasoner replica
       +-- VAE encode + noise + pack (overlap)                    |
       |<----------- per-layer K/V or cache URI -----------------+
       +-- validate fingerprint --> GEN-only forward/backward
```

Operational requirements:

- replicate the 8B Reasoner one per service GPU; prefer replica/data parallelism over
  tensor parallelism because Nano fits on one modern accelerator;
- dynamic-batch requests by total UND tokens, not request count;
- use immutable request IDs, bounded queues, backpressure, deadlines, and idempotent
  retries;
- keep a memory and/or NVMe content-addressed cache in front of Reasoner execution;
- expose queue time, Reasoner compute time, serialization time, bytes sent, cache-hit
  rate, and client wait time; and
- fail closed on version mismatches. An unavailable service may fall back to an exact
  disk hit, but not silently to a different Reasoner or quantization.

For a 1,024-token prompt the BF16 response is 144 MiB. At `R` examples/s, required
payload bandwidth is approximately `144 * R MiB/s`, before transport framing. This is
usually modest relative to long-video generator step time, but it must be measured at
the intended number of training ranks. Start with a simple streaming RPC into pinned
CPU buffers plus asynchronous H2D copies. Add CUDA IPC/NVLink for same-node workers or
CUDA-aware UCX/RDMA for cross-node workers only if profiling shows transport on the
critical path.

When a node has exactly eight GPUs, reserving GPU 0 for the service means the generator
job must be launched as a seven-rank job (with FSDP degree adjusted accordingly). A
cleaner production topology is a separate Reasoner pool shared by several eight-GPU
generator nodes.

## Code ownership and status

Keep the feature contract in training/model code; a training job must not import Ray or
other heavyweight inference-only dependencies.

| Area | Current status |
|---|---|
| `configs/base/defaults/model_config.py` | Implemented typed runtime configuration and compatibility validation. |
| `configs/toml_config/sft_config.py` | Implemented the flat `[model.reasoner_conditioning]` TOML schema. |
| `model/generator/mot/unified_mot.py` | Implemented optional UND construction, GEN-only execution, and `prune_und_pathway_`. |
| `model/generator/mot/cosmos3_vfm_network.py` | Implemented generator-only packed layout without text embedding. |
| `model/generator/omni_mot_model.py` | Implemented inline/offline lifecycle, pre-noise submission, synchronized resolution, structural pruning, and startup guards. Remote/read-through remain pending. |
| `model/generator/reasoner_features.py` | Implemented request/result types, UND-only extraction, inline/static memory, and exact two-way external-K/V attention. |
| `model/generator/reasoner_feature_cache.py` | Implemented immutable cache identity, fingerprints, bounded/resumable sharded writers, distributed finalization, manifest validation, and offline provider. |
| `model/generator/mot/multiview_attention.py` | Pending cache-aware per-view/maskless/Flex support. |
| `data/generator/local_datasets/sft_reasoner_documents.py` | Implemented finite deterministic standard-SFT document enumeration with training framing parity. |
| `data/generator/` | Centralized multi-batch prefetch/cache-handle plumbing remains pending. |
| `checkpoint/reasoner_only.py` | Implemented strict regular/EMA Reasoner-only local-DCP loading with independent reads and safe Generator/visual source-leaf omission; generator resume/composition validation remains pending. |
| `scripts/extract_reasoner_features.py` | Implemented resumable distributed shared-POSIX cache extraction and atomic publication without NCCL collectives. |
| `examples/toml/sft_config/` | Pending opt-in Nano cached-Reasoner recipe after full-scale parity passes. |

An optional remote server can live under the inference/serving tree, but it should
implement the neutral wire schema without making the training client depend on that
server implementation.

## Implementation sequence

### Phase 0: golden reference and instrumentation — partial

- Implemented a fixed-seed tiny H20 parity harness and provider/capture wait timers.
- Implemented overall step/VAE timing and per-rank peak allocated/reserved memory on
  a short real-Nano eager run.
- Still pending comprehensive UND/GEN/H2D/backward/optimizer/EMA timing on a
  representative compile-enabled run.

### Phase 1: common boundary and inline replay — implemented

- `ReasonerFeatureRequest`, `ReasonerFeatureBatch`, and `ReasonerFeatureProvider`
  define the common contract.
- `CapturingReasonerKVMemoryState` detaches exact per-layer UND K/V, while
  `StaticReasonerKVMemoryState` replays them with live GEN Q/K/V and autograd.
- Single- and multi-sample packing, fingerprint/offset validation, and unsupported
  CP/attention modes fail closed.
- A tiny CUDA test validates GEN output and every generator gradient across the full
  cached and structurally pruned executions.

### Phase 2: generator-only model and offline cache — MVP implemented

Completed:

- meta-device UND pruning for regular and EMA models;
- generator-only VFM/model execution;
- immutable manifest plus sharded safetensors writer/reader;
- bounded-memory incremental shards, resume markers, and distributed finalization;
- deterministic SFT document enumeration and exact training text framing;
- strict Reasoner-only regular/EMA DCP loading;
- the eight-rank-capable extraction CLI with per-rank timings/memory statistics;
- an eight-H20 end-to-end validation on the official eight-video BridgeData sample
  (nine records including the shared CFG-null prompt, 8,594 tokens, eight shards,
  atomic publication, and successful eager revalidation);
- an eight-H20 FSDP training A/B with full activation checkpointing, FP32 EMA,
  and gradient accumulation two: steps 5--10 used identical token traces and
  measured 57.887 seconds/step for joint versus 48.015 seconds/step for offline,
  while average peak allocated memory fell from 52.128 GiB/GPU to
  36.498 GiB/GPU;
- strict cache identity, per-record fingerprints, and shard checksums; and
- offline provider integration into the SFT step.

Still pending:

- a real full-corpus eight-GPU extraction run and optimizer-step numerical
  cache-vs-joint audit;
- automatic fingerprint derivation and token-budget dynamic batching;
- DCP/safetensors full-checkpoint warm-start and generator-only resume tests;
- an opt-in `vision_sft_nano_reasoner_cache` example recipe/launcher; and
- a representative-duration, compile-enabled full-size Nano benchmark.

In the small eight-GPU validation, every rank independently loaded the 8.19B BF16
Reasoner in 10.45--10.91 seconds. Rank-local extraction took 1.15--2.77 seconds and
peak allocated memory was 15.55--15.76 GiB (maximum reserved 15.77 GiB). This proves
the multi-worker coordination/publication path, but is not a substitute for the
pending full-corpus throughput and storage benchmark.

The short training A/B used the full Nano model, FSDP, full activation
checkpointing, FP32 EMA, and eight H20 GPUs, but deliberately disabled compilation
and repeatedly sampled the eight-video cache. Across the six paired steady steps,
offline reduced mean step time by 17.05%, increased aggregate token throughput by
20.56%, and reduced average peak allocated memory by 15.630 GiB/GPU. This validates
the expected direction and magnitude, but the hot 1.18-GiB cache is not a proxy for
full-corpus shared-filesystem behavior.

### Phase 3: remote/read-through provider — pending

1. Reuse the implemented request/result and fingerprint schema.
2. Add remote client/service transport and asynchronous prefetch.
3. Benchmark one service GPU against increasing generator-rank counts and scale
   replicas based on measured queueing, not a fixed assumed ratio.
4. Add persistent read-through writes only after cache-key and atomicity tests pass.

### Phase 4: broaden support — pending

Add separately gated parity suites for multiview masks and per-view captions, context
parallelism, causal/teacher-forcing training, MoE tiers, quantized K/V, and layerwise
H2D staging.

## Validation gates

Validated in the current implementation:

- canonical K/V shape/dtype/device/offset validation and multi-sample isolation;
- inline capture followed by static GEN-only replay;
- tiny dense Qwen output and all-generator-gradient parity between the full cached
  model and `prune_und_pathway_` model on one H20;
- absence of UND parameters from the structurally pruned state dict; and
- immutable-cache round trips, batching across shards, content-addressed prompt
  reuse, cache misses, stale fingerprints, malformed manifests, and corrupt shard
  checksums.

Remaining correctness gates:

- one optimizer step produces equivalent selected weights;
- normal prompt, CFG-null prompt, maximum-length prompt, and per-view captions;
- full checkpoint to generator-only warm start, generator-only resume, and full-model
  inference composition; and
- end-to-end stale cache/config provenance is rejected during distributed startup
  and resume, before any FSDP forward.

Distributed gates:

- eight-GPU DP/FSDP with activation checkpointing and EMA is validated in eager
  mode; compile-enabled validation remains pending;
- no Reasoner parameter appears in either regular or EMA model state on training ranks;
- no collective divergence when a cache/service request fails; and
- deterministic sample-to-feature association across resume and dataloader workers.

Performance report:

- peak allocated and reserved GPU memory for every rank;
- wall time split into data, VAE, provider wait, H2D, GEN forward, backward, optimizer,
  and EMA;
- cache bytes/read bandwidth or service queue/compute/transport percentiles; and
- throughput against the unmodified joint baseline.

The first production promotion should require zero cache misses, no silent fallbacks,
successful checkpoint resume, and a measured reduction in both peak memory and step
time on the real Nano SFT workload.
