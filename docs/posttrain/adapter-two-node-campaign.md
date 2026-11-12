# Two-node full-data campaign

This is a new qualification campaign. The historical single-host adapter results remain attached to their original revisions. No local CPU mesh, source download, or infrastructure allocation is a two-host training pass.

The working inventory and machine-readable evidence live at `/localhome/local-rarunachalam/cosmos-adapter-all-family-2node-20261111`. PR #342 remains closed; this work is local.

## Finite training coverage

`SFTConfig.training_epochs` opts into finite, whole-corpus passes. `FiniteSFTCursor` uses the existing `IndexedCursor` order and dataset identity. Every real row contributes once per epoch; the final global optimizer batch has explicit zero-weight padding. Padding does not consume or wrap the source. Real and padded rows both carry the weight field, so mixed microbatches are well-defined. The number of updates is `epochs * ceil(rows / (DP world size * micro_batch_size * grad_accum))`; configuration must match it.

The causal objective accepts a globally padding-only accumulation chunk only in this mode. Its forward/backward graph still participates in collectives. The trainer rejects a whole optimizer update containing only padding. Positive denominator windows retain the existing TOKEN_SUM accumulation and global weighting. Router auxiliary loss must be disabled because that separate objective does not yet mask training padding.

Finite exact resume preflights complete rank-local RNG state, the existing trainer update counters, the complete data cursor and its relation to committed updates inside the checkpoint transaction, before model loading. This mode requires completed updates without skipped attempts; it does not claim mid-accumulation recovery. A fresh factory-owned cursor may be restored during preflight, but refusal leaves model, optimizer and global RNG untouched.

Scheduler construction identity is now saved and checked, including cosine cycle length and linear warmup length. Legacy checkpoints made by a bound scheduler without this identity must resume using their original runtime/configuration or use a model/adapter warm start. This is an explicit compatibility boundary, including adapter-disabled legacy resume. Fresh adapter-disabled behavior and unbound legacy scheduler behavior are preserved.

Flow collation failures converge across ranks before a collective-bearing model forward. This does not attempt to recover a worker that fails inside a distributed forward; scheduler-level failure and checkpoint recovery remain separate campaign gates.

## Validation so far

Focused CPU tests cover uneven sequence lengths, real/padded mixed batches, globally empty accumulated chunks, exact ID coverage, optimizer parameters and moments, and checkpoint pre-mutation refusal. An independent reviewer additionally exercised actual CPU FSDP2/HSDP meshes (one physical host, Gloo, no CUDA). The report and executable probes are in `review/distributed-change-review.{md,json}` in the campaign directory. These checks do not qualify any GPU, model binding, full dataset, or two-node layout.

## Physical data and accumulated updates

Flow SFT now permits gradient accumulation through the existing TOKEN_SUM reducer. Controlled tests fix each example's time/noise and compare the complete global masked-action objective, gradients, AdamW parameters and moments against two-rank accumulated execution with uneven masks and an incomplete tail. This does not assert that changing microbatch shape preserves a stochastic model's RNG consumption. Exact resume binds the unchanged batch/accumulation configuration and restores rank RNG.

The LIBERO converter streams a single episode across parquet shard boundaries rather than retaining the complete image-bearing dataset in RAM. The v2 preparation identity records source ordering and source-frame count; it rejects nonmonotonic episodes, discontinuous frames, duplicate selected IDs and incomplete source coverage. It writes one action window at a time. Source hashes and an end-of-preparation manifest remain mandatory; a partial output directory is not a usable dataset.

## Ordered HF VLM execution

`rl.adapters.model_execution=qwen_vl_hf_ordered_v1` binds the exact Transformers 5.8.1 Qwen2-VL, Qwen2.5-VL and Qwen3-VL classes, PyTorch 2.10.0, and SDPA. Image, video and language calls have a consistent order, with graph-connected dummy vision calls for missing modalities. This is an explicit new profile; it does not reinterpret the legacy default profile or perform the rounded-weight profile's normalization conversion. Factorized components and dense PiSSA components can be mixed under this profile.

The family SFT recipe selects this profile for Qwen2/2.5-VL, enables deterministic four-frame video sampling, and preserves Qwen2.5's processor-produced `second_per_grid_ts` through training and prefix reuse. Original public checkpoints use `visual.*` and `model.layers.*`; loading and bounded export map these names to the exact TF5 owners before resolving ties. Duplicate source aliases refuse rather than overwrite another tensor. Native TF5 RMSNorm types are admitted as separately trained language extras.

Portable artifacts use the existing v3 execution identity. Their merged exports use FP32 stock HF weights and SDPA, preserving the processor and temporal intervals; the public Cosmos merged loader validates that identity. Other deployment precision profiles are not implied. CPU tests and independent review cover image/video/text execution, activation checkpointing, mixed adapters plus a norm extra, tied embeddings, warm reload, bounded export/reuse, and temporal prefix reuse. Installed-package, real-checkpoint and two-physical-node qualification are separate campaign gates.

The exporter rejects BF16 output for the FP32-only `qwen_vl_hf_ordered_v1` and `qwen3_vl_hf_stock_v1` merged-loader contracts before materialization. Previously it could emit a BF16 artifact that the declared Framework loader then rejected. Use FP32 for those artifacts. A separately supported BF16 execution contract requires its own artifact identity; changing metadata does not convert existing training semantics. Untagged legacy BF16 candidates and the rounded v4 contract retain their existing admission rules and qualification limits.

## Native PhysicalAI observations

The Alpamayo recipe accepts `model.model_config.observation_profile=physicalai_four_camera_v1` for the pinned PhysicalAI driving corpus. It binds cameras `[0,1,2,6]` (cross-left, front-wide, cross-right, front-tele), four preceding frames per camera in camera-major order, actual frame timestamps, source intrinsic/extrinsic calibration, and 16 history/64 future poses in the vehicle frame at the fixed keyframe. The native model still owns acceleration/curvature conversion, normalization and flow loss. This profile does not add reasoning supervision. `front_frames_v1` remains the default and retains its original artifact identity.

Use `python -m cosmos_framework.posttrain.scripts.prepare_physicalai_flow_sft --source-manifest SOURCE.json --clip-ids IDS.json --cache-dir CACHE --out PREPARED` in the physical runtime with authorized Hugging Face access. The manifest pins the repository, immutable revision and required archive sizes/hashes; the IDs are a fixed unique JSON list. The exact pinned native Alpamayo loader selects frames and converts poses. Each clip is committed only after a lossless PNG round trip and file hashes; source archive digests are repository-declared while SDK ZIP reads validate CRC. A completed clip is verified and reused, and incomplete attempts remain separate. Alternate feature-source routing is rejected rather than attributed to the wrong dataset. A full split is assembled only after its complete expected ID inventory is verified.

The observation profile enters both the model and preprocessing identities, so warm start, exact continuation and export cannot silently switch camera contracts. A native first-clip preprocessing check and focused CPU lifecycle tests are preflight evidence, not full driving training, trajectory quality or closed-loop safety results.

## Controlled continuation and relocated warm starts

`rl.deterministic=true rl.deterministic_strict=true` requires PyTorch deterministic kernels. The previous warning-only policy remains the default: in PyTorch 2.10 it permits nondeterministic FlashAttention backward even when a deterministic implementation exists. Strict mode fails when no deterministic kernel is available and records its choice in the objective's exact-resume identity. Switching between strict and legacy modes requires a warm start; it is not exact continuation. These settings do not guarantee bitwise equality across hardware or world sizes.

The first two-host SmolLM2 pilot completed six updates but its warning-only continuation failed the second resumed update's gradient/optimizer tolerance. Input IDs, batch tensors, RNG, data cursor, scheduler and optimizer groups matched. This failed pilot remains evidence; the strict rerun and independent global-objective comparison are separate gates, not inherited passes.

Public SFT portable warm start can relocate an identical local HF tokenizer snapshot. The compatibility check reconstructs only recorded source-location labels on a detached tokenizer, verifies all non-location semantics, and still requires exact base/processor file hashes and adapter settings. Unknown or external tokenizer-file fields refuse. The live tokenizer and new run identity retain the current path; subsequent checkpoints resume with `warm_start` cleared. Legacy artifact bytes and exact-resume identities are unchanged. CPU tests cover the public builder, relocated import, a complete checkpoint and the next resumed update; real installed-package and distributed qualification are recorded separately in the campaign evidence.

## Selected LayerNorm extras

The two-node Pi0.5 mixed recipe exposed a CUDA dtype failure in the selected SigLIP `encoder.layers.26.layer_norm2`: its separately trained weight and bias use FP32 storage while incoming activations use BF16. PyTorch 2.10 CUDA LayerNorm rejects this combination, although its CPU implementation accepts it.

Selected ordinary LayerNorm extras now call their native class implementation with FP32 inputs and return the incoming activation dtype. Parameters are neither replaced nor rounded, so optimizer ownership, aliases, portable state and FP32 merged loading retain the existing contract. RMSNorm extras and the explicitly rounded-weight execution profiles keep their existing implementations. Adapter-disabled modules are unchanged.

Focused CPU/A100 tests cover an independent FP64 normalization reference, finite affine/input gradients, multiple affine updates, nonreentrant activation replay, parameter identity, frozen state and unchanged FP32 subclass behavior. The actual two-node Pi0.5 mixed profiles still require an installed revised-package rerun; the failed initial-validation attempts remain in the campaign evidence. This fix does not itself establish full-data training, export or task qualification.

## Architecture-default embedding ties during bounded export

The two-node Gemma and Gemma2 mixed-recipe checkpoint probes passed portable extraction and reload, then exposed a bounded-export defect: their original `config.json` omits `tie_word_embeddings`, whose architecture default is true. Treating an absent raw JSON field as false incorrectly rejected the intentionally omitted `lm_head.weight` alias.

The bounded tensor source now resolves local `AutoConfig` defaults as the supported stock loader does. It still verifies both tensors when a tied checkpoint serializes both names, and explicit untied configurations still require an independent head. No full model is materialized and no adapter arithmetic changes. Focused tests cover Gemma/Gemma2 default ties, explicit untied heads, missing-head refusal, public export/reuse and strict stock loading. The four affected real mixed-recipe exports (Gemma/Gemma2, FSDP2/HSDP) subsequently passed fresh merge, idempotent reuse and strict stock loading under installed revision `b58329c` in job `38375453`. These are bounded training-checkpoint export probes, not full-dataset qualification. Existing training results retain their original source revision.

## Campaign evidence checkpoint

The sealed all-family campaign contains 16 bindings and 368 required primary cells. The local `campaign-results.json` and `campaign-report.md` in the campaign directory distinguish bounded preflights, complete primary training schedules, full-run continuation and final deployment/task gates. Successful single-host results are not inherited by this matrix.

The final full-data SmolLM2 mixed/FSDP checkpoint at update 935 passes exact portable extraction/reload and FP32 stock-merged comparisons on its fixed 34-case population. Its original execution fails the full-versus-incremental and split-prefix numerical tolerance despite token agreement and immutable prefix state. That failure remains preserved. The opt-in [Smol consistency profile](adapter-smollm-consistent.md), qualified under installed revision `af278a5`, passes all 34 cache cases through both public portable and merged loaders. Its complete matched task evaluation is separate from the earlier profile's complete 1,319-question evaluation; shards from different profiles must not be combined.

The earlier Lustre UID inode exhaustion was reproduced with failed writes and remains part of the historical failure evidence. On November 12, real shared writes and the installed packed-staging job succeeded; the measured quota headroom exceeded five million entries. This removes the previous blanket storage block without deleting campaign data. A separate AlpaSim inventory still identifies 58 unavailable required scene assets. It blocks the complete declared closed-loop suite, not independent offline preparation or other families. PR #342 is not changed or published by this campaign.

Twenty complete primary schedules have now been independently audited: 18 SmolLM2 cells, one Qwen2.5 rsLoRA/HSDP cell and one Qwen3 rsPiSSA/HSDP cell. Nineteen have passed their full-run three-update midpoint continuation checks. The remaining Smol rsPiSSA/FSDP continuation fails at update 470 after exact initial restoration: input IDs, RNG, clocks and earlier updates agree, but a tiny parameter difference crosses a BF16 projection rounding boundary. The rounding mechanism is diagnosed; the source of the earlier gradient difference remains unresolved. No full campaign cell has passed every required gate.

Qwen3 job `38452721` used installed revision `5fbdf7d`, two physical A100 hosts and HSDP `[2,2]`. It completed all 935 updates, all 7,473 training rows plus seven zero-weight padding slots, and both complete 1,319-row validations. Validation loss changed from `1.6426915` to `0.5083566`. These are loss/coverage results, not a task-accuracy claim. Its final deployment and task gates remain pending.

## Evidence-driven completion

`tools/adapter_campaign` verifies the sealed 368-cell contract from content-addressed receipts, source/runtime admission and authoritative training lineage. The collector has a tested success path; it no longer assumes that all future campaigns have zero passes. Missing, contradictory, foreign, incomplete or stale receipts cannot certify a cell. Synthetic success and rejection fixtures are explicitly separate from real results.

The initial real import covers the three primary-training gates for the earlier 19 complete schedules. It does not import or clear historical preflight failures, recovery failures, deployment results or external blockers. `campaign-final-gates-current.json` in the campaign directory identifies this partial import and its independent review. The original orchestration ledger retains the original execution revisions.

## Packed driving media

The native PhysicalAI preparer also accepts `--storage packed_v1 --scratch-dir OWNED_SCRATCH --clips-per-shard 256`. It streams the existing lossless PNG bytes and native row/calibration/action metadata into indexed shards; the logical sample identity and split order are unchanged. The default `clips_v1` representation remains available. Readers validate the complete marker, index, member bounds and content hashes before decoding.

On filesystems without atomic no-replace directory rename, a verified payload is published through an atomic relative symlink. The link and its uniquely named payload directory form one artifact and must be retained together. A published payload can retain an `.incomplete-*` directory name; that name is not permission to delete it. Interrupted unpublished attempts are retained separately, and completed shards are verified before reuse.

Installed revision `48f6aa0` passed 49 CPU tests and exact three-clip native-data/reuse checks on actual Lustre in job `38453087`. The frozen remaining staging plan preserves 27,888 earlier committed clips and assigns 278,263 future clips to 1,113 packed shards. It covers the same 306,151-clip corpus. The approximately 7.3 TB mean-size estimate and 10.97 TB sample-maximum envelope are capacity estimates, not formal upper bounds or completed staging measurements. Full split assembly, driving training and simulator evaluation remain separate gates.
