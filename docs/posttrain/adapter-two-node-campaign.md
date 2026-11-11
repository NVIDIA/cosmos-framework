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

## Native PhysicalAI observations

The Alpamayo recipe accepts `model.model_config.observation_profile=physicalai_four_camera_v1` for the pinned PhysicalAI driving corpus. It binds cameras `[0,1,2,6]` (cross-left, front-wide, cross-right, front-tele), four preceding frames per camera in camera-major order, actual frame timestamps, source intrinsic/extrinsic calibration, and 16 history/64 future poses in the vehicle frame at the fixed keyframe. The native model still owns acceleration/curvature conversion, normalization and flow loss. This profile does not add reasoning supervision. `front_frames_v1` remains the default and retains its original artifact identity.

Use `python -m cosmos_framework.posttrain.scripts.prepare_physicalai_flow_sft --source-manifest SOURCE.json --clip-ids IDS.json --cache-dir CACHE --out PREPARED` in the physical runtime with authorized Hugging Face access. The manifest pins the repository, immutable revision and required archive sizes/hashes; the IDs are a fixed unique JSON list. The exact pinned native Alpamayo loader selects frames and converts poses. Each clip is committed only after a lossless PNG round trip and file hashes; source archive digests are repository-declared while SDK ZIP reads validate CRC. A completed clip is verified and reused, and incomplete attempts remain separate. Alternate feature-source routing is rejected rather than attributed to the wrong dataset. A full split is assembled only after its complete expected ID inventory is verified.

The observation profile enters both the model and preprocessing identities, so warm start, exact continuation and export cannot silently switch camera contracts. A native first-clip preprocessing check and focused CPU lifecycle tests are preflight evidence, not full driving training, trajectory quality or closed-loop safety results.

## Controlled continuation and relocated warm starts

`rl.deterministic=true rl.deterministic_strict=true` requires PyTorch deterministic kernels. The previous warning-only policy remains the default: in PyTorch 2.10 it permits nondeterministic FlashAttention backward even when a deterministic implementation exists. Strict mode fails when no deterministic kernel is available and records its choice in the objective's exact-resume identity. Switching between strict and legacy modes requires a warm start; it is not exact continuation. These settings do not guarantee bitwise equality across hardware or world sizes.

The first two-host SmolLM2 pilot completed six updates but its warning-only continuation failed the second resumed update's gradient/optimizer tolerance. Input IDs, batch tensors, RNG, data cursor, scheduler and optimizer groups matched. This failed pilot remains evidence; the strict rerun and independent global-objective comparison are separate gates, not inherited passes.

Public SFT portable warm start can relocate an identical local HF tokenizer snapshot. The compatibility check reconstructs only recorded source-location labels on a detached tokenizer, verifies all non-location semantics, and still requires exact base/processor file hashes and adapter settings. Unknown or external tokenizer-file fields refuse. The live tokenizer and new run identity retain the current path; subsequent checkpoints resume with `warm_start` cleared. Legacy artifact bytes and exact-resume identities are unchanged. CPU tests cover the public builder, relocated import, a complete checkpoint and the next resumed update; real installed-package and distributed qualification are recorded separately in the campaign evidence.
